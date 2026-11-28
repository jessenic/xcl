//! lab: `XCL_LIT_TRACE=<seconds>[,<x>,<y>]` - which command of a frame writes extreme values into
//! the 1280x720 R16G16B16A16_FLOAT target the title lights its scene in.
//!
//! For ONE frame - from the first present after `<seconds>` of uptime to the next present - a
//! window of 64x64 texels centred on `<x>,<y>` (default 380,470) of up to `TRACKED` such
//! textures (2-D, single-sampled, format 9 / 10, 1280x720: first seen as render target 0, or
//! named by a dispatch's tables with a UAV flag) is copied into the next slot of one readback buffer:
//! when the texture is bound as a render target (the contents before the draws that follow),
//! after every draw / DispatchMesh / ExecuteIndirect recorded while it is bound as one, and
//! after every dispatch whose descriptor tables name it (named: read or written - "changed"
//! tells which). The copy is recorded on the title's list right after the command, the source in
//! whatever state it is in (as `list_x::pass_snapshot`; a UAV barrier first after compute).
//!
//! Lists are recorded on several threads: the order the snapshots are recorded in is not the
//! order the GPU runs them in. `submitted` stamps each list's snapshots at ExecuteCommandLists;
//! the report - one present after the frame's end, once the queue has run - lists them in
//! submit order (submit, index in the batch, order on the list), one line each, then names the
//! first whose window holds a texel over 100 or a NaN / Inf, and the snapshot before it.
//! Nothing but one `OnceLock` read when the variable is not set.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};

/// the window's side, texels
const SIDE: u32 = 64;
/// bytes of a slot: `SIDE` rows of `SIDE` 8-byte texels (the row pitch 512 and the slot's
/// offset are multiples of what a placed footprint asks: 256 and 512)
const SLOT: usize = (SIDE * SIDE * 8) as usize;
/// snapshots in a trace at most (64 MiB of readback memory)
const SLOTS: usize = 6144;
/// textures followed at most
const TRACKED: usize = 12;
/// PC list slots (as `list_x`): CopyTextureRegion, ResourceBarrier
const PC_COPY_TEXTURE_REGION: usize = 16;
const PC_RESOURCE_BARRIER: usize = 26;

pub(super) const BIND: u8 = 0;
pub(super) const DRAW: u8 = 1;
pub(super) const MESH_DRAW: u8 = 2;
pub(super) const INDIRECT_DRAW: u8 = 3;
pub(super) const DISPATCH: u8 = 4;
pub(super) const INDIRECT_DISPATCH: u8 = 5;
/// before the dispatch runs (what the texture holds when the program starts)
pub(super) const BEFORE_DISPATCH: u8 = 6;
pub(super) const BEFORE_INDIRECT_DISPATCH: u8 = 7;
/// the list is closed: every followed texture, as the list leaves it
pub(super) const CLOSE: u8 = 8;

fn kind_name(kind: u8) -> &'static str {
    match kind {
        BIND => "rt bind",
        DRAW => "draw",
        MESH_DRAW => "mesh draw",
        INDIRECT_DRAW => "indirect draw",
        DISPATCH => "dispatch",
        INDIRECT_DISPATCH => "indirect dispatch",
        BEFORE_DISPATCH => "before dispatch",
        CLOSE => "list close",
        _ => "before indirect dispatch",
    }
}

/// (seconds of uptime, window centre x, y)
fn config() -> Option<(f64, u32, u32)> {
    static C: OnceLock<Option<(f64, u32, u32)>> = OnceLock::new();
    *C.get_or_init(|| {
        let v = std::env::var("XCL_LIT_TRACE").ok()?;
        let mut f = v.split(',').map(|s| s.trim());
        let t: f64 = f.next()?.parse().ok()?;
        let x = f.next().and_then(|s| s.parse().ok()).unwrap_or(380);
        let y = f.next().and_then(|s| s.parse().ok()).unwrap_or(470);
        Some((t, x, y))
    })
}

const WAITING: u8 = 0;
const TRACING: u8 = 1;
/// the frame is over: lists recorded in it may still be submitted, the report is at the next present
const DRAINING: u8 = 2;
const DONE: u8 = 3;
static PHASE: AtomicU8 = AtomicU8::new(WAITING);

/// the console (vs, ps) hashes of the graphics pipelines (a map of its own: `real::PSO_SHADERS`
/// having them changes what other lab code does)
static GRAPHICS: Mutex<Option<HashMap<usize, (u64, u64)>>> = Mutex::new(None);

pub(super) fn note_graphics(pso: usize, vs: u64, ps: u64) {
    GRAPHICS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, (vs, ps));
}

/// The console hashes of `pso` (the title's object or what it resolves to).
pub(super) fn shaders(pso: usize) -> Option<(u64, u64)> {
    let own = |p: usize| GRAPHICS.lock().unwrap().as_ref().and_then(|m| m.get(&p).copied());
    let resolved = super::pso_async::resolve(pso);
    own(pso).or_else(|| own(resolved)).or_else(|| super::real::pso_shaders(pso)).or_else(|| super::real::pso_shaders(resolved))
}

/// `XCL_LIT_TRACE` is set.
pub(super) fn on() -> bool {
    config().is_some()
}

/// The traced frame is being recorded (the call sites ask before they look anything up).
pub(super) fn tracing() -> bool {
    config().is_some() && PHASE.load(Ordering::Relaxed) == TRACING
}

struct Tracked {
    res: usize,
    /// the window: left, top, width, height
    window: [u32; 4],
    format: DXGI_FORMAT,
}

struct Snap {
    list: usize,
    /// its number among the snapshots recorded on `list`
    list_seq: u32,
    kind: u8,
    pso: usize,
    /// console record hashes: (vs / ms, ps) or (cs, 0)
    hashes: Option<(u64, u64)>,
    /// index in `Trace::tracked`
    tracked: usize,
    /// the render target index the texture is bound at; 0xFF: named by the dispatch's tables
    role: u8,
    /// named by a dispatch: table << 8 | slot
    place: u16,
    /// the console descriptor that names it
    words: [u32; 8],
    /// (submit number, index in its batch, queue) once its list was submitted
    submit: Option<(u32, u32, usize)>,
}

struct Trace {
    dev: ID3D12Device,
    rb: ID3D12Resource,
    tracked: Vec<Tracked>,
    snaps: Vec<Snap>,
    per_list: HashMap<usize, u32>,
    submits: u32,
    /// commands past `SLOTS`
    dropped: u64,
    /// multisampled 1280x720 float targets seen (not copied)
    multisampled: u64,
    /// textures of the lit target's shape that were not followed: (resource, how first seen)
    passed: Vec<(usize, String)>,
    /// calls of `after` by kind; those of a draw kind on a list with no known render target
    calls: [u64; 8],
    /// per list: (kind, hashes or (pso, 0)) of its draws and dispatches, each once, and their number
    cmds: HashMap<usize, (Vec<(u8, u64, u64)>, u32)>,
    /// per list: the (submit, index) it was last submitted at
    list_submit: HashMap<usize, (u32, u32)>,
    /// descriptor word 0 (the address) of the descriptors that named a followed texture
    addresses: Vec<u32>,
    /// (resource, cs hash) of the uses of such an address through another resource, logged
    others: Vec<(usize, u64)>,
    no_target: u64,
    /// uptime at its start
    began: f64,
}
unsafe impl Send for Trace {}
static TRACE: Mutex<Option<Trace>> = Mutex::new(None);

unsafe fn add_ref(p: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(1));
    f(p);
}

/// The lit target's shape: the window of `res` when it is one (Err(true): but multisampled).
unsafe fn lit_window(res: usize, need_uav: bool, x: u32, y: u32) -> Result<([u32; 4], DXGI_FORMAT), bool> {
    let r = res as *mut c_void;
    let Some(d) = ID3D12Resource::from_raw_borrowed(&r).map(|t| t.GetDesc()) else { return Err(false) };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !matches!(d.Format.0, 9 | 10) || d.Width < 960 || d.Height < 540 {
        return Err(false);
    }
    if need_uav && d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        return Err(false);
    }
    if d.SampleDesc.Count > 1 {
        return Err(true);
    }
    let (w, h) = (SIDE.min(d.Width as u32), SIDE.min(d.Height));
    // (`x`, `y` are of a 1280x720 image: the same place in a larger one)
    let (x, y) = ((x as u64 * d.Width / 1280) as u32, y * d.Height / 720);
    let left = x.saturating_sub(w / 2).min(d.Width as u32 - w);
    let top = y.saturating_sub(h / 2).min(d.Height - h);
    Ok(([left, top, w, h], d.Format))
}

/// After a command recorded on the PC list `this` (`kind`; `pso`: its pipeline, `hashes`: its
/// console record hashes where the caller knows them better than `real::pso_shaders`): a
/// snapshot of every followed texture bound as a render target (not for a dispatch) or among
/// `named` (the textures the dispatch's tables name).
pub(super) unsafe fn after(this: *mut c_void, kind: u8, pso: usize, hashes: Option<(u64, u64)>, named: &[(usize, u16, [u32; 8])]) {
    let Some((_, x, y)) = config() else { return };
    if PHASE.load(Ordering::Relaxed) != TRACING {
        return;
    }
    // (render target index or 0xFF, resource)
    let mut found: Vec<(u8, usize, u16, [u32; 8])> = Vec::new();
    if kind < DISPATCH {
        for (k, h) in super::list_x::bound_render_targets(this).into_iter().enumerate() {
            if let Some(r) = super::present::rtv_resource(h) {
                found.push((k as u8, r, 0, [0; 8]));
            }
        }
    }
    found.extend(named.iter().map(|&(r, at, w)| (0xFF, r, at, w)));
    let mut g = TRACE.lock().unwrap();
    let Some(t) = g.as_mut() else { return };
    if kind == CLOSE {
        found = t.tracked.iter().map(|k| (0xFE, k.res, 0, [0; 8])).collect();
    }
    if PHASE.load(Ordering::Relaxed) != TRACING {
        return;
    }
    if kind != CLOSE {
        t.calls[kind as usize % 8] += 1;
    }
    if kind < DISPATCH && found.is_empty() {
        t.no_target += 1;
    }
    if (DRAW..=INDIRECT_DISPATCH).contains(&kind) {
        let h = hashes.or_else(|| shaders(pso)).unwrap_or((pso as u64, 0));
        let e = t.cmds.entry(this as usize).or_default();
        e.1 += 1;
        if e.0.len() < 40 && !e.0.contains(&(kind, h.0, h.1)) {
            e.0.push((kind, h.0, h.1));
        }
    }
    let hashes = if kind == CLOSE { None } else { hashes.or_else(|| shaders(pso)) };
    let mut after_compute = kind >= DISPATCH;
    // lab: `XCL_LIT_TRACE_INPUTS=<cs hash>`: every texture the tables of that program's dispatch name, once
    if matches!(kind, BEFORE_DISPATCH | BEFORE_INDIRECT_DISPATCH) {
        static WANT: OnceLock<u64> = OnceLock::new();
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let want = *WANT.get_or_init(|| std::env::var("XCL_LIT_TRACE_INPUTS").ok().and_then(|v| u64::from_str_radix(v.trim(), 16).ok()).unwrap_or(0));
        if want != 0 && hashes.map(|h| h.0) == Some(want) && !SAID.swap(true, Ordering::Relaxed) {
            for &(_, res, place, words) in found.iter().filter(|f| f.0 == 0xFF) {
                let r = res as *mut c_void;
                let d = ID3D12Resource::from_raw_borrowed(&r).map(|x| x.GetDesc());
                crate::xlog_info!(
                    "lab: lit trace: inputs of {want:016x}: table {} slot {} [{:08x} {:08x} {:08x} {:08x}] {res:#x} {}",
                    place >> 8,
                    place & 0xFF,
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                    d.map_or("?".to_string(), |d| format!("{}x{}x{} format {} mips {} flags {:#x} dim {}", d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels, d.Flags.0, d.Dimension.0))
                );
            }
        }
    }
    // a followed texture's address named through another resource (a view of another format?)
    for &(_, res, place, words) in found.iter().filter(|f| f.0 == 0xFF) {
        if words[0] == 0 || !t.addresses.contains(&words[0]) || t.tracked.iter().any(|k| k.res == res) {
            continue;
        }
        let cs = hashes.map_or(0, |h| h.0);
        if t.others.len() < 200 && !t.others.contains(&(res, cs)) {
            t.others.push((res, cs));
            let r = res as *mut c_void;
            let d = ID3D12Resource::from_raw_borrowed(&r).map(|x| x.GetDesc());
            crate::xlog_info!(
                "lab: lit trace: OTHER resource {res:#x} at a followed address: {} cs {cs:016x} list {:#x} table {} slot {} [{}] desc {}",
                kind_name(kind),
                this as usize,
                place >> 8,
                place & 0xFF,
                words.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" "),
                d.map_or("?".to_string(), |d| format!("{}x{} format {} flags {:#x} dim {}", d.Width, d.Height, d.Format.0, d.Flags.0, d.Dimension.0))
            );
        }
    }
    for (role, res, place, words) in found {
        let at = match t.tracked.iter().position(|k| k.res == res) {
            Some(at) => at,
            // (found as render target 0 or through a dispatch's tables with a UAV flag only: a
            // second target of a draw - a G-buffer plane, a history - does not take a place)
            None => match lit_window(res, role == 0xFF, x, y) {
                Ok(_) if role != 0 && role != 0xFF || t.tracked.len() >= TRACKED => {
                    if t.passed.len() < 16 && !t.passed.iter().any(|p| p.0 == res) {
                        let how = if role == 0xFF { "named by a dispatch".to_string() } else { format!("render target {role}") };
                        t.passed.push((res, format!("{how} at {} on list {:#x}", kind_name(kind), this as usize)));
                    }
                    continue;
                }
                Ok((window, format)) => {
                    // (ours until the process ends: the lists of the frame name it)
                    add_ref(res);
                    t.tracked.push(Tracked { res, window, format });
                    crate::xlog_info!(
                        "lab: lit trace: follows {res:#x} ({}x{} format {}), first seen {} on list {:#x} at {}: window {}x{} at {},{}",
                        ID3D12Resource::from_raw_borrowed(&(res as *mut c_void)).map_or(0, |r| r.GetDesc().Width),
                        ID3D12Resource::from_raw_borrowed(&(res as *mut c_void)).map_or(0, |r| r.GetDesc().Height),
                        format.0,
                        if role == 0xFF { "named by a dispatch".to_string() } else { format!("as render target {role}") },
                        this as usize,
                        kind_name(kind),
                        window[2],
                        window[3],
                        window[0],
                        window[1]
                    );
                    t.tracked.len() - 1
                }
                Err(multisampled) => {
                    if multisampled {
                        t.multisampled += 1;
                    }
                    continue;
                }
            },
        };
        if role == 0xFF && words[0] != 0 && !t.addresses.contains(&words[0]) {
            t.addresses.push(words[0]);
        }
        let slot = t.snaps.len();
        if slot >= SLOTS {
            t.dropped += 1;
            continue;
        }
        if after_compute {
            // (a UAV barrier first: the pass's writes are done before the copy reads)
            let uav: [u32; 8] = [2, 0, 0, 0, 0, 0, 0, 0];
            let f: unsafe extern "system" fn(*mut c_void, u32, *const [u32; 8]) = std::mem::transmute(super::real::orig_fn(this, PC_RESOURCE_BARRIER));
            f(this, 1, &uav);
            after_compute = false;
        }
        let k = &t.tracked[at];
        let [left, top, w, h] = k.window;
        let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: (slot * SLOT) as u64,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: k.format, Width: w, Height: h, Depth: 1, RowPitch: SIDE * 8 },
        };
        let rb = t.rb.as_raw() as usize;
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        let from = D3D12_BOX { left, top, front: 0, right: left + w, bottom: top + h, back: 1 };
        let f: unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX) =
            std::mem::transmute(super::real::orig_fn(this, PC_COPY_TEXTURE_REGION));
        f(this, &dst, 0, 0, 0, &src, &from);
        std::mem::forget((dst, src));
        let seq = t.per_list.entry(this as usize).or_insert(0);
        let list_seq = *seq;
        *seq += 1;
        t.snaps.push(Snap { list: this as usize, list_seq, kind, pso, hashes, tracked: at, role, place, words, submit: None });
    }
}

/// ExecuteCommandLists of the PC lists `lists` on `queue`: their snapshots get the submit's number.
pub(super) fn submitted(queue: *mut c_void, lists: &[*mut c_void]) {
    if config().is_none() || !matches!(PHASE.load(Ordering::Relaxed), TRACING | DRAINING) {
        return;
    }
    let mut g = TRACE.lock().unwrap();
    let Some(t) = g.as_mut() else { return };
    let n = t.submits;
    t.submits += 1;
    let mut with: Vec<String> = Vec::new();
    for (i, &l) in lists.iter().enumerate() {
        t.list_submit.entry(l as usize).or_insert((n, i as u32));
        let mut stamped = 0;
        for s in t.snaps.iter_mut().filter(|s| s.list == l as usize && s.submit.is_none()) {
            s.submit = Some((n, i as u32, queue as usize));
            stamped += 1;
        }
        if stamped > 0 {
            with.push(format!("[{i}] {:#x} ({stamped} snapshots)", l as usize));
        }
    }
    if !with.is_empty() {
        crate::xlog_info!("lab: lit trace: submit {n} on queue {:#x} at {:.3} s, {} lists; with snapshots: {}", queue as usize, crate::log::uptime(), lists.len(), with.join(", "));
    }
}

/// The title presents on `queue`: the trace starts at the first present after its time, stops
/// recording at the next, and is read and logged at the one after.
pub(super) unsafe fn present(queue: *mut c_void) {
    let Some((from, x, y)) = config() else { return };
    match PHASE.load(Ordering::Relaxed) {
        WAITING if crate::log::uptime() >= from => {
            let Some(trace) = begin() else {
                crate::xlog_warn!("lab: lit trace: no readback buffer - off");
                PHASE.store(DONE, Ordering::Relaxed);
                return;
            };
            *TRACE.lock().unwrap() = Some(trace);
            PHASE.store(TRACING, Ordering::Relaxed);
            crate::xlog_info!(
                "lab: lit trace: from this present ({:.3} s) to the next: {SIDE}x{SIDE} texels around {x},{y} of up to {TRACKED} 1280x720 format 9/10 targets, after every command that can write them ({SLOTS} at most). Recorded on several threads: # below is the order the GPU ran them in (submit, index in the batch, order on the list), rec the order they were recorded in",
                crate::log::uptime()
            );
        }
        TRACING => {
            // (presents come in uneven pairs - 40 ms, then the rest of a 200 ms frame: the trace
            // runs until the first present 400 ms or more after its start, two frames or so)
            let g = TRACE.lock().unwrap();
            if g.as_ref().is_some_and(|t| crate::log::uptime() - t.began < 0.4) {
                return;
            }
            PHASE.store(DRAINING, Ordering::Relaxed);
            if let Some(t) = g.as_ref() {
                crate::xlog_info!(
                    "lab: lit trace: frame over at {:.3} s: {} snapshots of {} textures on {} lists; commands seen: {} rt binds, {} draws, {} mesh draws, {} indirect draws, {} dispatches, {} indirect dispatches ({} of the binds and draws on a list with no known render target){}{}{}; read at the next present",
                    crate::log::uptime(),
                    t.snaps.len(),
                    t.tracked.len(),
                    t.per_list.len(),
                    t.calls[0],
                    t.calls[1],
                    t.calls[2],
                    t.calls[3],
                    t.calls[4],
                    t.calls[5],
                    t.no_target,
                    if t.passed.is_empty() { String::new() } else { format!("; 1280x720 format 9/10 textures NOT followed: {}", t.passed.iter().map(|p| format!("{:#x} ({})", p.0, p.1)).collect::<Vec<_>>().join(", ")) },
                    if t.dropped > 0 { format!(", {} MORE NOT TAKEN (the {SLOTS} slots were full)", t.dropped) } else { String::new() },
                    if t.multisampled > 0 { format!(", {} uses of a multisampled 1280x720 float target not followed", t.multisampled) } else { String::new() }
                );
            }
        }
        DRAINING => {
            PHASE.store(DONE, Ordering::Relaxed);
            let g = TRACE.lock().unwrap();
            let Some(t) = g.as_ref() else { return };
            // what was submitted so far has run before the buffer is read
            let done = (|| -> Result<(), String> {
                let q = ID3D12CommandQueue::from_raw_borrowed(&queue).ok_or("no queue")?;
                let fence: ID3D12Fence = t.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| e.to_string())?;
                q.Signal(&fence, 1).map_err(|e| e.to_string())?;
                crate::gpu::fence::wait(&fence, 1, &t.dev, None, 30000).map_err(|e| e.to_string())
            })();
            if let Err(e) = done {
                crate::xlog_warn!("lab: lit trace: the queue was not waited for ({e}): slots may read as not run");
            }
            report(t);
            // (the buffer and the trace stay: a list recorded in the frame may still be submitted)
        }
        _ => {}
    }
}

unsafe fn begin() -> Option<Trace> {
    let raw = super::real::device_raw();
    let dev = ID3D12Device::from_raw_borrowed(&raw)?.clone();
    let mut rb: Option<ID3D12Resource> = None;
    super::title_textures::own_work(|| {
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: (SLOTS * SLOT) as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
    })
    .ok()?;
    let rb = rb?;
    // marks: a slot the GPU has not written reads 0xEE in every byte
    let mut p: *mut c_void = std::ptr::null_mut();
    rb.Map(0, None, Some(&mut p)).ok()?;
    if p.is_null() {
        return None;
    }
    std::ptr::write_bytes(p as *mut u8, 0xEE, SLOTS * SLOT);
    rb.Unmap(0, None);
    Some(Trace { dev, rb, tracked: Vec::new(), snaps: Vec::new(), per_list: HashMap::new(), submits: 0, dropped: 0, multisampled: 0, passed: Vec::new(), calls: [0; 8], cmds: HashMap::new(), list_submit: HashMap::new(), addresses: Vec::new(), others: Vec::new(), no_target: 0, began: crate::log::uptime() })
}

/// What a window holds.
#[derive(Default)]
struct Stats {
    max: [f32; 3],
    mean: [f64; 3],
    over: usize,
    /// texels with a NaN or an Inf in r, g or b
    not_finite: usize,
    /// the first texel over 100 or not finite, in the texture
    first_bad: Option<(u32, u32)>,
    /// mean |r - r of the texel to the right| over the mean r, finite texels (speckle: near 1)
    rough: f64,
}

/// A texel (8 bytes, R16G16B16A16_FLOAT) with r, g or b over 100, or not finite.
fn bad(t: &[u8]) -> bool {
    (0..3).any(|k| {
        let v = u16::from_le_bytes([t[2 * k], t[2 * k + 1]]);
        v >> 10 & 31 == 31 || super::present::half(v) > 100.0
    })
}

/// `texels`: rows of `w` 8-byte R16G16B16A16_FLOAT texels, the window at `left`, `top`.
fn stats(texels: &[u8], w: u32, left: u32, top: u32) -> Stats {
    let mut s = Stats { max: [f32::MIN; 3], ..Default::default() };
    let mut finite = 0usize;
    for (i, t) in texels.chunks_exact(8).enumerate() {
        let c: [u16; 3] = std::array::from_fn(|k| u16::from_le_bytes([t[2 * k], t[2 * k + 1]]));
        // (`present::half` reads exponent 31 as 65504: Inf and NaN are told apart here, before it)
        let bad = if c.iter().any(|v| v >> 10 & 31 == 31) {
            s.not_finite += 1;
            true
        } else {
            let v = c.map(super::present::half);
            finite += 1;
            for k in 0..3 {
                s.max[k] = s.max[k].max(v[k]);
                s.mean[k] += v[k] as f64;
            }
            let over = v.iter().any(|&x| x > 100.0);
            s.over += over as usize;
            over
        };
        if bad && s.first_bad.is_none() {
            s.first_bad = Some((left + i as u32 % w, top + i as u32 / w));
        }
    }
    let (mut diff, mut pairs) = (0f64, 0usize);
    for row in texels.chunks_exact(8 * w as usize) {
        for x in 0..w as usize - 1 {
            let (a, b) = (u16::from_le_bytes([row[8 * x], row[8 * x + 1]]), u16::from_le_bytes([row[8 * x + 8], row[8 * x + 9]]));
            if a >> 10 & 31 != 31 && b >> 10 & 31 != 31 {
                diff += (super::present::half(a) as f64 - super::present::half(b) as f64).abs();
                pairs += 1;
            }
        }
    }
    s.rough = diff / pairs.max(1) as f64 / (s.mean[0] / finite.max(1) as f64).abs().max(1e-9);
    for k in 0..3 {
        s.mean[k] /= finite.max(1) as f64;
        if finite == 0 {
            s.max[k] = 0.0;
        }
    }
    s
}

unsafe fn report(t: &Trace) {
    let mut p: *mut c_void = std::ptr::null_mut();
    if t.rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
        crate::xlog_warn!("lab: lit trace: the readback buffer could not be mapped");
        return;
    }
    let slot = |i: usize| std::slice::from_raw_parts((p as *const u8).add(i * SLOT), SLOT);
    // the order the GPU ran them in; lists not submitted yet last, as recorded
    let mut order: Vec<usize> = (0..t.snaps.len()).collect();
    order.sort_by_key(|&i| (t.snaps[i].submit.map_or((u32::MAX, 0), |s| (s.0, s.1)), i));
    crate::xlog_info!(
        "lab: lit trace: {} snapshots, {} submits seen, textures: {}",
        order.len(),
        t.submits,
        t.tracked.iter().enumerate().map(|(k, x)| format!("t{k} = {:#x}", x.res)).collect::<Vec<_>>().join(", ")
    );
    // per texture: the last snapshot that ran (slot, its number in the order), the first bad one and the one before it
    let mut last: Vec<Option<(usize, usize)>> = vec![None; t.tracked.len()];
    let mut first_bad: Vec<Option<(usize, Option<usize>)>> = vec![None; t.tracked.len()];
    // the first snapshot in which a texel is bad that was not, or held other bits, in the one before
    let mut first_new: Vec<Option<(usize, usize)>> = vec![None; t.tracked.len()];
    let mut lines: Vec<String> = Vec::with_capacity(order.len());
    for (n, &i) in order.iter().enumerate() {
        let s = &t.snaps[i];
        let k = &t.tracked[s.tracked];
        let name = match s.hashes {
            Some((a, b)) if s.kind >= DISPATCH => format!("vs/ms {:016x} ps/cs {a:016x}", b),
            Some((a, b)) => format!("vs/ms {a:016x} ps/cs {b:016x}"),
            None if s.kind == BIND || s.kind == CLOSE => "vs/ms - ps/cs -".to_string(),
            None => format!("vs/ms ? ps/cs ? (pso {:#x})", s.pso),
        };
        let role = if s.role == 0xFE { "at close".to_string() } else if s.role == 0xFF { format!("named table {} slot {} [{}]", s.place >> 8, s.place & 0xFF, s.words.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" ")) } else { format!("rt{}", s.role) };
        let place = match s.submit {
            Some((sub, at, _)) => format!("list {:#x} (submit {sub}[{at}], #{} on it, rec {i})", s.list, s.list_seq),
            None => format!("list {:#x} (NOT SUBMITTED by the report, #{} on it, rec {i})", s.list, s.list_seq),
        };
        let head = format!("lab: lit trace #{n} {} {name} {place} t{} {role}", kind_name(s.kind), s.tracked);
        let data = slot(i);
        if data.iter().all(|&b| b == 0xEE) {
            lines.push(format!("{head}: not run by the GPU when read"));
            crate::xlog_info!("{}", lines[n]);
            continue;
        }
        let [left, top, w, h] = k.window;
        let texels = &data[..(w * h * 8) as usize];
        if last[s.tracked].is_none() {
            if let Some(boot) = crate::state::boot() {
                let _ = std::fs::write(boot.logs_dir().join(format!("lit_trace_t{}_{w}x{h}_at_{left}_{top}.raw", s.tracked)), texels);
            }
        }
        let st = stats(texels, w, left, top);
        let (changed, new_bad) = match last[s.tracked] {
            Some((prev, _)) => {
                let pairs = || texels.chunks_exact(8).zip(slot(prev).chunks_exact(8)).filter(|(a, b)| a != b);
                let new_bad = pairs().filter(|(a, _)| bad(a)).count();
                (format!("{} texels, {new_bad} of them now over 100 / NaN / Inf", pairs().count()), new_bad)
            }
            None => ("- (first)".to_string(), 0),
        };
        if new_bad > 0 && first_new[s.tracked].is_none() {
            first_new[s.tracked] = last[s.tracked].map(|l| (n, l.1));
        }
        lines.push(format!(
            "{head}: max rgb {:.4e} {:.4e} {:.4e}, mean rgb {:.4e} {:.4e} {:.4e}, rough {:.3}, texels over 100: {}, NaN/Inf: {}, changed vs previous: {changed}{}",
            st.max[0],
            st.max[1],
            st.max[2],
            st.mean[0],
            st.mean[1],
            st.mean[2],
            st.rough,
            st.over,
            st.not_finite,
            st.first_bad.map_or(String::new(), |(x, y)| format!(", first such texel {x},{y}"))
        ));
        crate::xlog_info!("{}", lines[n]);
        if first_bad[s.tracked].is_none() && (st.over > 0 || st.not_finite > 0) {
            first_bad[s.tracked] = Some((n, last[s.tracked].map(|l| l.1)));
        }
        last[s.tracked] = Some((i, n));
    }
    t.rb.Unmap(0, None);
    for (k, x) in t.tracked.iter().enumerate() {
        match first_bad[k] {
            None => crate::xlog_info!("lab: lit trace: summary t{k} {:#x}: no snapshot of the window holds a texel over 100 or a NaN / Inf", x.res),
            Some((n, None)) => crate::xlog_info!("lab: lit trace: summary t{k} {:#x}: the FIRST snapshot already holds them (written before it, or by its command): {}", x.res, lines[n]),
            Some((n, Some(before))) => {
                crate::xlog_info!("lab: lit trace: summary t{k} {:#x}: first with a texel over 100 or a NaN / Inf: {}", x.res, lines[n]);
                crate::xlog_info!("lab: lit trace: summary t{k} {:#x}: the snapshot before it: {}", x.res, lines[before]);
            }
        }
        // (the target may still hold the frame before's values when it is bound: the command that WRITES them)
        match first_new[k] {
            Some((n, before)) => {
                crate::xlog_info!("lab: lit trace: summary t{k} {:#x}: first that CHANGED a texel to over 100 / NaN / Inf: {}", x.res, lines[n]);
                crate::xlog_info!("lab: lit trace: summary t{k} {:#x}: the snapshot before that: {}", x.res, lines[before]);
            }
            None => crate::xlog_info!("lab: lit trace: summary t{k} {:#x}: no snapshot changed a texel of the window to over 100 / NaN / Inf", x.res),
        }
    }
    let mut ls: Vec<_> = t.cmds.iter().collect();
    ls.sort_by_key(|(l, _)| t.list_submit.get(l).copied().unwrap_or((u32::MAX, 0)));
    for (l, (c, n)) in ls {
        let at = t.list_submit.get(l).map_or("not submitted".to_string(), |s| format!("submit {}[{}]", s.0, s.1));
        crate::xlog_info!("lab: lit trace: list {l:#x} ({at}): {n} draws / dispatches: {}", c.iter().map(|&(k, a, b)| format!("{}:{a:016x}/{b:016x}", kind_name(k))).collect::<Vec<_>>().join(", "));
    }
    if t.tracked.is_empty() {
        crate::xlog_info!("lab: lit trace: no 1280x720 format 9/10 texture was bound as a render target or named by a dispatch with a UAV flag in the frame");
    }
    crate::log::flush();
}
