//! `XCL_GPU_CULPRIT=1` (diagnostics, off by default): which dispatches a stalled queue has not
//! finished. Around every dispatch XCL records (title lists, console-recorded lists,
//! ExecuteIndirect of a translated compute pipeline) the list writes a running number into a
//! readback buffer with `WriteBufferImmediate`: `MARKER_IN` before (the work before it has
//! started), `MARKER_OUT` after (the work before it is done), one pair of words per PC list.
//! When the GPU-wait watchdog releases a wait, a list whose last number in is above its last
//! number out has work that started and is not done: those dispatches are logged with their
//! console compute shader (`fnv64` of the title's CS bytecode, the name of its file in the
//! translated-shader cache) and their groups.
//!
//! LOCAL: the same markers go around the draws and the title's other GPU operations (`marked`:
//! copies, CopyTiles, clears, resolves, DiscardResource, the console's CopyMemoryToMemoryX /
//! FillMemory / Write32/64 methods), on every list type, COPY lists included.
//!
//! DRED breadcrumbs would say the same on a device removal, but they change the driver's
//! pipeline cache keys: every pipeline is built again, and a retail title's start then takes
//! minutes.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("XCL_GPU_CULPRIT").is_ok_and(|v| v.trim() == "1"))
}

/// Lists with a slot in the readback buffer (two words each).
const SLOTS: usize = 1 << 14;
/// Dispatches remembered by number (the oldest are forgotten).
const REMEMBERED: usize = 1 << 20;

/// The console CS of the compute pipelines the title made, by the pointer it holds.
static CS: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);
/// LOCAL: the console VS / PS of the graphics pipelines the title made, by the pointer it holds.
static GFX: Mutex<Option<HashMap<usize, (u64, u64)>>> = Mutex::new(None);

/// LOCAL: a graphics pipeline the title made: its console VS and PS, for the draw markers.
pub unsafe fn note_graphics(pso: usize, desc: *const c_void) {
    if !enabled() || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = std::ptr::read(desc as *const D3D12_GRAPHICS_PIPELINE_STATE_DESC);
    let h = |b: &D3D12_SHADER_BYTECODE| {
        let (p, n) = (b.pShaderBytecode as usize, b.BytecodeLength);
        if p == 0 || n == 0 || !crate::gpu::shader::readable(p, n) { 0 } else { crate::gpu::shader::fnv64(std::slice::from_raw_parts(p as *const u8, n)) }
    };
    GFX.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, (h(&d.VS), h(&d.PS)));
}

/// LOCAL: "vs … ps …" of a graphics pipeline, for the removal tripwire.
pub fn gfx_names(pso: usize) -> String {
    let g = GFX.lock().unwrap();
    match g.as_ref().and_then(|m| m.get(&pso)) {
        Some((vs, ps)) => format!("vs {vs:016x} ps {ps:016x}"),
        None => String::new(),
    }
}

fn gfx_of(title_pso: usize, pso: usize) -> (u64, u64) {
    let g = GFX.lock().unwrap();
    let Some(m) = g.as_ref() else { return (0, 0) };
    [title_pso, pso, super::pso_async::resolve(title_pso)].iter().find_map(|p| m.get(p)).copied().unwrap_or((0, 0))
}

/// LOCAL: before a draw on `list`, as `before_dispatch`.
pub unsafe fn before_draw(list: &ID3D12GraphicsCommandList, title_pso: usize, pso: usize, count: u32, instances: u32) -> u32 {
    if !enabled() {
        return 0;
    }
    let (vs, ps) = gfx_of(title_pso, pso);
    before_op(list, format!("draw vs {vs:016x} ps {ps:016x}, pipeline {pso:#x}, {count} x{instances}"))
}

/// A compute pipeline the title made: its console CS, for the log.
pub unsafe fn note_compute(pso: usize, desc: *const c_void) {
    if !(enabled() || !skip_list().is_empty()) || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let (p, n) = (d.CS.pShaderBytecode as usize, d.CS.BytecodeLength);
    if p == 0 || n == 0 || !crate::gpu::shader::readable(p, n) {
        return;
    }
    let h = crate::gpu::shader::fnv64(std::slice::from_raw_parts(p as *const u8, n));
    CS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, h);
}

/// LOCAL: `XCL_SKIP_CS=<hash>[,<hash>...]` (16-digit console CS hashes, as XCL_GPU_CULPRIT
/// prints them): dispatches of those programs are not recorded.
fn skip_list() -> &'static Vec<u64> {
    static S: OnceLock<Vec<u64>> = OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_SKIP_CS").map(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect()).unwrap_or_default())
}


/// LOCAL: `XCL_BOUND_CODE=1` – the driver code size (`GetCachedBlob`) of each PC pipeline at its
/// first bind, summed (never subtracted): logged every 10 MB and at a device removal (#505: the
/// removal seems to come at a constant total of bound shader code, about 14 MB per GB of VRAM).
static BOUND: Mutex<Option<(std::collections::HashSet<usize>, u64, u64, u32, u32)>> = Mutex::new(None);

pub fn bound_code_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_BOUND_CODE").is_ok_and(|v| v.trim() == "1"))
}

/// LOCAL: a PC pipeline `pso` is set on a list (`XCL_BOUND_CODE`).
pub unsafe fn note_bound(pso: &ID3D12PipelineState) {
    if !bound_code_on() {
        return;
    }
    let raw = pso.as_raw() as usize;
    {
        let g = BOUND.lock().unwrap();
        if g.as_ref().is_some_and(|b| b.0.contains(&raw)) {
            return;
        }
    }
    let bytes = pso.GetCachedBlob().map(|b| b.GetBufferSize() as u64).unwrap_or(0);
    let mut g = BOUND.lock().unwrap();
    let b = g.get_or_insert_with(|| (std::collections::HashSet::new(), 0, 0, 0, 0));
    if !b.0.insert(raw) {
        return;
    }
    b.1 += bytes;
    b.3 += 1;
    if bytes >= 1 << 20 {
        b.4 += 1;
    }
    if b.1 >= b.2 + (10 << 20) {
        b.2 = b.1;
        crate::xlog_info!("bound code: {} MB in {} pipelines ({} of 1 MB or more)", b.1 >> 20, b.3, b.4);
    }
}

/// LOCAL: the bound-code total, for the removal log.
pub fn bound_code_summary() -> String {
    match BOUND.lock().unwrap().as_ref() {
        Some(b) => format!("{} MB in {} pipelines ({} of 1 MB or more)", b.1 >> 20, b.3, b.4),
        None => "none counted".into(),
    }
}

static SKIPPED: AtomicU32 = AtomicU32::new(0);

/// Whether the dispatch of this pipeline is left out (`XCL_SKIP_CS`).
pub fn skip(title_pso: usize, pso: usize) -> bool {
    let list = skip_list();
    if list.is_empty() {
        return false;
    }
    let h = cs_of(title_pso, pso);
    if h == 0 || !list.contains(&h) {
        return false;
    }
    let n = SKIPPED.fetch_add(1, Ordering::Relaxed);
    if n.is_power_of_two() || n == 0 {
        crate::xlog_info!("[SKIP_CS] dispatch of cs {h:016x} left out ({} so far)", n + 1);
    }
    true
}

/// LOCAL: `XCL_REFUSE_CS=<path>|<hash>[,<path>|<hash>...]`: console compute programs (16-digit
/// `fnv64` of the CS bytecode, the `record` hash of `driver compile start` and of the refused
/// compute PSO warning) that never reach the driver: their translation is not even attempted, the
/// title gets main's stand-in of a console CS without a translation, and their dispatches are
/// skipped as for a failed build. A token of exactly 16 hex digits is a hash, any other a text
/// file of one hash a line (`#` to the end of a line is a comment; blank lines are passed over).
fn refuse_list() -> &'static std::collections::HashSet<u64> {
    static S: OnceLock<std::collections::HashSet<u64>> = OnceLock::new();
    S.get_or_init(|| {
        let Ok(v) = std::env::var("XCL_REFUSE_CS") else { return Default::default() };
        let (set, bad) = parse_refuse(&v, |p| std::fs::read_to_string(p).ok());
        for b in &bad {
            crate::xlog_warn!("XCL_REFUSE_CS: {b} passed over");
        }
        crate::xlog_info!("XCL_REFUSE_CS: {} console compute programs are left out", set.len());
        set
    })
}

/// LOCAL: the hashes of an `XCL_REFUSE_CS` value (`read`: a file's text) and what was not one.
fn parse_refuse(value: &str, read: impl Fn(&str) -> Option<String>) -> (std::collections::HashSet<u64>, Vec<String>) {
    let hash = |t: &str| {
        let h = t.strip_prefix("0x").unwrap_or(t);
        (h.len() == 16 && h.bytes().all(|c| c.is_ascii_hexdigit())).then(|| u64::from_str_radix(h, 16).ok()).flatten()
    };
    let (mut set, mut bad) = (std::collections::HashSet::new(), Vec::new());
    for tok in value.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if let Some(h) = hash(tok) {
            set.insert(h);
            continue;
        }
        let Some(text) = read(tok) else {
            bad.push(format!("{tok:?} (not a 16-digit hash nor a readable file)"));
            continue;
        };
        for (n, line) in text.lines().enumerate() {
            let l = line.split('#').next().unwrap_or("").trim();
            if l.is_empty() {
                continue;
            }
            match hash(l) {
                Some(h) => {
                    set.insert(h);
                }
                None => bad.push(format!("{tok}:{}: {l:?}", n + 1)),
            }
        }
    }
    (set, bad)
}

/// LOCAL: whether the console compute program `cs` is listed in `XCL_REFUSE_CS` (logged once a
/// program): its pipeline is not handed to the driver.
pub fn refused_cs(cs: &[u8]) -> bool {
    let list = refuse_list();
    if list.is_empty() {
        return false;
    }
    let h = crate::gpu::shader::fnv64(cs);
    if !list.contains(&h) {
        return false;
    }
    static LOGGED: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
    if LOGGED.lock().unwrap().get_or_insert_with(Default::default).insert(h) {
        crate::xlog_info!("XCL_REFUSE_CS: {h:016x} left out");
    }
    true
}

/// The title's pipeline, or the one it stands for (a background build).
fn cs_of(title_pso: usize, pso: usize) -> u64 {
    let g = CS.lock().unwrap();
    let Some(m) = g.as_ref() else { return 0 };
    [title_pso, pso, super::pso_async::resolve(title_pso)].iter().find_map(|p| m.get(p)).copied().unwrap_or(0)
}

struct Markers {
    _buf: ID3D12Resource,
    gpu: u64,
    cpu: *const u32,
    /// PC list → its slot
    slots: HashMap<usize, u32>,
    /// dispatch number → (list, what it was)
    what: HashMap<u32, (usize, String)>,
    order: VecDeque<u32>,
}
// SAFETY: `cpu` is the persistent map of `_buf`, read only under the mutex
unsafe impl Send for Markers {}

static MARKERS: Mutex<Option<Markers>> = Mutex::new(None);
static NEXT: AtomicU32 = AtomicU32::new(1);

unsafe fn make_markers(dev: &ID3D12Device) -> Option<Markers> {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (SLOTS * 8) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buf: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buf) {
        crate::xlog_warn!("GPU culprit: no readback buffer for the dispatch markers: {e}");
        return None;
    }
    let buf = buf?;
    let mut p: *mut c_void = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    crate::xlog_info!("GPU culprit: dispatch markers on ({SLOTS} list slots)");
    Some(Markers { gpu: buf.GetGPUVirtualAddress(), _buf: buf, cpu: p as *const u32, slots: HashMap::new(), what: HashMap::new(), order: VecDeque::new() })
}

/// Before a dispatch on `list`: a marker in with a new number, which the caller hands to
/// `after_dispatch` once the dispatch is recorded (0: no marker). `groups` `[0, 0, n]`: an
/// indirect one of at most `n` dispatches.
pub unsafe fn before_dispatch(list: &ID3D12GraphicsCommandList, title_pso: usize, pso: usize, groups: [u32; 3]) -> u32 {
    if !enabled() {
        return 0;
    }
    let size = match groups {
        [0, 0, n] => format!("indirect, up to {n} dispatches"),
        [x, y, z] => format!("groups {x}x{y}x{z}"),
    };
    before_op(list, format!("cs {:016x}, pipeline {pso:#x}, {size}", cs_of(title_pso, pso)))
}

/// A marker in with a new number before an operation described by `what` (LOCAL: shared by the
/// dispatch and draw markers).
pub unsafe fn before_op(list: &ID3D12GraphicsCommandList, what: String) -> u32 {
    if !enabled() {
        return 0;
    }
    let Ok(l2) = list.cast::<ID3D12GraphicsCommandList2>() else { return 0 };
    let mut g = MARKERS.lock().unwrap();
    if g.is_none() {
        let mut dev: Option<ID3D12Device> = None;
        if list.GetDevice(&mut dev).is_err() {
            return 0;
        }
        *g = make_markers(&dev.unwrap());
    }
    let Some(m) = g.as_mut() else { return 0 };
    mark_in(m, list, &l2, what)
}

/// LOCAL: the marker in of `before_op` into `m` (split out for the copy-list test).
unsafe fn mark_in(m: &mut Markers, list: &ID3D12GraphicsCommandList, l2: &ID3D12GraphicsCommandList2, what: String) -> u32 {
    let key = list.as_raw() as usize;
    let Some(slot) = slot_for(&mut m.slots, key, SLOTS) else { return 0 };
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    m.what.insert(id, (key, what));
    m.order.push_back(id);
    if m.order.len() > REMEMBERED {
        let old = m.order.pop_front().unwrap();
        m.what.remove(&old);
    }
    let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + slot as u64 * 8, Value: id };
    l2.WriteBufferImmediate(1, &p, Some(&D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN));
    id
}

/// After the dispatch: a marker out with its number.
pub unsafe fn after_dispatch(list: &ID3D12GraphicsCommandList, id: u32) {
    if id == 0 {
        return;
    }
    let Ok(l2) = list.cast::<ID3D12GraphicsCommandList2>() else { return };
    let g = MARKERS.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    mark_out(m, list, &l2, id);
}

/// LOCAL: the marker out of `after_dispatch` into `m`.
unsafe fn mark_out(m: &Markers, list: &ID3D12GraphicsCommandList, l2: &ID3D12GraphicsCommandList2, id: u32) {
    let Some(&slot) = m.slots.get(&(list.as_raw() as usize)) else { return };
    let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + slot as u64 * 8 + 4, Value: id };
    l2.WriteBufferImmediate(1, &p, Some(&D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT));
}

// ---------------------------------------------------------------------------------------
// LOCAL: markers around the non-draw, non-dispatch operations (copies, clears, resolves,
// discards, the console's memory copies / fills / immediate writes) – same switch
// (XCL_GPU_CULPRIT=1), same readback buffer. WriteBufferImmediate and a READBACK buffer in
// COPY_DEST are valid on COPY lists too (`gpu_markers_on_a_copy_list_match`).
// ---------------------------------------------------------------------------------------

/// LOCAL: `op` on the (adopted PC) list `list` between a marker in and a marker out; `what`
/// (the label, logged only for an op started and not done) is made only when the markers are on.
/// Only the PC call(s) go in `op`: an early return between the markers, or a marked op inside
/// another, would leave the list's number in above its number out.
pub unsafe fn marked<R>(list: *mut c_void, what: impl FnOnce() -> String, op: impl FnOnce() -> R) -> R {
    let _m = mark(list, what);
    op()
}

/// LOCAL: a marker in now, its marker out when this is dropped (for an op with several exits
/// once it is recorded)
pub struct Mark(*mut c_void, u32);

impl Drop for Mark {
    fn drop(&mut self) {
        if self.1 != 0 {
            // SAFETY: the list `mark` was given, still recording
            unsafe {
                if let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&self.0) {
                    after_dispatch(l, self.1);
                }
            }
        }
    }
}

/// LOCAL: the marker in of `marked`, the marker out when the result is dropped
pub unsafe fn mark(list: *mut c_void, what: impl FnOnce() -> String) -> Mark {
    if !enabled() {
        return Mark(list, 0);
    }
    let Some(l) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return Mark(list, 0) };
    Mark(list, before_op(l, format!("{} {}", list_kind(l), what())))
}

/// LOCAL: the list's type, short (`copy` lists: the title's DMA / copy queues)
unsafe fn list_kind(l: &ID3D12GraphicsCommandList) -> &'static str {
    match l.GetType() {
        D3D12_COMMAND_LIST_TYPE_DIRECT => "[direct]",
        D3D12_COMMAND_LIST_TYPE_COMPUTE => "[compute]",
        D3D12_COMMAND_LIST_TYPE_COPY => "[copy]",
        _ => "[other]",
    }
}

/// LOCAL: whether bytes `off .. off + n` leave a buffer of `width` bytes (an overflow does too).
pub fn out_of_bounds(off: u64, n: u64, width: u64) -> bool {
    off.checked_add(n).is_none_or(|end| end > width)
}

/// LOCAL: the least end of a copy footprint in its buffer: the offset, every row but the last
/// one's pitch, and a byte of the last (block-compressed formats: a row is 4 texel rows).
pub fn footprint_end(offset: u64, format: i32, height: u32, depth: u32, row_pitch: u32) -> u64 {
    let bc = matches!(format, 70..=84 | 94..=99);
    let rows = if bc { height.div_ceil(4) } else { height }.max(1) as u64;
    let rows = rows * depth.max(1) as u64;
    offset.saturating_add(row_pitch as u64 * (rows - 1)).saturating_add(1)
}

/// LOCAL: what the label says of a resource
struct Res {
    buffer: bool,
    /// the PC GPU address (buffers; 0 otherwise)
    va: u64,
    width: u64,
    height: u32,
    subresources: u32,
    reserved: bool,
}

unsafe fn res(r: usize) -> Option<Res> {
    let p = r as *mut c_void;
    let x = ID3D12Resource::from_raw_borrowed(&p)?;
    let d = x.GetDesc();
    let buffer = d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    // the PC method: an adopted resource's slot 11 is the console's address for title calls
    let va = if !buffer {
        0
    } else if super::real::is_adopted(p) {
        std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> u64>(super::real::orig_fn(p, 11))(p)
    } else {
        x.GetGPUVirtualAddress()
    };
    let array = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { d.DepthOrArraySize.max(1) as u32 };
    Some(Res { buffer, va, width: d.Width, height: d.Height, subresources: d.MipLevels.max(1) as u32 * array, reserved: super::real::is_reserved(p) })
}

/// LOCAL: a buffer range of a copy: "dst 0x… va 0x…+0x… @off+n" and the flags (OOB: past the
/// buffer's width; reserved: a tiled resource, whether the range is mapped is not checked)
pub unsafe fn buffer_range(tag: &str, r: usize, off: u64, n: u64) -> String {
    match res(r) {
        None => format!("{tag} {r:#x} (no resource) @{off:#x}+{n:#x}"),
        Some(x) => format!(
            "{tag} {r:#x} va {:#x}+{:#x} @{off:#x}+{n:#x}{}{}{}",
            x.va,
            x.width,
            if !x.buffer { " NOT-A-BUFFER" } else { "" },
            if out_of_bounds(off, n, x.width) { " OOB" } else { "" },
            if x.reserved { " reserved(range unchecked)" } else { "" }
        ),
    }
}

/// LOCAL: a whole resource: buffer "va …+width", texture "WxH, N subresources"
pub unsafe fn resource(tag: &str, r: usize) -> String {
    match res(r) {
        None => format!("{tag} {r:#x} (no resource)"),
        Some(x) if x.buffer => format!("{tag} {r:#x} va {:#x}+{:#x}{}", x.va, x.width, if x.reserved { " reserved(unchecked)" } else { "" }),
        Some(x) => format!("{tag} {r:#x} tex {}x{} {} subres{}", x.width, x.height, x.subresources, if x.reserved { " reserved(unchecked)" } else { "" }),
    }
}

/// LOCAL: a subresource of a texture: "tex 0x… WxH sub i/n" (BADSUB: past its subresources;
/// depth-stencil planes are not counted, so a plane-1 copy shows it too)
pub unsafe fn subresource(tag: &str, r: usize, sub: u32) -> String {
    match res(r) {
        None => format!("{tag} {r:#x} (no resource) sub {sub}"),
        Some(x) => format!(
            "{tag} {r:#x} {}x{} sub {sub}/{}{}{}",
            x.width,
            x.height,
            x.subresources,
            if sub >= x.subresources { " BADSUB" } else { "" },
            if x.reserved { " reserved(unchecked)" } else { "" }
        ),
    }
}

/// LOCAL: a copy location of CopyTextureRegion
pub unsafe fn location(tag: &str, l: *const D3D12_TEXTURE_COPY_LOCATION) -> String {
    let Some(l) = l.as_ref() else { return format!("{tag} null") };
    let r: usize = std::mem::transmute_copy(&l.pResource);
    if l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
        let f = l.Anonymous.PlacedFootprint;
        let end = footprint_end(f.Offset, f.Footprint.Format.0, f.Footprint.Height, f.Footprint.Depth, f.Footprint.RowPitch);
        format!(
            "{} fp fmt {} {}x{}x{} pitch {:#x}",
            buffer_range(tag, r, f.Offset, end - f.Offset),
            f.Footprint.Format.0,
            f.Footprint.Width,
            f.Footprint.Height,
            f.Footprint.Depth,
            f.Footprint.RowPitch
        )
    } else {
        subresource(tag, r, l.Anonymous.SubresourceIndex)
    }
}

/// The slot of `list`: its own, or the next free one; none once `cap` lists have one (the list is
/// then not recorded, so every recorded slot stays inside the readback buffer).
fn slot_for(slots: &mut HashMap<usize, u32>, list: usize, cap: usize) -> Option<u32> {
    if let Some(&s) = slots.get(&list) {
        return Some(s);
    }
    let n = slots.len();
    (n < cap).then(|| {
        slots.insert(list, n as u32);
        n as u32
    })
}

/// The lists with work started and not done: (list, last number in, last number out), from
/// each slot's pair of words. A marker in only says that the work before it has started, so a
/// list counts when its number in is above its number out (in and out written out of order
/// otherwise look open).
fn open_lists(slots: &HashMap<usize, u32>, words: impl Fn(u32) -> (u32, u32)) -> Vec<(usize, u32, u32)> {
    let mut open: Vec<(usize, u32, u32)> = slots
        .iter()
        .filter_map(|(&list, &slot)| {
            let (i, o) = words(slot);
            (i > o).then_some((list, i, o))
        })
        .collect();
    open.sort_by_key(|&(_, i, _)| i);
    open
}

/// The dispatches of `list` numbered after `out` up to `in_` that are still remembered.
fn not_done(what: &HashMap<u32, (usize, String)>, list: usize, in_: u32, out: u32) -> Vec<(u32, &str)> {
    ((out + 1)..=in_).filter_map(|id| what.get(&id).filter(|(l, _)| *l == list).map(|(_, s)| (id, s.as_str()))).collect()
}

/// The watchdog released GPU waits: log the dispatches started and not done (once per burst,
/// 4 s apart, at most 8 times).
pub fn on_watchdog_release() {
    static BURSTS: AtomicU32 = AtomicU32::new(0);
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    if !enabled() {
        return;
    }
    {
        let mut l = LAST.lock().unwrap();
        if l.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(4)) {
            return;
        }
        *l = Some(std::time::Instant::now());
    }
    if BURSTS.fetch_add(1, Ordering::Relaxed) >= 8 {
        return;
    }
    let g = MARKERS.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    // SAFETY: slots index the mapped buffer of SLOTS pairs
    let open = open_lists(&m.slots, |s| unsafe { (std::ptr::read_volatile(m.cpu.add(s as usize * 2)), std::ptr::read_volatile(m.cpu.add(s as usize * 2 + 1))) });
    crate::xlog_warn!("GPU culprit: {} of {} lists have dispatches started and not done", open.len(), m.slots.len());
    for (list, i, o) in open.iter().take(16) {
        let left = not_done(&m.what, *list, *i, *o);
        crate::xlog_warn!("GPU culprit: list {list:#x}: marker in #{i}, last marker out #{o}; {} of its dispatches not known done:", left.len());
        for (id, s) in left.iter().take(24) {
            crate::xlog_warn!("GPU culprit:   #{id}: {s}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LOCAL: `XCL_REFUSE_CS` takes inline hashes and hash files with comments and blank lines.
    #[test]
    fn refuse_cs_list_takes_inline_hashes_and_files_with_comments() {
        let file = "# CS_RaytracingResolve\n\n00112233445566aa # CS_RaytracingResolve_a\n  FFEEDDCCBBAA9988\n#ffffffffffffffff\nnot-a-hash\n";
        let read = |p: &str| (p == "rt.txt").then(|| file.to_string());
        let (set, bad) = parse_refuse("rt.txt", read);
        assert_eq!(set, [0x0011_2233_4455_66aa, 0xffee_ddcc_bbaa_9988].into_iter().collect());
        assert_eq!(bad, vec!["rt.txt:6: \"not-a-hash\"".to_string()]);

        let (set, bad) = parse_refuse(" 0123456789abcdef, 0xfedcba9876543210 ,,rt.txt, 1234, missing.txt", read);
        assert_eq!(set.len(), 4);
        assert!(set.contains(&0x0123_4567_89ab_cdef) && set.contains(&0xfedc_ba98_7654_3210) && set.contains(&0x0011_2233_4455_66aa));
        assert_eq!(bad.len(), 3, "{bad:?}"); // the file's bad line, "1234", missing.txt

        // a real file on disk, read as the runtime reads it
        let path = std::env::temp_dir().join(format!("xcl-refuse-cs-{}.txt", std::process::id()));
        std::fs::write(&path, "\r\n# header\r\n0a0b0c0d0e0f1011 # name\r\n").unwrap();
        let (set, bad) = parse_refuse(path.to_str().unwrap(), |p| std::fs::read_to_string(p).ok());
        let _ = std::fs::remove_file(&path);
        assert_eq!(set, [0x0a0b_0c0d_0e0f_1011].into_iter().collect());
        assert!(bad.is_empty());
    }

    /// Only a list whose number in is above its number out has work in flight, and the
    /// dispatches logged for it are its own, after the out.
    #[test]
    fn a_list_with_a_marker_in_above_its_marker_out_is_open() {
        let slots: HashMap<usize, u32> = [(0xA, 0), (0xB, 1), (0xC, 2), (0xD, 3)].into_iter().collect();
        // A: done; B: 12 started, 10 done; C: in and out written out of order; D: never ran
        let words = |s: u32| [(9, 9), (12, 10), (20, 21), (0, 0)][s as usize];
        assert_eq!(open_lists(&slots, words), vec![(0xB, 12, 10)]);

        let what: HashMap<u32, (usize, String)> =
            [(10, (0xB, "b10".to_string())), (11, (0xA, "a11".to_string())), (12, (0xB, "b12".to_string()))].into_iter().collect();
        assert_eq!(not_done(&what, 0xB, 12, 10), vec![(12, "b12")], "11 belongs to another list");
    }

    /// Once every slot is taken, a new list gets none and is not recorded, and the lists that
    /// have one keep it: no slot past the readback buffer is ever read.
    #[test]
    fn a_list_past_the_last_slot_gets_none() {
        let mut slots = HashMap::new();
        assert_eq!(slot_for(&mut slots, 0xA, 2), Some(0));
        assert_eq!(slot_for(&mut slots, 0xB, 2), Some(1));
        assert_eq!(slot_for(&mut slots, 0xC, 2), None);
        assert_eq!(slots.len(), 2, "the list without a slot is not recorded");
        assert_eq!(slot_for(&mut slots, 0xA, 2), Some(0));
        assert!(slots.values().all(|&s| (s as usize) < 2));
    }

    /// The markers reach the readback buffer: a list run to its end leaves its numbers in and
    /// out equal, so nothing is reported for it.
    #[test]
    fn gpu_markers_of_a_finished_list_match() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let Some(m) = make_markers(&dev) else { panic!("readback buffer") };
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let l2: ID3D12GraphicsCommandList2 = list.cast().unwrap();
            for (i, mode) in [(0u64, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN), (4, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT)] {
                let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + 8 + i, Value: 77 };
                l2.WriteBufferImmediate(1, &p, Some(&mode));
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(t.elapsed().as_secs() < 10, "the list did not finish");
                std::thread::yield_now();
            }
            let words = |s: u32| (std::ptr::read_volatile(m.cpu.add(s as usize * 2)), std::ptr::read_volatile(m.cpu.add(s as usize * 2 + 1)));
            assert_eq!(words(1), (77, 77));
            let slots: HashMap<usize, u32> = [(list.as_raw() as usize, 1)].into_iter().collect();
            assert!(open_lists(&slots, words).is_empty());
        }
    }

    /// LOCAL: a range past the buffer's end (or one whose end overflows) is out of bounds; one
    /// that ends at it is not.
    #[test]
    fn a_copy_range_past_the_buffer_is_out_of_bounds() {
        assert!(!out_of_bounds(0, 0x100, 0x100));
        assert!(!out_of_bounds(0xF0, 0x10, 0x100));
        assert!(out_of_bounds(0xF0, 0x11, 0x100));
        assert!(out_of_bounds(0x100, 1, 0x100));
        assert!(out_of_bounds(u64::MAX, 2, u64::MAX), "an end that overflows is out");
    }

    /// LOCAL: a footprint ends at least one byte into its last row: rows of texels, rows of
    /// 4x4 blocks for BC formats, times the depth.
    #[test]
    fn a_footprint_ends_in_its_last_row() {
        // R8G8B8A8_UNORM (28), 4 rows of 256 bytes at 0x200: the last row starts at 0x500
        assert_eq!(footprint_end(0x200, 28, 4, 1, 256), 0x501);
        // BC1 (71), 16 texel rows = 4 block rows
        assert_eq!(footprint_end(0, 71, 16, 1, 256), 3 * 256 + 1);
        // BC7 (98), 6 texel rows round up to 2 block rows, 3 slices
        assert_eq!(footprint_end(0, 98, 6, 3, 512), 5 * 512 + 1);
        // height 0 counts as one row
        assert_eq!(footprint_end(0x40, 28, 0, 0, 256), 0x41);
        assert!(out_of_bounds(0x200, footprint_end(0x200, 28, 4, 1, 256) - 0x200, 0x500), "the footprint leaves a 0x500-byte buffer");
    }

    /// LOCAL: the markers work on a COPY list (the title's DMA / copy queues): WriteBufferImmediate
    /// into the READBACK buffer in COPY_DEST around a CopyBufferRegion records, closes, runs, and
    /// leaves the list's numbers in and out equal, with the device still there.
    #[test]
    fn gpu_markers_on_a_copy_list_match() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let Some(mut m) = make_markers(&dev) else { panic!("readback buffer") };
            let buffer = |heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 0x1000,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                let mut b: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut b).unwrap();
                b.unwrap()
            };
            let src = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            let dst = buffer(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
            let ty = D3D12_COMMAND_LIST_TYPE_COPY;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let l2: ID3D12GraphicsCommandList2 = list.cast().unwrap();
            assert_eq!(list_kind(&list), "[copy]");
            let a = mark_in(&mut m, &list, &l2, "copy a".into());
            list.CopyBufferRegion(&dst, 0, &src, 0, 0x1000);
            mark_out(&m, &list, &l2, a);
            let b = mark_in(&mut m, &list, &l2, "copy b".into());
            list.CopyBufferRegion(&dst, 0x800, &src, 0, 0x800);
            mark_out(&m, &list, &l2, b);
            assert!(b > a && a != 0);
            list.Close().expect("a COPY list with the markers closes");
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(t.elapsed().as_secs() < 10, "the list did not finish");
                std::thread::yield_now();
            }
            dev.GetDeviceRemovedReason().expect("the device is still there");
            let slot = m.slots[&(list.as_raw() as usize)];
            let words = |s: u32| (std::ptr::read_volatile(m.cpu.add(s as usize * 2)), std::ptr::read_volatile(m.cpu.add(s as usize * 2 + 1)));
            assert_eq!(words(slot), (b, b), "the last op's numbers in and out");
            assert!(open_lists(&m.slots, words).is_empty(), "nothing started and not done");
            assert_eq!(not_done(&m.what, list.as_raw() as usize, b, a), vec![(b, "copy b")], "the labels are kept by number");
        }
    }
}
