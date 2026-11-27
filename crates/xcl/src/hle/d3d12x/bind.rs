//! Root arguments of console code, bound when it dispatches with a translated pipeline.
//!
//! A translated console shader reads its root arguments the console's way (user SGPRs s4…,
//! the rest from a table at s[0:1]: `console_list::user_data`) and memory through XCL's
//! shader ABI (u0 io, u1 res, u2… unified memory). So on a command list of console code the
//! compute root signature and root arguments are kept here – packed by the root signature's
//! layout – instead of going to the PC list, and each Dispatch with a translated pipeline
//! binds the ABI root signature, writes the user SGPRs (`io`) and the argument table to
//! unified memory and dispatches. That memory stays until the list's allocator is reset.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param};

#[derive(Default)]
struct State {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    /// the graphics root signature's layout and arguments
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    /// the title's graphics root signature (a PC object), bound before a draw with an
    /// untranslated pipeline – created with it – since SetGraphicsRootSignature binds nothing
    rs_gfx: usize,
    pso: usize,
    /// the PC allocator this recording is on (`alloc_pool`: not always the title's allocator)
    allocator: usize,
    /// unified memory of this recording (io, argument tables)
    uma: Vec<usize>,
    /// where the next draw's io and argument table go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
    res_buf: Option<(ID3D12Resource, u64, usize)>,
    res_retired: Vec<(ID3D12Resource, u64, usize)>,
    /// the next dispatch's thread limits (`IO_THREAD_LIMIT`; a `DispatchX` with thread
    /// dimensions), taken by its `bind_abi`
    limits: [u32; 3],
}

static ALLOC_RES: Mutex<Option<HashMap<usize, Vec<(ID3D12Resource, u64, usize)>>>> = Mutex::new(None);
static FREE_RES: Mutex<Vec<(ID3D12Resource, u64, usize)>> = Mutex::new(Vec::new());

fn hold_res_until_reset(allocator: usize, mut bufs: Vec<(ID3D12Resource, u64, usize)>) {
    if bufs.is_empty() || allocator == 0 {
        return;
    }
    ALLOC_RES.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().append(&mut bufs);
}

pub(super) fn release_allocator_res(allocator: usize) {
    if let Some(bufs) = ALLOC_RES.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator)) {
        let over = give_back(&mut FREE_RES.lock().unwrap(), FREE_RES_KEPT, bufs);
        RES_LIVE.fetch_sub(over.iter().map(|b| b.1 as i64).sum(), Ordering::Relaxed);
    }
    if let Some(bufs) = ALLOC_SCALED.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator)) {
        let over = give_back(&mut FREE_SCALED.lock().unwrap(), FREE_SCALED_KEPT, bufs);
        SCALED_STATS[2].fetch_add(over.len() as u64, Ordering::Relaxed);
    }
}

/// free record buffers kept for reuse (the rest are released)
const FREE_RES_KEPT: usize = 16;
/// free scaled-argument buffers kept for reuse (`XCL_SPLIT_FREE_LISTS=1`; small, one a call)
const FREE_SCALED_KEPT: usize = 64;

/// `bufs` an allocator held, given back to `free` up to `kept` in all; what did not fit (to be
/// released).
fn give_back<T>(free: &mut Vec<T>, kept: usize, bufs: Vec<T>) -> Vec<T> {
    let mut over = Vec::new();
    for b in bufs {
        if free.len() < kept {
            free.push(b);
        } else {
            over.push(b);
        }
    }
    over
}

/// The smallest buffer of `free` (`(buffer, bytes, device)`) of device `dev` with `need` bytes
/// at least, taken out of it.
fn take_fitting<T>(free: &mut Vec<(T, u64, usize)>, dev: usize, need: u64) -> Option<(T, u64, usize)> {
    let i = free.iter().enumerate().filter(|(_, (_, cap, d))| *d == dev && *cap >= need).min_by_key(|(_, (_, cap, _))| *cap).map(|(i, _)| i)?;
    Some(free.swap_remove(i))
}

/// lab: `XCL_SPLIT_FREE_LISTS=1` (off unless set): the buffers of the scaled mesh arguments
/// (`make_scaled_args_buffer`, one a two-pass ExecuteIndirect) go to a free list of their own
/// and are reused from it. Unset: they are made for each call and given back to the record
/// buffers' free list (16), which they filled – no record buffer fitted there and none could
/// be kept, so a new 4 MiB one was made for nearly every recording (a retail title's world:
/// 16,128 of them in 640 s, "16 free" in every line).
fn split_free_lists() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPLIT_FREE_LISTS").is_ok_and(|v| v.trim() == "1"))
}

/// `XCL_SPLIT_FREE_LISTS=1`: scaled-argument buffers held until their allocator's reset, and
/// those free again
static ALLOC_SCALED: Mutex<Option<HashMap<usize, Vec<(ID3D12Resource, u64, usize)>>>> = Mutex::new(None);
static FREE_SCALED: Mutex<Vec<(ID3D12Resource, u64, usize)>> = Mutex::new(Vec::new());
/// scaled-argument buffers made, reused, released for a full free list
static SCALED_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A scaled-argument buffer of `bytes` at least (`XCL_SPLIT_FREE_LISTS=1`): a free one, or a new
/// one of the next power of two (64 KiB at least, as a committed buffer takes anyway).
unsafe fn scaled_args_buffer(dev: &ID3D12Device, bytes: u64) -> Result<(ID3D12Resource, u64, usize), String> {
    let dev_key = dev.as_raw() as usize;
    if let Some(e) = take_fitting(&mut FREE_SCALED.lock().unwrap(), dev_key, bytes) {
        SCALED_STATS[1].fetch_add(1, Ordering::Relaxed);
        return Ok(e);
    }
    let size = bytes.next_power_of_two().max(64 << 10);
    let n = SCALED_STATS[0].fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_info!(
            "D3D12.X: scaled-argument buffer #{n} of {} KiB made (XCL_SPLIT_FREE_LISTS=1: {} reused, {} released, {} free)",
            size >> 10,
            SCALED_STATS[1].load(Ordering::Relaxed),
            SCALED_STATS[2].load(Ordering::Relaxed),
            FREE_SCALED.lock().unwrap().len()
        );
    }
    Ok((default_uav_buffer(dev, size, "scaled args buffer")?, size, dev_key))
}

/// lab: bytes of two-pass record buffers that exist (in use, held until their allocator's
/// reset, or free)
static RES_LIVE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// lab: (bytes of the bounded copies' buffers that exist, buffers made so far) – `XCL_ACTIVITY`
pub fn clamp_stats() -> (u64, u64) {
    super::indirect_clamp::scratch_stats()
}

/// lab: (bytes of record buffers that exist, allocators holding some, free ones) – `XCL_ACTIVITY`
pub fn res_stats() -> (u64, usize, usize) {
    let held = ALLOC_RES.lock().unwrap().as_ref().map_or(0, |m| m.len());
    (RES_LIVE.load(Ordering::Relaxed).max(0) as u64, held, FREE_RES.lock().unwrap().len())
}

static STATES: Mutex<Option<HashMap<usize, State>>> = Mutex::new(None);

/// draws / dispatches of console code: with a translated pipeline, skipped, with a PC one
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// draws and dispatches of console code so far (`XCL_ACTIVITY`)
pub fn activity() -> (u64, u64) {
    (COUNTS[DRAWS].load(Ordering::Relaxed), COUNTS[DISPATCHES].load(Ordering::Relaxed))
}
const DRAWS: usize = 0;
const DRAWS_SKIPPED: usize = 1;
const DRAWS_PC: usize = 2;
const DISPATCHES: usize = 3;
const DISPATCHES_SKIPPED: usize = 4;

fn count(k: usize) {
    let n = COUNTS[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let (allocators, blocks) = console_list::pending_blocks();
        let (coherent_used, coherent_total, sparse_used) = crate::gpu::uma::get()
            .map_or((0, 0, 0), |u| (u.used() >> 20, u.size() >> 20, u.sparse_used() >> 20));
        crate::xlog_info!(
            "D3D12.X: console code so far: {} draws with translated pipelines ({} skipped, {} with PC pipelines), {} dispatches ({} skipped); coherent {coherent_used}/{coherent_total} MB, sparse {sparse_used} MB, {blocks} blocks held for {allocators} allocators",
            c[DRAWS], c[DRAWS_SKIPPED], c[DRAWS_PC], c[DISPATCHES], c[DISPATCHES_SKIPPED],
        );
    }
}

/// lab: arena blocks the lists hold themselves (a recording's blocks go to its allocator at the
/// list's next reset); 0 when the map is in use.
pub(super) fn blocks_in_lists() -> usize {
    STATES.try_lock().ok().and_then(|g| g.as_ref().map(|m| m.values().map(|s| s.uma.len()).sum())).unwrap_or(0)
}

fn with<R>(list: *mut c_void, f: impl FnOnce(&mut State) -> R) -> R {
    f(STATES.lock().unwrap().get_or_insert_with(HashMap::new).entry(list as usize).or_default())
}

/// The list was created or reset onto the PC allocator `allocator` with `initial_pso`: the
/// previous recording's memory is the previous allocator's until the GPU is done with it, and
/// pipeline and root signature state reset per D3D12 `CreateCommandList` / `Reset` semantics.
pub(super) fn reset(list: *mut c_void, allocator: usize, initial_pso: usize) {
    super::indirect_clamp::list_reset(list as usize);
    super::gpu_profile::list_reset(list);
    let (old, blocks, res_bufs) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.pso = initial_pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        let mut res_bufs = std::mem::take(&mut s.res_retired);
        if let Some(b) = s.res_buf.take() {
            res_bufs.push(b);
        }
        (old, std::mem::take(&mut s.uma), res_bufs)
    });
    console_list::hold_until_reset(old, blocks);
    hold_res_until_reset(old, res_bufs);
}

/// `ClearState(pso)`: resets pipeline and root-argument state back to creation defaults
/// with `pso` bound, keeping the current recording's allocator and UMA arena intact.
pub(super) fn clear_state(list: *mut c_void, pso: usize) {
    with(list, |s| {
        s.pso = pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
    });
}

/// Carry bound pipeline and root-signature / root-argument state from `from` to `to` across a
/// deferred mid-list split (`to` records into `allocator` with its own fresh UMA arena while
/// `from` retains its previously allocated blocks until `allocator` resets).
pub(super) fn inherit_for_split(from: *mut c_void, to: *mut c_void, allocator: usize) {
    let (layout, args, layout_gfx, args_gfx, rs_gfx, pso) = with(from, |s| {
        (s.layout.clone(), s.args.clone(), s.layout_gfx.clone(), s.args_gfx.clone(), s.rs_gfx, s.pso)
    });
    let (old, blocks) = with(to, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.layout = layout;
        s.args = args;
        s.layout_gfx = layout_gfx;
        s.args_gfx = args_gfx;
        s.rs_gfx = rs_gfx;
        s.pso = pso;
        (old, std::mem::take(&mut s.uma))
    });
    console_list::hold_until_reset(old, blocks);
}

/// A submit is about to wait for the pipeline `this` has bound.
pub(super) fn want_bound(this: *mut c_void) {
    super::pso_async::want(with(this, |s| s.pso));
}

/// lab `XCL_RT_WATCH`: the pipeline `this` has bound.
pub(super) fn bound_pso(this: *mut c_void) -> usize {
    with(this, |s| s.pso)
}

/// lab (`XCL_SKIP_LOG=1`): one line the first time a pipeline's draw or dispatch is skipped at
/// a place in this file (the source line), with its console shaders' hashes when they are known
/// and the uptime: which programs a run lost at world entry, to compare between runs.
pub(super) fn skip_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SKIP_LOG").is_ok_and(|v| v.trim() == "1"))
}

fn skip_note(this: *mut c_void, line: u32) {
    if !skip_log() {
        return;
    }
    let pso = with(this, |s| s.pso);
    skip_note_pso(pso, line, "skipped");
}

/// A call of the list's bound pipeline was put off to the submit (`list_x::defer`).
pub(super) fn skip_note_put_off(this: *mut c_void) {
    if skip_log() {
        skip_note_pso(with(this, |s| s.pso), line!(), "put off to the submit");
    }
}

/// [`skip_note`] for a pipeline by itself (`what`: skipped, put off).
pub(super) fn skip_note_pso(pso: usize, line: u32, what: &str) {
    if !skip_log() {
        return;
    }
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<(u64, u64, usize, u32)>>> = std::sync::Mutex::new(None);
    let shaders = super::real::pso_shaders(pso);
    let key = shaders.map_or((0, 0, pso, line), |(a, b)| (a, b, 0, line));
    if !SEEN.lock().unwrap().get_or_insert_with(Default::default).insert(key) {
        return;
    }
    let (a, b) = shaders.unwrap_or((0, 0));
    crate::xlog_info!(
        "lab: skip log: {what} first at line {line}: pipeline {pso:#x} shaders {a:016x} {b:016x}{} at {:.1} s",
        if super::pso_async::is_pending(pso) { " (still being built)" } else { "" },
        crate::log::uptime()
    );
}

/// A put-off dispatch that could not be recorded.
pub(super) fn dispatch_dropped() {
    count(DISPATCHES_SKIPPED);
    super::rt_watch::dropped();
}

/// `to` goes on in the arena `from` took its pieces from (`from` takes no more).
pub(super) fn pass_arena(from: *mut c_void, to: *mut c_void) {
    let arena = with(from, |s| std::mem::take(&mut s.arena));
    with(to, |s| s.arena = arena);
}

/// The list is gone.
pub(super) fn forget(list: *mut c_void) {
    if let Some(mut s) = STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize))) {
        console_list::hold_until_reset(s.allocator, s.uma);
        if let Some(b) = s.res_buf.take() {
            s.res_retired.push(b);
        }
        hold_res_until_reset(s.allocator, s.res_retired);
    }
}

#[cfg(test)]
pub(crate) fn seed_reset_state_for_test(list: *mut c_void, allocator: usize, pso: usize, uma: Vec<usize>) {
    with(list, |s| {
        s.allocator = allocator;
        s.pso = pso;
        s.layout = Some(Arc::new(Layout::default()));
        s.args = vec![1, 2, 3, 4];
        s.layout_gfx = Some(Arc::new(Layout::default()));
        s.args_gfx = vec![5, 6, 7, 8];
        s.uma = uma;
    });
}

#[cfg(test)]
pub(crate) fn reset_state_for_test(list: *mut c_void) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<usize>) {
    with(list, |s| (s.allocator, s.pso, s.args.clone(), s.args_gfx.clone(), s.uma.clone()))
}

#[cfg(test)]
pub(crate) fn remove_state_for_test(list: *mut c_void) {
    STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize)));
}

/// bytes of parameter `i` at constant `offset` (of the compute or the graphics arguments)
fn put_in(list: *mut c_void, gfx: bool, i: u32, offset: u32, bytes: &[u8]) {
    with(list, |s| {
        let (layout, args) = if gfx { (&s.layout_gfx, &mut s.args_gfx) } else { (&s.layout, &mut s.args) };
        let Some(at) = layout.as_ref().and_then(|l| l.params.get(i as usize)).map(|p| match *p {
            Param::Constants { at, .. } => at as usize + 4 * offset as usize,
            Param::Descriptor { at } | Param::Table { at } => at as usize,
        }) else {
            return;
        };
        if let Some(t) = args.get_mut(at..at + bytes.len()) {
            t.copy_from_slice(bytes);
        }
    })
}

fn put(list: *mut c_void, i: u32, offset: u32, bytes: &[u8]) {
    put_in(list, false, i, offset, bytes)
}

/// #34 SetGraphicsRootSignature (PC #30)
pub(super) unsafe extern "system" fn set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args_gfx = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout_gfx = layout;
        s.rs_gfx = rs as usize;
    });
}

/// #36 SetGraphicsRootDescriptorTable (PC #32)
pub(super) unsafe extern "system" fn set_graphics_root_table(this: *mut c_void, i: u32, handle: u64) {
    put_in(this, true, i, 0, &(handle as u32).to_le_bytes());
}

/// #38 SetGraphicsRoot32BitConstant (PC #34)
pub(super) unsafe extern "system" fn set_graphics_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put_in(this, true, i, offset, &value.to_le_bytes());
}

/// #40 SetGraphicsRoot32BitConstants (PC #36)
pub(super) unsafe extern "system" fn set_graphics_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put_in(this, true, i, offset, &bytes);
    }
}

/// #42 / #44 / #46 SetGraphicsRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #38 /
/// #40 / #42)
pub(super) unsafe extern "system" fn set_graphics_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put_in(this, true, i, 0, &va.to_le_bytes());
}

/// `XCL_TRACE_ADDRESSES=1`: root descriptor addresses of console code – in unified memory
/// (translated shaders see them) or not – counted and logged
fn note_address(va: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static IN: AtomicU64 = AtomicU64::new(0);
    static OUT: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("XCL_TRACE_ADDRESSES").is_some()) {
        return;
    }
    let inside = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    let (i, o) = if inside { (IN.fetch_add(1, Ordering::Relaxed) + 1, OUT.load(Ordering::Relaxed)) } else { (IN.load(Ordering::Relaxed), OUT.fetch_add(1, Ordering::Relaxed) + 1) };
    if (i + o).is_power_of_two() {
        crate::xlog_info!("D3D12.X: root descriptor addresses: {i} in unified memory, {o} outside (latest {va:#x})");
    }
}

/// command signatures that change root arguments (they need the title's root signature)
static ROOT_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchMeshSig {
    companion_dispatch: usize,
    byte_stride: u32,
    mesh_offset: u32,
}

static DISPATCH_MESH_SIGS: Mutex<Option<HashMap<usize, DispatchMeshSig>>> = Mutex::new(None);

pub(super) fn note_root_changing_signature(sig: usize) {
    ROOT_SIGNATURES.lock().unwrap().get_or_insert_with(Default::default).insert(sig);
}

/// command signatures with a Dispatch argument (noted at every creation, so a signature made
/// at a freed one's address does not keep its kind)
static DISPATCH_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(super) fn note_signature_dispatches(sig: usize, dispatches: bool) {
    let mut s = DISPATCH_SIGNATURES.lock().unwrap();
    let s = s.get_or_insert_with(Default::default);
    if dispatches {
        s.insert(sig);
    } else {
        s.remove(&sig);
    }
}

fn signature_dispatches(sig: usize) -> bool {
    DISPATCH_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig))
}

pub(super) fn note_dispatch_mesh_signature(sig: usize, companion_dispatch: usize, byte_stride: u32, mesh_offset: u32) {
    DISPATCH_MESH_SIGS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(sig, DispatchMeshSig { companion_dispatch, byte_stride, mesh_offset });
}

fn dispatch_mesh_sig_of(sig: usize) -> Option<DispatchMeshSig> {
    DISPATCH_MESH_SIGS.lock().unwrap().as_ref()?.get(&sig).copied()
}

unsafe fn emit_uav_barrier(list: &ID3D12GraphicsCommandList, res: Option<&ID3D12Resource>) {
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(res.cloned()),
            }),
        },
    };
    list.ResourceBarrier(&[std::ptr::read(&b)]);
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.UAV).pResource);
}

unsafe fn emit_transition(list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let mut b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    list.ResourceBarrier(&[std::ptr::read(&b)]);
    std::mem::ManuallyDrop::drop(&mut (*b.Anonymous.Transition).pResource);
}

unsafe fn default_uav_buffer(dev: &ID3D12Device, bytes: u64, what: &str) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut res: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        None,
        &mut res,
    )
    .map_err(|e| format!("{what} ({bytes:#x} B): {e}"))?;
    res.ok_or_else(|| format!("null {what}"))
}

unsafe fn ensure_res_buffer(this: *mut c_void, need_bytes: u64) -> Result<(ID3D12Resource, u64), String> {
    let need = need_bytes.max(4 << 20).next_multiple_of(1 << 20).min(128 << 20);
    if let Some((buf, _cap, _dev)) = with(this, |s| s.res_buf.as_ref().filter(|(_, cap, _)| *cap >= need).cloned()) {
        let va = buf.GetGPUVirtualAddress();
        return Ok((buf, va));
    }
    let dev = super::real::list_device(this).ok_or("no device for two-pass buffer")?;
    let dev_key = dev.as_raw() as usize;
    let picked = {
        let mut free = FREE_RES.lock().unwrap();
        free.iter().position(|(_, cap, d)| *d == dev_key && *cap >= need).map(|i| free.swap_remove(i))
    };
    let entry = match picked {
        Some(e) => e,
        None => {
            // lab: how much video memory the record buffers take
            static MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let (n, made) = (N.fetch_add(1, Ordering::Relaxed) + 1, MADE.fetch_add(need, Ordering::Relaxed) + need);
            if n <= 4 || n.is_power_of_two() || n % 32 == 0 {
                let (free, smaller) = {
                    let f = FREE_RES.lock().unwrap();
                    (f.len(), f.iter().filter(|(_, cap, _)| *cap < need).count())
                };
                crate::xlog_info!("D3D12.X: two-pass record buffer #{n} of {} MiB made ({} MiB made so far, {free} free, {smaller} of them smaller)", need >> 20, made >> 20);
            }
            RES_LIVE.fetch_add(need as i64, Ordering::Relaxed);
            (default_uav_buffer(&dev, need, "two-pass res buffer")?, need, dev_key)
        }
    };
    let buf = entry.0.clone();
    let va = buf.GetGPUVirtualAddress();
    with(this, |s| {
        if let Some(old) = s.res_buf.replace(entry) {
            s.res_retired.push(old);
        }
    });
    Ok((buf, va))
}

/// lab: the first user SGPR after the root arguments gets the address of the title's bound
/// view heap (`list_x::title_view_heap`; `XCL_HEAP_SGPR=0`: not) – it was 0, and the world mesh
/// programs that take their meshlet from a visible list (flags bit 28) read the list's V#
/// from address 0 + slot * 32: every group then drew meshlet 0.
/// The SGPR after it gets the address of the bound sampler heap (`list_x::title_sampler_heap`;
/// `XCL_HEAP_SGPR=1`: the view heap alone): the world pixel shaders index both heaps themselves
/// (264 of the 362 dumped: T# = s15 + i * 32, S# = s16 + i * 16, prim mask s17, after seven
/// words of root arguments in s8..s14) – with 0 there every S# was read from the first bytes
/// of window 0.
fn heap_sgpr(list: *mut c_void, io: &mut Vec<u32>, known_layout: bool) {
    static MODE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| std::env::var("XCL_HEAP_SGPR").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2u32));
    if !known_layout || mode == 0 {
        return;
    }
    let base = super::list_x::title_view_heap(list);
    let samplers = if mode >= 2 { super::list_x::title_sampler_heap(list) } else { 0 };
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: heap SGPRs: s{} = view heap {base:#x}, s{} = sampler heap {samplers:#x} ({n} so far)", io.len(), io.len() + 1);
    }
    if base != 0 && io.len() + 1 < crate::gpu::recomp::IO_UMA_BASE_HI {
        io.push(base as u32);
        io.push(samplers as u32);
    }
}

unsafe fn bind_two_pass_stage(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    real_mesh_pso: usize,
    tp: &super::real::TwoPassPso,
    res_va: u64,
    compute_pass: bool,
) -> Result<(), String> {
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take_or(u, bytes.len().max(16), 256, &mut s.uma, super::sparse::arena_block)).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    if compute_pass {
        // lab: what the program reads of the title's memory outside the unified windows
        super::view_mem::ensure_for_draw(layout.as_deref(), &args);
    }
    // page budget: the textures in the bound descriptor tables are sampled, by the compute pass
    // (the console MS) and by the pixel shader of the mesh pass alike (as `bind_abi`; CharlyRien's
    // finding on #505: these draws never touched their tables)
    if let Some(l) = layout.as_ref().filter(|_| super::vram::tracking()) {
        super::title_textures::tables_drawn(&table_addresses(l, &args, u.base()));
    }
    let inline = layout
        .as_ref()
        .map_or(0, |l| if compute_pass { l.inline_ms } else { l.inline_gfx } as usize)
        .min(args.len());
    let table_at = copy(&args[inline..])?;
    let mut io = match layout.as_ref() {
        Some(l) if compute_pass => console_list::user_data_ms(l, &args, table_at as u64),
        Some(l) => console_list::user_data_gfx(l, &args, table_at as u64),
        None if compute_pass => {
            let mut v = vec![0u32; 16];
            v[8] = table_at as u32;
            v[9] = (table_at >> 32) as u32;
            v
        }
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    let user = io.len();
    heap_sgpr(this, &mut io, layout.is_some());
    if !compute_pass {
        ps_probe(tp, &io);
        if io.len() > user {
            draw_textures(&io, user);
        }
    }
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
    let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let rs_raw = if compute_pass { tp.compute_rs } else { tp.mesh_rs } as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs_raw).ok_or("two-pass root signature")?;
    let views = [va(io_at), res_va].into_iter().chain((0..4).map(|k| windows.get(k).copied().filter(|&v| v != 0).unwrap_or(first)));
    if compute_pass {
        list.SetComputeRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, v);
        }
        let pso_raw = tp.compute as *mut c_void;
        let pc = ID3D12PipelineState::from_raw_borrowed(&pso_raw).ok_or("two-pass compute pipeline")?;
        list.SetPipelineState(pc);
    } else {
        list.SetGraphicsRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, v);
        }
        let pso_raw = real_mesh_pso as *mut c_void;
        let pc = ID3D12PipelineState::from_raw_borrowed(&pso_raw).ok_or("two-pass mesh pipeline")?;
        list.SetPipelineState(pc);
    }
    Ok(())
}

/// `XCL_NATIVE_MESH`: the bindings of a pipeline whose mesh shader is the console program
/// itself (`TwoPassPso::native`). The mesh program and the pixel shader take different user
/// SGPRs (`console_list::user_data_ms` / `user_data_gfx`) and one root signature has one `io`:
/// the pixel shader's block first, as in any draw, then the mesh program's
/// (`recomp::MeshNative::io_base`). `res` is not used (window 0 stands in).
unsafe fn bind_native_mesh(this: *mut c_void, list: &ID3D12GraphicsCommandList, real_mesh_pso: usize, tp: &super::real::TwoPassPso) -> Result<(), String> {
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take_or(u, bytes.len().max(16), 256, &mut s.uma, super::sparse::arena_block)).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    // lab: what the program reads of the title's memory outside the unified windows
    super::view_mem::ensure_for_draw(layout.as_deref(), &args);
    let mut io = Vec::with_capacity(2 * crate::gpu::recomp::KERNEL_SGPRS);
    for mesh in [false, true] {
        let inline = layout.as_ref().map_or(0, |l| if mesh { l.inline_ms } else { l.inline_gfx } as usize).min(args.len());
        let table_at = copy(&args[inline..])?;
        let mut block = match layout.as_ref() {
            Some(l) if mesh => console_list::user_data_ms(l, &args, table_at as u64),
            Some(l) => console_list::user_data_gfx(l, &args, table_at as u64),
            None if mesh => {
                let mut v = vec![0u32; 16];
                v[8] = table_at as u32;
                v[9] = (table_at >> 32) as u32;
                v
            }
            None => vec![table_at as u32, (table_at >> 32) as u32],
        };
        let user = block.len();
        heap_sgpr(this, &mut block, layout.is_some());
        if !mesh {
            ps_probe(tp, &block);
            if block.len() > user {
                draw_textures(&block, user);
            }
        }
        block.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
        block[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
        io.extend(block);
    }
    let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    let rs_raw = tp.mesh_rs as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs_raw).ok_or("native mesh root signature")?;
    list.SetGraphicsRootSignature(rs);
    let views = [super::real::uma_va_of(io_at).unwrap_or(first), first].into_iter().chain((0..4).map(|k| windows.get(k).copied().filter(|&v| v != 0).unwrap_or(first)));
    for (k, v) in views.enumerate() {
        list.SetGraphicsRootUnorderedAccessView(k as u32, v);
    }
    let pso_raw = real_mesh_pso as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&pso_raw).ok_or("native mesh pipeline")?;
    list.SetPipelineState(pc);
    Ok(())
}

/// Native mesh draws so far (direct, indirect), logged at powers of two.
fn native_mesh_drawn(indirect: bool, groups: u32) {
    static N: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    let n = N[indirect as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: native mesh {} #{n} ({groups} {})", if indirect { "ExecuteIndirect(DispatchMesh)" } else { "DispatchMesh" }, if indirect { "records at most" } else { "groups" });
    }
}

unsafe fn scale_indirect_pipeline(dev: &ID3D12Device) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    static CACHE: Mutex<Option<HashMap<usize, (ID3D12RootSignature, ID3D12PipelineState)>>> = Mutex::new(None);
    let key = dev.as_raw() as usize;
    if let Some(pair) = CACHE.lock().unwrap().as_ref().and_then(|m| m.get(&key).cloned()) {
        return Ok(pair);
    }
    let hlsl = "cbuffer C : register(b0) { uint count; uint stride; uint offset; uint mult; };\nByteAddressBuffer src : register(t0);\nRWByteAddressBuffer dst : register(u0);\n[numthreads(64, 1, 1)]\nvoid main(uint tid : SV_DispatchThreadID) {\n    if (tid >= count) return;\n    uint base = tid * stride;\n    uint dwords = stride / 4u;\n    for (uint i = 0u; i < dwords; ++i) {\n        uint w = src.Load(base + i * 4u);\n        if (i * 4u == offset) w = w * mult;\n        dst.Store(base + i * 4u, w);\n    }\n}\n";
    let dxil = crate::gpu::recomp::dxc::compile_cached(hlsl, "main", "cs_6_6")?.0;
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        ..Default::default()
    };
    let mut blob = None;
    D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).map_err(|e| format!("scale rs serialize: {e}"))?;
    let blob = blob.ok_or("scale rs blob")?;
    let rs: ID3D12RootSignature = dev
        .CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()))
        .map_err(|e| format!("scale rs create: {e}"))?;
    let mut cdesc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let res_pso: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&cdesc);
    std::mem::ManuallyDrop::drop(&mut cdesc.pRootSignature);
    let pso = res_pso.map_err(|e| format!("scale pso create: {e}"))?;
    // LOCAL (claudetools-gh, #2952 entry 6): the first pair put in the cache wins. With `insert` a
    // second thread that built its own at the same moment replaced the first one, which was then
    // freed while lists recorded with it were still to run (a GPU hang at world entry).
    Ok(CACHE.lock().unwrap().get_or_insert_with(HashMap::new).entry(key).or_insert((rs, pso)).clone())
}

unsafe fn make_scaled_args_buffer(this: *mut c_void, list: &ID3D12GraphicsCommandList, src_raw: usize, src_off: u64, max: u32, stride: u32, mesh_offset: u32, mult: u32) -> Result<ID3D12Resource, String> {
    let dev = super::real::list_device(this).ok_or("no device")?;
    let (rs, pso) = scale_indirect_pipeline(&dev)?;
    let src_ptr = src_raw as *mut c_void;
    let src = ID3D12Resource::from_raw_borrowed(&src_ptr).ok_or("indirect args buffer")?;
    let bytes = (max as u64).saturating_mul(stride as u64).max(256).next_multiple_of(256);
    let split = split_free_lists();
    let (dst, bytes) = if split {
        let (b, cap, _) = scaled_args_buffer(&dev, bytes)?;
        (b, cap)
    } else {
        (default_uav_buffer(&dev, bytes, "scaled args buffer")?, bytes)
    };
    emit_transition(list, src, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list.SetComputeRootSignature(&rs);
    list.SetPipelineState(&pso);
    let consts = [max, stride, mesh_offset, mult];
    list.SetComputeRoot32BitConstants(0, 4, consts.as_ptr() as *const c_void, 0);
    list.SetComputeRootShaderResourceView(1, src.GetGPUVirtualAddress() + src_off);
    list.SetComputeRootUnorderedAccessView(2, dst.GetGPUVirtualAddress());
    list.Dispatch(max.div_ceil(64), 1, 1);
    emit_transition(list, &dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    emit_transition(list, src, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    let dev_key = dev.as_raw() as usize;
    if split {
        // (held for the allocator this recording is on, as the record buffers are at its reset)
        let allocator = with(this, |s| s.allocator);
        if allocator != 0 {
            ALLOC_SCALED.lock().unwrap().get_or_insert_with(HashMap::new).entry(allocator).or_default().push((dst.clone(), bytes, dev_key));
        }
    } else {
        with(this, |s| s.res_retired.push((dst.clone(), bytes, dev_key)));
    }
    Ok(dst)
}

pub(super) unsafe extern "system" fn dispatch_mesh(this: *mut c_void, x: u32, y: u32, z: u32) {
    type PcDm = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
    let pc_dm: PcDm = std::mem::transmute(super::real::orig_fn(this, 79));
    let raw_pso = with(this, |s| s.pso);
    let pso = pso_for_draw(raw_pso);
    if let Some(tp) = super::real::two_pass_of(pso) {
        let groups = x.saturating_mul(y.max(1)).saturating_mul(z.max(1));
        if groups == 0 {
            return;
        }
        let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
        if tp.native {
            // `XCL_NATIVE_MESH`: the console program is the pipeline's mesh shader – one draw
            match bind_native_mesh(this, list, pso, &tp) {
                Ok(()) => {
                    pc_dm(this, groups.min(65535), 1, 1);
                    super::targets::note_draw(this, super::targets::RAN);
                    count(DRAWS);
                    native_mesh_drawn(false, groups);
                }
                Err(e) => {
                    super::targets::note_draw(this, super::targets::SKIPPED);
                    skip_note(this, line!());
                    count(DRAWS_SKIPPED);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::xlog_warn!("D3D12.X: native mesh DispatchMesh skipped: {e}");
                    }
                }
            }
            return;
        }
        // one record per group into `res` (a root UAV: no bounds check), whose size
        // `ensure_res_buffer` caps at 128 MiB: groups past it would write past the buffer
        let rec_bytes = (tp.record.record_dwords() as u64 * 4).max(4);
        let fit = ((128u64 << 20) / rec_bytes).min(u64::from(u32::MAX)) as u32;
        if groups > fit {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::xlog_warn!("D3D12.X: two-pass DispatchMesh of {groups} groups drawn as {fit}: the record buffer holds no more");
            }
        }
        let groups = groups.min(fit);
        let need = (groups as u64).saturating_mul(rec_bytes);
        let run = (|| -> Result<(), String> {
            let (res_buf, res_va) = ensure_res_buffer(this, need)?;
            ms_probe(this, "DM", &tp, groups);
            gp_mark(this, gp::Label::of(gp::Kind::Other));
            bind_two_pass_stage(this, list, pso, &tp, res_va, true)?;
            list.Dispatch(groups, 1, 1);
            emit_uav_barrier(list, Some(&res_buf));
            gp_mark(this, gp::Label::op(gp::Kind::TwoPassCs, gp_two_pass(&tp), [groups, 0, 0]));
            {
                // lab: as above, for a direct two-pass DispatchMesh (kinds 8..11)
                let (r, vd) = (res_buf.as_raw() as usize, tp.record.vert_dwords() as u64 * 4);
                ei_peek(this, 8, groups, r, 0, 0, 0);
                ei_peek(this, 9, groups, r, 16, r, 28);
                ei_peek(this, 10, groups, r, 16 + vd, r, 28 + vd);
                ei_peek(this, 11, groups, r, 16 + 2 * vd, r, 28 + 2 * vd);
            }
            bind_two_pass_stage(this, list, pso, &tp, res_va, false)?;
            pc_dm(this, groups.saturating_mul(tp.mesh_groups_per_record), 1, 1);
            emit_uav_barrier(list, Some(&res_buf));
            gp_mark(this, gp::Label::op(gp::Kind::TwoPassMs, gp_two_pass(&tp), [groups, 0, 0]));
            Ok(())
        })();
        match run {
            Ok(()) => {
                super::targets::note_draw(this, super::targets::RAN);
                count(DRAWS);
                static LOGGED: AtomicU64 = AtomicU64::new(0);
                let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 8 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: two-pass DispatchMesh #{n} ({groups} groups, split {}, kind {})", tp.mesh_groups_per_record, tp.kind);
                }
            }
            Err(e) => {
                super::targets::note_draw(this, super::targets::SKIPPED);
                skip_note(this, line!());
                count(DRAWS_SKIPPED);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!("D3D12.X: two-pass DispatchMesh skipped: {e}");
                }
            }
        }
        return;
    }
    if super::pso_async::is_pending(raw_pso) {
        super::targets::note_draw(this, super::targets::STANDIN);
        skip_note(this, line!());
        count(DRAWS_SKIPPED);
        return;
    }
    let rs = with(this, |s| s.rs_gfx) as *mut c_void;
    if let (Some(list), Some(r)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), ID3D12RootSignature::from_raw_borrowed(&rs)) {
        list.SetGraphicsRootSignature(r);
    }
    super::targets::note_draw(this, super::targets::PC);
    count(DRAWS_PC);
    gp_mark(this, gp::Label::of(gp::Kind::Other));
    pc_dm(this, x, y, z);
    gp_mark(this, gp::Label::op(gp::Kind::PcDraw, (0, 0), [x, y, z]));
}

/// #59 ExecuteIndirect(signature, max count, arguments, offset, count buffer, offset): with a
/// translated pipeline, on XCL's ABI with the kept root arguments (signatures that change root
/// arguments themselves cannot be – skipped, logged)
pub(super) unsafe extern "system" fn execute_indirect(this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    let pc: F = std::mem::transmute(super::real::orig_fn(this, 59));
    super::real::use_buffer(args);
    super::real::use_buffer(count_buf);
    let raw_pso = with(this, |s| s.pso);
    let unbuilt_compute = signature_dispatches(sig) && super::pso_async::is_pending(raw_pso) && super::real::is_console_compute(raw_pso);
    if unbuilt_compute && super::list_x::defer(this, super::list_x::Deferred::Indirect(sig, max, args, args_off, count_buf, count_off)) {
        return super::pso_async::need_by_submit(raw_pso);
    }
    // (#1798: a graphics pipeline still being built is not waited for; a compute one is)
    let pso = pso_for_draw(raw_pso);
    // (any other ExecuteIndirect ends a run of two-pass ones: a dispatch may write their records)
    if (super::clamp_runs::on() || super::indirect_clamp::batch_on()) && !(super::real::two_pass_of(pso).is_some() && dispatch_mesh_sig_of(sig).is_some()) {
        super::clamp_runs::write(this as usize);
        super::indirect_clamp::run_stop(this as usize);
    }
    let (console_args_off, console_count_off) = (args_off, count_off);
    let (args_off, count_off) = (args_off + super::real::buffer_delta(args), count_off + super::real::buffer_delta(count_buf));
    if let Some(tp) = super::real::two_pass_of(pso) {
        if let Some(msig) = dispatch_mesh_sig_of(sig) {
            let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
            super::clamp_runs::call(this as usize);
            // the records bounded on the GPU first, as for other ExecuteIndirect calls below: a
            // garbage record hung the GPU here (0x887a0006). Both passes read the bounded copy.
            // The compute pass writes one record per group into `res` (a root UAV: no bounds
            // check), so the groups are also limited to the records the buffer holds.
            // lab: `XCL_TWO_PASS_RES_MB` (default 32)
            static RES_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            #[allow(non_snake_case)]
            let RES_BYTES: u64 = *RES_MB.get_or_init(|| std::env::var("XCL_TWO_PASS_RES_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(32u64).clamp(1, 128)) << 20;
            let rec_bytes = (tp.record.record_dwords() as u64 * 4).max(4);
            let budget = (RES_BYTES / rec_bytes).saturating_mul(128).min(u64::from(super::indirect_clamp::budget())) as u32;
            // (`XCL_NATIVE_MESH`: no record buffer to fit, the budget of any other call)
            let budget = if tp.native { super::indirect_clamp::budget() } else { budget };
            ei_peek(this, 3, max, args, args_off + u64::from(msig.mesh_offset), count_buf, count_off);
            // lab: `XCL_TWO_PASS_UNBOUNDED=1`: the passes read the title's records as they are
            static UNBOUNDED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let unbounded = *UNBOUNDED.get_or_init(|| std::env::var_os("XCL_TWO_PASS_UNBOUNDED").is_some());
            gp_mark(this, gp::Label::of(gp::Kind::Other));
            let bounded = if unbounded { None } else { super::indirect_clamp::bound_in_run(list, with(this, |s| s.allocator), sig, max, args, console_args_off, count_buf, console_count_off, budget) };
            if bounded.is_none() && !unbounded {
                // records we cannot bound could write past `res` and hang the GPU (0x887a0006)
                ei_work(this, sig, super::targets::SKIPPED);
                skip_note(this, line!());
                count(DRAWS_SKIPPED);
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 8 {
                    crate::xlog_warn!("D3D12.X: two-pass ExecuteIndirect skipped: its records could not be bounded");
                }
                return;
            }
            let gp_name = gp_two_pass(&tp);
            gp_mark(this, gp::Label::op(gp::Kind::Clamp, gp_name, [max, 1, 0]));
            let (args, args_off) = match &bounded {
                Some(b) => (b.raw(), b.offset()),
                None => (args, args_off),
            };
            // a run's copy (`XCL_CLAMP_BATCH`) holds the count too, and has been readable since
            // the run's pass: its calls need no wait of their own
            let (count_buf, count_off) = match bounded.as_ref().and_then(|b| b.count().map(|at| (b.raw(), at))) {
                Some(c) => c,
                None => (count_buf, count_off),
            };
            let in_run = bounded.as_ref().is_some_and(|b| b.batched());
            let ordered = |op: &dyn Fn()| if in_run { op() } else { fresh_indirect_arguments(this, || op()) };
            ei_peek(this, 0, max, args, args_off + u64::from(msig.mesh_offset), count_buf, count_off);
            // lab: the groups of a single record as the CPU sees them now (a record the GPU
            // writes later may differ): the record buffer is made for them when they need more
            // than `RES_BYTES` (unbounded, the compute pass wrote past a smaller buffer)
            let cpu_groups = if unbounded && max == 1 && count_buf == 0 {
                super::real::console_buffer(args).and_then(|(p, len)| {
                    let o = console_args_off as usize + msig.mesh_offset as usize;
                    (o + 12 <= len && crate::gpu::shader::readable(p + o, 12)).then(|| ((p + o) as *const [u32; 3]).read_unaligned())
                })
            } else {
                None
            };
            let cpu_need = cpu_groups.map_or(0, |g| u64::from(g[0]).saturating_mul(u64::from(g[1].max(1))).saturating_mul(u64::from(g[2].max(1))).saturating_mul(rec_bytes));
            {
                static SEEN: AtomicU64 = AtomicU64::new(0);
                static UNKNOWN: AtomicU64 = AtomicU64::new(0);
                static OVER: AtomicU64 = AtomicU64::new(0);
                static MOST: AtomicU64 = AtomicU64::new(0);
                let n = SEEN.fetch_add(1, Ordering::Relaxed) + 1;
                if cpu_groups.is_none() {
                    UNKNOWN.fetch_add(1, Ordering::Relaxed);
                }
                if cpu_need > RES_BYTES {
                    OVER.fetch_add(1, Ordering::Relaxed);
                }
                let most = MOST.fetch_max(cpu_need, Ordering::Relaxed).max(cpu_need);
                if n.is_power_of_two() || (cpu_need > RES_BYTES && cpu_need == most) {
                    crate::xlog_info!(
                        "D3D12.X: two-pass EI records on the CPU: {n} calls, {} not readable, {} need more than {} MiB, most {} KiB (this one {:x?}, record {} B)",
                        UNKNOWN.load(Ordering::Relaxed), OVER.load(Ordering::Relaxed), RES_BYTES >> 20, most >> 10, cpu_groups, rec_bytes
                    );
                }
            }
            let run = (|| -> Result<(), String> {
                if tp.native {
                    // `XCL_NATIVE_MESH`: the console program is the pipeline's mesh shader – the
                    // title's call on the bounded records, nothing before or after it
                    bind_native_mesh(this, list, pso, &tp)?;
                    fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
                    native_mesh_drawn(true, max);
                    return Ok(());
                }
                let (res_buf, res_va) = ensure_res_buffer(this, RES_BYTES.max(cpu_need))?;
                ms_probe(this, "EI", &tp, max);
                bind_two_pass_stage(this, list, pso, &tp, res_va, true)?;
                // both passes read the records a dispatch before them wrote (the title's own, or
                // the bounded copy): ordered after those writes as any other ExecuteIndirect is
                // (`fresh_indirect_arguments`; NVIDIA runs the old records otherwise)
                ordered(&|| pc(this, msig.companion_dispatch, max, args, args_off, count_buf, count_off));
                emit_uav_barrier(list, Some(&res_buf));
                gp_mark(this, gp::Label::op(gp::Kind::TwoPassCs, gp_name, [max, 0, 0]));
                {
                    // lab: record 0 as the compute pass left it (kind 4: vertices, primitives, spare; 5/6/7: position of vertex 0/1/2, w as the "count")
                    let (r, vd) = (res_buf.as_raw() as usize, tp.record.vert_dwords() as u64 * 4);
                    ei_peek(this, 4, max, r, 0, 0, 0);
                    ei_peek(this, 5, max, r, 16, r, 28);
                    ei_peek(this, 6, max, r, 16 + vd, r, 28 + vd);
                    ei_peek(this, 7, max, r, 16 + 2 * vd, r, 28 + 2 * vd);
                }
                if tp.mesh_groups_per_record > 1 && max > 0 && args != 0 {
                    let scaled = make_scaled_args_buffer(this, list, args, args_off, max, msig.byte_stride, msig.mesh_offset, tp.mesh_groups_per_record)?;
                    gp_mark(this, gp::Label::op(gp::Kind::TwoPassScale, gp_name, [max, 0, 0]));
                    bind_two_pass_stage(this, list, pso, &tp, res_va, false)?;
                    fresh_indirect_arguments(this, || pc(this, sig, max, scaled.as_raw() as usize, 0, count_buf, count_off));
                } else {
                    bind_two_pass_stage(this, list, pso, &tp, res_va, false)?;
                    ordered(&|| pc(this, sig, max, args, args_off, count_buf, count_off));
                }
                emit_uav_barrier(list, Some(&res_buf));
                gp_mark(this, gp::Label::op(gp::Kind::TwoPassMs, gp_name, [max, 0, 0]));
                Ok(())
            })();
            if let Some(b) = bounded {
                b.finish(list);
                gp_mark(this, gp::Label::op(gp::Kind::ClampFinish, gp_name, [max, 1, 0]));
            }
            match run {
                Ok(()) => {
                    ei_work(this, sig, super::targets::RAN);
                    count(DRAWS);
                    static LOGGED: AtomicU64 = AtomicU64::new(0);
                    let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 8 || n.is_power_of_two() {
                        crate::xlog_info!("D3D12.X: two-pass ExecuteIndirect(DispatchMesh) #{n} (max {max}, split {}, kind {})", tp.mesh_groups_per_record, tp.kind);
                    }
                }
                Err(e) => {
                    ei_work(this, sig, super::targets::SKIPPED);
                    skip_note(this, line!());
                    count(DRAWS_SKIPPED);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, Ordering::Relaxed) < 8 {
                        crate::xlog_warn!("D3D12.X: two-pass ExecuteIndirect skipped: {e}");
                    }
                }
            }
            return;
        }
    }
    let (gfx, compute) = (super::real::is_translated_graphics(pso), super::real::is_translated_compute(pso));
    if !gfx && !compute {
        // A pipeline still being built (or whose build failed): its stand-in draws nothing, and
        // the list may hold XCL's ABI root signature from a translated draw before it – the PC
        // call with that pair is invalid (debug layer 953, "the currently set Root Signature
        // doesn't match the currently set Pipeline State Object"; a retail title's world then
        // hung the GPU, 0x887a0006). Skipped, like a draw or dispatch of such a pipeline.
        if super::pso_async::is_pending(raw_pso) {
            ei_work(this, sig, super::targets::STANDIN);
            skip_note(this, line!());
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped");
            }
            return;
        }
        // A finished two-pass MS/GS pipeline is a PC mesh shader pipeline (MS+PS) registered in
        // TWO_PASS rather than TRANSLATED_GRAPHICS. Falling through to raw PC ExecuteIndirect
        // with the title's graphics root signature (e.g. a non-DispatchMesh indirect draw or a
        // root-changing signature) invalidates the command list (Close → E_INVALIDARG).
        if let Some(tp) = super::real::two_pass_of(pso) {
            skip_note(this, line!());
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!(
                    "D3D12.X: ExecuteIndirect with two-pass {} pipeline ({pso:#x}) skipped – PC pipeline is a mesh shader",
                    tp.kind
                );
            }
            return;
        }
        // a console compute program without a translation (its build failed): nothing to run,
        // as for Dispatch. The PC call would run it with the list's compute root signature, which
        // can still be XCL's ABI one from a translated dispatch before it (debug layer 953).
        // A native PC compute pipeline is not one of these and still goes to the PC call.
        // (`CONSOLE_COMPUTE` holds the title's object, which for a shared stand-in differs from
        // the driver pipeline `resolve_wait` returned.)
        if signature_dispatches(sig) && (super::real::is_console_compute(raw_pso) || super::real::is_console_compute(pso)) {
        super::targets::note_dispatch();
            skip_note(this, line!());
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect dispatch with a pipeline that has no translated console shader ({raw_pso:#x}) skipped");
            }
            return;
        }
        // a PC pipeline: the title's root signature, as for a draw (`graphics_bound`)
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let (Some(list), Some(r)) = (ID3D12GraphicsCommandList::from_raw_borrowed(&this), ID3D12RootSignature::from_raw_borrowed(&rs)) {
            list.SetGraphicsRootSignature(r);
        }
        ei_work(this, sig, super::targets::PC);
        gp_mark(this, gp::Label::of(gp::Kind::Other));
        fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
        gp_mark(this, gp::Label::op(gp::Kind::PcIndirect, (0, 0), [max, 0, 0]));
        return;
    }
    if ROOT_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig)) {
        ei_work(this, sig, super::targets::SKIPPED);
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect that changes root arguments with a translated pipeline skipped");
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    // the records bounded on the GPU first (a garbage record would hang it): this changes the
    // list's compute bindings, which `bind_abi` sets again
    // lab (`XCL_FOLD_GROUPS`): a program with a folded group id gets its wide records as rows
    let fold = if !gfx && super::real::pso_shaders(pso).is_some_and(|s| crate::gpu::dxil::kernel::fold_groups(s.0)) { crate::gpu::dxil::kernel::FOLD_ROW } else { 0 };
    let gp_name = if gp::on() { super::real::pso_shaders(super::pso_async::resolve(pso)).unwrap_or((0, 0)) } else { (0, 0) };
    let gp_kind = if gfx { gp::Kind::IndirectDraw } else { gp::Kind::IndirectDispatch };
    gp_mark(this, gp::Label::of(gp::Kind::Other));
    let bounded = super::indirect_clamp::with_fold(fold, || super::indirect_clamp::bound(list, with(this, |s| s.allocator), sig, max, args, console_args_off, count_buf, console_count_off));
    if bounded.is_some() {
        gp_mark(this, gp::Label::op(gp::Kind::Clamp, gp_name, [max, 0, 0]));
    }
    match &bounded {
        Some(b) => ei_peek(this, if gfx { 1 } else { 2 }, max, b.raw(), b.offset(), count_buf, count_off),
        None => ei_peek(this, if gfx { 1 } else { 2 }, max, args, args_off, count_buf, count_off),
    }
    // lab (`XCL_EI_PEEK`): the first 3 calls of each pipeline with its shader (PS / CS record hash)
    if ei_peek_from().is_some_and(|t| crate::log::uptime() >= t) {
        if let Some((a, b)) = super::real::pso_shaders(pso) {
            let hash = if gfx { b } else { a };
            static SEEN: Mutex<Option<HashMap<u64, u32>>> = Mutex::new(None);
            let n = {
                let mut g = SEEN.lock().unwrap();
                let n = g.get_or_insert_with(HashMap::new).entry(hash).or_insert(0);
                *n += 1;
                *n
            };
            if n <= 3 {
                let slot = match &bounded {
                    Some(b) => ei_peek_slot(this, if gfx { 20 } else { 21 }, max, b.raw(), b.offset(), count_buf, count_off),
                    None => ei_peek_slot(this, if gfx { 20 } else { 21 }, max, args, args_off, count_buf, count_off),
                };
                if let Some(slot) = slot {
                    EI_NAMED.lock().unwrap().push((slot, hash, gfx));
                }
                // lab: the SAME call's record as the title wrote it, next to the bounded one
                // ("ei pair"): the bounded record of a folded program read [0 | 174 | 1, 1, 1]
                // where the title's records were said to ask for ~100000 groups
                if let (Some(b), Some(_)) = (slot, &bounded) {
                    if let Some(t) = ei_peek_slot(this, 41, max, args, args_off, count_buf, count_off) {
                        EI_PAIRS.lock().unwrap().push((t, b, hash));
                    }
                }
            }
            // lab: a compute call's record as the title wrote it (before it is bounded), up to 64
            // a program: how many groups its dispatches ask for (D3D12 takes 65535 a dimension,
            // `indirect_clamp` cuts there; the console takes more)
            if !gfx && n <= 64 && bounded.is_some() {
                if let Some(slot) = ei_peek_slot(this, 40, max, args, args_off, count_buf, count_off) {
                    EI_WIDE.lock().unwrap().push((slot, hash));
                }
            }
        }
    }
    if gfx {
        rt_writers_probe(this, pso);
    }
    if compute && !gfx {
        // lab: indirect dispatches in the probes too (groups: u32::MAX, max count, 0)
        dispatch_probe(this, pso, [u32::MAX, max, 0]);
        rt_dump_note_compute(this);
    }
    let twins = if compute && !gfx { depth_twins_named(this) } else { Vec::new() };
    for &(res, plane) in &twins {
        super::list_x::depth_twin_fill(this, res, plane);
    }
    if compute && !gfx {
        pass_fill_at(this, pso, true);
    }
    let timed = if compute && !gfx { gpu_time_begin(this, pso) } else { None };
    // (a compute program split into a dispatch chain runs whole through ExecuteIndirect: `dispatch_chain::whole`)
    let result = bind_abi(this, list, pso, gfx).and_then(|()| {
        if compute {
            if let Some(whole) = super::real::dispatch_chain::whole(pso).map_err(|e| format!("whole compute PSO: {e}"))? {
                list.SetPipelineState(&whole);
            }
        }
        Ok(())
    });
    if let Ok(()) = result {
        ei_work(this, sig, super::targets::RAN);
        count(if gfx { DRAWS } else { DISPATCHES });
        gp_mark(this, gp::Label::of(gp::Kind::Other));
        fresh_indirect_arguments(this, || match &bounded {
            Some(b) => pc(this, sig, max, b.raw(), b.offset(), count_buf, count_off),
            None => pc(this, sig, max, args, args_off, count_buf, count_off),
        });
        gp_mark(this, gp::Label::op(gp_kind, gp_name, [max, 0, 0]));
        for (res, plane) in twins {
            super::list_x::depth_twin_flush(this, res, plane);
        }
        gpu_time_end(this, timed);
        if compute && !gfx {
            pass_snap(this, pso);
            pass_fill(this, pso);
        }
    }
    if let Some(b) = bounded {
        b.finish(list);
        gp_mark(this, gp::Label::op(gp::Kind::ClampFinish, gp_name, [max, 0, 0]));
    }
    if let Err(e) = result {
        ei_work(this, sig, super::targets::SKIPPED);
        skip_note(this, line!());
        count(if gfx { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect with a translated pipeline skipped: {e}");
        }
    }
}

// lab: `XCL_EI_PEEK=t` (seconds of uptime): from then on the arguments (12 bytes at the action)
// and the count of up to 4096 ExecuteIndirect calls are copied to a readback buffer when the
// call runs on the GPU, and logged 8, 16 and 30 s after the first.
struct EiPeek {
    rb: ID3D12Resource,
    /// (uptime, kind 0 two-pass mesh / 1 graphics / 2 compute, max, has count)
    meta: Vec<(f64, u8, u32, bool)>,
}
unsafe impl Send for EiPeek {}
static EI_PEEK: Mutex<Option<EiPeek>> = Mutex::new(None);
const EI_PEEK_SLOTS: usize = 32768;

fn ei_peek_from() -> Option<f64> {
    static T: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("XCL_EI_PEEK").ok().and_then(|v| v.trim().parse().ok()))
}

unsafe fn ei_peek(this: *mut c_void, kind: u8, max: u32, args: usize, args_at: u64, count_buf: usize, count_off: u64) {
    let Some(from) = ei_peek_from() else { return };
    if crate::log::uptime() < from || args == 0 {
        return;
    }
    // every n-th call of a kind (`XCL_EI_PEEK_EVERY`, default 4)
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    static CALLS: [AtomicU64; 16] = [Z; 16];
    let every = *EVERY.get_or_init(|| std::env::var("XCL_EI_PEEK_EVERY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4u64).max(1));
    if CALLS[kind as usize & 15].fetch_add(1, Ordering::Relaxed) % every != 0 {
        return;
    }
    ei_peek_slot(this, kind, max, args, args_at, count_buf, count_off);
}

/// lab: one peek slot (no sampling); its index
unsafe fn ei_peek_slot(this: *mut c_void, kind: u8, max: u32, args: usize, args_at: u64, count_buf: usize, count_off: u64) -> Option<usize> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?;
    let mut g = EI_PEEK.lock().unwrap();
    if g.is_none() {
        let dev = super::real::list_device(this)?;
        let mut rb: Option<ID3D12Resource> = None;
        let made = dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: (EI_PEEK_SLOTS * 32) as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        );
        let (Ok(()), Some(rb)) = (made, rb) else { return None };
        // marks: a slot the GPU has not written yet reads 0xEEEEEEEE
        let mut p: *mut c_void = std::ptr::null_mut();
        if rb.Map(0, None, Some(&mut p)).is_ok() && !p.is_null() {
            std::ptr::write_bytes(p as *mut u8, 0xEE, EI_PEEK_SLOTS * 32);
            rb.Unmap(0, None);
        }
        *g = Some(EiPeek { rb, meta: Vec::new() });
        let _ = std::thread::Builder::new().name("xcl-ei-peek".into()).spawn(|| {
            for wait in [8u64, 8, 14, 15, 15, 20, 20, 30] {
                std::thread::sleep(std::time::Duration::from_secs(wait));
                ei_peek_report();
            }
        });
    }
    let pk = g.as_mut()?;
    let slot = pk.meta.len();
    if slot >= EI_PEEK_SLOTS {
        return None;
    }
    let a = args as *mut c_void;
    let a = ID3D12Resource::from_raw_borrowed(&a)?;
    list.CopyBufferRegion(&pk.rb, (slot * 32) as u64, a, args_at, 12);
    let c = count_buf as *mut c_void;
    let has_count = match ID3D12Resource::from_raw_borrowed(&c) {
        Some(c) if count_buf != 0 => {
            list.CopyBufferRegion(&pk.rb, (slot * 32 + 16) as u64, c, count_off, 4);
            true
        }
        _ => false,
    };
    pk.meta.push((crate::log::uptime(), kind, max, has_count));
    Some(slot)
}

/// lab (`XCL_DRAW_TOUCH=0`: not): the textures a draw of the retail title samples, told to the
/// page budget (`title_textures::drawn`). Its pixel shaders index the view heap themselves:
/// the material's record – at the offset in the last root argument word (`io[heap - 1]`), in
/// the buffer whose V# is at the global table (s8) + 0xac0 – holds the heap slots of its
/// textures (239 of the 362 dumped PS load that V#; the record offsets they read are 0..64),
/// T# = view heap (`io[heap]`) + slot * 32. Every dword of the record's first 68 bytes that is
/// the slot of an image T# naming a texture on pages counts (one that is no slot names none,
/// or at worst keeps one more texture). A record is looked at once in 250 ms.
unsafe fn draw_textures(io: &[u32], heap: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| !std::env::var("XCL_DRAW_TOUCH").is_ok_and(|v| v.trim() == "0")) || !super::vram::tracking() || heap < 10 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let at = |a: u64| if a >> 32 == 0 { u.base() + a as usize } else { a as usize };
    let cpu = |p: usize, n: usize| -> Option<Vec<u32>> {
        crate::gpu::shader::readable(p, 4 * n).then(|| (0..n).map(|i| ((p + 4 * i) as *const u32).read_unaligned()).collect())
    };
    let (table, record, view) = (io[8], io[heap - 1], io[heap]);
    {
        static SEEN: Mutex<Option<HashMap<(u32, u32), std::time::Instant>>> = Mutex::new(None);
        let now = std::time::Instant::now();
        let mut g = SEEN.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        if m.get(&(table, record)).is_some_and(|t| now.duration_since(*t) < std::time::Duration::from_millis(250)) {
            return;
        }
        if m.len() > 1 << 16 {
            m.clear();
        }
        m.insert((table, record), now);
    }
    let Some(v) = cpu(at(table as u64) + 0xac0, 4) else { return };
    let stride = (v[1] >> 16) & 0x3FFF;
    let limit = if stride == 0 { v[2] } else { v[2].wrapping_mul(stride) };
    let n = (limit.saturating_sub(record) / 4).min(17) as usize;
    if n == 0 {
        return;
    }
    let buf = at((u64::from(v[1] & 0xFFFF) << 32) | u64::from(v[0]));
    let Some(rec) = cpu(buf.wrapping_add(record as usize), n) else { return };
    let mut named = Vec::new();
    for slot in rec {
        if slot == 0 || slot >= 1 << 22 {
            continue;
        }
        // an image descriptor (as `console_heap::referenced` tells one)
        let Some(t) = cpu(at(view as u64) + slot as usize * 32, 4).filter(|t| t[3] >> 28 >= 8) else { continue };
        if let Some(res) = super::console_heap::index_resource(t[0]) {
            if !named.contains(&res) {
                named.push(res);
            }
        }
    }
    super::title_textures::drawn(&named);
}

/// LOCAL (`XCL_SKIN_INPUTS=1`; `=probe`: log only): what the title's ped skinning compute programs
/// read through the view heap, which no table names: two buffers (heap slots at +0x40 / +0x44 of
/// the buffer whose V# is in slot 8 of the second table, s5) – kept in view memory
/// (`view_mem::ensure`) – and one 2-D texture per record (heap slot at +0x8c of each 0x90-byte
/// record of the buffer in slot 8 of the first table, s4; count: dword 0 of the buffer in its
/// slot 3) – told to the page budget (`title_textures::drawn`). `heap`: the index of the view
/// heap's SGPR in `io`.
unsafe fn skin_inputs(pso: usize, io: &[u32], heap: usize) {
    const SKINNING: [u64; 10] = [
        0x1404420b38db8597, 0xcc047f458010c5d2, 0x34207613d17f8b96, 0x94f612c85468fb5b, 0x36353c66e9666102,
        0x44ca55bfefe67bd5, 0xbd968d88fabf544f, 0x9144739038da8a66, 0x4585e350d1b864e3, 0xb6101d83892d44a5,
    ];
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| match std::env::var("XCL_SKIN_INPUTS").as_deref().map(str::trim) {
        Ok("1") => 2,
        Ok("probe") => 1,
        _ => 0,
    });
    if mode == 0 || heap < 2 || heap >= io.len() {
        return;
    }
    let Some((cs, _)) = super::real::pso_shaders(super::pso_async::resolve(pso)).filter(|s| SKINNING.contains(&s.0)) else { return };
    let Some(u) = crate::gpu::uma::get() else { return };
    let at = |a: u64| if a >> 32 == 0 { u.base() + a as usize } else { a as usize };
    // (a buffer in the title's GPU view is read where the CPU has its bytes)
    let cpu = |p: usize, n: usize| -> Option<Vec<u32>> {
        let p = if u.window_of(p).is_some() { p } else { super::view_mem::cpu_at(p)? };
        crate::gpu::shader::readable(p, 4 * n).then(|| (0..n).map(|i| ((p + 4 * i) as *const u32).read_unaligned()).collect())
    };
    let va = |d: &[u32]| ((d[1] as usize & 0xFFFF) << 32) | d[0] as usize;
    let bytes = |d: &[u32]| {
        let stride = (d[1] >> 16) as usize & 0x3FFF;
        if stride == 0 { d[2] as usize } else { (d[2] as usize).saturating_mul(stride) }
    };
    let (a, b, view) = (at(io[heap - 2] as u64), at(io[heap - 1] as u64), at(io[heap] as u64));
    let mut line = String::new();
    // the two buffers
    if let Some(v) = cpu(b + 0x100, 4).filter(|v| v[3] >> 30 == 0 && (v[0] | v[1]) != 0) {
        if let Some(slots) = cpu(va(&v) + 0x40, 2) {
            for slot in slots {
                let Some(d) = cpu(view + slot as usize * 32, 4).filter(|d| slot < 1 << 22 && d[3] >> 30 == 0) else {
                    line += &format!(" buf slot {slot:#x}: no V#;");
                    continue;
                };
                let (p, n) = (va(&d), bytes(&d).clamp(4, 64 << 20));
                line += &format!(" buf slot {slot:#x}: {p:#x}+{n:#x} win {:?}", u.window_of(p));
                if u.window_of(p).is_none() {
                    // which of its 64 KiB pages the CPU cannot read (the GPU then reads zeros there)
                    let pages: Vec<usize> = ((p & !0xFFFF)..p + n).step_by(0x1_0000).collect();
                    let missing: Vec<usize> = pages.iter().copied().filter(|&q| super::view_mem::cpu_at(q).is_none()).collect();
                    line += &format!(" pages {} unreadable {}", pages.len(), missing.len());
                    if let Some(&q) = missing.first() {
                        static M: AtomicU64 = AtomicU64::new(0);
                        if M.fetch_add(1, Ordering::Relaxed) < 24 {
                            let cpu_view = crate::hle::kernelx::mirrors::source(q);
                            crate::xlog_info!("lab: skin inputs: {cs:016x} buffer {p:#x}+{n:#x}: page {q:#x} unreadable; CPU view {cpu_view:x?}: {}", cpu_view.map(crate::hle::kernelx::describe_address).unwrap_or_default());
                        }
                    }
                }
                line += ";";
                if mode == 2 && p != 0 {
                    super::view_mem::ensure(p, n);
                }
            }
        } else {
            line += " record buffer unreadable;";
        }
    } else {
        line += " no record V#;";
    }
    // the textures
    let count = cpu(a + 0x60, 4).filter(|v| v[3] >> 30 == 0).and_then(|v| cpu(va(&v), 1)).map_or(0, |c| c[0].min(64) as usize);
    let (mut named, mut evicted) = (Vec::new(), 0);
    if let Some(v) = cpu(a + 0x100, 4).filter(|v| v[3] >> 30 == 0 && (v[0] | v[1]) != 0) {
        for i in 0..count {
            let Some(slot) = cpu(va(&v) + i * 0x90 + 0x8c, 1).map(|s| s[0]).filter(|&s| s != 0 && s < 1 << 22) else { continue };
            let Some(t) = cpu(view + slot as usize * 32, 4).filter(|t| t[3] >> 28 >= 8) else { continue };
            if let Some(res) = super::console_heap::index_resource(t[0]) {
                if !named.contains(&res) {
                    evicted += super::vram::is_evicted(res) as u32;
                    named.push(res);
                }
            }
        }
    }
    if mode == 2 && super::vram::tracking() {
        super::title_textures::drawn(&named);
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    // every buffer V# of both tables' first 32 slots: where it lives (the outputs are A's slots 24 / 25)
    if n < 24 || (n % 4096 == 0 && n < 1 << 20) {
        let mut all = String::new();
        for (name, t) in [("A", a), ("B", b)] {
            for k in 0..32usize {
                let Some(d) = cpu(t + 32 * k, 4).filter(|d| d[3] >> 30 == 0 && (d[0] | d[1]) != 0) else { continue };
                let (p, len) = (va(&d), bytes(&d));
                let place = match u.window_of(p) {
                    Some(w) => format!("win {w}{}", if u.is_sparse(p) { " sparse" } else { "" }),
                    None => format!("view, cpu {}", if super::view_mem::cpu_at(p).is_some() { "ok" } else { "none" }),
                };
                all += &format!(" {name}{k}: {p:#x}+{len:#x} ({place});");
            }
        }
        crate::xlog_info!("lab: skin tables #{n} {cs:016x}:{all}");
    }
    // the outputs (A's slots 24 / 25): how many of their tiles have video memory now
    {
        let mut out = String::new();
        let mut missing = 0;
        for k in [24usize, 25] {
            let Some(d) = cpu(a + 32 * k, 4).filter(|d| d[3] >> 30 == 0 && (d[0] | d[1]) != 0) else { continue };
            let (p, len) = (va(&d), bytes(&d));
            if !u.is_sparse(p) {
                continue;
            }
            let (all, mapped, kept, parked) = super::sparse::tile_counts(p, len);
            missing += all - mapped.min(all);
            out += &format!(" A{k} {p:#x}+{len:#x}: {all} tiles, {mapped} mapped, {kept} kept, {parked} parked;");
        }
        static LAST: AtomicU64 = AtomicU64::new(u64::MAX);
        static LINES: AtomicU64 = AtomicU64::new(0);
        if LAST.swap(missing as u64, Ordering::Relaxed) != missing as u64 && LINES.fetch_add(1, Ordering::Relaxed) < 400 {
            crate::xlog_info!("lab: skin out #{n} {cs:016x}:{out} ({missing} without video memory)");
        }
    }
    if n < 48 || n.is_power_of_two() || evicted > 0 && n % 64 == 0 || line.contains("unreadable ") && !line.contains("unreadable 0") && n % 256 == 0 {
        crate::xlog_info!("lab: skin inputs #{n} {cs:016x}: tables {a:#x} {b:#x}, heap {view:#x};{line} {count} records, {} textures named, {evicted} evicted", named.len());
    }
}

/// lab: `XCL_PS_PROBE=t` (seconds of uptime; `XCL_PS_PROBE_HASH`, default e1e597d212161b7e: the
/// G-buffer pixel shader of MS_VS_TransformD; `XCL_PS_PROBE_EVERY`, default 8): where that
/// program's textures come from, as the CPU sees the memory when the mesh pass is bound.
/// `io` = its SGPRs: s8 the global table, s14 the material record's offset in the buffer whose
/// V# is at table + 0xac0 (dwords 0, 1, 3, 4, 5 of the record = heap slots of the albedo
/// (t111), t105, detail, t110 and normal (t144) textures), s15 / s16 the view and sampler heap
/// (`heap_sgpr`): T# = s15 + slot * 32, S# = s16 + 0x430 / 0x4b0 / 0x520. Per T#: its 8 words,
/// the heap slot it is in (PC index a view made there has), and the texture word 0 names.
unsafe fn ps_probe(tp: &super::real::TwoPassPso, io: &[u32]) {
    static CFG: std::sync::OnceLock<Option<(f64, u64, u64)>> = std::sync::OnceLock::new();
    let Some((from, hash, every)) = *CFG.get_or_init(|| {
        let from = std::env::var("XCL_PS_PROBE").ok()?.trim().parse().ok()?;
        let hash = std::env::var("XCL_PS_PROBE_HASH").ok().and_then(|v| u64::from_str_radix(v.trim(), 16).ok()).unwrap_or(0xe1e5_97d2_1216_1b7e);
        let every = std::env::var("XCL_PS_PROBE_EVERY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8u64).max(1);
        Some((from, hash, every))
    }) else {
        return;
    };
    if crate::log::uptime() < from || super::real::two_pass_ps_hash(tp.compute) != hash {
        return;
    }
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if CALLS.fetch_add(1, Ordering::Relaxed) % every != 0 || LOGGED.fetch_add(1, Ordering::Relaxed) >= 48 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let s = |i: usize| io.get(i).copied().unwrap_or(0);
    // a 32-bit console address is in window 0 (as the shaders read it)
    let at = |a: u64| if a >> 32 == 0 { u.base() + a as usize } else { a as usize };
    let cpu = |p: usize, n: usize| -> Option<Vec<u32>> {
        crate::gpu::shader::readable(p, 4 * n).then(|| (0..n).map(|i| ((p + 4 * i) as *const u32).read_unaligned()).collect())
    };
    let fmt = |v: &Option<Vec<u32>>| v.as_ref().map_or("unreadable".to_string(), |v| v.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(" "));
    let heap = |p: usize| match super::console_heap::probe_slot(p as u64) {
        Some((obj, k, sampler, visible, index)) => format!("{} heap {obj:#x} slot {k} visible {visible} pc index {index:?}", if sampler { "sampler" } else { "view" }),
        None => "in no console heap".to_string(),
    };
    let mut line = format!("ps probe: PS {hash:016x} MS {:016x}: s8.. = [{}], {} SGPRs", tp.ms_hash, (8..18).map(|i| format!("{:x}", s(i))).collect::<Vec<_>>().join(" "), io.len());
    let v = cpu(at(s(8) as u64) + 0xac0, 4);
    line += &format!(" | index buffer V# [{}]", fmt(&v));
    let mut slots = Vec::new();
    if let Some(v) = &v {
        let buf = at((u64::from(v[1] & 0xFFFF) << 32) | u64::from(v[0]));
        let stride = (v[1] >> 16) & 0x3FFF;
        let limit = if stride == 0 { v[2] } else { v[2].wrapping_mul(stride) };
        let rec = cpu(buf.wrapping_add(s(14) as usize), 6);
        line += &format!(" -> {buf:#x} win {:?} limit {limit:#x}, record +{:#x} = [{}]", u.window_of(buf), s(14), fmt(&rec));
        if let Some(r) = rec.filter(|_| s(14).checked_add(24).is_some_and(|e| e <= limit)) {
            slots = vec![("albedo", r[0]), ("t105", r[1]), ("detail", r[3]), ("t110", r[4]), ("normal", r[5])];
        } else {
            line += " (past the limit or unreadable: the program reads slot 0 five times)";
            slots = vec![("slot 0", 0)];
        }
    }
    for (name, slot) in slots {
        let p = at(s(15) as u64).wrapping_add(slot as usize * 32);
        let t = cpu(p, 8);
        line += &format!(" | {name} slot {slot:#x}: T# at {p:#x} win {:?} [{}], {}", u.window_of(p), fmt(&t), heap(p));
        let Some(t) = t else { continue };
        match super::console_heap::index_resource(t[0]) {
            Some(res) => {
                line += &format!(", word 0 names texture {res:#x} evicted {}", super::vram::is_evicted(res));
                match super::title_textures::probe_state(res) {
                    Some((va, src, pending, gpu_written, unmapped, shadowed)) => {
                        line += &format!(" at {va:#x} read from {src:#x} pending {pending} gpu-written {gpu_written} unmapped {unmapped} shadowed {shadowed}");
                        let r = res as *mut c_void;
                        if let Some(r) = ID3D12Resource::from_raw_borrowed(&r) {
                            let d = r.GetDesc();
                            line += &format!(" {}x{} format {} mips {}", d.Width, d.Height, d.Format.0, d.MipLevels);
                        }
                    }
                    None => line += " (not kept in step with title memory)",
                }
            }
            None => line += ", word 0 names no texture on pages",
        }
    }
    for off in [0x430usize, 0x4b0, 0x520] {
        let p = at(s(16) as u64) + off;
        let w = cpu(p, 4);
        line += &format!(" | S# +{off:#x} at {p:#x} [{}] -> sampler {:?}, {}", fmt(&w), w.as_ref().map(|w| w[3] & 0xFFF), heap(p));
    }
    crate::xlog_info!("{line}");
}

/// lab: `XCL_MS_PROBE=t`: what a two-pass mesh draw's program is given, for the retail title's
/// wrapped-VS mesh shaders (s17 = descriptor table, +0x60 = the meshlet buffer; its dword 0 =
/// meshlet count, +12.. = the meshlets, 16 bytes each): the CPU's view logged here, the GPU's
/// view of the same bytes through the window buffer in peek slots (kind 12: dwords 0..3,
/// kind 13: dwords 4..7 of the meshlet buffer; 14: dwords 0..3 of the descriptor at +0x60).
unsafe fn ms_probe(this: *mut c_void, what: &str, tp: &super::real::TwoPassPso, groups: u32) {
    static FROM: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let Some(from) = *FROM.get_or_init(|| std::env::var("XCL_MS_PROBE").ok().and_then(|v| v.trim().parse().ok())) else { return };
    if crate::log::uptime() < from {
        return;
    }
    {
        // which pipelines the two-pass draws use: (MS, PS) → (draws, groups), the most used logged
        static USE: Mutex<Option<HashMap<(u64, u64), (u64, u64)>>> = Mutex::new(None);
        static TOTAL: AtomicU64 = AtomicU64::new(0);
        let ps = super::real::two_pass_ps_hash(tp.compute);
        let mut g = USE.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let e = m.entry((tp.ms_hash, ps)).or_default();
        e.0 += 1;
        e.1 += u64::from(groups);
        let n = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        if n >= 4096 && n.is_power_of_two() {
            let mut v: Vec<_> = m.iter().map(|(k, v)| (*v, *k)).collect();
            v.sort_unstable_by(|a, b| b.cmp(a));
            crate::xlog_info!("ms probe: {n} two-pass draws with {} (MS, PS) pairs; most used (draws/groups MS+PS): {}", v.len(), v.iter().take(40).map(|((d, gr), (ms, ps))| format!("{d}/{gr} {ms:016x}+{ps:016x}")).collect::<Vec<_>>().join(", "));
        }
    }
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| std::env::var("XCL_MS_PROBE_EVERY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(16u64).max(1));
    if CALLS.fetch_add(1, Ordering::Relaxed) % every != 0 || LOGGED.load(Ordering::Relaxed) >= 300 {
        return;
    }
    LOGGED.fetch_add(1, Ordering::Relaxed);
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let inline_ms = layout.as_ref().map_or(0, |l| l.inline_ms as usize).min(args.len());
    let w: Vec<u32> = args[..inline_ms].chunks(4).map(|c| { let mut b = [0u8; 4]; b[..c.len()].copy_from_slice(c); u32::from_le_bytes(b) }).collect();
    let g = |i: usize| w.get(i).copied().unwrap_or(0);
    let cpu = |p: usize, n: usize| -> Option<Vec<u32>> {
        crate::gpu::shader::readable(p, 4 * n).then(|| (0..n).map(|i| ((p + 4 * i) as *const u32).read_unaligned()).collect())
    };
    let fmt = |v: &Option<Vec<u32>>| v.as_ref().map_or("unreadable".to_string(), |v| v.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(" "));
    let table = u.base() + g(1) as usize;
    let mut line = format!("ms probe: {what} MS {:016x} groups {groups} threads {} maxv {} maxp {} vd {}; inline {} B of {}: s16.. = [{}]; table {table:#x}", tp.ms_hash, tp.record.threads, tp.record.max_vertices, tp.record.max_primitives, tp.record.vert_dwords(), inline_ms, args.len(), w.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(" "));
    // the eight V#s of the table (the meshlet buffer's slot differs per program)
    for k in 0..8usize {
        let desc = cpu(table + 0x20 * k, 4);
        let Some(d) = &desc else { continue };
        let buf = ((d[1] as usize & 0xFFFF) << 32) | d[0] as usize;
        if buf == 0 {
            continue;
        }
        // an address with no high part is in window 0 (as the shaders read it)
        let buf = if buf >> 32 == 0 { u.base() + buf } else { buf };
        let m = cpu(buf, 16);
        line += &format!(" | +{:#x} V# [{}] -> {buf:#x} win {:?} sparse {} cpu [{}]", 0x20 * k, fmt(&desc), u.window_of(buf), u.is_sparse(buf), fmt(&m));
        if let Some((b, o)) = super::real::uma_window_buffer(buf, 64) {
            let s0 = ei_peek_slot(this, 12, groups, b, o, b, o + 12);
            let s1 = ei_peek_slot(this, 13, groups, b, o + 16, b, o + 28);
            let s2 = ei_peek_slot(this, 14, groups, b, o + 48, b, o + 60);
            line += &format!(" gpu slots {} {} {}", s0.unwrap_or(0), s1.unwrap_or(0), s2.unwrap_or(0));
        }
    }
    crate::xlog_info!("{line}");
}


/// lab: (peek slot, PS / CS record hash, graphics) of the calls peeked by pipeline
static EI_NAMED: Mutex<Vec<(usize, u64, bool)>> = Mutex::new(Vec::new());
static EI_NAMED_DONE: AtomicU64 = AtomicU64::new(0);

/// lab (`XCL_EI_PEEK`): (slot, CS record hash) of compute records as the title wrote them
static EI_WIDE: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
/// lab: (slot of the title's record, slot of the bounded record, shader record hash) of one call
static EI_PAIRS: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());
static EI_PAIRS_DONE: AtomicU64 = AtomicU64::new(0);

fn ei_peek_report() {
    let g = EI_PEEK.lock().unwrap();
    let Some(pk) = g.as_ref() else { return };
    unsafe {
        let mut p: *mut c_void = std::ptr::null_mut();
        if pk.rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
            return;
        }
        let w = |slot: usize, k: usize| (p as *const u32).add(slot * 8 + k).read_unaligned();
        {
            // CS -> (records read, with more than 65535 groups in a dimension, the widest)
            let mut by: std::collections::BTreeMap<u64, (u32, u32, [u32; 3])> = Default::default();
            for &(slot, hash) in EI_WIDE.lock().unwrap().iter() {
                let a = [w(slot, 0), w(slot, 1), w(slot, 2)];
                if a[0] == 0xEEEE_EEEE {
                    continue;
                }
                let e = by.entry(hash).or_default();
                e.0 += 1;
                if a.iter().any(|&v| v > 65535) {
                    e.1 += 1;
                }
                if a[0] >= e.2[0] {
                    e.2 = a;
                }
            }
            let read: u32 = by.values().map(|e| e.0).sum();
            let over: u32 = by.values().map(|e| e.1).sum();
            let mut wide: Vec<(u64, (u32, u32, [u32; 3]))> = by.iter().filter(|e| e.1 .1 > 0).map(|(&h, &e)| (h, e)).collect();
            wide.sort_by_key(|e| std::cmp::Reverse(e.1 .2[0]));
            crate::xlog_info!(
                "ei wide: {read} compute records of {} programs read as the title wrote them, {over} ask for more than 65535 groups in a dimension ({} programs): {}",
                by.len(),
                wide.len(),
                wide.iter().take(24).map(|(h, e)| format!("{h:016x} {} of {} widest {:?}", e.1, e.0, e.2)).collect::<Vec<_>>().join("; ")
            );
        }
        {
            // only the pairs whose two records differ, and every pair of a folded program
            let pairs = EI_PAIRS.lock().unwrap();
            let from = EI_PAIRS_DONE.load(Ordering::Relaxed) as usize;
            let mut done = from;
            for &(t, b, hash) in pairs.iter().skip(from) {
                if w(t, 0) == 0xEEEE_EEEE || w(b, 0) == 0xEEEE_EEEE {
                    break;
                }
                done += 1;
                let (tr, br) = ([w(t, 0), w(t, 1), w(t, 2)], [w(b, 0), w(b, 1), w(b, 2)]);
                if tr != br || crate::gpu::dxil::kernel::fold_groups(hash) {
                    crate::xlog_info!("ei pair: {hash:016x} at {:.1} s: the title's record {tr:?}, the bounded one {br:?}", pk.meta[t].0);
                }
            }
            EI_PAIRS_DONE.store(done as u64, Ordering::Relaxed);
        }
        {
            let named = EI_NAMED.lock().unwrap();
            let from = EI_NAMED_DONE.load(Ordering::Relaxed) as usize;
            let mut done = from;
            for &(slot, hash, gfx) in named.iter().skip(from) {
                if w(slot, 0) == 0xEEEE_EEEE {
                    break;
                }
                done += 1;
                let (t, _, max, has_count) = pk.meta[slot];
                crate::xlog_info!("ei named: {} {hash:016x} at {t:.1} s: max {max} [{} {} {}]{}", if gfx { "ps" } else { "cs" }, w(slot, 0), w(slot, 1), w(slot, 2), if has_count { format!(" count {}", w(slot, 4)) } else { String::new() });
            }
            EI_NAMED_DONE.store(done as u64, Ordering::Relaxed);
        }
        // (kind, 4 s bucket) -> (calls, not run yet, empty arguments, count 0, work, examples)
        let mut rows: std::collections::BTreeMap<(u8, u64), (u32, u32, u32, u32, u64, Vec<String>)> = Default::default();
        for (slot, &(t, kind, max, has_count)) in pk.meta.iter().enumerate() {
            let row = rows.entry((kind, (t / 4.0) as u64 * 4)).or_default();
            row.0 += 1;
            let (x, y, z, c) = (w(slot, 0), w(slot, 1), w(slot, 2), w(slot, 4));
            if x == 0xEEEE_EEEE {
                row.1 += 1;
                continue;
            }
            let empty = if kind == 1 { x == 0 || y == 0 } else { x == 0 || y == 0 || z == 0 };
            if empty {
                row.2 += 1;
            }
            if has_count && c == 0 {
                row.3 += 1;
            }
            if !empty && !(has_count && c == 0) {
                row.4 += x as u64 * if kind == 1 { y as u64 } else { y as u64 * z as u64 };
            }
            if row.5.len() < if kind >= 4 { 48 } else { 10 } && (!empty || row.0 % 97 == 1 || kind >= 4) {
                if matches!(kind, 5..=7 | 9..=11 | 30..=32) {
                    row.5.push(format!("({:.4e} {:.4e} {:.4e} {:.4e})", f32::from_bits(x), f32::from_bits(y), f32::from_bits(z), f32::from_bits(c)));
                } else {
                    row.5.push(format!("max {max} [{x:#x} {y:#x} {z:#x}]{}", if has_count { format!(" count {c:#x}") } else { String::new() }));
                }
            }
        }
        for (slot, &(_, kind, _, _)) in pk.meta.iter().enumerate() {
            if (12..=14).contains(&kind) {
                crate::xlog_info!("ms probe: gpu slot {slot} kind {kind}: [{:x} {:x} {:x} {:x}]", w(slot, 0), w(slot, 1), w(slot, 2), w(slot, 4));
            }
        }
        crate::xlog_info!("ei peek: {} calls sampled (kind 0 two-pass mesh bounded, 1 graphics, 2 compute, 3 two-pass mesh as the title wrote them; record 0 after the compute pass: 4 header + 5/6/7 positions (EI), 8 header + 9/10/11 positions (direct, max = groups))", pk.meta.len());
        for ((kind, t), (n, pending, empty, zero_count, work, examples)) in rows {
            crate::xlog_info!("ei peek: kind {kind} at {t}-{} s: {n} calls, {pending} not run yet, {empty} with empty arguments, {zero_count} with count 0, work of the rest {work}; {}", t + 4, examples.join("; "));
        }
        pk.rb.Unmap(0, None);
    }
}

/// `XCL_DUMP_TARGETS`: an ExecuteIndirect with `outcome` (a dispatch signature's: a dispatch)
unsafe fn ei_work(this: *mut c_void, sig: usize, outcome: usize) {
    if !super::targets::enabled() {
        return;
    }
    if signature_dispatches(sig) {
        super::targets::note_dispatch();
    } else {
        super::targets::note_draw(this, outcome);
    }
}

/// XCL's own buffer for `fresh_indirect_arguments` (device, buffer): COMMON between its uses
static ARGUMENT_ORDER: Mutex<Option<(usize, ID3D12Resource)>> = Mutex::new(None);

/// `op` (an ExecuteIndirect on `list`) reading the arguments and count the list wrote before it.
///
/// The console orders a dispatch that writes indirect arguments before the ExecuteIndirect that
/// reads them with a barrier that reaches the PC as a UAV barrier (`list_x::resource_barrier_x`).
/// NVIDIA's driver (RTX 5090) fetches an ExecuteIndirect's arguments without waiting for that
/// barrier: it ran the count that was there before the list in all but a few of 100 tries, also
/// when the dispatch wrote them through another resource over the same memory, as translated
/// shaders do through the memory window buffers (`execute_indirect_reads_the_arguments_written_before_it`).
/// A retail title's ExecuteIndirect after the dispatch that writes its group count then runs
/// whatever count the memory held before. A transition to INDIRECT_ARGUMENT makes the driver
/// wait for the writes – of any buffer, as measured there. The title's buffer is in the title's
/// state (its transitions reach the PC list), so a buffer of XCL's own takes it, COMMON ->
/// INDIRECT_ARGUMENT before and back after. Only on a list XCL made (`list_x::on_create`) and
/// not a bundle, which cannot hold barriers; its type comes from XCL's state, not a call to it.
unsafe fn fresh_indirect_arguments(list: *mut c_void, op: impl FnOnce()) {
    let tracked = super::list_x::tracked_list_type(list).is_some_and(|t| t != D3D12_COMMAND_LIST_TYPE_BUNDLE);
    let buffer = if tracked { ID3D12GraphicsCommandList::from_raw_borrowed(&list).and_then(|l| argument_order_buffer(l)) } else { None };
    let Some(buffer) = buffer else { return op() };
    let b = buffer.as_raw() as usize;
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    op();
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
}

/// lab (`XCL_CLAMP_ORDER=0`: off): the same wait for a dispatch of XCL's own that READS the
/// arguments - `indirect_clamp` bounds a record in a compute pass before the call, and that pass
/// came before the transition above: it read the count while the title's passes were still
/// writing it (runs Q4 / Q5, "ei pair": the final gather's [97776, 1, 1] bounded to [174, 1, 1],
/// other programs' records bounded to MORE than they held), so the call ran a stale count.
pub(super) unsafe fn wait_for_argument_writes(list: &ID3D12GraphicsCommandList) {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var("XCL_CLAMP_ORDER").is_ok_and(|v| v.trim() == "0")) {
        return;
    }
    let raw = list.as_raw();
    if !super::list_x::tracked_list_type(raw).is_some_and(|t| t != D3D12_COMMAND_LIST_TYPE_BUNDLE) {
        return;
    }
    let Some(buffer) = argument_order_buffer(list) else { return };
    let b = buffer.as_raw() as usize;
    super::list_x::transition(raw, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    super::list_x::transition(raw, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
}

/// The device's buffer for `fresh_indirect_arguments` (made on first use)
unsafe fn argument_order_buffer(list: &ID3D12GraphicsCommandList) -> Option<ID3D12Resource> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let mut g = ARGUMENT_ORDER.lock().unwrap();
    if let Some((_, b)) = g.as_ref().filter(|(d, _)| *d == dev.as_raw() as usize) {
        return Some(b.clone());
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: no buffer to order its arguments after the writes before it ({e}) – it may read old ones");
        }
        return None;
    }
    let b = b?;
    *g = Some((dev.as_raw() as usize, b.clone()));
    Some(b)
}

/// A DrawInstanced of the console's rect list (three corners of each rectangle, the fourth
/// implied) as two triangles a rectangle; false: not a rect list, or one this cannot expand
/// (the caller draws the three vertices as a triangle – half of each rectangle).
///
/// The PC has no rect lists. The fourth corner is made here, on the CPU, from the title's
/// vertex data: v3 = v0 + v2 − v1 (seen in a retail title's full-screen passes: (−1,1) (−1,−1)
/// (1,−1) with texture coordinates (0,0) (0,1) (1,1) – the missing corner is (1,1), (1,0)).
/// The vertex layout is not looked at: each 4-byte word that is the same in all three vertices
/// is kept (a packed colour), any other is taken as a float. Vertices a shader makes from the
/// vertex index alone (no vertex buffer) are not expanded.
unsafe fn draw_rects(this: *mut c_void, list: &ID3D12GraphicsCommandList, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) -> bool {
    if !cpu_rect_expansion(with(this, |s| s.pso)) {
        return false;
    }
    let Some(vbs) = super::list_x::rect_list_state(this) else { return false };
    let rects = (vertices / 3) as usize;
    if vbs.is_empty() || rects == 0 || rects > 4096 {
        return false;
    }
    let mut views = Vec::new();
    for &(slot, (va, bytes, stride), _) in &vbs {
        let (s, at) = (stride as usize, va as usize + first_vertex as usize * stride as usize);
        let len = rects * 3 * s;
        if va == 0 || s == 0 || s % 4 != 0 || first_vertex as usize * s + len > bytes as usize || !crate::gpu::shader::readable(at, len) {
            return false;
        }
        let p = allocate_upload_memory(this, rects * 6 * s, 16);
        if p.is_null() {
            return false;
        }
        let word = |v: usize, w: usize| (at as *const u8).add(v * s + w * 4).cast::<u32>().read_unaligned();
        for r in 0..rects {
            let out = p.add(r * 6 * s);
            let src = (at as *const u8).add(r * 3 * s);
            // v0 v1 v2, then v0 v2 v3
            std::ptr::copy_nonoverlapping(src, out, 3 * s);
            std::ptr::copy_nonoverlapping(src, out.add(3 * s), s);
            std::ptr::copy_nonoverlapping(src.add(2 * s), out.add(4 * s), s);
            for w in 0..s / 4 {
                let (a, b, c) = (word(r * 3, w), word(r * 3 + 1, w), word(r * 3 + 2, w));
                let d = if a == b && b == c { a } else { (f32::from_bits(a) + f32::from_bits(c) - f32::from_bits(b)).to_bits() };
                out.add(5 * s + w * 4).cast::<u32>().write_unaligned(d);
            }
        }
        views.push((slot, D3D12_VERTEX_BUFFER_VIEW { BufferLocation: super::list_x::pc_va_of(p as u64), SizeInBytes: (rects * 6 * s) as u32, StrideInBytes: stride }));
    }
    for (slot, v) in &views {
        list.IASetVertexBuffers(*slot, Some(&[*v]));
    }
    list.DrawInstanced(rects as u32 * 6, instances, 0, first_instance);
    for &(slot, _, pc) in &vbs {
        if let Some(v) = pc {
            list.IASetVertexBuffers(slot, Some(&[v]));
        }
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: rect lists are drawn as two triangles a rectangle, the fourth corner made on the CPU");
    }
    true
}

/// Whether a rect list drawn with the bound pipeline `pso` is expanded on the CPU (`draw_rects`):
/// not with the rect list GS (`real::has_rect_list_gs`), which completes each rect itself – both
/// would draw every rect twice (each CPU half is three corners of the rect).
fn cpu_rect_expansion(pso: usize) -> bool {
    !super::real::has_rect_list_gs(super::pso_async::resolve(pso))
}

/// #12 DrawInstanced
pub(super) unsafe extern "system" fn draw_instanced(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    DRAW_ARGS.with(|d| d.set((false, vertices, instances, first_vertex, 0)));
    if let Some(list) = graphics_bound(this) {
        gp_mark(this, gp::Label::of(gp::Kind::Other));
        if draw_rects(this, &list, vertices, instances, first_vertex, first_instance) {
            gp_draw(this, [vertices, instances, 0]);
            return;
        }
        super::list_x::sync_title_buffers(this, None, first_vertex, vertices, if first_instance != 0 { instances.max(2) } else { instances });
        list.DrawInstanced(vertices, instances, first_vertex, first_instance);
        gp_draw(this, [vertices, instances, 0]);
    }
}

use super::gpu_profile as gp;

/// lab (`XCL_GPU_PROFILE`): a mark on `this` (nothing when the profile is off)
unsafe fn gp_mark(this: *mut c_void, label: gp::Label) {
    if gp::on() {
        gp::mark(this, label);
    }
}

/// lab (`XCL_GPU_PROFILE`): the mark after a draw on `this` with its bound pipeline
unsafe fn gp_draw(this: *mut c_void, size: [u32; 3]) {
    if !gp::on() {
        return;
    }
    let pso = super::pso_async::resolve(with(this, |s| s.pso));
    let label = if super::real::is_translated_graphics(pso) {
        gp::Label::op(gp::Kind::Draw, super::real::pso_shaders(pso).unwrap_or((0, 0)), size)
    } else {
        gp::Label::op(gp::Kind::PcDraw, (0, 0), size)
    };
    gp::mark(this, label);
}

/// lab (`XCL_GPU_PROFILE`): the console shaders of a compute pipeline
fn gp_cs(pso: usize) -> (u64, u64) {
    if !gp::on() {
        return (0, 0);
    }
    super::real::pso_shaders(super::pso_async::resolve(pso)).map_or((0, 0), |s| (s.0, 0))
}

/// lab (`XCL_GPU_PROFILE`): what a two-pass pipeline is named by (MS, PS)
fn gp_two_pass(tp: &super::real::TwoPassPso) -> (u64, u64) {
    if !gp::on() {
        return (0, 0);
    }
    (tp.ms_hash, super::real::two_pass_ps_hash(tp.compute))
}

thread_local! {
    /// the draw being bound: (indexed, count, instances, first index / vertex, base vertex) – for the frame trace
    static DRAW_ARGS: std::cell::Cell<(bool, u32, u32, u32, i32)> = const { std::cell::Cell::new((false, 0, 0, 0, 0)) };
}

/// #13 DrawIndexedInstanced
pub(super) unsafe extern "system" fn draw_indexed_instanced(this: *mut c_void, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    if let Some(list) = prepare_indexed_draw(this, indices, instances, first_index, base_vertex) {
        super::list_x::sync_title_buffers(this, Some((first_index, indices, base_vertex)), 0, 0, if first_instance != 0 { instances.max(2) } else { instances });
        gp_mark(this, gp::Label::of(gp::Kind::Other));
        emit_prepared_indexed_draw(&list, indices, instances, first_index, base_vertex, first_instance);
        gp_draw(this, [indices, instances, 0]);
    }
}

/// Prepare an indexed draw once, before callers make temporary native state changes.
pub(super) unsafe fn prepare_indexed_draw(
    this: *mut c_void,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
) -> Option<ID3D12GraphicsCommandList> {
    DRAW_ARGS.with(|d| d.set((true, indices, instances, first_index, base_vertex)));
    graphics_bound(this)
}

/// Submit an indexed draw whose translated graphics state has already been prepared.
pub(super) unsafe fn emit_prepared_indexed_draw(
    list: &ID3D12GraphicsCommandList,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    list.DrawIndexedInstanced(indices, instances, first_index, base_vertex, first_instance);
}

/// The pipeline a draw (or an ExecuteIndirect) uses now: a graphics pipeline still being built
/// is not waited for – its build goes to the workers and its stand-in, which draws nothing, is
/// used until it is built (`pso_async`). A compute one is waited for. Waiting held a retail
/// title's threads 1 – 5 s per draw in the driver's compiler (33 s in all as its world loaded,
/// on an NVIDIA GPU), and its hang watchdog ended it.
fn pso_for_draw(pso: usize) -> usize {
    super::pso_async::resolve_for_bind(pso)
}

/// Before a draw: with a translated pipeline its ABI root signature and the kept graphics
/// arguments are bound (None: the draw is skipped – a console pipeline without a
/// translation, or no unified memory).
unsafe fn graphics_bound(this: *mut c_void) -> Option<ID3D12GraphicsCommandList> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?.clone();
    let pso = with(this, |s| s.pso);
    // a graphics pipeline still being built in the background is not waited for (`pso_for_draw`)
    let real = pso_for_draw(pso);
    if !super::real::is_translated_graphics(real) {
        if super::targets::enabled() {
            super::targets::note_draw(this, if super::pso_async::is_pending(pso) { super::targets::STANDIN } else { super::targets::PC });
        }
        // lab (`XCL_SKIP_LOG`): a draw with a graphics pipeline still being built draws nothing
        if skip_log() && super::pso_async::is_pending(pso) {
            skip_note_pso(pso, line!(), "drawn with the stand-in (nothing drawn)");
        }
        // A finished two-pass MS/GS pipeline is a PC mesh shader pipeline (MS+PS) registered in
        // TWO_PASS rather than TRANSLATED_GRAPHICS. Binding the title's graphics RS and issuing
        // DrawInstanced / DrawIndexedInstanced against a mesh PSO invalidates the command list
        // (Close → E_INVALIDARG).
        if let Some(tp) = super::real::two_pass_of(real) {
            skip_note(this, line!());
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!(
                    "D3D12.X: draw with two-pass {} pipeline ({real:#x}) skipped – PC pipeline is a mesh shader",
                    tp.kind
                );
            }
            return None;
        }
        count(DRAWS_PC);
        if super::present::trace_frame() {
            // which pipelines draw without a translation (first 12 distinct)
            static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
            let mut g = SEEN.lock().unwrap();
            if !g.contains(&pso) && g.len() < 12 {
                g.push(pso);
                let real = super::pso_async::resolve(pso);
                crate::xlog_info!(
                    "frame trace: draw with an untranslated pipeline {pso:#x} (resolves to {real:#x}; pending {}, two-pass {}, RT0 {:?})",
                    super::pso_async::is_pending(pso),
                    super::real::two_pass_of(real).is_some(),
                    super::present::rt0_resource(this as usize).map(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }))
                );
            }
        }
        // a PC pipeline (also a stand-in or fallback: made with the title's root signature):
        // the title's root signature, which nothing else binds on the PC list – a draw without
        // one is invalid (the debug layer reports it on every GPU; WARP faults on it)
        if super::pso_async::is_pending(pso) {
            super::list_x::note_dropped_draw(this);
        }
        let rs = with(this, |s| s.rs_gfx) as *mut c_void;
        if let Some(r) = ID3D12RootSignature::from_raw_borrowed(&rs) {
            list.SetGraphicsRootSignature(r);
        }
        return Some(list);
    }
    if super::present::trace_frame() {
        trace_ui_draw(this, pso);
    }
    rt_writers_probe(this, pso);
    post_probe(this, pso);
    match bind_abi(this, &list, pso, true) {
        Ok(()) => {
            super::targets::note_draw(this, super::targets::RAN);
            count(DRAWS);
            Some(list)
        }
        Err(e) => {
            super::targets::note_draw(this, super::targets::SKIPPED);
            super::list_x::note_dropped_draw(this);
            skip_note(this, line!());
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: draw with a translated pipeline skipped: {e}");
            }
            None
        }
    }
}

/// `XCL_TRACE_FRAME`: draws into a presented buffer (the UI) once the title has presented for a
/// while – their pipeline, root arguments and the resources their descriptor tables start with
/// (format, size, kept in step with title memory or not). The first 24.
/// `XCL_TRACE_FRAME`, after 300 presents: one line per distinct (VS, PS, render target) drawn –
/// the shaders' dump hashes, the target and the textures of the first descriptor table.
unsafe fn trace_draw_kinds(this: *mut c_void, pso: usize) {
    // keyed by the target's shape, not the object: a UI library's transient targets (Renoir's
    // layers) are new objects every frame and would fill the list at once
    static SEEN: Mutex<Vec<(u64, u64, Option<(u64, u32, i32, u32)>)>> = Mutex::new(Vec::new());
    if super::present::activity().0 < 300 {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.SampleDesc.Count) });
    {
        let mut g = SEEN.lock().unwrap();
        let key = (vs, ps, desc(rt));
        if g.len() >= 160 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut tex = Vec::new();
    if let (Some(layout), Some(u)) = (layout, crate::gpu::uma::get()) {
        for p in layout.params.iter() {
            if let Param::Table { at } = *p {
                let t = word(at as usize) as usize;
                for k in 0..4usize {
                    if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) {
                        tex.push(format!("{:?}{}", desc(r), super::title_textures::address(r).map_or(String::new(), |a| format!(" at {a:#x}"))));
                    }
                }
            }
            if tex.len() >= 4 {
                break;
            }
        }
    }
    crate::xlog_info!(
        "frame trace: draws vs {vs:016x} ps {ps:016x} into {:?}{} – textures {}; {}",
        desc(rt),
        if super::present::is_presented(rt) { " (presented)" } else { "" },
        tex.join(", "),
        super::list_x::state_summary(this)
    );
}

/// `XCL_TRACE_SHADERS=hash,…` (dump hashes, hex): the draws traced with their root arguments
/// are those whose vertex or pixel shader is listed – into any target, at any time (the first
/// 24); without it, draws into a presented target (or any – `XCL_TRACE_UI_ANY`) after 300
/// presents.
fn traced_shaders() -> &'static [u64] {
    static S: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_TRACE_SHADERS").unwrap_or_default().split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect())
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): draws whose descriptor
/// tables name it – that object, or another at its title address (the first 16).
unsafe fn trace_reads_of_watched(this: *mut c_void, pso: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let w_addr = super::real::title_address(w).or_else(|| super::title_textures::address(w));
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..16usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let addr = super::real::title_address(r).or_else(|| super::title_textures::address(r));
            if super::present::is_watched(r) || (addr.is_some() && addr == w_addr) {
                let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
                // each (source, target, shaders) once
                static SEEN: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
                {
                    let mut g = SEEN.lock().unwrap();
                    if g.len() >= 32 || g.contains(&(r, rt, pso)) {
                        return;
                    }
                    g.push((r, rt, pso));
                }
                if !super::present::is_presented(rt) {
                    super::present::follow(rt);
                }
                if let Some(dsv) = super::list_x::bound_dsv(this) {
                    super::list_x::watch_dsv(dsv);
                }
                let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.Flags.0) });
                crate::xlog_info!(
                    "frame trace: a draw (shaders {:x?}, {}) reads a watched target {r:#x} {:?} at {addr:x?} (slot {k}), drawing into {rt:#x} {:?}{} – {}",
                    super::real::pso_shaders(super::pso_async::resolve(pso)),
                    super::list_x::state_summary(this),
                    d(r),
                    d(rt),
                    if super::present::is_presented(rt) { " (presented)" } else { "" },
                    root_args_summary(this, 32)
                );
                let args = DRAW_ARGS.with(|d| d.get());
                crate::xlog_info!(
                    "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
                    super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
                    super::list_x::geometry_summary(this, args)
                );
                return;
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: render targets a draw into the watched target samples (a UI library's
/// layers) are dumped with it (`present::follow`).
unsafe fn follow_layer_sources(this: *mut c_void) {
    let (Some(w), Some(rt)) = (super::present::watched_rt(), super::present::rt0_resource(this as usize)) else { return };
    if rt != w {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..4usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let layer = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| {
                let d = x.GetDesc();
                d.Flags.0 & 1 != 0 && d.Width >= 64 && d.Height >= 32 && d.SampleDesc.Count == 1
            });
            if layer && r != w {
                super::present::follow(r);
            }
        }
    }
}

unsafe fn trace_ui_draw(this: *mut c_void, pso: usize) {
    trace_draw_kinds(this, pso);
    trace_reads_of_watched(this, pso);
    follow_layer_sources(this);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // `XCL_TRACE_MANY=n`: n draws instead of 24 (a whole frame of a UI)
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| std::env::var("XCL_TRACE_MANY").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
    if N.load(std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let Some(rt) = super::present::rt0_resource(this as usize) else { return };
    let wanted = traced_shaders();
    if wanted.is_empty() {
        // draws into a presented target, or – `XCL_TRACE_UI_ANY` – into any target
        if super::present::activity().0 < 300 || (!super::present::is_presented(rt) && std::env::var_os("XCL_TRACE_UI_ANY").is_none()) {
            return;
        }
    } else if !super::real::pso_shaders(super::pso_async::resolve(pso)).is_some_and(|(v, p)| wanted.contains(&v) || wanted.contains(&p)) {
        return;
    } else {
        super::present::watch_rt(rt);
    }
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sh = super::real::pso_shaders(super::pso_async::resolve(pso)).map(|(v, p)| format!("vs {v:016x} ps {p:016x}"));
    crate::xlog_info!("frame trace: UI draw (pipeline {pso:#x}, {}): {}", sh.unwrap_or_default(), root_args_summary(this, 16));
    if !wanted.is_empty() {
        // the picked shaders' geometry too (vertices dumped at the next present)
        let args = DRAW_ARGS.with(|d| d.get());
        crate::xlog_info!(
            "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
            super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
            super::list_x::geometry_summary(this, args)
        );
    }
}

/// A draw's root arguments for the frame trace: constants, root buffers (their first words, as
/// the CPU sees them), the first slots of descriptor tables (the PC resources they name).
unsafe fn root_args_summary(this: *mut c_void, buffer_words: usize) -> String {
    let Some(u) = crate::gpu::uma::get() else { return String::new() };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return String::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut what = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => what.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                let w: Vec<u32> = if va != 0 && crate::gpu::shader::readable(va, 4 * buffer_words) { (0..buffer_words).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect() } else { Vec::new() };
                what.push(format!("p{i} buffer {va:#x} {w:x?}"));
            }
            Param::Table { at } => {
                let t = word(at as usize) as usize;
                let slots: Vec<String> = (0..4usize)
                    .map(|k| {
                        let at = u.base() + t + 32 * k;
                        let w = super::console_heap::words_at((t + 32 * k) as u32);
                        match super::console_heap::resource_at(at) {
                            Some(r) => {
                                if super::title_textures::address(r).is_some() {
                                    super::title_textures::probe_gpu_content(r);
                                }
                                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
                                format!("[{k}] {r:#x} {:?} title {:?} T# {:x?}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), super::title_textures::address(r).map(|a| format!("{a:#x}")), w.map(|w| w[..4].to_vec()))
                            }
                            None => format!("[{k}] - T# {:x?}", w.map(|w| w[..4].to_vec())),
                        }
                    })
                    .collect();
                what.push(format!("p{i} table {t:#x}: {}", slots.join(", ")));
            }
        }
    }
    what.join("; ")
}

/// #33 SetComputeRootSignature (PC #29)
pub(super) unsafe extern "system" fn set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout = layout;
    });
}

/// #35 SetComputeRootDescriptorTable (PC #31): the table's console address, 32 bits
pub(super) unsafe extern "system" fn set_compute_root_table(this: *mut c_void, i: u32, handle: u64) {
    put(this, i, 0, &(handle as u32).to_le_bytes());
}

/// #37 SetComputeRoot32BitConstant (PC #33)
pub(super) unsafe extern "system" fn set_compute_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put(this, i, offset, &value.to_le_bytes());
}

/// #39 SetComputeRoot32BitConstants (PC #35)
pub(super) unsafe extern "system" fn set_compute_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put(this, i, offset, &bytes);
    }
}

/// #41 / #43 / #45 SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #37 /
/// #39 / #41): the 64-bit GPU address
pub(super) unsafe extern "system" fn set_compute_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put(this, i, 0, &va.to_le_bytes());
}

/// #25 SetPipelineState: noted, then bound as on the PC
pub(super) unsafe extern "system" fn set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    with(this, |s| s.pso = pso as usize);
    super::real::bind_pipeline(this, pso);
}

/// #14 Dispatch: with a translated pipeline, on XCL's ABI with the kept root arguments
pub(super) unsafe extern "system" fn dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    dispatch_flags(this, x, y, z, 0);
}

/// `DispatchX` flag: x, y, z are thread counts, not group counts; the console launches whole
/// groups and leaves the lanes past the count out of EXEC (0x8000, wave32, needs nothing here)
const DISPATCH_THREAD_DIMENSIONS: u32 = 0x20;

/// What a dispatch of `dims` with `flags` launches on pipeline `pso`: its groups and the thread
/// limits for `IO_THREAD_LIMIT` (0: that axis ends on a whole group). None: it cannot be run
/// as the console would (thread dimensions with a group shape not known).
fn dispatch_groups(pso: usize, dims: [u32; 3], flags: u32) -> Option<([u32; 3], [u32; 3])> {
    if flags & DISPATCH_THREAD_DIMENSIONS == 0 {
        return Some((dims, [0; 3]));
    }
    let threads = super::real::pso_threads(pso)?.map(|t| t.max(1));
    let groups = std::array::from_fn(|k| dims[k].div_ceil(threads[k]));
    let limits = std::array::from_fn(|k| if dims[k] % threads[k] != 0 { dims[k] } else { 0 });
    Some((groups, limits))
}

/// lab: `XCL_DISPATCHX_THREADS=1` launches a `DispatchX` with thread dimensions as whole groups
/// (`dispatch_groups`); unset, its counts are launched as groups, as the lab's runs did before.
fn thread_dimensions() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        // LOCAL: `XCL_DISPATCHX_FLAGS=1` (the name of lab patch 0029, whose code this tree already
        // has from #2610) turns the same thing on
        let on = ["XCL_DISPATCHX_THREADS", "XCL_DISPATCHX_FLAGS"].iter().any(|k| std::env::var(k).is_ok_and(|v| v.trim() == "1"));
        crate::xlog_info!("D3D12.X: DispatchX thread dimensions (0x20) {}", if on { "launched as whole groups (XCL_DISPATCHX_THREADS=1)" } else { "launched as group counts (lab: XCL_DISPATCHX_THREADS=1 converts them)" });
        on
    })
}

/// lab: the flags `dispatch_groups` takes: without `on` (`thread_dimensions`), no thread dimensions.
fn lab_dispatch_flags(flags: u32, on: bool) -> u32 {
    if on { flags } else { flags & !DISPATCH_THREAD_DIMENSIONS }
}

/// `Dispatch` / `DispatchX(x, y, z, flags)`.
pub(super) unsafe fn dispatch_flags(this: *mut c_void, x: u32, y: u32, z: u32, flags: u32) {
    super::targets::note_dispatch();
    let pso = with(this, |s| s.pso);
    // lab XCL_RT_WATCH: the record of a watched pipeline (None when off)
    let watched = super::rt_watch::watched(pso);
    if let Some(r) = watched {
        if let (Some(layout), args) = with(this, |s| (s.layout.clone(), s.args.clone())) {
            super::rt_watch::note_args(r, &layout.params, &args);
        }
    }
    if super::pso_async::is_pending(pso) && super::list_x::defer(this, super::list_x::Deferred::Dispatch(x, y, z, flags)) {
        super::rt_watch::seen(watched, super::rt_watch::Outcome::PutOff);
        return super::pso_async::need_by_submit(pso);
    }
    let pso = super::pso_async::resolve_wait(pso);
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !super::real::is_translated_compute(pso) {
        super::rt_watch::seen(watched, super::rt_watch::Outcome::Untranslated);
        // a console pipeline without a translation (including a background build that
        // failed): nothing to run (its root arguments were not given to the PC list either)
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: Dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
        }
        skip_note(this, line!());
        count(DISPATCHES_SKIPPED);
        return;
    }
    let Some((groups, limits)) = dispatch_groups(pso, [x, y, z], lab_dispatch_flags(flags, thread_dimensions())) else {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: DispatchX of {x}x{y}x{z} threads (flags {flags:#x}) with a pipeline whose group shape is not known ({pso:#x}) skipped");
        }
        skip_note(this, line!());
        count(DISPATCHES_SKIPPED);
        super::rt_watch::seen(watched, super::rt_watch::Outcome::Failed);
        return;
    };
    if flags != 0 {
        // each kind of flags once: what titles pass, and what it launches
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let first = { let mut g = SEEN.lock().unwrap(); !g.contains(&flags) && { g.push(flags); true } };
        if first {
            crate::xlog_info!("D3D12.X: DispatchX with flags {flags:#x}: {x}x{y}x{z} launched as {}x{}x{} groups (thread limits {limits:?})", groups[0], groups[1], groups[2]);
        }
    }
    with(this, |s| s.limits = limits);
    let done = translated_dispatch(this, list, pso, groups[0], groups[1], groups[2]);
    with(this, |s| s.limits = [0; 3]);
    match done {
        Ok(()) => {
            count(DISPATCHES);
            super::rt_watch::seen(watched, super::rt_watch::Outcome::Launched);
        }
        Err(e) => {
            skip_note(this, line!());
            count(DISPATCHES_SKIPPED);
            super::rt_watch::seen(watched, super::rt_watch::Outcome::Failed);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: Dispatch with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: a full-screen dispatch (enough groups to cover a 4K target in tiles of up
/// to 128 pixels) – what its root arguments say. A bindless title passes the indices of its
/// descriptors in root constants and constant buffers: every small value there is looked up
/// in the dispatch's descriptor tables, and the resources found (the pass's inputs and
/// outputs) are logged.
unsafe fn trace_fullscreen(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&pso) || g.len() >= 12 {
            return;
        }
        g.push(pso);
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | hi << 32) as usize;
                let n = 64usize;
                if va != 0 && crate::gpu::shader::readable(va, 4 * n) {
                    let w: Vec<u32> = (0..n).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect();
                    what.push(format!("p{i} buffer {va:#x}: {w:x?}"));
                    values.extend(&w);
                } else {
                    what.push(format!("p{i} buffer {va:#x} (not readable)"));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    let mut found: Vec<(u32, usize)> = Vec::new();
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            let slot = u.base() + t as usize + 32 * v as usize;
            if let Some(r) = super::console_heap::resource_at(slot) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((v, r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .map(|&(v, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
            format!("[{v}] {r:#x} {:?}{}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), if super::present::is_presented(r) { " PRESENTED" } else { "" })
        })
        .collect();
    crate::xlog_info!("frame trace: full-screen dispatch {groups:?}, pipeline {pso:#x}: {}", what.join("; "));
    crate::xlog_info!("frame trace: … its indices name: {}", if res.is_empty() { "nothing".to_string() } else { res.join("; ") });
}

/// #148 AllocateUploadMemory(size, alignment) → CPU/GPU pointer in unified memory (used by
/// `XboxBvhBuilder.dll` and other console middleware on default title command lists).
/// Freed automatically when the list's command allocator is reset.
pub(super) unsafe extern "system" fn allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else {
        return std::ptr::null_mut();
    };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with(this, |s| s.arena.take(u, size, align, &mut s.uma)) {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with(this, |s| s.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with(this, |s| s.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

/// lab: `XCL_POST_PROBE=<seconds>` – from then on, what the tone-map draw reads, as the CPU
/// sees it (24 times, every 4th draw of PS_Composite 8d7a9eda9b47fd27 / PS_CompositeNMB
/// 4adcdc125462510a; `XCL_POST_PROBE_HASH` = another PS). The program (disassembly in
/// scratchpad\agents\exposure\rec): s8 = the frame's table, s9 = the pass's table.
/// - its constants (V# at s9 + 0x540): +0x514 != 0 chooses the LUT path (scene x exposure,
///   bloom, 3-D LUT at +0x240, then the output LUT at +0x3e0), 0 the curve of the buffer at
///   +0x20 (written by PS_CalculateExposure with a pixel-shader UAV store);
/// - the exposure is texel (0, 0) of the 1x1 RGBA32F target at s9 + 0x380 (written by
///   PS_CalculateExposure / PS_ResetExposure: x = exposure, y = its log2), loaded by the VERTEX
///   shader (VS_PassthroughComposite) and passed down as TEXCOORD0.z (the LUT path multiplies
///   the scene by it: 0 there = a constant image) and TEXCOORD1.x (log2, the curve path);
/// - the pre-exposure scalar is dword 0 of the buffer at s8 + 0x80.
/// Per texture: size, format, evicted, and the state of a texture kept in step. With
/// `XCL_RT_DUMP` the textures it samples are dumped too (`rtd_<present>_<k>_s_...`): the raw
/// texel of the 1x1 exposure target is in the "most common" list (w z y x, x = exposure).
/// lab: `XCL_RT_WRITERS=<seconds>` – from then on, once per (VS, PS, target) of a translated
/// draw into a target at least 1920 wide: the shaders' record hashes and the target (which pass
/// writes the lit target), and again at its 50th draw.
unsafe fn rt_writers_probe(this: *mut c_void, pso: usize) {
    static FROM: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let Some(from) = *FROM.get_or_init(|| std::env::var("XCL_RT_WRITERS").ok().and_then(|v| v.trim().parse().ok())) else { return };
    if crate::log::uptime() < from {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    let Some(r) = super::present::rt0_resource(this as usize) else { return };
    let Some(d) = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc()) else { return };
    if d.Width < 1920 {
        return;
    }
    static SEEN: Mutex<Option<HashMap<(u64, u64, usize), u32>>> = Mutex::new(None);
    let n = {
        let mut g = SEEN.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        if m.len() >= 600 && !m.contains_key(&(vs, ps, r)) {
            return;
        }
        let n = m.entry((vs, ps, r)).or_insert(0);
        *n += 1;
        *n
    };
    if n == 1 || n == 50 {
        crate::xlog_info!("rt writer: vs {vs:016x} ps {ps:016x} into {r:#x} {}x{} f{} at {:.1} s (draw {n})", d.Width, d.Height, d.Format.0, crate::log::uptime());
    }
}

unsafe fn post_probe(this: *mut c_void, pso: usize) {
    static CFG: std::sync::OnceLock<Option<(f64, Option<u64>)>> = std::sync::OnceLock::new();
    let Some((from, only)) = *CFG.get_or_init(|| {
        let from = std::env::var("XCL_POST_PROBE").ok()?.trim().parse().ok()?;
        Some((from, std::env::var("XCL_POST_PROBE_HASH").ok().and_then(|v| u64::from_str_radix(v.trim(), 16).ok())))
    }) else {
        return;
    };
    if crate::log::uptime() < from {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    if !only.map_or(matches!(ps, 0x8d7a_9eda_9b47_fd27 | 0x4adc_dc12_5462_510a), |h| h == ps) {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let args = with(this, |s| s.args_gfx.clone());
    let word = |i: usize| args.get(4 * i..4 * i + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    let (t0, t1) = (u.base() + word(0) as usize, u.base() + word(1) as usize);
    // lab: a T# in a table is a copy of a heap slot's: its resource is the one of the slot of the
    // title's bound view heap that holds the same index (word 0)
    let heap = super::list_x::title_view_heap(this) as usize;
    let resolve = |slot: usize| -> Option<usize> {
        if let Some(r) = super::console_heap::resource_at(slot) {
            return Some(r);
        }
        if !crate::gpu::shader::readable(slot, 4) {
            return None;
        }
        let w0 = (slot as *const u32).read_unaligned();
        if let Some(r) = super::console_heap::resource_by_word0(w0) {
            return Some(r);
        }
        if heap == 0 {
            return None;
        }
        let mut n = 65536usize;
        while n > 256 && !crate::gpu::shader::readable(heap, 32 * n) {
            n /= 2;
        }
        (0..n).find(|&k| ((heap + 32 * k) as *const u32).read_unaligned() == w0 && super::console_heap::resource_at(heap + 32 * k).is_some()).and_then(|k| super::console_heap::resource_at(heap + 32 * k))
    };
    // with `XCL_RT_DUMP`: the exposure texel, the LUTs (3-D: their first slice), the curve
    // and vignette textures, bloom and the scene go into the next dump as kind `s`
    if super::present::rt_dump_on() {
        for off in [0x380usize, 0x240, 0x300, 0xc0, 0x3e0, 0x2c0, 0x1c0, 0xa0, 0x4e0, 0x4c0] {
            if let Some(r) = resolve(t1 + off) {
                super::present::note_rt_dump(r, b's');
            }
        }
    }
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if CALLS.fetch_add(1, Ordering::Relaxed) % 4 != 0 || LOGGED.fetch_add(1, Ordering::Relaxed) >= 24 {
        return;
    }
    let cpu = |p: usize, n: usize| -> Option<Vec<u32>> {
        crate::gpu::shader::readable(p, 4 * n).then(|| (0..n).map(|i| ((p + 4 * i) as *const u32).read_unaligned()).collect())
    };
    let hex = |v: &Option<Vec<u32>>| v.as_ref().map_or("unreadable".to_string(), |v| v.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(" "));
    // a V#: its words, where its bytes are (a 32-bit address is in window 0) and the first `n` dwords from `off`
    let buffer = |slot: usize, off: usize, n: usize| -> String {
        let v = cpu(slot, 4);
        let Some(w) = &v else { return "V# unreadable".into() };
        let a = (u64::from(w[1] & 0xFFFF) << 32) | u64::from(w[0]);
        let p = if a >> 32 == 0 { u.base() + a as usize } else { a as usize };
        format!("V# [{}] at {p:#x} win {:?} +{off:#x}: [{}]", hex(&v), p.checked_sub(u.base()).map(|d| d >> 32), hex(&cpu(p + off, n)))
    };
    let texture = |slot: usize| -> String {
        let Some(r) = resolve(slot) else { return format!("no view (T# [{}])", hex(&cpu(slot, 8))) };
        let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
        let mut s = d.map_or("?".to_string(), |d| format!("dim {} {}x{}x{} f{} mips {}", d.Dimension.0, d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels));
        if super::vram::is_evicted(r) {
            s += " EVICTED";
        }
        if let Some((va, src, pending, gpu_written, unmapped, shadowed)) = super::title_textures::probe_state(r) {
            s += &format!(" kept at {va:#x} src {src:#x} pending {pending} gpu-written {gpu_written} unmapped {unmapped} shadowed {shadowed}");
        }
        s
    };
    let rt = super::present::rt0_resource(this as usize).and_then(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }));
    crate::xlog_info!("post probe: vs {vs:016x} ps {ps:016x} into {rt:?} at {:.1} s: s8 {:#x} s9 {:#x}; pre-exposure (s8+0x80) {}; frame constants (s8+0xc0) {}", crate::log::uptime(), word(0), word(1), buffer(t0 + 0x80, 0, 4), buffer(t0 + 0xc0, 0x5f0, 1));
    // lab (with `XCL_EI_PEEK`): the same buffers as the GPU holds them (the CPU does not see GPU
    // writes into shadowed unified memory): 12 bytes each, logged by `ei_peek_report` as kind
    // 30 (pre-exposure, s8 + 0x80), 31 (s9 + 0x0), 32 (the curve, s9 + 0x20)
    if ei_peek_from().is_some() {
        for (kind, slot) in [(30u8, t0 + 0x80), (31, t1), (32, t1 + 0x20)] {
            let Some(w) = cpu(slot, 4) else { continue };
            let a = (u64::from(w[1] & 0xFFFF) << 32) | u64::from(w[0]);
            let p = if a >> 32 == 0 { u.base() + a as usize } else { a as usize };
            if let Some((win, at)) = super::real::uma_window_buffer(p, 16) {
                ei_peek_slot(this, kind, 0, win, at, 0, 0);
            }
        }
    }
    let cb = t1 + 0x540;
    crate::xlog_info!(
        "post probe: constants {} | +0x40 {} | +0x220 {} | +0x3c0 {} | +0x3e0 {} | +0x400 {} | +0x480 {} | +0x500 {}",
        buffer(cb, 0, 4),
        buffer(cb, 0x40, 2),
        buffer(cb, 0x220, 4),
        buffer(cb, 0x3c0, 5),
        buffer(cb, 0x3e0, 4),
        buffer(cb, 0x400, 12),
        buffer(cb, 0x480, 8),
        buffer(cb, 0x500, 18)
    );
    crate::xlog_info!("post probe: buffers: +0x0 {} | +0x20 (curve) {} | +0x60 {} | +0x500 {} | +0x520 {}", buffer(t1, 0, 8), buffer(t1 + 0x20, 0, 8), buffer(t1 + 0x60, 0, 16), buffer(t1 + 0x500, 0, 16), buffer(t1 + 0x520, 0, 20));
    let names = [
        (0x380, "exposure 1x1 (VS)"),
        (0x4c0, "scene"),
        (0x400, "scene lz"),
        (0x80, "image_load"),
        (0x480, "blurred"),
        (0x100, "local"),
        (0x1c0, "curve lut"),
        (0x4a0, "grid a"),
        (0x3a0, "grid b"),
        (0x4e0, "bloom"),
        (0x2c0, "splines"),
        (0x240, "LUT 1"),
        (0x300, "LUT 1b"),
        (0xc0, "LUT 1c"),
        (0x460, "LUT mask"),
        (0x3e0, "LUT 2 (output)"),
        (0xa0, "vignette"),
        (0x420, "overlay"),
    ];
    let t: Vec<String> = names.iter().map(|&(off, name)| format!("{name} +{off:#x}: {}", texture(t1 + off))).collect();
    crate::xlog_info!("post probe: textures: {} | s8+0x2a0 noise: {}", t.join(" | "), texture(t0 + 0x2a0));
}

/// lab: `XCL_DISPATCH_PROBE=<seconds>` – from then on, once per compute pipeline (up to 800):
/// its console CS (record hash), groups, and the textures its root arguments can name: every
/// small value of its constants and root buffers looked up as a slot of its descriptor tables
/// and of the title's bound view heap (as `trace_fullscreen` does), plus the first 48 slots of
/// each table. `WxH fN` per texture; the ones the title presents are marked.
unsafe fn dispatch_probe(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    static FROM: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let Some(from) = *FROM.get_or_init(|| std::env::var("XCL_DISPATCH_PROBE").ok().and_then(|v| v.trim().parse().ok())) else { return };
    if crate::log::uptime() < from {
        return;
    }
    let real = super::pso_async::resolve(pso);
    static SEEN: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);
    {
        let mut g = SEEN.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        if m.len() >= 800 && !m.contains_key(&real) {
            return;
        }
        let n = m.entry(real).or_insert(0);
        *n += 1;
        if *n != 1 && *n != 40 {
            return;
        }
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<usize> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    for p in layout.params.iter() {
        match *p {
            Param::Constants { at, count } => values.extend((0..count as usize).filter_map(|k| word(at as usize + 4 * k))),
            Param::Descriptor { at } => {
                let va = (word(at as usize).unwrap_or(0) as u64 | (word(at as usize + 4).unwrap_or(0) as u64) << 32) as usize;
                if va != 0 && crate::gpu::shader::readable(va, 4 * 96) {
                    values.extend((0..96).map(|k| ((va + 4 * k) as *const u32).read_unaligned()));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                if v != 0 {
                    tables.push(u.base() + v as usize);
                }
            }
        }
    }
    let heap = super::list_x::title_view_heap(this) as usize;
    let mut found: Vec<(String, usize)> = Vec::new();
    let mut add = |tag: String, slot: usize, found: &mut Vec<(String, usize)>| {
        // a table's T# is a copy: its resource is the one of any slot with the same index (word 0)
        let by_index = || {
            let w0 = crate::gpu::shader::readable(slot, 4).then(|| (slot as *const u32).read_unaligned())?;
            (w0 != 0).then(|| super::console_heap::resource_by_word0_cached(w0)).flatten()
        };
        if let Some(r) = super::console_heap::resource_at(slot).or_else(by_index) {
            if !found.iter().any(|x| x.1 == r) {
                found.push((tag, r));
            }
        }
    };
    for (ti, &t) in tables.iter().enumerate() {
        for k in 0..48usize {
            add(format!("t{ti}[{k}]"), t + 32 * k, &mut found);
        }
        for &v in values.iter().filter(|&&v| v < 1 << 20) {
            add(format!("t{ti}[{v}]"), t + 32 * v as usize, &mut found);
        }
    }
    if heap != 0 {
        for &v in values.iter().filter(|&&v| v < 1 << 20) {
            add(format!("h[{v}]"), heap + 32 * v as usize, &mut found);
        }
    }
    let res: Vec<String> = found
        .iter()
        .filter_map(|(tag, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(*r as *mut c_void)).map(|x| x.GetDesc())?;
            (d.Dimension.0 >= 2).then(|| format!("{tag} {r:#x} {}x{} f{}{}", d.Width, d.Height, d.Format.0, if super::present::is_presented(*r) { " PRESENTED" } else { "" }))
        })
        .collect();
    let cs = super::real::pso_shaders(real).map_or(0, |s| s.0);
    crate::xlog_info!("dispatch probe: cs {cs:016x} groups {groups:?} at {:.1} s: {}", crate::log::uptime(), res.join(", "));
}

/// lab (`XCL_RT_DUMP`): within a second before a dump is due, the textures in the first 64
/// slots of a compute dispatch's descriptor tables go into the dump as kind `u` (what compute
/// passes read and write: light lists, probe atlases, volumes), whatever their size.
/// lab (`depth_twin`): the (depth buffer, plane) pairs whose twin a view in the descriptor tables
/// of the compute dispatch being bound names (the first 256 slots of each table).
unsafe fn depth_twins_named(this: *mut c_void) -> Vec<(usize, u32)> {
    let mut out: Vec<(usize, u32)> = Vec::new();
    if !super::depth_twin::any() {
        return out;
    }
    let Some(u) = crate::gpu::uma::get() else { return out };
    let (layout, args, pso) = with(this, |s| (s.layout.clone(), s.args.clone(), s.pso));
    let Some(layout) = layout else { return out };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let v = word(at as usize).unwrap_or(0);
        if v == 0 {
            continue;
        }
        let t = u.base() + v as usize;
        let mut n = 256usize;
        while n > 0 && !crate::gpu::shader::readable(t, 32 * n) {
            n /= 2;
        }
        for k in 0..n {
            let d = ((t + 32 * k) as *const [u32; 4]).read_unaligned();
            // (an image descriptor: type in the top bits of word 3)
            if d[3] >> 28 < 8 {
                continue;
            }
            if let Some(x) = super::depth_twin::by_word0(d[0]) {
                if !out.contains(&x) {
                    out.push(x);
                    // (alias pairs, plane 2, are named all the time: the others have their own lines)
                    static N: AtomicU64 = AtomicU64::new(0);
                    static OTHER: AtomicU64 = AtomicU64::new(0);
                    if (if x.1 == 2 { &N } else { &OTHER }).fetch_add(1, Ordering::Relaxed) < 40 {
                        let cs = super::real::pso_shaders(super::pso_async::resolve(pso)).map_or(0, |s| s.0);
                        crate::xlog_info!("lab: depth twin view of {:#x} plane {} in slot {k} of a table of cs {cs:016x} at {:.1} s", x.0, x.1, crate::log::uptime());
                    }
                }
            }
        }
    }
    out
}

unsafe fn rt_dump_note_compute(this: *mut c_void) {
    if !super::present::rt_dump_due_soon() {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args, pso) = with(this, |s| (s.layout.clone(), s.args.clone(), s.pso));
    let Some(layout) = layout else { return };
    // lab: `XCL_RT_DUMP_ONLY=<cs record hash>,...`: only what these passes name (every texture in
    // the first 192 slots of their tables), each also as a `.raw` file
    let only = super::present::rt_dump_only();
    if !only.is_empty() {
        let cs = super::real::pso_shaders(super::pso_async::resolve(pso)).map_or(0, |s| s.0);
        if !only.contains(&cs) {
            return;
        }
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let v = word(at as usize).unwrap_or(0);
        if v == 0 {
            continue;
        }
        let t = u.base() + v as usize;
        for k in 0..if only.is_empty() { 128usize } else { 192 } {
            let slot = t + 32 * k;
            if !crate::gpu::shader::readable(slot, 32) {
                break;
            }
            let w0 = (slot as *const u32).read_unaligned();
            let r = super::console_heap::resource_at(slot).or_else(|| (w0 != 0).then(|| super::console_heap::resource_by_word0_cached(w0)).flatten());
            // (render targets, depth buffers and UAV textures: not the streamed ones)
            // (a slot may still name a texture the title released: its object must at least
            // have a method table – a run ended in GetDesc of such a one)
            let alive = |r: usize| crate::gpu::shader::readable(r, 8) && crate::gpu::shader::readable((r as *const usize).read_unaligned(), 8 * 12);
            if let Some(r) = r.filter(|&r| alive(r) && ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| { let d = x.GetDesc(); !only.is_empty() && d.Dimension.0 >= 2 || d.Flags.0 & 7 != 0 || d.Format.0 == 67 })) {
                super::present::note_rt_dump(r, if only.is_empty() { b'u' } else { b'o' });
            }
        }
    }
}

unsafe fn translated_dispatch(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, x: u32, y: u32, z: u32) -> Result<(), String> {
    if super::present::trace_frame() {
        trace_fullscreen(this, pso, [x, y, z]);
    }
    dispatch_probe(this, pso, [x, y, z]);
    super::clamp_runs::write(this as usize);
    super::indirect_clamp::run_stop(this as usize);
    rt_dump_note_compute(this);
    let twins = depth_twins_named(this);
    for &(res, plane) in &twins {
        super::list_x::depth_twin_fill(this, res, plane);
    }
    pass_fill_at(this, pso, true);
    let timed = gpu_time_begin(this, pso);
    bind_abi(this, list, pso, false)?;
    gp_mark(this, gp::Label::of(gp::Kind::Other));
    let cs = gp_cs(pso);
    let mut pieces = 0u32;
    let ran = super::real::dispatch_chain::dispatch(list, pso, with(this, |s| s.allocator), [x, y, z], |i| {
        pieces += 1;
        gp_mark(this, gp::Label::op(gp::Kind::ChainPiece, (cs.0, i as u64), [x, y, z]));
    });
    if pieces == 0 {
        gp_mark(this, gp::Label::op(gp::Kind::Dispatch, cs, [x, y, z]));
    }
    gpu_time_end(this, timed);
    for (res, plane) in twins {
        super::list_x::depth_twin_flush(this, res, plane);
    }
    pass_snap(this, pso);
    pass_fill(this, pso);
    ran
}

/// lab: `XCL_PASS_FILL=<cs record hash>:t<i>[<slot>][=<hex dword>]:...,<hash>:...` - right after
/// every dispatch of a named compute program, the textures in the named slots of its descriptor
/// tables (numbered as the dispatch probe and the pass snapshots print them) are filled with the
/// dword (default 0 = cleared). To take a pass's output out of the frame (its garbage with it),
/// or to put a constant in its place (an R9G9B9E5 / R11G11B10 value as hex).
/// `XCL_PASS_PREFILL` (the same form): the fill BEFORE each dispatch of the program - a marker in
/// its outputs; a snapshot after the pass then shows which texels the pass wrote.
unsafe fn pass_fill(this: *mut c_void, pso: usize) {
    pass_fill_at(this, pso, false)
}

unsafe fn pass_fill_at(this: *mut c_void, pso: usize, before: bool) {
    static SETS: [std::sync::OnceLock<Vec<(u64, Vec<(usize, usize, u32)>)>>; 2] = [std::sync::OnceLock::new(), std::sync::OnceLock::new()];
    let set = SETS[before as usize].get_or_init(|| {
        std::env::var(if before { "XCL_PASS_PREFILL" } else { "XCL_PASS_FILL" })
            .unwrap_or_default()
            .split(',')
            .filter_map(|e| {
                let mut f = e.trim().split(':');
                let cs = u64::from_str_radix(f.next()?.trim(), 16).ok()?;
                let slots = f
                    .filter_map(|s| {
                        let (slot, pat) = s.split_once('=').map_or((s, 0), |(a, b)| (a, u32::from_str_radix(b.trim(), 16).unwrap_or(0)));
                        let (t, k) = slot.trim().strip_prefix('t')?.strip_suffix(']')?.split_once('[')?;
                        Some((t.parse().ok()?, k.parse().ok()?, pat))
                    })
                    .collect();
                Some((cs, slots))
            })
            .collect()
    });
    if set.is_empty() {
        return;
    }
    let cs = super::real::pso_shaders(super::pso_async::resolve(pso)).map_or(0, |s| s.0);
    let Some((_, slots)) = set.iter().find(|e| e.0 == cs) else { return };
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let tables: Vec<usize> = layout
        .params
        .iter()
        .filter_map(|p| match *p {
            Param::Table { at } => word(at as usize).filter(|&v| v != 0).map(|v| u.base() + v as usize),
            _ => None,
        })
        .collect();
    let alive = |r: usize| crate::gpu::shader::readable(r, 8) && crate::gpu::shader::readable((r as *const usize).read_unaligned(), 8 * 12);
    for &(ti, k, pat) in slots {
        let Some(&t) = tables.get(ti) else { continue };
        let slot = t + 32 * k;
        if !crate::gpu::shader::readable(slot, 32) {
            continue;
        }
        let w0 = (slot as *const u32).read_unaligned();
        let r = super::console_heap::resource_at(slot).or_else(|| (w0 != 0).then(|| super::console_heap::resource_by_word0_cached(w0)).flatten());
        let done = r.filter(|&r| alive(r)).is_some_and(|r| super::list_x::pass_fill(this, r, pat));
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 24 || n.is_power_of_two() {
            crate::xlog_info!("pass {}fill #{n}: cs {cs:016x} t{ti}[{k}] {:#x} with {pat:#010x} at {:.1} s: {}", if before { "pre" } else { "" }, r.unwrap_or(0), crate::log::uptime(), if done { "done" } else { "NOT done" });
        }
    }
}

/// lab: `XCL_GPU_TIME=<seconds>` - GPU time by compute program. From that time, for 2.5 s (12000
/// calls at most), a timestamp query before and after each translated dispatch / indirect dispatch of
/// the title's lists; the present after that resolves them (`gpu_time_present`, on the presenter's
/// list) and one 4 s later prints the sums by program: "gpu time: ...". Ticks of the queue's
/// timestamp counter; passes on several queues overlap, so the shares are a guide, not a budget.
struct GpuTime {
    heap: ID3D12QueryHeap,
    rb: ID3D12Resource,
    /// the program of each pair
    meta: Vec<u64>,
    /// 0 collecting, 1 resolved (at `at`), 2 printed
    state: u8,
    at: f64,
}
static GPU_TIME: Mutex<Option<GpuTime>> = Mutex::new(None);
const GPU_TIME_PAIRS: usize = 12000;

fn gpu_time_from() -> Option<f64> {
    static FROM: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *FROM.get_or_init(|| std::env::var("XCL_GPU_TIME").ok().and_then(|v| v.trim().parse().ok()))
}

/// The pair's index when this call is timed (the first query is recorded).
unsafe fn gpu_time_begin(this: *mut c_void, pso: usize) -> Option<u32> {
    let from = gpu_time_from()?;
    let now = crate::log::uptime();
    if now < from || now > from + 2.5 {
        return None;
    }
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?;
    let mut g = GPU_TIME.lock().unwrap();
    if g.is_none() {
        let dev = super::real::list_device(this)?;
        let mut heap: Option<ID3D12QueryHeap> = None;
        dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2 * GPU_TIME_PAIRS as u32, NodeMask: 0 }, &mut heap).ok()?;
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 16 * GPU_TIME_PAIRS as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .ok()?;
        *g = Some(GpuTime { heap: heap?, rb: rb?, meta: Vec::new(), state: 0, at: 0.0 });
    }
    let t = g.as_mut()?;
    if t.state != 0 || t.meta.len() >= GPU_TIME_PAIRS {
        return None;
    }
    let i = t.meta.len() as u32;
    t.meta.push(super::real::pso_shaders(super::pso_async::resolve(pso)).map_or(0, |s| s.0));
    list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i);
    Some(i)
}

unsafe fn gpu_time_end(this: *mut c_void, i: Option<u32>) {
    let Some(i) = i else { return };
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if let Some(t) = GPU_TIME.lock().unwrap().as_ref() {
        list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i + 1);
    }
}

/// At a present (the presenter's list, recorded after the frame's lists were submitted).
pub(super) unsafe fn gpu_time_present(list: &ID3D12GraphicsCommandList) {
    let Some(from) = gpu_time_from() else { return };
    let now = crate::log::uptime();
    let mut g = GPU_TIME.lock().unwrap();
    let Some(t) = g.as_mut() else { return };
    if t.state == 0 && !t.meta.is_empty() && now > from + 3.5 {
        list.ResolveQueryData(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2 * t.meta.len() as u32, &t.rb, 0);
        t.state = 1;
        t.at = now;
    } else if t.state == 1 && now > t.at + 4.0 {
        t.state = 2;
        let mut p: *mut c_void = std::ptr::null_mut();
        if t.rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
            return;
        }
        let q = |k: usize| (p as *const u64).add(k).read_unaligned();
        let mut by: HashMap<u64, (u64, u32)> = HashMap::new();
        let (mut total, mut bad, mut first, mut last) = (0u64, 0u32, u64::MAX, 0u64);
        for (i, &cs) in t.meta.iter().enumerate() {
            let (a, b) = (q(2 * i), q(2 * i + 1));
            if a == 0 || b < a {
                bad += 1;
                continue;
            }
            let e = by.entry(cs).or_default();
            e.0 += b - a;
            e.1 += 1;
            total += b - a;
            first = first.min(a);
            last = last.max(b);
        }
        // the time BETWEEN the timed calls, in the order the GPU ran them: the largest gaps and the
        // programs on either side (what lies there - draws, copies, waits - is the rest of the frame)
        let mut line: Vec<(u64, u64, u64)> = t.meta.iter().enumerate().map(|(i, &cs)| (q(2 * i), q(2 * i + 1), cs)).filter(|&(a, b, _)| a != 0 && b >= a).collect();
        line.sort_unstable();
        let mut gaps: Vec<(u64, u64, u64)> = line.windows(2).filter(|w| w[1].0 > w[0].1).map(|w| (w[1].0 - w[0].1, w[0].2, w[1].2)).collect();
        gaps.sort_unstable_by_key(|g| std::cmp::Reverse(g.0));
        let between: u64 = gaps.iter().map(|g| g.0).sum();
        crate::xlog_info!(
            "gpu time gaps: {between} ticks between the timed calls in {} gaps; the largest (ticks, after program, before program): {}",
            gaps.len(),
            gaps.iter().take(30).map(|(n, a, b)| format!("{n} {a:016x} {b:016x}")).collect::<Vec<_>>().join("; ")
        );
        t.rb.Unmap(0, None);
        let mut v: Vec<_> = by.into_iter().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1 .0));
        crate::xlog_info!(
            "gpu time: {} calls timed from {from:.1} s ({bad} without a result), {total} ticks in them, {} ticks from the first to the last; by program (ticks, share, calls): {}",
            t.meta.len(),
            last.saturating_sub(first.min(last)),
            v.iter().take(40).map(|(cs, (ticks, n))| format!("{cs:016x} {ticks} {:.1}% {n}", 100.0 * *ticks as f64 / total.max(1) as f64)).collect::<Vec<_>>().join("; ")
        );
    }
}

/// lab: `XCL_PASS_SNAP=<cs record hash>,...` (hex; with `XCL_RT_DUMP`) - what the large targets a
/// compute pass names hold RIGHT AFTER that pass, not at the present (the title places several
/// transient targets at one address: at the present a target shows whatever used its memory
/// last). Within a second before a dump is due, after the first dispatch of a named
/// program: every single-sampled 2-D colour / UAV target at least 1280 wide in the first 64
/// slots of its tables (4 at most) is copied in the list to a texture of its own, which goes
/// into the dump as kind `p`. The log line maps the copies to the pass and the slot.
unsafe fn pass_snap(this: *mut c_void, pso: usize) {
    // (`<hash>:t1[12]:t1[13]`: exactly these slots of the program's tables, whatever they hold -
    // the automatic choice stops at the first 4 / 12 textures and names a texture by its first slot)
    static SET: std::sync::OnceLock<Vec<(u64, Vec<(usize, usize)>)>> = std::sync::OnceLock::new();
    let set = SET.get_or_init(|| {
        std::env::var("XCL_PASS_SNAP")
            .unwrap_or_default()
            .split(',')
            .filter_map(|e| {
                let mut f = e.trim().split(':');
                let cs = u64::from_str_radix(f.next()?.trim(), 16).ok()?;
                let slots = f
                    .filter_map(|s| {
                        let (t, k) = s.trim().strip_prefix('t')?.strip_suffix(']')?.split_once('[')?;
                        Some((t.parse().ok()?, k.parse().ok()?))
                    })
                    .collect();
                Some((cs, slots))
            })
            .collect()
    });
    if set.is_empty() || !super::present::rt_dump_due_soon() {
        return;
    }
    let cs = super::real::pso_shaders(super::pso_async::resolve(pso)).map_or(0, |s| s.0);
    let Some((_, named)) = set.iter().find(|e| e.0 == cs) else { return };
    let now = crate::log::uptime();
    // (program, time) of the calls taken; the copies made, kept until their lists have long run
    static CALLS: Mutex<Vec<(u64, f64)>> = Mutex::new(Vec::new());
    static KEPT: Mutex<Vec<(usize, f64)>> = Mutex::new(Vec::new());
    let call = {
        let mut g = CALLS.lock().unwrap();
        // (the first call of a frame: a present between the copy and the dump lets the set go)
        // (4 s: one set a dump. Once a frame (0.3 s) ended run Q1 in the driver at the dump)
        g.retain(|&(_, t)| now - t < 4.0);
        let n = g.iter().filter(|c| c.0 == cs).count();
        if n >= 1 {
            return;
        }
        g.push((cs, now));
        n
    };
    {
        let mut g = KEPT.lock().unwrap();
        let old: Vec<usize> = g.iter().filter(|&&(_, t)| now - t > 30.0).map(|&(r, _)| r).collect();
        g.retain(|&(_, t)| now - t <= 30.0);
        for r in old {
            drop(ID3D12Resource::from_raw(r as *mut c_void));
        }
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let alive = |r: usize| crate::gpu::shader::readable(r, 8) && crate::gpu::shader::readable((r as *const usize).read_unaligned(), 8 * 12);
    // `XCL_PASS_SNAP_ANY=1`: every 2-D / 3-D texture at least 16 wide, a target or not (12 at most):
    // for small passes (the colour LUT bakes and what they read)
    static ANY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let any = *ANY.get_or_init(|| std::env::var("XCL_PASS_SNAP_ANY").is_ok_and(|v| v.trim() == "1"));
    let (most, least) = if any { (12, 16) } else { (4, 1280) };
    let mut seen: Vec<usize> = Vec::new();
    let mut ti = 0;
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let v = word(at as usize).unwrap_or(0);
        if v == 0 {
            continue;
        }
        let t = u.base() + v as usize;
        let ks: Vec<usize> = if named.is_empty() { (0..64).collect() } else { named.iter().filter(|s| s.0 == ti).map(|s| s.1).collect() };
        for k in ks {
            let slot = t + 32 * k;
            if named.is_empty() && seen.len() >= most || !crate::gpu::shader::readable(slot, 32) {
                break;
            }
            let w0 = (slot as *const u32).read_unaligned();
            let r = super::console_heap::resource_at(slot).or_else(|| (w0 != 0).then(|| super::console_heap::resource_by_word0_cached(w0)).flatten());
            let Some(r) = r.filter(|&r| alive(r) && (!named.is_empty() || !seen.contains(&r))) else { continue };
            let rr = r as *mut c_void;
            let Some(d) = ID3D12Resource::from_raw_borrowed(&rr).map(|x| x.GetDesc()) else { continue };
            let target = d.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) != 0 && d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 == 0;
            let shape = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D || any && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            let depth = d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0;
            let shape = shape || !named.is_empty() && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            if !shape || d.SampleDesc.Count > 1 || depth || named.is_empty() && (d.Width < least || !(target || any)) {
                continue;
            }
            seen.push(r);
            let Some(snap) = super::list_x::pass_snapshot(this, r) else { continue };
            KEPT.lock().unwrap().push((snap, now));
            super::present::note_rt_dump(snap, b'p');
            crate::xlog_info!("pass snap: cs {cs:016x} call {call} at {now:.2} s: t{ti}[{k}] {r:#x} {}x{} f{} copied to {snap:#x}{}", d.Width, d.Height, d.Format.0, if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { format!(" (3-D, {} slices)", d.DepthOrArraySize) } else { String::new() });
        }
        ti += 1;
    }
}

/// The CPU addresses of the descriptor tables root arguments `args` of `layout` bind (32-bit
/// console addresses, in unified memory at `base`); tables at 0 are none.
fn table_addresses(layout: &Layout, args: &[u8], base: usize) -> Vec<usize> {
    let word = |at: usize| args.get(at..at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    layout
        .params
        .iter()
        .filter_map(|p| match *p {
            Param::Table { at } => Some(word(at as usize)),
            _ => None,
        })
        .filter(|&t| t != 0)
        .map(|t| base + t as usize)
        .collect()
}

/// XCL's ABI for a translated pipeline: its root signature, the user SGPRs (`io`, the kept
/// arguments the console's way – compute or graphics) and the argument table in unified
/// memory, u1 res, u2… the memory windows; then the pipeline.
unsafe fn bind_abi(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, gfx: bool) -> Result<(), String> {
    super::list_x::ensure_view_heap(this);
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    // shadowed unified memory whose sparse windows could not be made (no reserved resources
    // either): descriptor tables are read from window 0, and a null root descriptor is not bound
    if first == 0 && u.shadowed() {
        return Err("window 0 of unified memory has no buffer on the GPU".into());
    }
    let (layout, mut args) = with(this, |s| if gfx { (s.layout_gfx.clone(), s.args_gfx.clone()) } else { (s.layout.clone(), s.args.clone()) });
    // lab: a root address in a buffer the title placed outside the windows: where that buffer
    // lives in them (`real::place_outside_buffer`)
    if let Some(l) = layout.as_deref() {
        for p in l.params.iter() {
            let Param::Descriptor { at } = *p else { continue };
            let at = at as usize;
            let Some(b) = args.get(at..at + 8) else { continue };
            let va = u64::from_le_bytes(b.try_into().unwrap()) as usize;
            if va >> 32 >= 0x14 {
                if let Some(q) = super::real::outside_buffer_address(va) {
                    args[at..at + 8].copy_from_slice(&(q as u64).to_le_bytes());
                    super::sparse::use_range(q, 1);
                    static N: AtomicU64 = AtomicU64::new(0);
                    let k = N.fetch_add(1, Ordering::Relaxed) + 1;
                    if k <= 8 || k.is_power_of_two() {
                        crate::xlog_info!("lab: root address #{k} {va:#x} in an outside buffer -> {q:#x}");
                    }
                }
            }
        }
    }
    // lab: lazily placed buffers the tables name get their video memory (`sparse::use_range`)
    if super::sparse::lazy_buffer_mb().0 != 0 {
        if let Some(l) = layout.as_deref() {
            // (lab XCL_USE_RANGE_HIT=1: each address once)
            let mut seen = super::sparse::use_range_hit().then(super::sparse::SeenAddrs::new);
            for table in table_addresses(l, &args, u.base()) {
                if !crate::gpu::shader::readable(table, 32 * 64) {
                    continue;
                }
                for k in 0..64 {
                    let d = ((table + 32 * k) as *const [u32; 4]).read_unaligned();
                    if d[3] >> 30 == 0 && (d[0] | d[1]) != 0 {
                        let va = ((d[1] as usize & 0xFFFF) << 32) | d[0] as usize;
                        if seen.as_mut().is_some_and(|s| !s.first(va)) {
                            continue;
                        }
                        super::sparse::use_range(va, 1);
                    }
                }
            }
        }
    }
    // lab: what the shaders read of the title's memory outside the unified windows
    super::view_mem::ensure_for_draw(layout.as_deref(), &args);
    // 32-bit console addresses: window 0; pieces of the list's arena (not a 64 KiB grain each)
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take_or(u, bytes.len().max(16), 256, &mut s.uma, super::sparse::arena_block)).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    // a pipeline still being built in the background is waited for
    let real = super::pso_async::resolve_wait(pso);
    // a compute shader takes no more arguments from registers than it has (`inline_for`)
    let user_sgprs = if gfx { None } else { super::real::pso_user_sgprs(real) };
    let inline = layout.as_ref().map_or(0, |l| if gfx { l.inline_gfx } else { console_list::inline_for(l, user_sgprs) } as usize).min(args.len());
    let table_at = copy(&args[inline..])?;
    let mut io = match layout.as_ref() {
        Some(l) if gfx => console_list::user_data_gfx(l, &args, table_at as u64),
        Some(l) => console_list::user_data_for(l, user_sgprs, &args, table_at as u64),
        None => vec![table_at as u32, (table_at >> 32) as u32],
    };
    let user = io.len();
    heap_sgpr(this, &mut io, layout.is_some());
    if gfx && io.len() > user {
        draw_textures(&io, user);
    }
    if !gfx && io.len() > user {
        skin_inputs(pso, &io, user);
    }
    // page budget: the textures in the bound descriptor tables are sampled
    if let Some(l) = layout.as_ref().filter(|_| super::vram::tracking()) {
        super::title_textures::tables_drawn(&table_addresses(l, &args, u.base()));
    }
    io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
    io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
    if !gfx {
        for (k, l) in with(this, |s| std::mem::take(&mut s.limits)).into_iter().enumerate() {
            io[crate::gpu::recomp::IO_THREAD_LIMIT[k]] = l;
        }
    }
    let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
    let res_at = copy(&[0u8; 256])?;
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let rs = super::real::pso_root_signature(real).or_else(|| super::real::abi_rs_cached().map(|p| p as usize)).ok_or("no XCL ABI root signature")?;
    let rs = rs as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs).ok_or("root signature")?;
    let views = [va(io_at), va(res_at)].into_iter().chain((0..4).map(|k| windows.get(k).copied().filter(|&v| v != 0).unwrap_or(first)));
    if gfx {
        list.SetGraphicsRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, v);
        }
    } else {
        list.SetComputeRootSignature(rs);
        for (k, v) in views.enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, v);
        }
    }
    let real = real as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&real).ok_or("pipeline")?;
    list.SetPipelineState(pc);
    Ok(())
}

#[cfg(test)]
pub(super) fn last_draw_args() -> (bool, u32, u32, u32, i32) {
    DRAW_ARGS.with(|d| d.get())
}

#[cfg(test)]
pub(super) fn counts_snapshot() -> [u64; 5] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

#[cfg(test)]
pub(super) fn tracked_pso(list: *mut c_void) -> usize {
    STATES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .map_or(0, |s| s.pso)
}

#[cfg(test)]
mod tests {
    use super::super::{device_x, list_x, pso_async, real, title_list};
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// One free list for both (as before `XCL_SPLIT_FREE_LISTS`): sixteen small scaled-argument
    /// buffers given back first fill it, the 4 MiB record buffer after them is released, and a
    /// recording that needs 4 MiB finds none. Two lists: the record buffer is kept and taken
    /// again, and a scaled-argument buffer is the smallest that fits, of the same device.
    #[test]
    fn scaled_argument_buffers_keep_out_of_the_record_buffers_free_list() {
        const MIB: u64 = 1 << 20;
        let (dev, other) = (0x1000usize, 0x2000usize);
        let scaled = |k: u32| (k, 64 << 10, dev);
        let record = (100u32, 4 * MIB, dev);
        // shared
        let mut shared = Vec::new();
        assert!(give_back(&mut shared, FREE_RES_KEPT, (0..16).map(scaled).collect()).is_empty());
        assert_eq!(give_back(&mut shared, FREE_RES_KEPT, vec![record]), vec![record], "the record buffer is released");
        assert_eq!(take_fitting(&mut shared, dev, 4 * MIB), None, "a new 4 MiB buffer is made");
        // split
        let (mut records, mut scaleds) = (Vec::new(), Vec::new());
        assert!(give_back(&mut scaleds, FREE_SCALED_KEPT, (0..16).map(scaled).collect()).is_empty());
        assert!(give_back(&mut records, FREE_RES_KEPT, vec![record]).is_empty());
        assert_eq!(take_fitting(&mut records, dev, 4 * MIB), Some(record));
        // the smallest that fits, of its device; over the limit is released
        assert!(give_back(&mut scaleds, FREE_SCALED_KEPT, vec![(50, 256 << 10, dev), (51, 128 << 10, other), (52, 128 << 10, dev)]).is_empty());
        assert_eq!(take_fitting(&mut scaleds, dev, 100 << 10), Some((52, 128 << 10, dev)));
        assert_eq!(take_fitting(&mut scaleds, dev, 200 << 10), Some((50, 256 << 10, dev)));
        assert_eq!(take_fitting(&mut scaleds, dev, 200 << 10), None);
        assert_eq!(scaleds.len(), 17);
        let over = give_back(&mut scaleds, FREE_SCALED_KEPT, (200..260).map(scaled).collect());
        assert_eq!((scaleds.len(), over.len()), (FREE_SCALED_KEPT, 17 + 60 - FREE_SCALED_KEPT));
    }

    /// The descriptor tables a draw binds are where its table arguments point (in unified
    /// memory), constants and root buffers are not tables, and an unset table is none.
    #[test]
    fn table_addresses_follow_the_table_arguments() {
        let layout = Layout {
            params: vec![Param::Table { at: 0 }, Param::Constants { at: 4, count: 2 }, Param::Descriptor { at: 12 }, Param::Table { at: 20 }, Param::Table { at: 24 }],
            size: 28,
            inline: 0,
            inline_gfx: 0,
            inline_ms: 0,
        };
        let mut args = vec![0u8; 28];
        args[0..4].copy_from_slice(&0x4dbc_9a80u32.to_le_bytes());
        args[4..12].copy_from_slice(&[7; 8]);
        args[12..20].copy_from_slice(&0x1234_5678_9abcu64.to_le_bytes());
        args[24..28].copy_from_slice(&0x0004_0000u32.to_le_bytes());
        let base = 0x7_0000_0000usize;
        assert_eq!(table_addresses(&layout, &args, base), vec![base + 0x4dbc_9a80, base + 0x4_0000]);
        assert!(table_addresses(&layout, &args[..2], base).is_empty(), "arguments not set yet");
    }

    /// A pipeline with the rect list GS draws its rect lists whole: draw_rects leaves them alone.
    #[test]
    fn rect_lists_with_the_rect_list_gs_are_not_expanded_on_the_cpu() {
        let (with_gs, without) = (0x7E57_4000usize, 0x7E57_5000usize);
        real::register_rect_list_gs_for_test(with_gs);
        assert!(!cpu_rect_expansion(with_gs));
        assert!(cpu_rect_expansion(without) && cpu_rect_expansion(0));
        real::forget_pso_for_test(with_gs);
        assert!(cpu_rect_expansion(with_gs));
    }

    /// An ExecuteIndirect whose bound pipeline is still being built is skipped (its stand-in
    /// with XCL's ABI root signature on the list is the debug layer's 953 and a hung GPU);
    /// a built or failed one goes to the PC call. The gate `execute_indirect` uses.
    #[test]
    fn execute_indirect_with_a_pipeline_still_building_is_skipped() {
        let (building, built, failed) = (0x7E57_6001usize, 0x7E57_6002usize, 0x7E57_6003usize);
        // lazy submits: no worker starts them, so the states stay under the test's control
        pso_async::submit_lazy(building, false, Box::new(|| 0x1234), 0);
        pso_async::submit_lazy(built, false, Box::new(|| 0x1235), 0);
        pso_async::submit_lazy(failed, false, Box::new(|| 0), 0);
        assert!(pso_async::is_pending(building));
        assert_eq!(pso_async::resolve_wait(built), 0x1235);
        assert_eq!(pso_async::resolve_wait(failed), failed);
        assert!(!pso_async::is_pending(built) && !pso_async::is_pending(failed));
        // never resolved: still pending (the skip case), then forgotten
        assert!(pso_async::is_pending(building));
        pso_async::released(building);
        pso_async::released(built);
        pso_async::released(failed);
        assert!(!pso_async::is_pending(building));
    }

    /// A dispatch writes an ExecuteIndirect's group count through another buffer over the same
    /// memory (as translated shaders write through the memory windows), a UAV barrier (what a
    /// console barrier becomes), then the ExecuteIndirect: it runs the count just written, not
    /// the one the memory held before the list (`fresh_indirect_arguments`; without it an RTX
    /// 5090 ran the old count in nearly every try).
    #[test]
    fn execute_indirect_reads_the_arguments_written_before_it() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use std::mem::{transmute_copy, ManuallyDrop};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory6, DXGI_CREATE_FACTORY_FLAGS};

        const TRIES: u32 = 40;
        // mode 0: write (x, 1, 1) at a[0..12]; mode 1: count the groups
        let src = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint mode; uint x; };
RWByteAddressBuffer a : register(u0);
RWByteAddressBuffer c : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint t : SV_GroupIndex) {
    if (t != 0u) return;
    if (mode == 0u) a.Store3(0, uint3(x, 1u, 1u)); else c.InterlockedAdd(0, 1u);
}
"#;
        if crate::gpu::recomp::dxc::location().is_err() {
            skip_test("no DXC");
            return;
        }
        unsafe {
            let setup = || -> Result<(ID3D12Device, String), String> {
                let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI factory unavailable: {e}"))?;
                let sel = crate::gpu::adapter::Selector::from_env_or("auto");
                let (adapter, name): (IDXGIAdapter1, String) = if sel == crate::gpu::adapter::Selector::Warp {
                    (factory.EnumWarpAdapter().map_err(|e| format!("WARP unavailable: {e}"))?, "WARP (software renderer)".into())
                } else {
                    let (adapter, desc) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next()
                        .ok_or_else(|| crate::gpu::adapter::no_match(&factory, &sel))?;
                    (adapter, crate::gpu::adapter::name(&desc))
                };
                let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_11_0).map_err(|e| format!("{name}: D3D12 device unavailable ({e})"))?;
                let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_6 };
                dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as _, std::mem::size_of_val(&sm) as u32)
                    .map_err(|e| format!("{name}: cannot query SM 6.6 ({e})"))?;
                if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
                    return Err(crate::gpu::agility::format_no_sm66_message(&name, crate::gpu::agility::state()));
                }
                Ok((dev, name))
            };
            let (dev, name) = match setup() {
                Ok(selected) => selected,
                Err(reason) => { skip_test(&reason); return; }
            };
            println!("ExecuteIndirect adapter: {name}");
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let cs = D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: cs, ..Default::default() }).unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
            let sig = sig.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = |width| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            // the arguments and the buffer the dispatch writes them through: one memory
            let mut heap: Option<ID3D12Heap> = None;
            let heap_desc = D3D12_HEAP_DESC { SizeInBytes: 1 << 16, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            dev.CreateHeap(&heap_desc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let placed = || {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreatePlacedResource(&heap, 0, &desc(1 << 16), D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (args, written) = (placed(), placed());
            // the group count, read on the CPU
            let cpu_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            let mut groups: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &desc(256), D3D12_RESOURCE_STATE_COMMON, None, &mut groups).unwrap();
            let groups = groups.unwrap();
            let mut count = std::ptr::null_mut();
            groups.Map(0, None, Some(&mut count)).unwrap();
            let count = count as *mut u32;
            let uav_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
            };
            let run = |list: &ID3D12GraphicsCommandList, close: &dyn Fn()| {
                close();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let v = done.GetCompletedValue() + 1;
                queue.Signal(&done, v).unwrap();
                fence::wait(&done, v, &dev, None, 10_000).unwrap();
            };
            let (mut old, mut wrong) = (0u32, Vec::new());
            for k in 0..TRIES {
                // each try writes a count the memory does not hold yet
                let new = if k % 2 == 0 { 4096 } else { 0 };
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let record = |x: u32| -> ID3D12GraphicsCommandList {
                    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
                    list.SetComputeRootSignature(&rs);
                    list.SetComputeRootUnorderedAccessView(1, written.GetGPUVirtualAddress());
                    list.SetComputeRootUnorderedAccessView(2, groups.GetGPUVirtualAddress());
                    list.SetComputeRoot32BitConstants(0, 2, [0u32, x].as_ptr() as _, 0);
                    list.Dispatch(1, 1, 1);
                    list
                };
                let before = record(old);
                run(&before, &|| before.Close().unwrap());
                alloc.Reset().unwrap();
                *count = 0;
                let list = record(new);
                list.ResourceBarrier(&[uav_barrier.clone()]);
                list.SetComputeRoot32BitConstants(0, 2, [1u32, 0].as_ptr() as _, 0);
                // from here as XCL records a title's ExecuteIndirect: on the adopted list
                let raw = list.as_raw();
                real::adopt(raw, real::Kind::CommandList);
                list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                fresh_indirect_arguments(raw, || ei(raw, sig.as_raw() as usize, 1, args.as_raw() as usize, 0, 0, 0));
                let close: Close = std::mem::transmute(real::orig_fn(raw, 9));
                run(&list, &|| close(raw).unwrap());
                let ran = std::ptr::read_volatile(count);
                if ran != new {
                    wrong.push((k, ran));
                }
                old = new;
            }
            groups.Unmap(0, None);
            assert!(wrong.is_empty(), "{} of {TRIES} ExecuteIndirects ran another group count than the one written before them (try, groups): {wrong:?}", wrong.len());
        }
    }

    /// A draw does not wait for a graphics pipeline still being built: its build goes to the
    /// workers and the draw gets the stand-in at once; once built, the real one. A compute
    /// pipeline is still waited for.
    #[test]
    fn a_draw_does_not_wait_for_a_graphics_pipeline_still_being_built() {
        let (graphics, compute) = (0x7E57_6101usize, 0x7E57_6102usize);
        pso_async::submit_lazy(graphics, false, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(1500));
            0x1301
        }), 0);
        pso_async::submit_lazy(compute, true, Box::new(|| 0x1302), 0);
        let t = std::time::Instant::now();
        assert_eq!(pso_for_draw(graphics), graphics, "the stand-in, not the real one");
        assert!(t.elapsed() < std::time::Duration::from_millis(500), "a draw waited {:?}", t.elapsed());
        // the draw sent it to the workers: it is built with no further draw
        let start = std::time::Instant::now();
        while pso_async::is_pending(graphics) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(pso_for_draw(graphics), 0x1301, "built by a worker");
        assert_eq!(pso_for_draw(compute), 0x1302, "a compute pipeline is waited for");
        pso_async::released(graphics);
        pso_async::released(compute);
    }

    /// Minimal RTS0 v2 root signature blob with one 4-constant parameter.
    fn test_rts0_blob() -> Vec<u8> {
        let words: [u32; 12] = [
            2, 1, 24, 0, 0, 0, // version=2, num_params=1, params_offset=24
            1, 0, 36,          // param 0: type=Constants(1), visibility=0, payload_offset=36
            0, 0, 4,           // shader_reg=0, reg_space=0, num_32bit_values=4
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn reset_and_clear_state_track_initial_pso_and_clear_root_args() {
        let list = 0x7100_0001usize as *mut c_void;
        let rs_compute = 0x7100_0010usize;
        let rs_gfx = 0x7100_0020usize;
        let alloc1 = 0x7100_0100usize;
        let alloc2 = 0x7100_0200usize;
        let pso_init = 0x7100_1000usize;
        let pso_mid = 0x7100_2000usize;
        let pso_cleared = 0x7100_3000usize;
        let pso_reset = 0x7100_4000usize;

        let blob = test_rts0_blob();
        console_list::note_root_signature(rs_compute, &blob);
        console_list::note_root_signature(rs_gfx, &blob);

        // Initial CreateCommandList / reset seeds PSO and allocator with clean root state.
        reset(list, alloc1, pso_init);
        with(list, |s| {
            assert_eq!(s.pso, pso_init);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Populate compute + graphics root signatures, constants, and a different PSO.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
        }
        with(list, |s| {
            s.pso = pso_mid;
            assert!(s.layout.is_some() && !s.args.is_empty());
            assert!(s.layout_gfx.is_some() && !s.args_gfx.is_empty());
        });

        // ClearState(pso_cleared) updates PSO and clears root signatures/args while keeping allocator.
        clear_state(list, pso_cleared);
        with(list, |s| {
            assert_eq!(s.pso, pso_cleared);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // ClearState(null) clears PSO to 0.
        clear_state(list, 0);
        assert_eq!(tracked_pso(list), 0);

        // Re-populate root args, then Reset(alloc2, pso_reset) switches allocator/PSO and clears root args.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0x1234_5678, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0x8765_4321, 0);
        }
        reset(list, alloc2, pso_reset);
        with(list, |s| {
            assert_eq!(s.pso, pso_reset);
            assert_eq!(s.allocator, alloc2);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Reset with null initial PSO clears stale PSO back to 0.
        reset(list, alloc2, 0);
        assert_eq!(tracked_pso(list), 0);

        forget(list);
    }

    #[repr(C)]
    struct FakeAdoptedList {
        vt: *const usize,
        refs: AtomicUsize,
        dispatch_calls: AtomicUsize,
        set_compute_rs_calls: AtomicUsize,
        set_compute_uav_calls: AtomicUsize,
        set_pso_calls: AtomicUsize,
        execute_indirect_calls: AtomicUsize,
    }

    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    unsafe extern "system" fn fake_dispatch(this: *mut c_void, _x: u32, _y: u32, _z: u32) {
        (*(this as *const FakeAdoptedList)).dispatch_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_pso(this: *mut c_void, _pso: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_pso_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_rs(this: *mut c_void, _rs: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_compute_rs_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_uav(this: *mut c_void, _slot: u32, _va: u64) {
        (*(this as *const FakeAdoptedList)).set_compute_uav_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_execute_indirect(
        this: *mut c_void,
        _sig: usize,
        _max: u32,
        _args: usize,
        _args_off: u64,
        _count_buf: usize,
        _count_off: u64,
    ) {
        (*(this as *const FakeAdoptedList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// `DispatchX` with thread dimensions (flag 0x20, as the BVH builder's rehydration passes
    /// use it): the console launches whole groups of the pipeline's `numthreads` and leaves
    /// the lanes past the count out; without the flag the counts are groups. With the flag
    /// and a group shape not known, nothing is launched (the counts as groups was the bug).
    #[test]
    fn dispatch_x_with_thread_dimensions_launches_whole_groups() {
        let pso = 0x7300_0001usize;
        real::register_threads_for_test(pso, [64, 1, 1]);
        // 640 threads in groups of 64: 10 groups, the last one whole
        assert_eq!(dispatch_groups(pso, [640, 1, 1], 0x20), Some(([10, 1, 1], [0, 0, 0])));
        // 100 threads: 2 groups, and the lanes past 100 to leave out
        assert_eq!(dispatch_groups(pso, [100, 1, 1], 0x20), Some(([2, 1, 1], [100, 0, 0])));
        // the wave32 bit (0x8000) changes nothing here
        assert_eq!(dispatch_groups(pso, [100, 1, 1], 0x8020), Some(([2, 1, 1], [100, 0, 0])));
        // a 2-D group shape
        let pso2 = 0x7300_0002usize;
        real::register_threads_for_test(pso2, [8, 8, 1]);
        assert_eq!(dispatch_groups(pso2, [20, 16, 3], 0x20), Some(([3, 2, 3], [20, 0, 0])));
        // group counts (Dispatch, DispatchX without the flag) are launched as they are
        assert_eq!(dispatch_groups(pso, [7, 2, 1], 0), Some(([7, 2, 1], [0, 0, 0])));
        assert_eq!(dispatch_groups(pso, [7, 2, 1], 0x8000), Some(([7, 2, 1], [0, 0, 0])));
        // no threads: no groups
        assert_eq!(dispatch_groups(pso, [0, 1, 1], 0x20), Some(([0, 1, 1], [0, 0, 0])));
        // thread dimensions on a pipeline whose group shape is not known: not launched
        assert_eq!(dispatch_groups(0x7300_00FF, [100, 1, 1], 0x20), None);
        assert_eq!(dispatch_groups(0x7300_00FF, [100, 1, 1], 0), Some(([100, 1, 1], [0, 0, 0])));
    }

    /// lab: without `XCL_DISPATCHX_THREADS=1` a DispatchX's counts are launched as groups, as
    /// before; with it, thread dimensions reach `dispatch_groups`.
    #[test]
    fn dispatch_x_thread_dimensions_follow_the_lab_switch() {
        let pso = 0x7300_0003usize;
        real::register_threads_for_test(pso, [64, 1, 1]);
        assert_eq!(lab_dispatch_flags(0x8020, false), 0x8000);
        assert_eq!(dispatch_groups(pso, [100, 1, 1], lab_dispatch_flags(0x20, false)), Some(([100, 1, 1], [0, 0, 0])));
        assert_eq!(lab_dispatch_flags(0x8020, true), 0x8020);
        assert_eq!(dispatch_groups(pso, [100, 1, 1], lab_dispatch_flags(0x20, true)), Some(([2, 1, 1], [100, 0, 0])));
    }

    #[test]
    fn async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        // Extended vtable layout expected by `real::orig_fn`: entry -1 holds `orig_vt` pointer,
        // and entries 0..64 hold the COM method pointers used via `ID3D12GraphicsCommandList`.
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // 1. Background compute build SUCCEEDS: real PSO R is registered in TRANSLATED_COMPUTE,
        // stand-in S is NOT. ExecuteIndirect resolves S -> R and treats it as translated.
        let standin_ok = 0x7200_0001usize;
        let real_ok = 0x7200_0002usize;
        let sig_root_changing = 0x7200_00F0usize;
        note_root_changing_signature(sig_root_changing);

        pso_async::submit(
            standin_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_ok), real_ok);
        assert!(!real::is_translated_compute(standin_ok));
        assert!(real::is_translated_compute(real_ok));

        reset(list_ptr, 0x7200_0100, standin_ok);
        unsafe {
            // With `sig_root_changing`, a translated compute PSO is caught by the root-changing
            // guard instead of falling through to `pc` (which happened when `R` wasn't registered).
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "ExecuteIndirect must recognize resolved async compute PSO R as translated"
        );

        // 2. Background compute build FAILS (returns 0): stand-in S_fail is NOT in
        // TRANSLATED_COMPUTE. Dispatch must skip cleanly without binding XCL's ABI RS/UAVs
        // onto S_fail, and ExecuteIndirect must not enter bind_abi either.
        let standin_fail = 0x7200_0003usize;
        pso_async::submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(pso_async::resolve_wait(standin_fail), standin_fail);
        assert!(!real::is_translated_compute(standin_fail));

        reset(list_ptr, 0x7200_0100, standin_fail);
        let before = counts_snapshot();
        unsafe {
            dispatch(list_ptr, 1, 1, 1);
        }
        let after = counts_snapshot();
        assert_eq!(after[DISPATCHES], before[DISPATCHES]);
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);

        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "failed async compute stand-in must not enter translated ExecuteIndirect path"
        );

        // 3. The same failed stand-in (made from a console shader) with a Dispatch signature:
        // nothing to run, as for Dispatch – the PC call would pair the stand-in with the list's
        // compute root signature (XCL's ABI one after a translated dispatch: debug layer 953)
        let sig_dispatch = 0x7200_00F8usize;
        real::register_console_compute_for_test(standin_fail);
        note_signature_dispatches(sig_dispatch, true);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1, "a dispatch of an untranslated console compute pipeline must not reach the PC ExecuteIndirect");
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 4. A native PC compute pipeline (not made from a console shader) with the same Dispatch
        // signature still goes to the PC call, as on main
        let native = 0x7200_0004usize;
        reset(list_ptr, 0x7200_0100, native);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 2, "a native PC compute pipeline must still reach the PC ExecuteIndirect");

        // a signature made later at the same address without a Dispatch argument goes to the PC
        reset(list_ptr, 0x7200_0100, standin_fail);
        note_signature_dispatches(sig_dispatch, false);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 3);

        forget(list_ptr);
        let _ = pso_async::released(standin_ok);
        let _ = pso_async::released(standin_fail);
        real::forget_pso_for_test(real_ok);
        real::forget_pso_for_test(standin_fail);
        assert!(!real::is_console_compute(standin_fail), "a released pipeline's address does not stay a console one");
    }

    #[test]
    fn execute_indirect_skips_failed_shared_console_compute_standin() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // With shared stand-ins (`standin::shell`), `dev_create_compute_pso` registers the
        // title-facing shell in `CONSOLE_COMPUTE`, while `pso_async::resolve_wait` returns the
        // shared driver pipeline when the background build fails.
        let shell_fail = 0x7210_0001usize;
        let shell_ok = 0x7210_0002usize;
        let shared_driver = 0x7210_0003usize;
        let real_ok = 0x7210_0004usize;
        let sig_dispatch = 0x7210_00F8usize;
        let sig_root_changing = 0x7210_00F0usize;
        note_signature_dispatches(sig_dispatch, true);
        note_root_changing_signature(sig_root_changing);

        pso_async::submit_lazy(shell_fail, true, Box::new(|| 0), 0);
        pso_async::shared(shell_fail, shared_driver);
        real::register_console_compute_for_test(shell_fail);

        pso_async::submit_lazy(
            shell_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
            0,
        );
        pso_async::shared(shell_ok, shared_driver);
        real::register_console_compute_for_test(shell_ok);

        assert_eq!(pso_async::resolve_wait(shell_fail), shared_driver);
        assert!(!pso_async::is_pending(shell_fail));
        assert!(real::is_console_compute(shell_fail));
        assert!(!real::is_console_compute(shared_driver));

        // 1. Failed shared console compute stand-in with a Dispatch signature must be skipped,
        // never forwarded to PC ExecuteIndirect with the shared driver pipeline.
        reset(list_ptr, 0x7210_0100, shell_fail);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "failed shared console compute stand-in must not reach PC ExecuteIndirect"
        );
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 2. Another shell sharing the same driver pipeline whose build succeeds still takes the
        // translated path (caught here by the root-changing guard rather than falling through to PC).
        reset(list_ptr, 0x7210_0100, shell_ok);
        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0);

        // 3. Releasing `shell_fail` forgets its `CONSOLE_COMPUTE` entry so a native PC compute
        // pipeline later allocated at the same address still reaches PC ExecuteIndirect.
        let _ = pso_async::released(shell_fail);
        real::forget_pso_for_test(shell_fail);
        assert!(!real::is_console_compute(shell_fail));
        reset(list_ptr, 0x7210_0100, shell_fail);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "native PC pipeline at a released shell's address must still reach PC ExecuteIndirect"
        );

        forget(list_ptr);
        note_signature_dispatches(sig_dispatch, false);
        let _ = pso_async::released(shell_ok);
        real::forget_pso_for_test(shell_ok);
        real::forget_pso_for_test(real_ok);
    }

    #[test]
    fn two_pass_pso_skips_direct_draw_and_unmatched_execute_indirect() {
        #[repr(C)]
        struct FakeTwoPassList {
            vt: *const usize,
            refs: AtomicUsize,
            set_graphics_rs_calls: AtomicUsize,
            execute_indirect_calls: AtomicUsize,
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            (*(this as *const FakeTwoPassList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            (*(this as *const FakeTwoPassList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }
        unsafe extern "system" fn set_graphics_rs(this: *mut c_void, _rs: *mut c_void) {
            (*(this as *const FakeTwoPassList)).set_graphics_rs_calls.fetch_add(1, Ordering::Relaxed);
        }
        unsafe extern "system" fn exec_indirect(
            this: *mut c_void,
            _sig: usize,
            _max: u32,
            _args: usize,
            _args_off: u64,
            _count_buf: usize,
            _count_off: u64,
        ) {
            (*(this as *const FakeTwoPassList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
        }

        let mut orig_vt = [0usize; 64];
        orig_vt[1] = add_ref as usize;
        orig_vt[2] = release as usize;
        orig_vt[30] = set_graphics_rs as usize;
        orig_vt[59] = exec_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeTwoPassList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            set_graphics_rs_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeTwoPassList as *mut c_void;

        // Register a finished two-pass GS/MS PSO behind an async stand-in (registered in TWO_PASS,
        // not TRANSLATED_GRAPHICS).
        let standin_tp = 0x7250_0001usize;
        let real_tp = 0x7250_0002usize;
        pso_async::submit(
            standin_tp,
            false,
            Box::new(move || {
                real::register_two_pass_for_test(real_tp, "GS");
                real_tp
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_tp), real_tp);
        assert!(real::two_pass_of(real_tp).is_some());
        assert!(!real::is_translated_graphics(real_tp));

        reset(list_ptr, 0x7250_0100, standin_tp);
        unsafe {
            // Seed a non-null title graphics RS so a fallthrough would visibly call SetGraphicsRootSignature.
            set_graphics_root_signature(list_ptr, list_ptr);
        }

        // 1. Direct draw (`graphics_bound`) must return None without binding the title's RS.
        let before = counts_snapshot();
        let bound = unsafe { graphics_bound(list_ptr) };
        let after = counts_snapshot();
        assert!(bound.is_none(), "two-pass MS/GS PSO must not fall through to PC DrawInstanced");
        assert!(after[DRAWS_SKIPPED] > before[DRAWS_SKIPPED]);
        assert_eq!(after[DRAWS_PC], before[DRAWS_PC]);
        assert_eq!(fake.set_graphics_rs_calls.load(Ordering::Relaxed), 0);

        // 2. Unmatched `ExecuteIndirect` must skip without binding the title's RS or calling PC ExecuteIndirect.
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, 0x7250_00F0, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert!(after[DRAWS_SKIPPED] > before[DRAWS_SKIPPED]);
        assert_eq!(fake.set_graphics_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 0);

        forget(list_ptr);
        let _ = pso_async::released(standin_tp);
        real::forget_pso_for_test(real_tp);
        assert!(real::two_pass_of(real_tp).is_none());
    }

    #[test]
    fn create_list_reset_and_clear_state_track_initial_pso_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        if console_list::enabled() {
            return;
        }
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            real::adopt(dev.as_raw(), real::Kind::Device);

            // Build two real PC compute PSOs from the inbox dummy CS bytecode.
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("CreateRootSignature");
            let cs = real::dummy_cs_bytecode();
            let make_pso = || -> ID3D12PipelineState {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                pso
            };
            let pso1 = make_pso();
            let pso2 = make_pso();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");

            // Stand-in synthetic key backed by real `pso1` in pso_async: CreateCommandList must
            // resolve it via resolve_for_bind for PC D3D12 while tracking `standin` in `bind`.
            let standin = 0x7300_0001usize;
            let pso1_raw = pso1.as_raw() as usize;
            pso_async::submit(standin, true, Box::new(move || pso1_raw));
            assert_eq!(pso_async::resolve_wait(standin), pso1_raw);

            let mut raw_list: *mut c_void = std::ptr::null_mut();
            let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
            let hr = device_x::create_command_list_pc(
                dev.as_raw(),
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32,
                alloc.as_raw() as usize,
                standin,
                &iid,
                &mut raw_list,
            );
            assert!(hr >= 0 && !raw_list.is_null(), "create_command_list_pc failed: {hr:#x}");
            let backing = title_list::backing(raw_list).expect("wrapped in title_list");
            assert_eq!(tracked_pso(backing), standin, "CreateCommandList must seed bind::State.pso");

            // ClearState(pso2) updates bind::State.pso to pso2 and clears root args.
            let rs_key = 0x7300_0010usize;
            console_list::note_root_signature(rs_key, &test_rts0_blob());
            set_compute_root_signature(backing, rs_key as *mut c_void);
            set_compute_root_constant(backing, 0, 0xABCD, 0);
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
            cl.ClearState(&pso2);
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "ClearState must update bind::State.pso");
            with(backing, |s| assert!(s.layout.is_none() && s.args.is_empty()));

            // ClearState(None) clears bind::State.pso to 0.
            cl.ClearState(None);
            assert_eq!(tracked_pso(backing), 0, "ClearState(None) must clear bind::State.pso");

            // Close + reset_x(alloc, pso2) updates bind::State.pso to pso2 and clears root args.
            set_graphics_root_signature(backing, rs_key as *mut c_void);
            set_graphics_root_constant(backing, 0, 0x1111, 0);
            cl.Close().expect("Close");
            let hr = list_x::reset_x(backing, alloc.as_raw() as usize, pso2.as_raw() as usize);
            assert!(hr >= 0, "reset_x failed: {hr:#x}");
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "Reset must update bind::State.pso");
            with(backing, |s| assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty()));

            // Release the wrapped title list cleanly (calls bind::forget + list_x::on_release).
            let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(raw_list as *const *const usize)).add(2));
            assert_eq!(release_fn(raw_list), 0);
            let _ = pso_async::released(standin);
        }
    }

    #[test]
    fn dispatch_mesh_signature_tracking_returns_companion_and_layout() {
        let sig = 0x7400_0001usize;
        let companion = 0x7400_0002usize;
        assert_eq!(dispatch_mesh_sig_of(sig), None);
        note_dispatch_mesh_signature(sig, companion, 28, 16);
        assert_eq!(
            dispatch_mesh_sig_of(sig),
            Some(DispatchMeshSig { companion_dispatch: companion, byte_stride: 28, mesh_offset: 16 })
        );
        DISPATCH_MESH_SIGS.lock().unwrap().as_mut().unwrap().remove(&sig);
        assert_eq!(dispatch_mesh_sig_of(sig), None);
    }
}
