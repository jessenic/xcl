//! lab: title memory outside the unified-memory windows, readable by translated shaders.
//!
//! A retail title keeps its streamed geometry (meshlet, index and vertex buffers) at
//! addresses of its 24 GB "GPU view" (`kernelx::mirrors`: 0xD_FFFF0000.., the GPU-only view
//! of the CPU view at 0x7_FFFF0000..). Translated shaders reach memory through the windows
//! (`xcl_w` = address >> 32 - the unified block's first window): only windows 0..3 existed, so
//! every load there read 0 – a mesh shader then saw meshlets of one vertex and one primitive,
//! and the world was not drawn. Here windows 4..15 (addresses 0x8_0000_0000..0x14_0000_0000)
//! are reserved buffers whose 64 KiB tiles are mapped and filled on demand: a draw names the
//! ranges it can read (`ensure`: the V#s of its descriptor table, its root addresses) and the
//! tiles under them are copied from where the title's bytes are (`source`) on a queue of XCL's
//! own, finished before `ensure` returns – so before the list being recorded can run.
//! Tiles are checked again (a sampled signature) at most once a second (`XCL_VIEW_FRESH_MS`).
//!
//! lab `XCL_VIEW_FULL_RANGE=1` (off by default): a root address has no size and was taken as 1 MiB
//! (`XCL_VIEW_ROOT_KB`), a V# as its records × stride up to 16 MiB: past that a tile never mapped
//! read 0, one mapped by another draw its last copy. The VIctory fork kelvaroq found the same 1 MiB
//! bound behind its "exploded triangles" and lets the view cover the rest of the buffer. With the
//! switch a range ends where the title's allocation that holds it ends when XCL knows it
//! (`real::outside_buffer_left`), else at a V#'s own size, else (a root, a V# over 16 MiB) at the
//! end of the run of tiles from its first one that have a source, at most `XCL_VIEW_FULL_MB`
//! (256 MiB; 0: no cap) for the pool's sake; never short of the old bounds unless an allocation
//! ends sooner. Which of those gave the size is counted in `stats`. With the switch a tile found
//! without a source is not checked again while fresh (`Misses`): a run with 64 MiB behind each
//! root checked 100 M such tiles.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

const TILE: usize = 0x1_0000;
const WINDOW: usize = 1 << 32;
/// first window here and one past the last (window k = addresses (base window + k) << 32)
pub const FIRST: usize = 4;
pub const END: usize = 16;
const POOL_TILES: u32 = 1024;
const STAGING_TILES: usize = 128;
/// lab (`XCL_VIEW_NO_WAIT=1`): upload batches in flight at once, each with its own list and
/// `STAGING_TILES` of the staging buffer
const RING: usize = 4;

struct Tile {
    sig: u64,
    checked: std::time::Instant,
    /// (pool heap, tile in it)
    slot: (u32, u32),
}

struct State {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// window FIRST + i
    windows: Vec<ID3D12Resource>,
    /// address of window 0 (the unified block's base)
    base: usize,
    pools: Vec<ID3D12Heap>,
    next: u32,
    /// pool tiles given back by `evict_stale`
    free: Vec<(u32, u32)>,
    evicted_at: Option<std::time::Instant>,
    tiles: HashMap<usize, Tile>,
    staging: ID3D12Resource,
    staging_at: *mut u8,
    cap_tiles: usize,
    /// (va, len, far) → when it was last looked at
    recent: HashMap<(usize, usize, usize), std::time::Instant>,
    /// lab (`XCL_VIEW_FULL_RANGE=1`): tiles found without a source
    misses: Misses,
    /// lab (`no_wait`): the upload batches' lists and staging parts, used in turn
    ring: Vec<RingSlot>,
    ring_next: usize,
}
unsafe impl Send for State {}

/// lab (`no_wait`): a batch's list, reused once the fence passed `value`; it copies from
/// staging part (its index × `STAGING_TILES` tiles).
struct RingSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    value: u64,
}

/// lab: `XCL_VIEW_NO_WAIT=1`: an upload is not waited for on the CPU, with the state's lock held,
/// before `ensure` returns: the title's queue waits for it on the GPU before the submit
/// (`before_execute`), and the staging is a ring whose parts are reused by fence value.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_VIEW_NO_WAIT").is_ok_and(|v| v.trim() == "1"))
}

/// lab (`no_wait`): the fence the uploads signal, and the last value signalled, for the title's
/// queues to wait for without the state's lock (`before_execute`)
static UPLOAD_FENCE: Mutex<Option<ID3D12Fence>> = Mutex::new(None);
static UPLOAD_LAST: AtomicU64 = AtomicU64::new(0);
/// lab: queue waits added for uploads, and batches that waited for a ring part on the CPU
static QUEUE_WAITS: AtomicU64 = AtomicU64::new(0);
static RING_WAITS: AtomicU64 = AtomicU64::new(0);

/// The value a title queue has to wait for before its submit (`before_execute`): the last
/// upload's, when the fence has not passed it yet (None: nothing to wait for).
fn upload_wait(last: u64, completed: u64) -> Option<u64> {
    (last != 0 && completed < last).then_some(last)
}

/// The value a ring part has to be waited for at before its reuse (None: free now).
fn ring_wait(slot_value: u64, completed: u64) -> Option<u64> {
    (slot_value != 0 && completed < slot_value).then_some(slot_value)
}

/// lab (`XCL_VIEW_NO_WAIT=1`): before the title's ExecuteCommandLists on `queue`: it waits on the
/// GPU for the view memory uploads recorded so far (its lists may read them). Without the lock.
pub unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    if !no_wait() {
        return;
    }
    let last = UPLOAD_LAST.load(Ordering::Acquire);
    if last == 0 {
        return;
    }
    let Some(fence) = UPLOAD_FENCE.lock().unwrap().clone() else { return };
    if let Some(v) = upload_wait(last, super::real::fence_completed(&fence)) {
        if queue.Wait(&fence, v).is_ok() {
            QUEUE_WAITS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static UPLOADED: AtomicU64 = AtomicU64::new(0);
static CHECKED: AtomicU64 = AtomicU64::new(0);
static NO_SOURCE: AtomicU64 = AtomicU64::new(0);
static FULL: AtomicU64 = AtomicU64::new(0);
/// tiles and pools of the state, for `stats` (which must not wait for an `ensure` in progress)
static TILES: AtomicU64 = AtomicU64::new(0);
static POOLS: AtomicU64 = AtomicU64::new(0);
static EVICTED: AtomicU64 = AtomicU64::new(0);
/// lab counters of the ranges (`stats`): root addresses named by draws, looked at (not seen in
/// the last `fresh`), of those with a sourced tile right past `XCL_VIEW_ROOT_KB` (cut short
/// without the switch); V#s over `VSHARP_MOST`; bytes past the old bounds and what gave the size
/// with the switch (an allocation, the V#, sourced tiles, the cap); ranges and tiles skipped as
/// fresh; rounds of `evict_stale`
static ROOTS_NAMED: AtomicU64 = AtomicU64::new(0);
static ROOTS_LOOKED: AtomicU64 = AtomicU64::new(0);
static ROOTS_CUT: AtomicU64 = AtomicU64::new(0);
static VSHARP_CUT: AtomicU64 = AtomicU64::new(0);
static ADDED: AtomicU64 = AtomicU64::new(0);
static SIZE_BY: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static RECENT: AtomicU64 = AtomicU64::new(0);
static FRESH_TILES: AtomicU64 = AtomicU64::new(0);
static ROUNDS: AtomicU64 = AtomicU64::new(0);
/// lab: tiles without a source found past the old bounds (where full ranges end), tiles without
/// a source not checked again (`Misses`), the longest range looked at
static PAST_MISS: AtomicU64 = AtomicU64::new(0);
static MEMO: AtomicU64 = AtomicU64::new(0);
static LONGEST: AtomicU64 = AtomicU64::new(0);

/// lab (with the switch): tiles found without a source → when, not checked again while fresh
#[derive(Default)]
struct Misses(HashMap<usize, std::time::Instant>);

impl Misses {
    fn known(&self, tile: usize, now: std::time::Instant, fresh: std::time::Duration) -> bool {
        self.0.get(&tile).is_some_and(|t| now.duration_since(*t) < fresh)
    }

    fn note(&mut self, tile: usize, now: std::time::Instant) {
        if self.0.len() > 200_000 {
            self.0.clear();
        }
        self.0.insert(tile, now);
    }
}

/// lab `XCL_VIEW_FULL_RANGE` when it is not set: turning the full ranges on by default is this line
const FULL_RANGE_BY_DEFAULT: bool = false;
/// a V#'s records × stride is taken up to this without the switch
const VSHARP_MOST: usize = 16 << 20;

/// The range switches, read once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cfg {
    /// bytes behind a root address (`XCL_VIEW_ROOT_KB`, 1024)
    root: usize,
    /// `XCL_VIEW_FULL_RANGE=1`
    full: bool,
    /// most bytes of a range with the switch (`XCL_VIEW_FULL_MB`, 256; 0: no cap)
    cap: usize,
    /// a range or a tile looked at less than this ago is not looked at again (`XCL_VIEW_FRESH_MS`, 1000)
    fresh: std::time::Duration,
}

fn cfg_from(var: impl Fn(&str) -> Option<String>) -> Cfg {
    let num = |k: &str| var(k).and_then(|v| v.trim().parse::<usize>().ok());
    Cfg {
        root: num("XCL_VIEW_ROOT_KB").unwrap_or(1024) << 10,
        full: var("XCL_VIEW_FULL_RANGE").map_or(FULL_RANGE_BY_DEFAULT, |v| v.trim() == "1"),
        cap: match num("XCL_VIEW_FULL_MB").unwrap_or(256) {
            0 => usize::MAX,
            mb => mb.saturating_mul(1 << 20),
        },
        fresh: std::time::Duration::from_millis(num("XCL_VIEW_FRESH_MS").unwrap_or(1000) as u64),
    }
}

fn cfg() -> &'static Cfg {
    static C: std::sync::OnceLock<Cfg> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let c = cfg_from(|k| std::env::var(k).ok());
        if c.full {
            crate::xlog_info!("lab XCL_VIEW_FULL_RANGE=1: a root address and a V# are loaded to the end of the title's allocation when known, else to the V#'s size, else to the end of their sourced tiles, at most {} MB (XCL_VIEW_FULL_MB); tiles fresh for {} ms", c.cap >> 20, c.fresh.as_millis());
        }
        c
    })
}

/// What gave a range its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Why {
    /// the end of the title's allocation
    Alloc,
    /// the V#'s records × stride
    Given,
    /// the end of the run of sourced tiles
    Tiles,
    /// `Cfg::cap`
    Cap,
    /// without the switch: the old bounds
    Bound,
}

/// What a draw names: a root address (no size) or a V# (records × stride bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Named {
    Root,
    View(usize),
}

/// `(bytes always covered, bytes covered while the tiles have a source, why)` of a range at an
/// address with `alloc_left` bytes left in its allocation (when known).
fn plan(c: &Cfg, named: Named, alloc_left: Option<usize>) -> (usize, usize, Why) {
    if !c.full {
        let n = match named {
            Named::Root => c.root,
            Named::View(n) => n.clamp(4, VSHARP_MOST),
        };
        return (n, n, Why::Bound);
    }
    if let Some(left) = alloc_left {
        let (want, why) = match named {
            Named::View(n) if n.max(4) < left => (n.max(4), Why::Given),
            _ => (left, Why::Alloc),
        };
        return if want > c.cap { (c.cap, c.cap, Why::Cap) } else { (want, want, why) };
    }
    match named {
        Named::Root => (c.root, c.cap.max(c.root), Why::Cap),
        Named::View(n) if n <= VSHARP_MOST => (n.max(4), n.max(4), Why::Given),
        Named::View(n) if n <= c.cap => (VSHARP_MOST, n, Why::Given),
        Named::View(_) => (VSHARP_MOST, c.cap.max(VSHARP_MOST), Why::Cap),
    }
}

/// The tiles one range covers: `[va, end)` (what was asked: tiles there without a source are
/// skipped), then up to `far` the tiles that follow while every tile from `va`'s on has had one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Reach {
    end: usize,
    far: usize,
    run: bool,
    /// the first tile past `end` without a source
    stop: Option<usize>,
}

impl Reach {
    fn new(va: usize, len: usize, far: usize, hi: usize) -> Reach {
        let end = va.saturating_add(len).min(hi);
        Reach { end, far: va.saturating_add(far).min(hi).max(end), run: true, stop: None }
    }

    /// Whether the tile at `tile` is in the range (before its source is known).
    fn covers(&self, tile: usize) -> bool {
        tile < self.end || (self.run && tile < self.far)
    }

    /// The tile at `tile` has no source: true when it is in what was asked (counted, logged);
    /// past that the range ends there.
    fn missing(&mut self, tile: usize) -> bool {
        self.run = false;
        if tile >= self.end && self.stop.is_none() {
            self.stop = Some(tile);
        }
        tile < self.end
    }

    /// Where the range ended, `next` being the tile after the last one looked at.
    fn reached(&self, next: usize) -> usize {
        match self.stop {
            Some(s) => s.clamp(self.end, self.far),
            None if !self.run => self.end,
            None => next.clamp(self.end, self.far),
        }
    }

    /// What gave the range its size: `planned` when it reached `far`, else its sourced tiles.
    fn why(&self, reached: usize, planned: Why) -> Why {
        if self.far > self.end && reached < self.far {
            Why::Tiles
        } else {
            planned
        }
    }
}

/// One range looked at (atomics only): `cut` – a root whose tile right past `XCL_VIEW_ROOT_KB`
/// has a source; `added` – bytes past the old bound.
fn count(root: bool, cut: bool, full: bool, why: Why, added: usize) {
    if root {
        ROOTS_LOOKED.fetch_add(1, Ordering::Relaxed);
        if cut {
            ROOTS_CUT.fetch_add(1, Ordering::Relaxed);
        }
    }
    if full {
        ADDED.fetch_add(added as u64, Ordering::Relaxed);
        let k = match why {
            Why::Alloc => 0,
            Why::Given => 1,
            Why::Tiles => 2,
            Why::Cap => 3,
            Why::Bound => return,
        };
        SIZE_BY[k].fetch_add(1, Ordering::Relaxed);
    }
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// `XCL_VIEW_MEM=0`: off.
fn enabled() -> bool {
    !std::env::var("XCL_VIEW_MEM").is_ok_and(|v| v == "0")
}

/// The windows' buffers `(resource, bytes)` for descriptors `MEM_HEAP_BASE + FIRST..END` of the
/// shared heap; empty when they could not be made.
pub unsafe fn init(dev: &ID3D12Device, base: usize) -> Vec<(usize, usize)> {
    if !enabled() {
        return Vec::new();
    }
    let made = (|| -> windows::core::Result<State> {
        let mut windows = Vec::new();
        for _ in FIRST..END {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc(WINDOW as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut r)?;
            windows.push(r.ok_or_else(windows::core::Error::empty)?);
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.Close()?;
        let mut staging: Option<ID3D12Resource> = None;
        let parts = if no_wait() { RING } else { 1 };
        let mut ring = Vec::new();
        if no_wait() {
            for _ in 0..RING {
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
                list.Close()?;
                ring.push(RingSlot { alloc, list, value: 0 });
            }
        }
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc((parts * STAGING_TILES * TILE) as u64, D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut staging,
        )?;
        let staging = staging.ok_or_else(windows::core::Error::empty)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        staging.Map(0, None, Some(&mut p))?;
        let cap_mb: usize = std::env::var("XCL_VIEW_POOL_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(3072);
        Ok(State {
            dev: dev.clone(),
            queue,
            fence,
            value: 0,
            alloc,
            list,
            windows,
            base,
            pools: Vec::new(),
            next: POOL_TILES,
            free: Vec::new(),
            evicted_at: None,
            tiles: HashMap::new(),
            staging,
            staging_at: p as *mut u8,
            cap_tiles: cap_mb << 4,
            recent: HashMap::new(),
            misses: Misses::default(),
            ring,
            ring_next: 0,
        })
    })();
    match made {
        Ok(s) => {
            let out = s.windows.iter().map(|w| (w.as_raw() as usize, WINDOW)).collect();
            crate::xlog_info!("D3D12.X: view memory: windows {FIRST}..{END} ({:#x}..{:#x}) are filled on demand, up to {} MB", base + FIRST * WINDOW, base + END * WINDOW, s.cap_tiles >> 4);
            if no_wait() {
                *UPLOAD_FENCE.lock().unwrap() = Some(s.fence.clone());
                crate::xlog_info!("D3D12.X: view memory: uploads are not waited for under the lock; the title's queues wait for them (XCL_VIEW_NO_WAIT=1, {RING} batches in flight)");
            }
            *STATE.lock().unwrap() = Some(s);
            out
        }
        Err(e) => {
            crate::xlog_warn!("D3D12.X: view memory: not available ({e})");
            Vec::new()
        }
    }
}

/// Where the 64 KiB page at `va` has its bytes for the CPU (`None`: nowhere readable).
fn source(va: usize) -> Option<usize> {
    use crate::hle::kernelx;
    let a = match kernelx::mirrors::mapped_source(va) {
        Some(shown) => shown?,
        None => match kernelx::mirrors::source(va) {
            Some(cpu) if !kernelx::physical_page_at(va) => cpu,
            _ => va,
        },
    };
    crate::gpu::shader::readable(a, TILE).then_some(a)
}

/// LOCAL: where the CPU reads the byte at `va` (the page's source plus the offset in it).
pub fn cpu_at(va: usize) -> Option<usize> {
    source(va & !(TILE - 1)).map(|p| p + (va & (TILE - 1)))
}

/// A sampled signature of a page (a word every 256 bytes and its last one), read from a copy
/// made without faulting: the title may decommit the page meanwhile (a run ended in memcpy here).
unsafe fn signature(src: usize) -> u64 {
    thread_local!(static COPY: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0u8; TILE]));
    COPY.with(|c| {
        let mut c = c.borrow_mut();
        c.fill(0);
        super::title_textures::copy_committed_region(src, c.as_mut_ptr(), TILE);
        signature_of(c.as_ptr() as usize)
    })
}

unsafe fn signature_of(a: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut o = 0;
    while o < TILE {
        h = (h ^ ((a + o) as *const u64).read_unaligned()).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(17);
        o += 256;
    }
    h ^ ((a + TILE - 8) as *const u64).read_unaligned()
}

/// The pool is full: tiles no draw has named for `STALE` give their pool tiles back (at most
/// one round a second). A list recorded earlier and not run yet reads 0 or newer bytes there.
unsafe fn evict_stale(s: &mut State, now: std::time::Instant) {
    const STALE: std::time::Duration = std::time::Duration::from_secs(6);
    if s.evicted_at.is_some_and(|t| now.duration_since(t) < std::time::Duration::from_secs(1)) {
        return;
    }
    s.evicted_at = Some(now);
    let old: Vec<usize> = s.tiles.iter().filter(|(_, t)| t.slot.0 != u32::MAX && now.duration_since(t.checked) >= STALE).map(|(&a, _)| a).collect();
    // window -> its tiles that go (one UpdateTileMappings per window)
    let mut by_window: HashMap<usize, Vec<u32>> = HashMap::new();
    for &tile in &old {
        let Some(t) = s.tiles.remove(&tile) else { continue };
        let k = (tile - s.base) / WINDOW - FIRST;
        by_window.entry(k).or_default().push(((tile - s.base) % WINDOW / TILE) as u32);
        s.free.push(t.slot);
    }
    for (k, mut xs) in by_window {
        xs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::new();
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::new();
        let mut counts: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < xs.len() {
            let mut run = 1u32;
            while i + (run as usize) < xs.len() && xs[i + run as usize] == xs[i] + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: xs[i], ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            counts.push(run);
            i += run as usize;
        }
        let flags = vec![D3D12_TILE_RANGE_FLAG_NULL; coords.len()];
        s.queue.UpdateTileMappings(&s.windows[k], coords.len() as u32, Some(coords.as_ptr()), Some(regions.as_ptr()), None, flags.len() as u32, Some(flags.as_ptr()), None, Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
    }
    s.recent.clear();
    EVICTED.fetch_add(old.len() as u64, Ordering::Relaxed);
    let n = ROUNDS.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: view memory: the pool is full – {} tiles not named for {} s given back ({} kept; round {n})", old.len(), STALE.as_secs(), s.tiles.len());
    }
}

/// Shaders may read `[va, va + len)`: the tiles there hold the title's bytes when this returns.
pub unsafe fn ensure(va: usize, len: usize) {
    ensure_range(va, len, len, len, false, Why::Bound);
}

/// `ensure` of `[va, va + len)` and, while their tiles have a source, up to `va + far` (`Reach`);
/// `old`: the bytes the range had without the switch, `root`: a root address (`count`).
unsafe fn ensure_range(va: usize, len: usize, far: usize, old: usize, root: bool, planned: Why) {
    let c = cfg();
    let mut g = STATE.lock().unwrap();
    super::real::removed_probe("view_mem::ensure: start");
    let Some(s) = g.as_mut() else { return };
    let (lo, hi) = (s.base + FIRST * WINDOW, s.base + END * WINDOW);
    if va < lo || va >= hi || len == 0 {
        return;
    }
    let now = std::time::Instant::now();
    let fresh = c.fresh;
    if s.recent.get(&(va, len, far)).is_some_and(|t| now.duration_since(*t) < fresh) {
        RECENT.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if s.recent.len() > 200_000 {
        s.recent.clear();
    }
    s.recent.insert((va, len, far), now);
    let mut reach = Reach::new(va, len, far, hi);
    let mut t = va & !(TILE - 1);
    let (mut fresh_tiles, mut memo) = (0u64, 0u64);
    // (tile, source, new)
    let mut todo: Vec<(usize, usize, bool)> = Vec::new();
    while reach.covers(t) {
        let tile = t;
        t += TILE;
        let known = s.tiles.get(&tile);
        if known.is_some_and(|k| now.duration_since(k.checked) < fresh) {
            fresh_tiles += 1;
            continue;
        }
        if c.full && s.misses.known(tile, now, fresh) {
            reach.missing(tile);
            memo += 1;
            continue;
        }
        let Some(src) = source(tile) else {
            if c.full {
                s.misses.note(tile, now);
            }
            // the first few say why (asked range, what the page maps show): a range past a
            // buffer's end is expected here, a mapped page that cannot be read is not (lab: past
            // what was asked, a tile without a source ends a full range: counted apart)
            let inside = reach.missing(tile);
            if !inside {
                PAST_MISS.fetch_add(1, Ordering::Relaxed);
            }
            if inside && NO_SOURCE.fetch_add(1, Ordering::Relaxed) < 12 {
                use crate::hle::kernelx;
                crate::xlog_info!(
                    "D3D12.X: view memory: no source for tile {tile:#x} of {va:#x}+{len:#x}: page mapping {:x?}, CPU view {:x?}, physical page here {}",
                    kernelx::mirrors::mapped_source(tile),
                    kernelx::mirrors::source(tile),
                    kernelx::physical_page_at(tile)
                );
            }
            continue;
        };
        CHECKED.fetch_add(1, Ordering::Relaxed);
        let sig = signature(src);
        match s.tiles.get_mut(&tile) {
            Some(k) if k.sig == sig => k.checked = now,
            Some(k) => {
                k.sig = sig;
                k.checked = now;
                todo.push((tile, src, false));
            }
            None => {
                if s.tiles.len() >= s.cap_tiles {
                    evict_stale(s, now);
                }
                if s.tiles.len() >= s.cap_tiles {
                    if FULL.fetch_add(1, Ordering::Relaxed) == 0 {
                        crate::xlog_warn!("D3D12.X: view memory: the pool is full ({} MB, XCL_VIEW_POOL_MB) – further ranges read 0", s.cap_tiles >> 4);
                    }
                    continue;
                }
                s.tiles.insert(tile, Tile { sig, checked: now, slot: (u32::MAX, 0) });
                todo.push((tile, src, true));
            }
        }
    }
    FRESH_TILES.fetch_add(fresh_tiles, Ordering::Relaxed);
    if memo != 0 {
        MEMO.fetch_add(memo, Ordering::Relaxed);
    }
    // lab counters: what a root without the switch left out, what the switch added
    let reached = reach.reached(t);
    let cut = root && {
        let past = va.saturating_add(c.root).saturating_add(TILE - 1) & !(TILE - 1);
        past < hi && (s.tiles.get(&past).is_some_and(|k| now.duration_since(k.checked) < fresh) || (!(c.full && s.misses.known(past, now, fresh)) && source(past).is_some()))
    };
    LONGEST.fetch_max((reached - va) as u64, Ordering::Relaxed);
    count(root, cut, c.full, reach.why(reached, planned), reached.saturating_sub(va.saturating_add(old).min(hi)));
    TILES.store(s.tiles.len() as u64, Ordering::Relaxed);
    if todo.is_empty() {
        return;
    }
    // new tiles: a pool tile each, mapped under the window - one UpdateTileMappings per window
    // and pool heap for all the tiles of this call (lab: it was one call per tile, 100-400 a
    // second in the world; `XCL_VIEW_MAP_EACH=1`: as before)
    static EACH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let each = *EACH.get_or_init(|| std::env::var("XCL_VIEW_MAP_EACH").is_ok_and(|v| v.trim() == "1"));
    // (window, pool heap) -> (tile of the window, tile of the heap)
    let mut maps: HashMap<(usize, u32), Vec<(u32, u32)>> = HashMap::new();
    for &(tile, _, new) in &todo {
        if !new {
            continue;
        }
        if let Some(slot) = s.free.pop() {
            let k = (tile - s.base) / WINDOW - FIRST;
            let coord = D3D12_TILED_RESOURCE_COORDINATE { X: ((tile - s.base) % WINDOW / TILE) as u32, ..Default::default() };
            let region = D3D12_TILE_REGION_SIZE { NumTiles: 1, UseBox: false.into(), ..Default::default() };
            let (flag, offset, count) = (D3D12_TILE_RANGE_FLAG_NONE, slot.1, 1u32);
            if each {
                s.queue.UpdateTileMappings(&s.windows[k], 1, Some(&coord), Some(&region), &s.pools[slot.0 as usize], 1, Some(&flag), Some(&offset), Some(&count), D3D12_TILE_MAPPING_FLAG_NONE);
            } else {
                maps.entry((k, slot.0)).or_default().push((coord.X, slot.1));
            }
            if let Some(t) = s.tiles.get_mut(&tile) {
                t.slot = slot;
            }
            continue;
        }
        if s.next >= POOL_TILES {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: POOL_TILES as u64 * TILE as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            if s.dev.CreateHeap(&desc, &mut heap).is_err() || heap.is_none() {
                crate::xlog_warn!("D3D12.X: view memory: no video memory for another pool");
                s.cap_tiles = 0;
                s.tiles.remove(&tile);
                continue;
            }
            s.pools.push(heap.unwrap());
            POOLS.store(s.pools.len() as u64, Ordering::Relaxed);
            s.next = 0;
        }
        let k = (tile - s.base) / WINDOW - FIRST;
        let coord = D3D12_TILED_RESOURCE_COORDINATE { X: ((tile - s.base) % WINDOW / TILE) as u32, ..Default::default() };
        let region = D3D12_TILE_REGION_SIZE { NumTiles: 1, UseBox: false.into(), ..Default::default() };
        let (flag, offset, count) = (D3D12_TILE_RANGE_FLAG_NONE, s.next, 1u32);
        if each {
            s.queue.UpdateTileMappings(&s.windows[k], 1, Some(&coord), Some(&region), s.pools.last().unwrap(), 1, Some(&flag), Some(&offset), Some(&count), D3D12_TILE_MAPPING_FLAG_NONE);
        } else {
            maps.entry((k, s.pools.len() as u32 - 1)).or_default().push((coord.X, s.next));
        }
        if let Some(t) = s.tiles.get_mut(&tile) {
            t.slot = (s.pools.len() as u32 - 1, s.next);
        }
        s.next += 1;
    }
    for ((k, heap), mut pairs) in maps {
        // runs of tiles that follow each other in the window and in the heap are one region
        pairs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::new();
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::new();
        let mut offsets: Vec<u32> = Vec::new();
        let mut counts: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < pairs.len() {
            let (x0, t0) = pairs[i];
            let mut run = 1u32;
            while i + (run as usize) < pairs.len() && pairs[i + run as usize] == (x0 + run, t0 + run) {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            offsets.push(t0);
            counts.push(run);
            i += run as usize;
        }
        let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; coords.len()];
        s.queue.UpdateTileMappings(&s.windows[k], coords.len() as u32, Some(coords.as_ptr()), Some(regions.as_ptr()), &s.pools[heap as usize], flags.len() as u32, Some(flags.as_ptr()), Some(offsets.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
    }
    super::real::removed_probe("view_mem::ensure: after the tile mappings");
    for batch in todo.chunks(STAGING_TILES) {
        // lab (`no_wait`): the next ring part, once the GPU is done with its last batch
        let (list, part) = if no_wait() {
            let i = s.ring_next;
            s.ring_next = (i + 1) % RING;
            if let Some(v) = ring_wait(s.ring[i].value, super::real::fence_completed(&s.fence)) {
                RING_WAITS.fetch_add(1, Ordering::Relaxed);
                let _ = crate::gpu::fence::wait(&s.fence, v, &s.dev, None, 5000);
            }
            let r = &s.ring[i];
            if r.alloc.Reset().is_err() || r.list.Reset(&r.alloc, None).is_err() {
                return;
            }
            (r.list.clone(), i * STAGING_TILES)
        } else {
            if s.alloc.Reset().is_err() || s.list.Reset(&s.alloc, None).is_err() {
                return;
            }
            (s.list.clone(), 0)
        };
        for (i, &(tile, src, _)) in batch.iter().enumerate() {
            if !s.tiles.contains_key(&tile) {
                continue;
            }
            let at = (part + i) * TILE;
            std::ptr::write_bytes(s.staging_at.add(at), 0, TILE);
            super::title_textures::copy_committed_region(src, s.staging_at.add(at), TILE);
            let k = (tile - s.base) / WINDOW - FIRST;
            list.CopyBufferRegion(&s.windows[k], ((tile - s.base) % WINDOW) as u64, &s.staging, at as u64, TILE as u64);
        }
        if list.Close().is_err() {
            return;
        }
        let Ok(cl) = list.cast::<ID3D12CommandList>() else { return };
        let was_removed = s.dev.GetDeviceRemovedReason().is_err();
        s.queue.ExecuteCommandLists(&[Some(cl)]);
        if !was_removed && s.dev.GetDeviceRemovedReason().is_err() {
            let first: Vec<String> = batch.iter().take(8).map(|&(tile, src, new)| format!("{tile:#x} (window {}, tile {}, from {src:#x}, new {new}, pool tile {:?})", (tile - s.base) / WINDOW, (tile - s.base) % WINDOW / TILE, s.tiles.get(&tile).map(|t| t.slot))).collect();
            crate::xlog_error!("lab: view memory: the device is removed right after the upload list of {} tiles ({} new in this call of {} tiles to do, range {va:#x}+{len:#x}); {} tiles in {} pools, next pool tile {}, fence value {}: {}", batch.len(), todo.iter().filter(|t| t.2).count(), todo.len(), s.tiles.len(), s.pools.len(), s.next, s.value, first.join("; "));
        }
        super::real::removed_probe("view_mem::ensure: after its ExecuteCommandLists");
        s.value += 1;
        if s.queue.Signal(&s.fence, s.value).is_err() {
            return;
        }
        if no_wait() {
            s.ring[part / STAGING_TILES].value = s.value;
            UPLOAD_LAST.store(s.value, Ordering::Release);
            UPLOADED.fetch_add(batch.len() as u64, Ordering::Relaxed);
            continue;
        }
        if let Err(e) = crate::gpu::fence::wait(&s.fence, s.value, &s.dev, None, 5000) {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: view memory: upload not finished: {e}");
            }
        }
        UPLOADED.fetch_add(batch.len() as u64, Ordering::Relaxed);
    }
}

/// The ranges a draw with these root arguments can read outside the unified windows: its root
/// addresses (`XCL_VIEW_ROOT_KB` each – their sizes are not known; lab `XCL_VIEW_FULL_RANGE=1`:
/// `plan`) and the buffer V#s (16 bytes at every 32 of a descriptor table: address, stride << 16
/// in word 1, records, type in the top bits of word 3) of its tables' first `slots` descriptors.
pub unsafe fn ensure_for_draw(layout: Option<&super::console_list::Layout>, args: &[u8]) {
    use super::console_list::Param;
    let (base, on) = match STATE.lock().unwrap().as_ref() {
        Some(s) => (s.base, true),
        None => (0, false),
    };
    let Some(layout) = layout.filter(|_| on) else { return };
    static SLOTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let c = cfg();
    // lab: the bytes left in the title's allocation at `va`, when XCL knows it (with the switch)
    let alloc_left = |va: usize| if c.full { super::real::outside_buffer_left(va) } else { None };
    let slots = *SLOTS.get_or_init(|| std::env::var("XCL_VIEW_TABLE_SLOTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let (lo, hi) = (base + FIRST * WINDOW, base + END * WINDOW);
    for p in &layout.params {
        match *p {
            Param::Descriptor { at } => {
                let (Some(l), Some(h)) = (word(at as usize), word(at as usize + 4)) else { continue };
                let va = ((h as usize & 0xFFFF) << 32) | l as usize;
                if va >= lo && va < hi {
                    ROOTS_NAMED.fetch_add(1, Ordering::Relaxed);
                    let (len, far, why) = plan(c, Named::Root, alloc_left(va));
                    ensure_range(va, len, far, c.root, true, why);
                }
            }
            Param::Table { at } => {
                let Some(t) = word(at as usize) else { continue };
                if t == 0 {
                    continue;
                }
                let table = base + t as usize;
                if !crate::gpu::shader::readable(table, 32 * slots) {
                    continue;
                }
                for k in 0..slots {
                    let d = ((table + 32 * k) as *const [u32; 4]).read_unaligned();
                    let va = ((d[1] as usize & 0xFFFF) << 32) | d[0] as usize;
                    if va < lo || va >= hi || d[3] >> 30 != 0 {
                        continue;
                    }
                    let stride = (d[1] >> 16) as usize & 0x3FFF;
                    let bytes = if stride == 0 { d[2] as usize } else { (d[2] as usize).saturating_mul(stride) };
                    if bytes > VSHARP_MOST {
                        VSHARP_CUT.fetch_add(1, Ordering::Relaxed);
                    }
                    let (len, far, why) = plan(c, Named::View(bytes), alloc_left(va));
                    ensure_range(va, len, far, bytes.clamp(4, VSHARP_MOST), false, why);
                }
            }
            Param::Constants { .. } => {}
        }
    }
}

/// One line for the activity log.
pub fn stats() -> String {
    let (tiles, pools) = (TILES.load(Ordering::Relaxed), POOLS.load(Ordering::Relaxed));
    let c = cfg();
    let by = |k: usize| SIZE_BY[k].load(Ordering::Relaxed);
    format!(
        "view memory: {} MB in {tiles} tiles ({pools} pools), {} given back, {} tile uploads, {} tile checks, {} without a source; lab ranges: roots {} named, {} looked at, {} with a sourced tile past XCL_VIEW_ROOT_KB; {} V#s over 16 MiB; full range {}: {} MB past the old bounds, sizes from an allocation {}, the V# {}, sourced tiles {}, the cap {}; fresh (< {} ms): {} ranges and {} tiles not looked at again; pool full: {} rounds, {} tiles refused; without a source: {} tiles past the old bounds, {} not checked again (memo); longest range {} MB{}",
        tiles >> 4,
        EVICTED.load(Ordering::Relaxed),
        UPLOADED.load(Ordering::Relaxed),
        CHECKED.load(Ordering::Relaxed),
        NO_SOURCE.load(Ordering::Relaxed),
        ROOTS_NAMED.load(Ordering::Relaxed),
        ROOTS_LOOKED.load(Ordering::Relaxed),
        ROOTS_CUT.load(Ordering::Relaxed),
        VSHARP_CUT.load(Ordering::Relaxed),
        if c.full { "on" } else { "off" },
        ADDED.load(Ordering::Relaxed) >> 20,
        by(0),
        by(1),
        by(2),
        by(3),
        c.fresh.as_millis(),
        RECENT.load(Ordering::Relaxed),
        FRESH_TILES.load(Ordering::Relaxed),
        ROUNDS.load(Ordering::Relaxed),
        FULL.load(Ordering::Relaxed),
        PAST_MISS.load(Ordering::Relaxed),
        MEMO.load(Ordering::Relaxed),
        LONGEST.load(Ordering::Relaxed) >> 20,
        if no_wait() { format!("; no-wait: {} queue waits, {} ring waits", QUEUE_WAITS.load(Ordering::Relaxed), RING_WAITS.load(Ordering::Relaxed)) } else { String::new() }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: usize = 1 << 20;

    /// lab (`XCL_VIEW_NO_WAIT=1`): a title queue waits for the last upload only while the fence
    /// has not passed it; a ring part is reused at once when its batch is done.
    #[test]
    fn uploads_are_waited_for_on_the_gpu_only_while_in_flight() {
        assert_eq!(upload_wait(0, 0), None, "no upload yet");
        assert_eq!(upload_wait(7, 6), Some(7));
        assert_eq!(upload_wait(7, 7), None, "done");
        assert_eq!(ring_wait(0, 0), None, "never used");
        assert_eq!(ring_wait(5, 4), Some(5));
        assert_eq!(ring_wait(5, 9), None);
    }

    fn cfg_of(vars: &[(&str, &str)]) -> Cfg {
        cfg_from(|k| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string()))
    }

    /// `ensure_range`'s walk over the tiles, with `sourced` for `source`: (tiles looked at, tiles
    /// without a source counted, where the range ended, why)
    fn walk(va: usize, plan: (usize, usize, Why), hi: usize, sourced: impl Fn(usize) -> bool) -> (Vec<usize>, usize, usize, Why) {
        let mut r = Reach::new(va, plan.0, plan.1, hi);
        let mut t = va & !(TILE - 1);
        let (mut seen, mut counted) = (Vec::new(), 0);
        while r.covers(t) {
            let tile = t;
            t += TILE;
            seen.push(tile);
            if !sourced(tile) && r.missing(tile) {
                counted += 1;
            }
        }
        let reached = r.reached(t);
        (seen, counted, reached, r.why(reached, plan.2))
    }

    #[test]
    fn view_ranges_keep_the_old_bounds_without_the_switch() {
        let c = cfg_of(&[]);
        assert_eq!(c, Cfg { root: MB, full: false, cap: 256 * MB, fresh: std::time::Duration::from_millis(1000) });
        assert!(!cfg_of(&[("XCL_VIEW_FULL_RANGE", "0")]).full);
        assert_eq!(plan(&c, Named::Root, None), (MB, MB, Why::Bound));
        assert_eq!(plan(&c, Named::Root, Some(64 * MB)), (MB, MB, Why::Bound));
        assert_eq!(plan(&c, Named::View(0), None), (4, 4, Why::Bound));
        assert_eq!(plan(&c, Named::View(40 * MB), None), (16 * MB, 16 * MB, Why::Bound));
        // the walk is the old `while t < end`: every tile of [va, va + 1 MiB), none after, holes counted
        let va = 0xD_0000_8000;
        let (seen, counted, reached, why) = walk(va, plan(&c, Named::Root, None), usize::MAX, |t| t != 0xD_0004_0000);
        let old: Vec<usize> = (0..).map(|k| (va & !(TILE - 1)) + k * TILE).take_while(|&t| t < va + MB).collect();
        assert_eq!(seen, old);
        assert_eq!((counted, reached, why), (1, va + MB, Why::Bound));
        // XCL_VIEW_ROOT_KB and XCL_VIEW_FRESH_MS as before / new
        let c = cfg_of(&[("XCL_VIEW_ROOT_KB", " 65536 "), ("XCL_VIEW_FRESH_MS", "250")]);
        assert_eq!((c.root, c.fresh.as_millis()), (64 * MB, 250));
    }

    #[test]
    fn full_view_ranges_end_at_the_allocation_the_vsharp_or_the_sourced_tiles() {
        let c = cfg_of(&[("XCL_VIEW_FULL_RANGE", "1")]);
        assert!(c.full);
        // the title's allocation first, a V#'s size inside it, never past it
        assert_eq!(plan(&c, Named::Root, Some(3 * MB)), (3 * MB, 3 * MB, Why::Alloc));
        assert_eq!(plan(&c, Named::View(100), Some(MB)), (100, 100, Why::Given));
        assert_eq!(plan(&c, Named::View(2 * MB), Some(MB)), (MB, MB, Why::Alloc));
        assert_eq!(plan(&c, Named::View(40 * MB), Some(64 * MB)), (40 * MB, 40 * MB, Why::Given));
        // a V# without an allocation: its size, the 16 MiB bound gone
        assert_eq!(plan(&c, Named::View(MB), None), (MB, MB, Why::Given));
        assert_eq!(plan(&c, Named::View(40 * MB), None), (16 * MB, 40 * MB, Why::Given));
        // a root without one: 1 MiB at least, then the run of sourced tiles
        let va = 0xD_0001_0000;
        let p = plan(&c, Named::Root, None);
        assert_eq!(p, (MB, 256 * MB, Why::Cap));
        let (seen, counted, reached, why) = walk(va, p, usize::MAX, |t| t < va + 5 * MB);
        assert_eq!((seen.len(), counted, reached, why), (81, 0, va + 5 * MB, Why::Tiles));
        // a hole in the first MiB: the old range only, the hole counted
        let (seen, counted, reached, why) = walk(va, p, usize::MAX, |t| t != va + 2 * TILE);
        assert_eq!((seen.len(), counted, reached, why), (16, 1, va + MB, Why::Tiles));
        // a V# over 16 MiB whose tiles all have a source reaches its size
        let (_, _, reached, why) = walk(va, plan(&c, Named::View(40 * MB), None), usize::MAX, |_| true);
        assert_eq!((reached, why), (va + 40 * MB, Why::Given));
        // nor past the end of the windows
        let (_, _, reached, _) = walk(va, p, va + 3 * MB, |_| true);
        assert_eq!(reached, va + 3 * MB);
    }

    #[test]
    fn full_view_ranges_stop_at_the_cap() {
        let c = cfg_of(&[("XCL_VIEW_FULL_RANGE", "1"), ("XCL_VIEW_FULL_MB", "8")]);
        assert_eq!(c.cap, 8 * MB);
        let va = 0xD_0000_0000;
        let p = plan(&c, Named::Root, None);
        let (seen, _, reached, why) = walk(va, p, usize::MAX, |_| true);
        assert_eq!((seen.len(), reached, why), (128, va + 8 * MB, Why::Cap));
        assert_eq!(plan(&c, Named::Root, Some(64 * MB)), (8 * MB, 8 * MB, Why::Cap));
        assert_eq!(plan(&c, Named::View(1 << 30), None), (16 * MB, 16 * MB, Why::Cap));
        // never short of the old bounds
        let c = cfg_of(&[("XCL_VIEW_FULL_RANGE", "1"), ("XCL_VIEW_FULL_MB", "0"), ("XCL_VIEW_ROOT_KB", "4096")]);
        assert_eq!(c.cap, usize::MAX);
        assert_eq!(plan(&c, Named::Root, None), (4 * MB, usize::MAX, Why::Cap));
        let c = cfg_of(&[("XCL_VIEW_FULL_RANGE", "1"), ("XCL_VIEW_FULL_MB", "1"), ("XCL_VIEW_ROOT_KB", "4096")]);
        assert_eq!(plan(&c, Named::Root, None), (4 * MB, 4 * MB, Why::Cap));
    }

    #[test]
    fn tiles_without_a_source_are_not_checked_again_while_fresh() {
        let fresh = std::time::Duration::from_millis(1000);
        let now = std::time::Instant::now();
        let mut m = Misses::default();
        assert!(!m.known(0xD_0000_0000, now, fresh));
        m.note(0xD_0000_0000, now);
        assert!(m.known(0xD_0000_0000, now, fresh));
        assert!(m.known(0xD_0000_0000, now + std::time::Duration::from_millis(999), fresh));
        assert!(!m.known(0xD_0000_0000, now + fresh, fresh));
        assert!(!m.known(0xD_0001_0000, now, fresh));
        // a memo hit ends a full range as a check would
        let mut r = Reach::new(0xD_0000_0000, MB, 8 * MB, usize::MAX);
        assert!(!r.missing(0xD_0010_0000));
        assert_eq!(r.reached(0xD_0011_0000), 0xD_0010_0000);
        // bounded: past 200 000 tiles it starts again (a memo of its own: `m` holds a tile already)
        let mut m = Misses::default();
        for k in 0..200_001 {
            m.note(k * TILE, now);
        }
        assert_eq!(m.0.len(), 200_001);
        m.note(0xE_0000_0000, now);
        assert_eq!(m.0.len(), 1);
        assert!(!m.known(0, now, fresh));
        assert!(m.known(0xE_0000_0000, now, fresh));
    }

    #[test]
    fn view_range_counters_go_to_the_stats_line() {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let before = (load(&ROOTS_LOOKED), load(&ROOTS_CUT), load(&ADDED), load(&SIZE_BY[0]), load(&SIZE_BY[1]), load(&SIZE_BY[2]), load(&SIZE_BY[3]));
        count(true, true, false, Why::Bound, 0);
        count(true, false, true, Why::Tiles, 3 * MB);
        count(false, true, true, Why::Alloc, 0);
        count(false, false, true, Why::Given, MB);
        count(false, false, true, Why::Cap, 0);
        count(false, false, true, Why::Bound, 0);
        let after = (load(&ROOTS_LOOKED), load(&ROOTS_CUT), load(&ADDED), load(&SIZE_BY[0]), load(&SIZE_BY[1]), load(&SIZE_BY[2]), load(&SIZE_BY[3]));
        assert_eq!(after, (before.0 + 2, before.1 + 1, before.2 + 4 * MB as u64, before.3 + 1, before.4 + 1, before.5 + 1, before.6 + 1));
        let line = stats();
        assert!(line.starts_with("view memory: "), "{line}");
        assert!(line.contains(" without a source; lab ranges: roots "), "{line}");
        assert!(line.contains(&format!(", {} with a sourced tile past XCL_VIEW_ROOT_KB; ", after.1)), "{line}");
        assert!(line.contains(&format!(" MB past the old bounds, sizes from an allocation {}, the V# {}, sourced tiles {}, the cap {}; ", after.3, after.4, after.5, after.6)), "{line}");
        assert!(line.contains(" pool full: ") && line.contains(" rounds, "), "{line}");
        assert!(line.contains("; without a source: ") && line.contains(" not checked again (memo); longest range "), "{line}");
    }
}
