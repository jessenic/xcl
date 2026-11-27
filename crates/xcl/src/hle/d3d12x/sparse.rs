//! The title's own memory, seen by the GPU: sparse windows.
//!
//! A console title keeps GPU data in memory it reserves and commits itself (XDefiant: one
//! 8 GB reservation of 64 KiB pages it commits in steps and places buffers in) – on the
//! console all of it is GPU-visible at the same addresses. XCL gives such reservations
//! addresses in the *sparse windows* of the unified-memory block (`gpu::uma`: windows 0..2
//! in `for_runtime`), so translated shaders reach them through `mem`…`mem2` like any unified
//! memory. On the GPU each sparse window is a reserved (tiled) buffer of 4 GiB whose 64 KiB
//! tiles are mapped onto video-memory heaps as the title commits pages.
//! The CPU's view stays the title's memory: what the CPU wrote (write-watch) is copied into
//! the tiles before the title's next ExecuteCommandLists via an asynchronous ring-buffered
//! staging upload queue, and the title's queues GPU-wait on the upload fence (no CPU stall).
//! Only the bytes the CPU changed go (64-byte chunks against a hash of what was sent last,
//! `changed_chunks`): the rest of a written page may hold what the GPU wrote there (counters,
//! indirect arguments) and must not be overwritten with the CPU's stale view of it.
//! Buffers the title places there get reserved resources over the same tiles, so PC-side
//! copies, views and barriers see the same memory.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_DECOMMIT, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READWRITE,
};

const TILE: usize = 0x1_0000;
/// tiles of a video-memory pool heap (64 MiB)
const POOL_TILES: u32 = 1024;
/// tiles decommitted while no title work was submitted that stay mapped (`State::parked`) before
/// they are unmapped together: 256 MB of video memory
const PARKED_MAX: usize = 4096;
/// bytes of the ring-buffered upload staging buffer
const STAGING: usize = 64 << 20;
/// alignment of sub-allocations inside the upload staging ring (write-combined cacheline)
const RING_ALIGN: usize = 64;
/// number of in-flight upload command allocators / lists
const UPLOAD_SLOTS: usize = 16;
/// bytes of the dedicated pre-zeroed buffer for tile/hole clears (1 MiB)
const ZERO_BUF: usize = 1 << 20;
/// bytes of the GPU→CPU readback staging buffer
const READBACK_STAGING: usize = 16 << 20;
/// bytes of the upload ring that are free before a part of the uploads starts (`upload_room`)
const RING_ROOM: usize = 1 << 20;

const PAGE: usize = 0x1000;
const PAGES_PER_TILE: usize = TILE / PAGE;
const CHUNK: usize = 64;
const CHUNKS: usize = PAGE / CHUNK;

/// Fast non-cryptographic integer hasher for page/tile lookup maps on the hot submit path.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        // Tile addresses are 64 KiB aligned. Move mixed bits into the low bucket-index bits.
        self.0.rotate_right(23)
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

struct TileEntry {
    pool: (u32, u32),
    /// bit `i` set => 4 KiB page `i` of this 64 KiB tile is uncommitted (a hole)
    holes: u16,
    /// per-4 KiB-page hashes of the 64-byte chunks last sent to the GPU (`None` = all zero)
    sent: [Option<Box<[u64; CHUNKS]>>; PAGES_PER_TILE],
}

impl TileEntry {
    fn with_holes(pool: (u32, u32), holes: u16) -> Self {
        Self { pool, holes, sent: std::array::from_fn(|_| None) }
    }
}

/// Query actual OS page commitment across the 64 KiB tile at `tile`.
/// Bit `i` in the returned mask is set whenever 4 KiB page `tile + i * PAGE` is not committed
/// and accessible (`MEM_RESERVE`, `MEM_FREE`, `PAGE_NOACCESS`, `PAGE_GUARD`, etc.).
unsafe fn query_tile_holes(tile: usize) -> u16 {
    let mut holes = 0xFFFFu16;
    let end = tile.saturating_add(TILE);
    let mut cur = tile;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        // SAFETY: `VirtualQuery` inspects process virtual memory metadata without dereferencing `cur`.
        let n = super::call_stats::cs!(VirtualQuery, VirtualQuery(cur as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()));
        if n == 0 || mbi.RegionSize == 0 {
            break;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(end);
        let accessible = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
        if accessible {
            let mut pg = cur;
            while pg < region_end {
                holes &= !(1u16 << ((pg - tile) / PAGE));
                pg += PAGE;
            }
        }
        if region_end <= cur {
            break;
        }
        cur = region_end;
    }
    holes
}

/// Update `mapped`, `free`, and `to_zero` for a decommit of `[a, a + n)`.
/// Returns the base addresses of tiles that became completely uncommitted (`holes == 0xFFFF`)
/// and were removed from `mapped` (ready to be unmapped from their sparse window buffers).
fn decommit_tiles(
    mapped: &mut FxHashMap<usize, TileEntry>,
    pinned: &FxHashMap<(u32, u32), u32>,
    free: &mut Vec<(u32, u32)>,
    to_zero: &mut Vec<(usize, usize)>,
    a: usize,
    n: usize,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut unmapped: Vec<usize> = Vec::new();
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = mapped.get_mut(&t) {
            let mut newly_holed = 0u16;
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                entry.sent[idx] = None;
                let bit = 1u16 << idx;
                if entry.holes & bit == 0 {
                    entry.holes |= bit;
                    newly_holed |= bit;
                }
                p += PAGE;
            }
            if entry.holes == 0xFFFF {
                unmapped.push(t);
            } else if newly_holed != 0 {
                for idx in 0..PAGES_PER_TILE {
                    if newly_holed & (1u16 << idx) != 0 {
                        to_zero.push((t + idx * PAGE, PAGE));
                    }
                }
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
    for &t in &unmapped {
        if let Some(entry) = mapped.remove(&t) {
            if !pinned.contains_key(&entry.pool) {
                free.push(entry.pool);
            }
        }
    }
    if !unmapped.is_empty() {
        to_zero.retain(|&(za, _)| mapped.contains_key(&(za & !(TILE - 1))));
    }
    unmapped
}

struct UploadSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_val: u64,
}

/// A copy back on `State::rbq`: its list, its readback buffer, the `rb_fence` value it waits
/// for, the `rbq` fence value that marks it done, and the copies into the title's memory
/// (address, offset in `buf`, bytes).
struct RbJob {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    buf: ID3D12Resource,
    ptr: *mut u8,
    cap: usize,
    rb_v: u64,
    done_v: u64,
    copies: Vec<(usize, usize, usize)>,
    /// the fences of its title queue's copy back queue (`State::rb_queues`): the first reaches
    /// `rb_v` when the lists it follows are done, the second `done_v` when the copy is. None:
    /// `rb_fence` and the fence of `rbq`
    fences: Option<(ID3D12Fence, ID3D12Fence)>,
}

/// The copies back of one title queue (`State::rb_queues`): a fence the title queue signals
/// after its lists, a queue of XCL's that waits for it and runs the copies, and the fence that
/// queue signals after them, with the last values.
struct RbQueue {
    title: ID3D12Fence,
    title_v: u64,
    queue: ID3D12CommandQueue,
    done: ID3D12Fence,
    done_v: u64,
}

/// How far `drain_readbacks` goes
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Drain {
    /// only copies the GPU has finished
    Finished,
    /// also those whose title work is done (waiting for the copy itself): before a CPU read
    /// the title made sure of, e.g. after its wait for the value a list wrote (outside tests the
    /// wait is made with `STATE` released instead: `settle_released`)
    #[cfg_attr(not(test), allow(dead_code))]
    TitleDone,
}

struct RetiredTiles {
    tiles: Vec<(u32, u32)>,
    /// Fence marks on `self.queue` and every known title queue (`None` = signal failed, keep held).
    marks: Option<Vec<(ID3D12Fence, u64)>>,
}

struct State {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    /// reusable auto-reset Win32 event for fence waits (no per-wait CreateEventW/CloseHandle)
    event: HANDLE,
    /// fence signalled on title queues after their lists, so a copy back waits for them
    rb_fence: ID3D12Fence,
    rb_value: u64,
    /// copies back after the title's submissions (`readback_after_execute`): on a queue of their
    /// own (`rbq`, None: on `queue`, waited for on the CPU), landing in the title's memory once
    /// done (`drain_readbacks`). The title's queues can wait for work the title submits after
    /// this one, from the same thread: waiting here for its lists stopped that submission – and,
    /// on `queue`, every upload behind it (`before_execute` waiting for ring room).
    rbq: Option<(ID3D12CommandQueue, ID3D12Fence)>,
    rbq_value: u64,
    /// title queue → its copy back queue and fences (used while `rbq` is there; `rbq` and
    /// `rb_fence` only when one cannot be made). With one fence for all title queues, a queue that
    /// got to its Signal first let the copy of a queue still waiting (for work the title submits
    /// later) run early: it read the bytes before that queue's lists wrote them, landed them, and
    /// nothing read them again (`Wait64BitValueX timed out … memory holds 0` once the title's
    /// queues wait for each other). With one copy queue for all, every copy waited behind such a
    /// queue.
    rb_queues: FxHashMap<usize, RbQueue>,
    /// submitted copies back, oldest first, and finished ones to reuse
    rb_jobs: VecDeque<RbJob>,
    rb_pool: Vec<RbJob>,
    /// (window start, reserved buffer) of each sparse window
    windows: Vec<(usize, ID3D12Resource)>,
    /// video-memory pool heaps (None: given back, its index free for the next one)
    pools: Vec<Option<ID3D12Heap>>,
    free: Vec<(u32, u32)>,
    /// pools all of whose tiles came free: released once the GPU is past `value` and a moment
    /// went by (titles' queues may still read what was mapped there)
    retired: Vec<(ID3D12Heap, u64, std::time::Instant)>,
    /// tile address → pool tile + uncommitted page hole mask + per-page sent chunk hashes
    mapped: FxHashMap<usize, TileEntry>,
    /// pool tiles also used by placed buffers (not freed while they are)
    pinned: FxHashMap<(u32, u32), u32>,
    /// tile address → pool tile of tiles decommitted since the last title submit, still mapped
    /// in their window: unmapped together (`flush_parked`) before the next submit, when the free
    /// list runs out, or past `PARKED_MAX`; a commit there before that takes its tile back with
    /// no new mapping. (A retail title moving physical pages one at a time while it loads
    /// decommits and commits again thousands of tiles a second, and each `UpdateTileMappings`
    /// took ~0.15 ms.)
    parked: FxHashMap<usize, (u32, u32)>,
    /// unmapped parked pool tiles awaiting `self.queue` and in-flight title queues before entering `free`
    quarantined: VecDeque<RetiredTiles>,
    /// lab (`XCL_SPARSE_COOL=1`): pool tiles a buffer gave back (`release_if_unused`), quarantined
    /// with the parked ones at the next flush instead of free at once
    released: Vec<(u32, u32)>,
    /// per-title-queue fences used to mark in-flight GPU work (`note_queue`)
    queue_fences: FxHashMap<usize, (ID3D12CommandQueue, ID3D12Fence, u64)>,
    /// tile address → pool tile of decommitted tiles a placed buffer uses: they stay mapped in
    /// their window until the buffer is gone. Memory committed there again is then the same for
    /// the window (what shaders write) and the buffer (what an ExecuteIndirect or a copy of the
    /// title reads); with a new tile in the window the buffer kept the old one, with old bytes
    kept: FxHashMap<usize, (u32, u32)>,
    /// placed buffers → the tiles they use (address, pool tile)
    buffers: FxHashMap<usize, Vec<(usize, (u32, u32))>>,
    /// reused tiles / decommitted partial pages to clear: (address, bytes)
    to_zero: Vec<(usize, usize)>,
    /// ring-buffered upload staging heap (`STAGING` bytes, persistently mapped)
    staging: Option<(ID3D12Resource, *mut u8)>,
    /// monotonic ring-buffer write/submitted/retired byte cursors
    ring_write: u64,
    ring_submitted: u64,
    ring_free: u64,
    /// submitted ring chunks awaiting GPU completion: `(fence_val, ring_end_pos)`
    inflight: VecDeque<(u64, u64)>,
    /// dedicated 1 MiB pre-zeroed buffer for tile/hole clears (zero staging bandwidth)
    zero_buf: Option<ID3D12Resource>,
    /// ring of async upload command allocators + lists
    slots: Vec<UploadSlot>,
    next_slot: usize,
    /// dedicated readback staging buffer + command allocator / list
    readback: Option<(ID3D12Resource, *mut u8)>,
    rb_alloc: ID3D12CommandAllocator,
    rb_list: ID3D12GraphicsCommandList,
    rb_last_fence: u64,
    /// the fence value each title queue last waited for
    waited: FxHashMap<usize, u64>,
    /// `mapped` as sorted (start, end) runs (None: to be made again)
    runs: Option<Vec<(usize, usize)>>,
    /// reusable buffer for dirty runs in `before_execute` (avoids per-submit heap alloc)
    scratch_runs: Vec<(usize, usize)>,
    /// lab: placed buffers without video memory yet (`place_buffer` with `lazy`): (resource,
    /// first tile, bytes from there); their tiles are mapped at the first use (`use_range`)
    lazy_bufs: Vec<(usize, usize, usize)>,
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

const K0: u64 = 0x9E37_79B9_7F4A_7C15;
const K1: u64 = 0xBF58_476D_1CE4_E5B9;
const K2: u64 = 0x94D0_49BB_1331_11EB;
const K3: u64 = 0x243F_6A88_85A3_08D3;

/// 4-way parallel hash over 8 `u64` words (64 bytes): breaks serial multiply latency chain.
#[inline(always)]
const fn hash_words(w: [u64; 8]) -> u64 {
    let a = (w[0] ^ K3).wrapping_mul(K0) ^ w[4].wrapping_mul(K1);
    let b = (w[1] ^ K0).wrapping_mul(K1) ^ w[5].wrapping_mul(K2);
    let c = (w[2] ^ K1).wrapping_mul(K2) ^ w[6].wrapping_mul(K3);
    let d = (w[3] ^ K2).wrapping_mul(K3) ^ w[7].wrapping_mul(K0);
    let h = a.rotate_left(27).wrapping_add(b.rotate_left(13)) ^ c.rotate_left(41).wrapping_add(d.rotate_left(53));
    h.wrapping_mul(K0) ^ (h >> 32)
}

const ZERO_HASH: u64 = hash_words([0; 8]);
const ZERO_HASHES: [u64; CHUNKS] = [ZERO_HASH; CHUNKS];

#[inline(always)]
fn read_chunk_words(c: &[u8]) -> [u64; 8] {
    debug_assert_eq!(c.len(), CHUNK);
    let p = c.as_ptr() as *const [u8; 8];
    unsafe {
        [
            u64::from_le_bytes(p.add(0).read_unaligned()),
            u64::from_le_bytes(p.add(1).read_unaligned()),
            u64::from_le_bytes(p.add(2).read_unaligned()),
            u64::from_le_bytes(p.add(3).read_unaligned()),
            u64::from_le_bytes(p.add(4).read_unaligned()),
            u64::from_le_bytes(p.add(5).read_unaligned()),
            u64::from_le_bytes(p.add(6).read_unaligned()),
            u64::from_le_bytes(p.add(7).read_unaligned()),
        ]
    }
}

/// hash of 64 bytes
#[inline(always)]
fn chunk_hash(c: &[u8]) -> u64 {
    hash_words(read_chunk_words(c))
}

/// lab: `XCL_READBACK_GUARD=1`: a synchronous copy back (`readback_released`) lands a 64-byte
/// chunk only if the CPU's view of it is as it was when the copy was submitted: the copy is
/// waited for with `STATE` released, and the title may write there meanwhile (that write is
/// kept, and goes to video memory with the next submit, as its `sent` hash stays).
fn readback_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_READBACK_GUARD").is_ok_and(|v| v.trim() == "1"))
}

/// lab (`readback_guard`): chunks a copy back did not land, as the CPU wrote them meanwhile
static GUARD_KEPT: AtomicU64 = AtomicU64::new(0);

/// The hashes of the 64-byte chunks that `parts` (address, offset, bytes) touch in the CPU view,
/// in order; 0 for a chunk of a page that is not `committed`.
unsafe fn chunk_hashes(parts: &[(usize, usize, usize)], committed: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut out = Vec::new();
    for &(dst, _, len) in parts {
        let mut c = dst & !(CHUNK - 1);
        while c < dst + len {
            out.push(if committed(c & !(PAGE - 1)) { chunk_hash(std::slice::from_raw_parts(c as *const u8, CHUNK)) } else { 0 });
            c += CHUNK;
        }
    }
    out
}

/// `parts` (address, offset at `p`, bytes) copied from `p` into the CPU view, a 64-byte chunk at
/// a time, where the chunk's page is `committed` and the chunk still hashes as `before`
/// (`chunk_hashes` when the copy was submitted). Returns the chunks landed (address, hash after)
/// and the number kept as the CPU wrote them.
unsafe fn land_unchanged(p: *const u8, parts: &[(usize, usize, usize)], before: &[u64], committed: impl Fn(usize) -> bool) -> (Vec<(usize, u64)>, usize) {
    let (mut landed, mut kept, mut i) = (Vec::new(), 0, 0);
    for &(dst, off, len) in parts {
        let end = dst + len;
        let mut c = dst & !(CHUNK - 1);
        while c < end {
            let was = before.get(i).copied();
            i += 1;
            if committed(c & !(PAGE - 1)) {
                let view = std::slice::from_raw_parts(c as *const u8, CHUNK);
                if was == Some(chunk_hash(view)) {
                    let (lo, hi) = (c.max(dst), (c + CHUNK).min(end));
                    std::ptr::copy_nonoverlapping(p.add(off + (lo - dst)), lo as *mut u8, hi - lo);
                    landed.push((c, chunk_hash(std::slice::from_raw_parts(c as *const u8, CHUNK))));
                } else {
                    kept += 1;
                }
            }
            c += CHUNK;
        }
    }
    (landed, kept)
}

#[inline(always)]
fn push_run(out: &mut Vec<(usize, usize)>, addr: usize, len: usize) {
    match out.last_mut() {
        Some(last) if last.0 + last.1 == addr => last.1 += len,
        _ => out.push((addr, len)),
    }
}

/// Scan a 4 KiB page at `pg`, appending any changed 64-byte chunk runs directly to `out`
/// (zero per-page heap allocation; skips allocating `slot` completely when a fresh zero page
/// is still all zeros).
#[inline]
fn scan_page(page: &[u8], pg: usize, slot: &mut Option<Box<[u64; CHUNKS]>>, out: &mut Vec<(usize, usize)>) {
    if let Some(sent) = slot.as_deref_mut() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            let h = chunk_hash(c);
            if h == sent[k] {
                continue;
            }
            sent[k] = h;
            push_run(out, pg + k * CHUNK, CHUNK);
        }
        return;
    }
    // Page has never had non-zero data sent: skip zero chunks with a cheap OR-reduction and
    // allocate the hash array only if at least one chunk is non-zero.
    let mut chunks = page.chunks_exact(CHUNK).enumerate();
    while let Some((k, c)) = chunks.next() {
        let w = read_chunk_words(c);
        if (w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0 {
            continue;
        }
        let h = hash_words(w);
        if h == ZERO_HASH {
            continue;
        }
        let sent = slot.insert(Box::new(ZERO_HASHES));
        sent[k] = h;
        push_run(out, pg + k * CHUNK, CHUNK);
        for (k2, c2) in chunks {
            let h2 = chunk_hash(c2);
            if h2 == ZERO_HASH {
                continue;
            }
            sent[k2] = h2;
            push_run(out, pg + k2 * CHUNK, CHUNK);
        }
        break;
    }
}

/// The byte ranges (offset, length) of `page` whose 64-byte chunks differ from `sent`
/// (updated to the page's chunks).
pub fn changed_chunks(page: &[u8], sent: &mut [u64; CHUNKS]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, c) in page.chunks_exact(CHUNK).enumerate() {
        let h = chunk_hash(c);
        if h == sent[k] {
            continue;
        }
        sent[k] = h;
        push_run(&mut out, k * CHUNK, CHUNK);
    }
    out
}

/// Plan a contiguous sub-allocation `(aligned_pos, chunk_len)` inside a ring buffer of `cap`
/// bytes starting at monotonic cursor `ring_write`, keeping each chunk `<= cap / 2` and
/// avoiding mid-copy wrap-around without wasting large tail gaps.
#[inline]
fn plan_ring_chunk(ring_write: u64, want: usize, cap: usize) -> (u64, usize) {
    let mut chunk = want.min(cap / 2);
    let mut aligned = (ring_write + (RING_ALIGN as u64 - 1)) & !(RING_ALIGN as u64 - 1);
    let off = (aligned as usize) % cap;
    if off + chunk > cap {
        let tail = cap - off;
        if tail >= PAGE {
            chunk = tail;
        } else {
            aligned += tail as u64;
        }
    }
    (aligned, chunk)
}

// D3D12 objects are free-threaded; staging pointers and event handle are only used under STATE lock
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// The state's queue again, outside `STATE` (`queue`). `STATE` is held around D3D12 calls (an
/// upload's allocator `Reset` in `begin_slot`), and `late_release` asks for the queue from a
/// resource's last `Release`, which D3D12 can make inside a call of its own on another thread
/// (a list's `Reset`, with `XCL_DRED=1` on every start): the two then waited for each other.
static QUEUE: Mutex<Option<ID3D12CommandQueue>> = Mutex::new(None);

/// lab: `XCL_SPARSE_COOL=1`: a pool tile a buffer gives back is used again only after the work
/// submitted before on the sparse queue and every known title queue is done (the quarantine of
/// parked tiles, #2345), not at once. nvidia-dev's cooldown (`9554f33e`, `76969bc1`: freed tiles
/// wait 2 s) covered every freed tile; here the parked ones have the quarantine already.
fn sparse_cool() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPARSE_COOL").is_ok_and(|v| v.trim() == "1"))
}

/// One copy back at a time (`readback_released`): the readback buffer and its list are in use
/// until the copy is done, and that is waited for with `STATE` released. Taken before `STATE`,
/// never under it.
static READBACK: Mutex<()> = Mutex::new(());

/// lab `XCL_SPARSE_MAP_NOWAIT=1` (off by default): 0 not read yet, 1 off, 2 on
static MAP_NOWAIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// lab `XCL_SPARSE_MAP_NOWAIT=1`: a title's Map of sparse memory (`readback_now`) copies back
/// on a copy queue of its own, ordered after the state's queue on the GPU (`map_copy_released`),
/// instead of on the state's queue, whose copy waits for the GPU's 3D engine (the title's lists)
/// to get to it; and a Map of a range none of whose tiles has video memory copies nothing back.
/// Not a skip of the copy when the asynchronous copies back cover the range: those only cover
/// what the title's copies, fills and value writes wrote (`list_x` note_write), not its shaders'
/// writes, which only the Map's copy brings back.
pub(super) fn map_nowait() -> bool {
    match MAP_NOWAIT.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("XCL_SPARSE_MAP_NOWAIT").is_ok_and(|v| v == "1");
            MAP_NOWAIT.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            if on {
                crate::xlog_info!("lab XCL_SPARSE_MAP_NOWAIT=1: a Map's copy back runs on a copy queue ordered after the sparse queue; a Map of a range without video memory copies nothing");
            }
            on
        }
        v => v == 2,
    }
}

/// tests: the switch as `on` says (None: read from the environment again)
#[cfg(test)]
fn set_map_nowait(on: Option<bool>) {
    MAP_NOWAIT.store(
        match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

/// lab `XCL_SPARSE_MAP_NOWAIT=1`: the copy queue of a Map's copy back, its fence and last
/// value, its list and its readback buffer (`READBACK_STAGING` bytes, mapped at `ptr`), for the
/// device `dev`. Made with the device's original creators (`real::pc_vtable`): the queue is not
/// adopted, so its submits do not go through the title's ExecuteCommandLists hook nor into
/// `list_x`'s queues (a title DMA list without a queue must never be given this one).
struct MapCopy {
    dev: usize,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    buf: ID3D12Resource,
    ptr: *mut u8,
}

unsafe impl Send for MapCopy {}

/// The Map copy queue (`MapCopy`), made at the first Map with the switch on. Taken right after
/// `READBACK`, before `STATE`.
static MAP_COPY: Mutex<Option<MapCopy>> = Mutex::new(None);

impl MapCopy {
    unsafe fn new(dev: &ID3D12Device) -> Option<MapCopy> {
        let d = dev.as_raw();
        let v = &*(super::real::pc_vtable(d) as *const ID3D12Device_Vtbl);
        let desc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() };
        let mut p: *mut c_void = std::ptr::null_mut();
        if (v.CreateCommandQueue)(d, &desc, &ID3D12CommandQueue::IID, &mut p).is_err() || p.is_null() {
            return None;
        }
        let queue = ID3D12CommandQueue::from_raw(p);
        let mut p: *mut c_void = std::ptr::null_mut();
        if (v.CreateFence)(d, 0, D3D12_FENCE_FLAG_NONE, &ID3D12Fence::IID, &mut p).is_err() || p.is_null() {
            return None;
        }
        let fence = ID3D12Fence::from_raw(p);
        let mut p: *mut c_void = std::ptr::null_mut();
        if (v.CreateCommandAllocator)(d, D3D12_COMMAND_LIST_TYPE_COPY, &ID3D12CommandAllocator::IID, &mut p).is_err() || p.is_null() {
            return None;
        }
        let alloc = ID3D12CommandAllocator::from_raw(p);
        let mut p: *mut c_void = std::ptr::null_mut();
        if (v.CreateCommandList)(d, 0, D3D12_COMMAND_LIST_TYPE_COPY, alloc.as_raw(), std::ptr::null_mut(), &ID3D12GraphicsCommandList::IID, &mut p).is_err() || p.is_null() {
            return None;
        }
        let list = ID3D12GraphicsCommandList::from_raw(p);
        list.Close().ok()?;
        let _ = queue.SetName(windows::core::w!("xcl sparse map copy queue"));
        let _ = list.SetName(windows::core::w!("xcl sparse map copy list"));
        // (the readback buffer as `readback_buf` makes the state's, with the original creator too)
        let mut bd = buffer_desc(READBACK_STAGING as u64);
        bd.Flags = D3D12_RESOURCE_FLAG_NONE;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let mut p: *mut c_void = std::ptr::null_mut();
        if (v.CreateCommittedResource)(d, &heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, std::ptr::null(), &ID3D12Resource::IID, &mut p).is_err() || p.is_null() {
            return None;
        }
        let buf = ID3D12Resource::from_raw(p);
        let mut mp: *mut c_void = std::ptr::null_mut();
        super::call_stats::cs!(Map, buf.Map(0, None, Some(&mut mp))).ok()?;
        if mp.is_null() {
            return None;
        }
        Some(MapCopy { dev: d as usize, queue, fence, value: 0, alloc, list, buf, ptr: mp as *mut u8 })
    }
}

/// Wait, 5 s at most, until `fence` has reached `target` (true: it has; a removed device counts).
/// Not for a holder of `STATE`: `wait_released`.
unsafe fn wait_fence(fence: &ID3D12Fence, target: u64) -> bool {
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    thread_local! {
        /// this thread's event for its waits (a wait given up leaves it registered with the
        /// fence: it is never closed, and a wait looks at the fence again when it is set)
        static EVENT: HANDLE = unsafe { CreateEventW(None, false, false, None) }.unwrap_or_default();
    }
    let event = EVENT.with(|e| *e);
    let start = std::time::Instant::now();
    loop {
        let at = super::real::fence_completed(fence);
        if at >= target {
            return true;
        }
        let left = 5000u32.saturating_sub(start.elapsed().as_millis() as u32);
        if left == 0 {
            crate::xlog_warn!("sparse: the queue has not reached fence value {target} after 5 s (at {at})");
            return false;
        }
        if !event.is_invalid() && super::call_stats::cs!(FenceSetEvent, fence.SetEventOnCompletion(target, event)).is_ok() {
            super::call_stats::cs!(EventWait, WaitForSingleObject(event, left));
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

/// Wait (`wait_fence`) until the state's queue has reached fence `value` with `STATE` released,
/// and take it again. The queue can stand still for as long as the GPU does, and the title's
/// other threads need the lock meanwhile – a submit, a commit, a `Map` (a retail title: a `Map`
/// waited 15 s for its copy back holding it, then a submit for room in the upload ring until the
/// title's watchdog ended it; a thread moving physical pages and the next `Map` stood at the
/// lock all that time). True: the queue of the same state got there. Anything the caller knew
/// of the state has to be looked at again.
unsafe fn wait_released(g: MutexGuard<'static, Option<State>>, value: u64) -> (MutexGuard<'static, Option<State>>, bool) {
    let Some(fence) = g.as_ref().map(|s| s.fence.clone()) else { return (g, false) };
    drop(g);
    let reached = wait_fence(&fence, value);
    let g = lock_state();
    let same = g.as_ref().is_some_and(|s| s.fence.as_raw() == fence.as_raw());
    (g, reached && same)
}

/// `upload_room(least)` of the state, waited for with `STATE` released (`wait_released`) – until
/// there is room if `patient`, else once. None: there is no state, or no room after that wait.
unsafe fn room_released(mut g: MutexGuard<'static, Option<State>>, least: usize, patient: bool) -> (MutexGuard<'static, Option<State>>, Option<usize>) {
    loop {
        let value = match g.as_mut().map(|s| s.upload_room(least)) {
            None => return (g, None),
            Some(Ok(room)) => return (g, Some(room)),
            Some(Err(value)) => value,
        };
        let (again, there) = wait_released(g, value);
        g = again;
        if !there && !patient {
            return (g, None);
        }
    }
}

/// Copy `ranges` (address, bytes) from sparse VRAM back into the CPU view, a readback buffer's
/// worth at a time: submitted, waited for with `STATE` released, and copied into the pages that
/// are still committed then. If the queue is not there after 5 s the CPU's view stays as it is
/// (the buffer does not hold the copy). The caller holds `READBACK`.
unsafe fn readback_released(mut g: MutexGuard<'static, Option<State>>, ranges: &[(usize, usize)]) {
    for &(a, n) in ranges {
        let mut at = a;
        while at < a + n {
            let Some(s) = g.as_mut() else { return };
            // (a copy given up earlier: its list and the buffer are still in use)
            let last = s.rb_last_fence;
            if last != 0 && super::real::fence_completed(&s.fence) < last {
                let (again, there) = wait_released(g, last);
                g = again;
                if !there {
                    return;
                }
                continue;
            }
            let Some((rb, p)) = s.readback_buf() else { return };
            let (next, parts) = s.readback_submit(&rb, at, a + n);
            at = next;
            if parts.is_empty() {
                continue;
            }
            let value = s.value;
            // lab (`readback_guard`): the CPU's view as the copy was submitted
            let before = readback_guard().then(|| chunk_hashes(&parts, |pg| s.page_committed(pg)));
            let (again, there) = wait_released(g, value);
            g = again;
            if !there {
                return;
            }
            if let Some(s) = g.as_mut() {
                match &before {
                    Some(before) => s.copy_back_unchanged(p, &parts, before),
                    None => s.copy_back(p, &parts),
                }
            }
        }
    }
}

/// (bytes of video memory in pool heaps, of those mapped) – `XCL_ACTIVITY`
pub fn pool_bytes() -> (usize, usize) {
    let g = lock_state();
    let Some(s) = g.as_ref() else { return (0, 0) };
    let pools = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize * TILE;
    (pools, pools.saturating_sub(s.free.len() * TILE))
}
/// commits before the device existed: the pages committed, as disjoint page ranges (start →
/// end) merged with their neighbours, so a commit or decommit costs O(log n) and not a scan of
/// every page committed so far. Like `committed` / `decommitted` with a state, a range counts
/// every page it touches; `init` commits them once the windows exist.
static EARLY: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// The pages `[a, a + n)` touches, as `[start, end)`.
fn page_span(a: usize, n: usize) -> (usize, usize) {
    (a & !(PAGE - 1), a.saturating_add(n).saturating_add(PAGE - 1) & !(PAGE - 1))
}

/// Add the pages `[a, b)` to `early`, merged with the ranges they overlap or touch.
fn early_add(early: &mut BTreeMap<usize, usize>, mut a: usize, mut b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..=a).next_back() {
        if e >= a {
            a = s;
            b = b.max(e);
            early.remove(&s);
        }
    }
    loop {
        let Some((s, e)) = early.range(a..=b).next().map(|(&s, &e)| (s, e)) else { break };
        b = b.max(e);
        early.remove(&s);
    }
    early.insert(a, b);
}

/// Take the pages `[a, b)` out of `early`; a range that overlaps it keeps the rest.
fn early_remove(early: &mut BTreeMap<usize, usize>, a: usize, b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..a).next_back() {
        if e > a {
            early.insert(s, a);
            if e > b {
                early.insert(b, e);
                return;
            }
        }
    }
    loop {
        let Some((s, e)) = early.range(a..b).next().map(|(&s, &e)| (s, e)) else { break };
        early.remove(&s);
        if e > b {
            early.insert(b, e);
            break;
        }
    }
}
/// generation counter incremented whenever the set of mapped tiles changes (for watch_ranges cache)
static MAPPED_GEN: AtomicU64 = AtomicU64::new(0);

/// Generation counter of `mapped_ranges()` (changes only when tiles are mapped or unmapped).
pub fn mapped_gen() -> u64 {
    MAPPED_GEN.load(Ordering::Acquire)
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

unsafe fn new_state(dev: &ID3D12Device, windows: Vec<(usize, ID3D12Resource)>) -> windows::core::Result<State> {
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
    let rbq = (|| -> windows::core::Result<(ID3D12CommandQueue, ID3D12Fence)> {
        let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let _ = q.SetName(windows::core::w!("xcl sparse readback queue"));
        Ok((q, dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?))
    })()
    .ok();
    let mut slots = Vec::with_capacity(UPLOAD_SLOTS);
    for _ in 0..UPLOAD_SLOTS {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.Close()?;
        let _ = list.SetName(windows::core::w!("xcl sparse upload list"));
        let _ = alloc.SetName(windows::core::w!("xcl sparse upload allocator"));
        slots.push(UploadSlot { alloc, list, fence_val: 0 });
    }
    let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
    let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None)?;
    rb_list.Close()?;
    let _ = rb_list.SetName(windows::core::w!("xcl sparse readback list"));
    let _ = rb_alloc.SetName(windows::core::w!("xcl sparse readback allocator"));
    let _ = queue.SetName(windows::core::w!("xcl sparse queue"));
    Ok(State {
        dev: dev.clone(),
        queue,
        fence,
        value: 0,
        event,
        rb_fence,
        rb_value: 0,
        rb_queues: FxHashMap::default(),
        rbq,
        rbq_value: 0,
        rb_jobs: VecDeque::new(),
        rb_pool: Vec::new(),
        windows,
        pools: Vec::new(),
        free: Vec::new(),
        retired: Vec::new(),
        mapped: FxHashMap::default(),
        pinned: FxHashMap::default(),
        parked: FxHashMap::default(),
        kept: FxHashMap::default(),
        quarantined: VecDeque::new(),
        released: Vec::new(),
        queue_fences: FxHashMap::default(),
        buffers: FxHashMap::default(),
        to_zero: Vec::new(),
        staging: None,
        ring_write: 0,
        ring_submitted: 0,
        ring_free: 0,
        inflight: VecDeque::new(),
        zero_buf: None,
        slots,
        next_slot: 0,
        readback: None,
        rb_alloc,
        rb_list,
        rb_last_fence: 0,
        waited: FxHashMap::default(),
        runs: None,
        scratch_runs: Vec::new(),
        lazy_bufs: Vec::new(),
    })
}

/// pool heap number of the tiles of a shadowed window: they are the window's own buffer, not
/// pool tiles (never mapped, freed or pinned)
const SHADOW_POOL: u32 = u32::MAX;

/// A committed window the GPU could not open as a heap (`real::open_uma`): `buf` in video
/// memory stands for `[base, base + size)` and is kept in step like the tiles of a sparse
/// window -- the CPU's writes (write-watch, changed 64-byte chunks) go to it before each
/// ExecuteCommandLists, GPU writes come back on Map and after the lists' memory operations.
/// All of it counts as mapped from the start: the window is committed. False: no queue for
/// the copies (the buffer then never sees the CPU's writes).
pub unsafe fn shadow_window(dev: &ID3D12Device, base: usize, size: usize, buf: &ID3D12Resource) -> bool {
    let mut g = lock_state();
    if g.is_none() {
        // no sparse window was made (`init`): the copies still need the queue and the lists
        match new_state(dev, Vec::new()) {
            Ok(s) => *g = Some(s),
            Err(e) => {
                crate::xlog_warn!("shadowed window at {base:#x}: {e}");
                return false;
            }
        }
    }
    let Some(s) = g.as_mut() else { return false };
    s.windows.push((base, buf.clone()));
    for (i, t) in (base..base + size).step_by(TILE).enumerate() {
        s.mapped.insert(t, TileEntry::with_holes((SHADOW_POOL, i as u32), 0));
    }
    s.runs = None;
    MAPPED_GEN.fetch_add(1, Ordering::Release);
    true
}

/// lab (`XCL_SHADOW_TILED=1`): the shadowed committed window `[lo, hi)` whose copy in video
/// memory is a reserved buffer with tiles mapped on demand (0, 0: none)
static LAZY: [std::sync::atomic::AtomicUsize; 2] = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];

/// lab: the window of `shadow_window_tiled`, if any.
pub fn lazy_range() -> Option<(usize, usize)> {
    let (lo, hi) = (LAZY[0].load(Ordering::Acquire), LAZY[1].load(Ordering::Acquire));
    (hi > lo).then_some((lo, hi))
}

/// lab: whether `va` is in the window of `shadow_window_tiled`.
pub fn is_lazy(va: usize) -> bool {
    lazy_range().is_some_and(|(lo, hi)| va >= lo && va < hi)
}

/// lab: `shadow_window` with a reserved buffer `buf` and no video memory to begin with: a
/// tile is mapped (a pool tile, as in the sparse windows) when the CPU first writes one of its
/// pages (`before_execute`) or the title places a buffer over it (`place_buffer`). The rest -
/// a retail title's render targets are 3.4 GB of this window's addresses and have pages of
/// their own (`vram`) - takes none. Memory only shaders write, with no buffer placed there,
/// is not kept.
pub unsafe fn shadow_window_tiled(dev: &ID3D12Device, base: usize, size: usize, buf: &ID3D12Resource) -> bool {
    let mut g = lock_state();
    if g.is_none() {
        match new_state(dev, Vec::new()) {
            Ok(s) => *g = Some(s),
            Err(e) => {
                crate::xlog_warn!("shadowed window at {base:#x}: {e}");
                return false;
            }
        }
    }
    let Some(s) = g.as_mut() else { return false };
    s.windows.push((base, buf.clone()));
    LAZY[0].store(base, Ordering::Release);
    LAZY[1].store(base + size, Ordering::Release);
    s.runs = None;
    MAPPED_GEN.fetch_add(1, Ordering::Release);
    true
}

/// The sparse windows' buffers on `dev`: `(window_index, buffer_raw, gpu_va)` for each
/// sparse window in `gpu::uma` (empty: no sparse windows).
pub unsafe fn init(dev: &ID3D12Device) -> Vec<(usize, usize, u64)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let mut out = Vec::new();
    let mut windows = Vec::new();
    for k in 0..u.shader_windows() {
        if !u.is_sparse_window(k) {
            continue;
        }
        let (base, size) = u.window(k);
        let mut r: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&buffer_desc(size as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r) {
            crate::xlog_warn!("sparse window {k}: CreateReservedResource: {e}");
            return Vec::new();
        }
        let Some(r) = r else { return Vec::new() };
        out.push((k, r.clone().into_raw() as usize, r.GetGPUVirtualAddress()));
        windows.push((base, r));
    }
    if out.is_empty() {
        return out;
    }
    let made = new_state(dev, windows);
    match made {
        Ok(s) => {
            crate::xlog_info!("D3D12.X: {} sparse window(s) for the title's own memory (tiled buffers on video memory)", out.len());
            // a state this replaces (the windows set up again) goes after the lock: the release of
            // its last buffers reaches `forget_buffer`, which takes the lock (#344)
            // (the queue first: `queue()` never answers with one older than the state's)
            let old_queue = QUEUE.lock().unwrap().replace(s.queue.clone());
            let old = STATE.lock().unwrap().replace(s);
            drop(old);
            drop(old_queue);
            for (a, b) in std::mem::take(&mut *EARLY.lock().unwrap()) {
                committed(a, b - a);
            }
            out
        }
        Err(e) => {
            crate::xlog_warn!("sparse windows: {e}");
            Vec::new()
        }
    }
}

impl State {
    #[inline]
    fn window_index(&self, a: usize) -> Option<usize> {
        self.windows.iter().position(|(b, _)| a >= *b && a < *b + crate::gpu::uma::WINDOW)
    }

    unsafe fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        self.drain_quarantined();
        if self.free.is_empty() && self.flush_parked() {
            self.drain_quarantined();
        }
        if let Some(t) = self.free.pop() {
            return Some((t, true));
        }
        // a pool retired a moment ago (its last tile came free) is taken back before a new
        // 64 MB heap is made: freeing and allocating a tile in turn made a new pool each time,
        // up to 18 GB of retired heaps in 2 s (#913). Its tiles are reused as freed tiles are
        // (zeroed), so this is no less safe than reusing a tile of a pool in use.
        if let Some((heap, _, _)) = self.retired.pop() {
            let h = self.add_pool(heap);
            return Some(((h, 0), true));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_TILES as u64 * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| self.dev.CreateHeap(&desc, &mut heap)) {
            crate::xlog_warn!("sparse windows: no video memory for more tiles ({e})");
            return None;
        }
        let heap = heap?;
        let h = self.add_pool(heap);
        Some(((h, 0), false))
    }

    /// `heap` becomes a pool (in a given-back pool's index if there is one); its tiles but the
    /// first, which the caller takes, go to the free list.
    fn add_pool(&mut self, heap: ID3D12Heap) -> u32 {
        let h = match self.pools.iter().position(|p| p.is_none()) {
            Some(i) => {
                self.pools[i] = Some(heap);
                i as u32
            }
            None => {
                self.pools.push(Some(heap));
                self.pools.len() as u32 - 1
            }
        };
        self.free.extend((1..POOL_TILES).rev().map(|t| (h, t)));
        h
    }

    /// Pools whose every tile is free again go (64 MB of video memory each – and as much of the
    /// system's commit: a retail game's homes of physical pages left 0.75 GB of pools for
    /// 0.29 GB of tiles in use). Call after their tiles are unmapped and signalled.
    fn retire_free_pools(&mut self) {
        let now = std::time::Instant::now();
        let done = unsafe { super::real::fence_completed(&self.fence) };
        self.retired.retain(|(_, v, at)| !(done >= *v && now.duration_since(*at).as_secs() >= 2));
        if self.free.len() < POOL_TILES as usize {
            return;
        }
        let mut count = vec![0u32; self.pools.len()];
        for &(h, _) in &self.free {
            count[h as usize] += 1;
        }
        let gone: Vec<usize> = (0..count.len()).filter(|&h| count[h] == POOL_TILES && self.pools[h].is_some()).collect();
        if gone.is_empty() {
            return;
        }
        self.free.retain(|&(h, _)| !gone.contains(&(h as usize)));
        for h in gone {
            if let Some(heap) = self.pools[h].take() {
                self.retired.push((heap, self.value, now));
            }
        }
    }

    /// Map `tiles` (tile addresses) of `res` (starting at `res_base`) onto pool tiles,
    /// batched per pool heap with contiguous runs merged into single regions.
    unsafe fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        if tiles.is_empty() {
            return;
        }
        super::real::removed_probe("sparse: before map_tiles");
        let mut by_heap: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        for &(a, (h, t)) in tiles {
            by_heap.entry(h).or_default().push((((a - res_base) / TILE) as u32, t));
        }
        for (h, mut pairs) in by_heap {
            pairs.sort_unstable_by_key(|&(x, _)| x);
            let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(pairs.len());
            let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(pairs.len());
            let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(pairs.len());
            let mut offsets: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut counts: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut i = 0;
            while i < pairs.len() {
                let (x0, t0) = pairs[i];
                let mut run = 1u32;
                while i + (run as usize) < pairs.len() {
                    let (nx, nt) = pairs[i + run as usize];
                    if nx == x0 + run && nt == t0 + run {
                        run += 1;
                    } else {
                        break;
                    }
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
                regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                offsets.push(t0);
                counts.push(run);
                i += run as usize;
            }
            super::call_stats::cs!(UpdateTileMappings, self.queue.UpdateTileMappings(
                res,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                self.pools[h as usize].as_ref().expect("a pool tile of a live pool"),
                flags.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            ));
        }
        super::real::removed_probe("sparse: after map_tiles");
    }

    unsafe fn unmap_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[usize]) {
        if tiles.is_empty() {
            return;
        }
        super::real::removed_probe("sparse: before unmap_tiles");
        let mut xs: Vec<u32> = tiles.iter().map(|&a| ((a - res_base) / TILE) as u32).collect();
        xs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(xs.len());
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(xs.len());
        let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(xs.len());
        let mut counts: Vec<u32> = Vec::with_capacity(xs.len());
        let mut i = 0;
        while i < xs.len() {
            let x0 = xs[i];
            let mut run = 1u32;
            while i + (run as usize) < xs.len() && xs[i + run as usize] == x0 + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            flags.push(D3D12_TILE_RANGE_FLAG_NULL);
            counts.push(run);
            i += run as usize;
        }
        super::call_stats::cs!(UpdateTileMappings, self.queue.UpdateTileMappings(
            res,
            coords.len() as u32,
            Some(coords.as_ptr()),
            Some(regions.as_ptr()),
            None,
            flags.len() as u32,
            Some(flags.as_ptr()),
            None,
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        ));
    }

    /// tiles of `[a, a + n)` get video memory (mapped in their window); `initial_holes` supplies
    /// the initial 16-bit hole mask for each newly mapped 64 KiB tile.
    unsafe fn ensure_mapped(&mut self, a: usize, n: usize, mut initial_holes: impl FnMut(usize) -> u16) {
        if n == 0 {
            return;
        }
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        let mut new: Vec<(usize, (u32, u32))> = Vec::new();
        let mut unparked = false;
        let mut t = first;
        while t < end {
            if !self.mapped.contains_key(&t) && self.window_index(t).is_some() {
                // still mapped there: no new mapping, what it held is cleared
                if let Some(tile) = self.parked.remove(&t).or_else(|| self.kept.remove(&t)) {
                    self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                    self.runs = None;
                    self.to_zero.push((t, TILE));
                    unparked = true;
                } else {
                    let Some((tile, reused)) = self.take_tile() else { break };
                    self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                    self.runs = None;
                    if reused {
                        self.to_zero.push((t, TILE));
                    }
                    new.push((t, tile));
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new.is_empty() {
            if unparked {
                MAPPED_GEN.fetch_add(1, Ordering::Release);
            }
            return;
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        // per window
        let mut by_window: FxHashMap<usize, Vec<(usize, (u32, u32))>> = FxHashMap::default();
        for x in new {
            if let Some(wi) = self.window_index(x.0) {
                by_window.entry(wi).or_default().push(x);
            }
        }
        for (wi, tiles) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.map_tiles(&r, b, &tiles);
        }
        self.signal();
    }

    #[inline]
    unsafe fn signal(&mut self) {
        self.value += 1;
        let _ = super::call_stats::cs!(QueueSignal, self.queue.Signal(&self.fence, self.value));
    }

    #[inline]
    unsafe fn wait_for(&self, target: u64) {
        if target == 0 || super::real::fence_completed(&self.fence) >= target {
            return;
        }
        // The event is one for all waits here, and a wait that gave up leaves its registration
        // behind: that one sets the event later, and the next wait would return at once with the
        // queue not there (10 such "after 5 s" lines within 100 ms in one run, each followed by a
        // Reset of an allocator the GPU was still in). So: until the fence says so, or 5 s.
        let t0 = std::time::Instant::now();
        loop {
            let _ = super::call_stats::cs!(FenceSetEvent, self.fence.SetEventOnCompletion(target, self.event));
            let left = 5000u64.saturating_sub(t0.elapsed().as_millis() as u64).max(1) as u32;
            super::call_stats::cs!(EventWait, windows::Win32::System::Threading::WaitForSingleObject(self.event, left));
            if super::real::fence_completed(&self.fence) >= target {
                return;
            }
            if t0.elapsed().as_millis() >= 5000 {
                break;
            }
        }
        crate::xlog_warn!("sparse: the queue has not reached fence value {target} after 5 s (at {})", super::real::fence_completed(&self.fence));
    }

    #[inline]
    unsafe fn reclaim_ring(&mut self) {
        if self.inflight.is_empty() {
            return;
        }
        let completed = super::real::fence_completed(&self.fence);
        while let Some(&(fv, end_pos)) = self.inflight.front() {
            if fv <= completed {
                self.ring_free = end_pos;
                self.inflight.pop_front();
            } else {
                break;
            }
        }
    }

    /// The bytes of CPU pages (`least` or more; 0: only tile clears, which take none of the ring)
    /// an `upload_zeros` and an `upload_dirty` can take now without waiting for the GPU: their
    /// two command slots are free and the ring has that much room. Err: the fence value to wait
    /// for first (`room_released`, not under the lock). Callers ask before each part of their
    /// uploads, so the waits in `begin_slot` and `upload_dirty`, which hold the lock, are not met.
    unsafe fn upload_room(&mut self, least: usize) -> Result<usize, u64> {
        if least == 0 && self.to_zero.is_empty() {
            return Ok(0);
        }
        self.reclaim_ring();
        let completed = super::real::fence_completed(&self.fence);
        for k in 0..self.slots.len().min(2) {
            let fv = self.slots[(self.next_slot + k) % self.slots.len()].fence_val;
            if fv > completed {
                return Err(fv);
            }
        }
        // (a chunk that does not fit before the ring's end skips less than a page of it)
        let room = (STAGING - (self.ring_write - self.ring_free) as usize).saturating_sub(PAGE);
        match self.inflight.front() {
            Some(&(fv, _)) if least > 0 && room < least.max(RING_ROOM) => Err(fv),
            _ => Ok(room),
        }
    }

    /// The tiles and pages waiting to be cleared (`to_zero`) are, before CPU writes into them
    /// are uploaded (overlapping CopyBufferRegion destinations: no WAW hazard). Returns them.
    unsafe fn flush_zeros(&mut self) -> Vec<(usize, usize)> {
        let zeroed = std::mem::take(&mut self.to_zero);
        for &(a, n) in &zeroed {
            self.clear_sent_range(a, n);
        }
        if !zeroed.is_empty() {
            self.upload_zeros(&zeroed);
        }
        zeroed
    }

    /// Acquire the next upload command slot, waiting only if that slot's previous submission is
    /// still executing on the GPU, then reset its allocator and list.
    unsafe fn begin_slot(&mut self) -> usize {
        let idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let fv = self.slots[idx].fence_val;
        if fv > 0 && super::real::fence_completed(&self.fence) < fv {
            self.wait_for(fv);
            self.reclaim_ring();
        }
        let slot = &self.slots[idx];
        let _ = slot.alloc.Reset();
        let _ = slot.list.Reset(&slot.alloc, None);
        crate::hle::d3d12x::gpu_profile::own_list_begin(slot.list.as_raw());
        idx
    }

    /// Close and submit upload slot `idx` asynchronously (NO CPU wait — callers order GPU work
    /// via `queue.Wait(&self.fence, self.value)`).
    unsafe fn submit_slot(&mut self, idx: usize) {
        crate::hle::d3d12x::gpu_profile::own_list_end(self.slots[idx].list.as_raw(), crate::hle::d3d12x::gpu_profile::Kind::Upload);
        let closed = self.slots[idx].list.Close().is_ok();
        if closed {
            if let Ok(l) = self.slots[idx].list.cast::<ID3D12CommandList>() {
                super::title_textures::own_work(|| {
                    self.queue.ExecuteCommandLists(&[Some(l)]);
                });
            }
        }
        self.signal();
        self.slots[idx].fence_val = self.value;
        if self.ring_write > self.ring_submitted {
            self.inflight.push_back((self.value, self.ring_write));
            self.ring_submitted = self.ring_write;
        }
    }

    unsafe fn staging_ptr(&mut self) -> Option<*mut u8> {
        if self.staging.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            super::call_stats::cs!(Map, r.Map(0, None, Some(&mut p))).ok()?;
            self.staging = Some((r, p as *mut u8));
        }
        self.staging.as_ref().map(|&(_, p)| p)
    }

    unsafe fn ensure_zero_buf(&mut self) -> bool {
        if self.zero_buf.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(ZERO_BUF as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            if self
                .dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .is_err()
            {
                return false;
            }
            let Some(r) = r else { return false };
            let mut p = std::ptr::null_mut();
            if super::call_stats::cs!(Map, r.Map(0, None, Some(&mut p))).is_err() {
                return false;
            }
            std::ptr::write_bytes(p as *mut u8, 0, ZERO_BUF);
            super::call_stats::cs!(Unmap, r.Unmap(0, None));
            self.zero_buf = Some(r);
        }
        true
    }

    unsafe fn readback_buf(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.readback.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(READBACK_STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            super::call_stats::cs!(Map, r.Map(0, None, Some(&mut p))).ok()?;
            self.readback = Some((r, p as *mut u8));
        }
        self.readback.clone()
    }

    /// Clear `sent` hashes for every page overlapping `[a, a + n)`.
    fn clear_sent_range(&mut self, a: usize, n: usize) {
        let mut pg = a & !(PAGE - 1);
        let end = a + n;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.sent[(p - t) / PAGE] = None;
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Update `sent` hashes for all 64-byte chunks overlapping `[va, va + len)` so the CPU
    /// copy-back we just wrote to `va` is not re-uploaded by the next `before_execute`.
    unsafe fn sync_sent_range(&mut self, va: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut pg = va & !(PAGE - 1);
        let end = va + len;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    let idx = (p - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                        let sent = entry.sent[idx].get_or_insert_with(|| Box::new(ZERO_HASHES));
                        let lo = va.saturating_sub(p) / CHUNK;
                        let hi = (end.min(p + PAGE) - p).div_ceil(CHUNK).min(CHUNKS);
                        for k in lo..hi {
                            sent[k] = chunk_hash(&bytes[k * CHUNK..(k + 1) * CHUNK]);
                        }
                    }
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Whether page `pg` is mapped here and committed (not a hole of its tile).
    fn page_committed(&self, pg: usize) -> bool {
        let t = pg & !(TILE - 1);
        self.mapped.get(&t).is_some_and(|e| e.holes & (1u16 << ((pg - t) / PAGE)) == 0)
    }

    /// lab (`readback_guard`): `copy_back` of the chunks the CPU did not write since `before`
    /// (`chunk_hashes` at the submit); `sent` hashes follow the chunks landed only.
    unsafe fn copy_back_unchanged(&mut self, p: *const u8, parts: &[(usize, usize, usize)], before: &[u64]) {
        let (landed, kept) = land_unchanged(p, parts, before, |pg| self.page_committed(pg));
        for (c, h) in landed {
            let pg = c & !(PAGE - 1);
            let t = pg & !(TILE - 1);
            if let Some(e) = self.mapped.get_mut(&t) {
                e.sent[(pg - t) / PAGE].get_or_insert_with(|| Box::new(ZERO_HASHES))[(c - pg) / CHUNK] = h;
            }
        }
        if kept > 0 {
            let n = GUARD_KEPT.fetch_add(kept as u64, Ordering::Relaxed) + kept as u64;
            if n == kept as u64 || (n - kept as u64).leading_zeros() != n.leading_zeros() {
                crate::xlog_info!("sparse: a copy back kept {kept} chunks the CPU wrote while it ran ({n} so far; XCL_READBACK_GUARD=1)");
            }
        }
    }

    /// `parts` (address, offset at `p`, bytes) read back from video memory into the CPU view
    /// (its committed pages), `sent` hashes following.
    unsafe fn copy_back(&mut self, p: *const u8, parts: &[(usize, usize, usize)]) {
        for &(dst, off, len) in parts {
            let mut at = dst;
            let mut src_off = off;
            let end = dst + len;
            while at < end {
                let pg = at & !(PAGE - 1);
                let t = pg & !(TILE - 1);
                let idx = (pg - t) / PAGE;
                let step = (pg + PAGE).min(end) - at;
                if self.mapped.get(&t).is_some_and(|e| e.holes & (1u16 << idx) == 0) {
                    std::ptr::copy_nonoverlapping(p.add(src_off), at as *mut u8, step);
                }
                at += step;
                src_off += step;
            }
            self.sync_sent_range(dst, len);
        }
    }

    /// A copy of what `[from, end)` has in sparse VRAM into the readback buffer `rb`, as much as
    /// fits, submitted on the state's queue and signalled (`self.value`). Returns where it
    /// stopped and the parts copied (address, offset in `rb`, bytes), for `copy_back` once the
    /// queue is there (`readback_released`). The copy before it is done (`rb_last_fence`): its
    /// list is used again. Caller has already ordered `self.queue` after any GPU writes if needed.
    unsafe fn readback_submit(&mut self, rb: &ID3D12Resource, from: usize, end: usize) -> (usize, Vec<(usize, usize, usize)>) {
        let mut parts: Vec<(usize, usize, usize)> = Vec::new();
        let mut used = 0usize;
        let mut cur = from;
        while cur < end && used < READBACK_STAGING {
            let tile = cur & !(TILE - 1);
            let Some(wi) = self.window_index(cur) else {
                cur = end;
                break;
            };
            let b = self.windows[wi].0;
            let to_tile_end = (tile + TILE) - cur;
            if !self.mapped.contains_key(&tile) {
                cur += (end - cur).min(to_tile_end);
                continue;
            }
            let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
            let chunk = (end - cur).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
            if parts.is_empty() {
                let _ = self.rb_alloc.Reset();
                let _ = self.rb_list.Reset(&self.rb_alloc, None);
                crate::hle::d3d12x::gpu_profile::own_list_begin(self.rb_list.as_raw());
            }
            super::call_stats::cs!(RecCopyBuffer, self.rb_list.CopyBufferRegion(rb, used as u64, &self.windows[wi].1, (cur - b) as u64, chunk as u64));
            parts.push((cur, used, chunk));
            used += chunk;
            cur += chunk;
        }
        if !parts.is_empty() {
            crate::hle::d3d12x::gpu_profile::own_list_end(self.rb_list.as_raw(), crate::hle::d3d12x::gpu_profile::Kind::Readback);
            if self.rb_list.Close().is_ok() {
                if let Ok(l) = self.rb_list.cast::<ID3D12CommandList>() {
                    super::title_textures::own_work(|| {
                        self.queue.ExecuteCommandLists(&[Some(l)]);
                    });
                }
            }
            self.signal();
            self.rb_last_fence = self.value;
        }
        (cur, parts)
    }

    /// Whether a tile of `[va, va + len)` has video memory (`mapped`)
    fn any_mapped(&self, va: usize, len: usize) -> bool {
        let end = va.saturating_add(len);
        let mut t = va & !(TILE - 1);
        while t < end {
            if self.mapped.contains_key(&t) {
                return true;
            }
            t += TILE;
        }
        false
    }

    /// `readback_submit` on the Map copy queue `m` (lab `XCL_SPARSE_MAP_NOWAIT=1`): what
    /// `[from, end)` has in sparse VRAM into `m`'s buffer, as much as fits, the same parts as
    /// `readback_submit` makes. The copy queue first waits, on the GPU, for a signal of the state's
    /// queue after all that was submitted to it so far (the Map's CPU writes uploaded, tiles mapped
    /// and cleared): the copy reads what the one on the state's queue would read, but does not wait
    /// behind the title's lists for the 3D engine. Later work on the state's queue is not ordered
    /// after it (nor needs to be: a copy back is a snapshot of VRAM after the Map was called).
    /// None: the copy queue could not be ordered after the state's queue (nothing submitted;
    /// the caller copies on the state's queue instead).
    unsafe fn map_copy_submit(&mut self, m: &mut MapCopy, from: usize, end: usize) -> Option<(usize, Vec<(usize, usize, usize)>)> {
        let mut parts: Vec<(usize, usize, usize)> = Vec::new();
        let mut used = 0usize;
        let mut cur = from;
        while cur < end && used < READBACK_STAGING {
            let tile = cur & !(TILE - 1);
            let Some(wi) = self.window_index(cur) else {
                cur = end;
                break;
            };
            let b = self.windows[wi].0;
            let to_tile_end = (tile + TILE) - cur;
            if !self.mapped.contains_key(&tile) {
                cur += (end - cur).min(to_tile_end);
                continue;
            }
            let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
            let chunk = (end - cur).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
            if parts.is_empty() {
                let _ = m.alloc.Reset();
                let _ = m.list.Reset(&m.alloc, None);
            }
            super::call_stats::cs!(RecCopyBuffer, m.list.CopyBufferRegion(&m.buf, used as u64, &self.windows[wi].1, (cur - b) as u64, chunk as u64));
            parts.push((cur, used, chunk));
            used += chunk;
            cur += chunk;
        }
        if parts.is_empty() {
            return Some((cur, parts));
        }
        let closed = m.list.Close().is_ok();
        // everything on the state's queue so far comes first
        self.signal();
        if !closed || super::call_stats::cs!(QueueWait, m.queue.Wait(&self.fence, self.value)).is_err() {
            return None;
        }
        let Ok(l) = m.list.cast::<ID3D12CommandList>() else { return None };
        m.queue.ExecuteCommandLists(&[Some(l)]);
        m.value += 1;
        let _ = super::call_stats::cs!(QueueSignal, m.queue.Signal(&m.fence, m.value));
        Some((cur, parts))
    }

    /// A copy back's list and a readback buffer of at least `need` bytes (reused when one fits)
    unsafe fn rb_job(&mut self, need: usize) -> Option<RbJob> {
        if let Some(i) = self.rb_pool.iter().enumerate().filter(|(_, j)| j.cap >= need).min_by_key(|(_, j)| j.cap).map(|(i, _)| i) {
            return Some(self.rb_pool.swap_remove(i));
        }
        let cap = need.max(64 << 10).next_power_of_two().min(READBACK_STAGING);
        let alloc: ID3D12CommandAllocator = self.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
        let list: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
        list.Close().ok()?;
        let _ = list.SetName(windows::core::w!("xcl sparse readback list"));
        let mut d = buffer_desc(cap as u64);
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        let mut r: Option<ID3D12Resource> = None;
        self.dev
            .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)
            .ok()?;
        let buf = r?;
        let mut p: *mut c_void = std::ptr::null_mut();
        super::call_stats::cs!(Map, buf.Map(0, None, Some(&mut p))).ok()?;
        Some(RbJob { alloc, list, buf, ptr: p as *mut u8, cap, rb_v: 0, done_v: 0, copies: Vec::new(), fences: None })
    }

    /// The fences a copy back waits on: its title queue's (reaches `rb_v` when the title's lists
    /// are done) and its copy queue's (reaches `done_v` when the copy is).
    fn rb_job_fences(&self, j: &RbJob) -> Option<(ID3D12Fence, ID3D12Fence)> {
        match &j.fences {
            Some(f) => Some(f.clone()),
            None => self.rbq.as_ref().map(|(_, done)| (self.rb_fence.clone(), done.clone())),
        }
    }

    /// Land submitted copies back in the title's memory as far as `how` says: every one that is
    /// done, oldest first (one title queue's in its order; a queue that waits does not hold back
    /// the copies of the others).
    unsafe fn drain_readbacks(&mut self, how: Drain) {
        if self.rbq.is_none() {
            return;
        }
        let mut i = 0;
        while i < self.rb_jobs.len() {
            let Some((title, done)) = self.rb_job_fences(&self.rb_jobs[i]) else { return };
            let (rb_v, done_v) = (self.rb_jobs[i].rb_v, self.rb_jobs[i].done_v);
            if super::real::fence_completed(&done) < done_v {
                if how == Drain::Finished || super::real::fence_completed(&title) < rb_v {
                    i += 1;
                    continue;
                }
                // the title's lists are done: only the copy itself is left
                // (`wait_fence`'s own event: `self.event` is the tile queue's, see `wait_for`)
                if !wait_fence(&done, done_v) {
                    crate::xlog_warn!("sparse: a copy back has not finished 5 s after the title's work it follows");
                    i += 1;
                    continue;
                }
            }
            let Some(mut job) = self.rb_jobs.remove(i) else { break };
            let copies = std::mem::take(&mut job.copies);
            self.copy_back(job.ptr, &copies);
            RB_PENDING.fetch_sub(1, Ordering::Release);
            job.copies = copies;
            job.copies.clear();
            self.rb_pool.push(job);
        }
    }

    /// The copy back queue of title queue `key` (made the first time) and the value its title
    /// fence gets next: (copy queue, title fence, value). None: it could not be made.
    unsafe fn rb_queue_for(&mut self, key: usize) -> Option<(ID3D12CommandQueue, ID3D12Fence, u64)> {
        if !self.rb_queues.contains_key(&key) {
            let title: ID3D12Fence = self.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let done: ID3D12Fence = self.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let queue: ID3D12CommandQueue = self.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
            let _ = queue.SetName(windows::core::w!("xcl sparse readback queue"));
            self.rb_queues.insert(key, RbQueue { title, title_v: 0, queue, done, done_v: 0 });
        }
        let rq = self.rb_queues.get_mut(&key)?;
        rq.title_v += 1;
        Some((rq.queue.clone(), rq.title.clone(), rq.title_v))
    }

    /// `readback_released` without a CPU wait: the copies run on the copy queue of title queue
    /// `key` (None: on `rbq`), after the title's queue signalled its fence (`rb_fence`) = `rb_v`,
    /// and land in the title's memory in `drain_readbacks`.
    unsafe fn readback_async(&mut self, ranges: &[(usize, usize)], key: Option<usize>, rb_v: u64) {
        let (q, title, done) = match key.and_then(|k| self.rb_queues.get(&k)) {
            Some(rq) => (rq.queue.clone(), Some(rq.title.clone()), rq.done.clone()),
            None => {
                let Some((q, done)) = self.rbq.clone() else { return };
                (q, None, done)
            }
        };
        // the mapped parts, in readback-buffer sized batches: (window, address, bytes)
        let mut batches: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new()];
        let mut used = 0usize;
        for &(a, n) in ranges {
            let mut at = 0usize;
            while at < n {
                let cur = a + at;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let to_tile_end = (tile + TILE) - cur;
                if !self.mapped.contains_key(&tile) {
                    at += (n - at).min(to_tile_end);
                    continue;
                }
                if used == READBACK_STAGING {
                    batches.push(Vec::new());
                    used = 0;
                }
                let to_win_end = (self.windows[wi].0 + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - at).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                batches.last_mut().expect("batch").push((wi, cur, chunk));
                used += chunk;
                at += chunk;
            }
        }
        for batch in batches.into_iter().filter(|b| !b.is_empty()) {
            let need = batch.iter().map(|c| c.2).sum::<usize>();
            let Some(mut job) = self.rb_job(need) else {
                crate::xlog_warn!("sparse: no readback buffer for a copy back of {need} bytes – left out");
                return;
            };
            let _ = job.alloc.Reset();
            let _ = job.list.Reset(&job.alloc, None);
            crate::hle::d3d12x::gpu_profile::own_list_begin(job.list.as_raw());
            let mut off = 0usize;
            for (wi, cur, chunk) in batch {
                let (base, res) = (self.windows[wi].0, self.windows[wi].1.clone());
                super::call_stats::cs!(RecCopyBuffer, job.list.CopyBufferRegion(&job.buf, off as u64, &res, (cur - base) as u64, chunk as u64));
                job.copies.push((cur, off, chunk));
                off += chunk;
            }
            crate::hle::d3d12x::gpu_profile::own_list_end(job.list.as_raw(), crate::hle::d3d12x::gpu_profile::Kind::Readback);
            if job.list.Close().is_ok() {
                if let Ok(l) = job.list.cast::<ID3D12CommandList>() {
                    super::title_textures::own_work(|| q.ExecuteCommandLists(&[Some(l)]));
                }
            }
            let done_v = match key.and_then(|k| self.rb_queues.get_mut(&k)) {
                Some(rq) => {
                    rq.done_v += 1;
                    rq.done_v
                }
                None => {
                    self.rbq_value += 1;
                    self.rbq_value
                }
            };
            let _ = super::call_stats::cs!(QueueSignal, q.Signal(&done, done_v));
            job.rb_v = rb_v;
            job.done_v = done_v;
            job.fences = title.clone().map(|t| (t, done.clone()));
            self.rb_jobs.push_back(job);
            RB_PENDING.fetch_add(1, Ordering::Release);
            RB_ASYNC.store(true, Ordering::Relaxed);
        }
    }

    /// Asynchronously zero `runs` (address, bytes) in sparse VRAM using the pre-zeroed `zero_buf`
    /// (zero staging ring bytes consumed, zero CPU stall).
    unsafe fn upload_zeros(&mut self, runs: &[(usize, usize)]) {
        if runs.is_empty() || !self.ensure_zero_buf() {
            return;
        }
        let slot_idx = self.begin_slot();
        let mut last_wi: Option<usize> = None;
        let zero_buf = self.zero_buf.as_ref().unwrap();
        let list = &self.slots[slot_idx].list;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(ZERO_BUF).min(to_win_end);
                super::call_stats::cs!(RecCopyBuffer, list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, zero_buf, 0, chunk as u64));
                done += chunk;
            }
        }
        self.submit_slot(slot_idx);
    }

    /// Asynchronously copy `runs` (address, bytes) from the CPU view into sparse VRAM tiles via
    /// the 64 MiB ring staging buffer (zero CPU stall on the common path).
    unsafe fn upload_dirty(&mut self, runs: &[(usize, usize)]) {
        if runs.is_empty() {
            return;
        }
        let Some(p) = self.staging_ptr() else { return };
        let mut open_slot: Option<usize> = None;
        let mut last_wi: Option<usize> = None;
        // the copies into the ring (source address, ring offset, bytes), done before their slot is
        // submitted, by the worker threads when there are many
        let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let (aligned, chunk) = plan_ring_chunk(self.ring_write, (n - done).min(to_win_end), STAGING);
                if (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                    self.reclaim_ring();
                    if (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                        if let Some(slot_idx) = open_slot.take() {
                            run_copies(&mut jobs, p);
                            self.submit_slot(slot_idx);
                            self.reclaim_ring();
                        }
                        while (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                            let Some(&(fv, _)) = self.inflight.front() else { break };
                            self.wait_for(fv);
                            self.reclaim_ring();
                        }
                    }
                }
                let slot_idx = match open_slot {
                    Some(idx) => idx,
                    None => {
                        let idx = self.begin_slot();
                        open_slot = Some(idx);
                        idx
                    }
                };
                let dst_off = (aligned as usize) % STAGING;
                jobs.push((cur, dst_off, chunk));
                let staging_res = &self.staging.as_ref().unwrap().0;
                super::call_stats::cs!(RecCopyBuffer, self.slots[slot_idx].list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, staging_res, dst_off as u64, chunk as u64));
                self.ring_write = aligned + chunk as u64;
                done += chunk;
            }
        }
        if let Some(slot_idx) = open_slot {
            run_copies(&mut jobs, p);
            self.submit_slot(slot_idx);
        }
    }
}

/// Allocate a zeroed, write-watched block in the sparse VRAM windows (never straddling a
/// 4 GiB window boundary when `size <= 4 GiB`). Commits the CPU pages and maps VRAM tiles
/// (or queues them until the D3D12 device exists). Returns `(address, rounded_size)`.
pub unsafe fn alloc(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// lab: a block for a list's arena when the coherent window is full (a retail title takes
/// 3934 of its 4095 MB at start; XCL's 64 MB pool + 97 MB were used up ~160 s into the world
/// and from then on a sixth of the draws and a third of the dispatches were skipped as
/// "unified memory is full"). `XCL_ARENA_SPARSE=0`: none, as before.
pub fn arena_block(bytes: usize) -> Option<usize> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("XCL_ARENA_SPARSE").map_or(true, |v| v != "0")) {
        return None;
    }
    static BLOCKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let p = unsafe { alloc(bytes, crate::gpu::uma::GRAIN) }.map(|(p, _)| p);
    let n = BLOCKS.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let (allocators, pending) = super::console_list::pending_blocks();
        crate::xlog_info!(
            "lab: arena block #{n} ({} KB) from the sparse windows: the coherent window is full ({pending} blocks held for {allocators} allocators, {} in lists that record or wait for their reset){}; the window by caller: {}",
            bytes >> 10,
            super::bind::blocks_in_lists() + super::console_list::blocks_in_lists(),
            if p.is_none() { " - none there either" } else { "" },
            crate::gpu::uma::used_by_site()
        );
    }
    p
}

/// Allocate a zeroed, write-watched block strictly inside sparse window `window` (e.g.
/// window 0 for 32-bit-addressed descriptor heaps). Returns `(address, rounded_size)`.
pub unsafe fn alloc_in(window: usize, size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc_in(window, size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Release a sparse allocation from `alloc` / `alloc_in` (false: not a sparse allocation).
pub unsafe fn free(p: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(p) {
        return false;
    }
    // decommitted before the address goes back: once released, another thread may get it,
    // commit it and write there
    let Some((start, len)) = u.sparse_reservation(p).filter(|&(start, _)| start == p) else { return false };
    decommitted(start, len);
    VirtualFree(p as *mut c_void, len, MEM_DECOMMIT);
    u.sparse_release(p).is_some()
}

/// After `queue.ExecuteCommandLists`: copy GPU-written sparse ranges (e.g. `Write32/64`,
/// `FillMemoryWith32BitValueX`, `CopyMemoryToMemoryX`) back to their CPU view so CPU reads
/// and spin-waits see the GPU writes without clobbering `sent` hashes.
/// The queue tile mappings go out on, once there is one. Answered without `STATE` (`QUEUE`):
/// a caller may be inside a D3D12 call that a holder of `STATE` waits for.
pub fn queue() -> Option<ID3D12CommandQueue> {
    QUEUE.lock().unwrap().clone()
}

pub unsafe fn readback_after_execute(queue: &ID3D12CommandQueue, ranges: &[(usize, usize)]) {
    if ranges.is_empty() {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    s.note_queue(queue);
    let merged = s.after_execute(queue, ranges);
    if merged.is_empty() {
        return;
    }
    // no readback queue: the copy back runs on the state's queue, ordered after the title's,
    // and is waited for with `STATE` released (`READBACK` is taken before `STATE`)
    drop(g);
    let _one = READBACK.lock().unwrap();
    readback_released(lock_state(), &merged);
}

/// copies back submitted and not landed yet, and whether any ran on the readback queue
static RB_PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static RB_ASYNC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether copies back land asynchronously (any ran on the readback queue so far)
pub fn readbacks_async() -> bool {
    RB_ASYNC.load(Ordering::Relaxed)
}

/// Copies back that are done on the GPU, into the title's memory, and those whose title work
/// is done (waiting for the copy): before a CPU read of memory the title's lists wrote.
pub unsafe fn settle_readbacks() {
    if RB_PENDING.load(Ordering::Acquire) == 0 {
        return;
    }
    settle_released(lock_state(), true);
}

/// `drain_readbacks(Drain::TitleDone)` of the state, the wait for a copy whose title work is
/// done made with `STATE` released (`wait_fence`, as in `wait_released`): the readback queue
/// stands still for as long as the GPU does, and the title's other threads need the lock
/// meanwhile. The lock is taken again after the wait, tried for ~2 ms unless `patient` (None:
/// not got).
unsafe fn settle_released(mut g: MutexGuard<'static, Option<State>>, patient: bool) -> Option<MutexGuard<'static, Option<State>>> {
    loop {
        let Some(s) = g.as_mut() else { return Some(g) };
        s.drain_readbacks(Drain::Finished);
        // the oldest copy whose title work is done and which is not done itself
        let mut next = None;
        for j in &s.rb_jobs {
            let Some((title, done)) = s.rb_job_fences(j) else { break };
            if super::real::fence_completed(&title) >= j.rb_v && super::real::fence_completed(&done) < j.done_v {
                next = Some((done, j.done_v));
                break;
            }
        }
        let Some((done, done_v)) = next else { return Some(g) };
        drop(g);
        // the title's lists are done: only the copy itself is left
        let reached = wait_fence(&done, done_v);
        g = if patient { lock_state() } else { try_state()? };
        if !reached {
            return Some(g);
        }
    }
}

/// `STATE`, tried for ~2 ms (None: not got)
fn try_state() -> Option<MutexGuard<'static, Option<State>>> {
    let t0 = std::time::Instant::now();
    loop {
        if let Ok(g) = STATE.try_lock() {
            return Some(g);
        }
        if t0.elapsed() >= std::time::Duration::from_millis(2) {
            return None;
        }
        std::thread::yield_now();
    }
}

/// `settle_readbacks` from a fence observation: the state's lock is only tried (for ~2 ms), as
/// XCL may be waiting on a fence of its own with the lock held.
pub unsafe fn settle_readbacks_try() {
    if RB_PENDING.load(Ordering::Acquire) == 0 {
        return;
    }
    if let Some(g) = try_state() {
        settle_released(g, false);
    }
}

/// The thread that moves finished copies back into the title's memory, so it sees what its
/// lists wrote without submitting again (started with the first copy back).
fn start_readback_thread() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-sparse-readback".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_micros(500));
            if let Ok(mut g) = STATE.try_lock() {
                if let Some(s) = g.as_mut().filter(|s| !s.rb_jobs.is_empty()) {
                    unsafe { s.drain_readbacks(Drain::Finished) };
                }
            }
        });
    });
}

impl State {
    /// `readback_after_execute` on this state. Returns the ranges still to copy back on `queue`
    /// (already ordered after the title's), when there is no readback queue (`readback_released`).
    unsafe fn after_execute(&mut self, queue: &ID3D12CommandQueue, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let s = self;
        let mut filtered: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(a, n)| n > 0 && s.window_index(a).is_some()).collect();
        if filtered.is_empty() {
            return Vec::new();
        }
        filtered.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(filtered.len());
        for (a, n) in filtered {
            match merged.last_mut() {
                Some(last) if a <= last.0.saturating_add(last.1) => {
                    let end = (last.0 + last.1).max(a + n);
                    last.1 = end - last.0;
                }
                _ => merged.push((a, n)),
            }
        }
        if s.rbq.is_some() {
            // this queue's own fence and copy queue (`rb_queues`)
            let key = queue.as_raw() as usize;
            if let Some((q, title, v)) = s.rb_queue_for(key) {
                if super::call_stats::cs!(QueueSignal, queue.Signal(&title, v)).is_ok() {
                    let _ = super::call_stats::cs!(QueueWait, q.Wait(&title, v));
                }
                s.readback_async(&merged, Some(key), v);
                start_readback_thread();
                return Vec::new();
            }
        }
        s.rb_value += 1;
        let v = s.rb_value;
        let signalled = super::call_stats::cs!(QueueSignal, queue.Signal(&s.rb_fence, v)).is_ok();
        if let Some((q, _)) = s.rbq.clone() {
            if signalled {
                let _ = super::call_stats::cs!(QueueWait, q.Wait(&s.rb_fence, v));
            }
            s.readback_async(&merged, None, v);
            start_readback_thread();
            return Vec::new();
        }
        if signalled {
            let _ = super::call_stats::cs!(QueueWait, s.queue.Wait(&s.rb_fence, v));
            super::list_x::note_internal_wait(s.queue.as_raw() as usize, "sparse readback behind a title queue", &s.rb_fence, v);
        }
        merged
    }
}

/// Synchronize `[va, va + len)` in a sparse window for a CPU read (`ID3D12Resource::Map` with
/// a non-empty read range, or `Map(0, NULL)` on a sparse readback resource): flushes pending
/// tile clears and unsubmitted CPU-changed chunks first so CPU writes are preserved, then
/// copies the range back from VRAM into CPU memory. The GPU is waited for with the state
/// unlocked; if the sparse queue is not there after 5 s, the CPU's view stays as it is.
pub unsafe fn readback_now(va: usize, len: usize) {
    if len == 0 {
        return;
    }
    let _one = READBACK.lock().unwrap();
    // lab XCL_SPARSE_MAP_NOWAIT=1: the Map copy queue (taken after READBACK, before STATE)
    let mut mc = map_nowait().then(|| MAP_COPY.lock().unwrap_or_else(|e| e.into_inner()));
    let mut g = lock_state();
    if g.as_ref().is_none_or(|s| s.window_index(va).is_none()) {
        return;
    }
    // copies back of earlier submissions first: their data is older than this one's
    let Some(again) = settle_released(g, true) else { return };
    g = again;
    let Some(mut s) = g.as_mut() else { return };
    if s.window_index(va).is_none() {
        return;
    }
    // lab XCL_SPARSE_MAP_NOWAIT=1: no tile of the range has video memory, so the copy below
    // would copy nothing (`readback_submit` and `copy_back` skip such tiles); the earlier copies
    // back are landed above as without the switch
    if mc.is_some() && !s.any_mapped(va, len) {
        note_map_without_vram(va, len);
        return;
    }
    // Preserve any unsubmitted CPU writes on pages overlapping `[va, va + len)`, a part at a
    // time: as much as the upload ring has room for, scanned and uploaded in one hold of the lock
    let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
    let mut pg = va & !(PAGE - 1);
    let end = va + len;
    let mut left = 0usize;
    while pg < end {
        if left < TILE {
            s.upload_dirty(&dirty_runs);
            dirty_runs.clear();
            // (no room after 5 s: no copy back over CPU writes that could not be sent first)
            let (again, room) = room_released(g, TILE, false);
            g = again;
            let (Some(next), Some(room)) = (g.as_mut(), room) else { return };
            s = next;
            left = room;
            s.flush_zeros();
        }
        left -= TILE;
        let t = pg & !(TILE - 1);
        let tile_end = (t + TILE).min(end);
        if let Some(entry) = s.mapped.get_mut(&t) {
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                if entry.holes & (1u16 << idx) == 0 {
                    let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                    scan_page(bytes, p, &mut entry.sent[idx], &mut dirty_runs);
                }
                p += PAGE;
            }
        }
        pg = tile_end;
    }
    if !dirty_runs.is_empty() {
        s.upload_dirty(&dirty_runs);
    }
    match mc.as_deref_mut() {
        Some(mc) => map_copy_released(g, mc, va, len),
        None => readback_released(g, &[(va, len)]),
    }
}

/// lab `XCL_SPARSE_MAP_NOWAIT=1`: Maps of ranges without video memory, no copy back made
static MAPS_WITHOUT_VRAM: AtomicU64 = AtomicU64::new(0);

fn note_map_without_vram(va: usize, len: usize) {
    let k = MAPS_WITHOUT_VRAM.fetch_add(1, Ordering::Relaxed) + 1;
    if k <= 8 || k.is_power_of_two() {
        crate::xlog_info!("lab XCL_SPARSE_MAP_NOWAIT=1: Map #{k} of {va:#x} ({len} bytes) has no tile with video memory: no copy back");
    }
}

/// `readback_released` of `[va, va + len)` on the Map copy queue (lab `XCL_SPARSE_MAP_NOWAIT=1`,
/// `State::map_copy_submit`), each part waited for with `STATE` released and copied into the
/// pages still committed then, as there. Where the copy queue cannot be made or ordered, the
/// rest goes on the state's queue (`readback_released`). The caller holds `READBACK` and
/// `MAP_COPY` (`mc`).
unsafe fn map_copy_released(mut g: MutexGuard<'static, Option<State>>, mc: &mut Option<MapCopy>, va: usize, len: usize) {
    let end = va + len;
    let mut at = va;
    while at < end {
        let Some(s) = g.as_mut() else { return };
        let dev = s.dev.as_raw() as usize;
        if mc.as_ref().is_none_or(|m| m.dev != dev) {
            *mc = MapCopy::new(&s.dev);
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_info!("lab XCL_SPARSE_MAP_NOWAIT=1: Map copy queue {}", if mc.is_some() { "made" } else { "not made: copies back on the sparse queue" });
            }
        }
        let Some(m) = mc.as_mut() else {
            return readback_released(g, &[(at, end - at)]);
        };
        // (a copy given up earlier: its list and buffer are still in use)
        if m.value != 0 && super::real::fence_completed(&m.fence) < m.value {
            let (f, v) = (m.fence.clone(), m.value);
            drop(g);
            let reached = wait_fence(&f, v);
            g = lock_state();
            if !reached {
                return;
            }
            continue;
        }
        let Some((next, parts)) = s.map_copy_submit(m, at, end) else {
            return readback_released(g, &[(at, end - at)]);
        };
        at = next;
        if parts.is_empty() {
            continue;
        }
        let (f, v) = (m.fence.clone(), m.value);
        drop(g);
        let reached = wait_fence(&f, v);
        g = lock_state();
        if !reached {
            return;
        }
        if let Some(s) = g.as_mut().filter(|s| s.dev.as_raw() as usize == m.dev) {
            s.copy_back(m.ptr, &parts);
        }
    }
}

/// The title committed `[a, a + n)` in a sparse window. The pages of tiles mapped for it that
/// lie outside the range are not committed (holes); the range's pages are.
pub unsafe fn committed(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    super::watch_trap::changed(a, n);
    let mut g = lock_state();
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_add(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.commit_range(a, n);
}

/// The title decommitted `[a, a + n)`: tiles whose pages are all uncommitted (whether in one
/// full-tile decommit or incrementally via sub-tile decommits) give their video memory back
/// (unless a placed buffer still pins the pool tile).
pub unsafe fn decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    super::watch_trap::changed(a, n);
    let mut g = lock_state();
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_remove(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.decommit_range(a, n);
}

impl State {
    /// `committed` with the state locked
    unsafe fn commit_range(&mut self, a: usize, n: usize) {
        self.ensure_mapped(a, n, |_| 0xFFFF);
        let mut pg = a & !(PAGE - 1);
        let end = a.saturating_add(n);
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = t.saturating_add(TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.holes &= !(1u16 << ((p - t) / PAGE));
                    p += PAGE;
                }
            }
            if tile_end <= pg {
                break;
            }
            pg = tile_end;
        }
    }

    /// `decommitted` with the state locked: the tiles that go are parked (`flush_parked` unmaps
    /// them and frees their pool tiles)
    unsafe fn decommit_range(&mut self, a: usize, n: usize) {
        let mut pools: Vec<(usize, (u32, u32))> = Vec::new();
        let mut t = a & !(TILE - 1);
        let end = a.saturating_add(n);
        while t < end {
            if let Some(e) = self.mapped.get(&t) {
                pools.push((t, e.pool));
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        let tiles = decommit_tiles(&mut self.mapped, &self.pinned, &mut Vec::new(), &mut self.to_zero, a, n);
        if tiles.is_empty() {
            return;
        }
        self.runs = None;
        for &t in &tiles {
            if let Ok(i) = pools.binary_search_by_key(&t, |p| p.0) {
                self.parked.insert(t, pools[i].1);
            }
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.drain_quarantined();
            self.retire_free_pools();
        }
    }

    /// A pool tile a buffer used at `t` goes back to the free list, unless its window maps it, it
    /// is parked there, or another buffer uses it. Kept mapped for the buffer: parked from here on.
    /// lab: with `XCL_SPARSE_COOL=1` it is quarantined at the next flush (`released`) instead.
    fn release_if_unused(&mut self, t: usize, tile: (u32, u32)) {
        self.release_if_unused_as(t, tile, sparse_cool());
    }

    /// `release_if_unused`, quarantined at the next flush if `cool` (`XCL_SPARSE_COOL`).
    fn release_if_unused_as(&mut self, t: usize, tile: (u32, u32), cool: bool) {
        if self.pinned.contains_key(&tile) {
            return;
        }
        if self.kept.get(&t) == Some(&tile) {
            self.kept.remove(&t);
            self.parked.insert(t, tile);
            return;
        }
        if self.mapped.get(&t).is_none_or(|e| e.pool != tile) && self.parked.get(&t) != Some(&tile) && !self.quarantined.iter().any(|b| b.tiles.contains(&tile)) {
            if !cool {
                self.free.push(tile);
            } else if !self.released.contains(&tile) {
                self.released.push(tile);
            }
        }
    }

    unsafe fn note_queue(&mut self, q: &ID3D12CommandQueue) -> bool {
        let key = q.as_raw() as usize;
        if self.queue_fences.contains_key(&key) {
            return true;
        }
        // Ignore queues belonging to a different device (e.g. isolated unit tests).
        let mut qdev: Option<ID3D12Device> = None;
        if q.GetDevice(&mut qdev).is_err() || qdev.as_ref().is_none_or(|d| d.as_raw() != self.dev.as_raw()) {
            return true;
        }
        match self.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) {
            Ok(fence) => {
                self.queue_fences.insert(key, (q.clone(), fence, 0));
                true
            }
            Err(_) => false,
        }
    }

    unsafe fn sync_known_queues(&mut self) -> bool {
        let mut ok = true;
        for raw in super::present::known_queues() {
            if !self.queue_fences.contains_key(&raw) {
                let ptr = raw as *mut c_void;
                if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&ptr) {
                    ok &= self.note_queue(q);
                }
            }
        }
        ok
    }

    unsafe fn quarantine_tiles(&mut self, tiles: Vec<(u32, u32)>) {
        if tiles.is_empty() {
            return;
        }
        let mut ok = self.sync_known_queues();
        let mut marks = Vec::with_capacity(1 + self.queue_fences.len());
        marks.push((self.fence.clone(), self.value));
        if ok {
            for (q, fence, val) in self.queue_fences.values_mut() {
                let next = *val + 1;
                if super::call_stats::cs!(QueueSignal, q.Signal(&*fence, next)).is_ok() {
                    *val = next;
                    marks.push((fence.clone(), next));
                } else {
                    ok = false;
                    break;
                }
            }
        }
        self.quarantined.push_back(RetiredTiles { tiles, marks: ok.then_some(marks) });
    }

    unsafe fn drain_quarantined(&mut self) -> bool {
        let mut freed = false;
        while let Some(front) = self.quarantined.front() {
            let ready = front.marks.as_ref().is_some_and(|marks| marks.iter().all(|(f, v)| super::real::probe_completed(f) >= *v));
            if !ready {
                break;
            }
            let batch = self.quarantined.pop_front().unwrap();
            for tile in batch.tiles {
                if !self.pinned.contains_key(&tile) {
                    self.free.push(tile);
                    freed = true;
                }
            }
        }
        freed
    }

    /// The parked tiles unmapped from their windows (one call per window) and their unpinned
    /// pool tiles quarantined behind `self.queue` and every known title queue's completion fence.
    /// Pools left all free are not retired here: callers do that (`retire_free_pools`) unless
    /// they take a tile right away.
    unsafe fn flush_parked(&mut self) -> bool {
        if self.parked.is_empty() {
            // lab (`XCL_SPARSE_COOL=1`): tiles buffers gave back, behind the same fences
            if !self.released.is_empty() {
                let tiles = std::mem::take(&mut self.released);
                self.quarantine_tiles(tiles);
            }
            return false;
        }
        let mut by_window: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut tiles: Vec<(u32, u32)> = std::mem::take(&mut self.released);
        for (t, tile) in std::mem::take(&mut self.parked) {
            // (a placed buffer uses it)
            if self.pinned.contains_key(&tile) {
                self.kept.insert(t, tile);
                continue;
            }
            if let Some(wi) = self.window_index(t) {
                by_window.entry(wi).or_default().push(t);
            }
            tiles.push(tile);
        }
        if by_window.is_empty() {
            if !tiles.is_empty() {
                self.quarantine_tiles(tiles);
            }
            return false;
        }
        for (wi, ts) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.unmap_tiles(&r, b, &ts);
        }
        self.signal();
        self.quarantine_tiles(tiles);
        true
    }
}

/// The sparse windows' tiles with video memory as sorted (start, end) runs: where a CPU write
/// has to reach the GPU.
pub fn mapped_ranges() -> Vec<(usize, usize)> {
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return Vec::new() };
    if s.runs.is_none() {
        let mut tiles: Vec<usize> = s.mapped.keys().copied().collect();
        tiles.sort_unstable();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for t in tiles {
            match runs.last_mut() {
                Some(last) if last.1 == t => last.1 = t + TILE,
                _ => runs.push((t, t + TILE)),
            }
        }
        s.runs = Some(runs);
    }
    s.runs.clone().unwrap_or_default()
}

/// time spent and pages found by the write-watch scans (`refresh_once`), for the timing line
static WATCH: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// [lock wait, chunk scan, upload] in µs, runs uploaded
static PHASES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Microseconds so far in [looks for CPU writes, waits for the state lock, chunk scans, uploads].
pub fn timing_totals() -> [u64; 4] {
    [WATCH[0].load(Ordering::Relaxed), PHASES[0].load(Ordering::Relaxed), PHASES[1].load(Ordering::Relaxed), PHASES[2].load(Ordering::Relaxed)]
}

pub fn note_watch(t: std::time::Duration, pages: usize) {
    WATCH[0].fetch_add(t.as_micros() as u64, Ordering::Relaxed);
    WATCH[1].fetch_add(pages as u64, Ordering::Relaxed);
}

/// One page to scan: its address and the slot of its hashes (disjoint for every page of one
/// submit: the pages are distinct and their tiles' entries are only touched through these).
struct PageSlot {
    pg: usize,
    slot: *mut Option<Box<[u64; CHUNKS]>>,
}

/// The pages of one submit, shared with the worker threads (`worker_pool`).
struct SharedPages<'a>(&'a [PageSlot]);
// SAFETY: the slots are disjoint, and the scan finishes before the state's lock is given up
unsafe impl Sync for SharedPages<'_> {}

/// Scan `pages` (`scan_page` on each); the changed runs go to `out` in address order.
unsafe fn scan_pages(pages: &[PageSlot], out: &mut Vec<(usize, usize)>) {
    /// pages a worker takes at a time, and the number from which the work is shared
    const PIECE: usize = 64;
    const SHARE_FROM: usize = 256;
    let scan = |p: &PageSlot, out: &mut Vec<(usize, usize)>| {
        let bytes = std::slice::from_raw_parts(p.pg as *const u8, PAGE);
        scan_page(bytes, p.pg, &mut *p.slot, out);
    };
    if pages.len() < SHARE_FROM {
        pages.iter().for_each(|p| scan(p, out));
        return;
    }
    let pieces = pages.len().div_ceil(PIECE);
    let outs: Vec<Mutex<Vec<(usize, usize)>>> = (0..pieces).map(|_| Mutex::new(Vec::new())).collect();
    let shared = SharedPages(pages);
    let work = |k: usize| {
        // (the whole struct, not its field: that is what is Sync)
        let shared = &shared;
        let mut mine = Vec::new();
        for p in &shared.0[k * PIECE..((k + 1) * PIECE).min(shared.0.len())] {
            scan(p, &mut mine);
        }
        *outs[k].lock().unwrap_or_else(|e| e.into_inner()) = mine;
    };
    super::worker_pool::run(pieces, &work);
    for o in outs {
        for (addr, len) in o.into_inner().unwrap_or_else(|e| e.into_inner()) {
            push_run(out, addr, len);
        }
    }
}

/// Copies from the title's memory into the staging ring: shared with the worker threads when
/// there are many bytes.
unsafe fn run_copies(jobs: &mut Vec<(usize, usize, usize)>, staging: *mut u8) {
    const SHARE_FROM: usize = 512 << 10;
    let total: usize = jobs.iter().map(|j| j.2).sum();
    if total < SHARE_FROM || jobs.len() < 2 {
        for &(src, off, n) in jobs.iter() {
            std::ptr::copy_nonoverlapping(src as *const u8, staging.add(off), n);
        }
    } else {
        // pieces of about the same bytes, in the order of the jobs
        let target = (total / 16).max(64 << 10);
        let mut pieces: Vec<(usize, usize)> = Vec::new();
        let (mut from, mut bytes) = (0usize, 0usize);
        for (k, j) in jobs.iter().enumerate() {
            bytes += j.2;
            if bytes >= target {
                pieces.push((from, k + 1));
                from = k + 1;
                bytes = 0;
            }
        }
        if from < jobs.len() {
            pieces.push((from, jobs.len()));
        }
        struct Staging(*mut u8);
        // SAFETY: the copies go to disjoint pieces of the ring, all from the caller's job list
        unsafe impl Sync for Staging {}
        let st = Staging(staging);
        let list: &[(usize, usize, usize)] = jobs;
        super::worker_pool::run(pieces.len(), &|k| {
            let st = &st;
            for &(src, off, n) in &list[pieces[k].0..pieces[k].1] {
                unsafe { std::ptr::copy_nonoverlapping(src as *const u8, st.0.add(off), n) };
            }
        });
    }
    jobs.clear();
}

/// Before the title's ExecuteCommandLists on `queue`: the pages the CPU wrote (`dirty`, 4 KiB
/// pages from the write-watch, sorted) and the reused tiles are copied asynchronously into video
/// memory, and `queue` GPU-waits for that and for new tile mappings. The pages go a part at a
/// time, each as large as the upload ring has room for: scanned and uploaded in one hold of the
/// lock, the room waited for with it released.
pub unsafe fn before_execute(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) {
    super::real::removed_probe("sparse::before_execute: start");
    before_execute_inner(queue, dirty, page);
    super::real::removed_probe("sparse::before_execute: end");
}

unsafe fn before_execute_inner(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) {
    let t0 = std::time::Instant::now();
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    s.note_queue(queue);
    s.drain_readbacks(Drain::Finished);
    // the title's work sees what was decommitted unmapped, as it was
    let flushed = s.flush_parked();
    let drained = s.drain_quarantined();
    if flushed || drained {
        s.retire_free_pools();
    }
    let t_locked = std::time::Instant::now();
    PHASES[0].fetch_add((t_locked - t0).as_micros() as u64, Ordering::Relaxed);
    let (again, room) = room_released(g, if dirty.is_empty() { 0 } else { page }, true);
    g = again;
    let (Some(mut s), Some(mut left)) = (g.as_mut(), room) else { return };
    // lab: pages written in the tiled shadow window get their tiles first (before the clears:
    // a reused tile is cleared, then written)
    if let Some((lo, hi)) = lazy_range() {
        let mut last = 0usize;
        let mut new = 0u64;
        for &a in dirty {
            let t = a & !(TILE - 1);
            if t != last && t >= lo && t < hi && !s.mapped.contains_key(&t) {
                s.ensure_mapped(t, TILE, |_| 0);
                new += 1;
            }
            last = t;
        }
        if new > 0 {
            static TILES: AtomicU64 = AtomicU64::new(0);
            let n = TILES.fetch_add(new, Ordering::Relaxed) + new;
            if (n - new).next_power_of_two() <= n || n.is_power_of_two() {
                crate::xlog_info!("lab: tiled shadow: {} MB of tiles mapped for CPU writes so far", n >> 4);
            }
        }
    }
    // Submit tile clears before CPU writes so overlapping CopyBufferRegion destinations have no WAW hazard.
    let zero_runs = s.flush_zeros();
    let mut dirty_runs = std::mem::take(&mut s.scratch_runs);
    dirty_runs.clear();
    // (runs, bytes) of the parts sent before the last
    let mut sent = (0usize, 0usize);
    let mut pages: Vec<PageSlot> = Vec::with_capacity(dirty.len());
    let mut i = 0;
    while i < dirty.len() {
        if left < page {
            // this part goes; the next has the room the ring has then
            scan_pages(&pages, &mut dirty_runs);
            pages.clear();
            s.upload_dirty(&dirty_runs);
            sent = (sent.0 + dirty_runs.len(), sent.1 + dirty_runs.iter().map(|r| r.1).sum::<usize>());
            dirty_runs.clear();
            let (again, room) = room_released(g, page, true);
            g = again;
            let (Some(next), Some(room)) = (g.as_mut(), room) else { return };
            s = next;
            left = room;
            s.flush_zeros();
        }
        let a = dirty[i];
        let mut n = page;
        i += 1;
        while i < dirty.len() && dirty[i] == a + n && n + page <= left {
            n += page;
            i += 1;
        }
        left -= n;
        // only what has video memory (committed tiles of a sparse window)
        let mut x = a;
        while x < a + n {
            let t = x & !(TILE - 1);
            let end = (t + TILE).min(a + n);
            if let Some(entry) = s.mapped.get_mut(&t) {
                let mut pg = x & !(PAGE - 1);
                while pg < end {
                    let idx = (pg - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        pages.push(PageSlot { pg, slot: &mut entry.sent[idx] });
                    }
                    pg += PAGE;
                }
            }
            x = end;
        }
    }
    scan_pages(&pages, &mut dirty_runs);
    let t_scanned = std::time::Instant::now();
    PHASES[1].fetch_add((t_scanned - t_locked).as_micros() as u64, Ordering::Relaxed);
    PHASES[3].fetch_add((zero_runs.len() + dirty_runs.len() + sent.0) as u64, Ordering::Relaxed);
    if !dirty_runs.is_empty() {
        s.upload_dirty(&dirty_runs);
    }
    PHASES[2].fetch_add(t_scanned.elapsed().as_micros() as u64, Ordering::Relaxed);
    {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        static BYTES: AtomicU64 = AtomicU64::new(0);
        static MICROS: AtomicU64 = AtomicU64::new(0);
        let bytes: usize = zero_runs.iter().chain(dirty_runs.iter()).map(|r| r.1).sum::<usize>() + sent.1;
        let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let b = BYTES.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
        let us = MICROS.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed) + t0.elapsed().as_micros() as u64;
        if n.is_power_of_two() && n >= 16 {
            crate::xlog_info!("sparse windows: {n} submits, {} MB of CPU writes uploaded, {} ms spent, {} tiles mapped", b >> 20, us / 1000, s.mapped.len());
            let ms = |x: &AtomicU64| x.load(Ordering::Relaxed) / 1000;
            crate::xlog_info!(
                "sparse windows timing: write-watch {} ms ({} dirty pages), lock wait {} ms, chunk scan {} ms, upload {} ms ({} runs)",
                ms(&WATCH[0]),
                WATCH[1].load(Ordering::Relaxed),
                ms(&PHASES[0]),
                ms(&PHASES[1]),
                ms(&PHASES[2]),
                PHASES[3].load(Ordering::Relaxed)
            );
        }
    }
    dirty_runs.clear();
    s.scratch_runs = dirty_runs;
    let q = queue.as_raw() as usize;
    if s.value > 0 && s.waited.get(&q).copied().unwrap_or(0) < s.value {
        let _ = super::call_stats::cs!(QueueWait, queue.Wait(&s.fence, s.value));
        super::list_x::note_internal_wait(q, "sparse uploads and tile mappings", &s.fence, s.value);
        s.waited.insert(q, s.value);
    }
}

/// A buffer the title places at `va` in a sparse window (`bytes` long, `desc` the PC
/// description): a reserved resource over the window's tiles there, starting at the 64 KiB
/// below `va` (returned: its distance from there).
pub unsafe fn place_buffer(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<(crate::util::HRESULT, usize)> {
    place_buffer_when(this, va, desc, state, riid, ppv, false)
}

/// lab: (start, end) around the addresses of the buffers of `place_buffer_when(.., lazy)` that
/// have no video memory yet (0, 0: none) - what `use_range` looks at before it takes the lock
static LAZY_BUFS: [std::sync::atomic::AtomicUsize; 2] = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];
/// lab: bytes of lazily placed buffers: placed, given video memory
static LAZY_BYTES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn lazy_bounds(s: &State) {
    // (lab XCL_USE_RANGE_HIT=1: the index first, so a `use_range` that sees the new bounds sees it)
    if use_range_hit() {
        *LAZY_INDEX.write().unwrap_or_else(|e| e.into_inner()) = lazy_index_build(&s.lazy_bufs);
    }
    let lo = s.lazy_bufs.iter().map(|b| b.1).min().unwrap_or(0);
    let hi = s.lazy_bufs.iter().map(|b| b.1 + b.2).max().unwrap_or(0);
    LAZY_BUFS[0].store(lo, Ordering::Release);
    LAZY_BUFS[1].store(hi, Ordering::Release);
}

/// lab `XCL_USE_RANGE_HIT=1` (off by default): 0 not read yet, 1 off, 2 on
static USE_RANGE_HIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// lab `XCL_USE_RANGE_HIT=1`: `use_range` takes the state only for an address in a lazily placed
/// buffer without video memory (`LAZY_INDEX`), not for every one between the lowest and the
/// highest of them (PROF1: ~7000 acquisitions a frame, nearly all finding nothing); `bind`
/// names each address of a draw's tables once (`SeenAddrs`).
pub(super) fn use_range_hit() -> bool {
    match USE_RANGE_HIT.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("XCL_USE_RANGE_HIT").is_ok_and(|v| v == "1");
            USE_RANGE_HIT.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            if on {
                crate::xlog_info!("lab XCL_USE_RANGE_HIT=1: use_range takes the sparse state only for an address in a lazily placed buffer without video memory");
            }
            on
        }
        v => v == 2,
    }
}

/// tests: the switch as `on` says (None: read from the environment again)
#[cfg(test)]
fn set_use_range_hit(on: Option<bool>) {
    USE_RANGE_HIT.store(
        match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

/// lab `XCL_USE_RANGE_HIT=1`: the lazily placed buffers without video memory, outside `STATE`:
/// (start, end, the largest end of this one and those before it), sorted by start; made again
/// with every change of `State::lazy_bufs` (`lazy_bounds`).
static LAZY_INDEX: std::sync::RwLock<Vec<(usize, usize, usize)>> = std::sync::RwLock::new(Vec::new());

/// `LAZY_INDEX` of `bufs` (resource, first tile, bytes from there)
fn lazy_index_build(bufs: &[(usize, usize, usize)]) -> Vec<(usize, usize, usize)> {
    let mut v: Vec<(usize, usize)> = bufs.iter().map(|b| (b.1, b.1 + b.2)).collect();
    v.sort_unstable();
    let mut hi = 0usize;
    v.into_iter()
        .map(|(s, e)| {
            hi = hi.max(e);
            (s, e, hi)
        })
        .collect()
}

/// Whether `[va, end)` meets a range of `index` (`lazy_index_build`): the test `use_range` makes
/// under the lock (`va < start + bytes && start < end`), for any of them. The ranges that start
/// before `end` are a prefix; one of them reaches past `va` if the largest end among them does.
fn lazy_index_hits(index: &[(usize, usize, usize)], va: usize, end: usize) -> bool {
    let n = index.partition_point(|r| r.0 < end);
    n > 0 && index[n - 1].2 > va
}

/// lab `XCL_USE_RANGE_HIT=1`: the addresses a draw's tables name, each passed on once. A small
/// direct-mapped set: an address seen is skipped, one evicted by another may come again (calling
/// `use_range` twice for an address is harmless: the second call finds nothing left to map).
pub(super) struct SeenAddrs([usize; 64]);

impl SeenAddrs {
    pub(super) fn new() -> Self {
        SeenAddrs([usize::MAX; 64])
    }

    /// true the first time `va` comes (as far as the set remembers)
    pub(super) fn first(&mut self, va: usize) -> bool {
        let k = ((va >> 4) ^ (va >> 12) ^ (va >> 20)) & 63;
        if self.0[k] == va {
            return false;
        }
        self.0[k] = va;
        true
    }
}

/// lab: (MB placed lazily, MB of them with video memory)
pub fn lazy_buffer_mb() -> (u64, u64) {
    (LAZY_BYTES[0].load(Ordering::Relaxed) >> 20, LAZY_BYTES[1].load(Ordering::Relaxed) >> 20)
}

/// lab: `[va, va + len)` of the sparse windows is about to be used by the GPU (a V# of a bound
/// table, a root address, a copy): the lazily placed buffers there get their video memory now.
/// Called while the list that uses them is recorded; the title's queue waits for the mapping at
/// its next submit (`before_execute`).
pub unsafe fn use_range(va: usize, len: usize) {
    let (lo, hi) = (LAZY_BUFS[0].load(Ordering::Acquire), LAZY_BUFS[1].load(Ordering::Acquire));
    if va >= hi || va.saturating_add(len.max(1)) <= lo {
        return;
    }
    // lab XCL_USE_RANGE_HIT=1: the state only when a buffer without video memory is there
    if use_range_hit() && !lazy_index_hits(&LAZY_INDEX.read().unwrap_or_else(|e| e.into_inner()), va, va.saturating_add(len.max(1))) {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    let end = va.saturating_add(len.max(1));
    let mut i = 0;
    let mut any = false;
    while i < s.lazy_bufs.len() {
        let (raw, start, bytes) = s.lazy_bufs[i];
        if va >= start + bytes || end <= start {
            i += 1;
            continue;
        }
        s.lazy_bufs.swap_remove(i);
        any = true;
        let ptr = raw as *mut c_void;
        let Some(res) = ID3D12Resource::from_raw_borrowed(&ptr).cloned() else { continue };
        s.ensure_mapped(start, bytes, |t| query_tile_holes(t));
        let tiles: Vec<(usize, (u32, u32))> = (start..start + bytes).step_by(TILE).filter_map(|t| s.mapped.get(&t).map(|e| (t, e.pool))).collect();
        s.map_tiles(&res, start, &tiles);
        for &(_, tile) in &tiles {
            *s.pinned.entry(tile).or_default() += 1;
        }
        s.buffers.insert(raw, tiles);
        let total = LAZY_BYTES[1].fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
        static N: AtomicU64 = AtomicU64::new(0);
        let k = N.fetch_add(1, Ordering::Relaxed) + 1;
        if k <= 200 || k.is_power_of_two() {
            crate::xlog_info!("lab: lazy buffer #{k} {raw:#x} at {start:#x} ({bytes} bytes) gets video memory at its first use ({va:#x}); {} MB of {} MB placed so far, {:.1} s", total >> 20, LAZY_BYTES[0].load(Ordering::Relaxed) >> 20, crate::log::uptime());
        }
    }
    if any {
        lazy_bounds(s);
        s.signal();
    }
}

/// `place_buffer`; lab: with `lazy` the buffer's tiles get no video memory until `use_range`
/// names an address of it (unmapped tiles read 0 and drop writes).
pub unsafe fn place_buffer_when(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void, lazy: bool) -> Option<(crate::util::HRESULT, usize)> {
    let mut g = lock_state();
    let s = g.as_mut()?;
    let start = va & !(TILE - 1);
    let delta = va - start;
    let bytes = desc.Width as usize;
    if bytes == 0 {
        return None;
    }
    s.window_index(start)?;
    s.window_index(va.checked_add(bytes)?.checked_sub(1)?)?;
    let mut d = *desc;
    d.Width = (delta + bytes) as u64;
    d.Alignment = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Reserved buffers only accept NONE or ALLOW_UNORDERED_ACCESS (strip console flags and ALLOW_CROSS_ADAPTER).
    d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if state & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32 | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32) != 0 {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    let hr = super::real::dev_create_reserved_resource(this, &d as *const _ as usize, state as usize, 0, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        return None;
    }
    if lazy {
        let raw = *ppv;
        s.lazy_bufs.push((raw as usize, start, delta + bytes));
        s.buffers.insert(raw as usize, Vec::new());
        LAZY_BYTES[0].fetch_add((delta + bytes) as u64, Ordering::Relaxed);
        lazy_bounds(s);
        return Some((hr, delta));
    }
    // its tiles: the window's video memory at the same addresses; probe actual OS commitment
    // so uncommitted pages (e.g. [start, va) or reserved tails) stay marked as holes
    s.ensure_mapped(start, delta + bytes, |t| query_tile_holes(t));
    let tiles: Vec<(usize, (u32, u32))> = (start..va + bytes).step_by(TILE).filter_map(|t| s.mapped.get(&t).map(|e| (t, e.pool))).collect();
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?.clone();
    s.map_tiles(&res, start, &tiles);
    for &(_, tile) in &tiles {
        *s.pinned.entry(tile).or_default() += 1;
    }
    s.buffers.insert(raw as usize, tiles);
    s.signal();
    Some((hr, delta))
}

/// Placed buffers released while `STATE` was held, let go by the next `lock_state`. Held only
/// to push or take, never across a call out.
static FORGOTTEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `STATE`, with the buffers released while it was held let go first.
fn lock_state() -> std::sync::MutexGuard<'static, Option<State>> {
    let mut g = super::call_stats::lock(&STATE, super::call_stats::C::LockState);
    forget_released(&mut g);
    g
}

fn forget_released(g: &mut Option<State>) {
    let gone = std::mem::take(&mut *FORGOTTEN.lock().unwrap());
    let Some(s) = g.as_mut() else { return };
    for res in gone {
        forget_in(s, res);
    }
}

/// A placed buffer is gone: its tiles are not pinned by it any more. This runs from the
/// resource's last Release, which D3D12 can make while it holds a lock of its own (in a title's
/// command list Reset), and `before_execute` calls D3D12 with `STATE` held: waiting for `STATE`
/// here deadlocked the two threads. So it does not wait: with `STATE` held, the holder's next
/// `lock_state` lets the buffer go.
pub fn forget_buffer(res: usize) {
    FORGOTTEN.lock().unwrap().push(res);
    match STATE.try_lock() {
        Ok(mut g) => forget_released(&mut g),
        Err(std::sync::TryLockError::WouldBlock) => {}
        Err(e) => panic!("sparse state: {e}"),
    }
}

fn forget_in(s: &mut State, res: usize) {
    if let Some(i) = s.lazy_bufs.iter().position(|b| b.0 == res) {
        let b = s.lazy_bufs.swap_remove(i);
        LAZY_BYTES[0].fetch_sub(b.2 as u64, Ordering::Relaxed);
        lazy_bounds(s);
    }
    let Some(tiles) = s.buffers.remove(&res) else { return };
    for (t, tile) in tiles {
        if let Some(c) = s.pinned.get_mut(&tile) {
            *c -= 1;
            if *c == 0 {
                s.pinned.remove(&tile);
                s.release_if_unused(t, tile);
            }
        }
    }
}

#[cfg(test)]
mod readback_guard_tests {
    use super::*;

    /// lab (`XCL_READBACK_GUARD=1`): a copy back lands the chunks the CPU did not write while it
    /// ran, and keeps the one it wrote; a part that starts and ends inside a chunk copies only its
    /// own bytes.
    #[test]
    fn a_copy_back_keeps_a_chunk_the_cpu_wrote_meanwhile() {
        unsafe {
            // a page-aligned "CPU view" of 4 chunks, and what video memory holds
            let mut page = vec![0u8; 2 * PAGE];
            let base = (page.as_mut_ptr() as usize + PAGE - 1) & !(PAGE - 1);
            let view = std::slice::from_raw_parts_mut(base as *mut u8, 4 * CHUNK);
            view.fill(1);
            let gpu = vec![9u8; 4 * CHUNK];
            // bytes 10 .. 4 chunks - 10: the first and last chunks only in part
            let parts = [(base + 10, 10usize, 4 * CHUNK - 20)];
            let all = |_: usize| true;
            let before = chunk_hashes(&parts, all);
            assert_eq!(before.len(), 4);
            // the CPU writes chunk 2 while the copy runs
            view[2 * CHUNK + 5] = 7;
            let (landed, kept) = land_unchanged(gpu.as_ptr(), &parts, &before, all);
            assert_eq!(kept, 1);
            assert_eq!(landed.iter().map(|l| (l.0 - base) / CHUNK).collect::<Vec<_>>(), vec![0, 1, 3]);
            assert_eq!(&view[..10], &[1u8; 10], "before the part: as it was");
            assert!(view[10..2 * CHUNK].iter().all(|&b| b == 9));
            assert_eq!(view[2 * CHUNK + 5], 7, "the CPU's write is kept");
            assert_eq!(view[2 * CHUNK], 1);
            assert!(view[3 * CHUNK..4 * CHUNK - 10].iter().all(|&b| b == 9));
            assert_eq!(&view[4 * CHUNK - 10..], &[1u8; 10], "after the part: as it was");
            assert_eq!(landed[0].1, chunk_hash(&view[..CHUNK]), "the hash after the landing");
            // a page that is not committed is neither read nor written
            let (landed, kept) = land_unchanged(gpu.as_ptr(), &parts, &chunk_hashes(&parts, |_| false), |_| false);
            assert!(landed.is_empty() && kept == 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f` on another thread while this one holds the sparse state (as `before_execute` does
    /// around its D3D12 calls, which can wait for that thread): did `f` finish without waiting
    /// for the state? The thread runs before the state is taken, so its start is not timed.
    fn returns_while_the_state_is_held(f: impl FnOnce() + Send + 'static) -> bool {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let _ = ready_tx.send(());
            let _ = go_rx.recv();
            f();
            let _ = tx.send(());
        });
        ready_rx.recv().unwrap();
        let held = STATE.lock().unwrap();
        go_tx.send(()).unwrap();
        let done = rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(held);
        let joined = t.join();
        assert!(joined.is_ok(), "the thread panicked: {joined:?}");
        done.is_ok()
    }

    /// A placed buffer released while another thread holds the sparse state: the release does
    /// not wait for the state, and the next `lock_state` lets the buffer go.
    #[test]
    fn a_buffer_released_while_the_state_is_held_does_not_wait_for_it() {
        const RES: usize = 0x7ff0_dead_0000;
        assert!(returns_while_the_state_is_held(|| forget_buffer(RES)), "forget_buffer waited for the sparse state's lock");
        drop(lock_state());
        assert!(!FORGOTTEN.lock().unwrap().contains(&RES));
    }

    /// A copy back after a title submission does not wait for the title's queue: that queue can
    /// be waiting for work the title submits next, from the thread that is in here. The copy
    /// lands once the title's work is done (`drain_readbacks`). Before, the sparse queue
    /// GPU-waited for the title's queue and this thread waited for the copy (5 s, then on).
    #[test]
    fn a_copy_back_does_not_wait_for_the_title_queue() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc(base as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc(TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let rbq_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("readback queue");
            let rbq_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("readback fence");

            let mut s = State {
                dev: dev.clone(),
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queues: FxHashMap::default(),
                rbq: Some((rbq_queue, rbq_fence)),
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                windows: vec![(base, reserved.clone())],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                quarantined: VecDeque::new(),
                released: Vec::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
                lazy_bufs: Vec::new(),
            };
            s.commit_range(base, TILE);
            assert!(s.mapped.contains_key(&base));
            std::ptr::write_bytes(base as *mut u8, 0, 512);

            // the title's queue: blocked on a fence nothing has signalled yet, then a copy of
            // 0..64 into the tile (a GPU write the title's next CPU read expects)
            let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate");
            let mut up: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &{ let mut d = buffer_desc(512); d.Flags = D3D12_RESOURCE_FLAG_NONE; d }, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up)
                .expect("upload buffer");
            let up = up.expect("upload buffer");
            let mut p: *mut c_void = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).expect("Map");
            for i in 0..64u32 {
                *(p as *mut u32).add(i as usize) = 0xC0DE_0000 | i;
            }
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&reserved, 0, &up, 0, 256);
            list.Close().expect("close");
            title.Wait(&s.fence, s.value).expect("wait for the tile mapping");
            title.Wait(&gate, 1).expect("wait for the gate");
            title.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);

            let t0 = std::time::Instant::now();
            s.after_execute(&title, &[(base, 256)]);
            let took = t0.elapsed();
            assert!(took < std::time::Duration::from_secs(1), "the copy back waited {took:?} for the title's queue");
            // the title's own fence after that submission
            let own: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("own fence");
            title.Signal(&own, 1).expect("own signal");
            s.drain_readbacks(Drain::Finished);
            assert_eq!(*(base as *const u32), 0, "nothing lands before the title's work is done");

            // another title queue that does not wait writes 256..512 and gets there first: its copy
            // lands, the blocked queue's copy neither runs early (it would read the bytes before
            // its list wrote them and land those) nor holds this one back
            let other: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("other title queue");
            for i in 0..64u32 {
                *(p as *mut u32).add(64 + i as usize) = 0xB0B0_0000 | i;
            }
            let alloc2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc2, None).expect("list");
            list2.CopyBufferRegion(&reserved, 256, &up, 256, 256);
            list2.Close().expect("close");
            other.Wait(&s.fence, s.value).expect("wait for the tile mapping");
            other.ExecuteCommandLists(&[Some(list2.cast::<ID3D12CommandList>().unwrap())]);
            s.after_execute(&other, &[(base + 256, 256)]);
            let own2: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("own fence");
            other.Signal(&own2, 1).expect("own signal");
            own2.SetEventOnCompletion(1, s.event).expect("SetEventOnCompletion");
            windows::Win32::System::Threading::WaitForSingleObject(s.event, 5000);
            s.drain_readbacks(Drain::TitleDone);
            for i in 0..64u32 {
                assert_eq!(*(base as *const u32).add(64 + i as usize), 0xB0B0_0000 | i, "dword {} of the other queue's copy back", 64 + i);
            }
            assert_eq!(s.rb_jobs.len(), 1, "the blocked queue's copy back is still to come");
            assert_eq!(*(base as *const u32), 0, "nothing of the blocked queue lands before its work is done");

            // the title's queue goes on: the copy back lands once it is done
            gate.Signal(1).expect("gate signal");
            // as the title would: wait for its own work
            own.SetEventOnCompletion(1, s.event).expect("SetEventOnCompletion");
            windows::Win32::System::Threading::WaitForSingleObject(s.event, 5000);
            s.drain_readbacks(Drain::TitleDone);
            for i in 0..64u32 {
                assert_eq!(*(base as *const u32).add(i as usize), 0xC0DE_0000 | i, "dword {i} after the copy back");
            }
            assert!(s.rb_jobs.is_empty(), "the copy backs were consumed");
            assert_eq!(s.rb_pool.len(), 2, "their buffers are kept for the next ones");
            drop(s);
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `queue()` answers while the state's lock is held: a resource's last `Release` asks for
    /// it (`late_release::drain_known`) from inside a D3D12 call that the lock's holder can be
    /// waiting for (a list's `Reset` against an upload's allocator `Reset`, with `XCL_DRED=1`).
    #[test]
    fn the_queue_is_answered_while_the_state_is_locked() {
        let held = STATE.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let asked = std::thread::spawn(move || {
            let _ = queue();
            let _ = tx.send(());
        });
        let answered = rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        drop(held);
        asked.join().unwrap();
        assert!(answered, "queue() waited for the sparse state's lock");
    }

    /// A block freed on one thread while another allocates: its address goes back to the
    /// windows only once it is decommitted (released first, the decommit came after another
    /// thread had the same address again: that thread's writes faulted).
    #[test]
    fn a_freed_block_is_decommitted_before_its_address_is_reused() {
        if crate::gpu::uma::get().is_none_or(|u| !u.has_sparse()) {
            return;
        }
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| unsafe {
                    for i in 0..2000u64 {
                        let Some((p, n)) = alloc(TILE, TILE) else { continue };
                        std::ptr::write_volatile(p as *mut u64, i);
                        std::ptr::write_volatile((p + n - 8) as *mut u64, i);
                        assert_eq!(std::ptr::read_volatile(p as *const u64), i);
                        assert!(free(p));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }


    /// Freeing the last tile in use of a pool and taking a tile again takes the retired pool
    /// back, not a new 64 MB heap each time (#913: 288 pools, 18 GB asked of a 12 GB card).
    #[test]
    fn a_pool_retired_a_moment_ago_is_taken_back() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queues: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                windows: Vec::new(),
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                quarantined: VecDeque::new(),
                released: Vec::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
                lazy_bufs: Vec::new(),
            };

            let (first, reused) = s.take_tile().expect("a tile from a new pool");
            assert!(!reused, "a new pool's tile needs no clearing");
            s.free.push(first);
            s.retire_free_pools();
            assert_eq!(s.retired.len(), 1, "an all-free pool is retired");
            for round in 0..8 {
                let (tile, reused) = s.take_tile().expect("a tile");
                assert!(reused, "round {round}: a tile of a retired pool is cleared before use");
                s.free.push(tile);
                s.retire_free_pools();
                assert_eq!(s.retired.len(), 1, "round {round}: the retired pool was taken back, not a new one made");
                assert_eq!(s.pools.iter().filter(|p| p.is_some()).count(), 0, "round {round}");
            }
        }
    }

    #[test]
    fn aligned_tile_addresses_spread_across_hash_buckets() {
        use std::hash::BuildHasher;

        let hasher = BuildHasherDefault::<FxHasher>::default();
        let mut used = [false; 256];
        for tile in 0..used.len() {
            let address = (1usize << 32) + tile * TILE;
            let bucket = hasher.hash_one(address) as usize & (used.len() - 1);
            used[bucket] = true;
        }
        let occupied = used.iter().filter(|&&bucket| bucket).count();
        assert!(occupied >= used.len() / 2, "256 aligned addresses used {occupied} buckets");
    }

    /// Only the 64-byte chunks the CPU changed are sent again – the rest of the page may hold
    /// what the GPU wrote.
    #[test]
    fn only_changed_chunks_are_sent() {
        let mut page = vec![0u8; PAGE];
        let mut sent = ZERO_HASHES;
        assert!(changed_chunks(&page, &mut sent).is_empty(), "a zero page is what the GPU has");
        page[10] = 1;
        page[200] = 2;
        page[255] = 3;
        page[4000] = 4;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(0, 64), (192, 64), (3968, 64)]);
        assert!(changed_chunks(&page, &mut sent).is_empty(), "nothing changed since");
        page[256] = 5; // next to the chunk at 192: one run
        page[200] = 9;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(192, 128)]);
    }

    /// `scan_page` skips allocating a hash box on all-zero pages, allocates on the first write,
    /// and coalesces adjacent dirty chunks across consecutive pages.
    #[test]
    fn scan_page_lazy_alloc_and_coalesce() {
        let mut page = vec![0u8; PAGE];
        let mut slot: Option<Box<[u64; CHUNKS]>> = None;
        let mut out = Vec::new();
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_none(), "all-zero page must not allocate hash box");
        assert!(out.is_empty());

        page[PAGE - 64] = 0xAA;
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_some());
        assert_eq!(out, vec![(0x1_0FC0, 64)]);

        // Adjacent chunk at the start of the next page coalesces into a single run.
        let mut next = vec![0u8; PAGE];
        next[0] = 0xBB;
        let mut next_slot: Option<Box<[u64; CHUNKS]>> = None;
        scan_page(&next, 0x1_1000, &mut next_slot, &mut out);
        assert_eq!(out, vec![(0x1_0FC0, 128)]);
    }

    /// Ring-buffer planner keeps every chunk contiguous, 64-byte aligned, <= cap/2, and splits
    /// cleanly across the end of the ring buffer without wrapping mid-copy.
    #[test]
    fn ring_planner_handles_wrap_and_large_chunks() {
        let cap = 64 << 20;
        // Normal aligned chunk
        let (a, n) = plan_ring_chunk(0, 4096, cap);
        assert_eq!((a, n), (0, 4096));
        // Unaligned cursor aligns up to 64
        let (a, n) = plan_ring_chunk(17, 256, cap);
        assert_eq!((a, n), (64, 256));
        // Oversized chunk capped at cap / 2
        let (a, n) = plan_ring_chunk(0, cap * 2, cap);
        assert_eq!((a, n), (0, cap / 2));
        // Near end of ring with >= 4 KiB left: shrinks to fill tail
        let near = (cap - 8192) as u64;
        let (a, n) = plan_ring_chunk(near, 16384, cap);
        assert_eq!((a, n), (near, 8192));
        assert_eq!((a as usize % cap) + n, cap);
        // Tiny tail (< 4 KiB): wraps cleanly to 0 of next cycle
        let tiny = (cap - 512) as u64;
        let (a, n) = plan_ring_chunk(tiny, 4096, cap);
        assert_eq!((a, n), (cap as u64, 4096));
        assert_eq!(a as usize % cap, 0);
    }

    /// `query_tile_holes` marks reserved, free, and `PAGE_NOACCESS` pages as holes while clearing
    /// bits only for committed, accessible 4 KiB pages.
    #[test]
    fn query_tile_holes_detects_reserved_noaccess_and_committed_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            // SAFETY: Reserving 2 tiles (128 KiB) with VirtualAlloc returns a 64 KiB-aligned VA.
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);

            // Initially every page in both reserved tiles is a hole.
            assert_eq!(query_tile_holes(base), 0xFFFF);
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 4..8 as PAGE_READWRITE and page 9 as PAGE_NOACCESS.
            let p4 = VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p4.is_null());
            let p9 = VirtualAlloc((base + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_NOACCESS);
            assert!(!p9.is_null());

            // Only bits 4..8 are cleared; reserved [0..4), PAGE_NOACCESS [9], and reserved [10..16) remain holes.
            assert_eq!(query_tile_holes(base), 0xFFFF & !(0b1111u16 << 4));
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 0..4 and confirm bits 0..4 clear too.
            let p0 = VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p0.is_null());
            assert_eq!(query_tile_holes(base), 0xFF00);

            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            // Released (MEM_FREE) memory reports all holes.
            assert_eq!(query_tile_holes(base), 0xFFFF);
        }
    }

    /// Mapping tiles for a placed buffer over partially-committed/reserved memory must preserve
    /// uncommitted hole bits so `readback_ranges` / `sync_sent_range` do not access-violate,
    /// and later commits clear those hole bits so subsequent writes scan cleanly.
    #[test]
    fn ensure_mapped_with_query_holes_skips_uncommitted_pages_on_readback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");

            // Reserve 2 tiles; commit only [base + 0x4000, base + 0x8000) to simulate a
            // non-64KiB-aligned placed buffer with uncommitted prefix [base, base + 0x4000) and
            // uncommitted second tile [base + TILE, base + 2*TILE).
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());

            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queues: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                quarantined: VecDeque::new(),
                released: Vec::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
                lazy_bufs: Vec::new(),
            };

            // Mirrors `place_buffer`: map both tiles probing real OS page commitment.
            s.ensure_mapped(base, 2 * TILE, |t| query_tile_holes(t));
            assert_eq!(s.mapped[&base].holes, 0xFF0F, "uncommitted prefix and suffix pages must stay holes");
            assert_eq!(s.mapped[&(base + TILE)].holes, 0xFFFF, "fully reserved tile must have all 16 holes set");

            // Reading back the full 2-tile span must not fault on uncommitted pages.
            let (rb, p) = s.readback_buf().expect("readback buffer");
            let (_, parts) = s.readback_submit(&rb, base, base + 2 * TILE);
            s.wait_for(s.value);
            s.copy_back(p, &parts);

            // Now incrementally commit [base, base + 4*PAGE) and clear its hole bits (as `committed` does).
            assert!(!VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let entry = s.mapped.get_mut(&base).unwrap();
            for i in 0..4 {
                entry.holes &= !(1u16 << i);
            }
            assert_eq!(entry.holes, 0xFF00);

            // Writing to the newly committed page is detected by `scan_page`.
            *(base as *mut u8) = 0x5A;
            let mut dirty = Vec::new();
            let page0 = std::slice::from_raw_parts(base as *const u8, PAGE);
            scan_page(page0, base, &mut entry.sent[0], &mut dirty);
            assert_eq!(dirty, vec![(base, CHUNK)]);

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A tile decommitted and committed again before the next submit gets its pool tile back
    /// with no new tile mapping (cleared instead). One that is not is unmapped and freed by
    /// `flush_parked` (before a submit), or when the free list runs out; a placed buffer's
    /// release does not free it while it is parked.
    #[test]
    fn a_tile_decommitted_and_committed_again_keeps_its_video_memory() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queues: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                quarantined: VecDeque::new(),
                released: Vec::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: None,
                scratch_runs: Vec::new(),
                lazy_bufs: Vec::new(),
            };

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            let v = s.value;
            assert!(v > 0, "a new tile is mapped");

            // decommitted: parked, still mapped, nothing sent to the GPU
            s.decommit_range(base, TILE);
            assert!(!s.mapped.contains_key(&base));
            assert_eq!(s.parked.get(&base), Some(&tile));
            assert!(!s.free.contains(&tile));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.release_if_unused(base, tile);
            assert!(!s.free.contains(&tile), "a parked tile is not free");

            // committed again: the same tile, cleared, no new mapping
            s.to_zero.clear();
            let gen = mapped_gen();
            s.commit_range(base, TILE);
            assert_eq!(s.mapped[&base].pool, tile);
            assert_eq!(s.mapped[&base].holes, 0);
            assert!(s.parked.is_empty());
            assert_eq!(s.to_zero, vec![(base, TILE)]);
            assert_eq!(s.value, v, "no tile mapping changed");
            assert!(mapped_gen() > gen, "readers of `mapped` see it changed");

            // decommitted and flushed (as before a submit): unmapped and free once unmap fence completes
            s.decommit_range(base, TILE);
            assert!(s.flush_parked());
            assert!(s.parked.is_empty());
            assert!(s.value > v, "the unmapping is sent");
            assert!(!s.flush_parked(), "nothing left to flush");
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert!(s.free.contains(&tile));

            // no free tile left: the parked one is flushed and reused once its unmap fence finishes
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            s.wait_for(s.value);
            s.commit_range(base + TILE, TILE);
            assert!(s.parked.is_empty());
            assert_eq!(s.mapped[&(base + TILE)].pool, tile);
            assert!(!s.mapped.contains_key(&base));

            // a placed buffer uses the tile: the window keeps it when it is decommitted and
            // flushed, memory committed there again is the buffer's, and it is free only once
            // the buffer is gone and it was decommitted
            *s.pinned.entry(tile).or_default() += 1;
            let at = base + TILE;
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked(), "nothing to unmap");
            assert_eq!(s.kept.get(&at), Some(&tile));
            assert!(!s.free.contains(&tile));
            let v = s.value;
            s.to_zero.clear();
            s.commit_range(at, TILE);
            assert_eq!(s.mapped[&at].pool, tile, "the buffer's tile");
            assert_eq!(s.to_zero, vec![(at, TILE)]);
            assert_eq!(s.value, v, "no tile mapping changed");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked());
            s.pinned.remove(&tile);
            s.release_if_unused(at, tile);
            assert!(s.kept.is_empty() && !s.free.contains(&tile));
            assert!(s.flush_parked());
            // (main's quarantine: free once the unmap is done on the sparse queue)
            assert!(!s.free.contains(&tile));
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert!(s.free.contains(&tile));

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    struct SignalOnDrop(ID3D12Fence, u64);

    impl Drop for SignalOnDrop {
        fn drop(&mut self) {
            unsafe {
                let _ = self.0.Signal(self.1);
            }
        }
    }

    unsafe fn test_state_with_reserved(tiles: usize) -> (usize, State) {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::MEM_RESERVE;

        let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
        }
        let dev = dev.expect("D3D12 device");
        let base = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
        assert!(base != 0 && base % TILE == 0);
        let mut reserved: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&buffer_desc((tiles * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
            .expect("CreateReservedResource");
        let reserved = reserved.expect("reserved buffer");
        // (lab: the state as `ensure` makes it – `new_state` – so its fields stay in one place)
        let s = new_state(&dev, vec![(base, reserved)]).expect("sparse state");
        (base, s)
    }

    /// `flush_parked` must keep unmapped pool tiles out of `free` (and `take_tile` must not reuse
    /// them) while the sparse queue's unmap fence is still in flight on the GPU.
    #[test]
    fn flush_parked_does_not_free_or_reuse_tiles_while_unmap_is_in_flight() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
            let _unblock = SignalOnDrop(gate.clone(), 1);
            s.queue.Wait(&gate, 1).expect("gate sparse queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            assert!(!s.free.contains(&tile), "unmapped tile must remain quarantined until unmap fence completes");

            // While the unmap is still blocked on `gate`, committing another tile must not steal `tile`.
            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "in-flight unmapped tile must not be reused before GPU completion");

            gate.Signal(1).expect("release sparse queue");
            s.wait_for(s.value);
            assert!(s.drain_quarantined(), "completed unmap batch should drain after fence completion");
            assert!(s.free.contains(&tile), "unmapped tile must return to free list once unmap fence completes");
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// lab (`XCL_SPARSE_COOL=1`): a pool tile a buffer gives back is not free at once: it is
    /// quarantined at the next flush, behind the work submitted before, as parked tiles are.
    #[test]
    fn a_tile_a_buffer_gives_back_is_quarantined_with_the_cool_switch() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            s.commit_range(base, TILE);
            s.wait_for(s.value);
            // a tile only a buffer used (its window maps another one at the address)
            let tile = s.mapped[&base].pool;
            let other = (tile.0, tile.1 + 1);
            s.free.retain(|&t| t != other);
            s.release_if_unused_as(base, other, false);
            assert!(s.free.contains(&other), "without the switch: free at once");
            s.free.retain(|&t| t != other);

            // a title queue whose work submitted before is still running
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            assert!(s.note_queue(&title_q));
            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
            let _unblock = SignalOnDrop(gate.clone(), 1);
            title_q.Wait(&gate, 1).expect("gate title queue");
            s.release_if_unused_as(base, other, true);
            s.release_if_unused_as(base, other, true);
            assert_eq!(s.released, vec![other], "once");
            assert!(!s.free.contains(&other));
            assert!(!s.flush_parked(), "nothing parked to unmap");
            assert!(s.released.is_empty());
            assert!(!s.drain_quarantined(), "the title queue is held: still quarantined");
            assert!(!s.free.contains(&other));
            gate.Signal(1).expect("release title queue");
            let t0 = std::time::Instant::now();
            while !s.drain_quarantined() {
                assert!(t0.elapsed().as_secs() < 5, "the title queue went on");
                std::thread::yield_now();
            }
            assert!(s.free.contains(&other), "free once the work before is done");
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `flush_parked` must keep unmapped pool tiles quarantined until every active title queue has
    /// finished its prior work, even after `s.queue` itself finishes unmapping.
    #[test]
    fn flush_parked_quarantines_tiles_until_title_queues_finish() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            assert!(s.note_queue(&title_q));

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let title_gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("title gate fence");
            let _unblock = SignalOnDrop(title_gate.clone(), 1);
            title_q.Wait(&title_gate, 1).expect("gate title queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());

            // Even after `s.queue` finishes `unmap_tiles`, `title_q` is still gated behind `title_gate`.
            s.wait_for(s.value);
            assert!(!s.drain_quarantined(), "tile must remain quarantined while title queue has work in flight");
            assert!(!s.free.contains(&tile));

            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "must not reuse tile still in use by title queue");

            title_gate.Signal(1).expect("release title queue");
            let start = std::time::Instant::now();
            while !s.drain_quarantined() && start.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert!(s.free.contains(&tile), "tile must return to free list after title queue completes");
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// Sub-tile decommits (two 32 KiB halves, sixteen 4 KiB pages, or decommitting the last
    /// remaining committed page) must unmap the tile and return unpinned pool tiles to `free`
    /// once `holes` reaches `0xFFFF`, pruning stale `to_zero` entries for unmapped tiles.
    #[test]
    fn decommit_sub_tile_reclaims_pool_tile_when_all_pages_become_holes() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let mut pinned: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut free: Vec<(u32, u32)> = Vec::new();
        let mut to_zero: Vec<(usize, usize)> = Vec::new();

        // 1. Two 32 KiB decommits across a full 64 KiB tile.
        let t0 = 0x10_0000usize;
        mapped.insert(t0, TileEntry::with_holes((0, 10), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0, TILE / 2).is_empty());
        assert_eq!(mapped[&t0].holes, 0x00FF);
        assert_eq!(to_zero.len(), 8);
        assert!(free.is_empty());

        let unmapped = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0 + TILE / 2, TILE / 2);
        assert_eq!(unmapped, vec![t0]);
        assert!(!mapped.contains_key(&t0));
        assert_eq!(free, vec![(0, 10)]);
        assert!(to_zero.is_empty(), "stale to_zero entries for the unmapped tile must be pruned");

        // 2. Sixteen individual 4 KiB decommits in reverse order.
        let t1 = 0x20_0000usize;
        free.clear();
        mapped.insert(t1, TileEntry::with_holes((0, 11), 0));
        for i in (1..PAGES_PER_TILE).rev() {
            let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1 + i * PAGE, PAGE);
            assert!(u.is_empty(), "tile still has committed pages after step {i}");
        }
        assert!(free.is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1, PAGE);
        assert_eq!(u, vec![t1]);
        assert!(!mapped.contains_key(&t1));
        assert_eq!(free, vec![(0, 11)]);
        assert!(to_zero.is_empty());

        // 3. Pinned tile decommitted in two halves: unmapped from `mapped`, NOT returned to `free`.
        let t2 = 0x30_0000usize;
        free.clear();
        mapped.insert(t2, TileEntry::with_holes((0, 12), 0));
        pinned.insert((0, 12), 1);
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2, TILE / 2).is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2 + TILE / 2, TILE / 2);
        assert_eq!(u, vec![t2]);
        assert!(!mapped.contains_key(&t2));
        assert!(free.is_empty(), "pinned pool tile must not return to free list");
        assert!(to_zero.is_empty());

        // 4. Single-committed-page tile decommit reclaims tile without dropping another tile's to_zero.
        let t3 = 0x40_0000usize;
        let t4 = 0x50_0000usize;
        mapped.insert(t3, TileEntry::with_holes((0, 13), 0xFFFF & !(1u16 << 2)));
        mapped.insert(t4, TileEntry::with_holes((0, 14), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t4, PAGE).is_empty());
        assert_eq!(to_zero, vec![(t4, PAGE)]);
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t3 + 2 * PAGE, PAGE);
        assert_eq!(u, vec![t3]);
        assert_eq!(free, vec![(0, 13)]);
        assert_eq!(to_zero, vec![(t4, PAGE)], "unrelated mapped tile's to_zero must remain");
    }

    /// The pages in `early`, checking that its ranges are page-aligned, not empty, disjoint and
    /// merged (no two touch).
    fn early_pages(early: &BTreeMap<usize, usize>) -> std::collections::BTreeSet<usize> {
        let mut last_end = None;
        let mut pages = std::collections::BTreeSet::new();
        for (&a, &b) in early {
            assert!(a % PAGE == 0 && b % PAGE == 0 && a < b, "range {a:#x}..{b:#x}");
            assert!(last_end.is_none_or(|e| e < a), "range {a:#x}..{b:#x} touches the one before");
            last_end = Some(b);
            pages.extend((a..b).step_by(PAGE));
        }
        pages
    }

    /// xorshift64*: the same sequence on every run
    fn next(x: &mut u64) -> u64 {
        *x ^= *x >> 12;
        *x ^= *x << 25;
        *x ^= *x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Before the device exists, commits and decommits leave the pages a plain page set would
    /// hold, with the live path's rule (a range counts every page it touches): random ranges,
    /// unaligned too, over 300 pages.
    #[test]
    fn early_commits_are_the_pages_a_page_set_holds() {
        let base = 0x4_2000_0000usize;
        for seed in 1..=8u64 {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let (mut early, mut model) = (BTreeMap::new(), std::collections::BTreeSet::new());
            for _ in 0..4000 {
                let a = base + (next(&mut x) % (300 * PAGE as u64)) as usize;
                let n = match next(&mut x) % 4 {
                    0 => PAGE,
                    1 => 1 + (next(&mut x) % 64) as usize,
                    _ => 1 + (next(&mut x) % (24 * PAGE as u64)) as usize,
                };
                let (start, end) = page_span(a, n);
                let pages = (start..end).step_by(PAGE);
                if next(&mut x) % 3 == 0 {
                    early_remove(&mut early, start, end);
                    pages.for_each(|p| {
                        model.remove(&p);
                    });
                } else {
                    early_add(&mut early, start, end);
                    model.extend(pages);
                }
                assert_eq!(early_pages(&early), model, "seed {seed}");
            }
        }
    }

    /// The same pages as the list this replaced, for what that list handled exactly: one
    /// entry per committed page (`XMemMapPhysicalPagesScatter`, `home_again`) and page-aligned
    /// decommits of any length.
    #[test]
    fn early_commits_match_the_old_list_for_page_entries() {
        let base = 0x4_2000_0000usize;
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        let (mut early, mut old): (BTreeMap<usize, usize>, Vec<(usize, usize)>) = (BTreeMap::new(), Vec::new());
        for _ in 0..20000 {
            let a = base + (next(&mut x) % 512) as usize * PAGE;
            if next(&mut x) % 2 == 0 {
                old.push((a, PAGE));
                early_add(&mut early, a, a + PAGE);
            } else {
                let n = (1 + next(&mut x) % 8) as usize * PAGE;
                old.retain(|&(y, m)| y + m <= a || y >= a + n);
                early_remove(&mut early, a, a + n);
            }
        }
        let old_pages: std::collections::BTreeSet<usize> = old.iter().map(|&(y, _)| y).collect();
        assert_eq!(early_pages(&early), old_pages);
    }

    /// A range committed in one call keeps its other pages when one page of it is decommitted
    /// (`commit_homes` commits several pages, `move_home_away` gives one back). The list this
    /// replaced dropped the whole entry, so `init` left those still-committed pages as holes.
    #[test]
    fn early_decommit_of_one_page_keeps_the_rest() {
        let a = 0x4_2000_0000usize;
        let mut early = BTreeMap::new();
        early_add(&mut early, a, a + 4 * TILE);
        early_remove(&mut early, a + TILE, a + 2 * TILE);
        assert_eq!(early.into_iter().collect::<Vec<_>>(), vec![(a, a + TILE), (a + 2 * TILE, a + 4 * TILE)]);
    }

    /// Time for n 64 KiB pages committed one at a time and decommitted one at a time in
    /// reverse, before the device exists (`cargo test --release -p xcl --lib --
    /// --ignored --exact hle::d3d12x::sparse::tests::early_commit_timing --nocapture`,
    /// `XCL_EARLY_PAGES=16384,32768`). Run it alone: a test that made the device keeps a state.
    #[test]
    #[ignore]
    fn early_commit_timing() {
        if STATE.lock().unwrap().is_some() {
            return eprintln!("early_commit_timing: a sparse state exists – run it alone");
        }
        let sizes: Vec<usize> = std::env::var("XCL_EARLY_PAGES").ok().map_or(vec![4096, 8192, 16384, 32768], |v| v.split(',').map(|s| s.trim().parse().unwrap()).collect());
        let base = 0x4_2000_0000usize;
        for n in sizes {
            let t = std::time::Instant::now();
            for i in 0..n {
                unsafe { committed(base + i * TILE, TILE) };
            }
            let commit = t.elapsed();
            let t = std::time::Instant::now();
            for i in (0..n).rev() {
                unsafe { decommitted(base + i * TILE, TILE) };
            }
            let decommit = t.elapsed();
            println!("early: {n} pages: commit {:.1} ms, decommit {:.1} ms", commit.as_secs_f64() * 1e3, decommit.as_secs_f64() * 1e3);
        }
    }

    /// A D3D12 device for the lab switch tests (WARP if there is no GPU).
    unsafe fn lab_device() -> ID3D12Device {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
        }
        dev.expect("D3D12 device")
    }

    /// Puts a lab switch back to "read from the environment" and `RB_PENDING` back, also when
    /// a test panics.
    struct LabRestore(fn(), usize);

    impl Drop for LabRestore {
        fn drop(&mut self) {
            (self.0)();
            RB_PENDING.fetch_sub(self.1, Ordering::Release);
        }
    }

    /// lab `XCL_FENCE_DIRECT=1`: XCL's own probe of an adopted fence reads the same value with
    /// the switch on and off. Off it is the COM call, the title's hook: with copies back pending
    /// and the sparse state held (here by this thread, as `drain_quarantined` holds it), the hook
    /// tries the lock for ~2 ms. On it reads the PC's value and does not wait for that lock.
    #[test]
    fn a_direct_fence_probe_reads_the_same_value_without_the_hooks_wait() {
        use crate::hle::d3d12x::real;
        unsafe {
            let dev = lab_device();
            let fence: ID3D12Fence = dev.CreateFence(5, D3D12_FENCE_FLAG_NONE).expect("fence");
            fence.Signal(9).expect("CPU signal");
            real::adopt(fence.as_raw(), real::Kind::Fence);
            assert!(real::is_adopted(fence.as_raw()), "the fence is adopted (the title's hooks)");
            RB_PENDING.fetch_add(1, Ordering::Release);
            let _restore = LabRestore(|| real::set_fence_direct(None), 1);
            let held = STATE.lock().unwrap_or_else(|e| e.into_inner());
            real::set_fence_direct(Some(false));
            let t = std::time::Instant::now();
            let off = real::probe_completed(&fence);
            let took_off = t.elapsed();
            real::set_fence_direct(Some(true));
            let t = std::time::Instant::now();
            let on = real::probe_completed(&fence);
            let took_on = t.elapsed();
            drop(held);
            assert_eq!((off, on), (9, 9), "the same completed value, switch off and on");
            assert_eq!(fence.GetCompletedValue(), 9, "the COM call agrees");
            assert!(took_off >= std::time::Duration::from_millis(2), "off: the probe is the title's hook, which tries the lock ~2 ms ({took_off:?})");
            assert!(took_on < std::time::Duration::from_millis(1), "on: the probe does not wait for the sparse state's lock ({took_on:?})");
            println!("fence probe with the state held and a copy back pending: off {took_off:?}, on {took_on:?}");
        }
    }

    /// An upload buffer of `words` dwords, `f(i)` in dword `i`.
    unsafe fn lab_upload(dev: &ID3D12Device, words: usize, f: impl Fn(u32) -> u32) -> ID3D12Resource {
        let mut up: Option<ID3D12Resource> = None;
        let mut d = buffer_desc((words * 4) as u64);
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up)
            .expect("upload buffer");
        let up = up.expect("upload buffer");
        let mut p: *mut c_void = std::ptr::null_mut();
        up.Map(0, None, Some(&mut p)).expect("Map");
        for i in 0..words {
            *(p as *mut u32).add(i) = f(i as u32);
        }
        up
    }

    /// lab `XCL_SPARSE_MAP_NOWAIT=1`: a Map's copy back on the Map copy queue reads what the copy
    /// on the state's queue reads (same parts, same bytes; the tile without video memory left
    /// out by both), and only once all that was on the state's queue before it is done.
    #[test]
    fn a_map_copy_on_the_copy_queue_reads_what_the_state_queue_copy_reads() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;
        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            // tile 0: committed, with video memory; tile 1: neither
            assert!(!VirtualAlloc(base as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            s.commit_range(base, TILE);
            assert!(s.any_mapped(base, 1) && s.any_mapped(base + TILE - 1, 2) && s.any_mapped(base, 2 * TILE));
            assert!(!s.any_mapped(base + TILE, TILE), "tile 1 has no video memory");
            let reserved = s.windows[0].1.clone();
            let dev = s.dev.clone();

            // the title's GPU writes 0..1024 of tile 0, and is done
            let up = lab_upload(&dev, 512, |i| if i < 256 { 0xA000_0000 | i } else { 0xB000_0000 | i });
            let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&reserved, 0, &up, 0, 1024);
            list.Close().expect("close");
            title.Wait(&s.fence, s.value).expect("wait for the tile mapping");
            title.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            let own: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("own fence");
            title.Signal(&own, 1).expect("own signal");
            assert!(wait_fence(&own, 1), "the title's copy finishes");
            std::ptr::write_bytes(base as *mut u8, 0, 1024);

            // switch off: the copy on the state's queue (`readback_released`'s steps)
            let (rb, p) = s.readback_buf().expect("readback buffer");
            let (next_off, parts_off) = s.readback_submit(&rb, base, base + 2 * TILE);
            s.wait_for(s.value);
            s.copy_back(p, &parts_off);
            let off: Vec<u32> = (0..256).map(|i| *(base as *const u32).add(i)).collect();
            assert_eq!(off[0], 0xA000_0000);
            assert_eq!(off[255], 0xA000_00FF);
            std::ptr::write_bytes(base as *mut u8, 0, 1024);

            // switch on: the same range on the Map copy queue
            let mut m = MapCopy::new(&dev).expect("Map copy queue");
            assert!(!super::super::real::is_adopted(m.queue.as_raw()), "the copy queue is not adopted");
            let (next_on, parts_on) = s.map_copy_submit(&mut m, base, base + 2 * TILE).expect("ordered after the state's queue");
            assert!(wait_fence(&m.fence, m.value), "the Map copy finishes");
            s.copy_back(m.ptr, &parts_on);
            let on: Vec<u32> = (0..256).map(|i| *(base as *const u32).add(i)).collect();
            assert_eq!((next_on, &parts_on), (next_off, &parts_off), "the same parts");
            assert_eq!(on, off, "the same bytes, switch on and off");

            // ordering: work on the state's queue held by a gate (a CPU write being uploaded) comes
            // first; the Map copy reads it once the gate opens, not before
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate");
            let alloc2: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc2, None).expect("list");
            list2.CopyBufferRegion(&reserved, 0, &up, 1024, 1024);
            list2.Close().expect("close");
            s.queue.Wait(&gate, 1).expect("gate the state's queue");
            s.queue.ExecuteCommandLists(&[Some(list2.cast::<ID3D12CommandList>().unwrap())]);
            let (_, parts) = s.map_copy_submit(&mut m, base, base + 1024).expect("ordered after the state's queue");
            assert_eq!(parts, vec![(base, 0, 1024)]);
            std::thread::sleep(std::time::Duration::from_millis(200));
            assert!(super::super::real::fence_completed(&m.fence) < m.value, "the Map copy does not run before the state's queue gets past the gate");
            gate.Signal(1).expect("open the gate");
            assert!(wait_fence(&m.fence, m.value), "the Map copy finishes after the gate");
            s.copy_back(m.ptr, &parts);
            for i in 0..256u32 {
                assert_eq!(*(base as *const u32).add(i as usize), 0xB000_0000 | (256 + i), "dword {i}: the upload ahead of the copy on the state's queue");
            }
            s.wait_for(s.value);
            drop(m);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// lab `XCL_SPARSE_MAP_NOWAIT`: off unless the environment says 1 (the tests run with it unset).
    #[test]
    fn the_map_switch_is_off_unless_set() {
        set_map_nowait(None);
        if std::env::var_os("XCL_SPARSE_MAP_NOWAIT").is_none() {
            assert!(!map_nowait(), "off by default");
        }
        set_map_nowait(Some(true));
        assert!(map_nowait());
        set_map_nowait(None);
    }

    /// lab `XCL_USE_RANGE_HIT`: the index answers what the scan under the lock finds, for random
    /// buffers (overlapping ones too) and random ranges.
    #[test]
    fn the_lazy_index_finds_what_the_scan_finds() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        for round in 0..300 {
            let n = (next(&mut x) % 12) as usize;
            let bufs: Vec<(usize, usize, usize)> = (0..n)
                .map(|i| {
                    let start = (next(&mut x) % 64) as usize * TILE;
                    let bytes = 1 + (next(&mut x) % (3 * TILE as u64)) as usize;
                    (0x1000 + i, start, bytes)
                })
                .collect();
            let index = lazy_index_build(&bufs);
            for _ in 0..200 {
                let va = (next(&mut x) % (70 * TILE as u64)) as usize;
                let end = va + 1 + (next(&mut x) % (2 * TILE as u64)) as usize;
                let scan = bufs.iter().any(|&(_, start, bytes)| !(va >= start + bytes || end <= start));
                assert_eq!(lazy_index_hits(&index, va, end), scan, "round {round}: [{va:#x}, {end:#x}) in {bufs:x?}");
            }
        }
        assert!(!lazy_index_hits(&[], 0, usize::MAX), "no buffers: no hit");
    }

    /// lab `XCL_USE_RANGE_HIT`: each address passes once; an address is never skipped the first
    /// time it comes.
    #[test]
    fn seen_addresses_pass_once() {
        let mut seen = SeenAddrs::new();
        let addrs = [0x10_0000_0000usize, 0x10_0000_0100, 0x10_0000_0000, 0x10_0001_0000, 0x10_0000_0100, 0x10_0000_0000];
        let passed: Vec<usize> = addrs.iter().copied().filter(|&a| seen.first(a)).collect();
        assert_eq!(passed, vec![0x10_0000_0000, 0x10_0000_0100, 0x10_0001_0000]);
        let mut seen = SeenAddrs::new();
        let mut x = 7u64;
        let mut firsts = std::collections::HashSet::new();
        for _ in 0..10_000 {
            let a = (next(&mut x) % 500) as usize * 0x40;
            if firsts.insert(a) {
                assert!(seen.first(a), "{a:#x} skipped the first time");
            } else {
                let _ = seen.first(a);
            }
        }
    }

    /// `f` on another thread while this one holds the sparse state: whether it finished within
    /// `ms` (then the state is let go, and the thread joined).
    fn finishes_within_while_the_state_is_held(ms: u64, f: impl FnOnce() + Send + 'static) -> bool {
        let held = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            f();
            let _ = tx.send(());
        });
        let done = rx.recv_timeout(std::time::Duration::from_millis(ms)).is_ok();
        drop(held);
        assert!(t.join().is_ok(), "the thread panicked");
        done
    }

    /// lab `XCL_USE_RANGE_HIT=1`: `use_range` of an address between the lazily placed buffers but
    /// in none of them returns while another thread holds the state; one in a buffer still takes
    /// the state, as does any address of the envelope with the switch off (the old path).
    #[test]
    fn use_range_outside_the_lazy_buffers_does_not_take_the_state() {
        const A: usize = 0x7e00_0000_0000;
        let bufs = [(0x1usize, A, TILE), (0x2usize, A + 8 * TILE, 2 * TILE)];
        let old = (LAZY_BUFS[0].load(Ordering::Acquire), LAZY_BUFS[1].load(Ordering::Acquire));
        let old_index = std::mem::replace(&mut *LAZY_INDEX.write().unwrap_or_else(|e| e.into_inner()), lazy_index_build(&bufs));
        LAZY_BUFS[0].store(A, Ordering::Release);
        LAZY_BUFS[1].store(A + 10 * TILE, Ordering::Release);
        let outside = A + 4 * TILE;
        let inside = A + 9 * TILE + 16;
        set_use_range_hit(Some(true));
        let on_outside = finishes_within_while_the_state_is_held(500, move || unsafe { use_range(outside, 1) });
        let on_inside = finishes_within_while_the_state_is_held(300, move || unsafe { use_range(inside, 1) });
        set_use_range_hit(Some(false));
        let off_outside = finishes_within_while_the_state_is_held(300, move || unsafe { use_range(outside, 1) });
        set_use_range_hit(None);
        LAZY_BUFS[0].store(old.0, Ordering::Release);
        LAZY_BUFS[1].store(old.1, Ordering::Release);
        *LAZY_INDEX.write().unwrap_or_else(|e| e.into_inner()) = old_index;
        assert!(on_outside, "on: an address in no lazy buffer does not wait for the state");
        assert!(!on_inside, "on: an address in a lazy buffer takes the state (to map it)");
        assert!(!off_outside, "off: every address of the envelope takes the state, as before");
    }
}
