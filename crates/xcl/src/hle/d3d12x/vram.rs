//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (32 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). The CPU never sees these textures
//! (on the console it could, but GPU-owned targets are not CPU-written).
//!
//! Arenas cost video memory for whole ranges, and a placement straddling two made a third over
//! both while the two lived on with their resources: a retail game's streaming targets, packed
//! back to back over 3.5 GB of its addresses, made 15 GB of arenas in two seconds – past the
//! card's memory and the system's commit, and the device was removed. So the first choice is
//! the console's own model (`Pages`): a 64 KiB page of video memory for each 64 KiB page of
//! the title's address a target covers, and each target a reserved (tiled) resource whose tiles
//! are mapped onto the pages of its addresses. Targets at one address share pages, a page
//! exists once however many targets cover it, and what no target covers takes nothing.
//! Arenas remain for what cannot be tiled (an address not 64 KiB aligned, a resource the
//! driver does not make reserved: D32S8, MSAA): heaps of the placement's size, shared by
//! placements inside them, let go 2 s after their last resource (256 MiB ones: 2.8 GB of D32S8).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// A heap of video memory for placements at title addresses `[base, base + size)`.
struct Arena<H> {
    dev: usize,
    base: usize,
    size: usize,
    /// the heap's alignment (a placement that needs more does not go in)
    align: usize,
    heap: H,
    /// resources placed in it
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    id: u64,
}

/// The arenas, apart from D3D12 (`H`: the heap; numbers in the tests).
struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas { list: Vec::new(), next_id: 0 }
    }

    /// Return expired heaps for `dev` (preserving insertion order) to retire/drop outside `ARENAS.lock()`.
    fn sweep_dropped_for(&mut self, dev: usize, now: std::time::Instant) -> Vec<H> {
        let is_expired = |a: &Arena<H>| a.dev == dev && a.users == 0 && a.idle_since.is_some_and(|t| now.duration_since(t) >= IDLE);
        if !self.list.iter().any(is_expired) {
            return Vec::new();
        }
        let mut dropped = Vec::new();
        let mut kept = Vec::with_capacity(self.list.len());
        for a in self.list.drain(..) {
            if is_expired(&a) {
                dropped.push(a.heap);
            } else {
                kept.push(a);
            }
        }
        self.list = kept;
        dropped
    }

    /// Place `bytes` at `va` (`align`) on `dev`: the newest arena of `dev` holding it at an offset
    /// `align` allows, else a new arena of its size, `make(size, align)`. Returns
    /// `((heap, offset, arena id, fresh), expired_heaps)`; None: no heap. Expired heaps are returned,
    /// not dropped, so the caller retires them outside `ARENAS.lock()`.
    fn take_for(&mut self, dev: usize, va: usize, bytes: usize, align: usize, now: std::time::Instant, make: impl FnOnce(usize, usize) -> Option<H>) -> (Option<(H, usize, u64, bool)>, Vec<H>) {
        let expired = self.sweep_dropped_for(dev, now);
        let fits = |a: &Arena<H>| a.dev == dev && a.align >= align && a.base <= va && va + bytes <= a.base + a.size && (va - a.base) % align == 0;
        let (i, fresh) = match self.list.iter().rposition(fits) {
            Some(i) => (i, false),
            None => {
                let size = bytes.div_ceil(align) * align;
                let Some(heap) = make(size, align) else { return (None, expired) };
                self.next_id += 1;
                self.list.push(Arena { dev, base: va, size, align, heap, users: 0, idle_since: None, id: self.next_id });
                (self.list.len() - 1, true)
            }
        };
        let a = &mut self.list[i];
        a.users += 1;
        a.idle_since = None;
        (Some((a.heap.clone(), va - a.base, a.id, fresh)), expired)
    }

    /// A resource in arena `id` released.
    fn release(&mut self, id: u64, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            a.users = a.users.saturating_sub(1);
            if a.users == 0 {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize { self.list.iter().map(|a| a.size).sum() }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
/// resource → its arena's id
static ARENA_OF: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
/// bytes of every arena made
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().bytes(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

pub(super) fn forget(resource: usize) {
    forget_small(resource);
    let placed = PLACED_TEXTURES.lock().unwrap().as_mut().is_some_and(|textures| textures.remove(&resource));
    if placed {
        forget_pages(resource);
        if let Some(id) = ARENA_OF.lock().unwrap().as_mut().and_then(|m| m.remove(&resource)) {
            ARENAS.lock().unwrap().release(id, std::time::Instant::now());
        }
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// A new arena's heap: `size` bytes of video memory for title addresses from `va`.
unsafe fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // 64 KiB, or 4 MiB for MSAA targets
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {va:#x}: CreateHeap: {e}");
        return None;
    }
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: video-memory arena #{n} for title addresses {va:#x}..{:#x} ({made} bytes of arenas made so far)", va + size);
    }
    heap
}

/// bytes of a page – of the title's addresses and of a tile
const PAGE: usize = 0x1_0000;
/// pages of a video-memory pool heap (64 MiB)
const POOL_PAGES: u32 = 1024;
/// a page no target uses any more is kept this long for the next target at its address (with
/// what it holds – a title may make a target again over memory it rendered to)
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// a pool tile given back is not reused before this (the GPU may still be at work on what the
/// title released)
const COOL: std::time::Duration = std::time::Duration::from_secs(2);
/// Page budget (`XCL_PAGE_BUDGET_MB`): a texture the CPU fills is not evicted before this
/// long after it was placed, remapped or used in a transition.
const EVICT_MIN_AGE: std::time::Duration = std::time::Duration::from_secs(5);
/// Page budget: textures smaller than this are never evicted (UI, fonts, small lookups).
const EVICT_MIN_TILES: u32 = 4;
/// Page budget: tiles freed per eviction round once the pools are at the budget.
const EVICT_BATCH: usize = 512;
/// Page budget: after a round that found nothing to evict, pools grow past the budget for this
/// long instead of trying again for every tile.
const EVICT_IDLE: std::time::Duration = std::time::Duration::from_secs(1);

/// Page budget: pool heaps the pages may take, from `XCL_PAGE_BUDGET_MB` (0: no limit).
fn pool_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let mb: usize = std::env::var("XCL_PAGE_BUDGET_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        if mb == 0 {
            return 0;
        }
        // evicting unmaps a texture from another thread: only safe while a texture's last
        // Release reaches `forget` before the object goes (`late_release` holds it until then)
        if !super::late_release::enabled() {
            crate::xlog_warn!("D3D12.X: XCL_PAGE_BUDGET_MB needs late release (XCL_LATE_RELEASE=0 is set): no page budget");
            return 0;
        }
        let pools = (mb << 20).div_ceil(POOL_PAGES as usize * PAGE).max(1);
        crate::xlog_info!("D3D12.X: page budget {mb} MB ({pools} pools of 64 MB): CPU-filled textures are evicted least recently used first beyond it");
        pools
    })
}

/// Page budget: textures whose tiles were evicted; the next upload or transition maps
/// them again and uploads them whole.
static EVICTED: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
/// the number of textures in `EVICTED` (callers skip the lookup while it is 0)
static EVICTED_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static EVICTIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REMAPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Page budget: textures with no T# in any view heap (while the heaps are looked at) may be
/// evicted this long after they were placed or last seen, small ones too.
const EVICT_UNSEEN_AGE: std::time::Duration = std::time::Duration::from_secs(2);
/// whether the console view heaps are looked at (`seen`): only then does "no T#" mean unused
static HEAPS_LOOKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Page budget: after a round that gave back fewer tiles than wanted (the rest of the
/// candidates' pages are shared), the next is not tried for this long: every tile taken at the
/// budget would otherwise sort and walk the same textures again.
const EVICT_RETRY: std::time::Duration = std::time::Duration::from_secs(1);

/// A pool tile: (pool heap, tile in it).
type Tile = (u32, u32);

struct Page {
    tile: Tile,
    /// targets using the page
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
}

/// A page of `Pages::pages` (its title address) or of `Pages::extra_pages` (its key there).
#[derive(Clone, Copy)]
enum PageKey {
    Own(usize),
    Extra(usize, usize),
}

/// The pages of `idle` – those whose last user went, oldest first, each with when – that no
/// resource has used for `IDLE` go back to the pools: out of their map, their tile to `cooling`.
/// An entry whose page was used again since (in use now, or idle again from a later moment, with
/// a later entry) is dropped. Only the queue is looked at: every page of both maps, looked at at
/// every placement, grew with the square of what a title streams.
fn sweep_idle(
    idle: &mut std::collections::VecDeque<(PageKey, std::time::Instant)>,
    pages: &mut std::collections::HashMap<usize, Page>,
    extra_pages: &mut std::collections::HashMap<(usize, usize), Page>,
    cooling: &mut std::collections::VecDeque<(Tile, std::time::Instant)>,
    now: std::time::Instant,
) {
    while idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
        let Some((key, since)) = idle.pop_front() else { break };
        let still_idle = |p: &Page| p.idle_since == Some(since);
        let page = match key {
            PageKey::Own(a) if pages.get(&a).is_some_and(still_idle) => pages.remove(&a),
            PageKey::Extra(va, i) if extra_pages.get(&(va, i)).is_some_and(still_idle) => extra_pages.remove(&(va, i)),
            _ => None,
        };
        if let Some(p) = page {
            cooling.push_back((p.tile, now));
        }
    }
}

struct Target {
    va: usize,
    console_bytes: Option<u64>,
    total: u32,
    mapped: bool,
    used: Vec<usize>,
    used_extra: Vec<(usize, usize)>,
    /// a texture the CPU fills (its texels can be uploaded again from the title's memory)
    evictable: bool,
    /// placed, remapped, transitioned or its T# seen in a shader-visible heap last
    touched: std::time::Instant,
    /// placed as GPU-owned: its pages are keyed by the physical page under them (`page_key`)
    gpu_owned: bool,
    /// where the last look at the view heaps found its T# (`seen`)
    seen: Seen,
    /// LOCAL (#1514): its owned pages in graphics reservations (`Pages::follow`)
    follows: Vec<usize>,
    /// LOCAL (`XCL_TEXTURE_FOLLOW=3`): its subresource groups (None: followed whole, or not
    /// mode 3). Tile i of a group keeps the bookkeeping key it has in every mode: `pages` at
    /// `first + i * PAGE` (i < own), else `extra_pages` at `(va, i)` – only an identifier of
    /// tile i there (its texels are not console page i's), never looked up as a title page
    groups: Option<std::sync::Arc<Vec<FollowGroup>>>,
}

/// LOCAL (#1514): a tile of no heap: the page it stands for is left unmapped (`Pages::map`).
const NO_TILE: Tile = (u32::MAX, 0);

/// LOCAL (#1514): `XCL_TEXTURE_FOLLOW=1`: the pages of sampled textures placed in the title's
/// graphics reservations get video memory only where the title has memory (`Follow`). Read once.
pub fn following() -> bool {
    follow_mode() != 0
}

/// LOCAL: `XCL_TEXTURE_FOLLOW`: 0 off, 1 per page (`follows_by_page` textures only), 2 whole
/// textures (any shape: all tiles mapped while any followed page has the title's memory, else
/// all NULL). Read once.
fn follow_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        // (first asked inside the title's VirtualAlloc: its last error stays as it was)
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
        let e = unsafe { GetLastError() };
        let mut mode = parse_follow_mode(std::env::var("XCL_TEXTURE_FOLLOW").ok().as_deref());
        // (textures are mapped again from the submitting thread, as an eviction does)
        if mode != 0 && !super::late_release::enabled() {
            crate::xlog_warn!("D3D12.X: XCL_TEXTURE_FOLLOW needs late release (XCL_LATE_RELEASE=0 is set): textures do not follow");
            mode = 0;
        } else if mode == 1 {
            crate::xlog_info!("D3D12.X: XCL_TEXTURE_FOLLOW=1: texture pages in graphics reservations follow the title's memory");
        } else if mode == 2 {
            crate::xlog_info!("D3D12.X: XCL_TEXTURE_FOLLOW=2: textures in graphics reservations follow the title's memory whole (mapped while any of their pages has memory)");
        } else if mode == 3 {
            // LOCAL
            crate::xlog_info!("D3D12.X: XCL_TEXTURE_FOLLOW=3: textures in graphics reservations follow the title's memory per subresource (a mip level's tiles mapped while any page of its console bytes has memory; packed mips together)");
        }
        unsafe { SetLastError(e) };
        mode
    })
}

/// LOCAL: the value of `XCL_TEXTURE_FOLLOW` → `follow_mode` (anything else: off).
fn parse_follow_mode(v: Option<&str>) -> u8 {
    match v.map(str::trim) {
        Some("1") => 1,
        Some("2") => 2,
        Some("3") => 3, // LOCAL
        _ => 0,
    }
}

/// LOCAL: whole-texture following (`XCL_TEXTURE_FOLLOW=2`).
fn follow_whole() -> bool {
    follow_mode() == 2
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=2`): whether a texture followed whole is backed: any of its
/// followed pages' `Follow::mapped`. None: no followed page (the texture does not follow).
fn whole_backed(page_mapped: impl IntoIterator<Item = bool>) -> Option<bool> {
    let mut any_page = false;
    for m in page_mapped {
        if m {
            return Some(true);
        }
        any_page = true;
    }
    any_page.then_some(false)
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=2`): what a texture followed whole changes to: Some(true) map all
/// its tiles, Some(false) NULL them all, None nothing. `holds_tiles`: it holds pages now.
fn whole_transition(backed: Option<bool>, holds_tiles: bool) -> Option<bool> {
    match backed {
        Some(b) if b != holds_tiles => Some(b),
        _ => None,
    }
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=3`): a group of a followed texture's PC tiles – one PC subresource
/// with tiles of its own, or all its packed mips (of every slice) – and the title's 64 KiB pages
/// over the console bytes of the same subresource(s). Its tiles are mapped together while any of
/// those pages has the title's memory (`group_backed`). A texture whose groups cannot be made
/// (`follow_groups_of`) is one group over all its tiles and its followed pages (`whole_group`):
/// `XCL_TEXTURE_FOLLOW=2`'s rule.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FollowGroup {
    /// (first overall tile, count), within the resource's tiles
    tiles: Vec<(u32, u32)>,
    /// the title's page addresses overlapping the console bytes, ascending
    pages: Vec<usize>,
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=3`): a group of PC tiles: its subresources (D3D12 order: mip +
/// slice × mips) and their tiles ((first overall tile, count)).
#[derive(Clone, Debug, PartialEq, Eq)]
struct TileGroup {
    subs: Vec<usize>,
    tiles: Vec<(u32, u32)>,
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=3`): the PC's tile groups of a texture from GetResourceTiling:
/// `subs[k]` the tiles of subresource k (Some((StartTileIndexInOverallResource, Width × Height
/// × DepthInTiles)), None: packed), `mips` × `array` subresources, `standard` / `packed` the
/// D3D12_PACKED_MIP_INFO mips, `packed_tiles` its NumTilesForPackedMips, `total` the
/// resource's tiles. One group per standard subresource; the packed group takes the tiles no
/// standard subresource has (each slice's packed tiles follow its standard mips), which must
/// be `packed_tiles` per slice (or in all). None: the tiling does not partition `[0, total)` so.
fn pc_tile_groups(subs: &[Option<(u32, u32)>], mips: usize, array: usize, standard: usize, packed: usize, packed_tiles: u32, total: u32) -> Option<Vec<TileGroup>> {
    if mips == 0 || array == 0 || subs.len() != mips * array || standard + packed != mips {
        return None;
    }
    let mut covered = vec![false; total as usize];
    let mut out = Vec::new();
    let mut packed_subs = Vec::new();
    for s in 0..array {
        for m in 0..mips {
            let k = m + s * mips;
            match (m < standard, subs[k]) {
                (true, Some((start, count))) if count > 0 => {
                    let end = start.checked_add(count).filter(|&e| e <= total)?;
                    for c in &mut covered[start as usize..end as usize] {
                        if std::mem::replace(c, true) {
                            return None;
                        }
                    }
                    out.push(TileGroup { subs: vec![k], tiles: vec![(start, count)] });
                }
                (false, None) => packed_subs.push(k),
                _ => return None,
            }
        }
    }
    let mut rest: Vec<(u32, u32)> = Vec::new();
    for (i, _) in covered.iter().enumerate().filter(|(_, c)| !**c) {
        match rest.last_mut() {
            Some(r) if r.0 + r.1 == i as u32 => r.1 += 1,
            _ => rest.push((i as u32, 1)),
        }
    }
    let n: u64 = rest.iter().map(|r| r.1 as u64).sum();
    let ok = if packed == 0 { n == 0 } else { n == packed_tiles as u64 * array as u64 || n == packed_tiles as u64 };
    if !ok {
        return None;
    }
    if n > 0 {
        out.push(TileGroup { subs: packed_subs, tiles: rest });
    }
    Some(out)
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=3`): the console bytes of each subresource (D3D12 order) of a 2D
/// surface `s` of a texture, from the texture's start (`at`: the surface's plane offset), as
/// `gpu::tiling::Chain` lays a slice out: the mip tail block first (`[0, block_bytes)`, shared by
/// every level in the tail), then the other levels from the smallest to mip 0 (level i up to the
/// start of level i − 1, mip 0 up to `slice_bytes`); slices `slice_bytes` apart.
fn console_sub_ranges(s: &crate::hle::xg::Surface, at: u64) -> Vec<(u64, u64)> {
    let c = &s.chain;
    let mips = c.levels.len();
    let level = |i: usize| -> (u64, u64) {
        if i >= s.first_tail {
            (0, c.layout.block_bytes.min(c.slice_bytes) as u64)
        } else {
            let end = if i == 0 { c.slice_bytes } else { c.levels[i - 1].offset };
            (c.levels[i].offset as u64, end as u64)
        }
    };
    let mut out = Vec::with_capacity(mips * s.array as usize);
    for slice in 0..s.array as u64 {
        let base = at + slice * c.slice_bytes as u64;
        for i in 0..mips {
            let (lo, hi) = level(i);
            out.push((base + lo, base + hi));
        }
    }
    out
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=3`): the tile groups of a texture at `va` with the title's pages
/// over their subresources' console bytes (`ranges`, from `console_sub_ranges`). None: a
/// subresource with no console range.
fn follow_groups(tiles: Vec<TileGroup>, ranges: &[(u64, u64)], va: usize) -> Option<Vec<FollowGroup>> {
    tiles
        .into_iter()
        .map(|g| {
            let mut pages = Vec::new();
            for &k in &g.subs {
                let &(lo, hi) = ranges.get(k)?;
                let (lo, hi) = (va.checked_add(usize::try_from(lo).ok()?)?, va.checked_add(usize::try_from(hi).ok()?)?);
                let mut p = lo & !(PAGE - 1);
                while p < hi {
                    pages.push(p);
                    p += PAGE;
                }
            }
            pages.sort_unstable();
            pages.dedup();
            Some(FollowGroup { tiles: g.tiles, pages })
        })
        .collect()
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=3`): whether a group is backed: any of its pages is (a page the
/// texture does not follow counts as backed: nothing tells of its changes). No page: backed.
fn group_backed(page_backed: impl IntoIterator<Item = bool>) -> bool {
    let mut any_page = false;
    for b in page_backed {
        if b {
            return true;
        }
        any_page = true;
    }
    !any_page
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=3`): the group of a texture without subresource groups: all its
/// `total` tiles, at its followed pages (mapped while any of them is backed, as mode 2).
fn whole_group(total: u32, follows: &[usize]) -> FollowGroup {
    FollowGroup { tiles: if total > 0 { vec![(0, total)] } else { Vec::new() }, pages: follows.to_vec() }
}

/// LOCAL (`XCL_TEXTURE_FOLLOW=3`): the subresource groups of the reserved texture `res` (`total`
/// tiles) placed at `va` for the console texture `console` (its D3D12_RESOURCE_DESC as the
/// console's, `console_bytes` its size there). None (followed whole instead): not a single-plane
/// 2D texture whose console mips and slices are the PC's, or a tiling `pc_tile_groups` refuses.
unsafe fn follow_groups_of(dev: &ID3D12Device, res: &ID3D12Resource, console: &[u8; 0x38], va: usize, console_bytes: Option<u64>, total: u32) -> Option<Vec<FollowGroup>> {
    let tex = crate::hle::xg::Texture::new(console).ok()?;
    if tex.planes.len() != 1 {
        return None;
    }
    let (s, at) = tex.plane_surface(tex.main as u32)?;
    let d = res.GetDesc();
    let (mips, array) = (d.MipLevels as usize, d.DepthOrArraySize as usize);
    if s.depth != 0 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || mips != s.chain.levels.len() || array != s.array as usize || total == 0 {
        return None;
    }
    let ranges = console_sub_ranges(s, at);
    let bytes = console_bytes?;
    if ranges.iter().any(|r| r.0 >= r.1 || r.1 > bytes) {
        return None;
    }
    let mut n = (mips * array) as u32;
    let mut all = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); n as usize];
    dev.GetResourceTiling(res, Some(&mut all), Some(&mut packed), None, Some(&mut n), 0, tilings.as_mut_ptr());
    if all != total || n as usize != mips * array {
        return None;
    }
    let subs: Vec<Option<(u32, u32)>> = tilings
        .iter()
        .map(|t| (t.StartTileIndexInOverallResource != D3D12_PACKED_TILE).then(|| (t.StartTileIndexInOverallResource, t.WidthInTiles * t.HeightInTiles as u32 * t.DepthInTiles as u32)))
        .collect();
    let groups = pc_tile_groups(&subs, mips, array, packed.NumStandardMips as usize, packed.NumPackedMips as usize, packed.NumTilesForPackedMips, total)?;
    follow_groups(groups, &ranges, va)
}

/// LOCAL (#1514): an owned page of placed textures in a graphics reservation: mapped only while
/// the title has memory there (committed, a physical page, or one mapped but not yet touched).
struct Follow {
    /// (texture, its tile at this page: the tile's index in the whole resource)
    users: Vec<(usize, u32)>,
    /// whether the textures' tiles there are on a page of video memory
    mapped: bool,
}

/// LOCAL (#1514): ranges of title memory a graphics reservation gained or lost memory in since
/// the last submit (`title_memory_changed`): the pages there are mapped or unmapped before the
/// next one.
static BACKING: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
/// LOCAL (#1514 port review): textures a page was mapped for again by `apply_backing`. The
/// tile may be recycled (another texture's texels) and the title may not write the page, so
/// `title_textures` uploads each whole at this submit (`take_reupload`).
static REUPLOAD: Mutex<Vec<usize>> = Mutex::new(Vec::new());

thread_local! {
    static TITLE_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// LOCAL: `before_execute` for the title's own submit (`title_textures::refresh_once`): only it
/// applies what the title changed of its memory to followed textures, as it then uploads the
/// textures whose tiles turned on (`take_reupload`) before the title's lists run.
pub(super) unsafe fn before_title_execute(queue: &ID3D12CommandQueue) {
    TITLE_SUBMIT.with(|c| c.set(true));
    before_execute(queue);
    TITLE_SUBMIT.with(|c| c.set(false));
}

pub(super) fn take_reupload() -> Vec<usize> {
    std::mem::take(&mut *REUPLOAD.lock().unwrap())
}

/// LOCAL: only textures whose PC tile i covers the bytes of console page i follow: one level,
/// one slice (the console keeps the mip tail first, the PC last – veyn-sudo on #1514).
unsafe fn follows_by_page(res: &ID3D12Resource) -> bool {
    let d = res.GetDesc();
    d.MipLevels == 1 && d.DepthOrArraySize == 1 && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
}
/// LOCAL (#1514): counts for the log: tiles left unmapped by `bind`, and tiles mapped and
/// unmapped by `apply_backing`
static FOLLOW_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// LOCAL (#1514): the title committed, mapped, decommitted or unmapped `[a, a + n)` of a graphics
/// reservation: the texture pages there follow it before the title's next submit
/// (`before_execute`). Nothing without `XCL_TEXTURE_FOLLOW=1`.
pub fn title_memory_changed(a: usize, n: usize) {
    if following() {
        note_changed(a, n);
    }
}

/// LOCAL (#1514): `title_memory_changed` past the switch. A CPU view's change is its GPU view's
/// too, and a GPU view's the CPU view's (a texture placed in a GPU view with nothing mapped there
/// is at the CPU view's address, `canonical_va`). (from 3L1A5-1's d656862a)
fn note_changed(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    // (the views are asked before `BACKING` is taken: `map_pages` calls this with `AWAY` held)
    let gpu = crate::hle::kernelx::mirrors::gpu_views_of(a);
    let cpu = crate::hle::kernelx::mirrors::source(a);
    let mut b = BACKING.lock().unwrap();
    b.push((a, n));
    b.extend(gpu.into_iter().chain(cpu).map(|x| (x, n)));
}

/// LOCAL (#1514): whether the title has memory at the 64 KiB page `a`: committed, a physical page
/// mapped there, or one mapped and not touched yet (`lazy`). A GPU view of the title's CPU memory
/// (`kernelx::mirrors`) shows what the queue's page-mapping method put under it, else a physical
/// page mapped there, else the memory of the CPU view behind it; an address of a CPU view (where
/// a texture of its GPU view is placed while nothing is mapped there, `canonical_va`) is backed
/// also when a GPU view of it shows memory. (3L1A5-1's d656862a form)
fn title_backed(a: usize) -> bool {
    use crate::hle::kernelx::{lazy, mirrors, physical_page_at};
    // what a GPU-view address shows (None: nothing says; the CPU view then)
    let shown = |g: usize| match mirrors::mapped_source(g) {
        Some(m) => Some(m.is_some()),
        None => physical_page_at(g).then_some(true),
    };
    if let Some(b) = shown(a) {
        return b;
    }
    let src = mirrors::source(a).unwrap_or(a);
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let committed = unsafe { VirtualQuery(src as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 && m.State == MEM_COMMIT;
    committed || lazy::is_lazy(src) || physical_page_at(src) || (src == a && mirrors::gpu_views_of(a).into_iter().any(|g| shown(g) == Some(true)))
}

/// LOCAL (#1514): the ranges of one UpdateTileMappings call over all `total` tiles of a resource:
/// `sel(i)` is the heap offset of tile i when this call sets it (Some), else it is skipped.
/// `flag` for the set runs (NONE: onto a heap; NULL: unmapped).
fn update_ranges(total: u32, flag: D3D12_TILE_RANGE_FLAGS, sel: impl Fn(u32) -> Option<u32>) -> Vec<(D3D12_TILE_RANGE_FLAGS, u32, u32)> {
    let mut out: Vec<(D3D12_TILE_RANGE_FLAGS, u32, u32)> = Vec::new();
    for i in 0..total {
        match (sel(i), out.last_mut()) {
            (Some(o), Some(r)) if r.0 == flag && (flag == D3D12_TILE_RANGE_FLAG_NULL || r.1 + r.2 == o) => r.2 += 1,
            (Some(o), _) => out.push((flag, if flag == D3D12_TILE_RANGE_FLAG_NULL { 0 } else { o }, 1)),
            (None, Some(r)) if r.0 == D3D12_TILE_RANGE_FLAG_SKIP => r.2 += 1,
            (None, _) => out.push((D3D12_TILE_RANGE_FLAG_SKIP, 0, 1)),
        }
    }
    out
}

/// Where a texture's T# was at the last look at the console view heaps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Seen {
    None,
    CpuOnly,
    Visible,
}

/// Per-heap free-tile allocator for `Pages`.
///
/// Keeping free tiles partitioned by pool heap (sorted descending so `.pop()` yields ascending
/// contiguous tile indices) prevents streaming churn across 100+ 64 MiB pools from scattering a
/// single texture across dozens of heaps — which otherwise explodes `Pages::map` into dozens of
/// `UpdateTileMappings` calls per texture.
struct TilePool {
    /// Free tile indices for each pool heap (descending when `!dirty_sort[h]`).
    free_by_heap: Vec<Vec<u32>>,
    /// Whether `free_by_heap[h]` must be sorted descending before popping.
    dirty_sort: Vec<bool>,
    /// Total free tiles currently available across all heaps.
    free_total: usize,
    /// Tiles given back, reusable after `COOL`.
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
}

impl TilePool {
    fn new() -> Self {
        TilePool {
            free_by_heap: Vec::new(),
            dirty_sort: Vec::new(),
            free_total: 0,
            cooling: std::collections::VecDeque::new(),
        }
    }

    fn add_heap_at(&mut self, slot: u32, pool_pages: u32) {
        let idx = slot as usize;
        if idx >= self.free_by_heap.len() {
            self.free_by_heap.resize_with(idx + 1, Vec::new);
            self.dirty_sort.resize(idx + 1, false);
        }
        debug_assert!(self.free_by_heap[idx].is_empty());
        self.free_by_heap[idx] = (0..pool_pages).rev().collect();
        self.dirty_sort[idx] = false;
        self.free_total += pool_pages as usize;
    }

    /// Reclaim completely empty cooled heaps (`free_by_heap[h].len() == pool_pages`), keeping up
    /// to `keep_spare` empty heaps warm so steady-state streaming never thrashes `CreateHeap`.
    /// Returns the reclaimed heap indices so the caller can drop their `ID3D12Heap`s.
    fn reclaim_empty_heaps(&mut self, keep_spare: usize, pool_pages: u32) -> Vec<u32> {
        let full = pool_pages as usize;
        let mut empty: Vec<usize> = self
            .free_by_heap
            .iter()
            .enumerate()
            .filter_map(|(h, tiles)| (tiles.len() == full).then_some(h))
            .collect();
        if empty.len() <= keep_spare {
            return Vec::new();
        }
        let mut reclaimed = Vec::with_capacity(empty.len() - keep_spare);
        for h in empty.drain(keep_spare..) {
            self.free_by_heap[h].clear();
            self.free_by_heap[h].shrink_to_fit();
            self.dirty_sort[h] = false;
            self.free_total -= full;
            reclaimed.push(h as u32);
        }
        reclaimed
    }

    fn push_cooling(&mut self, tile: Tile, now: std::time::Instant) {
        self.cooling.push_back((tile, now));
    }

    /// Move every cooled tile (`>= COOL`) into its pool heap's free list so best-fit allocation
    /// sees all ready tiles at once.
    fn drain_cooling(&mut self, now: std::time::Instant) {
        while self.cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL) {
            let Some(((h, t), _)) = self.cooling.pop_front() else { break };
            let idx = h as usize;
            if idx < self.free_by_heap.len() {
                self.free_by_heap[idx].push(t);
                self.dirty_sort[idx] = true;
                self.free_total += 1;
            }
        }
    }

    fn return_tiles(&mut self, tiles: &[Tile]) {
        for &(h, t) in tiles {
            let idx = h as usize;
            if idx < self.free_by_heap.len() {
                self.free_by_heap[idx].push(t);
                self.dirty_sort[idx] = true;
                self.free_total += 1;
            }
        }
    }

    /// Pick the pool heap to take up to `rem` tiles from:
    /// 1. `preferred` if it fits all `rem` tiles (keeps partial-overlap placements on 1 heap),
    /// 2. otherwise best-fit heap with `len >= rem` (fits the whole texture in 1 heap without
    ///    splitting fuller pools),
    /// 3. otherwise the heap with the most free tiles (minimizing distinct heaps touched).
    fn pick_heap(&self, rem: usize, preferred: Option<u32>) -> Option<usize> {
        if self.free_total == 0 {
            return None;
        }
        if let Some(p) = preferred.map(|h| h as usize).filter(|&h| h < self.free_by_heap.len()) {
            if self.free_by_heap[p].len() >= rem {
                return Some(p);
            }
        }
        let mut best_fit: Option<(usize, usize)> = None;
        let mut largest: Option<(usize, usize)> = None;
        for (h, tiles) in self.free_by_heap.iter().enumerate() {
            let len = tiles.len();
            if len == 0 {
                continue;
            }
            if len >= rem {
                if best_fit.is_none_or(|(_, blen)| len < blen) {
                    best_fit = Some((h, len));
                }
            } else if largest.is_none_or(|(lh, llen)| len > llen || (len == llen && preferred == Some(h as u32) && preferred != Some(lh as u32))) {
                largest = Some((h, len));
            }
        }
        best_fit.or(largest).map(|(h, _)| h)
    }

    /// Allocate `count` tiles, preferring a single heap and ascending contiguous offsets.
    /// `make_heap` allocates a new backing pool heap of `pool_pages` tiles when `free_total == 0`.
    /// On failure, any tiles already taken are returned untouched.
    fn take_batch(
        &mut self,
        count: usize,
        preferred: Option<u32>,
        now: std::time::Instant,
        pool_pages: u32,
        mut make_heap: impl FnMut() -> Option<u32>,
    ) -> Option<Vec<Tile>> {
        if count == 0 {
            return Some(Vec::new());
        }
        self.drain_cooling(now);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let rem = count - out.len();
            let h = match self.pick_heap(rem, preferred) {
                Some(h) => h,
                None => {
                    let Some(new_h) = make_heap() else {
                        self.return_tiles(&out);
                        return None;
                    };
                    self.add_heap_at(new_h, pool_pages);
                    new_h as usize
                }
            };
            if self.dirty_sort[h] {
                self.free_by_heap[h].sort_unstable_by(|a, b| b.cmp(a));
                self.dirty_sort[h] = false;
            }
            let free = &mut self.free_by_heap[h];
            let take = rem.min(free.len());
            for _ in 0..take {
                let t = free.pop().unwrap();
                out.push((h as u32, t));
            }
            self.free_total -= take;
        }
        Some(out)
    }
}

/// Allocate one 64 MiB default heap for `Pages`.
unsafe fn alloc_pool_heap(dev: &ID3D12Device) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Alignment: 0,
        // every kind of texture (resource heap tier 2: the only flags for both)
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&desc, &mut heap)) {
        crate::xlog_warn!("D3D12.X: no video memory for more pages of title textures ({e})");
        return None;
    }
    heap
}

/// Backing 64 MiB default heaps for `Pages`, supporting slot reuse when completely empty heaps
/// are reclaimed back to local VRAM. `len()` reports the count of live heaps.
struct PoolHeaps {
    slots: Vec<Option<ID3D12Heap>>,
    free_slots: Vec<u32>,
    live: usize,
}

impl PoolHeaps {
    fn new() -> Self {
        Self { slots: Vec::new(), free_slots: Vec::new(), live: 0 }
    }

    fn len(&self) -> usize {
        self.live
    }

    fn push_heap(&mut self, heap: ID3D12Heap) -> u32 {
        self.live += 1;
        if let Some(idx) = self.free_slots.pop() {
            debug_assert!(self.slots[idx as usize].is_none());
            self.slots[idx as usize] = Some(heap);
            idx
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Some(heap));
            idx
        }
    }

    fn take_heap(&mut self, idx: u32) -> Option<ID3D12Heap> {
        let h = self.slots.get_mut(idx as usize)?.take()?;
        self.live -= 1;
        self.free_slots.push(idx);
        Some(h)
    }
}

impl std::ops::Index<usize> for PoolHeaps {
    type Output = ID3D12Heap;
    fn index(&self, idx: usize) -> &Self::Output {
        self.slots[idx].as_ref().expect("live VRAM pool heap")
    }
}

fn keep_spare_pools() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("XCL_VRAM_KEEP_SPARE_POOLS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1)
    })
}

/// The title's pages of video memory for tiled textures (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    /// per-device timestamp of the last periodic reclaim check
    last_reclaim: std::time::Instant,
    pools: PoolHeaps,
    pool: TilePool,
    /// title page address → its video memory
    pages: std::collections::HashMap<usize, Page>,
    /// extra / sub-64 KiB tiles keyed by `(va, tile_index)` so repeated placements at the same
    /// address share them without aliasing unrelated addresses
    extra_pages: std::collections::HashMap<(usize, usize), Page>,
    /// pages of both maps whose last user went, oldest first, with when (`sweep_idle`)
    idle: std::collections::VecDeque<(PageKey, std::time::Instant)>,
    /// resource → its mapping metadata and currently held pages / extra keys
    targets: std::collections::HashMap<usize, Target>,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
    /// cached `reservable()` answers for `(DXGI_FORMAT, SampleCount)` on this device
    reservable_cache: std::collections::HashMap<(i32, u32), bool>,
    /// reusable scratch buffers for `map()`
    scratch_heaps: Vec<u32>,
    scratch_flags: Vec<D3D12_TILE_RANGE_FLAGS>,
    scratch_offsets: Vec<u32>,
    scratch_counts: Vec<u32>,
    /// page budget: no eviction round before this (the last one found nothing to evict, or
    /// fell short: `EVICT_RETRY`)
    evict_idle_until: Option<std::time::Instant>,
    /// page budget: pool heaps the pages may take (`pool_cap`; 0: no limit)
    cap: usize,
    /// LOCAL (#1514): owned pages of textures in graphics reservations → their textures (`Follow`)
    follow: std::collections::HashMap<usize, Follow>,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// device raw pointer → its pages (None inside: tiled tier < 2 on this device)
static PAGES: Mutex<std::collections::BTreeMap<usize, Option<Pages>>> = Mutex::new(std::collections::BTreeMap::new());

impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        // tier 2: unmapped tiles read as zero, writes to them go nowhere
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            crate::xlog_info!("D3D12.X: tiled resources tier {}: GPU-owned textures in arenas", o.TiledResourcesTier.0);
            return None;
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let _ = queue.SetName(windows::core::w!("xcl vram tile-mapping queue"));
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        crate::xlog_info!("D3D12.X: textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages {
            queue,
            fence,
            value: 0,
            last_reclaim: std::time::Instant::now(),
            pools: PoolHeaps::new(),
            pool: TilePool::new(),
            pages: std::collections::HashMap::new(),
            extra_pages: std::collections::HashMap::new(),
            idle: std::collections::VecDeque::new(),
            targets: std::collections::HashMap::new(),
            waited: std::collections::HashMap::new(),
            reservable_cache: std::collections::HashMap::new(),
            scratch_heaps: Vec::new(),
            scratch_flags: Vec::new(),
            scratch_offsets: Vec::new(),
            scratch_counts: Vec::new(),
            evict_idle_until: None,
            cap: pool_cap(),
            follow: std::collections::HashMap::new(),
        })
    }

    /// Pages no resource used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        sweep_idle(&mut self.idle, &mut self.pages, &mut self.extra_pages, &mut self.pool.cooling, now);
    }

    /// Sweep expired idle pages, drain cooled tiles, and reclaim any completely empty 64 MiB
    /// pool heaps (above the warm spare count), returning their `ID3D12Heap`s to be dropped
    /// outside `PAGES.lock()`.
    fn sweep_and_reclaim(&mut self, now: std::time::Instant) -> Vec<ID3D12Heap> {
        self.sweep(now);
        self.pool.drain_cooling(now);
        let reclaimed_ids = self.pool.reclaim_empty_heaps(keep_spare_pools(), POOL_PAGES);
        let mut dropped = Vec::with_capacity(reclaimed_ids.len());
        for idx in reclaimed_ids {
            if let Some(h) = self.pools.take_heap(idx) {
                dropped.push(h);
            }
        }
        if !dropped.is_empty() {
            let mb = (dropped.len() * POOL_PAGES as usize * PAGE) >> 20;
            crate::xlog_info!(
                "D3D12.X: reclaimed {} empty 64 MiB VRAM pool heap(s) ({mb} MiB returned to GPU; {} pools active)",
                dropped.len(),
                self.pools.len()
            );
        }
        dropped
    }

    /// Page budget: room for `want` tiles of the next placement in the pool heaps the cap still
    /// allows (what is still missing then comes from new pool heaps, past the budget). Tiles
    /// not cooled yet are taken first; when there are none, one round gives back a batch of
    /// tiles (to `cooling`), which the next tiles take first; a round that found nothing to
    /// evict is not repeated for a second: each one walks every target.
    unsafe fn budget_room(&mut self, want: usize, now: std::time::Instant) {
        if self.cap == 0 {
            return;
        }
        // the pool heaps still allowed take this many; only the rest needs free tiles
        let want = want.saturating_sub(self.cap.saturating_sub(self.pools.len()) * POOL_PAGES as usize);
        if want == 0 {
            return;
        }
        self.pool.drain_cooling(now);
        while self.pool.free_total < want {
            if self.pool.cooling.is_empty() {
                if self.evict_idle_until.is_some_and(|t| now < t) {
                    return;
                }
                if self.evict_lru(EVICT_BATCH) == 0 {
                    self.evict_idle_until = Some(now + EVICT_IDLE);
                    return;
                }
            }
            // at the budget a tile is reused before it cooled: a draw still in flight may read
            // the newer texels there for a frame
            let n = (want - self.pool.free_total).min(self.pool.cooling.len());
            let tiles: Vec<Tile> = self.pool.cooling.drain(..n).map(|(t, _)| t).collect();
            self.pool.return_tiles(&tiles);
        }
    }

    /// Map every tile of `res` (`tiles[i]`: overall tile i) – one call per pool heap. LOCAL
    /// (#1514): a tile `NO_TILE` is skipped (stays unmapped).
    unsafe fn map(&mut self, res: &ID3D12Resource, tiles: &[Tile]) {
        super::real::crumb(false, || format!("vram map {:p} x{}", res.as_raw(), tiles.len())); // LOCAL
        if tiles.is_empty() {
            return;
        }
        self.scratch_heaps.clear();
        self.scratch_heaps.extend(tiles.iter().map(|t| t.0).filter(|&h| h != NO_TILE.0));
        self.scratch_heaps.sort_unstable();
        self.scratch_heaps.dedup();
        for i in 0..self.scratch_heaps.len() {
            let h = self.scratch_heaps[i];
            let ranges = heap_ranges_trimmed(h, tiles);
            if ranges.is_empty() {
                continue;
            }
            self.scratch_flags.clear();
            self.scratch_offsets.clear();
            self.scratch_counts.clear();
            let mut num_tiles = 0u32;
            for &(mapped, off, count) in &ranges {
                self.scratch_flags.push(if mapped { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP });
                self.scratch_offsets.push(off);
                self.scratch_counts.push(count);
                num_tiles += count;
            }
            let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
            let region = D3D12_TILE_REGION_SIZE { NumTiles: num_tiles, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(self.scratch_flags.as_ptr()),
                Some(self.scratch_offsets.as_ptr()),
                Some(self.scratch_counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    /// Unmap every tile of `res` (`D3D12_TILE_RANGE_FLAG_NULL`) so its unshared tiles can return
    /// to the pool even while the title keeps the COM object alive.
    unsafe fn unmap(&mut self, res: &ID3D12Resource, total: u32) {
        super::real::crumb(false, || format!("vram unmap {:p} x{total}", res.as_raw())); // LOCAL
        if total == 0 {
            return;
        }
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: total, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&total),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    fn release_tiles(&mut self, used: Vec<usize>, used_extra: Vec<(usize, usize)>, now: std::time::Instant, immediate_cool: bool) {
        for a in used {
            if let Some(p) = self.pages.get_mut(&a) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.pages.remove(&a) {
                            self.pool.push_cooling(p.tile, now);
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Own(a), now));
                    }
                }
            }
        }
        for k in used_extra {
            if let Some(p) = self.extra_pages.get_mut(&k) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.extra_pages.remove(&k) {
                            self.pool.push_cooling(p.tile, now);
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Extra(k.0, k.1), now));
                    }
                }
            }
        }
    }

    /// Page budget: unmap CPU-filled textures, least recently touched first, until `want`
    /// tiles came back (pages other mapped textures still share stay). Each goes to `EVICTED`.
    /// After a round that fell short, `take_tile` tries none for `EVICT_RETRY` (`evict_idle_until`).
    unsafe fn evict_lru(&mut self, want: usize) -> usize {
        let now = std::time::Instant::now();
        let before = self.pool.free_total + self.pool.cooling.len();
        // with the view heaps looked at, a texture whose T# is in none cannot be sampled: it
        // goes first, small or not, once `EVICT_UNSEEN_AGE` old
        let heaps = HEAPS_LOOKED.load(std::sync::atomic::Ordering::Relaxed);
        let mut order: Vec<(bool, std::time::Instant, usize)> = self
            .targets
            .iter()
            .filter_map(|(&k, t)| {
                if !t.evictable || !t.mapped {
                    return None;
                }
                let unseen = heaps && t.seen == Seen::None;
                let age = now.duration_since(t.touched);
                let ok = if unseen { age >= EVICT_UNSEEN_AGE } else { t.total >= EVICT_MIN_TILES && age >= EVICT_MIN_AGE };
                ok.then_some((!unseen, t.touched, k))
            })
            .collect();
        order.sort_unstable();
        let mut evicted = Vec::new();
        let mut shared = 0u64;
        for (_, _, key) in order {
            if self.pool.free_total + self.pool.cooling.len() - before >= want {
                break;
            }
            let Some(t) = self.targets.get(&key) else { continue };
            // a texture whose pages all are other mapped textures' too gives nothing back
            let alone = t.used.iter().any(|a| self.pages.get(a).is_some_and(|p| p.users <= 1))
                || t.used_extra.iter().any(|k| self.extra_pages.get(k).is_some_and(|p| p.users <= 1));
            if !alone {
                shared += 1;
                continue;
            }
            let Some(t) = self.targets.get_mut(&key) else { continue };
            let total = t.total;
            let used = std::mem::take(&mut t.used);
            let used_extra = std::mem::take(&mut t.used_extra);
            let follows = std::mem::take(&mut t.follows); // LOCAL (#1514)
            t.mapped = false;
            let raw = key as *mut c_void;
            if let Some(res) = ID3D12Resource::from_raw_borrowed(&raw) {
                self.unmap(res, total);
            }
            self.release_tiles(used, used_extra, now, true);
            self.unfollow(key, &follows); // LOCAL (#1514)
            evicted.push(key);
        }
        let freed = self.pool.free_total + self.pool.cooling.len() - before;
        if freed < want {
            self.evict_idle_until = Some(now + EVICT_RETRY);
        }
        if !evicted.is_empty() {
            let n = EVICTIONS.fetch_add(evicted.len() as u64, std::sync::atomic::Ordering::Relaxed) + evicted.len() as u64;
            static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let step = LOGGED.load(std::sync::atomic::Ordering::Relaxed);
            if n >= step {
                LOGGED.store(n.max(1) * 2, std::sync::atomic::Ordering::Relaxed);
                crate::xlog_info!("D3D12.X: page budget: {} textures evicted ({} MB back) this round, {n} so far; {shared} passed over, all their pages shared", evicted.len(), (freed * PAGE) >> 20);
            }
            let mut g = EVICTED.lock().unwrap();
            let s = g.get_or_insert_with(HashSet::new);
            s.extend(evicted);
            EVICTED_LEN.store(s.len(), std::sync::atomic::Ordering::Relaxed);
        }
        freed
    }

    /// Acquire/share all `total` tiles for `res` at `va` and map them on the `Pages` queue
    /// (`phys[i]`: the title's physical page at owned page i, if any – see `page_key`).
    /// The bool is true when every tile was newly taken rather than shared with an existing page.
    /// LOCAL (#1514): with `follow` (`XCL_TEXTURE_FOLLOW=1`, a sampled texture `key`), an owned
    /// page in a graphics reservation the title has no memory at yet is left unmapped (tier 2:
    /// reads zero) and mapped when the title puts memory there (`apply_backing`); the pages
    /// followed so are returned last.
    #[allow(clippy::too_many_arguments)]
    /// LOCAL (`XCL_TEXTURE_FOLLOW=3`): with `follow`, the tiles of each group in `groups` (None:
    /// one group of all tiles, `whole_group`) are left unmapped while the group is not backed.
    unsafe fn bind(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, key: usize, va: usize, console_bytes: Option<u64>, total: u32, phys: &[Option<usize>], follow: bool, groups: Option<&[FollowGroup]>) -> Option<(Vec<usize>, Vec<(usize, usize)>, bool, Vec<usize>)> {
        let (first, own) = (total > 0).then(|| owned_pages(va, console_bytes, total as usize)).flatten()?;
        let now = std::time::Instant::now();
        self.sweep(now);

        // LOCAL (#1514): the owned pages in graphics reservations are followed; those whose
        // `Follow` is not mapped are left unmapped: no video memory taken for them. A new
        // `Follow` starts as the title's memory and other textures there are now; a texture
        // placed at a page already followed takes its state, so a backing change queued for the
        // next submit reaches all of them together (Filip722's 9c553339)
        let mut follows = Vec::new();
        let mut unbacked = Vec::new();
        if follow {
            unbacked = vec![false; total as usize];
            for (i, u) in unbacked.iter_mut().enumerate().take(own) {
                let a = first + i * PAGE;
                if !crate::hle::kernelx::in_graphics_reservation(a) {
                    continue;
                }
                follows.push(a);
                // (LOCAL, mode 3: the tile keyed at `a` holds no texels of page `a`: not a sign
                // the title has memory there)
                let shared = self.pages.get(&a).is_some_and(|p| p.users > 0) && follow_mode() != 3;
                let f = self.follow.entry(a).or_insert_with(|| Follow { users: Vec::new(), mapped: shared || title_backed(a) });
                f.users.push((key, i as u32));
                *u = !f.mapped;
            }
            // LOCAL (`XCL_TEXTURE_FOLLOW=2`): the whole texture is mapped while any followed page
            // is backed, else every tile (extra ones too) is NULL
            if follow_whole() {
                if let Some(backed) = whole_backed(follows.iter().map(|a| self.follow.get(a).is_some_and(|f| f.mapped))) {
                    unbacked.fill(!backed);
                }
            } else if follow_mode() == 3 {
                // LOCAL (`XCL_TEXTURE_FOLLOW=3`): per group, not per page
                unbacked.fill(false);
                // a group's pages in graphics reservations past the owned ones (the unaligned
                // head page; pages past `total` tiles, where the console keeps mip 0) are
                // followed too: counted as backed unfollowed, they pinned their groups mapped
                if let Some(gs) = groups {
                    let mut more: Vec<usize> = gs.iter().flat_map(|g| g.pages.iter().copied()).filter(|a| follows.binary_search(a).is_err()).collect();
                    more.sort_unstable();
                    more.dedup();
                    for a in more {
                        if !crate::hle::kernelx::in_graphics_reservation(a) {
                            continue;
                        }
                        let f = self.follow.entry(a).or_insert_with(|| Follow { users: Vec::new(), mapped: title_backed(a) });
                        // (no tile of its own: mode 3 does not read it)
                        f.users.push((key, u32::MAX));
                        follows.push(a);
                    }
                    follows.sort_unstable();
                }
                let whole;
                let gs: &[FollowGroup] = match groups {
                    Some(g) => g,
                    None => {
                        whole = [whole_group(total, &follows)];
                        &whole
                    }
                };
                for g in gs {
                    if !self.group_backed_now(g, &follows) {
                        for &(s, c) in &g.tiles {
                            if let Some(u) = unbacked.get_mut(s as usize..(s as usize + c as usize)) {
                                u.fill(true);
                            }
                        }
                    }
                }
            }
        }
        let unbacked_at = |i: usize| unbacked.get(i).copied().unwrap_or(false);

        let count = |pages: &std::collections::HashMap<usize, Page>, extra_pages: &std::collections::HashMap<(usize, usize), Page>| {
            let mut missing = 0usize;
            let mut preferred: Option<u32> = None;
            for i in 0..total as usize {
                if unbacked_at(i) {
                    continue;
                }
                let existing = if i < own {
                    pages.get(&page_key(first + i * PAGE, phys.get(i).copied().flatten())).map(|p| p.tile.0)
                } else {
                    extra_pages.get(&(va, i)).map(|p| p.tile.0)
                };
                match existing {
                    Some(h) if preferred.is_none() => preferred = Some(h),
                    Some(_) => {}
                    None => missing += 1,
                }
            }
            (missing, preferred)
        };
        let (mut missing, mut preferred) = count(&self.pages, &self.extra_pages);
        if self.cap != 0 {
            // an eviction can take back pages this placement shares (an older placement at the
            // same addresses): count again until what is missing has room
            loop {
                self.budget_room(missing, now);
                let (m, p) = count(&self.pages, &self.extra_pages);
                let more = m > missing;
                (missing, preferred) = (m, p);
                if !more {
                    break;
                }
            }
        }
        let pools = &mut self.pools;
        let pool = &mut self.pool;
        let Some(fresh) = pool.take_batch(missing, preferred, now, POOL_PAGES, || {
            let heap = alloc_pool_heap(dev)?;
            Some(pools.push_heap(heap))
        }) else {
            self.unfollow(key, &follows); // LOCAL (#1514)
            return None;
        };
        let mut fresh_iter = fresh.into_iter();

        let mut tiles = Vec::with_capacity(total as usize);
        let mut used = Vec::with_capacity(own);
        let mut used_extra = Vec::new();
        let mut reused = false;
        let mut left = 0u64;
        for i in 0..total as usize {
            // LOCAL (#1514): a followed page the title has no memory at
            if unbacked_at(i) {
                tiles.push(NO_TILE);
                left += 1;
                continue;
            }
            let tile = if i < own {
                let a = page_key(first + i * PAGE, phys.get(i).copied().flatten());
                let t = match self.pages.get_mut(&a) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        reused = true;
                        p.tile
                    }
                    None => {
                        let t = fresh_iter.next().unwrap();
                        self.pages.insert(a, Page { tile: t, users: 1, idle_since: None });
                        t
                    }
                };
                used.push(a);
                t
            } else {
                let k = (va, i);
                let t = match self.extra_pages.get_mut(&k) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        reused = true;
                        p.tile
                    }
                    None => {
                        let t = fresh_iter.next().unwrap();
                        self.extra_pages.insert(k, Page { tile: t, users: 1, idle_since: None });
                        t
                    }
                };
                used_extra.push(k);
                t
            };
            tiles.push(tile);
        }
        self.map(res, &tiles);
        if left > 0 {
            FOLLOW_STATS[0].fetch_add(left, std::sync::atomic::Ordering::Relaxed);
        }
        Some((used, used_extra, !reused, follows))
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=3`): whether group `g` of a texture following the pages
    /// `follows` (ascending) is backed now: any of its pages followed and `Follow::mapped`, or
    /// not followed.
    fn group_backed_now(&self, g: &FollowGroup, follows: &[usize]) -> bool {
        group_backed(g.pages.iter().map(|a| follows.binary_search(a).is_err() || self.follow.get(a).is_some_and(|f| f.mapped)))
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=3`): the groups of mapped texture `k` whose backing differs
    /// from what it holds (a group's tiles are held all or none: its first tile tells), each with
    /// whether it is to be mapped.
    fn group_flips(&self, k: usize) -> Vec<(FollowGroup, bool)> {
        let Some(t) = self.targets.get(&k).filter(|t| t.mapped && !t.follows.is_empty()) else { return Vec::new() };
        let Some((first, own)) = owned_pages(t.va, t.console_bytes, t.total as usize) else { return Vec::new() };
        let whole;
        let gs: &[FollowGroup] = match &t.groups {
            Some(g) => g,
            None => {
                whole = [whole_group(t.total, &t.follows)];
                &whole
            }
        };
        let used: HashSet<usize> = t.used.iter().copied().collect();
        let used_extra: HashSet<(usize, usize)> = t.used_extra.iter().copied().collect();
        gs.iter()
            .filter_map(|g| {
                let j = g.tiles.first()?.0 as usize;
                let holds = if j < own { used.contains(&(first + j * PAGE)) } else { used_extra.contains(&(t.va, j)) };
                let backed = self.group_backed_now(g, &t.follows);
                (backed != holds).then(|| (g.clone(), backed))
            })
            .collect()
    }

    /// LOCAL (#1514): texture `key` no longer at its followed pages `addrs`.
    fn unfollow(&mut self, key: usize, addrs: &[usize]) {
        for a in addrs {
            if let Some(f) = self.follow.get_mut(a) {
                f.users.retain(|u| u.0 != key);
                if f.users.is_empty() {
                    self.follow.remove(a);
                }
            }
        }
    }

    /// LOCAL (#1514): the followed pages in `ranges` mapped where the title has memory now and
    /// unmapped where it has none (their video memory idle, reused if the title maps there again
    /// within `IDLE`): one UpdateTileMappings per texture and heap, then the fence signalled.
    /// The page budget makes room for the new tiles once, before any texture is changed (an
    /// eviction changes the textures and pages looked at here).
    unsafe fn apply_backing(&mut self, dev: &ID3D12Device, ranges: &[(usize, usize)]) {
        let mut addrs: Vec<usize> = Vec::new();
        for &(a, n) in ranges {
            let mut p = a & !(PAGE - 1);
            while p < a.saturating_add(n) {
                if self.follow.contains_key(&p) {
                    addrs.push(p);
                }
                p += PAGE;
            }
        }
        if addrs.is_empty() {
            return;
        }
        addrs.sort_unstable();
        addrs.dedup();
        if follow_whole() {
            self.apply_backing_whole(dev, &addrs); // LOCAL
            return;
        }
        if follow_mode() == 3 {
            self.apply_backing_groups(dev, &addrs); // LOCAL
            return;
        }
        // (page, whether the title has memory there now) where that differs from the mapping
        let flips: Vec<(usize, bool)> = addrs.into_iter().filter_map(|a| {
            let mapped = self.follow.get(&a)?.mapped;
            let backed = title_backed(a);
            (backed != mapped).then_some((a, backed))
        }).collect();
        if flips.is_empty() {
            return;
        }
        let now = std::time::Instant::now();
        let need = flips.iter().filter(|&&(a, backed)| backed && !self.pages.contains_key(&a)).count();
        if need > 0 {
            self.budget_room(need, now);
        }
        let (mut on, mut off) = (0u64, 0u64);
        let mut retry: Vec<(usize, usize)> = Vec::new();
        // texture → (tile, its new page or None: unmapped)
        let mut changes: std::collections::HashMap<usize, Vec<(u32, Option<Tile>)>> = std::collections::HashMap::new();
        for (a, backed) in flips {
            // (an eviction above may have taken some of them, or all)
            let Some(f) = self.follow.get(&a) else { continue };
            // the textures that are mapped now (an unmapped one is bound again by `bind`)
            let users: Vec<(usize, u32)> = f.users.iter().copied().filter(|u| self.targets.get(&u.0).is_some_and(|t| t.mapped)).collect();
            if backed && !users.is_empty() {
                let tile = match self.pages.get_mut(&a) {
                    Some(p) => {
                        p.idle_since = None;
                        p.tile
                    }
                    None => {
                        let pools = &mut self.pools;
                        let taken = self.pool.take_batch(1, None, now, POOL_PAGES, || {
                            let heap = alloc_pool_heap(dev)?;
                            Some(pools.push_heap(heap))
                        });
                        // (none: tried again at the next submit – LOCAL, review)
                        let Some(t) = taken.and_then(|v| v.first().copied()) else {
                            retry.push((a, PAGE));
                            continue;
                        };
                        self.pages.insert(a, Page { tile: t, users: 0, idle_since: None });
                        t
                    }
                };
                for &(k, i) in &users {
                    if let Some(t) = self.targets.get_mut(&k) {
                        t.used.push(a);
                        if let Some(p) = self.pages.get_mut(&a) {
                            p.users += 1;
                        }
                        changes.entry(k).or_default().push((i, Some(tile)));
                        REUPLOAD.lock().unwrap().push(k);
                        on += 1;
                    }
                }
            } else if !backed {
                for &(k, i) in &users {
                    if let Some(t) = self.targets.get_mut(&k) {
                        if let Some(at) = t.used.iter().position(|&u| u == a) {
                            t.used.swap_remove(at);
                            self.release_tiles(vec![a], Vec::new(), now, false);
                        }
                        changes.entry(k).or_default().push((i, None));
                        off += 1;
                    }
                }
            }
            if let Some(f) = self.follow.get_mut(&a) {
                f.mapped = backed;
            }
        }
        if !retry.is_empty() {
            BACKING.lock().unwrap().extend(retry);
        }
        FOLLOW_STATS[1].fetch_add(on, std::sync::atomic::Ordering::Relaxed);
        FOLLOW_STATS[2].fetch_add(off, std::sync::atomic::Ordering::Relaxed);
        if changes.is_empty() {
            return;
        }
        self.update_tiles(changes); // LOCAL: shared with `apply_backing_groups`
    }

    /// LOCAL (#1514): for each texture in `changes` its tiles (tile, its new page or None:
    /// unmapped): one UpdateTileMappings per texture and heap (and one for the NULL ones), then
    /// the fence signalled.
    unsafe fn update_tiles(&mut self, changes: std::collections::HashMap<usize, Vec<(u32, Option<Tile>)>>) {
        for (k, tiles) in changes {
            let Some(total) = self.targets.get(&k).filter(|t| t.mapped).map(|t| t.total) else { continue };
            let raw = k as *mut c_void;
            let Some(res) = ID3D12Resource::from_raw_borrowed(&raw) else { continue };
            let at: std::collections::HashMap<u32, Option<Tile>> = tiles.into_iter().collect();
            let mut heaps: Vec<u32> = at.values().flatten().map(|t| t.0).collect();
            heaps.sort_unstable();
            heaps.dedup();
            let mut calls: Vec<(Option<u32>, Vec<(D3D12_TILE_RANGE_FLAGS, u32, u32)>)> = heaps
                .into_iter()
                .map(|h| (Some(h), update_ranges(total, D3D12_TILE_RANGE_FLAG_NONE, |i| at.get(&i).copied().flatten().filter(|t| t.0 == h).map(|t| t.1))))
                .collect();
            if at.values().any(|t| t.is_none()) {
                calls.push((None, update_ranges(total, D3D12_TILE_RANGE_FLAG_NULL, |i| at.get(&i).filter(|t| t.is_none()).map(|_| 0))));
            }
            for (h, mut ranges) in calls {
                // (only up to the last tile set, as `heap_ranges_trimmed`)
                while ranges.last().is_some_and(|r| r.0 == D3D12_TILE_RANGE_FLAG_SKIP) {
                    ranges.pop();
                }
                let num: u32 = ranges.iter().map(|r| r.2).sum();
                if num == 0 {
                    continue;
                }
                let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                let region = D3D12_TILE_REGION_SIZE { NumTiles: num, UseBox: false.into(), ..Default::default() };
                let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|r| r.0).collect();
                let offsets: Vec<u32> = ranges.iter().map(|r| r.1).collect();
                let counts: Vec<u32> = ranges.iter().map(|r| r.2).collect();
                self.queue.UpdateTileMappings(
                    res,
                    1,
                    Some(&coord),
                    Some(&region),
                    h.map(|h| &self.pools[h as usize]),
                    ranges.len() as u32,
                    Some(flags.as_ptr()),
                    Some(offsets.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        }
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=2`): `apply_backing` for textures followed whole. Each
    /// followed page in `addrs` takes the title's memory state; each mapped texture there is
    /// then mapped whole while any of its followed pages is backed (`bind` without following,
    /// and uploaded whole: `REUPLOAD`), and NULLed whole otherwise (its pages idle, as the
    /// per-page path's). What it is now is what it holds (no `used`/`used_extra`: NULL), so
    /// every texture at a page in `addrs` is looked at, not only those at pages that flipped:
    /// one whose tiles could not be taken is tried again at the next submit (`BACKING`).
    unsafe fn apply_backing_whole(&mut self, dev: &ID3D12Device, addrs: &[usize]) {
        let mut keys: Vec<usize> = Vec::new();
        for &a in addrs {
            let backed = title_backed(a);
            if let Some(f) = self.follow.get_mut(&a) {
                f.mapped = backed;
                keys.extend(f.users.iter().map(|u| u.0));
            }
        }
        keys.sort_unstable();
        keys.dedup();
        let now = std::time::Instant::now();
        let (mut on, mut off) = (0u64, 0u64);
        let mut retry: Vec<(usize, usize)> = Vec::new();
        let mut reupload: Vec<usize> = Vec::new();
        for k in keys {
            // (looked up again for each: a `bind` below may evict textures for room)
            let Some(t) = self.targets.get(&k).filter(|t| t.mapped && !t.follows.is_empty()) else { continue };
            let holds = !t.used.is_empty() || !t.used_extra.is_empty();
            let backed = whole_backed(t.follows.iter().map(|a| self.follow.get(a).is_some_and(|f| f.mapped)));
            let Some(to) = whole_transition(backed, holds) else { continue };
            let (va, console_bytes, total) = (t.va, t.console_bytes, t.total);
            let raw = k as *mut c_void;
            let Some(res) = ID3D12Resource::from_raw_borrowed(&raw) else { continue };
            if to {
                // (takes the pages still there, makes budget room for the rest, maps every tile)
                match self.bind(dev, res, k, va, console_bytes, total, &[], false, None) {
                    Some((used, used_extra, _, _)) => {
                        if let Some(t) = self.targets.get_mut(&k) {
                            t.used = used;
                            t.used_extra = used_extra;
                        }
                        reupload.push(k);
                        on += total as u64;
                    }
                    None => {
                        if let Some(t) = self.targets.get(&k) {
                            retry.extend(t.follows.iter().map(|&a| (a, PAGE)));
                        }
                    }
                }
            } else {
                let Some(t) = self.targets.get_mut(&k) else { continue };
                let used = std::mem::take(&mut t.used);
                let used_extra = std::mem::take(&mut t.used_extra);
                self.unmap(res, total);
                self.release_tiles(used, used_extra, now, false);
                off += total as u64;
            }
        }
        if !retry.is_empty() {
            BACKING.lock().unwrap().extend(retry);
        }
        if !reupload.is_empty() {
            REUPLOAD.lock().unwrap().extend(reupload);
        }
        FOLLOW_STATS[1].fetch_add(on, std::sync::atomic::Ordering::Relaxed);
        FOLLOW_STATS[2].fetch_add(off, std::sync::atomic::Ordering::Relaxed);
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=3`): `apply_backing` per subresource group. Each followed page
    /// in `addrs` takes the title's memory state; each mapped texture there then has every group
    /// whose backing differs from what it holds (`group_flips`) changed: a group backed now takes
    /// its tiles (the pages still there under their keys, fresh tiles for the rest; the texture
    /// uploaded whole: `REUPLOAD`), one not backed has them NULLed and released (idle, as the
    /// per-page path's). A group whose tiles could not be taken is tried again at the next submit
    /// (`BACKING`). The page budget makes room once, before any texture is changed.
    unsafe fn apply_backing_groups(&mut self, dev: &ID3D12Device, addrs: &[usize]) {
        let mut keys: Vec<usize> = Vec::new();
        for &a in addrs {
            let backed = title_backed(a);
            if let Some(f) = self.follow.get_mut(&a) {
                f.mapped = backed;
                keys.extend(f.users.iter().map(|u| u.0));
            }
        }
        keys.sort_unstable();
        keys.dedup();
        let now = std::time::Instant::now();
        let tiles_of = |g: &FollowGroup| -> Vec<usize> { g.tiles.iter().flat_map(|&(s, c)| s as usize..s as usize + c as usize).collect() };
        // the tiles to take: those of groups turning on that are not still there under their keys
        // (deduplicated: aliased textures share keys) – LOCAL, review: counting every tile made
        // the page budget evict far more than needed
        let need: usize = {
            let mut want: std::collections::HashSet<Result<usize, (usize, usize)>> = std::collections::HashSet::new();
            for &k in &keys {
                let Some(t) = self.targets.get(&k) else { continue };
                let Some((first, own)) = owned_pages(t.va, t.console_bytes, t.total as usize) else { continue };
                let va = t.va;
                for (g, to) in self.group_flips(k) {
                    if !to {
                        continue;
                    }
                    for j in tiles_of(&g) {
                        let key = if j < own { Ok(first + j * PAGE) } else { Err((va, j)) };
                        let there = match key {
                            Ok(a) => self.pages.contains_key(&a),
                            Err(e) => self.extra_pages.contains_key(&e),
                        };
                        if !there {
                            want.insert(key);
                        }
                    }
                }
            }
            want.len()
        };
        if need > 0 {
            self.budget_room(need, now);
        }
        let (mut on, mut off) = (0u64, 0u64);
        let mut retry: Vec<(usize, usize)> = Vec::new();
        let mut reupload: Vec<usize> = Vec::new();
        let mut changes: std::collections::HashMap<usize, Vec<(u32, Option<Tile>)>> = std::collections::HashMap::new();
        for k in keys {
            // (looked at again: an eviction above may have unmapped it)
            let flips = self.group_flips(k);
            let Some(t) = self.targets.get(&k).filter(|_| !flips.is_empty()) else { continue };
            let (va, follows) = (t.va, t.follows.clone());
            let Some((first, own)) = owned_pages(t.va, t.console_bytes, t.total as usize) else { continue };
            // tile j's bookkeeping key (`Target::groups`)
            let key_of = |j: usize| if j < own { Ok(first + j * PAGE) } else { Err((va, j)) };
            for (g, to) in flips {
                let js = tiles_of(&g);
                if to {
                    let missing = js
                        .iter()
                        .filter(|&&j| match key_of(j) {
                            Ok(a) => !self.pages.contains_key(&a),
                            Err(e) => !self.extra_pages.contains_key(&e),
                        })
                        .count();
                    let pools = &mut self.pools;
                    let taken = self.pool.take_batch(missing, None, now, POOL_PAGES, || {
                        let heap = alloc_pool_heap(dev)?;
                        Some(pools.push_heap(heap))
                    });
                    let Some(fresh) = taken else {
                        retry.extend(g.pages.iter().filter(|a| follows.binary_search(a).is_ok()).map(|&a| (a, PAGE)));
                        continue;
                    };
                    let mut fresh = fresh.into_iter();
                    let (mut used, mut used_extra) = (Vec::new(), Vec::new());
                    let set = changes.entry(k).or_default();
                    for &j in &js {
                        let p = match key_of(j) {
                            Ok(a) => {
                                used.push(a);
                                self.pages.entry(a).or_insert_with(|| Page { tile: fresh.next().unwrap(), users: 0, idle_since: None })
                            }
                            Err(e) => {
                                used_extra.push(e);
                                self.extra_pages.entry(e).or_insert_with(|| Page { tile: fresh.next().unwrap(), users: 0, idle_since: None })
                            }
                        };
                        p.users += 1;
                        p.idle_since = None;
                        set.push((j as u32, Some(p.tile)));
                    }
                    if let Some(t) = self.targets.get_mut(&k) {
                        t.used.extend(used);
                        t.used_extra.extend(used_extra);
                    }
                    reupload.push(k);
                    on += js.len() as u64;
                } else {
                    let (mut gone, mut gone_extra) = (HashSet::new(), HashSet::new());
                    for &j in &js {
                        match key_of(j) {
                            Ok(a) => gone.insert(a),
                            Err(e) => gone_extra.insert(e),
                        };
                    }
                    let Some(t) = self.targets.get_mut(&k) else { continue };
                    let (mut rel, mut rel_extra) = (Vec::new(), Vec::new());
                    t.used.retain(|a| !gone.contains(a) || {
                        rel.push(*a);
                        false
                    });
                    t.used_extra.retain(|e| !gone_extra.contains(e) || {
                        rel_extra.push(*e);
                        false
                    });
                    self.release_tiles(rel, rel_extra, now, false);
                    changes.entry(k).or_default().extend(js.iter().map(|&j| (j as u32, None)));
                    off += js.len() as u64;
                }
            }
        }
        if !retry.is_empty() {
            BACKING.lock().unwrap().extend(retry);
        }
        reupload.dedup();
        if !reupload.is_empty() {
            REUPLOAD.lock().unwrap().extend(reupload);
        }
        FOLLOW_STATS[1].fetch_add(on, std::sync::atomic::Ordering::Relaxed);
        FOLLOW_STATS[2].fetch_add(off, std::sync::atomic::Ordering::Relaxed);
        changes.retain(|_, v| !v.is_empty());
        if !changes.is_empty() {
            self.update_tiles(changes);
        }
    }
}

/// The ranges of one UpdateTileMappings call for pool heap `heap` over a resource whose overall
/// tile i goes to `tiles[i]`: (mapped, tile offset in the heap, count) – runs of consecutive
/// tiles of the heap, and skipped runs for tiles of other heaps.
fn heap_ranges(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &(h, t) in tiles {
        let mine = h == heap;
        match out.last_mut() {
            Some(r) if mine && r.0 && r.1 + r.2 == t => r.2 += 1,
            Some(r) if !mine && !r.0 => r.2 += 1,
            _ => out.push((mine, if mine { t } else { 0 }, 1)),
        }
    }
    out
}

/// Like `heap_ranges`, with any trailing `SKIP` range trimmed so `UpdateTileMappings` only walks
/// up to the last tile belonging to `heap`.
fn heap_ranges_trimmed(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out = heap_ranges(heap, tiles);
    while out.last().is_some_and(|r| !r.0) {
        out.pop();
    }
    out
}

/// 64 KiB page boundaries `P` with `va <= P < va + bytes`, capped at `total` tiles:
/// `(first_page, count)`. Because every `P` lies strictly inside `[va, va + bytes)`, two
/// non-overlapping ranges never share a page here even when `va` is only 4 KiB aligned.
fn owned_pages(va: usize, console_bytes: Option<u64>, total: usize) -> Option<(usize, usize)> {
    match console_bytes {
        Some(b) => {
            let bytes = usize::try_from(b).ok()?;
            let first = va.checked_next_multiple_of(PAGE)?;
            let end = va.saturating_add(bytes);
            let n = if first < end { (end - first).div_ceil(PAGE) } else { 0 };
            Some((first, n.min(total)))
        }
        None => (va % PAGE == 0).then_some((va, total)),
    }
}

/// A reserved texture needs TILED in the format's support (`support2`) and, multisampled, a
/// tiled quality level at its sample count.
fn can_reserve(support2: i32, samples: u32, tiled_msaa_levels: u32) -> bool {
    support2 & D3D12_FORMAT_SUPPORT2_TILED.0 != 0 && (samples <= 1 || tiled_msaa_levels > 0)
}

/// Whether the device can make `desc` a reserved texture, asked before CreateReservedResource:
/// refused, it is an error in the debug layer (#940: R32G8X24_TYPELESS depth, 1 and 4 samples).
/// A query that fails says yes, and creation decides.
unsafe fn reservable(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    let mut f = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: desc.Format, ..Default::default() };
    if dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &mut f as *mut _ as *mut c_void, std::mem::size_of_val(&f) as u32).is_err() {
        return true;
    }
    // a format without TILED is not reservable whatever the sample count's query says
    if !can_reserve(f.Support2.0, 1, 0) {
        return false;
    }
    let samples = desc.SampleDesc.Count;
    let mut m = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: desc.Format, SampleCount: samples, Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE, NumQualityLevels: 0 };
    if samples > 1 && dev.CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &mut m as *mut _ as *mut c_void, std::mem::size_of_val(&m) as u32).is_err() {
        return true;
    }
    can_reserve(f.Support2.0, samples, m.NumQualityLevels)
}

/// Canonical physical-backing address for `va`: when `va` lies inside an unmapped GPU mirror view
/// of title CPU memory (`kernelx::mirrors::source`), map both views to the same backing address so
/// shared 64 KiB VRAM pages and arenas alias instead of duplicating video memory. If distinct
/// physical or committed pages are mapped directly at `va`, keep `va` unaliased (matching
/// `title_textures::mirror_source`).
fn canonical_va(va: usize) -> usize {
    let Some(src) = crate::hle::kernelx::mirrors::source(va) else { return va };
    if crate::hle::kernelx::physical_page_at(va) {
        return va;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let mapped = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 && m.State == MEM_COMMIT;
    if mapped { va } else { src }
}

struct DiscardQueue { queue: ID3D12CommandQueue, alloc: ID3D12CommandAllocator, list: ID3D12GraphicsCommandList, fence: ID3D12Fence, value: u64 }
unsafe impl Send for DiscardQueue {}
static DISCARD: Mutex<std::collections::BTreeMap<usize, Option<DiscardQueue>>> = Mutex::new(std::collections::BTreeMap::new());
static DISCARDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
fn discards() -> u64 { DISCARDS.load(std::sync::atomic::Ordering::Relaxed) }

impl DiscardQueue {
    unsafe fn new(dev: &ID3D12Device) -> Option<DiscardQueue> {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
        list.Close().ok()?;
        Some(DiscardQueue { queue, alloc, list, fence: dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?, value: 0 })
    }
}

/// State in which a newly backed RT or DS placement is discarded (#940: debug error 1422).
fn discard_target_state(flags: D3D12_RESOURCE_FLAGS, fresh: bool) -> Option<D3D12_RESOURCE_STATES> {
    if !fresh { None }
    else if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 { Some(D3D12_RESOURCE_STATE_DEPTH_WRITE) }
    else if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 != 0 { Some(D3D12_RESOURCE_STATE_RENDER_TARGET) }
    else { None }
}

/// Discard a newly backed RT or DS placement before any queue touches it (#940: debug error 1422),
/// leaving its subresources in `initial` and waiting for completion (`after_map`: tile mapping fence).
unsafe fn discard_fresh(dev: &ID3D12Device, res: &ID3D12Resource, initial: D3D12_RESOURCE_STATES, target: D3D12_RESOURCE_STATES, after_map: Option<(ID3D12Fence, u64)>) {
    super::title_textures::own_work(|| {
        let mut g = DISCARD.lock().unwrap();
        let Some(dq) = g.entry(dev.as_raw() as usize).or_insert_with(|| DiscardQueue::new(dev)).as_mut() else { return };
        if let Some((f, v)) = after_map { let _ = dq.queue.Wait(&f, v); }
        if dq.alloc.Reset().is_err() || dq.list.Reset(&dq.alloc, None).is_err() { return; }
        let transition = |from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
            let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(res), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
            D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
        };
        if initial != target { dq.list.ResourceBarrier(&[transition(initial, target)]); }
        dq.list.DiscardResource(res, None);
        if initial != target { dq.list.ResourceBarrier(&[transition(target, initial)]); }
        let (Ok(()), Ok(cmd)) = (dq.list.Close(), dq.list.cast()) else { return };
        dq.queue.ExecuteCommandLists(&[Some(cmd)]);
        dq.value += 1;
        if dq.queue.Signal(&dq.fence, dq.value).is_ok() {
            let _ = dq.fence.SetEventOnCompletion(dq.value, windows::Win32::Foundation::HANDLE::default());
        }
        DISCARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
}

/// Keys of `Pages::pages` for the title's physical pages (title addresses never have bit 63).
const PHYS_KEY: usize = 1 << 63;

/// The key of the page of video memory for title page address `a`: the physical page mapped
/// there (`phys`), else the address. A retail game maps one physical page under render targets
/// at two addresses (aliasing, as on the console): keyed by address, each got video memory of
/// its own (1.85 GB more in its first 40 s) and what the GPU wrote through one never showed in
/// the other.
fn page_key(a: usize, phys: Option<usize>) -> usize {
    phys.map_or(a, |p| PHYS_KEY | p)
}

/// The physical pages under a GPU-owned texture's owned pages (`require_aligned`), asked before
/// `PAGES` is taken; empty for others (keyed by address) and without the console size.
fn physical_pages_under(va: usize, console_bytes: Option<u64>, require_aligned: bool) -> Vec<Option<usize>> {
    if !require_aligned {
        return Vec::new();
    }
    match console_bytes.and_then(|b| owned_pages(va, Some(b), usize::MAX)) {
        Some((first, n)) if n > 0 => crate::hle::kernelx::physical_pages(first, n),
        _ => Vec::new(),
    }
}

/// A texture at `va` as a reserved resource on the title's shared 64 KiB pages
/// (`console_bytes`: its size on the console; tiles past the owned page boundaries are shared by
/// `(va, tile_index)` so repeated placements at the same address do not leak). None: it cannot be.
#[allow(clippy::too_many_arguments)]
unsafe fn place_on_pages(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    require_aligned: bool,
    console: Option<&[u8; 0x38]>, // LOCAL (`XCL_TEXTURE_FOLLOW=3`): the console's desc
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let va = canonical_va(va);
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D {
        return None;
    }
    if (require_aligned || console_bytes.is_none()) && va % PAGE != 0 {
        return None;
    }
    let phys = physical_pages_under(va, console_bytes, require_aligned);
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let log = |why: &str| {
        if require_aligned && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_info!("D3D12.X: GPU-owned texture at {va:#x} ({}x{}, format {}, {} samples) not reserved ({why}): in an arena", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
        }
    };
    let dev_key = dev.as_raw() as usize;
    let is_reservable = {
        let mut g = PAGES.lock().unwrap();
        let pages = g.entry(dev_key).or_insert_with(|| Pages::new(dev)).as_mut()?;
        *pages.reservable_cache.entry((tiled.Format.0, tiled.SampleDesc.Count)).or_insert_with(|| reservable(dev, &tiled))
    };
    if !is_reservable {
        log("no tiled support for its format or sample count");
        return None;
    }
    // CreateReservedResource and GetResourceTiling do not touch `PAGES`; keep them outside the
    // lock so concurrent streaming threads, `before_execute`, and `page_bytes` do not serialize.
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        log(&format!("{hr:#x}"));
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut total = 0u32;
    dev.GetResourceTiling(res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
    // LOCAL (`XCL_TEXTURE_FOLLOW=3`): the subresource groups (outside `PAGES` too); None: whole
    let groups = if follow_mode() == 3 && !require_aligned {
        console.and_then(|c| follow_groups_of(dev, res, c, va, console_bytes, total)).map(std::sync::Arc::new)
    } else {
        None
    };
    let bound = {
        let mut g = PAGES.lock().unwrap();
        if let Some(Some(pages)) = g.get_mut(&dev_key) {
            // LOCAL (#1514): only sampled textures follow the title's memory
            let follow = following() && !require_aligned && (follow_whole() || follow_mode() == 3 || follows_by_page(res)); // LOCAL: any shape whole / per group
            if let Some((used, used_extra, fresh, follows)) = pages.bind(dev, res, raw as usize, va, console_bytes, total, &phys, follow, groups.as_deref().map(|g| g.as_slice())) {
                let after_map = fresh.then(|| (pages.fence.clone(), pages.value));
                pages.targets.insert(raw as usize, Target { va, console_bytes, total, mapped: true, used, used_extra, evictable: false, touched: std::time::Instant::now(), gpu_owned: require_aligned, seen: Seen::None, follows, groups });
                Some((fresh, after_map))
            } else {
                None
            }
        } else {
            None
        }
    };
    let Some((fresh, after_map)) = bound else {
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    };
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    if let Some(target) = discard_target_state(desc.Flags, fresh) {
        discard_fresh(dev, res, D3D12_RESOURCE_STATES(state as i32), target, after_map);
    }
    Some(hr)
}

/// Unmap a placed texture superseded by a placement with a different address range, returning its
/// unshared tiles (`extra_pages` and unshared tail pages) to the cooling pool immediately.
pub(super) unsafe fn unmap_superseded(res: &ID3D12Resource) {
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.get_mut(&key) else { continue };
        if !t.mapped {
            return;
        }
        let total = t.total;
        let used = std::mem::take(&mut t.used);
        let used_extra = std::mem::take(&mut t.used_extra);
        let follows = std::mem::take(&mut t.follows); // LOCAL (#1514)
        t.mapped = false;
        pages.unmap(res, total);
        pages.release_tiles(used, used_extra, now, true);
        pages.unfollow(key, &follows); // LOCAL (#1514)
        return;
    }
}

/// Re-map a previously unmapped placed texture when the newer placement shadowing it is released.
pub(super) unsafe fn remap_if_unmapped(res: &ID3D12Resource) -> bool {
    let mut dev: Option<ID3D12Device> = None;
    if res.GetDevice(&mut dev).is_err() {
        return false;
    }
    let Some(dev) = dev else { return false };
    let key = res.as_raw() as usize;
    let target = |g: &mut std::sync::MutexGuard<'_, std::collections::BTreeMap<usize, Option<Pages>>>| {
        let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return None };
        pages.targets.get(&key).map(|t| (t.va, t.console_bytes, t.total, t.mapped, t.gpu_owned))
    };
    // (the physical pages are asked without `PAGES` held)
    let Some((va, console_bytes, _, mapped, gpu_owned)) = target(&mut PAGES.lock().unwrap()) else { return false };
    if mapped {
        return true;
    }
    let phys = physical_pages_under(va, console_bytes, gpu_owned);
    let mut g = PAGES.lock().unwrap();
    let Some((va, console_bytes, total, mapped, gpu_owned)) = target(&mut g) else { return false };
    if mapped {
        return true;
    }
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return false };
    let follow = following() && !gpu_owned && (follow_whole() || follow_mode() == 3 || follows_by_page(res)); // LOCAL (#1514; any shape whole / per group)
    let groups = pages.targets.get(&key).and_then(|t| t.groups.clone()); // LOCAL (`XCL_TEXTURE_FOLLOW=3`)
    let Some((used, used_extra, _, follows)) = pages.bind(&dev, res, key, va, console_bytes, total, &phys, follow, groups.as_deref().map(|g| g.as_slice())) else {
        return false;
    };
    if let Some(t) = pages.targets.get_mut(&key) {
        t.used = used;
        t.used_extra = used_extra;
        t.follows = follows; // LOCAL (#1514)
        t.mapped = true;
        t.touched = std::time::Instant::now();
    }
    true
}

/// Page budget: `res` was evicted by the page budget.
pub(super) fn is_evicted(res: usize) -> bool {
    any_evicted() && EVICTED.lock().unwrap().as_ref().is_some_and(|s| s.contains(&res))
}

/// Page budget: map `res` again if the page budget evicted it. True when it was evicted and is
/// mapped again (the caller uploads it whole); a failed remap leaves it evicted, to be tried
/// again at its next write, transition or copy.
pub(super) unsafe fn revive(res: &ID3D12Resource) -> bool {
    let key = res.as_raw() as usize;
    if !is_evicted(key) || !remap_if_unmapped(res) {
        return false;
    }
    let mut g = EVICTED.lock().unwrap();
    let Some(s) = g.as_mut() else { return false };
    if !s.remove(&key) {
        return false;
    }
    EVICTED_LEN.store(s.len(), std::sync::atomic::Ordering::Relaxed);
    REMAPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Page budget: whether any texture is evicted (callers skip the lookup otherwise).
pub(super) fn any_evicted() -> bool {
    EVICTED_LEN.load(std::sync::atomic::Ordering::Relaxed) != 0
}

/// Whether the page budget follows the use of textures.
pub(super) fn tracking() -> bool {
    pool_cap() != 0
}

/// Page budget: the T# of the textures `visible` is in a shader-visible view heap (they may be
/// sampled: recent now), that of `cpu_only` only in CPU-only heaps. Returns the evicted ones of
/// `visible`, which the caller maps again (`revive`) and uploads before the title's next work;
/// they stay evicted until then.
pub(super) fn seen(visible: &[usize], cpu_only: &[usize]) -> Vec<usize> {
    let now = std::time::Instant::now();
    HEAPS_LOOKED.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut g = PAGES.lock().unwrap();
    for pages in g.values_mut().flatten() {
        for t in pages.targets.values_mut() {
            t.seen = Seen::None;
        }
        for r in visible {
            if let Some(t) = pages.targets.get_mut(r) {
                t.touched = now;
                t.seen = Seen::Visible;
            }
        }
        for r in cpu_only {
            if let Some(t) = pages.targets.get_mut(r) {
                t.seen = Seen::CpuOnly;
            }
        }
    }
    drop(g);
    if !any_evicted() {
        return Vec::new();
    }
    let e = EVICTED.lock().unwrap();
    let Some(e) = e.as_ref() else { return Vec::new() };
    visible.iter().copied().filter(|r| e.contains(r)).collect()
}

/// Page budget: the textures `res` were used (a transition the title submitted, a CPU write
/// uploaded): later to be evicted.
pub(super) fn touch(res: impl IntoIterator<Item = usize>) {
    if pool_cap() == 0 {
        return;
    }
    let now = std::time::Instant::now();
    let mut g = PAGES.lock().unwrap();
    for r in res {
        for pages in g.values_mut().flatten() {
            if let Some(t) = pages.targets.get_mut(&r) {
                t.touched = now;
                break;
            }
        }
    }
}

/// Page budget: `res` is a texture `title_textures` keeps in step with the title's memory, so
/// its texels can be uploaded again after an eviction.
pub(super) fn set_evictable(res: usize) {
    set_evictable_to(res, true);
}

/// Page budget: the GPU wrote `res`; its texels are not in the title's memory, so it is
/// never evicted.
pub(super) fn pin(res: usize) {
    set_evictable_to(res, false);
}

fn set_evictable_to(res: usize, on: bool) {
    if pool_cap() == 0 {
        return;
    }
    let mut g = PAGES.lock().unwrap();
    for pages in g.values_mut().flatten() {
        if let Some(t) = pages.targets.get_mut(&res) {
            t.evictable = on;
            return;
        }
    }
}

/// Page budget: (textures evicted, textures mapped again) so far.
pub fn eviction_counts() -> (u64, u64) {
    (EVICTIONS.load(std::sync::atomic::Ordering::Relaxed), REMAPS.load(std::sync::atomic::Ordering::Relaxed))
}

/// A resource released: its pages are no longer its.
fn forget_pages(resource: usize) {
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.remove(&resource) else { continue };
        if t.mapped {
            pages.release_tiles(t.used, t.used_extra, now, false);
        }
        pages.unfollow(resource, &t.follows); // LOCAL (#1514)
        drop(g);
        if let Some(s) = EVICTED.lock().unwrap().as_mut() {
            s.remove(&resource);
            EVICTED_LEN.store(s.len(), std::sync::atomic::Ordering::Relaxed);
        }
        return;
    }
}

/// Reclaimed heaps waiting for `unmap_mark` (if any) and every live queue's fence mark on one device.
struct RetiringBatch<H, F> { heaps: Vec<H>, unmap_mark: Option<(F, u64)>, marks: Vec<(F, u64)> }

/// Per-device multi-queue retirement state (`H` = heap, `F` = fence; pure-CPU types in tests).
struct RetireQueue<H, F> { fences: std::collections::HashMap<usize, (F, u64)>, batches: Vec<RetiringBatch<H, F>> }

impl<H, F: Clone> RetireQueue<H, F> {
    fn new() -> Self { Self { fences: std::collections::HashMap::new(), batches: Vec::new() } }

    /// Seal `heaps` behind `unmap_mark` and one fence signal per queue (signaled BEFORE queueing).
    /// Returns `heaps` immediately when zero marks were signaled and `unmap_mark` is `None`.
    fn retire_with<Q>(&mut self, heaps: Vec<H>, unmap_mark: Option<(F, u64)>, queues: &[(usize, Q)], mut make_fence: impl FnMut() -> Option<F>, mut signal: impl FnMut(&Q, &F, u64) -> bool) -> Vec<H> {
        if heaps.is_empty() { return Vec::new(); }
        let mut marks = Vec::with_capacity(queues.len());
        for (k, q) in queues {
            let (f, v) = match self.fences.entry(*k) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => { let Some(f) = make_fence() else { continue }; e.insert((f, 0)) }
            };
            if signal(q, f, *v + 1) { *v += 1; marks.push((f.clone(), *v)); }
        }
        if marks.is_empty() && unmap_mark.is_none() { return heaps; }
        self.batches.push(RetiringBatch { heaps, unmap_mark, marks });
        Vec::new()
    }

    /// Reap completed batches and prune dead-queue fences once completed; caller drops outside locks.
    fn sweep_with(&mut self, live_queues: &HashSet<usize>, mut completed: impl FnMut(&F) -> u64) -> (Vec<RetiringBatch<H, F>>, Vec<F>) {
        let (mut done, mut kept) = (Vec::new(), Vec::new());
        for b in std::mem::take(&mut self.batches) {
            if b.unmap_mark.as_ref().is_none_or(|(f, v)| completed(f) >= *v) && b.marks.iter().all(|(f, v)| completed(f) >= *v) { done.push(b); } else { kept.push(b); }
        }
        self.batches = kept;
        let stale: Vec<usize> = self.fences.iter().filter(|(k, (f, v))| !live_queues.contains(k) && completed(f) >= *v).map(|(&k, _)| k).collect();
        let pruned = stale.into_iter().filter_map(|k| self.fences.remove(&k).map(|(f, _)| f)).collect();
        (done, pruned)
    }
}

unsafe impl Send for RetireQueue<ID3D12Heap, ID3D12Fence> {}

static RETIRE: Mutex<Option<std::collections::HashMap<usize, RetireQueue<ID3D12Heap, ID3D12Fence>>>> = Mutex::new(None);
static RETIRING_BATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Collect all known live command queues belonging to `dev_key` (deduplicated by raw ptr).
unsafe fn collect_device_queues(dev_key: usize, extra: Option<&ID3D12CommandQueue>) -> Vec<(usize, ID3D12CommandQueue)> {
    let (mut seen, mut out) = (HashSet::new(), Vec::new());
    let mut add = |q: ID3D12CommandQueue| {
        let key = q.as_raw() as usize;
        let mut q_dev: Option<ID3D12Device> = None;
        if seen.insert(key) && q.GetDevice(&mut q_dev).is_ok() && q_dev.is_some_and(|d| d.as_raw() as usize == dev_key) {
            out.push((key, q));
        }
    };
    for raw in super::present::known_queues() {
        let raw = raw as *mut c_void;
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) { add(q.clone()); }
    }
    for q in extra.into_iter().cloned().chain(super::title_textures::upload_queues()).chain(super::sparse::queue()) {
        add(q);
    }
    out
}

/// Retire `newly_reclaimed` across all live queues on `dev_key`, sweep completed batches, and prune
/// dead-queue fences; drops every finished `ID3D12Heap` and `ID3D12Fence` outside all locks.
unsafe fn retire_heaps_and_sweep(dev: &ID3D12Device, dev_key: usize, newly_reclaimed: Vec<ID3D12Heap>, unmap_mark: Option<(ID3D12Fence, u64)>, extra_queue: Option<&ID3D12CommandQueue>) {
    let queues = collect_device_queues(dev_key, extra_queue);
    let live_keys: HashSet<usize> = queues.iter().map(|&(k, _)| k).collect();
    let _drop_outside_locks = {
        let mut guard = RETIRE.lock().unwrap();
        let map = guard.get_or_insert_with(std::collections::HashMap::new);
        if newly_reclaimed.is_empty() && !map.contains_key(&dev_key) { return; }
        let rq = map.entry(dev_key).or_insert_with(RetireQueue::new);
        let before = rq.batches.len();
        let immediate = rq.retire_with(newly_reclaimed, unmap_mark, &queues, || dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok(), |q, f, v| q.Signal(f, v).is_ok());
        let (done, pruned) = rq.sweep_with(&live_keys, |f| f.GetCompletedValue());
        RETIRING_BATCHES.fetch_add(rq.batches.len().wrapping_sub(before), std::sync::atomic::Ordering::AcqRel);
        (immediate, done, pruned)
    };
}

/// Before GPU work on `queue`: have it wait for every tile mapping made so far on its device,
/// release any previously retired heaps whose GPU fences have completed, and periodically (once
/// per second per device) retire completely empty 64 MiB VRAM pool heaps and idle arenas behind
/// per-queue GPU fence signals across all live queues on the device.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut dev: Option<ID3D12Device> = None;
    if queue.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let dev_key = dev.as_raw() as usize;
    let now = std::time::Instant::now();
    let mut newly_reclaimed = Vec::new();
    let mut do_reclaim = false;
    let mut unmap_mark: Option<(ID3D12Fence, u64)> = None;
    // LOCAL (#1514 port review): taken even when this device has no pages yet, so it never grows
    // (only before the title's own submit, `before_title_execute`: that path uploads the textures
    // whose tiles turn on in the same submit – LOCAL, review)
    let mut changed_all = if following() && TITLE_SUBMIT.with(|c| c.get()) { std::mem::take(&mut *BACKING.lock().unwrap()) } else { Vec::new() };
    {
        let mut g = PAGES.lock().unwrap();
        if let Some(Some(pages)) = g.get_mut(&dev_key) {
            // LOCAL (#1514, XCL_TEXTURE_FOLLOW): the followed texture pages follow what the
            // title changed since the last submit (before this queue waits for the mappings)
            if following() {
                let changed = std::mem::take(&mut changed_all);
                if !changed.is_empty() && !pages.follow.is_empty() {
                    pages.apply_backing(&dev, &changed);
                }
                log_follow(pages, now);
            }
            let key = queue.as_raw() as usize;
            if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
                let _ = queue.Wait(&pages.fence, pages.value);
                super::list_x::note_internal_wait(key, "texture page mappings", &pages.fence, pages.value);
                pages.waited.insert(key, pages.value);
            }
            if now.duration_since(pages.last_reclaim) >= std::time::Duration::from_secs(1) {
                pages.last_reclaim = now;
                do_reclaim = true;
                newly_reclaimed = pages.sweep_and_reclaim(now);
                if !newly_reclaimed.is_empty() {
                    unmap_mark = Some((pages.fence.clone(), pages.value));
                }
            }
        }
    }
    if do_reclaim {
        newly_reclaimed.extend(ARENAS.lock().unwrap().sweep_dropped_for(dev_key, now));
    }
    if do_reclaim || !newly_reclaimed.is_empty() || RETIRING_BATCHES.load(std::sync::atomic::Ordering::Acquire) > 0 {
        retire_heaps_and_sweep(&dev, dev_key, newly_reclaimed, unmap_mark, Some(queue));
    }
}

/// LOCAL (#1514): every 5 s while `XCL_TEXTURE_FOLLOW=1`: textures following the title's
/// memory, their pages left without video memory now, and the tiles mapped and unmapped so far.
fn log_follow(pages: &Pages, now: std::time::Instant) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    {
        let mut last = LAST.lock().unwrap();
        if last.is_some_and(|t| now.duration_since(t) < std::time::Duration::from_secs(5)) {
            return;
        }
        *last = Some(now);
    }
    let textures = pages.targets.values().filter(|t| t.mapped && !t.follows.is_empty()).count();
    let unbacked = pages.follow.values().filter(|f| !f.mapped).count();
    // LOCAL: per texture: those holding no tile at all, and the tiles followed textures hold NULL
    let (mut off, mut null_tiles) = (0usize, 0usize);
    for t in pages.targets.values().filter(|t| t.mapped && !t.follows.is_empty()) {
        let held = t.used.len() + t.used_extra.len();
        off += usize::from(held == 0);
        null_tiles += (t.total as usize).saturating_sub(held);
    }
    let s = |i: usize| FOLLOW_STATS[i].load(std::sync::atomic::Ordering::Relaxed);
    crate::xlog_info!(
        "D3D12.X: texture follow (mode {}): {textures} textures on {} followed pages, {unbacked} pages ({} MB) without video memory now; {off} of the textures wholly unbacked, {} MB of the textures' tiles NULL; so far {} tiles left unmapped at placement, {} mapped and {} unmapped as the title's memory changed",
        follow_mode(),
        pages.follow.len(),
        (unbacked * PAGE) >> 20,
        (null_tiles * PAGE) >> 20,
        s(0),
        s(1),
        s(2)
    );
    // LOCAL (`XCL_TEXTURE_FOLLOW=3`): the textures followed per subresource group, and those
    // followed whole for want of groups
    if follow_mode() == 3 {
        let (mut by_group, mut whole, mut groups, mut groups_off) = (0usize, 0usize, 0usize, 0usize);
        for t in pages.targets.values().filter(|t| t.mapped && !t.follows.is_empty()) {
            match &t.groups {
                Some(gs) => {
                    by_group += 1;
                    groups += gs.len();
                    groups_off += gs.iter().filter(|g| !pages.group_backed_now(g, &t.follows)).count();
                }
                None => whole += 1,
            }
        }
        crate::xlog_info!(
            "D3D12.X: texture follow (mode 3): {by_group} textures followed per subresource, {groups} groups, {groups_off} of them without the title's memory now; {} MB of followed textures' tiles NULL; {whole} textures followed whole (no subresource groups)",
            (null_tiles * PAGE) >> 20
        );
    }
}

#[cfg(test)]
pub(super) static VRAM_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Page budget, for tests: evict `res` now (as the least recently touched texture the CPU
/// fills); whether it was.
#[cfg(test)]
pub(super) unsafe fn evict_for_test(res: usize) -> bool {
    let mut g = PAGES.lock().unwrap();
    for pages in g.values_mut().flatten() {
        if let Some(t) = pages.targets.get_mut(&res) {
            t.evictable = true;
            t.touched = std::time::Instant::now().checked_sub(std::time::Duration::from_secs(600)).unwrap();
            pages.evict_lru(1);
            return pages.targets.get(&res).is_some_and(|t| !t.mapped);
        }
    }
    false
}

#[cfg(test)]
pub(super) fn page_bytes_for(device: usize) -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get(&device) else { return (0, 0) };
    ((p.pages.len() + p.extra_pages.len()) * PAGE, p.pools.len() * POOL_PAGES as usize * PAGE)
}

/// Pool tiles of `device` that are neither free, cooling, nor any page's (taken and lost).
#[cfg(test)]
pub(super) fn tiles_unaccounted_for(device: usize) -> usize {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get(&device) else { return 0 };
    (p.pools.len() * POOL_PAGES as usize).saturating_sub(p.pool.free_total + p.pool.cooling.len() + p.pages.len() + p.extra_pages.len())
}

/// (bytes of pages in use by tiled textures and of small-texture slots, bytes of their heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let (small, small_heaps) = small_bytes();
    let g = PAGES.lock().unwrap();
    g.values().flatten().fold((small, small_heaps), |(used, pools), p| {
        (used + (p.pages.len() + p.extra_pages.len()) * PAGE, pools + p.pools.len() * POOL_PAGES as usize * PAGE)
    })
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A sampled texture of the title at `va` (`desc`: the PC description, `console_bytes` its size on
/// the console) backed by shared 64 KiB video-memory pages; None when reserved placement is not
/// available (the caller falls back to a committed resource).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place_pages(
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    place_pages_console(this, va, desc, state, clear, console_bytes, None, riid, ppv) // LOCAL
}

/// LOCAL: `place_pages` with the console's D3D12_RESOURCE_DESC of the texture (`console`), for
/// its subresource groups (`XCL_TEXTURE_FOLLOW=3`; without it the texture is followed whole).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place_pages_console(
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    console: Option<&[u8; 0x38]>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) || std::env::var_os("XCL_VRAM_ARENAS").is_some() {
        return None;
    }
    if console_has_packed_3d_mips(desc) {
        return None;
    }
    if let Some(hr) = place_small(dev, this, va, desc, state, clear, console_bytes, riid, ppv) {
        return Some(hr);
    }
    place_on_pages(dev, this, va, desc, state, clear, console_bytes, false, console, riid, ppv)
}

/// XG and the PC driver pack volume mip tails differently. Keep these textures on the
/// detile/upload path rather than mapping title pages directly onto the driver's packed tail.
fn console_has_packed_3d_mips(desc: &D3D12_RESOURCE_DESC) -> bool {
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || desc.MipLevels <= 1 {
        return false;
    }
    let mut console_desc = [0u8; 0x38];
    unsafe {
        std::ptr::copy_nonoverlapping(
            desc as *const _ as *const u8,
            console_desc.as_mut_ptr(),
            console_desc.len(),
        )
    };
    crate::hle::xg::Texture::new(&console_desc)
        .ok()
        .and_then(|t| t.texels().map(|s| s.depth > 0 && s.first_tail < s.chain.levels.len()))
        .unwrap_or(false)
}

/// A slot of the small-texture heaps (4 KiB).
const SLOT: usize = 0x1000;
/// slots of one small-texture heap (4 MiB)
const SLOTS_PER_HEAP: u32 = 1024;

struct SmallSlot {
    slot: u32,
    /// textures at its address
    users: u32,
}

/// The 4 KiB slots of the small-texture heaps: one per title address (textures at one address
/// share it, as they share a page's tile), given back slots reusable after `COOL`.
#[derive(Default)]
struct Slots {
    /// slots handed out so far (the next new one; a new heap at each `SLOTS_PER_HEAP`)
    next: u32,
    free: Vec<u32>,
    cooling: std::collections::VecDeque<(u32, std::time::Instant)>,
    by_va: std::collections::HashMap<usize, SmallSlot>,
}

impl Slots {
    /// The slot of `va` for one more texture there: (slot, whether its heap must be made first).
    fn take(&mut self, va: usize, now: std::time::Instant) -> (u32, bool) {
        if let Some(s) = self.by_va.get_mut(&va) {
            s.users += 1;
            return (s.slot, false);
        }
        while self.cooling.front().is_some_and(|(_, at)| now.duration_since(*at) >= COOL) {
            self.free.extend(self.cooling.pop_front().map(|(s, _)| s));
        }
        let (slot, new_heap) = match self.free.pop() {
            Some(s) => (s, false),
            None => {
                self.next += 1;
                (self.next - 1, (self.next - 1) % SLOTS_PER_HEAP == 0)
            }
        };
        self.by_va.insert(va, SmallSlot { slot, users: 1 });
        (slot, new_heap)
    }

    /// One texture at `va` fewer: its slot cools once no texture uses it.
    fn give_back(&mut self, va: usize, now: std::time::Instant) {
        if let Some(s) = self.by_va.get_mut(&va) {
            s.users -= 1;
            if s.users == 0 {
                let slot = s.slot;
                self.by_va.remove(&va);
                self.cooling.push_back((slot, now));
            }
        }
    }
}

/// The small-texture heaps of a device.
struct SmallHeaps {
    heaps: Vec<ID3D12Heap>,
    slots: Slots,
    /// resource → its address
    of: std::collections::HashMap<usize, usize>,
}

// D3D12 heaps are free-threaded
unsafe impl Send for SmallHeaps {}

static SMALL: Mutex<std::collections::BTreeMap<usize, SmallHeaps>> = Mutex::new(std::collections::BTreeMap::new());

/// A sampled texture of at most 4 KiB inside one 64 KiB page of the title's (no page boundary
/// within it, so on the pages it is one extra tile keyed by its address) as a 4 KiB-aligned
/// placement in a shared heap instead of that 64 KiB tile: a retail game makes ~12 000 views of
/// 16x4 textures of 1–2.5 KB while it loads its world – 768 MiB of tiles for 20 MiB. Textures
/// at one address share the slot as they share the tile. None: not such a texture, or the PC
/// wants more than 4 KiB for it.
#[allow(clippy::too_many_arguments)]
unsafe fn place_small(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let bytes = console_bytes?;
    if bytes > SLOT as u64 || desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.SampleDesc.Count > 1 || owned_pages(va, Some(bytes), usize::MAX)?.1 != 0 {
        return None;
    }
    let small = D3D12_RESOURCE_DESC { Alignment: SLOT as u64, ..*desc };
    let info = dev.GetResourceAllocationInfo(0, &[small]);
    if info.Alignment != SLOT as u64 || info.SizeInBytes == 0 || info.SizeInBytes > SLOT as u64 {
        return None;
    }
    let mut g = SMALL.lock().unwrap();
    let h = g.entry(dev.as_raw() as usize).or_insert_with(|| SmallHeaps { heaps: Vec::new(), slots: Slots::default(), of: std::collections::HashMap::new() });
    let now = std::time::Instant::now();
    let (slot, new_heap) = h.slots.take(va, now);
    if new_heap {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: SLOTS_PER_HEAP as u64 * SLOT as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
        };
        let mut made: Option<ID3D12Heap> = None;
        if super::title_textures::own_work(|| dev.CreateHeap(&desc, &mut made)).is_err() || made.is_none() {
            // the slot's heap was not made: hand that slot (and so that heap) out again next time
            h.slots.by_va.remove(&va);
            h.slots.next -= 1;
            return None;
        }
        h.heaps.push(made.unwrap());
    }
    let heap = &h.heaps[(slot / SLOTS_PER_HEAP) as usize];
    let offset = (slot % SLOTS_PER_HEAP) as usize * SLOT;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset, &small as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        h.slots.give_back(va, now);
        return None;
    }
    h.of.insert(*ppv as usize, va);
    drop(g);
    // aliased by address like the pages' textures
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    Some(hr)
}

/// A released small texture: its slot goes back when no texture at its address is left.
fn forget_small(resource: usize) {
    let mut g = SMALL.lock().unwrap();
    for h in g.values_mut() {
        if let Some(va) = h.of.remove(&resource) {
            h.slots.give_back(va, std::time::Instant::now());
            return;
        }
    }
}

/// (bytes of small-texture slots in use, bytes of small-texture heaps)
fn small_bytes() -> (usize, usize) {
    let g = SMALL.lock().unwrap();
    g.values().fold((0, 0), |(used, heaps), h| (used + h.slots.by_va.len() * SLOT, heaps + h.heaps.len() * SLOTS_PER_HEAP as usize * SLOT))
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let va = canonical_va(va);
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let Some(hr) = place_on_pages(dev, this, va, desc, state, clear, console_bytes, true, None, riid, ppv) {
            return Some(hr);
        }
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let align = (info.Alignment as usize).max(PAGE);
    let now = std::time::Instant::now();
    let dev_key = dev.as_raw() as usize;
    let (taken, expired) = ARENAS.lock().unwrap().take_for(dev_key, va, info.SizeInBytes as usize, align, now, |size, align| arena_heap(dev, va, size, align));
    if !expired.is_empty() {
        retire_heaps_and_sweep(dev, dev_key, expired, None, None);
    }
    let (heap, offset, id, fresh) = taken?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        ARENAS.lock().unwrap().release(id, now);
        return None;
    }
    let raw = *ppv;
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    ARENA_OF.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(raw as usize, id);
    if let (Some(target), Some(res)) = (discard_target_state(desc.Flags, fresh), ID3D12Resource::from_raw_borrowed(&raw)) {
        discard_fresh(dev, res, D3D12_RESOURCE_STATES(state as i32), target, None);
    }
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const MIB: usize = 1 << 20;

    #[test]
    fn packed_console_3d_mip_tails_use_the_detile_upload_path() {
        fn desc(
            dimension: u32,
            width: u64,
            height: u32,
            depth: u16,
            mips: u16,
            format: u32,
            layout: u32,
        ) -> D3D12_RESOURCE_DESC {
            let mut bytes = [0u8; 0x38];
            bytes[0..4].copy_from_slice(&dimension.to_le_bytes());
            bytes[0x10..0x18].copy_from_slice(&width.to_le_bytes());
            bytes[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
            bytes[0x1C..0x1E].copy_from_slice(&depth.to_le_bytes());
            bytes[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
            bytes[0x20..0x24].copy_from_slice(&format.to_le_bytes());
            bytes[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            bytes[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
            unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const D3D12_RESOURCE_DESC) }
        }

        let packed_3d = desc(4, 64, 64, 64, 4, 28, 0x105);
        assert!(console_has_packed_3d_mips(&packed_3d));
        let single_mip_3d = desc(4, 64, 64, 64, 1, 28, 0x105);
        assert!(!console_has_packed_3d_mips(&single_mip_3d));
        let packed_2d = desc(3, 64, 64, 1, 7, 28, 0x105);
        assert!(!console_has_packed_3d_mips(&packed_2d));
    }

    /// Small textures take one 4 KiB slot per address (shared there, as a page's tile is), 1024
    /// to a heap; a slot no texture uses any more is reused after `COOL`.
    #[test]
    fn small_texture_slots_are_per_address_and_come_back_cooled() {
        let mut s = Slots::default();
        let t0 = Instant::now();
        assert_eq!(s.take(0x1000, t0), (0, true));
        assert_eq!(s.take(0x1000, t0), (0, false), "a second texture at an address shares its slot");
        // 12 000 addresses: 12 heaps, not 12 000 pages
        let new_heaps = (1..12_000).filter(|k| s.take(0x1000 + k * 0x400, t0).1).count();
        assert_eq!((new_heaps, s.next), (11, 12_000));
        s.give_back(0x1000, t0);
        assert!(s.by_va.contains_key(&0x1000), "still used by the other texture there");
        s.give_back(0x1000, t0);
        assert_eq!(s.take(0x9_0000_0000, t0 + Duration::from_millis(100)), (12_000, false), "reused before COOL");
        assert_eq!(s.take(0xA_0000_0000, t0 + COOL), (0, false), "reused after COOL");
    }

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        assert_eq!(heap_ranges_trimmed(1, &tiles), [(false, 0, 2), (true, 0, 2)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]), [(true, 3, 1), (true, 1, 2)]);
    }

    /// Streaming churn across 100+ 64 MiB pool heaps: cooling tiles returned in interleaved
    /// cross-heap order are regrouped per heap and sorted into contiguous ascending runs, so new
    /// textures stay inside 1 pool heap (1 UpdateTileMappings call, 1 contiguous range) instead of
    /// scattering across dozens of heaps.
    #[test]
    fn tile_pool_keeps_churned_allocations_in_single_heap_and_contiguous() {
        let t0 = Instant::now();
        let mut pool = TilePool::new();
        let heaps_made = std::cell::Cell::new(0u32);
        let mut make = || {
            let h = heaps_made.get();
            heaps_made.set(h + 1);
            Some(h)
        };

        // Fill 100 pools (64 tiles each for a fast unit test, same multi-heap topology as 1024).
        const P: u32 = 64;
        let mut live: Vec<Vec<Tile>> = Vec::new();
        for _ in 0..100 {
            live.push(pool.take_batch(P as usize, None, t0, P, &mut make).unwrap());
        }
        assert_eq!(heaps_made.get(), 100);
        assert_eq!(pool.free_total, 0);

        // Free 32 tiles from every pool in interleaved round-robin order (worst-case churn across
        // all 100 pools in `cooling`).
        for tile_idx in 0..32usize {
            for h in 0..100usize {
                pool.push_cooling(live[h][tile_idx], t0);
            }
        }
        assert_eq!(pool.cooling.len(), 3200);

        // After COOL, allocating 100 textures of 32 tiles each must:
        // - allocate ZERO new heaps,
        // - place EVERY texture into exactly 1 single pool heap (1 UpdateTileMappings call),
        // - yield 1 single contiguous tile range (0..32) per texture!
        let t1 = t0 + COOL + Duration::from_millis(1);
        for _ in 0..100 {
            let alloc = pool.take_batch(32, None, t1, P, &mut make).unwrap();
            assert_eq!(alloc.len(), 32);
            let h0 = alloc[0].0;
            assert!(alloc.iter().all(|&(h, _)| h == h0), "texture scattered across multiple heaps");
            let ranges = heap_ranges_trimmed(h0, &alloc);
            assert_eq!(ranges, [(true, 0, 32)], "expected 1 contiguous ascending range");
        }
        assert_eq!(heaps_made.get(), 100, "must reuse cooled tiles without making new heaps");
        assert_eq!(pool.free_total, 0);

        // Partial overlap preference: if heap 7 has free tiles and preferred=Some(7), use heap 7.
        for t in 10..18u32 {
            pool.push_cooling((7, t), t1);
            pool.push_cooling((3, t), t1);
        }
        let t2 = t1 + COOL + Duration::from_millis(1);
        let pref = pool.take_batch(8, Some(7), t2, P, &mut make).unwrap();
        assert!(pref.iter().all(|&(h, _)| h == 7), "preferred heap should be chosen");
        assert_eq!(heap_ranges_trimmed(7, &pref), [(true, 10, 8)]);

        // Rollback on OOM: requesting more tiles than available when make_heap fails returns every
        // popped tile cleanly so free_total is preserved.
        assert_eq!(pool.free_total, 8);
        let fail = pool.take_batch(16, None, t2, P, || None);
        assert!(fail.is_none());
        assert_eq!(pool.free_total, 8, "OOM rollback must return all popped tiles");
    }

    /// Empty cooled pool heaps (above the warm spare count) are reclaimed cleanly back to VRAM and
    /// their slot indices can be reused without growing the heap table.
    #[test]
    fn tile_pool_reclaims_empty_heaps_keeping_warm_spare_and_reuses_slots() {
        let t0 = Instant::now();
        let mut pool = TilePool::new();
        let free_slots: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(Vec::new());
        let next_slot = std::cell::Cell::new(0u32);
        let mut alloc_slot = || {
            free_slots.borrow_mut().pop().or_else(|| {
                let h = next_slot.get();
                next_slot.set(h + 1);
                Some(h)
            })
        };

        const P: u32 = 64;
        let h0 = pool.take_batch(P as usize, None, t0, P, &mut alloc_slot).unwrap();
        let h1 = pool.take_batch(P as usize, None, t0, P, &mut alloc_slot).unwrap();
        let h2 = pool.take_batch(P as usize, None, t0, P, &mut alloc_slot).unwrap();
        let h3 = pool.take_batch(P as usize, None, t0, P, &mut alloc_slot).unwrap();
        assert_eq!(next_slot.get(), 4);
        assert_eq!(pool.free_total, 0);

        // Keep 1 tile live in heap 0; free all tiles in heaps 1, 2, 3.
        for &tile in &h0[1..] {
            pool.push_cooling(tile, t0);
        }
        for batch in [&h1, &h2, &h3] {
            for &tile in batch {
                pool.push_cooling(tile, t0);
            }
        }
        let t1 = t0 + COOL + Duration::from_millis(1);
        pool.drain_cooling(t1);
        assert_eq!(pool.free_total, 63 + 64 * 3);

        // Reclaim empty heaps keeping 1 warm spare: heaps 2 and 3 reclaim, heap 1 stays warm,
        // heap 0 still has 1 live tile so it is untouched.
        let reclaimed = pool.reclaim_empty_heaps(1, P);
        assert_eq!(reclaimed, vec![2, 3]);
        assert_eq!(pool.free_total, 63 + 64);
        free_slots.borrow_mut().extend(reclaimed);

        // First 64-tile allocation reuses warm spare heap 1 (no new heap allocated).
        let reuse_spare = pool.take_batch(P as usize, None, t1, P, &mut alloc_slot).unwrap();
        assert!(reuse_spare.iter().all(|&(h, _)| h == 1));
        assert_eq!(free_slots.borrow().len(), 2);

        // Next 64-tile allocation consumes heap 0's 63 free tiles plus 1 tile from reclaimed slot 3
        // without advancing `next_slot`.
        let reuse_slot = pool.take_batch(P as usize, None, t1, P, &mut alloc_slot).unwrap();
        assert_eq!(reuse_slot.iter().filter(|&&(h, _)| h == 0).count(), 63);
        assert_eq!(reuse_slot.iter().filter(|&&(h, _)| h == 3).count(), 1);
        assert_eq!(next_slot.get(), 4);
        assert_eq!(free_slots.borrow().len(), 1);
    }

    #[test]
    fn arenas_sweep_dropped_returns_expired_heaps() {
        let t0 = Instant::now();
        let mut arenas: Arenas<u32> = Arenas::new();
        let (_, _, id1, _) = arenas.take_for(1, 0x10_0000, 0x1_0000, PAGE, t0, |_, _| Some(101)).0.unwrap();
        let (_, _, id2, _) = arenas.take_for(1, 0x20_0000, 0x1_0000, PAGE, t0, |_, _| Some(202)).0.unwrap();
        arenas.release(id1, t0);
        arenas.release(id2, t0 + Duration::from_secs(1));
        assert!(arenas.sweep_dropped_for(1, t0 + Duration::from_millis(1900)).is_empty());
        assert_eq!(arenas.sweep_dropped_for(1, t0 + IDLE + Duration::from_millis(10)), vec![101]);
        assert_eq!(arenas.bytes(), 0x1_0000);
        assert_eq!(arenas.sweep_dropped_for(1, t0 + Duration::from_secs(1) + IDLE + Duration::from_millis(10)), vec![202]);
        assert_eq!(arenas.bytes(), 0);
    }

    #[test]
    fn arenas_sweep_dropped_preserves_newest_fitting_order() {
        let t0 = Instant::now();
        let mut arenas: Arenas<u32> = Arenas::new();
        let id1 = arenas.take_for(1, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(101)).0.unwrap().2;
        let id2 = arenas.take_for(1, 0x20_0000, PAGE, PAGE, t0, |_, _| Some(202)).0.unwrap().2;
        let id3 = arenas.take_for(1, 0x20_0000, 2 * PAGE, PAGE, t0, |_, _| Some(303)).0.unwrap().2;
        arenas.release(id1, t0);
        let t1 = t0 + IDLE + Duration::from_millis(10);
        assert_eq!(arenas.sweep_dropped_for(1, t1), vec![101]);
        let (heap, off, picked_id, _) = arenas.take_for(1, 0x20_0000, PAGE, PAGE, t1, |_, _| panic!("must reuse existing arena")).0.unwrap();
        assert_eq!((heap, off, picked_id), (303, 0, id3), "sweep must preserve insertion order so rposition selects the newest fitting arena");
        assert_ne!(picked_id, id2);
    }

    #[test]
    fn arenas_isolate_devices_and_take_for_returns_expired_heaps() {
        use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
        struct DropFlag(Arc<AtomicUsize>);
        impl Drop for DropFlag { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }
        let (t0, d1, d2) = (Instant::now(), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let mut arenas: Arenas<Arc<DropFlag>> = Arenas::new();
        let (h1, _, id1, _) = arenas.take_for(0x1000, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(Arc::new(DropFlag(d1.clone())))).0.unwrap();
        let (h2, _, id2, _) = arenas.take_for(0x2000, 0x10_0000, PAGE, PAGE, t0, |_, _| Some(Arc::new(DropFlag(d2.clone())))).0.unwrap();
        assert_ne!(id1, id2, "distinct devices at the same VA must never alias arenas");
        drop((h1, h2));
        arenas.release(id1, t0);
        arenas.release(id2, t0);
        let t1 = t0 + IDLE + Duration::from_millis(10);
        let exp1 = arenas.sweep_dropped_for(0x1000, t1);
        assert_eq!((exp1.len(), arenas.bytes(), d1.load(Ordering::SeqCst), d2.load(Ordering::SeqCst)), (1, PAGE, 0, 0));
        let (_, exp2) = arenas.take_for(0x2000, 0x20_0000, PAGE, PAGE, t1, |_, _| Some(Arc::new(DropFlag(Arc::new(AtomicUsize::new(0))))));
        assert_eq!((exp2.len(), d2.load(Ordering::SeqCst)), (1, 0), "take_for must not drop expired heaps inside ARENAS");
        drop((exp1, exp2));
        assert_eq!((d1.load(Ordering::SeqCst), d2.load(Ordering::SeqCst)), (1, 1));
    }

    #[test]
    fn retire_queue_waits_for_all_queues_and_prunes_dead_queue_fences() {
        use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
        let (mut rq, unmap_f) = (RetireQueue::<u32, Arc<AtomicU64>>::new(), Arc::new(AtomicU64::new(0)));
        let mut qf: std::collections::HashMap<usize, Arc<AtomicU64>> = std::collections::HashMap::new();
        let mk = || Some(Arc::new(AtomicU64::new(0)));
        assert!(rq.retire_with(vec![101], Some((unmap_f.clone(), 1)), &[(1, 1usize), (2, 2), (3, 3)], mk, |q, f, _| { qf.entry(*q).or_insert_with(|| f.clone()); true }).is_empty());
        assert!(rq.retire_with(vec![202], None, &[(1, 1usize), (2, 2), (4, 4)], mk, |q, f, _| { qf.entry(*q).or_insert_with(|| f.clone()); true }).is_empty());
        qf[&2].store(2, Ordering::SeqCst);
        qf[&3].store(1, Ordering::SeqCst);
        unmap_f.store(1, Ordering::SeqCst);
        let live: HashSet<usize> = [1, 2].into_iter().collect();
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert!(done.is_empty() && pruned.len() == 1 && Arc::ptr_eq(&pruned[0], &qf[&3]) && rq.fences.contains_key(&4));
        drop(pruned);
        qf[&1].store(1, Ordering::SeqCst);
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert_eq!((done.len(), done[0].heaps.as_slice(), pruned.len()), (1, &[101][..], 0));
        assert!(Arc::strong_count(&qf[&3]) > 1, "completed batch fences must drop outside lock");
        drop(done);
        qf[&1].store(2, Ordering::SeqCst);
        qf[&4].store(1, Ordering::SeqCst);
        let (done, pruned) = rq.sweep_with(&live, |f| f.load(Ordering::SeqCst));
        assert_eq!((done[0].heaps.as_slice(), pruned.len(), rq.batches.is_empty()), (&[202][..], 1, true));
        assert!(Arc::ptr_eq(&pruned[0], &qf[&4]));
        assert_eq!(rq.retire_with(vec![303], None, &[] as &[(usize, usize)], mk, |_, _, _| true), vec![303]);
    }

    #[test]
    fn gpu_retire_queue_holds_heap_until_blocked_queue_passes_mark() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
        struct GuardedHeap(#[allow(dead_code)] ID3D12Heap, Arc<AtomicBool>);
        impl Drop for GuardedHeap { fn drop(&mut self) { self.1.store(true, Ordering::SeqCst); } }
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let q1: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let q2: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let probe: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = D3D12_HEAP_DESC { SizeInBytes: PAGE as u64, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: PAGE as u64, Flags: D3D12_HEAP_FLAG_NONE };
            let mut heap: Option<ID3D12Heap> = None;
            dev.CreateHeap(&desc, &mut heap).unwrap();
            let dropped = Arc::new(AtomicBool::new(false));
            q1.Wait(&gate, 1).unwrap();
            let (k1, k2) = (q1.as_raw() as usize, q2.as_raw() as usize);
            let mut rq: RetireQueue<GuardedHeap, ID3D12Fence> = RetireQueue::new();
            assert!(rq.retire_with(vec![GuardedHeap(heap.unwrap(), dropped.clone())], None, &[(k1, q1.clone()), (k2, q2.clone())], || dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok(), |q, f, v| q.Signal(f, v).is_ok()).is_empty());
            q2.Signal(&probe, 1).unwrap();
            probe.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let (done, pruned) = rq.sweep_with(&[k1, k2].into_iter().collect(), |f| f.GetCompletedValue());
            assert!(done.is_empty() && pruned.is_empty() && !dropped.load(Ordering::SeqCst), "heap dropped while Q1 was still blocked");
            gate.Signal(1).unwrap();
            q1.Signal(&probe, 2).unwrap();
            probe.SetEventOnCompletion(2, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let (done, pruned) = rq.sweep_with(&[k1].into_iter().collect(), |f| f.GetCompletedValue());
            assert_eq!((done.len(), pruned.len(), dropped.load(Ordering::SeqCst)), (1, 1, false));
            drop((done, pruned));
            assert!(dropped.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn canonical_va_aliases_unmapped_gpu_view_but_preserves_committed_override() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const PAGES_64K: u32 = 0x2040_0000;
        const GRAPHICS_NOACCESS: u32 = 0x800;
        unsafe {
            let cpu = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            let gpu_unmapped = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(cpu != 0 && gpu_unmapped != 0);
            crate::hle::kernelx::mirrors::reserved(cpu, PAGE, PAGES_64K, 0x04);
            crate::hle::kernelx::mirrors::reserved(gpu_unmapped, PAGE, PAGES_64K, GRAPHICS_NOACCESS | 0x01);

            // Unmapped GPU view aliases to its CPU view, and reverse lookup finds the GPU view.
            assert_eq!(canonical_va(gpu_unmapped), cpu);
            assert!(crate::hle::kernelx::mirrors::gpu_views_of(cpu).contains(&gpu_unmapped));

            // Once distinct committed memory is mapped directly at the GPU view, canonical_va must
            // NOT alias it back to the CPU view.
            assert!(!VirtualAlloc(gpu_unmapped as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert_eq!(canonical_va(gpu_unmapped), gpu_unmapped);

            crate::hle::kernelx::mirrors::released(gpu_unmapped);
            crate::hle::kernelx::mirrors::released(cpu);
            VirtualFree(gpu_unmapped as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// Idle pages go back after `IDLE`, and a sweep looks at the queue of idle pages alone: one
    /// used again stays, one idle again later goes when that is `IDLE` ago, and the sweeps of
    /// 20 000 placements beside 200 000 pages in use take no time to speak of (a look at every
    /// page at each of them: 4 000 000 000 looks).
    #[test]
    fn idle_pages_are_swept_from_their_queue() {
        use std::collections::{HashMap, VecDeque};
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut idle: VecDeque<(PageKey, Instant)> = VecDeque::new();
        let mut pages: HashMap<usize, Page> = (0..200_000usize).map(|i| (i * PAGE, Page { tile: (0, i as u32), users: 1, idle_since: None })).collect();
        let mut extra: HashMap<(usize, usize), Page> = HashMap::from([((7 * PAGE, 3), Page { tile: (1, 0), users: 1, idle_since: None })]);
        let mut cooling: VecDeque<(Tile, Instant)> = VecDeque::new();
        // what `release_tiles` and `bind` do to a page
        fn release(p: &mut Page, idle: &mut VecDeque<(PageKey, Instant)>, key: PageKey, when: Instant) {
            (p.users, p.idle_since) = (0, Some(when));
            idle.push_back((key, when));
        }
        fn use_again(p: &mut Page) {
            (p.users, p.idle_since) = (1, None);
        }
        // pages 1, 2, 3 idle at 0 ms and an extra page at 100 ms; page 2 is used again at
        // 500 ms; page 3 too, and is idle again at 1500 ms
        for a in [PAGE, 2 * PAGE, 3 * PAGE] {
            release(pages.get_mut(&a).unwrap(), &mut idle, PageKey::Own(a), at(0));
        }
        release(extra.get_mut(&(7 * PAGE, 3)).unwrap(), &mut idle, PageKey::Extra(7 * PAGE, 3), at(100));
        use_again(pages.get_mut(&(2 * PAGE)).unwrap());
        use_again(pages.get_mut(&(3 * PAGE)).unwrap());
        release(pages.get_mut(&(3 * PAGE)).unwrap(), &mut idle, PageKey::Own(3 * PAGE), at(1500));

        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(1900));
        assert!(cooling.is_empty() && pages.len() == 200_000 && idle.len() == 5, "nothing is idle for IDLE yet");
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2050));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1)], "page 1 alone: 2 is in use, 3 idle since 1500 ms");
        assert!(!pages.contains_key(&PAGE) && pages.contains_key(&(2 * PAGE)) && pages.contains_key(&(3 * PAGE)) && extra.len() == 1);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2150));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0)], "the extra page, idle since 100 ms");
        assert!(extra.is_empty() && pages.len() == 199_999);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3499));
        assert!(pages.contains_key(&(3 * PAGE)));
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3500));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0), (0, 3)]);
        assert!(idle.is_empty() && pages.len() == 199_998 && pages[&(2 * PAGE)].users == 1);

        // every placement sweeps: with nothing idle that is nothing, whatever is in use
        let t = Instant::now();
        for k in 0..20_000u64 {
            sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(4000 + k));
        }
        assert!(t.elapsed() < Duration::from_secs(2), "{:?} for 20 000 sweeps", t.elapsed());
        assert_eq!(pages.len(), 199_998);
    }

    /// LOCAL (#1514): a followed page's tiles change alone: the rest of the resource is skipped,
    /// consecutive heap tiles merge into one range, unmapped ones into a NULL range.
    #[test]
    fn update_ranges_set_only_the_changed_tiles() {
        let (none, null, skip) = (D3D12_TILE_RANGE_FLAG_NONE, D3D12_TILE_RANGE_FLAG_NULL, D3D12_TILE_RANGE_FLAG_SKIP);
        // tiles 2 and 3 onto heap tiles 7 and 8, tile 5 onto 1
        let map = |i: u32| match i {
            2 => Some(7),
            3 => Some(8),
            5 => Some(1),
            _ => None,
        };
        assert_eq!(update_ranges(7, none, map), [(skip, 0, 2), (none, 7, 2), (skip, 0, 1), (none, 1, 1), (skip, 0, 1)]);
        // tiles 0, 1 and 4 unmapped
        assert_eq!(update_ranges(5, null, |i| [0, 1, 4].contains(&i).then_some(0)), [(null, 0, 2), (skip, 0, 2), (null, 0, 1)]);
    }

    /// LOCAL (#1514): a tile left out (`NO_TILE`) is skipped in every pool heap's call, so the
    /// page stays unmapped.
    #[test]
    fn no_tile_is_skipped_by_every_heap() {
        let tiles = [(0, 4), NO_TILE, (0, 5), (1, 0), NO_TILE];
        assert_eq!(heap_ranges_trimmed(0, &tiles), [(true, 4, 1), (false, 0, 1), (true, 5, 1)]);
        assert_eq!(heap_ranges_trimmed(1, &tiles), [(false, 0, 3), (true, 0, 1)]);
    }

    /// LOCAL (#1514, from 3L1A5-1's d656862a): what a page of a GPU view shows (`title_backed`):
    /// the pool page the queue's page-mapping method put there, nothing where it unmapped it (or
    /// never mapped anything), and the CPU view's address of that page (where a texture of the
    /// GPU view is placed while nothing is mapped there) follows what its GPU view shows.
    #[test]
    fn a_page_the_page_mapping_method_mapped_in_a_gpu_view_is_backed() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::hle::kernelx::mirrors;
        #[repr(C)]
        struct Batch {
            dest: u64,
            ranges_n: u32,
            _pad: u32,
            ranges: usize,
        }
        // (addresses far above anything Windows hands out, and a size no other test uses)
        let (cpu, gpu, size) = (0x7190_0000_0000usize, 0x7196_0000_0000usize, 0x129 * PAGE);
        mirrors::reserved(cpu, size, 0x2040_2000, 0x04);
        mirrors::reserved(gpu, size, 0x2040_2000, 0x801);
        let pool: Vec<u8> = vec![0; 4 * PAGE];
        let p = pool.as_ptr() as usize;
        let call = |at: usize, r: &[[u32; 3]]| unsafe {
            let b = Batch { dest: at as u64, ranges_n: r.len() as u32, _pad: 0, ranges: r.as_ptr() as usize };
            assert!(mirrors::apply_batches(1, &b as *const Batch as usize, p, 4).is_some());
        };
        let (g, c) = (gpu + 2 * PAGE, cpu + 2 * PAGE);
        assert!(!title_backed(g) && !title_backed(c), "nothing mapped in either view");
        call(g, &[[1, 1, 1]]);
        assert!(title_backed(g), "a pool page mapped by the method");
        assert!(title_backed(c), "the CPU view's page of a GPU-view page that shows memory");
        assert!(!title_backed(gpu + 3 * PAGE) && !title_backed(cpu + 3 * PAGE), "the next page is not");
        call(g, &[[2, 0, 1]]);
        assert!(!title_backed(g) && !title_backed(c), "unmapped by the method");
        // the CPU view's change and the GPU view's both reach the pages of either (other tests
        // may note their own ranges here too: only these are looked for)
        note_changed(g, PAGE);
        assert!(BACKING.lock().unwrap().contains(&(c, PAGE)), "a GPU view's change is its CPU view's");
        note_changed(c, PAGE);
        assert!(BACKING.lock().unwrap().contains(&(g, PAGE)), "a CPU view's change is its GPU view's");
        BACKING.lock().unwrap().retain(|&(a, _)| a != g && a != c);
        mirrors::released(gpu);
        mirrors::released(cpu);
    }

    /// Any 64 KiB page boundary inside `[va, va + bytes)` belongs to that range alone, so
    /// non-overlapping ranges never share an owned page even when only 4 KiB aligned.
    /// LOCAL: `XCL_TEXTURE_FOLLOW` values → mode
    #[test]
    fn follow_mode_parses_one_and_two() {
        assert_eq!(parse_follow_mode(None), 0);
        assert_eq!(parse_follow_mode(Some("0")), 0);
        assert_eq!(parse_follow_mode(Some("1")), 1);
        assert_eq!(parse_follow_mode(Some(" 2 ")), 2);
        assert_eq!(parse_follow_mode(Some("3")), 3); // LOCAL: per subresource
        assert_eq!(parse_follow_mode(Some("4")), 0);
        assert_eq!(parse_follow_mode(Some("yes")), 0);
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=3`): one group per standard subresource, all packed mips of
    /// every slice in one group with the tiles no standard subresource has
    #[test]
    fn pc_tile_groups_from_a_synthetic_tiling() {
        // 4 mips × 2 slices: mips 0–1 standard (4 + 1 tiles), mips 2–3 packed (1 tile a slice)
        let subs = [Some((0, 4)), Some((4, 1)), None, None, Some((6, 4)), Some((10, 1)), None, None];
        let g = pc_tile_groups(&subs, 4, 2, 2, 2, 1, 12).unwrap();
        assert_eq!(
            g,
            vec![
                TileGroup { subs: vec![0], tiles: vec![(0, 4)] },
                TileGroup { subs: vec![1], tiles: vec![(4, 1)] },
                TileGroup { subs: vec![4], tiles: vec![(6, 4)] },
                TileGroup { subs: vec![5], tiles: vec![(10, 1)] },
                TileGroup { subs: vec![2, 3, 6, 7], tiles: vec![(5, 1), (11, 1)] },
            ]
        );
        // one slice, no packed mips: no packed group
        let g = pc_tile_groups(&[Some((0, 2)), Some((2, 1))], 2, 1, 2, 0, 0, 3).unwrap();
        assert_eq!(g.len(), 2);
        // all packed: one group of every tile
        let g = pc_tile_groups(&[None, None, None], 3, 1, 0, 3, 2, 2).unwrap();
        assert_eq!(g, vec![TileGroup { subs: vec![0, 1, 2], tiles: vec![(0, 2)] }]);
        // refused: overlapping tiles, past the end, a tile of no one, the wrong packed count,
        // a standard mip without tiles, mip counts that do not add up
        assert_eq!(pc_tile_groups(&[Some((0, 2)), Some((1, 1))], 2, 1, 2, 0, 0, 3), None);
        assert_eq!(pc_tile_groups(&[Some((0, 2)), Some((2, 2))], 2, 1, 2, 0, 0, 3), None);
        assert_eq!(pc_tile_groups(&[Some((0, 2)), Some((2, 1))], 2, 1, 2, 0, 0, 4), None);
        assert_eq!(pc_tile_groups(&[Some((0, 2)), None], 2, 1, 1, 1, 2, 3), None);
        assert_eq!(pc_tile_groups(&[None, None], 2, 1, 1, 1, 1, 1), None);
        assert_eq!(pc_tile_groups(&[Some((0, 2)), None], 2, 1, 2, 1, 1, 3), None);
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=3`): a subresource's console range holds every element of it
    /// (`xg::Texture::element_offset`), and only the tail levels share bytes
    #[test]
    fn console_sub_ranges_hold_their_elements() {
        fn console_desc(w: u64, h: u32, array: u16, mips: u16, format: u32, layout: u32) -> [u8; 0x38] {
            let mut d = [0u8; 0x38];
            d[0..4].copy_from_slice(&3u32.to_le_bytes());
            d[0x10..0x18].copy_from_slice(&w.to_le_bytes());
            d[0x18..0x1C].copy_from_slice(&h.to_le_bytes());
            d[0x1C..0x1E].copy_from_slice(&array.to_le_bytes());
            d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
            d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
            d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            d[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
            d
        }
        // (RGBA8 and BC1 in the 64 KiB modes, an array; R8 in a 4 KiB one)
        for (w, h, array, mips, format, layout) in [(512u64, 256u32, 1u16, 10u16, 28u32, 0x119u32), (1024, 1024, 2, 11, 71, 0x11B), (256, 256, 1, 9, 61, 0x105), (300, 200, 3, 5, 28, 0x119)] {
            let tex = crate::hle::xg::Texture::new(&console_desc(w, h, array, mips, format, layout)).unwrap();
            let (s, at) = tex.plane_surface(tex.main as u32).unwrap();
            let r = console_sub_ranges(s, at);
            let n = s.chain.levels.len();
            assert_eq!(r.len(), n * array as usize);
            for slice in 0..array as usize {
                for (i, l) in s.chain.levels.iter().enumerate() {
                    let (lo, hi) = r[i + slice * n];
                    assert!(lo < hi && hi <= tex.size(), "{w}x{h} mip {i} slice {slice}: {lo:#x}..{hi:#x} of {:#x}", tex.size());
                    let e = s.bpe as u64;
                    for y in 0..l.height {
                        for x in 0..l.width {
                            let o = tex.element_offset(i, x, y, slice);
                            assert!(o >= lo && o + e <= hi, "{w}x{h} mip {i} slice {slice} ({x}, {y}): {o:#x} not in {lo:#x}..{hi:#x}");
                        }
                    }
                }
            }
            // levels outside the tail have bytes of their own
            for a in 0..r.len() {
                for b in 0..a {
                    let tail = |k: usize| k % n >= s.first_tail;
                    if !(tail(a) && tail(b) && a / n == b / n) {
                        assert!(r[a].1 <= r[b].0 || r[b].1 <= r[a].0, "{w}x{h}: subresources {a} and {b} overlap");
                    }
                }
            }
        }
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=3`): a group's pages are every 64 KiB page its subresources'
    /// console bytes overlap; it is backed when any of them is (none: backed)
    #[test]
    fn follow_groups_take_the_pages_of_their_console_bytes() {
        let va = 0x7000_0000usize;
        let tiles = vec![
            TileGroup { subs: vec![0], tiles: vec![(0, 3)] },
            TileGroup { subs: vec![1, 2], tiles: vec![(3, 1)] },
        ];
        // mip 0 over pages 1–3 (ending inside page 3), mips 1 and 2 in page 0
        let ranges = [(PAGE as u64, 3 * PAGE as u64 + 0x100), (0x8000, 0x9000), (0, 0x100)];
        let g = follow_groups(tiles, &ranges, va).unwrap();
        assert_eq!(g[0].pages, vec![va + PAGE, va + 2 * PAGE, va + 3 * PAGE]);
        assert_eq!(g[1].pages, vec![va]);
        assert_eq!(g[0].tiles, vec![(0, 3)]);
        // a subresource with no range
        assert_eq!(follow_groups(vec![TileGroup { subs: vec![5], tiles: vec![(0, 1)] }], &ranges, va), None);

        assert!(group_backed([]), "no page: backed");
        assert!(!group_backed([false, false]));
        assert!(group_backed([false, true, false]), "any page backed");
        assert!(group_backed([true]));
        assert_eq!(whole_group(5, &[va, va + PAGE]), FollowGroup { tiles: vec![(0, 5)], pages: vec![va, va + PAGE] });
        assert!(whole_group(0, &[]).tiles.is_empty());
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=2`): a texture is backed when any followed page is; with no
    /// followed page it does not follow (never NULLed for good)
    #[test]
    fn whole_texture_backed_by_any_page() {
        assert_eq!(whole_backed([]), None);
        assert_eq!(whole_backed([false]), Some(false));
        assert_eq!(whole_backed([false, false, false]), Some(false));
        assert_eq!(whole_backed([false, true, false]), Some(true));
        assert_eq!(whole_backed([true, true]), Some(true));
    }

    /// LOCAL (`XCL_TEXTURE_FOLLOW=2`): only a change of state maps or NULLs the whole texture;
    /// a texture holding no tile while backed (its tiles not taken last time) is mapped again
    #[test]
    fn whole_texture_transitions() {
        assert_eq!(whole_transition(Some(true), false), Some(true), "off → on");
        assert_eq!(whole_transition(Some(false), true), Some(false), "on → off");
        assert_eq!(whole_transition(Some(true), true), None);
        assert_eq!(whole_transition(Some(false), false), None);
        assert_eq!(whole_transition(None, true), None, "not followed");
        assert_eq!(whole_transition(None, false), None, "not followed");
    }

    #[test]
    fn owned_pages_never_alias_disjoint_ranges() {
        assert_eq!(owned_pages(0x1_0000, Some(0x1_0000), 4), Some((0x1_0000, 1)));
        assert_eq!(owned_pages(0x1_0000, Some(0x1_4000), 4), Some((0x1_0000, 2)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4000), 4), Some((0x2_0000, 0)));
        assert_eq!(owned_pages(0x1_4000, Some(0x1_0000), 4), Some((0x2_0000, 1)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4_0000), 4), Some((0x2_0000, 4)));
        assert_eq!(owned_pages(0x1_4000, None, 4), None);
        // Exhaustively check 4 KiB-aligned disjoint pairs over 256 KiB.
        const K4: usize = 0x1000;
        for a in (0..16).map(|i| 0x10_0000 + i * K4) {
            for sa in [K4, 4 * K4, 16 * K4, 20 * K4] {
                let (fa, na) = owned_pages(a, Some(sa as u64), 8).unwrap();
                let pa: Vec<usize> = (0..na).map(|i| fa + i * PAGE).collect();
                for b in (0..16).map(|i| 0x10_0000 + i * K4) {
                    for sb in [K4, 4 * K4, 16 * K4, 20 * K4] {
                        if a + sa <= b || b + sb <= a {
                            let (fb, nb) = owned_pages(b, Some(sb as u64), 8).unwrap();
                            for j in 0..nb {
                                assert!(!pa.contains(&(fb + j * PAGE)), "disjoint [{a:#x}..{:#x}) and [{b:#x}..{:#x}) shared page", a + sa, b + sb);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Reserved only with TILED support, and for MSAA a tiled sample count (#940: the driver
    /// refused R32G8X24_TYPELESS reserved, at 1 and 4 samples, and the debug layer said so).
    #[test]
    fn reserved_needs_tiled_format_and_tiled_msaa() {
        let tiled = D3D12_FORMAT_SUPPORT2_TILED.0;
        assert!(can_reserve(tiled, 1, 0));
        assert!(!can_reserve(0, 1, 0), "format without TILED");
        assert!(can_reserve(tiled, 4, 1));
        assert!(!can_reserve(tiled, 4, 0), "no tiled quality level at 4 samples");
        assert!(!can_reserve(0, 4, 1));
    }

    /// The device's answer agrees with CreateReservedResource: what `reservable` turns away the
    /// driver refuses, what it lets through the driver makes (WARP or a hardware adapter).
    #[test]
    fn reservable_agrees_with_the_driver() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: windows::Win32::Graphics::Dxgi::IDXGIFactory4 = windows::Win32::Graphics::Dxgi::CreateDXGIFactory1().expect("DXGI factory");
                let Ok(warp) = factory.EnumWarpAdapter::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>() else { return };
                let _ = D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev);
            }
            let Some(dev) = dev else { return };
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_err() || o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
                return;
            }
            for (format, samples) in [(DXGI_FORMAT_R32G8X24_TYPELESS, 1), (DXGI_FORMAT_R32G8X24_TYPELESS, 4), (DXGI_FORMAT_R32_TYPELESS, 1), (DXGI_FORMAT_R8G8B8A8_UNORM, 1)] {
                let flags = if format == DXGI_FORMAT_R8G8B8A8_UNORM { D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET } else { D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 128, Height: 128, DepthOrArraySize: 1, MipLevels: 1, Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, Flags: flags, ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                let made = dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).is_ok() && r.is_some();
                assert_eq!(reservable(&dev, &desc), made, "format {} x{samples}: reservable vs CreateReservedResource", format.0);
            }
        }
    }

    /// A placement's arena id, with heaps numbered as they are made.
    fn take(a: &mut Arenas<u32>, va: usize, bytes: usize, align: usize, now: Instant) -> u64 {
        let n = a.list.len() as u32;
        a.take_for(1, va, bytes, align, now, |_, _| Some(n)).0.unwrap().2
    }

    #[test]
    fn an_arena_is_shared_at_its_addresses_and_otherwise_its_placements_size() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let first = take(&mut a, a0, 70 * MIB, PAGE, t);
        assert_eq!(a.bytes(), 70 * MIB, "the placement's size, not 256 MiB");
        // the same address, and a smaller placement inside: the same memory
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t), first);
        assert_eq!(take(&mut a, a0 + 4 * MIB, 8 * MIB, PAGE, t), first);
        // one straddling its end gets an arena of its own size; so does an MSAA target that needs 4 MiB
        let straddle = take(&mut a, a0 + 64 * MIB, 70 * MIB, PAGE, t);
        let msaa = take(&mut a, a0 + 8 * MIB, 8 * MIB, 4 * MIB, t);
        assert!(straddle != first && msaa != first && msaa != straddle);
        assert_eq!(a.bytes(), (70 + 70 + 8) * MIB);
        // the newest arena holding an address is the one placements there share
        assert_eq!(take(&mut a, a0 + 64 * MIB, 4 * MIB, PAGE, t), straddle);
    }

    #[test]
    fn an_arena_no_resource_uses_goes_after_two_seconds() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let id = take(&mut a, a0, 70 * MIB, PAGE, t);
        a.release(id, t);
        // placed again within 2 s: the same memory (with what it holds)
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t + Duration::from_secs(1)), id);
        a.release(id, t + Duration::from_secs(1));
        take(&mut a, a0 + 512 * MIB, MIB, PAGE, t + Duration::from_secs(3) + Duration::from_millis(1));
        assert_eq!(a.bytes(), MIB, "the idle arena went");
    }

    /// The retail game's pattern (HANDOFF): targets packed back to back over 3.5 GB of its
    /// addresses and placed again and again, each round at the last round's addresses or a third
    /// of a target further on, so placements straddle earlier ones. Main's 256 MiB arenas merged
    /// straddled arenas into growing unions while the old ones lived on (15 GB in 2 s).
    #[test]
    fn streamed_targets_stay_bounded() {
        const S: usize = 64 * MIB; // a 3840x2160 D32S8, rounded
        const K: usize = 8; // targets a round
        const SPAN: usize = 3584 * MIB;
        let (t0, a0) = (Instant::now(), 0x1_0000_0000usize);
        let mut a = Arenas::new();
        let mut live: std::collections::VecDeque<Vec<u64>> = std::collections::VecDeque::new();
        let mut bytes = Vec::new();
        for r in 0..400usize {
            let now = t0 + Duration::from_millis(500 * r as u64);
            let start = ((r / 2) * S / 3) % (SPAN - K * S) & !(PAGE - 1);
            live.push_back((0..K).map(|k| take(&mut a, a0 + start + k * S, S, PAGE, now)).collect());
            // a target lives three rounds (1.5 s)
            if live.len() > 3 {
                for id in live.pop_front().unwrap() {
                    a.release(id, now);
                }
            }
            bytes.push(a.bytes());
        }
        // live: 3 rounds of targets; idle and kept (< 2 s): 4 more
        let most = *bytes.iter().max().unwrap();
        assert!(most <= 7 * K * S, "{most} bytes of arenas");
        let early = *bytes[20..120].iter().max().unwrap();
        let late = *bytes[300..].iter().max().unwrap();
        assert!(late <= early, "arenas grow over time: {early} then {late} bytes");
    }

    /// On the GPU: D32S8 targets 256 MiB apart take their own size of video memory, not an arena
    /// of 256 MiB each; depth and stencil survive a clear, CopyResource and a read of both planes;
    /// arenas no target uses go. Needs a hardware adapter (skips on WARP).
    /// Sampled textures of 4 KiB or less inside one page of the title's take 4 KiB slots of shared
    /// heaps (not a 64 KiB tile each), one per address; neighbours keep their own texels, and one
    /// over a page boundary stays on the pages.
    #[test]
    fn small_sampled_textures_take_slots_not_pages() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            // a retail game's: 16x4 R16G16B16A16_FLOAT, 1 KiB on the console, 256-byte aligned
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R16G16B16A16_FLOAT, SampleDesc: sample, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC, bytes: u64| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32, std::ptr::null(), Some(bytes), riid, &mut p);
                (hr.is_some_and(|hr| hr >= 0) && !p.is_null()).then(|| ID3D12Resource::from_raw(p))
            };
            let small = tex(16, 4);
            if dev.GetResourceAllocationInfo(0, &[D3D12_RESOURCE_DESC { Alignment: SLOT as u64, ..small }]).Alignment != SLOT as u64 {
                return crate::gpu::recomp::runner::skip_test("no 4 KiB texture placement on this adapter");
            }
            let key = dev.as_raw() as usize;
            let (pages0, _) = page_bytes_for(key);
            let used0 = SMALL.lock().unwrap().get(&key).map_or(0, |h| h.slots.by_va.len());
            let base = 0x50_0000_0000usize;
            let textures: Vec<ID3D12Resource> = (0..64).map(|k| place_at(base + k * PAGE + 0xa00, &small, 1024).expect("small texture")).collect();
            assert_eq!(page_bytes_for(key).0, pages0, "small textures took pages");
            assert_eq!(SMALL.lock().unwrap()[&key].slots.by_va.len() - used0, 64, "one slot per small texture");
            assert!(textures.iter().all(|t| is_placed_texture(t.as_raw() as usize)), "a small texture is aliased by address like the pages' ones");
            // a second texture at an address shares its slot
            let again = place_at(base + 0xa00, &small, 1024).expect("small texture again");
            assert_eq!(SMALL.lock().unwrap()[&key].slots.by_va.len() - used0, 64, "a second texture at an address took a slot");
            // over a page boundary: still the pages
            let across = place_at(base + 0x100 * PAGE + PAGE - 0x200, &small, 1024).expect("texture over a page boundary");
            assert!(is_placed_texture(across.as_raw() as usize));

            // two neighbours: each keeps what was copied into it
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
            let buffer = |t: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
                let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 2 * total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: t, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, state, None, &mut r).unwrap();
                r.unwrap()
            };
            let (upload, readback) = (buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ), buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST));
            let mut p = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut p)).unwrap();
            for k in 0..2usize {
                std::slice::from_raw_parts_mut((p as *mut u8).add(k * total as usize), total as usize).fill(0x11 * (k as u8 + 1));
            }
            upload.Unmap(0, None);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let loc = |r: &ID3D12Resource, offset: u64| D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: offset, ..fp } } };
            let sub = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            for k in 0..2 {
                list.CopyTextureRegion(&sub(&textures[k]), 0, 0, 0, &loc(&upload, k as u64 * total), None);
            }
            let to_src = |r: &ID3D12Resource| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: D3D12_RESOURCE_STATE_COPY_DEST, StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ResourceBarrier(&[to_src(&textures[0]), to_src(&textures[1])]);
            for k in 0..2 {
                list.CopyTextureRegion(&loc(&readback, k as u64 * total), 0, 0, 0, &sub(&textures[k]), None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert_eq!(fence.GetCompletedValue(), 1, "GPU work did not finish");
            readback.Map(0, None, Some(&mut p)).unwrap();
            for k in 0..2usize {
                let row = std::slice::from_raw_parts((p as *const u8).add(k * total as usize), 16 * 8);
                assert!(row.iter().all(|&b| b == 0x11 * (k as u8 + 1)), "texture {k} read back {:02x?}", &row[..16]);
            }
            readback.Unmap(0, None);

            // released: the slots go back
            for t in textures.iter().chain([&again, &across]) {
                forget(t.as_raw() as usize);
            }
            assert_eq!(SMALL.lock().unwrap()[&key].slots.by_va.len(), used0);
        }
    }

    #[test]
    fn d32s8_targets_take_their_own_size_of_video_memory() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32, &clear as *const _ as *const c_void, None, riid, &mut p);
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "D32S8 target at {va:#x}");
                ID3D12Resource::from_raw(p)
            };

            // 14 4K targets over 3.5 GB of addresses
            let big = tex(3840, 2160);
            let size = dev.GetResourceAllocationInfo(0, &[big]).SizeInBytes as usize;
            let (used0, arenas0) = (local(), arena_bytes().0);
            let targets: Vec<ID3D12Resource> = (0..14).map(|k| place_at(0x40_0000_0000 + k * 256 * MIB, &big)).collect();
            let (used, arenas) = (local().saturating_sub(used0) as usize, arena_bytes().0 - arenas0);
            println!("14 D32S8 3840x2160 targets ({size} bytes each): {arenas} bytes of arenas, {used} bytes more local video memory");
            assert!(arenas <= 14 * size.next_multiple_of(PAGE), "{arenas} bytes of arenas");
            assert!(used < 14 * 256 * MIB, "{used} bytes of video memory");
            drop(targets);

            // clear one, copy it to another, read both planes back
            let small = tex(64, 64);
            let (a, b) = (place_at(0x48_0000_0000, &small), place_at(0x48_1000_0000, &small));
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&a, Some(&view), dsv);
            let mut layouts = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 2, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 0x5A, None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(&b, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyResource(&b, &a);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (plane, layout) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *layout } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // no event: returns when the fence has the value
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p as *const u8, total as usize);
            let row = |plane: usize, y: usize| layouts[plane].Offset as usize + y * layouts[plane].Footprint.RowPitch as usize;
            for y in [0, 31, 63] {
                let depth = f32::from_le_bytes(bytes[row(0, y)..row(0, y) + 4].try_into().unwrap());
                assert_eq!((depth, bytes[row(1, y)]), (0.25, 0x5A), "row {y}: depth, stencil");
            }
            readback.Unmap(0, None);
            drop((a, b));

            // 2 s later the arenas no target uses are let go: one new target's is all that is left
            std::thread::sleep(IDLE + Duration::from_millis(100));
            let last = place_at(0x50_0000_0000, &small);
            let small_size = dev.GetResourceAllocationInfo(0, &[small]).SizeInBytes as usize;
            assert_eq!(arena_bytes().0 - arenas0, small_size.next_multiple_of(PAGE));
            drop(last);
        }
    }

    /// Reproduces the multi-pool streaming churn pattern on the active D3D12 driver (comparing
    /// scattered cross-heap `UpdateTileMappings` against single-heap contiguous
    /// `UpdateTileMappings`, both with `FLAG_NONE` and per-texture `Signal`):
    /// `cargo test -p xcl --lib -- --ignored --exact hle::d3d12x::vram::tests::churn_update_tile_mappings_timing --nocapture`
    #[test]
    #[ignore]
    fn churn_update_tile_mappings_timing() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("D3D12 device");
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            const HEAPS: usize = 64;
            const TILES_PER_TEX: usize = 32; // 2 MiB tiled texture
            const TEX_COUNT: usize = 256;
            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: 64 * PAGE as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_NONE,
            };
            let mut heaps = Vec::with_capacity(HEAPS);
            for _ in 0..HEAPS {
                let mut h: Option<ID3D12Heap> = None;
                dev.CreateHeap(&heap_desc, &mut h).unwrap();
                heaps.push(h.unwrap());
            }
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 1024,
                Height: 512,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let make_textures = || -> Vec<ID3D12Resource> {
                (0..TEX_COUNT)
                    .map(|_| {
                        let mut r: Option<ID3D12Resource> = None;
                        dev.CreateReservedResource(&tex_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                        r.unwrap()
                    })
                    .collect()
            };

            // BEFORE pattern: flat-FIFO cooling scatters each 32-tile texture across 32 heaps,
            // calling UpdateTileMappings(FLAG_NONE) per heap + Signal per texture.
            let old_res = make_textures();
            let mut fval = 0u64;
            let t0 = Instant::now();
            let mut old_calls = 0usize;
            for (k, r) in old_res.iter().enumerate() {
                let tiles: Vec<Tile> = (0..TILES_PER_TEX).map(|i| (((k + i) % HEAPS) as u32, (i % 64) as u32)).collect();
                let mut hs: Vec<u32> = tiles.iter().map(|t| t.0).collect();
                hs.sort_unstable();
                hs.dedup();
                for h in hs {
                    let ranges = heap_ranges(h, &tiles);
                    let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|x| if x.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
                    let offsets: Vec<u32> = ranges.iter().map(|x| x.1).collect();
                    let counts: Vec<u32> = ranges.iter().map(|x| x.2).collect();
                    let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                    let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
                    queue.UpdateTileMappings(r, 1, Some(&coord), Some(&region), &heaps[h as usize], ranges.len() as u32, Some(flags.as_ptr()), Some(offsets.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
                    old_calls += 1;
                }
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
            }
            let old_submit = t0.elapsed();
            fence.SetEventOnCompletion(fval, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let old_total = t0.elapsed();

            // AFTER pattern: per-heap contiguous TilePool -> 1 heap, 1 contiguous range,
            // D3D12_TILE_MAPPING_FLAG_NONE, per-texture Signal.
            let new_res = make_textures();
            let t1 = Instant::now();
            let mut new_calls = 0usize;
            for (k, r) in new_res.iter().enumerate() {
                let h = k % HEAPS;
                let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                let region = D3D12_TILE_REGION_SIZE { NumTiles: TILES_PER_TEX as u32, UseBox: false.into(), ..Default::default() };
                let flag = D3D12_TILE_RANGE_FLAG_NONE;
                let off = 0u32;
                let cnt = TILES_PER_TEX as u32;
                queue.UpdateTileMappings(r, 1, Some(&coord), Some(&region), &heaps[h], 1, Some(&flag), Some(&off), Some(&cnt), D3D12_TILE_MAPPING_FLAG_NONE);
                new_calls += 1;
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
            }
            let new_submit = t1.elapsed();
            fence.SetEventOnCompletion(fval, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let new_total = t1.elapsed();

            println!(
                "BEFORE ({TEX_COUNT} x 2MiB textures, {old_calls} UpdateTileMappings + {TEX_COUNT} Signals): submit={old_submit:?} gpu_done={old_total:?}"
            );
            println!(
                "AFTER  ({TEX_COUNT} x 2MiB textures, {new_calls} UpdateTileMappings + {TEX_COUNT} Signals): submit={new_submit:?} gpu_done={new_total:?}"
            );
            assert!(new_calls < old_calls);
        }
    }

    /// Page budget: with the view heaps looked at, a texture whose T# is in none goes first; one
    /// whose pages all are shared is passed over, and a round that fell short holds the next back.
    #[test]
    fn page_budget_takes_unseen_textures_first_and_passes_over_shared_pages() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(1 << 20), riid, &mut p);
                if !hr.is_some_and(|hr| hr >= 0) || p.is_null() {
                    return None;
                }
                Some(ID3D12Resource::from_raw(p))
            };
            // two textures at the same address share every page
            let (Some(a), Some(b), Some(unseen), Some(own)) = (place_at(0x61_0000_0000), place_at(0x61_0000_0000), place_at(0x61_1000_0000), place_at(0x61_2000_0000)) else {
                crate::gpu::recomp::runner::skip_test("no reserved textures");
                return;
            };
            let key = |r: &ID3D12Resource| r.as_raw() as usize;
            let dev_key = dev.as_raw() as usize;
            let now = Instant::now();
            HEAPS_LOOKED.store(true, std::sync::atomic::Ordering::Relaxed);
            {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                // (`set_evictable` does nothing without XCL_PAGE_BUDGET_MB)
                for (r, age, seen) in [(&a, 60, Seen::CpuOnly), (&b, 60, Seen::CpuOnly), (&unseen, 3, Seen::None), (&own, 30, Seen::CpuOnly)] {
                    let t = pages.targets.get_mut(&key(r)).unwrap();
                    t.touched = now - Duration::from_secs(age);
                    t.seen = seen;
                    t.evictable = true;
                }
                pages.evict_idle_until = None;
                pages.evict_lru(1);
                assert!(!pages.targets[&key(&unseen)].mapped, "a texture with no T# goes first, though younger");
                assert!(pages.targets[&key(&a)].mapped && pages.targets[&key(&b)].mapped && pages.targets[&key(&own)].mapped);
                pages.evict_lru(1);
                assert!(pages.targets[&key(&a)].mapped && pages.targets[&key(&b)].mapped, "shared pages give nothing back: passed over");
                assert!(!pages.targets[&key(&own)].mapped, "the next with pages of its own is evicted");
                assert!(pages.evict_idle_until.is_none(), "rounds that gave back what was wanted are not held back");
                pages.evict_lru(usize::MAX);
                assert!(pages.evict_idle_until.is_some_and(|t| t > Instant::now()), "a round that fell short holds the next back");
                assert!(pages.targets[&key(&a)].mapped && pages.targets[&key(&b)].mapped);
            }
            HEAPS_LOOKED.store(false, std::sync::atomic::Ordering::Relaxed);
            assert!(is_evicted(key(&unseen)) && is_evicted(key(&own)) && !is_evicted(key(&a)) && !is_evicted(key(&b)));
            assert!(revive(&unseen) && revive(&own), "mapped again");
            drop((a, b, unseen, own));
        }
    }

    /// Page budget: the least recently touched CPU-filled texture gives its
    /// pages back first, a GPU-written one never does, and an evicted one maps again.
    #[test]
    fn page_budget_evicts_the_least_recently_touched_cpu_filled_texture() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(1 << 20), riid, &mut p);
                if !hr.is_some_and(|hr| hr >= 0) || p.is_null() {
                    return None;
                }
                Some(ID3D12Resource::from_raw(p))
            };
            let (Some(old), Some(newer), Some(written)) = (place_at(0x60_0000_0000), place_at(0x60_1000_0000), place_at(0x60_2000_0000)) else {
                crate::gpu::recomp::runner::skip_test("no reserved textures");
                return;
            };
            let key = |r: &ID3D12Resource| r.as_raw() as usize;
            let dev_key = dev.as_raw() as usize;
            let now = Instant::now();
            {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                // (`set_evictable` / `pin` do nothing without XCL_PAGE_BUDGET_MB)
                for (r, age, evictable) in [(&old, 30, true), (&newer, 20, true), (&written, 40, false)] {
                    let t = pages.targets.get_mut(&key(r)).unwrap();
                    t.touched = now - Duration::from_secs(age);
                    t.evictable = evictable;
                }
                let used_before = pages.pages.len();
                assert!(pages.evict_lru(1) > 0, "tiles came back");
                assert!(!pages.targets[&key(&old)].mapped, "the oldest CPU-filled texture is evicted");
                assert!(pages.targets[&key(&newer)].mapped, "one is enough for one tile");
                assert!(pages.targets[&key(&written)].mapped, "a GPU-written texture is never evicted");
                assert!(pages.pages.len() < used_before, "its pages went back");
                // nothing left to evict: the round says so (take_tile then waits before the next)
                pages.targets.get_mut(&key(&newer)).unwrap().evictable = false;
                assert_eq!(pages.evict_lru(1), 0, "nothing evictable");
            }
            assert!(is_evicted(key(&old)) && !is_evicted(key(&newer)));
            assert!(revive(&old), "mapped again");
            assert!(!is_evicted(key(&old)) && !revive(&old), "once");
            assert!(PAGES.lock().unwrap().get(&dev_key).unwrap().as_ref().unwrap().targets[&key(&old)].mapped, "mapped again");
            drop((old, newer, written));
        }
    }

    /// Page budget: at the cap, one texture placed evicts one batch, and its other tiles come
    /// from that batch – not one more round per tile (#640 review: one placement evicted every
    /// eligible texture).
    #[test]
    fn page_budget_one_placement_at_the_cap_evicts_one_batch() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            // 512 x 512 RGBA8: 16 tiles each
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(1 << 20), riid, &mut p);
                if !hr.is_some_and(|hr| hr >= 0) || p.is_null() {
                    return None;
                }
                Some(ID3D12Resource::from_raw(p))
            };
            // 40 x 16 = 640 tiles: more than one batch of 512
            let old: Vec<ID3D12Resource> = (0..40).filter_map(|i| place_at(0x61_0000_0000 + i * 0x10_0000)).collect();
            if old.len() != 40 {
                crate::gpu::recomp::runner::skip_test("no reserved textures");
                return;
            }
            let key = |r: &ID3D12Resource| r.as_raw() as usize;
            let dev_key = dev.as_raw() as usize;
            let now = Instant::now();
            let (saved_free, saved_cooling) = {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                for (i, r) in old.iter().enumerate() {
                    let t = pages.targets.get_mut(&key(r)).unwrap();
                    t.touched = now - Duration::from_secs(100 - i as u64);
                    t.evictable = true;
                }
                // at the cap with no spare tile: the next tile has to come from an eviction
                pages.cap = pages.pools.len();
                pages.evict_idle_until = None;
                let free: Vec<Vec<u32>> = pages.pool.free_by_heap.iter_mut().map(std::mem::take).collect();
                pages.pool.free_total = 0;
                (free, std::mem::take(&mut pages.pool.cooling))
            };
            let new = place_at(0x61_8000_0000);
            let mapped: Vec<bool> = {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                let mapped = old.iter().map(|r| pages.targets[&key(r)].mapped).collect();
                pages.cap = 0;
                for (h, free) in saved_free.into_iter().enumerate() {
                    pages.pool.free_total += free.len();
                    pages.pool.free_by_heap[h].extend(free);
                    pages.pool.dirty_sort[h] = true;
                }
                pages.pool.cooling.extend(saved_cooling);
                mapped
            };
            assert!(new.is_some(), "placed from the evicted tiles");
            let evicted = mapped.iter().filter(|m| !**m).count();
            assert_eq!(evicted, 32, "one batch of 512 tiles: the 32 least recently touched");
            assert!(mapped[..32].iter().all(|m| !m) && mapped[32..].iter().all(|m| *m), "oldest first");
            for r in &old {
                let _ = revive(r);
            }
            drop((old, new));
        }
    }

    /// Page budget: a placement whose eviction takes back pages it shares with the evicted
    /// texture (an older placement over the same addresses) counts them again and takes them
    /// from the evicted tiles, within the cap (the port of #640 counted once and panicked).
    #[test]
    fn page_budget_eviction_of_a_shared_page_counts_it_again() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            // 512 x 512 RGBA8: 16 tiles, all on the title's pages
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(1 << 20), riid, &mut p);
                if !hr.is_some_and(|hr| hr >= 0) || p.is_null() {
                    return None;
                }
                Some(ID3D12Resource::from_raw(p))
            };
            let Some(old) = place_at(0x62_0000_0000) else {
                crate::gpu::recomp::runner::skip_test("no reserved textures");
                return;
            };
            let dev_key = dev.as_raw() as usize;
            let (saved_free, saved_cooling, cap) = {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                let t = pages.targets.get_mut(&(old.as_raw() as usize)).unwrap();
                t.touched = Instant::now() - Duration::from_secs(100);
                t.evictable = true;
                // at the cap with no spare tile
                pages.cap = pages.pools.len();
                pages.evict_idle_until = None;
                let free: Vec<Vec<u32>> = pages.pool.free_by_heap.iter_mut().map(std::mem::take).collect();
                pages.pool.free_total = 0;
                (free, std::mem::take(&mut pages.pool.cooling), pages.cap)
            };
            // its second half is on the old texture's last 8 pages, which the eviction takes back
            let new = place_at(0x62_0000_0000 + 8 * PAGE);
            let (old_mapped, pools) = {
                let mut g = PAGES.lock().unwrap_or_else(|e| e.into_inner());
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                let out = (pages.targets[&(old.as_raw() as usize)].mapped, pages.pools.len());
                pages.cap = 0;
                for (h, free) in saved_free.into_iter().enumerate() {
                    pages.pool.free_total += free.len();
                    pages.pool.free_by_heap[h].extend(free);
                    pages.pool.dirty_sort[h] = true;
                }
                pages.pool.cooling.extend(saved_cooling);
                out
            };
            assert!(new.is_some(), "placed");
            assert!(!old_mapped, "the older placement was evicted");
            assert_eq!(pools, cap, "its tiles came from the eviction, not a new pool heap");
            let _ = revive(&old);
            drop((old, new));
        }
    }

    /// Page budget: below the cap, a placement that needs more than the pool heaps the cap still
    /// allows makes only those and evicts for the rest.
    #[test]
    fn page_budget_new_pool_heaps_stop_at_the_cap() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, width: u64, height: u32| {
                let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: width, Height: height, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(1 << 20), riid, &mut p);
                if !hr.is_some_and(|hr| hr >= 0) || p.is_null() {
                    return None;
                }
                Some(ID3D12Resource::from_raw(p))
            };
            // two of 16 tiles to evict
            let (Some(a), Some(b)) = (place_at(0x63_0000_0000, 512, 512), place_at(0x63_1000_0000, 512, 512)) else {
                crate::gpu::recomp::runner::skip_test("no reserved textures");
                return;
            };
            let dev_key = dev.as_raw() as usize;
            let (saved_free, saved_cooling, cap) = {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                for r in [&a, &b] {
                    let t = pages.targets.get_mut(&(r.as_raw() as usize)).unwrap();
                    t.touched = Instant::now() - Duration::from_secs(100);
                    t.evictable = true;
                }
                // one more pool heap allowed, no spare tile
                pages.cap = pages.pools.len() + 1;
                pages.evict_idle_until = None;
                let free: Vec<Vec<u32>> = pages.pool.free_by_heap.iter_mut().map(std::mem::take).collect();
                pages.pool.free_total = 0;
                (free, std::mem::take(&mut pages.pool.cooling), pages.cap)
            };
            // 4096 x 4224 RGBA8: 32 x 33 = 1056 tiles, 32 more than one pool heap
            let big = place_at(0x63_2000_0000, 4096, 4224);
            let (mapped, pools) = {
                let mut g = PAGES.lock().unwrap_or_else(|e| e.into_inner());
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                let out = ([&a, &b].map(|r| pages.targets[&(r.as_raw() as usize)].mapped), pages.pools.len());
                pages.cap = 0;
                for (h, free) in saved_free.into_iter().enumerate() {
                    pages.pool.free_total += free.len();
                    pages.pool.free_by_heap[h].extend(free);
                    pages.pool.dirty_sort[h] = true;
                }
                pages.pool.cooling.extend(saved_cooling);
                out
            };
            assert!(big.is_some(), "placed");
            assert_eq!(pools, cap, "one new pool heap, up to the cap");
            assert_eq!(mapped, [false, false], "the other 32 tiles came from evictions");
            let _ = (revive(&a), revive(&b));
            drop((a, b, big));
        }
    }

    #[test]
    fn fresh_targets_need_discard_once_and_aliased_arenas_do_not() {
        let (t, a0, mut a) = (Instant::now(), 0x2_0000_0000, Arenas::new());
        let take_f = |a: &mut Arenas<u32>, va: usize, bytes: usize, now: Instant| {
            let n = a.list.len() as u32;
            let (_, _, id, fresh) = a.take_for(1, va, bytes, PAGE, now, |_, _| Some(n)).0.unwrap();
            (id, fresh)
        };
        let (id, f0) = take_f(&mut a, a0, 8 * MIB, t);
        assert!(f0 && take_f(&mut a, a0, 8 * MIB, t) == (id, false) && take_f(&mut a, a0 + MIB, 2 * MIB, t) == (id, false));
        for _ in 0..3 { a.release(id, t); }
        assert_eq!(take_f(&mut a, a0, 8 * MIB, t + Duration::from_secs(1)), (id, false));
        a.release(id, t + Duration::from_secs(1));
        let (id2, f1) = take_f(&mut a, a0, 8 * MIB, t + Duration::from_secs(4));
        assert!(f1 && id2 != id);
        for (flags, fresh, want) in [
            (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, true, Some(D3D12_RESOURCE_STATE_RENDER_TARGET)),
            (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, true, Some(D3D12_RESOURCE_STATE_DEPTH_WRITE)),
            (D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, true, None),
            (D3D12_RESOURCE_FLAG_NONE, true, None),
            (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, false, None),
            (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, false, None),
        ] {
            assert_eq!(discard_target_state(flags, fresh), want);
        }
    }

    /// Newly backed placed and reserved RT/DS targets run `DiscardResource` once while keeping
    /// their initial state; aliased placements at the same address do not discard earlier writes.
    #[test]
    fn fresh_placed_and_reserved_targets_discard_without_wiping_aliases() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::{Common::*, *};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return crate::gpu::recomp::runner::skip_test("software adapter");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return crate::gpu::recomp::runner::skip_test("no tier-2 heap GPU") };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let (riid, sample) = (&ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID, DXGI_SAMPLE_DESC { Count: 1, Quality: 0 });
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.Close().unwrap();
            let (fence, fval): (ID3D12Fence, std::cell::Cell<u64>) = (dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(), std::cell::Cell::new(0));
            let run = |rec: &dyn Fn(&ID3D12GraphicsCommandList)| {
                before_execute(&queue);
                alloc.Reset().unwrap();
                list.Reset(&alloc, None).unwrap();
                rec(&list);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                fval.set(fval.get() + 1);
                queue.Signal(&fence, fval.get()).unwrap();
                fence.SetEventOnCompletion(fval.get(), windows::Win32::Foundation::HANDLE::default()).unwrap();
            };
            let place_one = |va: usize, d: &D3D12_RESOURCE_DESC, st: D3D12_RESOURCE_STATES, cv: *const c_void, cb: Option<u64>| {
                let mut p = std::ptr::null_mut();
                assert!(place(dev.as_raw(), va, d, st.0 as u32, cv, cb, riid, &mut p).is_some_and(|hr| hr >= 0) && !p.is_null());
                ID3D12Resource::from_raw(p)
            };
            let read_back = |r: &ID3D12Resource, d: &D3D12_RESOURCE_DESC, fp: &mut [D3D12_PLACED_SUBRESOURCE_FOOTPRINT]| -> Vec<u8> {
                let (mut bytes, mut rb) = (0u64, None::<ID3D12Resource>);
                dev.GetCopyableFootprints(d, 0, fp.len() as u32, 0, Some(fp.as_mut_ptr()), None, None, Some(&mut bytes));
                let bd = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
                let rb = rb.unwrap();
                run(&|l| for (plane, f) in fp.iter().enumerate() {
                    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *f } };
                    let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                    l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                });
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).unwrap();
                let out = std::slice::from_raw_parts(p as *const u8, bytes as usize).to_vec();
                rb.Unmap(0, None);
                out
            };

            let ds_desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds_cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } } };
            let (d0, ds0) = (discards(), place_one(0x6A_0000_0000, &ds_desc, D3D12_RESOURCE_STATE_DEPTH_READ, &ds_cv as *const _ as *const c_void, None));
            assert_eq!(discards(), d0 + 1, "fresh arena depth target must discard");
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            dev.CreateDepthStencilView(&ds0, Some(&D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() }), dsv);
            run(&|l| {
                l.ResourceBarrier(&[barrier(&ds0, D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE)]);
                l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.375, 0x3C, None);
            });
            let ds1 = place_one(0x6A_0000_0000, &ds_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &ds_cv as *const _ as *const c_void, None);
            assert_eq!(discards(), d0 + 1, "aliased arena depth target must not discard");
            run(&|l| l.ResourceBarrier(&[barrier(&ds1, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE)]));
            let mut ds_fp = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let s = read_back(&ds1, &ds_desc, &mut ds_fp);
            assert_eq!((f32::from_le_bytes(s[ds_fp[0].Offset as usize..ds_fp[0].Offset as usize + 4].try_into().unwrap()), s[ds_fp[1].Offset as usize]), (0.375, 0x3C));

            let mut opt = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opt as *mut _ as *mut c_void, std::mem::size_of_val(&opt) as u32).is_ok() && opt.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_2.0 {
                let rt_desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..Default::default() };
                let rgba = [0.2f32, 0.4, 0.6, 1.0];
                let rt_cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Anonymous: D3D12_CLEAR_VALUE_0 { Color: rgba } };
                let (d1, rt0) = (discards(), place_one(0x6B_0000_0000, &rt_desc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &rt_cv as *const _ as *const c_void, Some(PAGE as u64)));
                assert_eq!(discards(), d1 + 1, "fresh tiled render target must discard");
                let rtv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).unwrap();
                let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
                dev.CreateRenderTargetView(&rt0, None, rtv);
                run(&|l| {
                    l.ResourceBarrier(&[barrier(&rt0, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
                    l.ClearRenderTargetView(rtv, &rgba, None);
                });
                let rt1 = place_one(0x6B_0000_0000, &rt_desc, D3D12_RESOURCE_STATE_COPY_SOURCE, &rt_cv as *const _ as *const c_void, Some(PAGE as u64));
                assert_eq!(discards(), d1 + 1, "aliased tiled render target must not discard");
                let mut rt_fp = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 1];
                let s = read_back(&rt1, &rt_desc, &mut rt_fp);
                assert_eq!(&s[rt_fp[0].Offset as usize..rt_fp[0].Offset as usize + 4], &[51, 102, 153, 255]);
                drop((rt0, rt1));
            }
            drop((ds0, ds1));
        }
    }
}
