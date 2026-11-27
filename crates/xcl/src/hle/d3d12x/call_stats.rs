//! lab: `XCL_CALL_STATS=1[,<seconds>]` – how many Win32 / D3D12 calls XCL makes around the
//! title's submits, and their time, by kind; and timers over the parts of the submit hook
//! (`real::q_execute_command_lists`) and of the refresh. Off unless set.
//!
//! Each counted call or part adds 1 and its nanoseconds to its kind (two relaxed adds), once for
//! all threads and once more when the current thread is inside a title ExecuteCommandLists
//! (the "in submits" bank). Times nest: a part counts the calls made inside it too. A lock is
//! first tried: got at once, it counts a call and no time; else the wait is timed. Every
//! `<seconds>` (5) the log has the totals since the start, with the presents so far, in one line:
//! `call stats at <s> s, <presents> presents: <name> <calls> <us> [<calls in submits> <us in
//! submits>]; …` (S/tools/callstats.py takes two such lines apart and gives them a frame).

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

macro_rules! kinds {
    ($($id:ident = $name:literal),* $(,)?) => {
        /// What is counted.
        #[allow(non_camel_case_types, clippy::upper_case_acronyms, dead_code)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(usize)]
        pub(crate) enum C { $($id),* }
        pub(crate) const NAMES: &[&str] = &[$($name),*];
    };
}

kinds! {
    // Win32 calls
    GetWriteWatch = "GetWriteWatch",
    VirtualProtect = "VirtualProtect",
    VirtualQuery = "VirtualQuery",
    ReadProcessMemory = "ReadProcessMemory",
    ReadCommitted = "read_committed",
    // D3D12 calls
    Map = "Resource::Map",
    Unmap = "Resource::Unmap",
    TitleFenceSignal = "title Fence::Signal",
    TitleFenceSetEvent = "title Fence::SetEventOnCompletion",
    TitleFenceCompleted = "title Fence::GetCompletedValue",
    FenceCompleted = "XCL fence_completed",
    FenceSetEvent = "XCL Fence::SetEventOnCompletion",
    EventWait = "XCL WaitForSingleObject (fence event)",
    QueueSignal = "XCL Queue::Signal",
    QueueWait = "XCL Queue::Wait",
    UpdateTileMappings = "UpdateTileMappings",
    CopyTiles = "CopyTiles",
    RecCopyBuffer = "XCL CopyBufferRegion recorded",
    RecCopyTexture = "XCL CopyTextureRegion recorded",
    OwnExecute = "ECL of XCL's own lists (hook)",
    TitleExecute = "ECL of the title (hook)",
    DriverExecute = "ECL driver call (title)",
    // locks
    LockState = "lock sparse STATE",
    LockTextures = "lock TEXTURES",
    LockRefresh = "lock REFRESH",
    // parts of the submit hook
    MapLists = "hook: pc_list / record_deferred / push_segments",
    ReviveListed = "hook: revive_listed",
    Refresh = "hook: refresh_for",
    NoteQueueForLists = "hook: note_queue_for_lists",
    GpuProfileNoteQueue = "hook: gpu_profile::note_queue",
    ExecuteWithWaits = "hook: execute_with_waits_on (all)",
    WaitAllOn = "hook: wait_all_on",
    NoteResubmits = "hook: alloc_pool::note_resubmits",
    CopyScratchSubmitted = "after: copy_scratch_submitted",
    AllocPoolExecuted = "after: alloc_pool::executed",
    SignalWrite64Fences = "after: signal_write64_fences",
    ReadbackSparseWrites = "after: readback_sparse_writes",
    CpuWritesAfter = "after: cpu_writes_after",
    RemovedProbe = "hook: removed_probe (all)",
    Submitted = "hook: title_textures::submitted",
    PresentNoteExecute = "hook: present::note_execute",
    // parts of the refresh
    LookJoin = "refresh: look_join covered / submit_done",
    RecheckViewSources = "refresh: recheck_gpu_view_sources",
    WatchRanges = "refresh: cached_watch_ranges",
    Look = "refresh: look (write watch)",
    SortDirty = "refresh: sort / dedup the written pages",
    ViewHeaps = "refresh: look_at_view_heaps",
    SparseBeforeExecute = "refresh: sparse::before_execute",
    VramBeforeExecute = "refresh: vram::before_execute",
    RecheckStreamed = "refresh: recheck_streamed_unwatched",
    Filter = "refresh: texture filter",
    Revive = "refresh: vram::revive",
    SeedRecheck = "refresh: seed_stream_recheck",
    UploadAll = "refresh: upload_all",
}

pub(crate) const N: usize = NAMES.len();

/// `cs!(Kind, expr)`: `expr`, timed as `C::Kind` (nothing but `expr` when off).
macro_rules! cs {
    ($k:ident, $e:expr) => {{
        let _cs = $crate::hle::d3d12x::call_stats::time($crate::hle::d3d12x::call_stats::C::$k);
        $e
    }};
}
pub(crate) use cs;


/// [calls, ns] of every kind: all threads, and inside a title submit
static ALL: [[AtomicU64; 2]; N] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; N];
static IN_SUBMIT: [[AtomicU64; 2]; N] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; N];

thread_local! {
    /// the current thread is inside a title ExecuteCommandLists
    static SUBMIT: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    static TEST_ON: Cell<bool> = const { Cell::new(false) };
}

/// `1` or `1,<seconds>`: the log's period (5 s); anything else: off.
pub(crate) fn parse(v: &str) -> Option<u64> {
    let mut f = v.trim().split(',');
    if f.next()?.trim() != "1" {
        return None;
    }
    match f.next() {
        None => Some(5),
        Some(s) => s.trim().parse::<u64>().ok().filter(|&s| s >= 1 && f.next().is_none()),
    }
}

fn period() -> Option<u64> {
    static P: OnceLock<Option<u64>> = OnceLock::new();
    *P.get_or_init(|| {
        let p = std::env::var("XCL_CALL_STATS").ok().and_then(|v| parse(&v));
        #[cfg(not(test))]
        if let Some(s) = p {
            let _ = std::thread::Builder::new().name("xcl-call-stats".into()).spawn(move || loop {
                std::thread::sleep(Duration::from_secs(s));
                crate::xlog_info!("{}", line(crate::log::uptime(), super::present::activity().0, &snapshot()));
            });
            crate::xlog_info!("call stats: Win32 / D3D12 calls and parts of the submit counted, totals in the log every {s} s (XCL_CALL_STATS)");
        }
        p
    })
}

#[inline]
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if TEST_ON.with(|c| c.get()) {
        return true;
    }
    period().is_some()
}

pub(crate) fn add(c: C, d: Duration) {
    let ns = d.as_nanos() as u64;
    let i = c as usize;
    ALL[i][0].fetch_add(1, Ordering::Relaxed);
    ALL[i][1].fetch_add(ns, Ordering::Relaxed);
    if SUBMIT.with(|s| s.get()) {
        IN_SUBMIT[i][0].fetch_add(1, Ordering::Relaxed);
        IN_SUBMIT[i][1].fetch_add(ns, Ordering::Relaxed);
    }
}

/// Times its kind until dropped (nothing when off).
pub(crate) struct Timer(Option<(C, Instant)>);

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some((c, t)) = self.0 {
            add(c, t.elapsed());
        }
    }
}

#[inline]
pub(crate) fn time(c: C) -> Timer {
    Timer(on().then(|| (c, Instant::now())))
}

/// `f`, timed as `c`.
#[inline]
#[allow(dead_code)]
pub(crate) fn timed<R>(c: C, f: impl FnOnce() -> R) -> R {
    let _t = time(c);
    f()
}

/// `m` locked; the wait counted as `c` (a lock got at once: a call, no time).
pub(crate) fn lock<'a, T>(m: &'a Mutex<T>, c: C) -> MutexGuard<'a, T> {
    if !on() {
        return m.lock().unwrap();
    }
    match m.try_lock() {
        Ok(g) => {
            add(c, Duration::ZERO);
            g
        }
        Err(std::sync::TryLockError::WouldBlock) => {
            let t = Instant::now();
            let g = m.lock().unwrap();
            add(c, t.elapsed());
            g
        }
        Err(std::sync::TryLockError::Poisoned(e)) => panic!("{e}"),
    }
}

/// While alive, the current thread is inside a title ExecuteCommandLists (the "in submits" bank).
pub(crate) struct InSubmit(bool);

impl Drop for InSubmit {
    fn drop(&mut self) {
        let was = self.0;
        SUBMIT.with(|s| s.set(was));
    }
}

pub(crate) fn in_submit() -> InSubmit {
    InSubmit(SUBMIT.with(|s| s.replace(true)))
}

/// [all calls, all ns, in-submit calls, in-submit ns] of every kind.
pub(crate) fn snapshot() -> Vec<[u64; 4]> {
    (0..N)
        .map(|i| [ALL[i][0].load(Ordering::Relaxed), ALL[i][1].load(Ordering::Relaxed), IN_SUBMIT[i][0].load(Ordering::Relaxed), IN_SUBMIT[i][1].load(Ordering::Relaxed)])
        .collect()
}

/// The log line of a snapshot (kinds never counted left out).
pub(crate) fn line(uptime: f64, presents: u64, s: &[[u64; 4]]) -> String {
    let parts: Vec<String> = s
        .iter()
        .enumerate()
        .filter(|(_, v)| v[0] > 0)
        .map(|(i, v)| format!("{} {} {} [{} {}]", NAMES[i], v[0], v[1] / 1000, v[2], v[3] / 1000))
        .collect();
    format!("call stats at {uptime:.1} s, {presents} presents: {}", parts.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(a: &[[u64; 4]], b: &[[u64; 4]], c: C) -> [u64; 4] {
        let i = c as usize;
        std::array::from_fn(|k| b[i][k] - a[i][k])
    }

    #[test]
    fn the_switch_is_one_with_a_period() {
        assert_eq!(parse("1"), Some(5));
        assert_eq!(parse(" 1 , 10 "), Some(10));
        assert_eq!(parse("0"), None);
        assert_eq!(parse("1,0"), None);
        assert_eq!(parse("1,x"), None);
        assert_eq!(parse("1,5,6"), None);
        assert_eq!(parse(""), None);
        assert_eq!(NAMES.len(), C::UploadAll as usize + 1, "one name a kind");
    }

    #[test]
    fn timers_locks_and_the_submit_bank_add_up() {
        // off (this thread): nothing is counted
        let s0 = snapshot();
        timed(C::Look, || std::thread::sleep(Duration::from_millis(2)));
        assert_eq!(delta(&s0, &snapshot(), C::Look), [0; 4]);
        TEST_ON.with(|c| c.set(true));
        let s0 = snapshot();
        timed(C::Look, || std::thread::sleep(Duration::from_millis(10)));
        {
            let _s = in_submit();
            let _t = time(C::Look);
            std::thread::sleep(Duration::from_millis(5));
        }
        // the bank ends with the guard
        timed(C::Look, || ());
        let d = delta(&s0, &snapshot(), C::Look);
        assert_eq!((d[0], d[2]), (3, 1));
        assert!(d[1] >= 15_000_000 && d[3] >= 5_000_000 && d[3] < d[1], "{d:?}");
        // a lock got at once: a call and no time; one held by another thread: the wait
        let m = std::sync::Arc::new(Mutex::new(0u32));
        drop(lock(&m, C::LockTextures));
        let m2 = m.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let h = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(30));
        });
        rx.recv().unwrap();
        *lock(&m, C::LockTextures) += 1;
        h.join().unwrap();
        let d = delta(&s0, &snapshot(), C::LockTextures);
        assert_eq!(d[0], 2);
        assert!(d[1] >= 20_000_000, "the wait was timed: {d:?}");
        TEST_ON.with(|c| c.set(false));
        let l = line(12.34, 7, &snapshot());
        assert!(l.starts_with("call stats at 12.3 s, 7 presents: "), "{l}");
        assert!(l.contains("refresh: look (write watch) "), "{l}");
        assert!(!l.contains("CopyTiles"), "kinds never counted are left out: {l}");
    }
}
