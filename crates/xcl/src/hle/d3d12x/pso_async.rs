//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for where it is used, a `Dispatch` at its submit (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    /// the driver pipeline binds get till then (`shared`); 0: the stand-in is one itself
    driver: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
    /// its dispatches may be left out while it is built (`image_only`): binds do not wait for it
    spared: std::sync::atomic::AtomicBool,
    demand: Mutex<Demand>,
    /// lab (`code`): builds the real pipeline again after it was given back to the driver
    again: Option<Again>,
    /// lab: bytes of the driver's compiled code of the real pipeline (0: not measured)
    code: AtomicUsize,
    /// lab: of that, its two-pass compute pass
    code_pass: AtomicUsize,
    /// lab: `code::now_ms` and the present of its last use with the real pipeline (0: never)
    last_use: std::sync::atomic::AtomicU64,
    last_frame: std::sync::atomic::AtomicU64,
    /// lab: its code counts as in use (used since it was built)
    in_use: std::sync::atomic::AtomicBool,
    /// lab: its real pipeline was given back and is not built again yet
    given: std::sync::atomic::AtomicBool,
    /// lab: `code::now_ms` when its real pipeline was built
    built_ms: std::sync::atomic::AtomicU64,
    /// lab: its last build left no driver blob for the next one (`real::BuildNote::blobless`):
    /// built again, it is compiled in full
    blobless: std::sync::atomic::AtomicBool,
}

/// Builds a real pipeline as often as asked (`submit_lazy_again`).
pub type Again = Arc<dyn Fn() -> usize + Send + Sync>;

/// Whether one of the title's threads waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    wanted: bool,
    /// a call put off to the submit needs it (`need_by_submit`)
    by_submit: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// This thread is a worker and one of the title's threads waits for what it is building: the
/// build must not stand back for the title's sake (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.demand.lock().unwrap().wanted))
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `build_is_wanted`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || p.demand.lock().unwrap().wanted);
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

/// A worker's build of `p` (`preload`: by one of the lowest-priority workers).
fn build(p: &Arc<Pending>, job: Job, preload: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
    if preload {
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    complete(p, job());
    if preload {
        // (under the lock: `want` does not raise this thread after it)
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = 0;
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        }
    }
    BUILDING.with(|b| *b.borrow_mut() = None);
}

impl Pending {
    /// One of the title's threads is about to wait for it: the worker building it goes on now,
    /// at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_PRIORITY_NORMAL, THREAD_SET_LIMITED_INFORMATION};
        let mut d = self.demand.lock().unwrap();
        if !std::mem::replace(&mut d.wanted, true) && d.preload_thread != 0 {
            unsafe {
                let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
                if !h.is_null() {
                    SetThreadPriority(h, THREAD_PRIORITY_NORMAL);
                    CloseHandle(h);
                }
            }
        }
    }

    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
        code::built(p, real);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// The PC calls of the title's threads (`pc_call`): the ones going on, and how long the ones
/// that ended took since the last frame was shown.
struct PcCalls {
    /// (id, when it began)
    now: Vec<(u64, std::time::Instant)>,
    /// the frames shown when `spent` began to count (`None` before the first observation)
    frame: Option<u64>,
    /// when the latest frame advance was first observed; calls that started earlier are clamped here
    frame_since: Option<std::time::Instant>,
    /// the time of the PC calls that ended since then
    spent: std::time::Duration,
}

impl PcCalls {
    const fn new() -> Self {
        Self { now: Vec::new(), frame: None, frame_since: None, spent: std::time::Duration::ZERO }
    }

    /// `frame` frames have been shown: the calls before the last of them are not this wait's.
    fn at(&mut self, frame: u64, now: std::time::Instant) {
        match self.frame {
            None => self.frame = Some(frame),
            Some(prev) if frame > prev => {
                self.frame = Some(frame);
                self.frame_since = Some(now);
                self.spent = std::time::Duration::ZERO;
            }
            Some(_) => {}
        }
    }

    fn clamp(&self, since: std::time::Instant) -> std::time::Instant {
        self.frame_since.map_or(since, |f| since.max(f))
    }

    fn begin(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        self.now.push((id, now));
    }

    fn end(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        if let Some(i) = self.now.iter().position(|c| c.0 == id) {
            let (_, since) = self.now.swap_remove(i);
            self.spent += now.saturating_duration_since(self.clamp(since));
        }
    }

    /// Whether a call going on holds the title: the PC calls since the last frame, it included,
    /// have taken `PC_CALL_HOLDS_AFTER`.
    fn holds(&mut self, frame: u64, now: std::time::Instant) -> bool {
        self.at(frame, now);
        let spent = self.spent;
        self.now.iter().any(|(_, since)| spent + now.saturating_duration_since(self.clamp(*since)) >= PC_CALL_HOLDS_AFTER)
    }
}

static PC_CALLS: Mutex<PcCalls> = Mutex::new(PcCalls::new());
/// From this much time in PC calls since the last frame on, a PC call holds the title: the
/// console's calls return at once.
const PC_CALL_HOLDS_AFTER: std::time::Duration = std::time::Duration::from_millis(250);

/// Whether one of the title's threads is held where a console title never waits: a bind is
/// waiting for a pipeline still being built, or a PC call is going on after such calls have
/// kept the title's threads for long (`pc_call`). `kernelx::Sleep` does not count that time.
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0 || held_by_pc_call()
}

fn held_by_pc_call() -> bool {
    PC_CALLS.lock().unwrap().holds(super::present::presents(), std::time::Instant::now())
}

/// A call of the title into D3D12 that XCL or the PC's driver can keep for long where the
/// console's takes no time: a submission (XCL's uploads and tile mappings before the lists, the
/// driver's compiling in them), a texture's placement, a pipeline built in the call. While it
/// lasts the title is held (`title_held`), once the PC calls since the last frame shown, this
/// one included, have taken `PC_CALL_HOLDS_AFTER`: one long call, or many short ones in a row
/// (a retail title placed 16 384 textures in 29 s without a frame, 1 - 50 ms each). A title
/// that shows its frames is not held by its submissions.
pub struct PcCall(u64);

pub fn pc_call() -> PcCall {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    PC_CALLS.lock().unwrap().begin(id, super::present::presents(), std::time::Instant::now());
    PcCall(id)
}

impl Drop for PcCall {
    fn drop(&mut self) {
        PC_CALLS.lock().unwrap().end(self.0, super::present::presents(), std::time::Instant::now());
    }
}

/// Whether this thread is building a pipeline for `build` (a worker, not one of the title's).
pub(super) fn building_here() -> bool {
    BUILDING.with(|b| b.borrow().is_some())
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

/// pipelines a submit will wait for: the workers take these before their queue (`need_by_submit`)
static NEEDED: Mutex<std::collections::VecDeque<Arc<Pending>>> = Mutex::new(std::collections::VecDeque::new());

fn next_needed() -> Option<Arc<Pending>> {
    NEEDED.lock().unwrap().pop_front()
}

/// Drop an unstarted orphaned job outside all locks and wake any concurrent waiter.
fn drop_unstarted_job(p: &Pending) {
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        {
            let mut done = p.done.lock().unwrap();
            *done = true;
            p.cv.notify_all();
        }
        drop(job);
    }
}

type Work = Box<dyn FnOnce() + Send>;

fn queue() -> &'static Mutex<mpsc::Sender<Work>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Work>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Work>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(work) = next else { return };
                while let Some(p) = next_needed() {
                    build_queued(&p);
                }
                work();
            });
        }
        Mutex::new(tx)
    })
}

fn enqueue(p: Arc<Pending>) {
    let _ = queue().lock().unwrap().send(Box::new(move || build_queued(&p)));
}

fn build_queued(p: &Arc<Pending>) {
    // (released by the title meanwhile)
    if p.orphan.load(Ordering::Acquire) {
        drop_unstarted_job(p);
        return;
    }
    // (None: a bind took it and builds it itself)
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        build(p, job, false);
    }
}

/// The caller also drains its batch. Multiple workers can therefore build
/// different chains without ever waiting for a free worker to start a piece.
pub(super) fn build_pieces(jobs: Vec<Job>) -> Vec<usize> {
    let count = jobs.len();
    let jobs = Arc::new(Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>()));
    let (send, recv) = mpsc::channel();
    let drain = move || loop {
        let job = jobs.lock().unwrap().pop();
        let Some((i, job)) = job else { return };
        let _ = send.send((i, job()));
    };
    // the helpers build for this worker's pipeline: the title waiting for it waits for them too
    let parent = BUILDING.with(|b| b.borrow().clone());
    for _ in 1..count.min(crate::gpu::recomp::dxc::compile_threads()) {
        let (drain, parent) = (drain.clone(), parent.clone());
        let _ = queue().lock().unwrap().send(Box::new(move || {
            let before = BUILDING.with(|b| b.replace(parent));
            drain();
            BUILDING.with(|b| *b.borrow_mut() = before);
        }));
    }
    drain();
    let mut results = vec![0; count];
    for (i, result) in recv.iter().take(count) { results[i] = result; }
    results
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0, None);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key, None);
}

/// `submit_lazy` with a build that can run again: the real pipeline may be given back to the
/// driver while nothing uses it and is built again at its next bind (`code`).
pub fn submit_lazy_again(standin: usize, compute: bool, again: Again, key: u64) {
    let first = again.clone();
    submit_as(standin, compute, Box::new(move || first()), false, key, Some(again));
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64, again: Option<Again>) {
    // (lab `code`: the build note of this build only, not of what ran on the thread before it)
    let job: Job = Box::new(move || {
        let _ = super::real::take_build_note();
        job()
    });
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        driver: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        spared: Default::default(),
        demand: Default::default(),
        again,
        code: Default::default(),
        code_pass: Default::default(),
        last_use: Default::default(),
        last_frame: Default::default(),
        in_use: Default::default(),
        given: Default::default(),
        built_ms: Default::default(),
        blobless: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
    } else if used::known(key) {
        if let Some(q) = preload() {
            q.list.lock().unwrap().push_back(p);
            q.cv.notify_one();
        }
    }
}

/// The shaders of a pipeline's set, told when the set's key is made (`real::OwnedDesc`).
pub(crate) fn shaders_of(key: u64, shaders: Vec<u64>) {
    used::parts(key, shaders);
}

/// The shaders of the pipelines earlier runs used, each by its own key (`used-shaders.txt`
/// beside the DXIL cache): what a title's shader preparation translates. Empty before a run
/// with this list has used a pipeline.
pub fn earlier_used_shaders() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-shaders.txt")).unwrap_or_default())
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` beside the DXIL cache):
/// what `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")).unwrap_or_default())
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
        /// the shaders of each set made in this run, by the set's key; the shaders of the sets
        /// used so far and `used-shaders.txt`, where they are kept for the next run
        parts: Mutex<std::collections::HashMap<u64, Vec<u64>>>,
        shaders: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)), parts: Default::default(), shaders: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            let shaders_path = path.with_file_name("used-shaders.txt");
            let shaders = parse(&std::fs::read_to_string(&shaders_path).unwrap_or_default());
            let shaders_file = std::fs::OpenOptions::new().create(true).append(true).open(&shaders_path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier), parts: Default::default(), shaders: Mutex::new((shaders, shaders_file)) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        {
            let mut g = get().now.lock().unwrap();
            if g.0.insert(key) {
                if let Some(f) = g.1.as_mut() {
                    let _ = writeln!(f, "{key:016x}");
                }
            }
        }
        // its shaders, each once (also those of a set an earlier run used: the list of
        // shaders is younger than the list of sets)
        let parts = get().parts.lock().unwrap().get(&key).cloned().unwrap_or_default();
        let mut g = get().shaders.lock().unwrap();
        for part in parts {
            if g.0.insert(part) {
                if let Some(f) = g.1.as_mut() {
                    let _ = writeln!(f, "{part:016x}");
                }
            }
        }
    }

    /// The shaders of the set `key`, each by its own key (`real::shader_key`).
    pub fn parts(key: u64, shaders: Vec<u64>) {
        if key != 0 {
            get().parts.lock().unwrap().entry(key).or_insert(shaders);
        }
    }

    /// (tests) This run used the shader `key`.
    #[cfg(test)]
    pub fn shader_noted(key: u64) -> bool {
        get().shaders.lock().unwrap().0.contains(&key)
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it), and the title's own threads are not either.
struct Preload {
    list: Mutex<std::collections::VecDeque<Arc<Pending>>>,
    cv: Condvar,
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.pop_front() {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        drop_unstarted_job(&p);
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        build(&p, job, true);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    /// The shaders of a set are kept when a pipeline of the set is used, each once, and not
    /// before: a set that is only made keeps nothing.
    #[test]
    fn the_shaders_of_a_used_set_are_kept_for_the_next_run() {
        let (set, other) = (0x5e70_0000_0000_0001u64, 0x5e70_0000_0000_0003u64);
        let (vs, ps, cs) = (0x5e70_0000_0000_0011u64, 0x5e70_0000_0000_0013u64, 0x5e70_0000_0000_0015u64);
        shaders_of(set, vec![vs, ps]);
        shaders_of(other, vec![cs]);
        assert!(!used::shader_noted(vs) && !used::shader_noted(ps), "made, not used yet");
        used::note(set);
        assert!(used::shader_noted(vs) && used::shader_noted(ps), "the shaders of the used set");
        assert!(!used::shader_noted(cs), "the other set was not used");
    }

    /// A PC call that has kept one of the title's threads for long holds the title, a short one
    /// does not, and the hold ends with the call.
    #[test]
    fn a_long_pc_call_holds_the_title() {
        let mut calls = PcCalls::new();
        let began = std::time::Instant::now();
        calls.begin(1, 0, began);
        calls.begin(2, 0, began + 2 * PC_CALL_HOLDS_AFTER);
        assert!(!calls.holds(0, began), "a call that has just begun holds the title");
        let later = began + 2 * PC_CALL_HOLDS_AFTER;
        assert!(calls.holds(0, later), "a PC call has lasted {:?}, but the title is not held", 2 * PC_CALL_HOLDS_AFTER);
        calls.end(1, 0, later);
        calls.end(2, 0, later);
        assert!(!calls.holds(0, later), "the calls are over, the title is still held");
    }

    /// Short PC calls add up between two frames: none of them is long, but after many in a row
    /// the one going on holds the title. The title's own time between them is not held, and a
    /// frame shown ends it.
    #[test]
    fn short_pc_calls_between_two_frames_hold_the_title() {
        let mut calls = PcCalls::new();
        let step = PC_CALL_HOLDS_AFTER / 10;
        let mut now = std::time::Instant::now();
        for id in 0..20 {
            calls.begin(id, 7, now);
            assert!(id >= 9 || !calls.holds(7, now + step), "{id} short calls hold the title already");
            now += step;
            calls.end(id, 7, now);
        }
        assert!(!calls.holds(7, now), "no PC call is going on, the title is held");
        calls.begin(20, 7, now);
        assert!(calls.holds(7, now), "twenty short PC calls since the last frame and one going on, but the title is not held");
        assert!(!calls.holds(8, now), "a frame was shown, the calls before it still hold the title");
        calls.end(20, 8, now);
    }

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = started.send(());
            let _ = finish.recv();
            0x5252
        }), 0);
        let bind = std::thread::spawn(move || resolve_wait_for(standin, None));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        // (built in the bind, or by a worker with the bind about to wait for it)
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    /// A stand-in that is no driver object (`shared`) never reaches the driver: binds get the
    /// driver pipeline shared with other stand-ins until the real one is built, and after a build
    /// that failed.
    #[test]
    fn a_bind_of_a_shared_stand_in_gets_its_driver_pipeline() {
        let (fails, builds, driver) = (0x7E57_B011usize, 0x7E57_B012usize, 0x7E57_B013usize);
        submit_lazy(fails, false, Box::new(|| 0), 0);
        submit_lazy(builds, false, Box::new(|| 0x1234), 0);
        for s in [fails, builds] {
            shared(s, driver);
            assert_eq!(entry(s).unwrap().stand(s), driver);
        }
        assert_eq!(resolve_wait_for(fails, None), driver, "its build failed");
        assert_eq!(resolve(fails), driver);
        assert_eq!(resolve_wait_for(builds, None), 0x1234);
        assert_eq!(resolve_for_bind(builds), 0x1234);
        // (0x1234 is no pipeline: taken, not released)
        assert_eq!((released(fails), released(builds)), (0, 0x1234));
    }

    /// A bind does not wait for a compute pipeline that writes only images: it gets the stand-in
    /// at once, with no budget, the build goes to the workers
    /// unwanted, and once built the bind gets the real one. Without `image_only` the same bind
    /// waits until the pipeline is built.
    #[test]
    fn a_bind_does_not_wait_for_a_pipeline_that_writes_only_images() {
        let (spared, waited) = (0x7E57_B0A1usize, 0x7E57_B0A2usize);
        let (go, finish) = mpsc::channel::<()>();
        let (tx, rx) = mpsc::channel();
        submit_lazy(spared, true, Box::new(move || {
            let _ = tx.send(wanted_here()());
            let _ = finish.recv();
            0x5A5A
        }), 0);
        image_only(spared);
        // (on a thread: a bind that waits would wait for `go`, below)
        let (bound_tx, bound) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = bound_tx.send((resolve_wait_for(spared, None), resolve_wait_for(spared, None)));
        });
        let binds = bound.recv_timeout(std::time::Duration::from_secs(5));
        if binds.is_err() {
            let _ = go.send(());
        }
        assert_eq!(binds.expect("the binds waited for the build"), (spared, spared), "still being built: the stand-in");
        assert!(is_pending(spared));
        assert!(!rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it"), "nobody waits: the build is not wanted");
        go.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(spared) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(spared, None), 0x5A5A, "built by a worker");
        // the same build without the mark: the bind waits for it
        submit_lazy(waited, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            0x5B5B
        }), 0);
        assert_eq!(resolve_wait_for(waited, None), 0x5B5B);
        let _ = released(spared);
        let _ = released(waited);
    }

    #[test]
    fn a_pipeline_a_submit_needs_is_built_before_the_queue() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let backlog = 8 * crate::gpu::recomp::dxc::compile_threads();
        let standins: Vec<usize> = (0..=backlog).map(|i| 0x7E57_E000usize + i).collect();
        for &standin in &standins {
            let order = order.clone();
            submit(standin, true, Box::new(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push(standin);
                0x6161
            }));
        }
        let last = *standins.last().unwrap();
        need_by_submit(last);
        let start = std::time::Instant::now();
        while standins.iter().any(|&s| is_pending(s)) && start.elapsed() < std::time::Duration::from_secs(30) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let order = order.lock().unwrap();
        assert_eq!(order.len(), standins.len());
        let at = order.iter().position(|&s| s == last).unwrap();
        assert!(at < order.len() / 2, "made last and needed by a submit, it was built {at} of {}", order.len());
        for &standin in &standins {
            let _ = released(standin);
        }
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, None), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, None), 0x5353, "the bind's wait reached the build");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    static WORKER_SATURATION_LOCK: Mutex<()> = Mutex::new(());

    /// A queued normal worker job whose stand-in was released before a worker starts it is
    /// skipped as orphaned instead of entering the compiler.
    #[test]
    fn a_released_queued_pipeline_is_not_built_by_normal_workers() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_9720usize + k;
            let started_tx = started_tx.clone();
            let gate = gate.clone();
            submit(standin, false, Box::new(move || {
                let _ = started_tx.send(());
                let mut open = gate.0.lock().unwrap();
                while !*open {
                    open = gate.1.wait(open).unwrap();
                }
                0x9720 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }

        let target = 0x7E57_97FFusize;
        let ran = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let ran_job = ran.clone();
        let drop_flag = DropFlag(dropped.clone());
        submit(target, false, Box::new(move || {
            let _keep = &drop_flag;
            ran_job.store(true, Ordering::Release);
            0x9721
        }));
        assert_eq!(released(target), 0);

        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !dropped.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(dropped.load(Ordering::Acquire), "queued orphan job was never drained");
        assert!(!ran.load(Ordering::Acquire), "released job reached the compiler");

        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }

    #[test]
    fn pc_call_crossing_frame_boundary_does_not_charge_prior_frame_time() {
        let mut calls = PcCalls::new();
        let t0 = std::time::Instant::now();
        let ms = std::time::Duration::from_millis;
        calls.begin(1, 7, t0);
        // Frame 8 shown after 290 ms; call 1 ends 10 ms later -> only 10 ms counts on frame 8.
        let t_frame8 = t0 + PC_CALL_HOLDS_AFTER + ms(40);
        assert!(!calls.holds(8, t_frame8), "call from frame 7 must not immediately hold frame 8");
        calls.end(1, 8, t_frame8 + ms(10));
        let t_next = t_frame8 + ms(11);
        calls.begin(2, 8, t_next);
        assert!(!calls.holds(8, t_next), "pre-frame duration leaked into frame 8's budget");
        // A call genuinely stuck >= PC_CALL_HOLDS_AFTER after frame 8 still holds the title.
        assert!(!calls.holds(8, t_next + PC_CALL_HOLDS_AFTER - ms(20)));
        assert!(calls.holds(8, t_next + PC_CALL_HOLDS_AFTER));
        calls.end(2, 8, t_next + PC_CALL_HOLDS_AFTER);

        // Also when frame 9 is first noticed inside end() itself, pre-frame time is clamped to 0.
        let t_f9 = t_next + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(3, 8, t_f9);
        let t_f9_end = t_f9 + PC_CALL_HOLDS_AFTER + ms(50);
        calls.end(3, 9, t_f9_end);
        calls.begin(4, 9, t_f9_end + ms(1));
        assert!(!calls.holds(9, t_f9_end + ms(1)), "end() on new frame must not charge prior-frame duration");
        calls.end(4, 9, t_f9_end + ms(1) + PC_CALL_HOLDS_AFTER);

        // A stale lower frame number must never rewind self.frame or wipe frame 9's spent budget.
        let t_stale = t_f9_end + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(5, 8, t_stale);
        assert!(calls.holds(9, t_stale), "stale lower frame rewound frame 9 or cleared its spent budget");
        calls.end(5, 9, t_stale);
    }

    #[test]
    fn released_queued_pipeline_drops_job_immediately_before_workers_drain() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        struct OpenGate(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for OpenGate {
            fn drop(&mut self) {
                *self.0 .0.lock().unwrap() = true;
                self.0 .1.notify_all();
            }
        }

        // Saturate normal workers and verify released() drops the queued job immediately.
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_normal = OpenGate(gate.clone());
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_A100usize + k;
            let (tx, g) = (started_tx.clone(), gate.clone());
            submit(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA100 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }
        let target_n = 0x7E57_A1FFusize;
        let (ran_n, dropped_n) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_n) = (ran_n.clone(), DropFlag(dropped_n.clone()));
        submit(target_n, false, Box::new(move || {
            let _keep = &flag_n;
            ran_job.store(true, Ordering::Release);
            0xA1FF
        }));
        assert_eq!(released(target_n), 0);
        assert!(dropped_n.load(Ordering::Acquire), "normal queued job was not dropped immediately on released()");
        drop(open_normal);
        assert!(!ran_n.load(Ordering::Acquire));
        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }

        // Saturate preload workers and verify released() drops the preloaded job immediately.
        let m = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        let (p_tx, p_rx) = mpsc::channel();
        let p_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_pre = OpenGate(p_gate.clone());
        let mut p_blockers = Vec::with_capacity(m);
        for k in 0..m {
            let standin = 0x7E57_A200usize + k;
            let key = 0x5EED_A200u64 + k as u64;
            used::earlier_run_used(key);
            let (tx, g) = (p_tx.clone(), p_gate.clone());
            submit_lazy(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA200 + k
            }), key);
            p_blockers.push(standin);
        }
        for _ in 0..m {
            p_rx.recv_timeout(Duration::from_secs(10)).expect("preload workers saturated");
        }
        let target_p = 0x7E57_A2FFusize;
        let key_p = 0x5EED_A2FFu64;
        used::earlier_run_used(key_p);
        let (ran_p, dropped_p) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_p) = (ran_p.clone(), DropFlag(dropped_p.clone()));
        submit_lazy(target_p, false, Box::new(move || {
            let _keep = &flag_p;
            ran_job.store(true, Ordering::Release);
            0xA2FF
        }), key_p);
        assert_eq!(released(target_p), 0);
        assert!(dropped_p.load(Ordering::Acquire), "preload queued job was not dropped immediately on released()");
        drop(open_pre);
        assert!(!ran_p.load(Ordering::Acquire));
        for standin in p_blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }
}

/// Lab: the compiled shader code the driver holds for the pipelines in use, and giving back
/// the ones nothing has used for a while.
///
/// NVIDIA's driver keeps the machine code of every pipeline from its first use until its
/// release in one heap of at most 512 MB, and removes the device (0x887a0005, nothing in DRED)
/// at the submit that needs more (xcl-dev/xcl issue 2403). A retail title's world uses ~1100
/// translated compute pipelines of ~435 MB of code (`GetCachedBlob`) on a 32-lane GPU.
///
/// `XCL_PSO_CODE=1`: a `shader code lab:` line every 2 s (`tick`). `XCL_PSO_CODE_MB=<n>`: also,
/// while the code in use is over n MB, the real pipelines of stand-ins (compute ones;
/// `XCL_PSO_CODE_GFX=1`: graphics ones too) not used for `XCL_PSO_IDLE_S` seconds (8) and 3
/// presents are released, longest unused first – a few seconds and 2 presents after they were
/// taken from their stand-in (lists recorded with them have run by then). The next bind builds
/// one again, from the driver's blob of the first build (milliseconds).
/// `XCL_PSO_CODE_CAP_MB=<n>`: a pipeline not in use whose code would bring the code in use (and
/// waiting for its release) over n MB is not used: its binds get the stand-in, as those of a
/// pipeline still being built (dispatches and draws skipped), until there is room.
/// `XCL_PSO_UNUSED_S=<n>`: a real pipeline nothing has used n seconds after it was built (built
/// ahead of its use because an earlier run used it) is given back whatever the budget: built
/// pipelines cost memory in the driver (~0.6 GB for a retail title's 4500).
/// `XCL_PSO_KEEP_BLOBLESS=1`: ... except one whose build left no driver blob for the next one
/// (`real::BuildNote::blobless`): built again at a bind, it is compiled in full on the binding
/// thread (4 - 7 s each for a retail title's mesh pipelines). The budget may still give it back.
/// A build again that takes 1 s or more is logged (`built again at bind:`), and the line counts
/// the pipelines given back without a blob.
mod code {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn var(name: &str) -> Option<u64> {
        std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
    }

    /// bytes of code in use to stay under (None: nothing is given back)
    fn budget() -> Option<usize> {
        static B: OnceLock<Option<usize>> = OnceLock::new();
        *B.get_or_init(|| var("XCL_PSO_CODE_MB").map(|mb| (mb as usize) << 20))
    }

    pub fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| budget().is_some() || var("XCL_PSO_CODE") == Some(1))
    }

    fn idle_ms() -> u64 {
        static I: OnceLock<u64> = OnceLock::new();
        *I.get_or_init(|| var("XCL_PSO_IDLE_S").unwrap_or(5) * 1000)
    }

    fn unused_ms() -> Option<u64> {
        static U: OnceLock<Option<u64>> = OnceLock::new();
        *U.get_or_init(|| var("XCL_PSO_UNUSED_S").map(|s| s * 1000))
    }

    fn cap() -> Option<usize> {
        static C: OnceLock<Option<usize>> = OnceLock::new();
        *C.get_or_init(|| var("XCL_PSO_CODE_CAP_MB").map(|mb| (mb as usize) << 20))
    }

    /// `XCL_PSO_KEEP_BLOBLESS=1`
    fn keep_blobless() -> bool {
        static K: OnceLock<bool> = OnceLock::new();
        *K.get_or_init(|| var("XCL_PSO_KEEP_BLOBLESS") == Some(1))
    }

    /// A build again of this long or more is logged.
    const SLOW_REBUILD: std::time::Duration = std::time::Duration::from_secs(1);

    /// Whether a built pipeline nothing has used is given back for `XCL_PSO_UNUSED_S` (`unused`
    /// in ms, None: off), `since_built` ms after its build: one that can be built again, and
    /// with `keep_blobless` not one whose build left no driver blob.
    fn given_back_unused(unused: Option<u64>, since_built: u64, can_again: bool, blobless: bool, keep_blobless: bool) -> bool {
        unused.is_some_and(|ms| since_built >= ms) && can_again && !(keep_blobless && blobless)
    }

    /// Whether the pipeline built by the build `note` is noted as without a blob: its build said
    /// so, or (`again`: a build again) its slowest compile took `SLOW_REBUILD` or more without
    /// a driver blob.
    fn blobless_after(note: &super::super::real::BuildNote, again: bool) -> bool {
        note.blobless || (again && note.slowest.as_ref().is_some_and(|s| !s.3 && s.2 >= SLOW_REBUILD))
    }

    /// The line of a build again that took `took` (None under `SLOW_REBUILD`): the slowest driver
    /// compile of `note`, the thread (`thread`: id and name).
    fn rebuild_line(took: std::time::Duration, note: &super::super::real::BuildNote, thread: (u32, Option<&str>)) -> Option<String> {
        if took < SLOW_REBUILD {
            return None;
        }
        let compile = match &note.slowest {
            Some((kind, record, t, _)) => format!("{kind}, record {record:016x}, {} ms (slowest driver compile {} ms)", took.as_millis(), t.as_millis()),
            None => format!("no driver compile on this thread, {} ms", took.as_millis()),
        };
        let blob = match &note.slowest {
            Some((_, _, _, true)) => "yes",
            Some(_) => "no",
            None => "?",
        };
        Some(format!("built again at bind: {compile}, thread {} ({}), blob {blob}", thread.0, thread.1.unwrap_or("unnamed")))
    }

    fn graphics_too() -> bool {
        static G: OnceLock<bool> = OnceLock::new();
        *G.get_or_init(|| var("XCL_PSO_CODE_GFX") == Some(1))
    }

    /// presents since its last use before a pipeline is taken from its stand-in
    const IDLE_FRAMES: u64 = 2;
    /// … and time and presents from then to its release
    const GRAVE_MS: u64 = 2000;
    const GRAVE_FRAMES: u64 = 2;

    /// milliseconds since the first call, never 0
    fn now_ms() -> u64 {
        static T0: OnceLock<std::time::Instant> = OnceLock::new();
        T0.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
    }

    fn frame() -> u64 {
        super::super::present::presents()
    }

    static GIVEN_BACK: AtomicUsize = AtomicUsize::new(0);
    /// of those, given back without a driver blob for their build again (`Pending::blobless`)
    static GIVEN_BACK_BLOBLESS: AtomicUsize = AtomicUsize::new(0);
    static REBUILT: AtomicUsize = AtomicUsize::new(0);
    static REBUILD_US: AtomicU64 = AtomicU64::new(0);
    static EVER_USED: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];
    /// bytes of code in use, and of pipelines waiting for their release (`GRAVE`)
    static IN_USE: AtomicUsize = AtomicUsize::new(0);
    static GRAVE_BYTES: AtomicUsize = AtomicUsize::new(0);
    /// binds that got the stand-in because of `cap`
    static HELD_BACK: AtomicUsize = AtomicUsize::new(0);
    /// pipelines released here that something else still held a reference to
    static STILL_HELD: AtomicUsize = AtomicUsize::new(0);

    /// real pipelines taken from their stand-ins, to release: (pipeline, when, present, code)
    static GRAVE: Mutex<Vec<(usize, u64, u64, usize)>> = Mutex::new(Vec::new());

    /// `real` was built for `p`: its code is measured once.
    pub fn built(p: &Pending, real: usize) {
        let again = p.given.swap(false, Ordering::AcqRel);
        p.built_ms.store(now_ms(), Ordering::Relaxed);
        p.blobless.store(blobless_after(&super::super::real::take_build_note(), again), Ordering::Relaxed);
        if !enabled() || cfg!(test) || p.code.load(Ordering::Relaxed) != 0 {
            return;
        }
        let (own, pass) = super::super::real::pso_code(real);
        p.code_pass.store(pass, Ordering::Relaxed);
        p.code.store((own + pass).max(1), Ordering::Relaxed);
    }

    /// A bind (or a lookup for one) got `real` for the stand-in `standin` of `p`: what it
    /// uses (`standin` when `cap` has no room for the code of `real`).
    pub fn used(p: &Pending, standin: usize, real: usize) -> usize {
        if !enabled() {
            return real;
        }
        let code = p.code.load(Ordering::Relaxed);
        if !p.in_use.load(Ordering::Relaxed) {
            if cap().is_some_and(|cap| IN_USE.load(Ordering::Relaxed) + GRAVE_BYTES.load(Ordering::Relaxed) + code > cap) {
                let n = HELD_BACK.fetch_add(1, Ordering::Relaxed);
                if n < 4 || (n + 1).is_power_of_two() {
                    crate::xlog_warn!("shader code lab: no room for the {} KB of a pipeline's code under the cap: its binds get the stand-in ({} so far)", code >> 10, n + 1);
                }
                return standin;
            }
        }
        p.last_frame.store(frame(), Ordering::Relaxed);
        p.last_use.store(now_ms(), Ordering::Relaxed);
        if !p.in_use.swap(true, Ordering::Relaxed) {
            IN_USE.fetch_add(code, Ordering::Relaxed);
            EVER_USED[if p.compute { 0 } else { 1 }].fetch_add(code, Ordering::Relaxed);
        }
        real
    }

    /// `p` is no longer in use.
    fn unused(p: &Pending) {
        if p.in_use.swap(false, Ordering::Relaxed) {
            let code = p.code.load(Ordering::Relaxed);
            let _ = IN_USE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(code)));
        }
    }

    /// The title released the stand-in of `p`.
    pub fn gone(p: &Pending) {
        unused(p);
    }

    /// Lab: the code in use right now, for the log line of a device removal.
    pub fn now() -> Option<String> {
        enabled().then(|| {
            let st = scan(None, 0, false, now_ms(), frame()).0;
            format!(
                "in use {} pipelines / {} KB (compute {} / {} KB, graphics {} / {} KB), waiting for release {} KB, held back by the cap {}",
                st.in_use[0].0 + st.in_use[1].0,
                (st.in_use[0].1 + st.in_use[1].1) >> 10,
                st.in_use[0].0,
                st.in_use[0].1 >> 10,
                st.in_use[1].0,
                st.in_use[1].1 >> 10,
                GRAVE_BYTES.load(Ordering::Relaxed) >> 10,
                HELD_BACK.load(Ordering::Relaxed)
            )
        })
    }

    /// Take the real pipeline from `p` (to release later): `p` is as before its first build.
    fn give_back(p: &Arc<Pending>) -> Option<usize> {
        let again = p.again.clone()?;
        let mut done = p.done.lock().unwrap();
        if !*done {
            return None;
        }
        let real = p.real.swap(0, Ordering::AcqRel);
        if real == 0 {
            return None;
        }
        *p.job.lock().unwrap() = Some(Box::new(move || {
            let _ = super::super::real::take_build_note();
            let t = std::time::Instant::now();
            let r = again();
            let took = t.elapsed();
            REBUILT.fetch_add(1, Ordering::Relaxed);
            REBUILD_US.fetch_add(took.as_micros() as u64, Ordering::Relaxed);
            let thread = std::thread::current();
            let id = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
            if let Some(line) = rebuild_line(took, &super::super::real::build_note(), (id, thread.name())) {
                crate::xlog_warn!("shader code lab: {line}");
            }
            r
        }));
        p.sent.store(false, Ordering::Release);
        p.late.store(false, Ordering::Release);
        p.given.store(true, Ordering::Release);
        unused(p);
        *done = false;
        GIVEN_BACK.fetch_add(1, Ordering::Relaxed);
        if p.blobless.load(Ordering::Relaxed) {
            GIVEN_BACK_BLOBLESS.fetch_add(1, Ordering::Relaxed);
        }
        Some(real)
    }

    #[derive(Default)]
    struct Stats {
        /// (pipelines, bytes) of compute and graphics: built now, in use now, used in the last 10 s
        built: [(usize, usize); 2],
        in_use: [(usize, usize); 2],
        recent: [(usize, usize); 2],
        /// in use, of 1 MB or more
        big: (usize, usize),
        /// in use: code of two-pass compute passes
        pass: usize,
    }

    /// The pipelines to give back so that the code in use is at most `budget`, longest unused
    /// first, and what there is now. `now` / `frame`: `now_ms` / presents.
    fn scan(budget: Option<usize>, idle_ms: u64, graphics: bool, now: u64, frame: u64) -> (Stats, Vec<Arc<Pending>>) {
        let mut st = Stats::default();
        let mut idle: Vec<(u64, Arc<Pending>)> = Vec::new();
        let mut never: Vec<Arc<Pending>> = Vec::new();
        let unused = if budget.is_some() { unused_ms() } else { None };
        {
            let g = PENDING.lock().unwrap();
            for p in g.iter().flat_map(|m| m.values()) {
                if p.real.load(Ordering::Acquire) == 0 {
                    continue;
                }
                let add = |e: &mut (usize, usize), n: usize| *e = (e.0 + 1, e.1 + n);
                let (k, code) = (if p.compute { 0 } else { 1 }, p.code.load(Ordering::Relaxed));
                add(&mut st.built[k], code);
                if !p.in_use.load(Ordering::Relaxed) {
                    let since_built = now.saturating_sub(p.built_ms.load(Ordering::Relaxed));
                    if given_back_unused(unused, since_built, p.again.is_some(), p.blobless.load(Ordering::Relaxed), keep_blobless()) {
                        never.push(p.clone());
                    }
                    continue;
                }
                add(&mut st.in_use[k], code);
                st.pass += p.code_pass.load(Ordering::Relaxed);
                if code >= 1 << 20 {
                    add(&mut st.big, code);
                }
                let last = p.last_use.load(Ordering::Relaxed);
                if now.saturating_sub(last) < 10_000 {
                    add(&mut st.recent[k], code);
                }
                let unused = now.saturating_sub(last) >= idle_ms && frame >= p.last_frame.load(Ordering::Relaxed) + IDLE_FRAMES;
                if unused && p.again.is_some() && (p.compute || graphics) {
                    idle.push((last, p.clone()));
                }
            }
        }
        let mut out = never;
        if let Some(budget) = budget {
            idle.sort_by_key(|e| e.0);
            let mut total = st.in_use[0].1 + st.in_use[1].1;
            for (_, p) in idle {
                if total <= budget {
                    break;
                }
                total = total.saturating_sub(p.code.load(Ordering::Relaxed));
                out.push(p);
            }
        }
        (st, out)
    }

    /// Every 2 s (the activity thread): give back and release what is due; the lab line.
    pub fn tick() -> Option<String> {
        if !enabled() {
            return None;
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        if keep_blobless() {
            ONCE.call_once(|| {
                crate::xlog_info!("shader code lab: XCL_PSO_KEEP_BLOBLESS=1: a pipeline whose build left no driver blob is not given back for XCL_PSO_UNUSED_S (the XCL_PSO_CODE_MB budget still may)");
            });
        }
        let (now, frame) = (now_ms(), frame());
        let (st, idle) = scan(budget(), idle_ms(), graphics_too(), now, frame);
        let mut given = (0usize, 0usize);
        for p in idle {
            let code = p.code.load(Ordering::Relaxed);
            if let Some(real) = give_back(&p) {
                GRAVE_BYTES.fetch_add(code, Ordering::Relaxed);
                GRAVE.lock().unwrap().push((real, now, frame, code));
                given = (given.0 + 1, given.1 + code);
            }
        }
        let all = std::mem::take(&mut *GRAVE.lock().unwrap());
        let (due, waiting): (Vec<_>, Vec<_>) = all.into_iter().partition(|&(_, t, f, _)| now.saturating_sub(t) >= GRAVE_MS && frame >= f + GRAVE_FRAMES);
        let held: usize = waiting.iter().map(|e| e.3).sum();
        GRAVE.lock().unwrap().extend(waiting);
        let released: usize = due.iter().map(|e| e.3).sum();
        for &(real, _, _, code) in &due {
            // (its last reference, if nothing else holds one)
            let left = unsafe {
                let f: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32 = std::mem::transmute(*(*(real as *const *const usize)).add(2));
                f(real as *mut std::ffi::c_void)
            };
            if left != 0 {
                STILL_HELD.fetch_add(1, Ordering::Relaxed);
            }
            let _ = GRAVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(code)));
        }
        let mb = |b: usize| b as f64 / (1 << 20) as f64;
        Some(format!(
            "in use {} pipelines / {:.0} MB (compute {} / {:.0}, graphics {} / {:.0}; of 1 MB or more {} / {:.0}; two-pass compute passes {:.0}), used in the last 10 s {:.0} + {:.0} MB; built {} / {:.0} + {} / {:.0} MB; ever used {:.0} + {:.0} MB; given back now {} / {:.0} MB, released now {} / {:.0} MB, waiting for release {:.0} MB; so far given back {}, built again {} in {} ms, held back by the cap {}, released but still referenced {}, given back without a blob {}",
            st.in_use[0].0 + st.in_use[1].0,
            mb(st.in_use[0].1 + st.in_use[1].1),
            st.in_use[0].0,
            mb(st.in_use[0].1),
            st.in_use[1].0,
            mb(st.in_use[1].1),
            st.big.0,
            mb(st.big.1),
            mb(st.pass),
            mb(st.recent[0].1),
            mb(st.recent[1].1),
            st.built[0].0,
            mb(st.built[0].1),
            st.built[1].0,
            mb(st.built[1].1),
            mb(EVER_USED[0].load(Ordering::Relaxed)),
            mb(EVER_USED[1].load(Ordering::Relaxed)),
            given.0,
            mb(given.1),
            due.len(),
            mb(released),
            mb(held),
            GIVEN_BACK.load(Ordering::Relaxed),
            REBUILT.load(Ordering::Relaxed),
            REBUILD_US.load(Ordering::Relaxed) / 1000,
            HELD_BACK.load(Ordering::Relaxed),
            STILL_HELD.load(Ordering::Relaxed),
            GIVEN_BACK_BLOBLESS.load(Ordering::Relaxed)
        ))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A compute pipeline nothing used for a while is taken from its stand-in when the code
        /// in use is over the budget – the longest unused one first, and no more than needed –
        /// and its next bind builds it again; one without a build that can run again is kept.
        #[test]
        fn an_unused_pipeline_is_given_back_and_built_again_at_its_next_bind() {
            let (old, new, once) = (0x7E57_C001usize, 0x7E57_C002usize, 0x7E57_C003usize);
            let builds = Arc::new(AtomicUsize::new(0));
            let n = builds.clone();
            submit_lazy_again(old, true, Arc::new(move || 0xA000 + n.fetch_add(1, Ordering::SeqCst)), 0);
            submit_lazy_again(new, true, Arc::new(|| 0xB000), 0);
            submit_lazy(once, true, Box::new(|| 0xC000), 0);
            assert_eq!(resolve_wait_for(old, None), 0xA000);
            assert_eq!(resolve_wait_for(new, None), 0xB000);
            assert_eq!(resolve_wait_for(once, None), 0xC000);
            // (the test build measures nothing and counts no use: set here)
            for (pso, last) in [(old, 10u64), (new, 20), (once, 5)] {
                let p = entry(pso).unwrap();
                p.code.store(300, Ordering::Relaxed);
                p.in_use.store(true, Ordering::Relaxed);
                p.last_use.store(last, Ordering::Relaxed);
                p.last_frame.store(1, Ordering::Relaxed);
            }
            let ours = |v: Vec<Arc<Pending>>| -> Vec<usize> { v.iter().map(|p| p.real.load(Ordering::Acquire)).filter(|r| [0xA000, 0xB000, 0xC000].contains(r)).collect() };
            // used too recently (time, then presents): nothing
            assert!(ours(scan(Some(0), 1000, false, 900, 100).1).is_empty());
            assert!(ours(scan(Some(0), 1000, false, 5000, IDLE_FRAMES).1).is_empty());
            // over the budget by one pipeline (other tests' stand-ins have no code): the oldest
            let total: usize = {
                let s = scan(None, 0, false, 5000, 100).0;
                s.in_use[0].1 + s.in_use[1].1
            };
            assert_eq!(ours(scan(Some(total - 1), 1000, false, 5000, 100).1), [0xA000]);
            assert!(ours(scan(Some(total), 1000, false, 5000, 100).1).is_empty());
            // all of it: both that can be built again, oldest first; never the other one
            assert_eq!(ours(scan(Some(0), 1000, false, 5000, 100).1), [0xA000, 0xB000]);
            let p = entry(old).unwrap();
            assert_eq!(give_back(&p), Some(0xA000));
            assert!(is_pending(old) && give_back(&p).is_none(), "as before its first build");
            assert!(give_back(&entry(once).unwrap()).is_none());
            assert_eq!(resolve_wait_for(old, None), 0xA001, "built again at the next bind");
            assert_eq!(builds.load(Ordering::SeqCst), 2);
            assert!(!is_pending(old));
            for pso in [old, new, once] {
                let _ = released(pso);
            }
        }

        /// `XCL_PSO_UNUSED_S` gives back a pipeline that can be built again once its delay is
        /// over; with `XCL_PSO_KEEP_BLOBLESS=1` not one whose build left no driver blob.
        #[test]
        fn the_unused_delay_keeps_a_pipeline_without_a_blob_only_with_the_switch() {
            // (unused, since built, can be built again, without a blob, switch) -> given back
            let cases = [
                (None, 99_000, true, false, false, false),
                (Some(30_000), 29_999, true, false, false, false),
                (Some(30_000), 30_000, true, false, false, true),
                (Some(30_000), 30_000, false, false, false, false),
                (Some(30_000), 30_000, true, true, false, true),
                (Some(30_000), 30_000, true, true, true, false),
                (Some(30_000), 30_000, true, false, true, true),
            ];
            for (unused, since, again, blobless, keep, want) in cases {
                assert_eq!(given_back_unused(unused, since, again, blobless, keep), want, "{unused:?} {since} {again} {blobless} {keep}");
            }
            // the switch is off unless set to 1
            assert_eq!(keep_blobless(), std::env::var("XCL_PSO_KEEP_BLOBLESS").ok().and_then(|v| v.trim().parse::<u64>().ok()) == Some(1));
        }

        /// A pipeline counts as without a blob when its build said so, or when its build again
        /// took 1 s or more in the driver without a blob; a quick one or one with a blob does not.
        #[test]
        fn a_pipeline_is_without_a_blob_when_its_build_says_so_or_a_slow_build_again_had_none() {
            use super::super::super::real::BuildNote;
            let slow = |blob: bool, ms: u64| BuildNote { slowest: Some(("MS".to_string(), 0xAB, std::time::Duration::from_millis(ms), blob)), blobless: false };
            assert!(!blobless_after(&BuildNote::default(), false));
            assert!(!blobless_after(&BuildNote::default(), true));
            assert!(blobless_after(&BuildNote { slowest: None, blobless: true }, false));
            assert!(!blobless_after(&slow(false, 5000), false), "a first build is compiled without a blob");
            assert!(blobless_after(&slow(false, 5000), true));
            assert!(!blobless_after(&slow(true, 5000), true));
            assert!(!blobless_after(&slow(false, 999), true));
        }

        /// The `built again at bind:` line: only from 1 s, with the slowest compile's kind,
        /// record, the time, the thread and whether a blob was supplied.
        #[test]
        fn a_slow_build_again_is_logged_with_its_compile_thread_and_blob() {
            use super::super::super::real::BuildNote;
            let note = BuildNote { slowest: Some(("MS without topology".to_string(), 0x1234_5678, std::time::Duration::from_millis(4199), false)), blobless: true };
            assert_eq!(rebuild_line(std::time::Duration::from_millis(999), &note, (7, Some("t"))), None);
            assert_eq!(
                rebuild_line(std::time::Duration::from_millis(4210), &note, (25412, None)).as_deref(),
                Some("built again at bind: MS without topology, record 0000000012345678, 4210 ms (slowest driver compile 4199 ms), thread 25412 (unnamed), blob no")
            );
            let with = BuildNote { slowest: Some(("MS".to_string(), 1, std::time::Duration::from_millis(1500), true)), blobless: false };
            assert!(rebuild_line(std::time::Duration::from_secs(2), &with, (3, Some("xcl-pso-0"))).unwrap().ends_with("thread 3 (xcl-pso-0), blob yes"));
            assert!(rebuild_line(std::time::Duration::from_secs(1), &BuildNote::default(), (3, None)).unwrap().contains("no driver compile on this thread, 1000 ms"));
        }

        /// A pipeline given back while its build left no blob is counted; its build again (no
        /// driver compile here) clears the mark.
        #[test]
        fn a_pipeline_given_back_without_a_blob_is_counted() {
            let pso = 0x7E57_C0B1usize;
            submit_lazy_again(pso, false, Arc::new(|| 0xB10B), 0);
            assert_eq!(resolve_wait_for(pso, None), 0xB10B);
            let p = entry(pso).unwrap();
            assert!(!p.blobless.load(Ordering::Relaxed), "no driver compile: not without a blob");
            p.blobless.store(true, Ordering::Relaxed);
            let before = GIVEN_BACK_BLOBLESS.load(Ordering::Relaxed);
            assert_eq!(give_back(&p), Some(0xB10B));
            assert_eq!(GIVEN_BACK_BLOBLESS.load(Ordering::Relaxed), before + 1);
            assert!(p.given.load(Ordering::Acquire));
            assert_eq!(resolve_wait_for(pso, None), 0xB10B, "built again at the next bind");
            assert!(!p.given.load(Ordering::Acquire) && !p.blobless.load(Ordering::Relaxed));
            let _ = released(pso);
        }
    }
}

/// Lab: the `shader code lab:` line of `code` (None: off).
pub fn code_tick() -> Option<String> {
    code::tick()
}

/// Lab: the code in use now (None: off), for the log of a device removal.
pub fn code_now() -> Option<String> {
    code::now()
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

impl Pending {
    /// What a bind of the stand-in `pso` gives the driver while the pipeline is not built.
    fn stand(&self, pso: usize) -> usize {
        match self.driver.load(Ordering::Acquire) {
            0 => pso,
            d => d,
        }
    }
}

/// Stand-in `standin` is no driver object (`standin::shell`): binds get `driver` in its place.
/// Told before the title has the stand-in.
pub fn shared(standin: usize, driver: usize) {
    if let Some(p) = entry(standin) {
        p.driver.store(driver, Ordering::Release);
    }
}

/// The pipeline to use for `pso` now: the real one once built, else the stand-in's driver
/// pipeline (`pso` itself when it is one).
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            let stand = p.stand(pso);
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p);
            }
            stand
        }
        r => code::used(&p, pso, r),
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget())
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    if p.spared.load(Ordering::Acquire) {
        return resolve_spared(&p, pso);
    }
    // (the title's thread is held from here until the pipeline is built or the budget is over)
    let _held = (!*p.done.lock().unwrap()).then(Held::new);
    if let Some(budget) = budget {
        return resolve_within(&p, pso, budget);
    }
    // (twice: a pipeline given back to the driver between its build and here is built again)
    for _ in 0..2 {
        let job = p.job.lock().unwrap().take();
        if let Some(job) = job {
            complete(&p, job());
        }
        p.want();
        let mut done = p.done.lock().unwrap();
        while !*done {
            done = p.cv.wait(done).unwrap();
        }
        drop(done);
        match p.real.load(Ordering::Acquire) {
            0 if p.job.lock().unwrap().is_some() => continue,
            0 => return p.stand(pso),
            r => return code::used(&p, pso, r),
        }
    }
    pso
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => code::used(p, pso, r),
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        return p.stand(pso);
    }
    p.want();
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
    if *done {
        return built(p);
    }
    drop(done);
    p.late.store(true, Ordering::Release);
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    p.stand(pso)
}

/// `XCL_PSO_IMAGE_SKIP=1`: a bind does not wait for a big compute program that writes nothing
/// but images (`gpu::effects::big_and_image_only`); its dispatches are left out until a worker
/// has built it. Off by default: what such a program draws is missing for that long.
///
/// A retail title's world start binds one ray-tracing resolve program per material (73 – 118
/// KiB of console code, about 1 MB of DXIL each): an RDNA2 driver takes 22 – 172 s for each,
/// one bind at a time, and the title stands still for all of them. `XCL_PSO_WAIT_MS` goes on without every late
/// program, also the culling passes whose counts an `ExecuteIndirect` reads next – a GPU hang.
pub fn image_skip() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("XCL_PSO_IMAGE_SKIP").is_ok_and(|v| v.trim() == "1"))
}

/// The compute pipeline behind stand-in `standin` writes nothing but images: binds go on
/// without it while it is built (`resolve_spared`). Told before the title has the stand-in.
pub fn image_only(standin: usize) {
    if let Some(p) = entry(standin) {
        p.spared.store(true, Ordering::Release);
    }
}

/// `resolve_wait` for a pipeline binds do not wait for (`image_only`): the real one once built,
/// else the stand-in, with the build sent to the workers. Nobody waits, so the build is not
/// `wanted`: a big one takes its turn among the driver's compiles and waits for free commit.
fn resolve_spared(p: &Arc<Pending>, pso: usize) -> usize {
    match p.real.load(Ordering::Acquire) {
        0 => {
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p.clone());
            }
            if !*p.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: a compute program that writes only images is still being built: its dispatches are left out until it is ({} so far)", n + 1);
                }
            }
            p.stand(pso)
        }
        r => r,
    }
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        // (lab: given back to the driver – built again here from the driver's blob, not by a
        // worker frames later)
        Some(p) if p.given.load(Ordering::Acquire) => resolve_wait_for(pso, None),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    code::gone(&p);
    {
        let done = p.done.lock().unwrap();
        if *done {
            return p.real.swap(0, Ordering::AcqRel);
        }
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
    }
    drop_unstarted_job(&p);
    0
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// One of the title's threads is about to wait for `pso`.
pub fn want(pso: usize) {
    if let Some(p) = entry(pso) {
        p.want();
    }
}

/// A call with `pso` was put off to the submit: its build goes before the workers' queue.
pub fn need_by_submit(pso: usize) {
    let Some(p) = entry(pso) else { return };
    if !std::mem::replace(&mut p.demand.lock().unwrap().by_submit, true) {
        NEEDED.lock().unwrap().push_back(p);
        let _ = queue().lock().unwrap().send(Box::new(|| ()));
    }
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}
