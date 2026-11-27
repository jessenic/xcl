//! lab: `XCL_THREAD_SAMPLE=<from s>,<period ms>,<count>` (+ `XCL_THREAD_SAMPLE_FILE=<path>`) –
//! where the CPU time of every thread goes, without a debugger, ETW or elevation. Off unless set.
//!
//! From log time `<from>`, every `<period>` ms (`<count>` times) a thread of its own takes, for
//! every other thread of the process: its cycle time since the previous sample
//! (`QueryThreadCycleTime`: the CPU it used, exact), and – when it ran since then – its registers
//! and the top of its stack, with the thread stopped only for that (`SuspendThread`,
//! `GetThreadContext`, a copy of at most 16 KiB of stack, `ResumeThread`). The frames are unwound
//! afterwards, thread running again, from that copy (`RtlVirtualUnwind` with the stack registers
//! moved into the copy, a step leaving the copy ends the walk): RIP and up to 7 return addresses.
//! A thread whose cycle time did not move and whose last RIP was in ntdll / win32u (a wait) is not
//! stopped: it did not run, so its stack is the one of its last sample (written as "unchanged").
//!
//! Everything goes to a binary file (`XCL_THREAD_SAMPLE_FILE`, else `xcl-thread-sample.bin` in the
//! temp dir), not the log; S/tools/thrprof.py reads it (modules, xcl.exe's PDB, waits). Format,
//! little-endian (`Out` writes it, `read` in the tests reads it back):
//!   header  `XCLTSMP1`, u32 version, u32 period ms, f64 from, u32 count, u32 pid, f64 uptime,
//!           u64 tsc, i64 qpc, i64 qpc frequency
//!   'M'     u64 base, u64 size, u8 kind (0 other, 1 XCL's own exe, 2 an image XCL mapped), u16 n, name
//!   'T'     u32 tid, u64 start address, u16 n, name (when first seen, and when the name changes)
//!   'S'     u32 index, f64 uptime, u64 tsc, u16 entries; each: u32 tid, u8 state (0 stopped and
//!           read, 1 unchanged, 2 failed), u8 frames, u64 cycles since its previous entry, frames x u64
//!   'E'     u32 samples, u32 threads, u64 stops, u64 unchanged, u64 failed, u64 sampler tsc,
//!           u64 longest stop tsc, u64 stops tsc, u64 late, u64 tsc, i64 qpc, f64 uptime
//! The log gets one line at the start and one with the counts at the end.

use std::collections::HashMap;
use std::ffi::c_void;
use std::io::Write;

pub(crate) const MAGIC: &[u8; 8] = b"XCLTSMP1";
pub(crate) const VERSION: u32 = 1;
/// RIP + return addresses kept a sample
const FRAMES: usize = 8;
/// stack copied while a thread stands
const COPY_BYTES: usize = 16 << 10;
/// a thread is stopped at least once every this many samples, unchanged or not
const MAX_UNCHANGED: u32 = 20;
/// cycles a waiting thread may use between two samples and still count as not having run: stopping it
/// (SuspendThread runs a kernel APC on it, ResumeThread wakes it) charges it some cycles of its own, so
/// an exact `== 0` would stop every waiter at every sample
const WAKE_NOISE_CYCLES: u64 = 1_000_000;
/// samples between two looks at the thread list / at the thread names / at the modules
const REFRESH_THREADS: u32 = 100;
const REFRESH_NAMES: u32 = 200;
const REFRESH_MODULES: u32 = 2000;

pub(crate) const STOPPED: u8 = 0;
pub(crate) const UNCHANGED: u8 = 1;
pub(crate) const FAILED: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Config {
    pub from: f64,
    pub period_ms: u32,
    pub count: u32,
}

/// `<from s>,<period ms>,<count>`: all three, period 1..=1000 ms, count >= 1.
pub(crate) fn parse(v: &str) -> Option<Config> {
    let f: Vec<&str> = v.split(',').map(str::trim).collect();
    if f.len() != 3 {
        return None;
    }
    let from = f[0].parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0)?;
    let period_ms = f[1].parse::<u32>().ok().filter(|p| (1..=1000).contains(p))?;
    let count = f[2].parse::<u32>().ok().filter(|c| *c >= 1)?;
    Some(Config { from, period_ms, count })
}

/// The binary file's writer (any `Write`: a file in the title, a Vec in the tests).
pub(crate) struct Out<W: Write> {
    w: W,
    pub ok: bool,
}

impl<W: Write> Out<W> {
    pub fn new(w: W) -> Self {
        Out { w, ok: true }
    }
    fn put(&mut self, b: &[u8]) {
        if self.ok && self.w.write_all(b).is_err() {
            self.ok = false;
        }
    }
    fn name(&mut self, s: &str) {
        let b = s.as_bytes();
        let n = b.len().min(u16::MAX as usize);
        self.put(&(n as u16).to_le_bytes());
        self.put(&b[..n]);
    }
    #[allow(clippy::too_many_arguments)]
    pub fn header(&mut self, c: &Config, pid: u32, uptime: f64, tsc: u64, qpc: i64, qpf: i64) {
        self.put(MAGIC);
        self.put(&VERSION.to_le_bytes());
        self.put(&c.period_ms.to_le_bytes());
        self.put(&c.from.to_le_bytes());
        self.put(&c.count.to_le_bytes());
        self.put(&pid.to_le_bytes());
        self.put(&uptime.to_le_bytes());
        self.put(&tsc.to_le_bytes());
        self.put(&qpc.to_le_bytes());
        self.put(&qpf.to_le_bytes());
    }
    pub fn module(&mut self, base: u64, size: u64, kind: u8, name: &str) {
        self.put(b"M");
        self.put(&base.to_le_bytes());
        self.put(&size.to_le_bytes());
        self.put(&[kind]);
        self.name(name);
    }
    pub fn thread(&mut self, tid: u32, start: u64, name: &str) {
        self.put(b"T");
        self.put(&tid.to_le_bytes());
        self.put(&start.to_le_bytes());
        self.name(name);
    }
    pub fn sample(&mut self, index: u32, uptime: f64, tsc: u64, entries: u16) {
        self.put(b"S");
        self.put(&index.to_le_bytes());
        self.put(&uptime.to_le_bytes());
        self.put(&tsc.to_le_bytes());
        self.put(&entries.to_le_bytes());
    }
    pub fn entry(&mut self, tid: u32, state: u8, cycles: u64, frames: &[u64]) {
        let n = frames.len().min(u8::MAX as usize);
        self.put(&tid.to_le_bytes());
        self.put(&[state, n as u8]);
        self.put(&cycles.to_le_bytes());
        for f in &frames[..n] {
            self.put(&f.to_le_bytes());
        }
    }
    pub fn end(&mut self, s: &Stats, tsc: u64, qpc: i64, uptime: f64) {
        self.put(b"E");
        self.put(&s.samples.to_le_bytes());
        self.put(&s.threads.to_le_bytes());
        for v in [s.stops, s.unchanged, s.failed, s.sampler_tsc, s.longest_stop_tsc, s.stops_tsc, s.late, tsc] {
            self.put(&v.to_le_bytes());
        }
        self.put(&qpc.to_le_bytes());
        self.put(&uptime.to_le_bytes());
    }
    pub fn flush(&mut self) {
        if self.ok && self.w.flush().is_err() {
            self.ok = false;
        }
    }
}

/// The counts of a sampling.
#[derive(Default, Clone, Debug, PartialEq)]
pub(crate) struct Stats {
    pub samples: u32,
    pub threads: u32,
    pub stops: u64,
    pub unchanged: u64,
    pub failed: u64,
    pub sampler_tsc: u64,
    pub longest_stop_tsc: u64,
    pub stops_tsc: u64,
    pub late: u64,
}

/// x64 CONTEXT's 16 integer registers, in register-number order (Rax, Rcx, Rdx, Rbx, Rsp, Rbp,
/// Rsi, Rdi, R8..R15: the CONTEXT keeps them in that order, one u64 after the other).
fn regs(c: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT) -> *mut u64 {
    std::ptr::addr_of_mut!(c.Rax)
}

/// Registers pointing into the original stack span [lo, lo + len) point into the copy at `copy`.
unsafe fn rebase(c: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT, lo: u64, len: u64, copy: u64) {
    let r = regs(c);
    for i in 0..16 {
        let v = *r.add(i);
        if v >= lo && v < lo + len {
            *r.add(i) = v - lo + copy;
        }
    }
}

/// RIP and the return addresses of up to `max - 1` frames, unwound from `c` whose stack
/// registers were moved into the copy [copy, copy + len). A step that would read outside the copy
/// ends the walk (`lo`: the original address of the copy's first byte).
unsafe fn walk(c: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT, lo: u64, copy: u64, len: u64, max: usize, out: &mut Vec<u64>) {
    use windows_sys::Win32::System::Diagnostics::Debug::{RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER};
    let inside = |v: u64, bytes: u64| v >= copy && v + bytes <= copy + len;
    while out.len() < max {
        let rip = c.Rip;
        if rip < 0x10000 {
            break;
        }
        out.push(rip);
        if out.len() == max || !inside(c.Rsp, 8) {
            break;
        }
        let mut base = 0u64;
        let f = RtlLookupFunctionEntry(rip, &mut base, std::ptr::null_mut());
        if f.is_null() {
            // no unwind data (a leaf, or code made at run time): the return address at RSP
            c.Rip = (c.Rsp as *const u64).read_unaligned();
            c.Rsp += 8;
            continue;
        }
        // a frame register (UNWIND_INFO byte 3, low nibble) must point into the copy too (an odd
        // address is an entry that points at another one: not read)
        let ua = (*f).Anonymous.UnwindInfoAddress;
        if ua & 1 == 0 {
            let fr = (*((base + ua as u64) as *const u8).add(3) & 0xF) as usize;
            if fr != 0 && !inside(*regs(c).add(fr), 8) {
                break;
            }
        }
        let mut data = std::ptr::null_mut();
        let mut frame = 0u64;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, rip, f, c, &mut data, &mut frame, std::ptr::null_mut());
        // registers restored from the copy hold original stack addresses
        rebase(c, lo, len, copy);
    }
}

#[link(name = "kernel32")]
extern "system" {
    fn QueryThreadCycleTime(thread: *mut c_void, cycles: *mut u64) -> i32;
}

/// [base, base + SizeOfImage) of a loaded module (0, 0: none).
unsafe fn image_span(base: usize) -> (u64, u64) {
    if base == 0 {
        return (0, 0);
    }
    let pe = base + (base as *const u8).add(0x3C).cast::<u32>().read_unaligned() as usize;
    // PE signature (4) + file header (20) + SizeOfImage's offset in the optional header (56)
    let size = (pe as *const u8).add(0x18 + 56).cast::<u32>().read_unaligned();
    (base as u64, size as u64)
}

unsafe fn module_span(name: &str) -> (u64, u64) {
    let w: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
    image_span(windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(w.as_ptr()) as usize)
}

fn tsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn qpc() -> i64 {
    let mut v = 0i64;
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut v) };
    v
}

struct Thread {
    h: *mut c_void,
    name: String,
    cycles: u64,
    /// the last stop's RIP was in a wait (ntdll / win32u)
    waiting: bool,
    unchanged_in_row: u32,
    alive: bool,
}

/// The modules: the loader's (toolhelp) and the images XCL mapped; each base once.
unsafe fn write_modules<W: Write>(out: &mut Out<W>, seen: &mut std::collections::HashSet<u64>) {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Module32FirstW, Module32NextW, MODULEENTRY32W, TH32CS_SNAPMODULE};
    let own = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null()) as u64;
    let snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    if snap != INVALID_HANDLE_VALUE {
        let mut me: MODULEENTRY32W = std::mem::zeroed();
        me.dwSize = std::mem::size_of::<MODULEENTRY32W>() as u32;
        let mut ok = Module32FirstW(snap, &mut me) != 0;
        while ok {
            let base = me.modBaseAddr as u64;
            if seen.insert(base) {
                // the full path: S/tools/thrprof.py reads the DLLs' export tables from it
                let n = me.szExePath.iter().position(|&c| c == 0).unwrap_or(me.szExePath.len());
                out.module(base, me.modBaseSize as u64, u8::from(base == own), &String::from_utf16_lossy(&me.szExePath[..n]));
            }
            ok = Module32NextW(snap, &mut me) != 0;
        }
        CloseHandle(snap);
    }
    for m in crate::modules::mapped_modules() {
        if seen.insert(m.base as u64) {
            out.module(m.base as u64, m.size as u64, 2, &m.name);
        }
    }
}

/// The thread's start address (NtQueryInformationThread, ThreadQuerySetWin32StartAddress).
unsafe fn start_address(h: *mut c_void) -> u64 {
    type F = unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32, *mut u32) -> i32;
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let f = *F.get_or_init(|| {
        let ntdll: Vec<u16> = "ntdll.dll\0".encode_utf16().collect();
        let m = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(ntdll.as_ptr());
        windows_sys::Win32::System::LibraryLoader::GetProcAddress(m, c"NtQueryInformationThread".as_ptr().cast()).map_or(0, |p| p as usize)
    });
    if f == 0 {
        return 0;
    }
    let mut a = 0u64;
    let st = std::mem::transmute::<usize, F>(f)(h, 9, (&mut a as *mut u64).cast(), 8, std::ptr::null_mut());
    if st >= 0 {
        a
    } else {
        0
    }
}

/// The ids of the other threads of this process.
unsafe fn thread_ids(me: u32) -> Vec<u32> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
    let pid = windows_sys::Win32::System::Threading::GetCurrentProcessId();
    let mut out = Vec::new();
    let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if snap == INVALID_HANDLE_VALUE {
        return out;
    }
    let mut te: THREADENTRY32 = std::mem::zeroed();
    te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
    let mut ok = Thread32First(snap, &mut te) != 0;
    while ok {
        if te.th32OwnerProcessID == pid && te.th32ThreadID != me {
            out.push(te.th32ThreadID);
        }
        ok = Thread32Next(snap, &mut te) != 0;
    }
    CloseHandle(snap);
    out
}

/// Run the sampling of `c` on this thread into `out` (from log time `c.from`); the counts.
pub(crate) fn run<W: Write>(c: Config, out: &mut Out<W>) -> Stats {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64};
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcessId, GetCurrentThreadId, GetExitCodeThread, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
    };
    let left = c.from - crate::log::uptime();
    if left > 0.0 {
        std::thread::sleep(std::time::Duration::from_secs_f64(left));
    }
    let mut st = Stats::default();
    unsafe {
        let me = GetCurrentThreadId();
        let mut qpf = 0i64;
        windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut qpf);
        out.header(&c, GetCurrentProcessId(), crate::log::uptime(), tsc(), qpc(), qpf);
        let mut modules = std::collections::HashSet::new();
        write_modules(out, &mut modules);
        let waits = [module_span("ntdll.dll"), module_span("win32u.dll")];
        let in_wait = |rip: u64| waits.iter().any(|&(b, n)| rip >= b && rip < b + n);
        let mut threads: HashMap<u32, Thread> = HashMap::new();
        let mut ids: Vec<u32> = Vec::new();
        let mut copy = vec![0u64; COPY_BYTES / 8];
        let mut frames: Vec<u64> = Vec::with_capacity(FRAMES);
        let mut rows: Vec<(u32, u8, u64, [u64; FRAMES], u8)> = Vec::new();
        let period = std::time::Duration::from_millis(c.period_ms as u64);
        let mut t0 = std::time::Instant::now();
        for i in 0..c.count {
            // on time: the i-th sample at t0 + i periods; more than a period late, from now on
            let due = t0 + period * i;
            let now = std::time::Instant::now();
            if now < due {
                std::thread::sleep(due - now);
            } else if now - due > period {
                st.late += 1;
                t0 = now - period * i;
            }
            let pass = tsc();
            if i % REFRESH_THREADS == 0 {
                ids = thread_ids(me);
                for &tid in &ids {
                    if threads.contains_key(&tid) {
                        continue;
                    }
                    let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, tid);
                    if h.is_null() {
                        continue;
                    }
                    let name = crate::eh::thread_name(h);
                    out.thread(tid, start_address(h), &name);
                    let mut cy = 0u64;
                    QueryThreadCycleTime(h, &mut cy);
                    threads.insert(tid, Thread { h, name, cycles: cy, waiting: false, unchanged_in_row: 0, alive: true });
                }
                st.threads = threads.len() as u32;
            }
            if i % REFRESH_NAMES == REFRESH_NAMES - 1 {
                for (&tid, t) in threads.iter_mut().filter(|t| t.1.alive) {
                    let name = crate::eh::thread_name(t.h);
                    if name != t.name {
                        out.thread(tid, start_address(t.h), &name);
                        t.name = name;
                    }
                }
            }
            if i % REFRESH_MODULES == REFRESH_MODULES - 1 {
                write_modules(out, &mut modules);
            }
            rows.clear();
            for &tid in &ids {
                let Some(t) = threads.get_mut(&tid) else { continue };
                if !t.alive {
                    continue;
                }
                let mut cy = 0u64;
                let have_cy = QueryThreadCycleTime(t.h, &mut cy) != 0;
                let ran = cy.wrapping_sub(t.cycles);
                if have_cy && ran < WAKE_NOISE_CYCLES && t.waiting && t.unchanged_in_row < MAX_UNCHANGED {
                    t.unchanged_in_row += 1;
                    st.unchanged += 1;
                    t.cycles = cy;
                    rows.push((tid, UNCHANGED, ran, [0; FRAMES], 0));
                    continue;
                }
                #[repr(align(16))]
                struct Aligned(CONTEXT);
                let mut ctx: Aligned = std::mem::zeroed();
                ctx.0.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
                let (mut lo, mut len) = (0u64, 0u64);
                // stopped only for its registers and the copy of its stack top: no lock, no heap
                let s0 = tsc();
                let got = SuspendThread(t.h) != u32::MAX && {
                    let ok = GetThreadContext(t.h, &mut ctx.0) != 0;
                    if ok {
                        let sp = ctx.0.Rsp & !7;
                        let mut q: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                        if sp != 0
                            && VirtualQuery(sp as *const c_void, &mut q, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) != 0
                            && q.State == MEM_COMMIT
                            && q.Protect & 0x104 == 0x4
                        {
                            let end = q.BaseAddress as u64 + q.RegionSize as u64;
                            len = (end - sp).min(COPY_BYTES as u64);
                            lo = sp;
                            std::ptr::copy_nonoverlapping(sp as *const u8, copy.as_mut_ptr().cast::<u8>(), len as usize);
                        }
                    }
                    ResumeThread(t.h);
                    ok
                };
                let stop = tsc().wrapping_sub(s0);
                if !got {
                    st.failed += 1;
                    let mut code = 0u32;
                    if GetExitCodeThread(t.h, &mut code) != 0 && code != 259 {
                        t.alive = false;
                    }
                    rows.push((tid, FAILED, if have_cy { ran } else { 0 }, [0; FRAMES], 0));
                    continue;
                }
                st.stops += 1;
                st.stops_tsc += stop;
                st.longest_stop_tsc = st.longest_stop_tsc.max(stop);
                frames.clear();
                let rip = ctx.0.Rip;
                if len >= 8 {
                    let base = copy.as_ptr() as u64;
                    rebase(&mut ctx.0, lo, len, base);
                    walk(&mut ctx.0, lo, base, len, FRAMES, &mut frames);
                } else {
                    frames.push(rip);
                }
                t.waiting = in_wait(rip);
                t.unchanged_in_row = 0;
                if have_cy {
                    t.cycles = cy;
                }
                let mut f = [0u64; FRAMES];
                f[..frames.len()].copy_from_slice(&frames);
                rows.push((tid, STOPPED, if have_cy { ran } else { 0 }, f, frames.len() as u8));
            }
            out.sample(i, crate::log::uptime(), pass, rows.len() as u16);
            for (tid, state, cy, f, n) in &rows {
                out.entry(*tid, *state, *cy, &f[..*n as usize]);
            }
            st.samples += 1;
            st.sampler_tsc += tsc().wrapping_sub(pass);
            if i % 200 == 199 {
                out.flush();
            }
            if !out.ok {
                break;
            }
        }
        write_modules(out, &mut modules);
        out.end(&st, tsc(), qpc(), crate::log::uptime());
        out.flush();
        for t in threads.values() {
            CloseHandle(t.h);
        }
    }
    st
}

/// `XCL_THREAD_SAMPLE` set: the sampler's thread.
pub fn start_from_env() {
    let Ok(v) = std::env::var("XCL_THREAD_SAMPLE") else { return };
    let Some(c) = parse(&v) else {
        crate::xlog_warn!("thread sample: XCL_THREAD_SAMPLE={v} is not <from s>,<period ms>,<count>: off");
        return;
    };
    let file = std::env::var_os("XCL_THREAD_SAMPLE_FILE").map_or_else(|| std::env::temp_dir().join("xcl-thread-sample.bin"), std::path::PathBuf::from);
    let _ = std::thread::Builder::new().name("xcl-thread-sample".into()).spawn(move || {
        crate::xlog_info!("thread sample: from {} s every {} ms, {} samples, to {}", c.from, c.period_ms, c.count, file.display());
        let f = match std::fs::File::create(&file) {
            Ok(f) => f,
            Err(e) => {
                crate::xlog_warn!("thread sample: cannot create {}: {e}", file.display());
                return;
            }
        };
        unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST};
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        }
        let mut out = Out::new(std::io::BufWriter::with_capacity(1 << 20, f));
        let t0 = (crate::log::uptime(), tsc());
        let s = run(c, &mut out);
        let t1 = (crate::log::uptime(), tsc());
        let hz = (t1.1 - t0.1) as f64 / (t1.0 - t0.0).max(1e-9);
        let us = |t: u64| t as f64 * 1e6 / hz.max(1.0);
        crate::xlog_info!(
            "thread sample: done: {} samples, {} threads, {} stops (mean {:.1} us, longest {:.0} us), {} unchanged, {} failed, {} late; sampler {:.0} us a sample; {} ({})",
            s.samples,
            s.threads,
            s.stops,
            us(s.stops_tsc) / s.stops.max(1) as f64,
            us(s.longest_stop_tsc),
            s.unchanged,
            s.failed,
            s.late,
            us(s.sampler_tsc) / s.samples.max(1) as f64,
            file.display(),
            if out.ok { "ok" } else { "WRITE FAILED" }
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the file holds, read back (the reader S/tools/thrprof.py has, in Rust).
    #[derive(Debug, Default)]
    struct File {
        period_ms: u32,
        count: u32,
        modules: Vec<(u64, u64, u8, String)>,
        threads: Vec<(u32, u64, String)>,
        samples: Vec<(u32, f64, u64, Vec<(u32, u8, u64, Vec<u64>)>)>,
        end: Option<(u32, u32, Vec<u64>)>,
    }

    struct Reader<'a> {
        b: &'a [u8],
        p: usize,
    }

    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let s = self.b.get(self.p..self.p + n)?;
            self.p += n;
            Some(s)
        }
    }

    fn read(b: &[u8]) -> Option<File> {
        let mut r = Reader { b, p: 0 };
        let mut take = |n: usize| r.take(n);
        macro_rules! u {
            ($t:ty) => {
                <$t>::from_le_bytes(take(std::mem::size_of::<$t>())?.try_into().ok()?)
            };
        }
        if take(8)? != MAGIC || u!(u32) != VERSION {
            return None;
        }
        let mut f = File { period_ms: u!(u32), ..Default::default() };
        let _from = u!(f64);
        f.count = u!(u32);
        let (_pid, _up, _tsc, _qpc, _qpf) = (u!(u32), u!(f64), u!(u64), u!(i64), u!(i64));
        loop {
            let Some(tag) = take(1) else { break };
            match tag[0] {
                b'M' => {
                    let (base, size, kind) = (u!(u64), u!(u64), take(1)?[0]);
                    let n = u!(u16) as usize;
                    f.modules.push((base, size, kind, String::from_utf8_lossy(take(n)?).into_owned()));
                }
                b'T' => {
                    let (tid, start) = (u!(u32), u!(u64));
                    let n = u!(u16) as usize;
                    f.threads.push((tid, start, String::from_utf8_lossy(take(n)?).into_owned()));
                }
                b'S' => {
                    let (i, up, t, n) = (u!(u32), u!(f64), u!(u64), u!(u16));
                    let mut es = Vec::new();
                    for _ in 0..n {
                        let tid = u!(u32);
                        let (state, nf) = (take(1)?[0], take(1)?[0]);
                        let cy = u!(u64);
                        let fr = (0..nf).map(|_| Some(u!(u64))).collect::<Option<Vec<u64>>>()?;
                        es.push((tid, state, cy, fr));
                    }
                    f.samples.push((i, up, t, es));
                }
                b'E' => {
                    let (s, t) = (u!(u32), u!(u32));
                    let v = (0..8).map(|_| Some(u!(u64))).collect::<Option<Vec<u64>>>()?;
                    let (_q, _u) = (u!(i64), u!(f64));
                    f.end = Some((s, t, v));
                }
                _ => return None,
            }
        }
        Some(f)
    }

    #[test]
    fn the_switch_takes_a_start_a_period_and_a_count() {
        assert_eq!(parse("360,5,12000"), Some(Config { from: 360.0, period_ms: 5, count: 12000 }));
        assert_eq!(parse(" 0.5 , 1 , 1 "), Some(Config { from: 0.5, period_ms: 1, count: 1 }));
        assert_eq!(parse("360,5"), None);
        assert_eq!(parse("360,5,12000,1"), None);
        assert_eq!(parse("360,0,10"), None);
        assert_eq!(parse("360,1001,10"), None);
        assert_eq!(parse("360,5,0"), None);
        assert_eq!(parse("-1,5,10"), None);
        assert_eq!(parse("x,5,10"), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn the_file_reads_back_record_by_record() {
        let mut out = Out::new(Vec::new());
        let c = Config { from: 12.5, period_ms: 5, count: 2 };
        out.header(&c, 77, 12.6, 1000, 2000, 10_000_000);
        out.module(0x1_4000_0000, 0x10_0000, 1, "xcl.exe");
        out.module(0x7ff8_0000_0000, 0x20_0000, 0, "ntdll.dll");
        out.thread(42, 0x1_4000_1234, "submit");
        out.thread(43, 0, "");
        out.sample(0, 12.7, 5000, 2);
        out.entry(42, STOPPED, 123_456, &[0x1_4000_2000, 0x1_4000_3000]);
        out.entry(43, UNCHANGED, 0, &[]);
        out.sample(1, 12.71, 6000, 1);
        out.entry(43, FAILED, 9, &[]);
        let s = Stats { samples: 2, threads: 2, stops: 1, unchanged: 1, failed: 1, sampler_tsc: 10, longest_stop_tsc: 3, stops_tsc: 3, late: 0 };
        out.end(&s, 7000, 3000, 12.8);
        out.flush();
        assert!(out.ok);
        let f = read(&out.w).expect("the file parses");
        assert_eq!((f.period_ms, f.count), (5, 2));
        assert_eq!(f.modules[0], (0x1_4000_0000, 0x10_0000, 1, "xcl.exe".into()));
        assert_eq!(f.threads, vec![(42, 0x1_4000_1234, "submit".into()), (43, 0, String::new())]);
        assert_eq!(f.samples.len(), 2);
        assert_eq!(f.samples[0].3[0], (42, STOPPED, 123_456, vec![0x1_4000_2000, 0x1_4000_3000]));
        assert_eq!(f.samples[0].3[1], (43, UNCHANGED, 0, vec![]));
        assert_eq!(f.samples[1].3, vec![(43, FAILED, 9, vec![])]);
        assert_eq!(f.end, Some((2, 2, vec![1, 1, 1, 10, 3, 3, 0, 7000])));
        // a truncated file stops at its last whole record
        assert!(read(&out.w[..out.w.len() - 3]).is_none());
    }

    #[inline(never)]
    fn spin(stop: &std::sync::atomic::AtomicBool) -> u64 {
        let mut x = 0u64;
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
        }
        x
    }

    /// This process sampled: a spinning thread is stopped and found running in this binary with
    /// its caller unwound, its cycle time moving; a thread blocked in a wait is found in ntdll
    /// and then left unchanged; the sampler's own thread is never in the file.
    /// `XCL_THREAD_SAMPLE_TEST_OUT=<path>` keeps the file (for S/tools/thrprof.py).
    #[test]
    fn a_sampling_of_this_process_finds_the_spinning_and_the_waiting_thread() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let s2 = stop.clone();
        let spinner = std::thread::Builder::new().name("ts-spinner".into()).spawn(move || spin(&s2)).unwrap();
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let waiter = std::thread::Builder::new().name("ts-waiter".into()).spawn(move || rx.recv().ok()).unwrap();
        // Rust names its threads with SetThreadDescription; wait until both are running
        std::thread::sleep(std::time::Duration::from_millis(50));
        let (sp_id, wt_id) = unsafe {
            use std::os::windows::io::AsRawHandle;
            let id = |h: *mut c_void| windows_sys::Win32::System::Threading::GetThreadId(h);
            (id(spinner.as_raw_handle()), id(waiter.as_raw_handle()))
        };
        let mut out = Out::new(Vec::new());
        let c = Config { from: 0.0, period_ms: 2, count: 150 };
        let (me, st) = std::thread::scope(|s| {
            s.spawn(|| (unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }, run(c, &mut out))).join().unwrap()
        });
        stop.store(true, Ordering::Relaxed);
        drop(tx);
        spinner.join().unwrap();
        waiter.join().unwrap();
        assert!(out.ok);
        if let Some(p) = std::env::var_os("XCL_THREAD_SAMPLE_TEST_OUT") {
            std::fs::write(p, &out.w).unwrap();
        }
        let f = read(&out.w).expect("the file parses");
        eprintln!("thread sample test: {st:?}; {} threads, {} modules", f.threads.len(), f.modules.len());
        assert_eq!(f.samples.len(), 150);
        assert_eq!(st.samples, 150);
        assert!(f.end.is_some());
        let own = f.modules.iter().find(|m| m.2 == 1).expect("the test binary is a kind 1 module").clone();
        let ntdll = f.modules.iter().find(|m| m.3.to_ascii_lowercase().ends_with("\\ntdll.dll")).expect("ntdll is listed").clone();
        let inside = |a: u64, m: &(u64, u64, u8, String)| a >= m.0 && a < m.0 + m.1;
        assert!(f.threads.iter().any(|t| t.0 == sp_id && t.2 == "ts-spinner"), "the spinner's name");
        assert!(f.threads.iter().all(|t| t.0 != me), "the sampler is not sampled");
        let mut spin_stopped = 0;
        let mut spin_in_own = 0;
        let mut spin_with_caller = 0;
        let mut spin_cycles = 0u64;
        let (mut wait_stopped, mut wait_in_ntdll, mut wait_unchanged) = (0, 0, 0);
        let mut wait_noise_max = 0u64;
        for (_, _, _, es) in &f.samples {
            assert!(es.iter().all(|e| e.0 != me));
            for e in es {
                if e.0 == sp_id {
                    spin_cycles += e.2;
                    if e.1 == STOPPED {
                        spin_stopped += 1;
                        spin_in_own += usize::from(inside(e.3[0], &own));
                        spin_with_caller += usize::from(e.3.len() >= 2 && inside(e.3[1], &own));
                    }
                }
                if e.0 == wt_id {
                    match e.1 {
                        STOPPED => {
                            wait_stopped += 1;
                            wait_in_ntdll += usize::from(inside(e.3[0], &ntdll));
                        }
                        UNCHANGED => {
                            wait_unchanged += 1;
                            wait_noise_max = wait_noise_max.max(e.2);
                        }
                        _ => {}
                    }
                }
            }
        }
        eprintln!(
            "spinner: {spin_stopped} stopped, {spin_in_own} in this binary, {spin_with_caller} with a caller, {spin_cycles} cycles; waiter: {wait_stopped} stopped ({wait_in_ntdll} in ntdll), {wait_unchanged} unchanged (at most {wait_noise_max} cycles between two)"
        );
        // the spinner runs all along: stopped at every sample (its cycles move), in spin()
        assert!(spin_stopped >= 140, "the spinner was stopped at nearly every sample");
        assert!(spin_in_own * 10 >= spin_stopped * 9, "the spinner's RIP is in this binary");
        assert!(spin_with_caller * 10 >= spin_stopped * 9, "its caller was unwound");
        assert!(spin_cycles > 0, "its cycle time moved");
        // the waiter: in ntdll when stopped, and mostly left unchanged
        assert!(wait_stopped >= 1 && wait_in_ntdll == wait_stopped, "the waiter waits in ntdll");
        assert!(wait_unchanged >= 100, "a thread that does not run is not stopped again");
    }
}
