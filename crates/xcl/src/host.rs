//! The host process: started by the launcher with the game's command line. Sets up
//! logging, the VFS, hooks and HLE, maps the title and jumps to its entry point.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::Boot;

pub static HOST_MODE: AtomicBool = AtomicBool::new(false);

/// The system's commit charge and its limit in bytes (RAM plus page files): what all programs'
/// memory, and all video memory, is counted against. (0, 0): not known.
pub fn system_commit() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    unsafe {
        let mut p: PERFORMANCE_INFORMATION = std::mem::zeroed();
        p.cb = std::mem::size_of_val(&p) as u32;
        if K32GetPerformanceInfo(&mut p, p.cb) == 0 {
            return (0, 0);
        }
        (p.CommitTotal as u64 * p.PageSize as u64, p.CommitLimit as u64 * p.PageSize as u64)
    }
}

/// Whether commit left over (`limit - commit`) is logged as low: under `LOW_COMMIT` the first
/// time, then each time it is half of what was logged last.
fn low_commit_logged(left: u64, last: u64) -> bool {
    left < LOW_COMMIT && (last == 0 || left <= last / 2)
}

/// Commit left under which the log warns: a retail title's world took the last 6 GB in seconds.
const LOW_COMMIT: u64 = 8 << 30;

/// The system is running out of commit: said in the log (once, then at each halving). When it
/// runs out, allocations fail for every program. (Measured on a retail title's world, whose
/// process held 45 GB beside 71 GB of other programs under a 147 GB limit: the process's own
/// memory counts against the commit, its 24 GB of video memory do not.)
fn warn_low_commit(commit: u64, limit: u64, private: u64, video: u64) {
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let left = limit.saturating_sub(commit);
    let last = LAST.load(Ordering::Relaxed);
    if !low_commit_logged(left, last) {
        return;
    }
    LAST.store(left.max(1), Ordering::Relaxed);
    crate::xlog_warn!(
        "memory: the system is running out of commit: {} MB left of {} MB (RAM + page file). This process holds {} MB (and {} MB of video memory, which do not count), other programs the rest. When it runs out allocations fail for every program: close other programs or give Windows a larger page file",
        left >> 20,
        limit >> 20,
        private >> 20,
        video >> 20
    );
}

pub fn host_main(boot_path: &Path) -> i32 {
    let boot: Boot = match std::fs::read_to_string(boot_path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("xcl host: cannot read {}: {e}", boot_path.display());
            return 0x7C1_0001;
        }
    };
    crate::log::init(
        Some(&boot.logs_dir().join("xcl.log")),
        crate::log::Level::parse(&boot.config.log_level),
        boot.config.log_to_stderr,
    );
    crate::log::log_panics();
    crate::xlog_info!("XCL {} – host for '{}'", env!("CARGO_PKG_VERSION"), boot.title);
    crate::xlog_info!("game: {} (read-only folder: {})", boot.exe_path.display(), boot.game_root.display());
    crate::xlog_info!("XCL data: {}", boot.data_dir.display());
    crate::xlog_info!(
        "console: {} (title memory {} MiB, mode {})",
        if boot.config.is_series_s() { "Xbox Series S (lockhart)" } else { "Xbox Series X (anaconda)" },
        boot.config.title_memory_mb,
        boot.config.title_memory
    );

    if std::env::var_os("XCL_UMA_MB").is_none() && boot.config.uma_mb != 0 {
        std::env::set_var("XCL_UMA_MB", boot.config.uma_mb.to_string());
    }
    crate::state::set_boot(boot.clone());
    if let Err(e) = crate::hle::d3d12x::present::preparation::before_guest(&boot) {
        crate::xlog_error!("Shader preparation: {e}");
        eprintln!("xcl: {e}");
        return 2;
    }
    let _ = std::thread::Builder::new().name("xcl-split-counts".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(10));
        crate::hle::d3d12x::real::log_dispatch_chain_counts();
    });
    // `XCL_ACTIVITY=1`: every 2 s what the title did since (a timeline of where it stops)
    if std::env::var_os("XCL_ACTIVITY").is_some() {
        let _ = std::thread::Builder::new().name("xcl-activity".into()).spawn(|| {
            let snap = || {
                let (p, e) = crate::hle::d3d12x::present::activity();
                let (d, x) = crate::hle::d3d12x::bind::activity();
                let f = crate::hle::d3d12x::frame::activity();
                let (r, b, fail) = crate::hle::dstorage::counters();
                [p, e, d, x, f[0], f[1], f[2], f[3], r, b >> 20, fail]
            };
            let mut last = snap();
            let mut ticks = 0u32;
            // lab: bytes of `[base, base + size)` that are in the process's working set
            let resident_bytes = |base: usize, size: usize| -> u64 {
                use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
                const PAGES: usize = 1 << 16;
                let mut buf: Vec<PSAPI_WORKING_SET_EX_INFORMATION> = vec![unsafe { std::mem::zeroed() }; PAGES.min(size.div_ceil(4096).max(1))];
                let (mut n, mut at) = (0u64, base);
                while at < base + size {
                    let count = PAGES.min((base + size - at) >> 12);
                    for (i, e) in buf[..count].iter_mut().enumerate() {
                        e.VirtualAddress = (at + (i << 12)) as *mut _;
                    }
                    let ok = unsafe { K32QueryWorkingSetEx(windows_sys::Win32::System::Threading::GetCurrentProcess(), buf.as_mut_ptr() as *mut _, (count * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32) };
                    if ok != 0 {
                        n += buf[..count].iter().filter(|e| unsafe { e.VirtualAttributes.Flags } & 1 != 0).count() as u64;
                    }
                    at += count << 12;
                }
                n << 12
            };
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = snap();
                let d: Vec<u64> = now.iter().zip(last).map(|(a, b)| a - b).collect();
                crate::xlog_info!(
                    "activity (2 s): {} presents, {} submits, {} draws, {} dispatches, {} frame-event waits ({} timed out), {} frame statistics ({} not flipped); DirectStorage {} requests, {} MB, {} failed",
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]
                );
                // (lab, never waits for a lock: what a stuck sparse queue or lock would show)
                // (lab: only with XCL_CALL_STATS, so a run without it logs and locks as the base did)
                if let Some(line) = crate::hle::d3d12x::sparse::fence_line_if(crate::hle::d3d12x::call_stats::on()) {
                    crate::xlog_info!("{line}");
                }
                // where the memory is: the process's commit, video memory, and XCL's parts of it
                let private = unsafe {
                    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
                    let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
                    c.cb = std::mem::size_of_val(&c) as u32;
                    K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
                    c.PrivateUsage
                };
                crate::xlog_info!("{}", crate::hle::d3d12x::view_mem::stats());
                let (local, non_local) = crate::hle::d3d12x::real::video_memory().unwrap_or((0, 0));
                let (arenas, arenas_made) = crate::hle::d3d12x::vram::arena_bytes();
                let (pools, pools_mapped) = crate::hle::d3d12x::sparse::pool_bytes();
                let (pages, page_pools) = crate::hle::d3d12x::vram::page_bytes();
                let mb = |b: u64| b >> 20;
                // the system's commit: this process's memory counts against it with every other
                // program's (video memory does not), and where it ends allocations fail (warned
                // when low; the lab keeps the memory line as it was)
                let (commit, commit_limit) = system_commit();
                if commit_limit != 0 {
                    warn_low_commit(commit, commit_limit, private as u64, local);
                }
                crate::xlog_info!(
                    "memory: {} MB committed by the process; video memory {} MB local, {} MB non-local; target pages {} MB (pools {} MB); arenas {} MB in use, {} MB made; sparse pools {} MB ({} MB mapped)",
                    mb(private as u64),
                    mb(local),
                    mb(non_local),
                    mb(pages as u64),
                    mb(page_pools as u64),
                    mb(arenas as u64),
                    mb(arenas_made as u64),
                    mb(pools as u64),
                    mb(pools_mapped as u64)
                );
                // lab: what is resident (the commit above is not), the card's budget, the parts
                let (ws, avail) = unsafe {
                    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
                    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
                    let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
                    c.cb = std::mem::size_of_val(&c) as u32;
                    K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c, c.cb);
                    let mut m: MEMORYSTATUSEX = std::mem::zeroed();
                    m.dwLength = std::mem::size_of_val(&m) as u32;
                    GlobalMemoryStatusEx(&mut m);
                    (c.WorkingSetSize as u64, m.ullAvailPhys)
                };
                let budget = crate::hle::d3d12x::real::video_budget().unwrap_or(0);
                let (res, res_held, res_free) = crate::hle::d3d12x::bind::res_stats();
                let (clamp, clamp_made) = crate::hle::d3d12x::bind::clamp_stats();
                crate::xlog_info!(
                    "memory lab: working set {} MB, system free {} MB, video budget {} MB; record buffers {} MB ({} allocators hold some, {} free); clamp buffers {} MB ({} made); {}",
                    mb(ws),
                    mb(avail),
                    mb(budget),
                    mb(res),
                    res_held,
                    res_free,
                    mb(clamp),
                    clamp_made,
                    crate::hle::d3d12x::vram::page_breakdown()
                );
                // lab: the compiled code of the pipelines in use (`XCL_PSO_CODE`, `XCL_PSO_CODE_MB`)
                if let Some(line) = crate::hle::d3d12x::pso_async::code_tick() {
                    crate::xlog_info!("shader code lab: {line}");
                }
                // lab: every 10 s, how much of each memory window is in the working set
                // (pages the title or XCL touched and Windows has not trimmed)
                ticks += 1;
                if ticks % 5 == 0 {
                    if let Some(u) = crate::gpu::uma::get() {
                        let parts: Vec<String> = (0..u.shader_windows())
                            .map(|k| {
                                let (base, size) = u.window(k);
                                format!("window {k}{} {} MB", if u.is_sparse_window(k) { " (sparse)" } else if u.is_committed_window(k) { " (coherent)" } else { "" }, mb(resident_bytes(base, size)))
                            })
                            .collect();
                        crate::xlog_info!("memory lab: in the working set: {}", parts.join(", "));
                        crate::xlog_info!("memory lab: {}", crate::hle::d3d12x::vram::pinned_sizes());
                        // the title's memory past the windows (the CPU view of its 24 GB GPU view)
                        let end = u.base() + (u.shader_windows() << 32);
                        crate::xlog_info!("memory lab: in the working set: {} MB of the 24 GB after the windows ({end:#x}..)", mb(resident_bytes(end, 24 << 30)));
                    }
                }
                // lab (`XCL_TRIM_VIEWS=1`): every 2 s, the mapped views of 256 MB or more are taken
                // out of the working set. The NVIDIA driver maps its 4 GB shader cache file
                // (NVIDIA/DXCache/*.nvph under %LOCALAPPDATA%) and reads 3.5 GB of it within
                // seconds some 30 s into a retail title's world: runs died of RAM there before
                // the 30 s walk below got to it.
                if std::env::var("XCL_TRIM_VIEWS").is_ok_and(|v| v.trim() == "1") {
                    use windows_sys::Win32::System::Memory::{VirtualQuery, VirtualUnlock, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_MAPPED};
                    let mut views: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                    let mut at = 0x1_0000usize;
                    while at < 0x7FFF_FFFE_0000 {
                        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                        if unsafe { VirtualQuery(at as *const _, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 || m.RegionSize == 0 {
                            break;
                        }
                        if m.State == MEM_COMMIT && m.Type == MEM_MAPPED {
                            let e = views.entry(m.AllocationBase as usize).or_insert(0);
                            *e = (*e).max(m.BaseAddress as usize + m.RegionSize - m.AllocationBase as usize);
                        }
                        at = m.BaseAddress as usize + m.RegionSize;
                    }
                    for (&b, &span) in views.iter().filter(|e| *e.1 >= 256 << 20) {
                        unsafe { VirtualUnlock(b as *const _, span) };
                        static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap();
                        if !seen.contains(&b) {
                            seen.push(b);
                            crate::xlog_info!("memory lab: mapped view of {} MB at {b:#x}: taken out of the working set every 2 s from now", span >> 20);
                        }
                    }
                }
                // lab: every 30 s, the working set by allocation (VirtualQuery over the whole
                // address space): by kind, and the 16 allocations with most resident bytes
                // (`XCL_MEM_REGIONS=1`; mapped views are counted as 0)
                if ticks % 15 == 7 && std::env::var_os("XCL_MEM_REGIONS").is_some() {
                    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_IMAGE, MEM_MAPPED};
                    let mut by_base: std::collections::HashMap<usize, (u64, u64, u64, u32)> = std::collections::HashMap::new();
                    let mut kinds = [0u64; 3];
                    let mut at = 0x1_0000usize;
                    while at < 0x7FFF_FFFE_0000 {
                        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                        if unsafe { VirtualQuery(at as *const _, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 || m.RegionSize == 0 {
                            break;
                        }
                        if m.State == MEM_COMMIT {
                            // not asked of mapped views: asking made 3.4 GB of a 4 GB view some
                            // other component maps part of the working set (run m6)
                            let r = if m.Type == MEM_MAPPED { 0 } else { resident_bytes(m.BaseAddress as usize, m.RegionSize) };
                            let e = by_base.entry(m.AllocationBase as usize).or_insert((0, 0, 0, m.Type));
                            e.0 += r;
                            e.1 += m.RegionSize as u64;
                            e.2 = e.2.max((m.BaseAddress as usize + m.RegionSize - m.AllocationBase as usize) as u64);
                            kinds[if m.Type == MEM_IMAGE { 0 } else if m.Type == MEM_MAPPED { 1 } else { 2 }] += r;
                        }
                        at = m.BaseAddress as usize + m.RegionSize;
                    }
                    // lab: sparse-window tiles that hold what textures were filled from
                    {
                        let runs = crate::hle::d3d12x::sparse::mapped_ranges();
                        let (bytes, whole, n) = crate::hle::d3d12x::title_textures::under_sources(&runs);
                        crate::xlog_info!(
                            "memory lab: sparse windows: {} MB of tiles mapped, {} MB of them under the bytes of {n} CPU-filled textures ({} MB in tiles wholly under them)",
                            mb(runs.iter().map(|r| (r.1 - r.0) as u64).sum()),
                            mb(bytes as u64),
                            mb((whole as u64) << 16)
                        );
                    }
                    // allocations by size class: how much of the working set each class holds
                    let mut classes = [(0u64, 0u32); 6];
                    for &(r, _, span, _) in by_base.values() {
                        let c = match span >> 20 {
                            0 => 0,
                            1..=15 => 1,
                            16..=63 => 2,
                            64..=255 => 3,
                            256..=1023 => 4,
                            _ => 5,
                        };
                        classes[c].0 += r;
                        classes[c].1 += 1;
                    }
                    // lab: the mapped views of 256 MB or more (some other component's: XCL maps
                    // none): where, how much is committed, and the file behind them if any;
                    // `XCL_TRIM_VIEWS=1`: their pages are taken out of the working set
                    // (VirtualUnlock on pages that are not locked does that)
                    for (&b, &(_, c, span, t)) in by_base.iter() {
                        if t != MEM_MAPPED || span < 256 << 20 {
                            continue;
                        }
                        let mut name = [0u16; 260];
                        let n = unsafe { windows_sys::Win32::System::ProcessStatus::K32GetMappedFileNameW(windows_sys::Win32::System::Threading::GetCurrentProcess(), b as *const _, name.as_mut_ptr(), 260) } as usize;
                        let trimmed = std::env::var("XCL_TRIM_VIEWS").is_ok_and(|v| v.trim() == "1") && {
                            unsafe { windows_sys::Win32::System::Memory::VirtualUnlock(b as *const _, span as usize) };
                            true
                        };
                        crate::xlog_info!("memory lab: mapped view at {b:#x}: {} of {} MB committed, file {:?}{}", mb(c), mb(span), String::from_utf16_lossy(&name[..n.min(260)]), if trimmed { ", taken out of the working set" } else { "" });
                    }
                    let mut v: Vec<(usize, (u64, u64, u64, u32))> = by_base.into_iter().collect();
                    v.sort_unstable_by_key(|&(_, (r, ..))| std::cmp::Reverse(r));
                    let top: Vec<String> = v.iter().take(16).map(|&(b, (r, c, span, t))| format!("{b:#x} {} of {} MB committed (span {} MB, {})", mb(r), mb(c), mb(span), if t == MEM_IMAGE { "image" } else if t == MEM_MAPPED { "mapped" } else { "private" })).collect();
                    crate::xlog_info!(
                        "memory lab: working set by kind: image {} MB, mapped {} MB, private {} MB; by allocation span: <1 MB {} MB ({}), <16 MB {} MB ({}), <64 MB {} MB ({}), <256 MB {} MB ({}), <1 GB {} MB ({}), larger {} MB ({}); most: {}",
                        mb(kinds[0]), mb(kinds[1]), mb(kinds[2]),
                        mb(classes[0].0), classes[0].1, mb(classes[1].0), classes[1].1, mb(classes[2].0), classes[2].1, mb(classes[3].0), classes[3].1, mb(classes[4].0), classes[4].1, mb(classes[5].0), classes[5].1,
                        top.join("; ")
                    );
                }
                let (evicted, mapped_again) = crate::hle::d3d12x::vram::eviction_counts();
                if evicted > 0 {
                    crate::xlog_info!("page budget: {evicted} textures evicted, {mapped_again} mapped again");
                }
                last = now;
            }
        });
    }
    // lab: `XCL_WS_MAX_MB=n` – a hard maximum for the working set: Windows takes the pages
    // used least recently out of it (to the page file; a later touch brings one back) instead
    // of the PC running out of RAM. A retail title's working set grows by ~20 MB a second in
    // its world, most of it memory written once (streamed textures the GPU already has).
    if let Some(mb) = std::env::var("XCL_WS_MAX_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n >= 1024) {
        #[link(name = "kernel32")]
        extern "system" {
            fn SetProcessWorkingSetSizeEx(process: *mut std::ffi::c_void, min: usize, max: usize, flags: u32) -> i32;
        }
        // QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE
        let ok = unsafe { SetProcessWorkingSetSizeEx(windows_sys::Win32::System::Threading::GetCurrentProcess() as *mut _, 256 << 20, mb << 20, 2 | 4) };
        crate::xlog_info!("memory lab: working set limited to {mb} MB (XCL_WS_MAX_MB): {}", if ok != 0 { "set".to_string() } else { format!("failed, {}", std::io::Error::last_os_error()) });
    }
    // lab: `XCL_MIN_FREE_MB=n` – the process ends itself when the PC's free RAM is under n MB at
    // four readings in a row (2 s) or under n / 2 MB once. The lab harness has this guard in a
    // PowerShell process, and that process was ended from outside twice (runs G2 / G3) while the
    // title ran on: this one lives and dies with the title.
    if let Some(mb) = std::env::var("XCL_MIN_FREE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&n| n > 0) {
        let _ = std::thread::Builder::new().name("xcl-ram-guard".into()).spawn(move || {
            use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
            crate::xlog_info!("memory lab: the process ends itself under {mb} MB of free RAM (XCL_MIN_FREE_MB)");
            let mut low = 0;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
                m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
                if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
                    continue;
                }
                let free = m.ullAvailPhys >> 20;
                low = if free < mb { low + 1 } else { 0 };
                if low >= 4 || free < mb / 2 {
                    crate::xlog_error!("XCL_MIN_FREE_MB: {free} MB of RAM free (limit {mb} MB) – the process ends itself");
                    crate::log::flush();
                    unsafe { windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), 0xE000_0001) };
                }
            }
        });
    }
    // `XCL_THREAD_DUMP_AT=s1,s2…`: where every thread is at those seconds (a stalled title)
    if let Ok(v) = std::env::var("XCL_THREAD_DUMP_AT") {
        let at: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let _ = std::thread::Builder::new().name("xcl-thread-dump".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            for s in at {
                let left = s - t0.elapsed().as_secs_f64();
                if left > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(left));
                }
                crate::eh::dump_threads();
            }
        });
    }
    // lab: `XCL_THREAD_SAMPLE=<from s>,<period ms>,<count>` – every thread's CPU and stacks, to a file (`thread_sample`)
    crate::thread_sample::start_from_env();
    // lab: `XCL_SAMPLE=<from s>,<seconds>[,<ms>]` – a sampling profile of every thread (`eh::sample_profile`)
    if let Ok(v) = std::env::var("XCL_SAMPLE") {
        let p: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if p.len() >= 2 {
            let file = std::env::var_os("XCL_SAMPLE_FILE").map_or_else(|| std::env::temp_dir().join("xcl-profile.txt"), std::path::PathBuf::from);
            let every = p.get(2).map_or(10, |&m| (m as u64).max(1));
            let _ = std::thread::Builder::new().name("xcl-sample".into()).spawn(move || crate::eh::sample_profile(p[0], p[1], every, file));
        }
    }
    crate::stubs::init_db(&boot.config.stub_returns);
    crate::hle::init(&boot.config);
    crate::hle::kernelx::fine_timer();
    crate::loader::set_cache_dir(boot.cache_dir());
    let mut dirs = vec![boot.game_root.clone()];
    if let Some(p) = boot.exe_path.parent() {
        if p != boot.game_root {
            dirs.insert(0, p.to_path_buf());
        }
    }
    crate::resolver::set_search_dirs(dirs);

    let mounts: Vec<(String, String)> = boot.config.mounts.iter().map(|(k, v)| (k.clone(), boot.expand(v))).collect();
    // (the shader cache is XCL's too: on a drive a console mount names it must stay where it is)
    crate::vfs::init(crate::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts).own_folder(&crate::gpu::recomp::dxc::cache_dir()));
    let _ = std::fs::create_dir_all(boot.data_dir.join("temp"));

    if let Err(e) = crate::vfs_hooks::install() {
        crate::xlog_error!("VFS: not all hooks installed: {e}");
    }
    if let Err(e) = crate::eh::install_pc_hook() {
        crate::xlog_error!("hook RtlPcToFileHeader: {e} – C++ exceptions in the game may not work");
    }
    crate::eh::install_crash_handler();
    crate::eh::start_probes();
    crate::hle::d3d12x::backend_before_guest();
    HOST_MODE.store(true, Ordering::Release);

    // `XCL_RUN_SECONDS=n`: the title ends itself after n seconds – the GPU drained, the log and
    // report written (a process killed from outside with GPU work in flight can hang in the driver)
    if let Some(secs) = std::env::var("XCL_RUN_SECONDS").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            let (shown, presented) = crate::hle::d3d12x::present::counters();
            crate::xlog_info!("XCL_RUN_SECONDS: {secs} s – ending the title ({shown} frames shown, {presented} PresentX calls)");
            crate::report::write_final(None);
            crate::log::flush();
            crate::hle::d3d12x::real::end_title(0);
        });
    }

    let main = match crate::loader::load_main(&boot.exe_path) {
        Ok(m) => m,
        Err(e) => {
            crate::xlog_error!("loading the game failed: {e}");
            crate::report::write_final(Some(format!("loader: {e}")));
            return 0x7C1_0002;
        }
    };
    unsafe { crate::hle::kernelx::adopt_title_hooks(&main) };
    crate::report::write_final(None);
    let code = crate::loader::run_main(&main);
    crate::xlog_info!("game entry point returned code {code}");
    crate::report::write_final(None);
    crate::log::flush();
    code as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_commit_is_logged_under_the_mark_and_at_each_halving() {
        assert!(!low_commit_logged(LOW_COMMIT, 0), "at the mark: not yet");
        assert!(low_commit_logged(LOW_COMMIT - 1, 0));
        assert!(!low_commit_logged(5 << 30, 7 << 30), "less, but not half of what was logged last");
        assert!(low_commit_logged(3 << 30, 7 << 30));
        assert!(low_commit_logged(0, 1), "none left");
        assert!(!low_commit_logged(9 << 30, 1 << 30), "back above the mark");
    }

    #[test]
    fn the_system_commit_is_known_and_under_its_limit() {
        let (commit, limit) = system_commit();
        assert!(commit > 0 && commit <= limit, "{commit} of {limit}");
    }
}
