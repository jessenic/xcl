//! lab `XCL_RT_WATCH=<frames>` (off by default): a probe of a title's ray-tracing structures,
//! nothing changed in what runs.
//!
//! A retail title builds its acceleration structures with the console's BVH builder (compute
//! programs) and reads, on the CPU, sizes those programs write (the compacted size). On a GPU
//! without host-memory heaps every unified-memory window is a copy in video memory, and what a
//! compute program stores there never comes back to the CPU's view (`list_x::note_write` takes
//! copies, fills and Write32/64 only). So the header of a structure as the CPU sees it says
//! nothing about the GPU's (PR #3026's watch read the CPU's). Every `<frames>` frames this logs:
//! * each structure a view names (`D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE`): words
//!   0, 1, 3, 12 (the root, @0x30) and 13 of its 256-byte header as the CPU sees them and as
//!   video memory holds them (`sparse::peek_gpu`, which leaves the CPU's view alone), how many of
//!   the 256 bytes had video memory under them, how many differ, and the watched records whose
//!   dispatches were given an address in that header as a root argument;
//! * for the records of `XCL_RT_WATCH_WRITERS` (record hashes): the first 16 bytes at each root
//!   address their dispatches were given, CPU against GPU (the compacted size the title reads);
//! * for the records of `XCL_RT_WATCH_RECORDS` (a file, one record hash first on each line: the
//!   builder's programs): pipelines created, dispatches launched, put off to the submit (counted
//!   again when recorded there), without a translated pipeline (skipped), failed; put-off
//!   dispatches dropped (any pipeline); their DispatchX calls: how many, how many with flag 0x20
//!   (thread dimensions: x counts threads; XCL launches x groups unless `XCL_DISPATCHX_FLAGS=1`),
//!   the largest x and the flags seen.
//!
//! Record hashes are XCL's (`fnv64` of the console shader container); the source names no
//! title's program.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::console_list::Param;

/// structures remembered (distinct addresses)
const MOST_STRUCTURES: usize = 256;
/// root addresses remembered per watched record
const MOST_TARGETS: usize = 32;
/// structures / writers' targets peeked at per line
const PEEKS: usize = 12;
/// bytes of a writer's target compared
const TARGET_BYTES: usize = 16;
/// the DispatchX flag counted: USE_THREAD_DIMENSIONS (0x8000 is the wave32 bit)
const FLAG_THREADS: u32 = 0x20;

/// The period of `XCL_RT_WATCH`: frames, at least 30 (`None`: off).
pub fn period_of(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok().filter(|&n| n > 0).map(|n| n.max(30))
}

/// Record hashes in `text`: the first token of each line or comma-separated item, 16 hex digits
/// (anything else is passed over).
pub fn records_of(text: &str) -> HashSet<u64> {
    text.split(['\n', ','])
        .filter_map(|l| l.split_whitespace().next())
        .filter(|t| t.len() == 16)
        .filter_map(|t| u64::from_str_radix(t, 16).ok())
        .collect()
}

/// The root addresses (CBV / SRV / UAV) in a dispatch's root arguments: 48-bit, as the
/// console's descriptors take them; 0 left out.
pub fn root_addresses(params: &[Param], args: &[u8]) -> Vec<usize> {
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    params
        .iter()
        .filter_map(|p| match *p {
            Param::Descriptor { at } => Some(((word(at as usize + 4)? as usize & 0xFFFF) << 32) | word(at as usize)? as usize),
            _ => None,
        })
        .filter(|&va| va != 0)
        .collect()
}

/// Words 0, 1, 3, 12 (the root, @0x30 = `gpu::bvh::ROOT_AT`) and 13 of a structure's header
/// (0 where the header is short).
pub fn header_words(h: &[u8]) -> [u32; 5] {
    let w = |at: usize| h.get(at..at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    [w(0), w(4), w(12), w(crate::gpu::bvh::ROOT_AT), w(0x34)]
}

/// `header_words` as a log field.
pub fn header_text(w: [u32; 5]) -> String {
    format!("w0 {:#x} w1 {:#x} w3 {:#x} root {:#x} w13 {:#x}", w[0], w[1], w[2], w[3], w[4])
}

/// Bytes that differ between `a` and `b` (over the shorter).
pub fn differing(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// What became of a dispatch of a watched pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Launched,
    PutOff,
    Untranslated,
    Failed,
}

/// Per record: pipelines created, dispatches by `Outcome`, DispatchX calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub pipelines: u64,
    pub launched: u64,
    pub put_off: u64,
    pub untranslated: u64,
    pub failed: u64,
    pub dx_calls: u64,
    pub dx_threads: u64,
    pub dx_max_x: u32,
    /// the flags of the first DispatchX calls, up to 4 distinct ones (0 = none)
    pub dx_flags: [u32; 4],
}

impl Counts {
    pub fn add(&mut self, o: Outcome) {
        match o {
            Outcome::Launched => self.launched += 1,
            Outcome::PutOff => self.put_off += 1,
            Outcome::Untranslated => self.untranslated += 1,
            Outcome::Failed => self.failed += 1,
        }
    }

    pub fn add_dispatch_x(&mut self, x: u32, flags: u32) {
        self.dx_calls += 1;
        if flags & FLAG_THREADS != 0 {
            self.dx_threads += 1;
        }
        self.dx_max_x = self.dx_max_x.max(x);
        // (flags 0 cannot be told from an empty slot: such calls count in dx_calls only)
        if flags != 0 && !self.dx_flags.contains(&flags) {
            if let Some(slot) = self.dx_flags.iter_mut().find(|f| **f == 0) {
                *slot = flags;
            }
        }
    }
}

/// The counters line of the builder's records: totals, records seen, records whose dispatches
/// never launched, up to 8 records with dispatches that did not launch, and the DispatchX
/// calls with thread dimensions (flag 0x20) by record.
pub fn counts_line(records: &BTreeMap<u64, Counts>, wanted: usize, dropped: u64) -> String {
    let mut t = Counts::default();
    for c in records.values() {
        t.pipelines += c.pipelines;
        t.launched += c.launched;
        t.put_off += c.put_off;
        t.untranslated += c.untranslated;
        t.failed += c.failed;
        t.dx_calls += c.dx_calls;
        t.dx_threads += c.dx_threads;
    }
    let made = records.values().filter(|c| c.pipelines > 0).count();
    let dispatched = records.values().filter(|c| c.launched + c.put_off + c.untranslated + c.failed > 0).count();
    let never: Vec<String> = records.iter().filter(|(_, c)| c.pipelines > 0 && c.launched == 0 && c.untranslated + c.failed + c.put_off > 0).map(|(h, _)| format!("{h:016x}")).collect();
    let bad: Vec<String> = records
        .iter()
        .filter(|(_, c)| c.untranslated + c.failed > 0)
        .take(8)
        .map(|(h, c)| format!("{h:016x} {}/{}/{}/{}", c.launched, c.put_off, c.untranslated, c.failed))
        .collect();
    let dx: Vec<String> = records
        .iter()
        .filter(|(_, c)| c.dx_threads > 0)
        .map(|(h, c)| {
            let flags: Vec<String> = c.dx_flags.iter().filter(|&&f| f != 0).map(|f| format!("{f:#x}")).collect();
            format!("{h:016x} {}/{} max x {} flags [{}]", c.dx_threads, c.dx_calls, c.dx_max_x, flags.join(" "))
        })
        .collect();
    format!(
        "builder records: {made} of {wanted} with a pipeline ({} pipelines), {dispatched} dispatched; dispatches launched {}, put off {}, without a translated pipeline {}, failed {}; put-off dispatches dropped (any pipeline) {dropped}; records never launched: [{}]; records with skips (launched/put off/untranslated/failed): [{}]; DispatchX {} calls, {} with flag 0x20 (thread counts) (record with/calls, largest x, flags): [{}]",
        t.pipelines,
        t.launched,
        t.put_off,
        t.untranslated,
        t.failed,
        never.join(" "),
        bad.join("; "),
        t.dx_calls,
        t.dx_threads,
        dx.join("; ")
    )
}

/// The records among `targets` (record → root addresses) given an address in `[va, va + len)`.
pub fn given(targets: &BTreeMap<u64, Vec<usize>>, va: usize, len: usize) -> Vec<u64> {
    targets.iter().filter(|(_, v)| v.iter().any(|&a| a >= va && a < va + len)).map(|(&r, _)| r).collect()
}

struct Config {
    every: u64,
    builder: HashSet<u64>,
    writers: HashSet<u64>,
}

fn config() -> Option<&'static Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    C.get_or_init(|| {
        let every = period_of(&std::env::var("XCL_RT_WATCH").ok()?)?;
        let file = std::env::var("XCL_RT_WATCH_RECORDS").ok();
        let builder = file.as_deref().and_then(|f| std::fs::read_to_string(f).ok()).map(|t| records_of(&t)).unwrap_or_default();
        let writers = std::env::var("XCL_RT_WATCH_WRITERS").ok().map(|t| records_of(&t)).unwrap_or_default();
        crate::xlog_info!(
            "lab XCL_RT_WATCH={every}: structures' headers and writers' targets, CPU against video memory, every {every} frames; {} builder record(s) from {}; {} writer record(s)",
            builder.len(),
            file.as_deref().unwrap_or("<no XCL_RT_WATCH_RECORDS>"),
            writers.len()
        );
        Some(Config { every, builder, writers })
    })
    .as_ref()
}

/// Whether the probe is on.
pub fn enabled() -> bool {
    config().is_some()
}

#[derive(Default)]
struct State {
    /// pipeline (as the title holds it) → record
    pipelines: HashMap<usize, u64>,
    counts: BTreeMap<u64, Counts>,
    structures: Vec<usize>,
    outside: u64,
    /// watched record → root addresses its dispatches were given
    targets: BTreeMap<u64, Vec<usize>>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static DROPPED: AtomicU64 = AtomicU64::new(0);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    f(STATE.lock().unwrap().get_or_insert_with(State::default))
}

/// A compute pipeline of record `record` was created; the title holds it as `pso`.
pub fn note_pipeline(record: u64, pso: usize) {
    let Some(c) = config() else { return };
    if pso == 0 || !(c.builder.contains(&record) || c.writers.contains(&record)) {
        return;
    }
    with(|s| {
        s.pipelines.insert(pso, record);
        if c.builder.contains(&record) {
            s.counts.entry(record).or_default().pipelines += 1;
        }
    });
}

/// The record of a watched pipeline (None: not watched, or the probe is off).
pub fn watched(pso: usize) -> Option<u64> {
    config()?;
    with(|s| s.pipelines.get(&pso).copied())
}

/// A dispatch of watched record `record` (from `watched`) ended as `o`.
pub fn seen(record: Option<u64>, o: Outcome) {
    let (Some(r), Some(c)) = (record, config()) else { return };
    if c.builder.contains(&r) {
        with(|s| s.counts.entry(r).or_default().add(o));
    }
}

/// DispatchX(x, …, flags) with `pso` bound.
pub fn dispatch_x(pso: usize, x: u32, flags: u32) {
    let Some(c) = config() else { return };
    let Some(r) = watched(pso) else { return };
    if c.builder.contains(&r) {
        with(|s| s.counts.entry(r).or_default().add_dispatch_x(x, flags));
    }
}

/// The root arguments of a dispatch of watched record `record`.
pub fn note_args(record: u64, params: &[Param], args: &[u8]) {
    let vas = root_addresses(params, args);
    if vas.is_empty() {
        return;
    }
    with(|s| {
        let t = s.targets.entry(record).or_default();
        for va in vas {
            if t.len() < MOST_TARGETS && !t.contains(&va) {
                t.push(va);
            }
        }
    });
}

/// A put-off dispatch was dropped (its pipeline is not known there).
pub fn dropped() {
    if enabled() {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// A view names the structure at `va` (`in_uma`: in unified memory, so shaders reach it).
pub fn note_structure(va: u64, in_uma: bool) {
    if !enabled() {
        return;
    }
    with(|s| {
        if !in_uma {
            s.outside += 1;
        } else if s.structures.len() < MOST_STRUCTURES && !s.structures.contains(&(va as usize)) {
            s.structures.push(va as usize);
        }
    });
}

/// Where `va` is: which kind of unified-memory window.
fn kind(va: usize) -> &'static str {
    let Some(u) = crate::gpu::uma::get() else { return "no unified memory" };
    if u.is_sparse(va) {
        "sparse window"
    } else if u.is_committed(va) && !u.shadowed() {
        "coherent window, shared"
    } else if u.is_committed(va) && super::sparse::is_lazy(va) {
        "coherent window, tiled copy"
    } else if u.is_committed(va) {
        "coherent window, copy"
    } else {
        "outside"
    }
}

/// `n` bytes at `va` as the CPU sees them (zeros where not readable).
fn cpu_bytes(va: usize, n: usize) -> Vec<u8> {
    if crate::gpu::shader::readable(va, n) {
        // SAFETY: `readable` just checked the range; title memory, only read
        unsafe { std::slice::from_raw_parts(va as *const u8, n).to_vec() }
    } else {
        vec![0; n]
    }
}

/// CPU and video memory at `va` (`n` bytes): (cpu, gpu and the bytes from video memory, or the
/// reason there is no GPU side).
fn both(va: usize, n: usize) -> (Vec<u8>, Result<(Vec<u8>, usize), &'static str>) {
    let cpu = cpu_bytes(va, n);
    // SAFETY: a copy from video memory into a vector of its own
    let gpu = unsafe { super::sparse::peek_gpu(va, n) }.ok_or("no copy in video memory to read (the GPU uses the CPU's memory here, or the queue did not answer)");
    (cpu, gpu)
}

fn hex_words(b: &[u8]) -> String {
    b.chunks(4).map(|w| format!("{:08x}", u32::from_le_bytes(w.try_into().unwrap_or([0; 4])))).collect::<Vec<_>>().join(" ")
}

/// Called at every frame boundary: the lines every `XCL_RT_WATCH` frames, once anything was seen.
pub fn tick(frames: u64) {
    let Some(c) = config() else { return };
    if frames % c.every != 0 {
        return;
    }
    let (structures, outside, counts, targets) = with(|s| (s.structures.clone(), s.outside, s.counts.clone(), s.targets.clone()));
    let dropped = DROPPED.load(Ordering::Relaxed);
    if structures.is_empty() && outside == 0 && counts.is_empty() && targets.is_empty() {
        return;
    }
    let t = crate::log::uptime();
    crate::xlog_info!("lab XCL_RT_WATCH: frame {frames}, {t:.1} s: {} structure(s) named in unified memory, {outside} view(s) outside it; {}", structures.len(), counts_line(&counts, c.builder.len(), dropped));
    let header = crate::gpu::bvh::HEADER_BYTES;
    for (i, &va) in structures.iter().take(PEEKS).enumerate() {
        let (cpu, gpu) = both(va, header);
        let by: Vec<String> = given(&targets, va, header).iter().map(|r| format!("{r:016x}")).collect();
        let gpu_side = match gpu {
            Ok((g, from_vram)) => format!("GPU {} | {from_vram} of {} bytes with video memory, {} differ", header_text(header_words(&g)), g.len(), differing(&cpu, &g)),
            Err(why) => format!("GPU: {why}"),
        };
        crate::xlog_info!("lab XCL_RT_WATCH: structure #{i} {va:#x} ({}): CPU {} | {gpu_side} | given as a root argument to: [{}]", kind(va), header_text(header_words(&cpu)), by.join(" "));
    }
    let mut left = PEEKS;
    for (r, vas) in targets.iter().filter(|(r, _)| c.writers.contains(*r)) {
        for &va in vas {
            if left == 0 {
                break;
            }
            left -= 1;
            let (cpu, gpu) = both(va, TARGET_BYTES);
            let gpu_side = match gpu {
                Ok((g, from_vram)) => format!("GPU [{}] | {from_vram} of {TARGET_BYTES} bytes with video memory, {} differ", hex_words(&g), differing(&cpu, &g)),
                Err(why) => format!("GPU: {why}"),
            };
            crate::xlog_info!("lab XCL_RT_WATCH: writer {r:016x} target {va:#x} ({}): CPU [{}] | {gpu_side}", kind(va), hex_words(&cpu));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_watch_period_is_off_unless_a_positive_number() {
        assert_eq!(period_of(""), None);
        assert_eq!(period_of("0"), None);
        assert_eq!(period_of("x"), None);
        assert_eq!(period_of(" 300 "), Some(300));
        assert_eq!(period_of("5"), Some(30), "at least 30 frames");
    }

    #[test]
    fn rt_watch_records_are_the_first_token_of_lines_or_items() {
        let r = records_of("0123456789abcdef some words\n\n1111222233334444\r\nnot-a-hash x\nfedcba9876543210,0a0b0c0d0e0f1011 y");
        let want: HashSet<u64> = [0x0123456789abcdef, 0x1111222233334444, 0xfedcba9876543210, 0x0a0b0c0d0e0f1011].into_iter().collect();
        assert_eq!(r, want);
        assert!(records_of("").is_empty());
        assert!(records_of("123 abc").is_empty(), "short tokens are passed over");
    }

    #[test]
    fn rt_watch_root_addresses_are_the_descriptors_48_bit_and_non_zero() {
        let mut args = vec![0u8; 40];
        args[0..4].copy_from_slice(&7u32.to_le_bytes()); // constants
        args[8..12].copy_from_slice(&0x2345_6700u32.to_le_bytes()); // descriptor lo
        args[12..16].copy_from_slice(&0xABCD_0009u32.to_le_bytes()); // hi: only the low 16 bits count
        args[24..28].copy_from_slice(&0x1000u32.to_le_bytes()); // table
        let params = [Param::Constants { at: 0, count: 2 }, Param::Descriptor { at: 8 }, Param::Descriptor { at: 16 }, Param::Table { at: 24 }, Param::Descriptor { at: 36 }];
        assert_eq!(root_addresses(&params, &args), vec![0x9_2345_6700], "the zero descriptor at 16 and the one cut short at 36 are left out");
    }

    #[test]
    fn rt_watch_header_words_and_differences() {
        let mut h = vec![0u8; crate::gpu::bvh::HEADER_BYTES];
        h[0..4].copy_from_slice(&6u32.to_le_bytes());
        h[4..8].copy_from_slice(&42u32.to_le_bytes());
        h[12..16].copy_from_slice(&862u32.to_le_bytes());
        h[0x30..0x34].copy_from_slice(&u32::MAX.to_le_bytes());
        h[0x34..0x38].copy_from_slice(&0x8c80u32.to_le_bytes());
        assert_eq!(header_words(&h), [6, 42, 862, u32::MAX, 0x8c80]);
        assert_eq!(header_text(header_words(&h)), "w0 0x6 w1 0x2a w3 0x35e root 0xffffffff w13 0x8c80");
        assert_eq!(header_words(&h[..8]), [6, 42, 0, 0, 0], "a short header has no root");
        let z = vec![0u8; crate::gpu::bvh::HEADER_BYTES];
        // 06 | 2a | 5e 03 | ff ff ff ff | 80 8c
        assert_eq!(differing(&h, &z), 10);
        // a structure built by XCL's own builder: the root the watch reads is the builder's
        let b = crate::gpu::bvh::build(&crate::gpu::bvh::tests::random_tris(8, 3), 0);
        assert_eq!(header_words(&b.data)[3], b.root);
    }

    #[test]
    fn rt_watch_counts_line_names_records_that_never_launched_and_dispatch_x_with_thread_counts() {
        let mut m = BTreeMap::new();
        let mut a = Counts { pipelines: 1, ..Default::default() };
        a.add(Outcome::Launched);
        a.add(Outcome::PutOff);
        a.add(Outcome::Launched);
        a.add_dispatch_x(5, 0x8000);
        let mut b = Counts { pipelines: 1, ..Default::default() };
        b.add(Outcome::Untranslated);
        b.add(Outcome::Failed);
        b.add_dispatch_x(1038, 0x20);
        b.add_dispatch_x(7, 0x20);
        b.add_dispatch_x(9, 0x8020);
        let c = Counts { pipelines: 1, ..Default::default() };
        m.insert(0x1111_1111_1111_1111u64, a);
        m.insert(0x2222_2222_2222_2222u64, b);
        m.insert(0x3333_3333_3333_3333u64, c);
        let l = counts_line(&m, 52, 4);
        assert!(l.starts_with("builder records: 3 of 52 with a pipeline (3 pipelines), 2 dispatched; dispatches launched 2, put off 1, without a translated pipeline 1, failed 1; put-off dispatches dropped (any pipeline) 4;"), "{l}");
        assert!(l.contains("records never launched: [2222222222222222]"), "{l}");
        assert!(l.contains("records with skips (launched/put off/untranslated/failed): [2222222222222222 0/0/1/1]"), "{l}");
        assert!(l.ends_with("DispatchX 4 calls, 3 with flag 0x20 (thread counts) (record with/calls, largest x, flags): [2222222222222222 3/3 max x 1038 flags [0x20 0x8020]]"), "{l}");
    }

    #[test]
    fn rt_watch_names_the_records_given_an_address_in_a_header() {
        let mut t = BTreeMap::new();
        t.insert(0xaaaa_aaaa_aaaa_aaaau64, vec![0x1000, 0x5000]);
        t.insert(0xbbbb_bbbb_bbbb_bbbbu64, vec![0x10f0]);
        t.insert(0xcccc_cccc_cccc_ccccu64, vec![0x1100]);
        assert_eq!(given(&t, 0x1000, 256), vec![0xaaaa_aaaa_aaaa_aaaa, 0xbbbb_bbbb_bbbb_bbbb], "0x1100 is past the header");
        assert!(given(&t, 0x9000, 256).is_empty());
    }
}
