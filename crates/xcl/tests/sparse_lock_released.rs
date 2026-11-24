//! XCL's queue for the sparse windows (tile mappings, uploads of CPU writes, copies back) can
//! stand still for as long as the GPU does. A thread that waits for it – a `Map` for its copy
//! back (`sparse::readback_now`), a submit for room in the upload ring (`sparse::before_execute`)
//! – must not hold the sparse state's lock meanwhile: the title's other threads (a commit, the
//! next submit) need it, and what they do may be what lets the GPU go on. And what the waits
//! are for still holds: the `Map` returns with the GPU's bytes, the submit with all of its
//! uploads made.

use std::sync::mpsc::channel;
use std::time::Duration;

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use xcl::hle::d3d12x::sparse;

const TILE: usize = 0x1_0000;
const PAGE: usize = 0x1000;
/// more than the upload ring holds (64 MiB)
const BIG: usize = 96 << 20;

/// A commit and a submit on another thread, as the title's other threads make them: whether
/// they were through within 2 s.
unsafe fn others_get_the_lock(k: usize, queue: &ID3D12CommandQueue) -> bool {
    let (tx, rx) = channel();
    let queue = queue.clone();
    let other = std::thread::spawn(move || {
        let _ = sparse::alloc_in(k, TILE, TILE);
        sparse::before_execute(&queue, &[], PAGE);
        let _ = tx.send(());
    });
    let through = rx.recv_timeout(Duration::from_secs(2)).is_ok();
    // (joined by the caller's release of the queue; a thread left behind ends with the process)
    if through {
        other.join().unwrap();
    }
    through
}

#[test]
fn a_wait_for_the_sparse_queue_does_not_hold_the_state() {
    unsafe {
        std::env::set_var("XCL_UMA_MB", "64");
        std::env::set_var("XCL_COHERENT_MB", "64");
        let Ok(dev) = xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            return xcl::gpu::recomp::runner::skip_test("no GPU");
        };
        let windows = sparse::init(&dev);
        let Some(&(k, raw, _)) = windows.first() else {
            return xcl::gpu::recomp::runner::skip_test("no GPU");
        };
        let base = xcl::gpu::uma::get().unwrap().window(k).0;
        let raw = raw as *mut std::ffi::c_void;
        let window = ID3D12Resource::from_raw_borrowed(&raw).unwrap();
        let (a, _) = sparse::alloc_in(k, TILE, TILE).expect("sparse allocation");
        let (big, _) = sparse::alloc_in(k, BIG, TILE).expect("sparse allocation");
        let sparse_queue = sparse::queue().expect("the sparse queue");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let other: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let ev = CreateEventW(None, false, false, None).unwrap();
        let mut value = 0u64;
        let mut wait = |what: &str| {
            value += 1;
            queue.Signal(&fence, value).unwrap();
            fence.SetEventOnCompletion(value, ev).unwrap();
            assert_eq!(WaitForSingleObject(ev, 20_000), WAIT_OBJECT_0, "the title's queue did not finish: {what}");
        };

        // a buffer of the title's (CPU-visible both ways) and a list of its queue
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let buffer = |heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap();
            let r = r.unwrap();
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).unwrap();
            (r, p as *mut u64)
        };
        let (src, src_p) = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let (dst, dst_p) = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let al: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &al, None).unwrap();

        // 1. a Map. The GPU writes 8 bytes at `a` (the CPU's view holds zeros)
        *src_p = 0x1122_3344_5566_7788;
        list.CopyBufferRegion(window, (a - base) as u64, &src, 0, 8);
        list.Close().unwrap();
        sparse::before_execute(&queue, &[], PAGE);
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        wait("the GPU's write");
        // the sparse queue stands still, and a thread of the title's maps the bytes to read them
        let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        sparse_queue.Wait(&gate, 1).unwrap();
        let map = std::thread::spawn(move || {
            sparse::readback_now(a, 8);
            *(a as *const u64)
        });
        std::thread::sleep(Duration::from_millis(500));
        let map_released = others_get_the_lock(k, &other);
        gate.Signal(1).unwrap();
        let read = map.join().unwrap();

        // 2. a submit with more CPU writes than the upload ring holds, the sparse queue still again
        for i in (0..BIG).step_by(8) {
            *((big + i) as *mut u64) = 0xA5A5_0000_0000_0000 | i as u64;
        }
        let dirty: Vec<usize> = (big..big + BIG).step_by(PAGE).collect();
        sparse_queue.Wait(&gate, 2).unwrap();
        let (tx, rx) = channel();
        let q = queue.clone();
        let submit = std::thread::spawn(move || {
            sparse::before_execute(&q, &dirty, PAGE);
            let _ = tx.send(());
        });
        std::thread::sleep(Duration::from_millis(500));
        let submit_released = others_get_the_lock(k, &other);
        let waited = rx.try_recv().is_err();
        gate.Signal(2).unwrap();
        let uploaded = rx.recv_timeout(Duration::from_secs(20)).is_ok();
        assert!(uploaded, "the submit's uploads did not finish once the sparse queue went on");
        submit.join().unwrap();
        // what video memory holds at the start and the end of the block, after the queue's wait
        al.Reset().unwrap();
        list.Reset(&al, None).unwrap();
        list.CopyBufferRegion(&dst, 0, window, (big - base) as u64, 8);
        list.CopyBufferRegion(&dst, 8, window, (big + BIG - 8 - base) as u64, 8);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        wait("the copy of the uploads");
        let _ = CloseHandle(ev);

        assert!(
            map_released && submit_released,
            "a commit and a submit on another thread waited for the sparse state's lock while its holder waited for the sparse queue \
             (through within 2 s – behind a Map: {map_released}, behind a submit: {submit_released})"
        );
        assert_eq!(read, 0x1122_3344_5566_7788, "the Map returned without the GPU's write in the CPU's view");
        assert!(waited, "the submit was to wait for room in the upload ring: the test does not cover it");
        assert_eq!(*dst_p, 0xA5A5_0000_0000_0000, "the first CPU write of the submit is not in video memory");
        assert_eq!(*dst_p.add(1), 0xA5A5_0000_0000_0000 | (BIG - 8) as u64, "the last CPU write of the submit is not in video memory");
    }
}
