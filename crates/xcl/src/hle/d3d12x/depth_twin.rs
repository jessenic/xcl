//! lab: a depth buffer the console also lets compute shaders write (ALLOW_DEPTH_STENCIL |
//! ALLOW_UNORDERED_ACCESS; the PC refuses the pair, `device_x::pc_resource_desc` keeps the
//! depth buffer) gets a twin texture with ALLOW_UNORDERED_ACCESS for the plane a UAV names:
//! the title's UAVs of the depth buffer are made of the twin, and after every compute dispatch
//! whose descriptor tables name such a view the twin is copied into the depth buffer's plane
//! (`list_x::depth_twin_flush`; the title moves these buffers through no state of its own).
//!
//! A retail title writes its material masks into the STENCIL of a 2560x1440 D16S8 buffer with a
//! compute pass (a X24_TYPELESS_G8_UINT UAV) and its lighting passes test / read that stencil;
//! it also blits cached local-light shadow maps into D16 buffers with compute (R16_UNORM UAVs).
//! With null views all of that was dropped. `XCL_DEPTH_TWIN=0`: off; `=s`: stencil twins only.
//!
//! `XCL_SRGB_TWIN=1` (lab, off by default): the same for a render target in R8G8B8A8_UNORM_SRGB
//! that the console also gives an R8G8B8A8_UNORM UAV (the PC has no UAV of an sRGB resource, so
//! `device_x::pc_resource_desc` drops the flag and the view was a null one). The retail title's
//! hair resolve (`hair_resolve:CS_ResolveToGBufferTiled_N`) writes the G-buffer's albedo that
//! way, and its depth through an R32_FLOAT_X8X24 UAV of the D32S8 buffer
//! (`XCL_DEPTH_TWIN_PLANE0=3`). The twin is copied from the target before such a dispatch and
//! back after it, and only once a barrier of the title has told us the target's state.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

#[derive(Clone, Copy)]
pub struct Info {
    pub twin: usize,
    /// stencil twins: the buffer the plane goes through (0: none, the formats are the same)
    pub staging: usize,
    /// subresources of the plane (mips x array), and the plane's first subresource in the depth buffer
    pub subs: u32,
    pub first: u32,
    pub fp_twin: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    pub fp_plane: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    /// lab: the depth plane of a console D16S8 buffer (R24G8 on the PC; an R16_UNORM twin): a DSV
    /// of the depth buffer for the pass that writes the twin's values as depth (0: no such pass)
    pub dsv: usize,
    /// lab: the plane is copied into the twin before a dispatch that names it (the depth of a
    /// D32S8 buffer: a pass may write only some texels)
    pub copy_in: bool,
    /// lab (`XCL_SRGB_TWIN`): the twin of a render target; its state is only what a barrier of
    /// the title left (`state_known`), never taken to be DEPTH_WRITE
    pub target: bool,
}

struct Twin {
    _res: ID3D12Resource,
    _staging: Option<ID3D12Resource>,
    _dsv_heap: Option<ID3D12DescriptorHeap>,
    shape: (u64, u32, u16, u16, i32),
    info: Info,
}
unsafe impl Send for Twin {}

/// (depth buffer, plane) → its twin
static TWINS: Mutex<Option<HashMap<(usize, u32), Twin>>> = Mutex::new(None);
static ANY: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicU64 = AtomicU64::new(0);
/// image descriptor word 0 (the slot's PC heap index) → (depth buffer, plane) its view is of
static VIEWS: Mutex<Option<HashMap<u32, (usize, u32)>>> = Mutex::new(None);
/// (depth buffer, subresource; u32::MAX: all) → the PC state the title's last barrier left
static STATES: Mutex<Option<HashMap<(usize, u32), u32>>> = Mutex::new(None);

fn mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("XCL_DEPTH_TWIN").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("s") => 1,
        _ => 2,
    })
}

/// lab: twins of the depth plane of two-plane buffers. `XCL_DEPTH_TWIN_PLANE0`: bit 0 (1) the
/// console's D16S8 (R24G8), bit 1 (2) D32S8 (R32G8X24); default 1.
fn plane0_bits() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("XCL_DEPTH_TWIN_PLANE0").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1))
}

/// lab (`XCL_SRGB_TWIN`): twins for the UNORM UAVs of sRGB render targets. 1: copied only once a
/// barrier of the title has told the target's state; 2: a target no barrier has named is taken to
/// be in RENDER_TARGET (the retail title names none of its twinned resources in a barrier).
fn srgb_mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("XCL_SRGB_TWIN").ok().and_then(|v| v.trim().parse().ok()).filter(|m| *m <= 2).unwrap_or(0))
}

fn srgb_on() -> bool {
    srgb_mode() != 0
}

/// The state of a render target with a twin for the copies around a dispatch: what a barrier
/// left (`known`), else RENDER_TARGET in mode 2, else None (the copies are not made).
fn target_state_from(known: Option<u32>, mode: u8) -> Option<u32> {
    known.or(if mode == 2 { Some(0x4) } else { None })
}

/// The PC state subresource `sub` of `res` is in for the copies between it and its twin (None: a
/// render target whose state is not known - no copies).
pub fn copy_state(info: &Info, res: usize, sub: u32) -> Option<u32> {
    if info.target { target_state_from(state_known(res, sub), srgb_mode()) } else { Some(state(res, sub)) }
}

/// lab: the first three times each thing happens to the lab's own twins (the depth of a D32S8
/// buffer, an sRGB target): which dispatches name them and whether the copies were made.
pub fn note_use(info: &Info, res: usize, plane: u32, what: &'static str) {
    if !info.copy_in {
        return;
    }
    static SEEN: Mutex<Option<HashMap<(usize, u32, &'static str), u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap();
    let n = g.get_or_insert_with(HashMap::new).entry((res, plane, what)).or_insert(0);
    *n += 1;
    if *n <= 3 {
        crate::xlog_info!("lab: twin use: {what} #{n} of {res:#x} plane {plane} ({}x{}, {}) at {:.1} s", info.fp_twin.Footprint.Width, info.fp_twin.Footprint.Height, if info.target { "sRGB target" } else { "D32S8 depth" }, crate::log::uptime());
    }
}

/// The plane and the twin's format for a UAV in `view` (the PC's format) of a resource in
/// `format` with one subresource a plane or not (`single`): None when there is no twin for it.
/// `mode`: `XCL_DEPTH_TWIN` (0 off, 1 stencil only, 2 all); `plane0`: `XCL_DEPTH_TWIN_PLANE0`.
fn twin_plane(format: i32, view: i32, single: bool, mode: u8, plane0: u32, srgb: bool) -> Option<(u32, DXGI_FORMAT)> {
    // R32_TYPELESS / R16_TYPELESS (D32, D16): the one plane, a twin in the same format.
    // R24G8_TYPELESS / R32G8X24_TYPELESS: the stencil (views X24_TYPELESS_G8_UINT /
    // X32_TYPELESS_G8X24_UINT), an R8_UINT twin; a UAV of their depth stays a null view.
    match (format, view) {
        _ if mode == 0 => None,
        (39 | 53, _) if mode == 2 => Some((0, DXGI_FORMAT(format))),
        (44, 47) | (19, 22) if single => Some((1, DXGI_FORMAT_R8_UINT)),
        // lab: the depth of a two-plane buffer. R24G8 (the console's D16S8: a retail title's material
        // permutation, z = n / 65535, which its tile-lighting draws EQUAL-test): an R16_UNORM twin,
        // written into the depth plane by a full-screen pass. R32G8X24: an R32_FLOAT twin, through a buffer.
        (44, 46) if mode == 2 && plane0 & 1 != 0 && single => Some((0, DXGI_FORMAT_R16_UNORM)),
        (19, 21) if mode == 2 && plane0 & 2 != 0 && single => Some((0, DXGI_FORMAT_R32_FLOAT)),
        // lab: an sRGB render target's UNORM UAV (the same bytes: no conversion on a store)
        (29, 28) if mode == 2 && srgb && single => Some((0, DXGI_FORMAT_R8G8B8A8_UNORM)),
        _ => None,
    }
}

/// The format of views of the twin of plane 0 of a depth buffer in `format` (None: the title's).
pub fn plane0_view_format(format: i32) -> Option<DXGI_FORMAT> {
    match format {
        44 => Some(DXGI_FORMAT_R16_UNORM),
        19 => Some(DXGI_FORMAT_R32_FLOAT),
        _ => None,
    }
}

/// lab: (root signature, pipeline) of the pass that writes an R16_UNORM twin's texels (in a buffer:
/// rows of `pitch` bytes) as the depth of a D24S8 buffer; the stencil is not touched.
pub unsafe fn depth_pass(dev: &ID3D12Device) -> Option<(usize, usize)> {
    static P: Mutex<Option<Option<(usize, usize)>>> = Mutex::new(None);
    let mut g = P.lock().unwrap();
    if let Some(p) = *g {
        return p;
    }
    let made = (|| -> Result<(usize, usize), String> {
        let vs = crate::gpu::recomp::dxc::compile_cached("float4 main(uint id : SV_VertexID) : SV_Position { return float4(id == 2u ? 3.0 : -1.0, id == 1u ? 3.0 : -1.0, 0.0, 1.0); }\n", "main", "vs_6_0")?.0;
        let ps = crate::gpu::recomp::dxc::compile_cached(
            "ByteAddressBuffer src : register(t0);\ncbuffer C : register(b0) { uint pitch; };\nfloat main(float4 pos : SV_Position) : SV_Depth {\n    uint2 p = uint2(pos.xy);\n    uint at = p.y * pitch + p.x * 2u;\n    uint w = src.Load(at & ~3u);\n    uint q = (w >> ((at & 2u) * 8u)) & 0xFFFFu;\n    return (float)q / 65535.0;\n}\n",
            "main",
            "ps_6_0",
        )?
        .0;
        let params = [
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 1 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL },
            D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL },
        ];
        let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), ..Default::default() };
        let mut blob = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).map_err(|e| format!("root signature: {e}"))?;
        let blob = blob.ok_or("root signature blob")?;
        let dev_raw = dev.as_raw();
        let create_rs: unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const windows::core::GUID, *mut *mut c_void) -> i32 = std::mem::transmute(super::real::orig_fn(dev_raw, 16));
        let mut rs_raw = std::ptr::null_mut();
        let hr = create_rs(dev_raw, 0, blob.GetBufferPointer(), blob.GetBufferSize(), &ID3D12RootSignature::IID, &mut rs_raw);
        if hr < 0 || rs_raw.is_null() {
            return Err(format!("CreateRootSignature {:#x}", hr as u32));
        }
        let rs = ID3D12RootSignature::from_raw(rs_raw);
        let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as *const c_void, BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as *const c_void, BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: false.into(), ..Default::default() },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL, DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS, StencilEnable: false.into(), ..Default::default() },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 0,
            DSVFormat: DXGI_FORMAT_D24_UNORM_S8_UINT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let create_pso: unsafe extern "system" fn(*mut c_void, *const D3D12_GRAPHICS_PIPELINE_STATE_DESC, *const windows::core::GUID, *mut *mut c_void) -> i32 = std::mem::transmute(super::real::orig_fn(dev_raw, 10));
        let mut pso_raw = std::ptr::null_mut();
        let hr = create_pso(dev_raw, &pd, &ID3D12PipelineState::IID, &mut pso_raw);
        std::mem::ManuallyDrop::drop(&mut pd.pRootSignature);
        if hr < 0 || pso_raw.is_null() {
            return Err(format!("CreateGraphicsPipelineState {:#x}", hr as u32));
        }
        // (kept for the life of the process)
        Ok((rs.into_raw() as usize, pso_raw as usize))
    })();
    if let Err(e) = &made {
        crate::xlog_warn!("D3D12.X: depth twin: no depth-write pass: {e}");
    }
    *g = Some(made.ok());
    g.unwrap()
}

fn shape(d: &D3D12_RESOURCE_DESC) -> (u64, u32, u16, u16, i32) {
    (d.Width, d.Height, d.DepthOrArraySize, d.MipLevels, d.Format.0)
}

pub fn any() -> bool {
    ANY.load(Ordering::Relaxed)
}

/// The twin of the plane of the depth buffer `res` that a UAV in `view_format` (the PC's) names,
/// made now if there is none, and that plane: None when there cannot be one.
pub unsafe fn get_or_make(res: *mut c_void, view_format: i32) -> Option<(*mut c_void, u32)> {
    if mode() == 0 {
        return None;
    }
    let r = ID3D12Resource::from_raw_borrowed(&res)?;
    let d = r.GetDesc();
    if d.SampleDesc.Count != 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        return None;
    }
    let (plane, twin_format) = twin_plane(d.Format.0, view_format, d.DepthOrArraySize == 1 && d.MipLevels == 1, mode(), plane0_bits(), srgb_on())?;
    let target = d.Format.0 == 29;
    let mut g = TWINS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(t) = m.get(&(res as usize, plane)) {
        if t.shape == shape(&d) {
            return Some((t.info.twin as *mut c_void, plane));
        }
    }
    let mut dev: Option<ID3D12Device> = None;
    r.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let td = D3D12_RESOURCE_DESC { Alignment: 0, Format: twin_format, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..d };
    let mut twin: Option<ID3D12Resource> = None;
    let made = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut twin);
    let Some(twin) = twin.filter(|_| made.is_ok()) else {
        crate::xlog_warn!("D3D12.X: depth twin: no texture for {}x{}x{} format {}: {made:?}", d.Width, d.Height, d.DepthOrArraySize, d.Format.0);
        return None;
    };
    let subs = d.DepthOrArraySize as u32 * d.MipLevels as u32;
    let first = plane * subs;
    let (mut fp_twin, mut fp_plane) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default());
    let (mut total_twin, mut total_plane) = (0u64, 0u64);
    dev.GetCopyableFootprints(&td, 0, 1, 0, Some(&mut fp_twin), None, None, Some(&mut total_twin));
    dev.GetCopyableFootprints(&d, first, 1, 0, Some(&mut fp_plane), None, None, Some(&mut total_plane));
    let mut staging: Option<ID3D12Resource> = None;
    let pass = plane == 0 && d.Format.0 == 44;
    if plane == 1 || twin_format != d.Format {
        if !pass && (fp_twin.Footprint.RowPitch != fp_plane.Footprint.RowPitch || total_plane == 0) {
            crate::xlog_warn!("D3D12.X: depth twin: stencil footprints differ ({:?} / {:?})", fp_twin.Footprint, fp_plane.Footprint);
            return None;
        }
        let bd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total_twin.max(total_plane).max(256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let made = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut staging);
        if made.is_err() || staging.is_none() {
            crate::xlog_warn!("D3D12.X: depth twin: no staging buffer: {made:?}");
            return None;
        }
    }
    // lab: the DSV of the depth-write pass
    let mut dsv_heap: Option<ID3D12DescriptorHeap> = None;
    let mut dsv = 0usize;
    if pass {
        let dev_raw = dev.as_raw();
        let hd = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 };
        let create_heap: unsafe extern "system" fn(*mut c_void, *const D3D12_DESCRIPTOR_HEAP_DESC, *const windows::core::GUID, *mut *mut c_void) -> i32 = std::mem::transmute(super::real::orig_fn(dev_raw, 14));
        let create_dsv: unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_DEPTH_STENCIL_VIEW_DESC, usize) = std::mem::transmute(super::real::orig_fn(dev_raw, 21));
        let mut raw = std::ptr::null_mut();
        if create_heap(dev_raw, &hd, &ID3D12DescriptorHeap::IID, &mut raw) < 0 || raw.is_null() {
            crate::xlog_warn!("D3D12.X: depth twin: no DSV heap");
            return None;
        }
        let heap = ID3D12DescriptorHeap::from_raw(raw);
        dsv = heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let vd = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D24_UNORM_S8_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, Flags: D3D12_DSV_FLAG_NONE, ..Default::default() };
        create_dsv(dev_raw, res, &vd, dsv);
        dsv_heap = Some(heap);
    }
    let texel = match twin_format.0 {
        28 | 39 | 41 => 4,
        53 | 56 => 2,
        _ => 1,
    };
    let bytes = d.Width * d.Height as u64 * subs as u64 * texel;
    let total = BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let info = Info { twin: twin.as_raw() as usize, staging: staging.as_ref().map_or(0, |s| s.as_raw() as usize), subs, first, fp_twin, fp_plane, dsv, copy_in: plane == 0 && matches!(d.Format.0, 19 | 29), target };
    m.insert((res as usize, plane), Twin { _res: twin, _staging: staging, _dsv_heap: dsv_heap, shape: shape(&d), info });
    ANY.store(true, Ordering::Relaxed);
    if m.len() <= 8 || m.len().is_power_of_two() || plane == 1 {
        crate::xlog_info!(
            "D3D12.X: depth twin #{} for {res:p} plane {plane}: {}x{}x{} format {} ({} MB in all; footprint formats {} / {}, row pitch {})",
            m.len(),
            d.Width,
            d.Height,
            d.DepthOrArraySize,
            d.Format.0,
            total >> 20,
            fp_twin.Footprint.Format.0,
            fp_plane.Footprint.Format.0,
            fp_plane.Footprint.RowPitch
        );
    }
    Some((info.twin as *mut c_void, plane))
}

pub fn info(res: usize, plane: u32) -> Option<Info> {
    TWINS.lock().unwrap().as_ref()?.get(&(res, plane)).map(|t| t.info)
}

pub fn has(res: usize) -> bool {
    any() && TWINS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&(res, 0)) || m.contains_key(&(res, 1)))
}

/// The PC state that stands for UNORDERED_ACCESS on a resource with a twin (the twin is the one
/// written): DEPTH_WRITE for a depth buffer, as `device_x::pc_initial_state` creates it; for a
/// render target (`XCL_SRGB_TWIN`) NON_PIXEL_SHADER_RESOURCE - DEPTH_WRITE is no state of a
/// colour texture.
pub fn ua_stand_in(res: usize) -> u32 {
    stand_in(TWINS.lock().unwrap().as_ref().and_then(|m| m.get(&(res, 0))).is_some_and(|t| t.info.target))
}

fn stand_in(target: bool) -> u32 {
    if target { 0x40 } else { 0x10 }
}

/// A view of the twin of (`res`, `plane`) is in the console slot whose descriptor starts with `w0`.
pub fn note_view(w0: u32, res: usize, plane: u32) {
    VIEWS.lock().unwrap().get_or_insert_with(HashMap::new).insert(w0, (res, plane));
}

/// The console slot whose image descriptor starts with `w0` is given another view.
pub fn forget_view(w0: u32) {
    if any() {
        if let Some(m) = VIEWS.lock().unwrap().as_mut() {
            m.remove(&w0);
        }
    }
}

/// (depth buffer, plane) whose twin a descriptor starting with `w0` names.
pub fn by_word0(w0: u32) -> Option<(usize, u32)> {
    VIEWS.lock().unwrap().as_ref()?.get(&w0).copied()
}

/// The title's barrier leaves subresource `sub` (u32::MAX: all) of `res` in the PC state `state`.
pub fn set_state(res: usize, sub: u32, state: u32) {
    let mut g = STATES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if sub == u32::MAX {
        m.retain(|k, _| k.0 != res);
    }
    m.insert((res, sub), state);
}

/// The PC state a barrier of the title left subresource `sub` of `res` in (None: it has not
/// moved it since the twin was made).
pub fn state_known(res: usize, sub: u32) -> Option<u32> {
    let g = STATES.lock().unwrap();
    g.as_ref().and_then(|m| m.get(&(res, sub)).or_else(|| m.get(&(res, u32::MAX))).copied())
}

/// The PC state of subresource `sub` of `res` as the title's barriers left it (DEPTH_WRITE
/// when it never moved it: what `device_x::pc_initial_state` creates these in).
pub fn state(res: usize, sub: u32) -> u32 {
    let g = STATES.lock().unwrap();
    g.as_ref().and_then(|m| m.get(&(res, sub)).or_else(|| m.get(&(res, u32::MAX))).copied()).unwrap_or(0x10)
}

// ---------------------------------------------------------------------------------------
// lab: alias pairs. The retail title places an R32_UINT texture with ALLOW_UNORDERED_ACCESS and
// an R9G9B9E5 texture without flags at ONE address: compute writes packed RGB9E5 through the
// UAV, shaders sample the other (its screen-probe radiance atlases, 2048x512 / 2304x576 ...).
// At an address that is not 64 KiB aligned the UAV texture becomes a committed resource
// (`vram::place` takes only aligned ones) and the sampled one a reserved texture on pages
// (`vram::place_pages`): two memories, the sampled view never sees the writes. Here the pair is
// remembered and the UAV texture is copied into the other (through a buffer: the formats are not
// castable) after each compute dispatch that names its UAV (`list_x::depth_twin_flush`, plane 2).
// `XCL_ALIAS_COPY=0`: off.
// ---------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct AliasInfo {
    pub dst: usize,
    pub staging: usize,
    pub src_fp: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    pub dst_fp: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
}

struct Alias {
    _src: ID3D12Resource,
    _dst: ID3D12Resource,
    _staging: ID3D12Resource,
    info: AliasInfo,
}
unsafe impl Send for Alias {}

/// title address → the GPU-owned R32_UINT UAV texture placed there last. The pointers are not
/// references: an entry goes when its texture's last reference does (`alias_forget`, from the
/// resource's Release), so one that is here is alive. (Before, a released source stayed here and
/// `alias_pair` called into the freed object when another texture was placed at its address –
/// CharlyRien's review on #505.)
static ALIAS_SRC: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// The last reference of resource `res` is gone (or held for a late release): it is no alias
/// source any more.
pub fn alias_forget(res: usize) {
    if let Some(m) = ALIAS_SRC.lock().unwrap().as_mut() {
        m.retain(|_, r| *r != res);
    }
}
/// UAV texture → its pair
static ALIASES: Mutex<Option<HashMap<usize, Alias>>> = Mutex::new(None);

fn alias_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_ALIAS_COPY").is_ok_and(|v| v.trim() == "0"))
}

/// A GPU-owned texture was placed at title address `va`.
pub unsafe fn alias_note_gpu_owned(va: usize, res: *mut c_void) {
    if !alias_on() {
        return;
    }
    let Some(r) = ID3D12Resource::from_raw_borrowed(&res) else { return };
    let d = r.GetDesc();
    let mut g = ALIAS_SRC.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if d.Format == DXGI_FORMAT_R32_UINT && d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0 && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        m.insert(va, res as usize);
    } else {
        m.remove(&va);
    }
}

/// A texture without flags was placed at `va`: paired with the R32_UINT UAV texture there when it
/// is an R9G9B9E5 one of the same shape and the two do not share memory.
pub unsafe fn alias_pair(va: usize, res: *mut c_void) {
    if !alias_on() {
        return;
    }
    // (the source gets a reference of ours while `ALIAS_SRC` is held: `alias_forget` cannot pass)
    let (src, s) = {
        let g = ALIAS_SRC.lock().unwrap();
        let Some(&src) = g.as_ref().and_then(|m| m.get(&va)) else { return };
        let src_raw = src as *mut c_void;
        let Some(s) = ID3D12Resource::from_raw_borrowed(&src_raw).cloned() else { return };
        (src, s)
    };
    let Some(d) = ID3D12Resource::from_raw_borrowed(&res) else { return };
    let (dd, sd) = (d.GetDesc(), s.GetDesc());
    if dd.Format != DXGI_FORMAT_R9G9B9E5_SHAREDEXP || (dd.Width, dd.Height, dd.DepthOrArraySize, dd.MipLevels) != (sd.Width, sd.Height, sd.DepthOrArraySize, sd.MipLevels) {
        return;
    }
    // both reserved on the pages of the address: taken as one memory already – but at such an
    // address the R9G9B9E5 view read zeros while the R32_UINT UAV held the title's packed words
    // (claudetools-gh's measurement on xcl-dev/xcl#505: a retail title's half-res DoF input, and
    // with it a view-dependent black scene). LOCAL: paired as well (`XCL_ALIAS_ALIGNED=0`: not).
    static ALIGNED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if va % 0x1_0000 == 0 && !*ALIGNED.get_or_init(|| !std::env::var("XCL_ALIAS_ALIGNED").is_ok_and(|v| v.trim() == "0")) {
        return;
    }
    let mut dev: Option<ID3D12Device> = None;
    if s.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let n = sd.MipLevels as u32 * sd.DepthOrArraySize as u32;
    let mut src_fp = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let mut dst_fp = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let (mut src_total, mut dst_total) = (0u64, 0u64);
    dev.GetCopyableFootprints(&sd, 0, n, 0, Some(src_fp.as_mut_ptr()), None, None, Some(&mut src_total));
    dev.GetCopyableFootprints(&dd, 0, n, 0, Some(dst_fp.as_mut_ptr()), None, None, Some(&mut dst_total));
    if src_total == 0 || src_total != dst_total || src_fp.iter().zip(&dst_fp).any(|(a, b)| a.Offset != b.Offset || a.Footprint.RowPitch != b.Footprint.RowPitch) {
        crate::xlog_warn!("D3D12.X: alias pair at {va:#x}: footprints differ ({src_total} / {dst_total} bytes)");
        return;
    }
    let bd = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: src_total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut staging: Option<ID3D12Resource> = None;
    let made = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut staging);
    let Some(staging) = staging.filter(|_| made.is_ok()) else { return };
    let info = AliasInfo { dst: res as usize, staging: staging.as_raw() as usize, src_fp, dst_fp };
    let mut g = ALIASES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    m.insert(src, Alias { _src: s.clone(), _dst: d.clone(), _staging: staging, info });
    ANY.store(true, Ordering::Relaxed);
    crate::xlog_info!("D3D12.X: alias pair #{} at {va:#x}: R32_UINT UAV texture {src:#x} → R9G9B9E5 texture {res:p}, {}x{}, {} mips ({} KiB a copy)", m.len(), sd.Width, sd.Height, sd.MipLevels, src_total >> 10);
}

/// Whether `res` is an R32_UINT UAV texture placed in title memory (it may be, or become, the
/// UAV texture of an alias pair: its UAVs are followed).
pub fn is_alias_src(res: usize) -> bool {
    alias_on() && ALIAS_SRC.lock().unwrap().as_ref().is_some_and(|m| m.values().any(|&r| r == res))
}

pub fn alias_info(res: usize) -> Option<AliasInfo> {
    ALIASES.lock().unwrap().as_ref()?.get(&res).map(|a| a.info.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An alias source is known only while its texture lives: the entry goes with the last
    /// reference (`alias_forget`, from the resource's Release), and only that texture's.
    #[test]
    fn an_alias_source_is_forgotten_with_its_last_reference() {
        let (a, b) = (0x7E57_A11A_5000usize, 0x7E57_A11A_6000usize);
        {
            let mut g = ALIAS_SRC.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            m.insert(0x7_1000_0000, a);
            m.insert(0x7_2000_0000, b);
        }
        assert!(is_alias_src(a) && is_alias_src(b));
        alias_forget(a);
        assert!(!is_alias_src(a), "a released texture is no source any more");
        assert!(is_alias_src(b), "another texture's entry stays");
        alias_forget(a);
        alias_forget(b);
        assert!(!is_alias_src(b));
    }

    #[test]
    fn twins_by_resource_and_view_format() {
        // as it was: one-plane depth buffers, the stencil of two-plane ones, the D16S8 depth
        assert_eq!(twin_plane(39, 41, false, 2, 1, false), Some((0, DXGI_FORMAT_R32_TYPELESS)));
        assert_eq!(twin_plane(53, 56, false, 2, 1, false), Some((0, DXGI_FORMAT_R16_TYPELESS)));
        assert_eq!(twin_plane(53, 56, false, 1, 1, false), None, "stencil twins only");
        assert_eq!(twin_plane(44, 47, true, 1, 1, false), Some((1, DXGI_FORMAT_R8_UINT)));
        assert_eq!(twin_plane(19, 22, true, 2, 1, false), Some((1, DXGI_FORMAT_R8_UINT)));
        assert_eq!(twin_plane(19, 22, false, 2, 1, false), None, "mips or slices");
        assert_eq!(twin_plane(44, 46, true, 2, 1, false), Some((0, DXGI_FORMAT_R16_UNORM)));
        assert_eq!(twin_plane(44, 46, true, 2, 2, false), None);
        // the D32S8 depth: only with bit 1 of XCL_DEPTH_TWIN_PLANE0
        assert_eq!(twin_plane(19, 21, true, 2, 1, false), None);
        assert_eq!(twin_plane(19, 21, true, 2, 3, false), Some((0, DXGI_FORMAT_R32_FLOAT)));
        // an sRGB target's UNORM UAV: only with XCL_SRGB_TWIN, one subresource, all twins on
        assert_eq!(twin_plane(29, 28, true, 2, 1, false), None);
        assert_eq!(twin_plane(29, 28, true, 2, 1, true), Some((0, DXGI_FORMAT_R8G8B8A8_UNORM)));
        assert_eq!(twin_plane(29, 28, false, 2, 1, true), None, "a mip chain stays a null view");
        assert_eq!(twin_plane(29, 28, true, 1, 1, true), None);
        assert_eq!(twin_plane(29, 29, true, 2, 1, true), None, "an sRGB UAV is not one the PC has");
        assert_eq!(twin_plane(28, 28, true, 2, 1, true), None, "a UNORM target has its own UAV");
        // off
        for (f, v) in [(39, 41), (44, 47), (19, 21), (29, 28)] {
            assert_eq!(twin_plane(f, v, true, 0, 3, true), None);
        }
    }

    #[test]
    fn a_target_state_is_unknown_until_a_barrier() {
        let res = 0x7A11_0000usize;
        assert_eq!(state_known(res, 0), None);
        assert_eq!(state(res, 0), 0x10, "a depth buffer the title never moved: DEPTH_WRITE");
        set_state(res, u32::MAX, 0x4);
        assert_eq!(state_known(res, 0), Some(0x4));
        set_state(res, 0, 0x80);
        assert_eq!((state_known(res, 0), state_known(res, 1)), (Some(0x80), Some(0x4)));
    }

    #[test]
    fn a_target_no_barrier_named_is_copied_only_in_mode_2() {
        assert_eq!(target_state_from(None, 1), None);
        assert_eq!(target_state_from(None, 2), Some(0x4), "taken to be in RENDER_TARGET");
        assert_eq!(target_state_from(Some(0x80), 1), Some(0x80));
        assert_eq!(target_state_from(Some(0x80), 2), Some(0x80), "a barrier's state comes first");
    }

    #[test]
    fn unordered_access_stands_as_a_state_the_resource_can_be_in() {
        assert_eq!(stand_in(false), 0x10, "a depth buffer: DEPTH_WRITE");
        assert_eq!(stand_in(true), 0x40, "a render target: a shader resource, never a depth state");
        assert_eq!(ua_stand_in(0x7A11_0001), 0x10, "no twin of a target: as it was");
    }
}
