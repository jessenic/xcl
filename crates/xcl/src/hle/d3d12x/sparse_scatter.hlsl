// Lab, XCL_SPARSE_SCATTER=1 (sparse.rs, `upload_dirty_with`): the changed runs of one sparse upload
// slot go from the staging ring into a sparse window by this dispatch instead of one
// CopyBufferRegion each. One group a run, 16 bytes a thread a step; every offset and length is a
// multiple of 16 (the caller checks), so the window receives exactly the bytes of the runs.

ByteAddressBuffer staging : register(t0);   // the upload slot's staging ring
ByteAddressBuffer runs : register(t1);      // 16 bytes a run: staging offset, window offset, bytes, 0
RWByteAddressBuffer window : register(u0);  // the sparse window (its tiles)

cbuffer Config : register(b0) {
    uint first; // the run of group 0 (dispatches go 65535 groups at a time)
};

[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint4 r = runs.Load4((first + group.x) * 16);
    for (uint o = lane * 16; o < r.z; o += 64 * 16) {
        window.Store4(r.y + o, staging.Load4(r.x + o));
    }
}
