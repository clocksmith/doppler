// Full-F32 GEMV: independent output rows share a workgroup, each reduced
// cooperatively across K. No f16 or subgroup capability is required.

override THREADS_PER_COL: u32 = 32u;
override COLS_PER_WG: u32 = 4u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transpose_b: u32,
    workgroups_x: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> partials: array<f32, THREADS_PER_COL * COLS_PER_WG>;

@compute @workgroup_size(THREADS_PER_COL, COLS_PER_WG, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let lane = local_id.x;
    let local_col = local_id.y;
    let group_linear = group_id.y * u.workgroups_x + group_id.x;
    let col = group_linear * COLS_PER_WG + local_col;
    let base = local_col * THREADS_PER_COL;
    var sum = 0.0;
    if (col < u.N) {
        for (var k = lane; k < u.K; k += THREADS_PER_COL) {
            if (u.transpose_b != 0u) {
                sum += A[k] * B[col * u.K + k];
            } else {
                sum += A[k] * B[k * u.N + col];
            }
        }
    }
    partials[base + lane] = sum;
    workgroupBarrier();

    // Retain an unpaired value at each level, including odd specializations.
    // Tail columns participate in every barrier but never read or write storage.
    var reduction_count = THREADS_PER_COL;
    loop {
        if (reduction_count <= 1u) { break; }
        let next = (reduction_count + 1u) / 2u;
        if (lane < reduction_count - next) {
            partials[base + lane] += partials[base + lane + next];
        }
        workgroupBarrier();
        reduction_count = next;
    }
    if (lane == 0u && col < u.N) {
        C[col] = partials[base] * u.alpha;
    }
}
