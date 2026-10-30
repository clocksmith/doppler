// Full-F32 GEMV with subgroup-owned outputs and four-component arithmetic.
// Exact scalar bindings support minimal buffers and unaligned row/K/N tails.
enable subgroups;
requires subgroup_id;

override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 4u;

struct Uniforms {
    M: u32, N: u32, K: u32, alpha: f32,
    transpose_b: u32, workgroups_x: u32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

fn input_four(index: u32) -> vec4<f32> {
    return vec4<f32>(A[index], A[index + 1u], A[index + 2u], A[index + 3u]);
}
fn weight_four(index: u32) -> vec4<f32> {
    return vec4<f32>(B[index], B[index + 1u], B[index + 2u], B[index + 3u]);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_id) subgroup: u32,
    @builtin(num_subgroups) subgroup_count: u32
) {
    // Compact active ranks also cover partial subgroups without assuming any
    // relation between subgroup IDs and workgroup-local invocation indices.
    let active_count = subgroupAdd(1u);
    let active_rank = subgroupExclusiveAdd(1u);
    let writer = subgroupElect();
    let group_base = (group.y * u.workgroups_x + group.x) * COLS_PER_WG;
    let output_width = select(4u, 1u, u.transpose_b != 0u);
    let slots = (COLS_PER_WG + output_width - 1u) / output_width;
    let batches = (slots + subgroup_count - 1u) / subgroup_count;
    for (var batch = 0u; batch < batches; batch++) {
        let local_col = (batch * subgroup_count + subgroup) * output_width;
        let col = group_base + local_col;
        var sums = vec4<f32>(0.0f);
        if (local_col < COLS_PER_WG && col < u.N) {
            if (u.transpose_b != 0u) {
                for (var k = active_rank * 4u; k < u.K; k += active_count * 4u) {
                    if (k + 3u < u.K) {
                        sums = fma(input_four(k), weight_four(col * u.K + k), sums);
                    } else {
                        for (var component = 0u; component < 4u; component++) {
                            if (k + component < u.K) {
                                sums[component] = fma(A[k + component], B[col * u.K + k + component], sums[component]);
                            }
                        }
                    }
                }
            } else {
                for (var k = active_rank; k < u.K; k += active_count) {
                    if (local_col + 3u < COLS_PER_WG && col + 3u < u.N) {
                        sums = fma(vec4<f32>(A[k]), weight_four(k * u.N + col), sums);
                    } else {
                        for (var component = 0u; component < 4u; component++) {
                            if (local_col + component < COLS_PER_WG && col + component < u.N) {
                                sums[component] = fma(A[k], B[k * u.N + col + component], sums[component]);
                            }
                        }
                    }
                }
            }
        }
        let total = subgroupAdd(sums);
        if (writer && local_col < COLS_PER_WG && col < u.N) {
            if (u.transpose_b != 0u) {
                C[col] = ((total.x + total.y) + total.z + total.w) * u.alpha;
            } else {
                for (var component = 0u; component < 4u; component++) {
                    if (local_col + component < COLS_PER_WG && col + component < u.N) {
                        C[col + component] = total[component] * u.alpha;
                    }
                }
            }
        }
    }
}
