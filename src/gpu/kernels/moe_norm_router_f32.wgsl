// Decode post-attention RMSNorm fused with the MoE expert-router and shared-expert
// router projections of one token: one dispatch for rmsnorm.wgsl main (PRE_RESIDUAL,
// OUTPUT_PRENORM) and two matmul_gemv_f32_subgroup.wgsl transposed GEMVs.
//
// Each workgroup recomputes the norm exactly as rmsnorm.wgsl main does with
// NORM_THREADS invocations: per-thread sums of squares over contiguous elements of
// input + residual, the same tree reduction and the refined inverse root, then
// x * inv_rms * weight. Workgroup 0 writes the normalized row and the pre-norm sum.
// Columns follow matmul_gemv_f32_subgroup.wgsl at WORKGROUP_SIZE / COLS_PER_WG: one
// subgroup per column, each lane four-wide fma over k = rank * 4 + n * count * 4,
// subgroupAdd, then ((x + y) + z) + w. Columns 0 .. num_experts - 1 are router
// logits; column num_experts is the shared-expert gate.
enable subgroups;
requires subgroup_id;

override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 4u;
override NORM_THREADS: u32 = 256u;
override HIDDEN_SIZE: u32;
override RMS_NORM_OFFSET: bool = false;
override WEIGHT_IS_F16: bool = false;

struct Uniforms {
    hidden_size: u32,
    num_experts: u32,
    eps: f32,
    workgroups_x: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read> norm_weight: array<u32>;
@group(0) @binding(4) var<storage, read> router_weight: array<f32>;
@group(0) @binding(5) var<storage, read> shared_router_weight: array<f32>;
@group(0) @binding(6) var<storage, read_write> normed: array<f32>;
@group(0) @binding(7) var<storage, read_write> prenorm: array<f32>;
@group(0) @binding(8) var<storage, read_write> logits: array<f32>;
@group(0) @binding(9) var<storage, read_write> shared_gate: array<f32>;

var<workgroup> partial_sums: array<f32, NORM_THREADS>;
var<workgroup> row: array<f32, HIDDEN_SIZE>;

fn apply_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    }
    return w;
}

fn load_weight(idx: u32) -> f32 {
    if (WEIGHT_IS_F16) {
        let pair = unpack2x16float(norm_weight[idx >> 1u]);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(norm_weight[idx]);
}

fn load_input(idx: u32) -> f32 {
    return input[idx] + residual[idx];
}

fn row_four(index: u32) -> vec4<f32> {
    return vec4<f32>(row[index], row[index + 1u], row[index + 2u], row[index + 3u]);
}

fn weight_value(is_shared: bool, index: u32) -> f32 {
    if (is_shared) {
        return shared_router_weight[index];
    }
    return router_weight[index];
}

fn weight_four(is_shared: bool, index: u32) -> vec4<f32> {
    return vec4<f32>(weight_value(is_shared, index), weight_value(is_shared, index + 1u),
        weight_value(is_shared, index + 2u), weight_value(is_shared, index + 3u));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(subgroup_id) subgroup: u32,
    @builtin(num_subgroups) subgroup_count: u32
) {
    let size = u.hidden_size;

    // rmsnorm.wgsl main with NORM_THREADS threads: thread t sums its contiguous elements.
    let elements_per_thread = (size + NORM_THREADS - 1u) / NORM_THREADS;
    for (var thread_idx = local; thread_idx < NORM_THREADS; thread_idx += WORKGROUP_SIZE) {
        var local_sum_sq: f32 = 0.0;
        for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
            let idx = thread_idx * elements_per_thread + i;
            if (idx < size) {
                let x = load_input(idx);
                local_sum_sq = local_sum_sq + x * x;
            }
        }
        partial_sums[thread_idx] = local_sum_sq;
    }
    workgroupBarrier();

    for (var stride: u32 = NORM_THREADS / 2u; stride > 0u; stride = stride >> 1u) {
        for (var thread_idx = local; thread_idx < stride; thread_idx += WORKGROUP_SIZE) {
            partial_sums[thread_idx] = partial_sums[thread_idx] + partial_sums[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let mean_sq = partial_sums[0] / f32(size);
    let epsilon_sum = mean_sq + u.eps;
    let estimate = inverseSqrt(epsilon_sum);
    let estimate_squared = estimate * estimate;
    let square_error = fma(estimate, estimate, -estimate_squared);
    let reciprocal_residual = fma(-epsilon_sum, estimate_squared, 1.0)
        - epsilon_sum * square_error;
    let inv_rms = fma(0.5 * estimate, reciprocal_residual, estimate);

    let writes_row = group.x == 0u && group.y == 0u;
    for (var idx = local; idx < size; idx += WORKGROUP_SIZE) {
        let x = load_input(idx);
        let result = x * inv_rms * apply_weight(load_weight(idx));
        row[idx] = result;
        if (writes_row) {
            prenorm[idx] = x;
            normed[idx] = result;
        }
    }
    workgroupBarrier();

    // matmul_gemv_f32_subgroup.wgsl, transposed, over num_experts + 1 columns.
    let active_count = subgroupAdd(1u);
    let active_rank = subgroupExclusiveAdd(1u);
    let writer = subgroupElect();
    let columns = u.num_experts + 1u;
    let group_base = (group.y * u.workgroups_x + group.x) * COLS_PER_WG;
    let batches = (COLS_PER_WG + subgroup_count - 1u) / subgroup_count;
    for (var batch = 0u; batch < batches; batch++) {
        let local_col = batch * subgroup_count + subgroup;
        let col = group_base + local_col;
        let is_shared = col == u.num_experts;
        let weight_base = select(col * size, 0u, is_shared);
        var sums = vec4<f32>(0.0f);
        if (local_col < COLS_PER_WG && col < columns) {
            for (var k = active_rank * 4u; k < size; k += active_count * 4u) {
                if (k + 3u < size) {
                    sums = fma(row_four(k), weight_four(is_shared, weight_base + k), sums);
                } else {
                    for (var component = 0u; component < 4u; component++) {
                        if (k + component < size) {
                            sums[component] = fma(row[k + component], weight_value(is_shared, weight_base + k + component), sums[component]);
                        }
                    }
                }
            }
        }
        let total = subgroupAdd(sums);
        if (writer && local_col < COLS_PER_WG && col < columns) {
            let value = (total.x + total.y) + total.z + total.w;
            if (is_shared) {
                shared_gate[0] = value;
            } else {
                logits[col] = value;
            }
        }
    }
}

// Single pass (variant f32_single_pass): every workgroup stages input + residual in
// workgroup memory four-wide and streams its ROWS router rows four-wide across
// invocations. The sum of squares is then taken from the staged row in rmsnorm.wgsl
// main's contiguous NORM_THREADS order, so the normalized row and pre-norm sum are
// rmsnorm's bits. Each row's dot is taken against input * weight and scaled by the
// inverse root at the end (Pooled engine/wgsl/moe.js normRouterKernel): the logits round
// differently from a GEMV of the normalized row. Row num_experts is the shared-expert gate.
override ROWS: u32 = 4u;
var<workgroup> row_sums: array<vec4<f32>, NORM_THREADS>;

fn row_weight(row: u32, idx: u32) -> f32 {
    if (row == u.num_experts) {
        return shared_router_weight[idx];
    }
    return router_weight[row * u.hidden_size + idx];
}

@compute @workgroup_size(NORM_THREADS, 1, 1)
fn single_pass(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) thread_idx: u32) {
    let size = u.hidden_size;
    let row0 = (group.y * u.workgroups_x + group.x) * ROWS;
    var dots = vec4<f32>(0.0);
    for (var c = thread_idx; c < size / 4u; c += NORM_THREADS) {
        let j = c * 4u;
        let x = vec4<f32>(input[j], input[j + 1u], input[j + 2u], input[j + 3u])
            + vec4<f32>(residual[j], residual[j + 1u], residual[j + 2u], residual[j + 3u]);
        row[j] = x.x;
        row[j + 1u] = x.y;
        row[j + 2u] = x.z;
        row[j + 3u] = x.w;
        let xw = x * vec4<f32>(apply_weight(load_weight(j)), apply_weight(load_weight(j + 1u)),
            apply_weight(load_weight(j + 2u)), apply_weight(load_weight(j + 3u)));
        for (var r = 0u; r < ROWS; r++) {
            let rr = row0 + r;
            if (rr <= u.num_experts) {
                dots[r] += dot(vec4<f32>(row_weight(rr, j), row_weight(rr, j + 1u),
                    row_weight(rr, j + 2u), row_weight(rr, j + 3u)), xw);
            }
        }
    }
    workgroupBarrier();
    let elements_per_thread = (size + NORM_THREADS - 1u) / NORM_THREADS;
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = row[idx];
            local_sum_sq = local_sum_sq + x * x;
        }
    }
    partial_sums[thread_idx] = local_sum_sq;
    row_sums[thread_idx] = dots;
    workgroupBarrier();
    for (var stride: u32 = NORM_THREADS / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            partial_sums[thread_idx] = partial_sums[thread_idx] + partial_sums[thread_idx + stride];
            row_sums[thread_idx] = row_sums[thread_idx] + row_sums[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let mean_sq = partial_sums[0] / f32(size);
    let epsilon_sum = mean_sq + u.eps;
    let estimate = inverseSqrt(epsilon_sum);
    let estimate_squared = estimate * estimate;
    let square_error = fma(estimate, estimate, -estimate_squared);
    let reciprocal_residual = fma(-epsilon_sum, estimate_squared, 1.0)
        - epsilon_sum * square_error;
    let inv_rms = fma(0.5 * estimate, reciprocal_residual, estimate);

    if (thread_idx < ROWS) {
        let rr = row0 + thread_idx;
        let value = row_sums[0][thread_idx] * inv_rms;
        if (rr < u.num_experts) {
            logits[rr] = value;
        } else if (rr == u.num_experts) {
            shared_gate[0] = value;
        }
    }
    if (group.x == 0u && group.y == 0u) {
        for (var idx = thread_idx; idx < size; idx += NORM_THREADS) {
            let x = row[idx];
            prenorm[idx] = x;
            normed[idx] = x * inv_rms * apply_weight(load_weight(idx));
        }
    }
}
