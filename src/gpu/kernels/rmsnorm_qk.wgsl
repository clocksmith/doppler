override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

override RMS_NORM_OFFSET: bool = false;
override Q_WEIGHT_IS_F16: bool = false;
override K_WEIGHT_IS_F16: bool = false;

struct Uniforms {
    q_rows: u32,
    k_rows: u32,
    head_dim: u32,
    row_stride: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> q_input: array<f32>;
@group(0) @binding(2) var<storage, read> q_weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> q_output: array<f32>;
@group(0) @binding(4) var<storage, read> k_input: array<f32>;
@group(0) @binding(5) var<storage, read> k_weight: array<u32>;
@group(0) @binding(6) var<storage, read_write> k_output: array<f32>;

var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_correction: array<f32, MAX_WORKGROUP_SIZE>;

fn apply_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    }
    return w;
}

fn row_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.y * max(u.row_stride, 1u) + wg_id.x;
}

fn load_value(row: u32, idx: u32) -> f32 {
    if (row < u.q_rows) {
        return q_input[row * u.head_dim + idx];
    }
    let k_row = row - u.q_rows;
    return k_input[k_row * u.head_dim + idx];
}

fn load_weight(row: u32, idx: u32) -> f32 {
    if (row < u.q_rows) {
        if (Q_WEIGHT_IS_F16) {
            let packed = q_weight[idx >> 1u];
            let pair = unpack2x16float(packed);
            return select(pair.x, pair.y, (idx & 1u) == 1u);
        }
        return bitcast<f32>(q_weight[idx]);
    }
    if (K_WEIGHT_IS_F16) {
        let packed = k_weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(k_weight[idx]);
}

fn store_value(row: u32, idx: u32, value: f32) {
    if (row < u.q_rows) {
        q_output[row * u.head_dim + idx] = value;
    } else {
        let k_row = row - u.q_rows;
        k_output[k_row * u.head_dim + idx] = value;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = row_index(wg_id);
    let total_rows = u.q_rows + u.k_rows;
    if (row >= total_rows) {
        return;
    }

    let thread_idx = local_id.x;
    let size = u.head_dim;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Keep rounded sums observable before recovering their low parts. Without
    // the shared-memory boundary, compiler reassociation can erase compensation.
    var local_sum_sq = 0.0;
    var correction = 0.0;
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = thread_idx * elements_per_thread + i;
        var x = 0.0;
        if (idx < size) { x = load_value(row, idx); }
        let square = x * x;
        shared_sum[thread_idx] = local_sum_sq + square;
        workgroupBarrier();
        let rounded = shared_sum[thread_idx];
        let error = fma(-1.0, rounded, max(local_sum_sq, square)) + min(local_sum_sq, square);
        correction += fma(x, x, -square) + error;
        local_sum_sq = rounded;
    }

    shared_sum[thread_idx] = local_sum_sq;
    shared_correction[thread_idx] = correction;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        let a = shared_sum[thread_idx];
        let b = shared_sum[min(thread_idx + stride, WORKGROUP_SIZE - 1u)];
        if (thread_idx < stride) { shared_sum[thread_idx] = a + b; }
        workgroupBarrier();
        if (thread_idx < stride) {
            let rounded = shared_sum[thread_idx];
            let error = fma(-1.0, rounded, max(a, b)) + min(a, b);
            shared_correction[thread_idx] += shared_correction[thread_idx + stride] + error;
        }
        workgroupBarrier();
    }

    let mean_sq = (shared_sum[0] + shared_correction[0]) / f32(size);
    let epsilon_sum = mean_sq + u.eps;
    let estimate = inverseSqrt(epsilon_sum);
    let estimate_squared = estimate * estimate;
    let square_error = fma(estimate, estimate, -estimate_squared);
    let reciprocal_residual = fma(-epsilon_sum, estimate_squared, 1.0)
        - epsilon_sum * square_error;
    let inv_rms = fma(0.5 * estimate, reciprocal_residual, estimate);

    workgroupBarrier();

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let result = load_value(row, idx) * inv_rms * apply_weight(load_weight(row, idx));
            store_value(row, idx, result);
        }
    }
}
