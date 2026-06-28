override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

override RMS_NORM_OFFSET: bool = false;
override Q_WEIGHT_IS_F16: bool = false;
override K_WEIGHT_IS_F16: bool = false;

struct Uniforms {
    num_tokens: u32,
    q_size: u32,
    k_size: u32,
    v_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    workgroup_stride: u32,
    qk_rows: u32,
    total_v: u32,
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> fused_qkv: array<f32>;
@group(0) @binding(2) var<storage, read> q_weight: array<u32>;
@group(0) @binding(3) var<storage, read> k_weight: array<u32>;
@group(0) @binding(4) var<storage, read_write> q_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> k_output: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_output: array<f32>;

var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;

fn flat_group_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.y * max(u.workgroup_stride, 1u) + wg_id.x;
}

fn q_rows() -> u32 {
    return u.num_tokens * u.num_heads;
}

fn qkv_size() -> u32 {
    return u.q_size + u.k_size + u.v_size;
}

fn apply_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    }
    return w;
}

fn load_qk_value(row: u32, idx: u32) -> f32 {
    let q_row_count = q_rows();
    if (row < q_row_count) {
        let token = row / u.num_heads;
        let head = row % u.num_heads;
        return fused_qkv[token * qkv_size() + head * u.head_dim + idx];
    }

    let k_row = row - q_row_count;
    let token = k_row / u.num_kv_heads;
    let head = k_row % u.num_kv_heads;
    return fused_qkv[token * qkv_size() + u.q_size + head * u.head_dim + idx];
}

fn load_weight(row: u32, idx: u32) -> f32 {
    if (row < q_rows()) {
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

fn store_qk_value(row: u32, idx: u32, value: f32) {
    if (row < q_rows()) {
        q_output[row * u.head_dim + idx] = value;
    } else {
        let k_row = row - q_rows();
        k_output[k_row * u.head_dim + idx] = value;
    }
}

fn copy_v(v_group: u32, thread_idx: u32) {
    let v_idx = v_group * WORKGROUP_SIZE + thread_idx;
    if (v_idx >= u.total_v) {
        return;
    }
    let token = v_idx / u.v_size;
    let elem = v_idx % u.v_size;
    let src_idx = token * qkv_size() + u.q_size + u.k_size + elem;
    v_output[v_idx] = fused_qkv[src_idx];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let group_idx = flat_group_index(wg_id);
    let thread_idx = local_id.x;

    if (group_idx >= u.qk_rows) {
        copy_v(group_idx - u.qk_rows, thread_idx);
        return;
    }

    let elements_per_thread = (u.head_dim + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < u.head_dim) {
            let x = load_qk_value(group_idx, idx);
            local_sum_sq = local_sum_sq + x * x;
        }
    }

    shared_sum[thread_idx] = local_sum_sq;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let inv_rms = 1.0 / sqrt(shared_sum[0] / f32(u.head_dim) + u.eps);
    workgroupBarrier();

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < u.head_dim) {
            let result = load_qk_value(group_idx, idx) * inv_rms * apply_weight(load_weight(group_idx, idx));
            store_qk_value(group_idx, idx, result);
        }
    }
}
