// KV Quantization Kernel (int8)
//
// Packs 4 int8 values into u32, with per-token+head scale (f16).

enable f16;

const MAX_WORKGROUP_SIZE: u32 = 256u;
const PACK_FACTOR: u32 = 4u;
const QMAX: f32 = 127.0;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    num_tokens: u32,
    packed_stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input_k: array<f16>;
@group(0) @binding(2) var<storage, read> input_v: array<f16>;
@group(0) @binding(3) var<storage, read_write> output_k: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_v: array<u32>;
@group(0) @binding(5) var<storage, read_write> scales_k: array<f16>;
@group(0) @binding(6) var<storage, read_write> scales_v: array<f16>;

var<workgroup> shared_k: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_v: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_abs_k: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_abs_v: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_qk: array<i32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_qv: array<i32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_scale_k: f32;
var<workgroup> shared_scale_v: f32;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let token_idx = workgroup_id.y;
    let tid = local_id.x;
    let head_dim = u.head_dim;

    if (head_idx >= u.num_kv_heads || token_idx >= u.num_tokens) {
        return;
    }
    if (head_dim > WORKGROUP_SIZE || WORKGROUP_SIZE > MAX_WORKGROUP_SIZE) {
        return;
    }

    let valid = tid < head_dim;
    let token_base = token_idx * u.num_kv_heads + head_idx;
    let input_base = token_base * head_dim;

    if (valid) {
        let idx = input_base + tid;
        let k_val = f32(input_k[idx]);
        let v_val = f32(input_v[idx]);
        shared_k[tid] = k_val;
        shared_v[tid] = v_val;
        shared_abs_k[tid] = abs(k_val);
        shared_abs_v[tid] = abs(v_val);
    }
    workgroupBarrier();

    if (tid == 0u) {
        var max_k: f32 = 0.0;
        var max_v: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d++) {
            max_k = max(max_k, shared_abs_k[d]);
            max_v = max(max_v, shared_abs_v[d]);
        }
        var scale_k = max_k / QMAX;
        var scale_v = max_v / QMAX;
        if (scale_k == 0.0) { scale_k = 1.0; }
        if (scale_v == 0.0) { scale_v = 1.0; }
        shared_scale_k = scale_k;
        shared_scale_v = scale_v;
        let scale_idx = (u.start_pos + token_idx) * u.num_kv_heads + head_idx;
        scales_k[scale_idx] = f16(scale_k);
        scales_v[scale_idx] = f16(scale_v);
    }
    workgroupBarrier();

    if (valid) {
        let scale_k = shared_scale_k;
        let scale_v = shared_scale_v;
        let qk = i32(round(clamp(shared_k[tid] / scale_k, -QMAX, QMAX)));
        let qv = i32(round(clamp(shared_v[tid] / scale_v, -QMAX, QMAX)));
        shared_qk[tid] = qk;
        shared_qv[tid] = qv;
    }
    workgroupBarrier();

    if ((tid % PACK_FACTOR) == 0u) {
        let pack_idx = tid / PACK_FACTOR;
        if (pack_idx < u.packed_stride) {
            var packed_k: u32 = 0u;
            var packed_v: u32 = 0u;
            for (var i: u32 = 0u; i < PACK_FACTOR; i++) {
                let lane = tid + i;
                let qk = select(0, shared_qk[lane], lane < head_dim);
                let qv = select(0, shared_qv[lane], lane < head_dim);
                let uk = u32(qk) & 0xffu;
                let uv = u32(qv) & 0xffu;
                packed_k = packed_k | (uk << (8u * i));
                packed_v = packed_v | (uv << (8u * i));
            }
            let packed_offset = ((u.start_pos + token_idx) * u.num_kv_heads + head_idx) * u.packed_stride + pack_idx;
            output_k[packed_offset] = packed_k;
            output_v[packed_offset] = packed_v;
        }
    }
}
