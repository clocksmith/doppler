// Packed FP8 code decoding with explicit block F32 multiply scales.
// This uses portable F32 arithmetic; it does not require native FP8 instructions.
override WORKGROUP_SIZE: u32 = 64u;
override E4M3FN: bool = true;
struct Uniforms {
    M: u32, N: u32, K: u32, block_n: u32,
    block_k: u32, blocks_k: u32, block_words: u32, _pad: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, WORKGROUP_SIZE>;

fn decode_fp8(code: u32) -> f32 {
    let sign = (code & 128u) << 24u;
    let magnitude = code & 127u;
    if (E4M3FN) {
        if (magnitude == 127u) { return bitcast<f32>(sign | 0x7fc00000u); }
        let exponent = magnitude >> 3u;
        let mantissa = magnitude & 7u;
        if (exponent == 0u) {
            return bitcast<f32>(sign | bitcast<u32>(f32(mantissa) * 0.001953125f));
        }
        return bitcast<f32>(sign | ((exponent + 120u) << 23u) | (mantissa << 20u));
    }
    let exponent = magnitude >> 2u;
    let mantissa = magnitude & 3u;
    if (exponent == 31u) {
        if (mantissa == 0u) { return bitcast<f32>(sign | 0x7f800000u); }
        return bitcast<f32>(sign | 0x7fc00000u);
    }
    if (exponent == 0u) {
        return bitcast<f32>(sign | bitcast<u32>(f32(mantissa) * 0.0000152587890625f));
    }
    return bitcast<f32>(sign | ((exponent + 112u) << 23u) | (mantissa << 21u));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let n = group.x;
    let m = group.y;
    if (n >= u.N || m >= u.M) { return; }
    var sum = 0.0f;
    for (var k = lane; k < u.K; k += WORKGROUP_SIZE) {
        let block = (n / u.block_n) * u.blocks_k + k / u.block_k;
        let base = block * u.block_words;
        let index = (n % u.block_n) * u.block_k + k % u.block_k;
        let code = (weights[base + 1u + index / 4u] >> ((index % 4u) * 8u)) & 255u;
        let weight = decode_fp8(code) * bitcast<f32>(weights[base]);
        sum = fma(input[m * u.K + k], weight, sum);
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) { output[m * u.N + n] = partial[0]; }
}
