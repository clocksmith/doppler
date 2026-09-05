override WORKGROUP_SIZE: u32 = 64u;
override PREDICTION_LENGTH: u32 = 64u;
override OUTPUT_QUANTILES: u32 = 3u;
struct Request { horizon: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<uniform> request: Request;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> moments: array<f32>;
@group(0) @binding(3) var<storage, read> quantile_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= request.horizon * OUTPUT_QUANTILES) { return; }
    let time = i / OUTPUT_QUANTILES;
    let quantile = quantile_indices[i % OUTPUT_QUANTILES];
    output[i] = input[quantile * PREDICTION_LENGTH + time] * moments[1] + moments[0];
}
