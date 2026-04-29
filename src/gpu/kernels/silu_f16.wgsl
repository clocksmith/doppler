// AUTO-GENERATED from src/gpu/kernels/silu.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
//
// F16 variant for reduced memory bandwidth when using F16 activations.
// Intermediate computations use F32 for precision.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;
override HAS_GATE: bool = false;
override GATE_USE_SIGMOID: bool = false;
override INPUT_USE_IDENTITY: bool = false;
override USE_SPLIT: bool = false;
override USE_VEC4: bool = false;
override USE_ROWSPLIT: bool = false;

struct Uniforms {
    size: u32,          // Total output elements
    rowsplit_dim: u32,  // Row-split dim or dispatch stride for non-row-split variants
    clamp_max: f32,     // SwiGLU clamp (0 = disabled)
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(3) var<storage, read> gate: array<f16>;

fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    return 1.0 / (1.0 + exp(-clamped));
}

fn silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

fn apply_input_activation(x: f32) -> f32 {
    return select(silu(x), x, INPUT_USE_IDENTITY);
}

fn clamp_swiglu(x: f32) -> f32 {
    if (u.clamp_max <= 0.0) {
        return x;
    }
    return clamp(x, -u.clamp_max, u.clamp_max);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let dispatch_stride = max(u.rowsplit_dim, 1u);
    if (USE_VEC4) {
        let base_idx = global_id.y * dispatch_stride + global_id.x * 4u;
        if (base_idx >= u.size) {
            return;
        }

        let remaining = min(4u, u.size - base_idx);
        for (var i: u32 = 0u; i < remaining; i = i + 1u) {
            let x = f32(input[base_idx + i]);
            output[base_idx + i] = f16(apply_input_activation(x));
        }
        return;
    }

    let idx = global_id.y * dispatch_stride + global_id.x;
    if (idx >= u.size) {
        return;
    }

    if (USE_ROWSPLIT) {
        if (u.rowsplit_dim == 0u) {
            return;
        }
        let dim = u.rowsplit_dim;
        let num_tokens = u.size / dim;
        let token_idx = global_id.y;
        let dim_idx = global_id.x;
        if (token_idx >= num_tokens || dim_idx >= dim) {
            return;
        }
        let row_base = token_idx * dim * 2u;
        let g = f32(input[row_base + dim_idx]);
        let up = f32(input[row_base + dim + dim_idx]);
        output[token_idx * dim + dim_idx] = f16(clamp_swiglu(silu(g) * up));
        return;
    }

    if (HAS_GATE) {
        let up = f32(input[idx]);
        let g = f32(gate[idx]);
        let gateAct = select(silu(g), sigmoid(g), GATE_USE_SIGMOID);
        output[idx] = f16(clamp_swiglu(gateAct * apply_input_activation(up)));
        return;
    }

    if (USE_SPLIT) {
        let g = f32(input[idx]);
        let up = f32(input[idx + u.size]);
        output[idx] = f16(clamp_swiglu(silu(g) * up));
        return;
    }

    let x = f32(input[idx]);
    output[idx] = f16(apply_input_activation(x));
}
