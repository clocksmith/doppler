// GeLU Activation Kernel
//
// GeLU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// Variants:
// - geglu: GeLU(gate) * up (separate gate buffer)
// - geglu_rowsplit: input = [numTokens, 2*dim] row-split

override WORKGROUP_SIZE: u32 = 256u;
override HAS_GATE: bool = false;
override USE_ROWSPLIT: bool = false;
override GELU_ERF: bool = false;

struct Uniforms {
    size: u32,          // Total output elements
    rowsplit_dim: u32,  // Dim for rowsplit variants (0 when unused)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;

// Source erf GELU, evaluated in f32 even for f16 storage. The polynomial
// approximates erf rather than substituting the distinct tanh GELU function.
fn erf_gelu(x: f32) -> f32 {
    let z = abs(x) * 0.7071067811865476;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t)
        + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let complement = polynomial * exp(-z * z);
    let cdf_twice = select(complement, 2.0 - complement, x >= 0.0);
    return 0.5 * x * cdf_twice;
}

fn gelu(x: f32) -> f32 {
    if (GELU_ERF) {
        return erf_gelu(x);
    }
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let c: f32 = 0.044715;
    let inner = sqrt_2_over_pi * (x + c * x * x * x);
    let inner_clamped = clamp(inner, -15.0, 15.0);
    return 0.5 * x * (1.0 + tanh(inner_clamped));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (USE_ROWSPLIT) {
        if (u.rowsplit_dim == 0u) {
            return;
        }
        let dim = u.rowsplit_dim;
        let token_idx = global_id.y;
        let dim_idx = global_id.x;
        let idx = token_idx * dim + dim_idx;
        if (idx >= u.size || dim_idx >= dim) {
            return;
        }
        let row_base = token_idx * dim * 2u;
        let g = input[row_base + dim_idx];
        let up = input[row_base + dim + dim_idx];
        output[idx] = gelu(g) * up;
        return;
    }

    let idx = global_id.x;
    if (idx >= u.size) {
        return;
    }

    if (HAS_GATE) {
        let up = input[idx];
        let g = gate[idx];
        output[idx] = gelu(g) * up;
        return;
    }

    let x = input[idx];
    output[idx] = gelu(x);
}
