// GELU Activation Kernel
//
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
// Uses override constants for compile-time feature selection:
// - HAS_GATE: Enable gated variant (GeGLU pattern)
// - LAYOUT: Input layout (0=separate, 1=interleaved, 2=rowsplit)

// =============================================================================
// Override Constants (compile-time configuration)
// =============================================================================

override WORKGROUP_SIZE: u32 = 256u;

// Feature flags
override HAS_GATE: bool = false;      // GELU(gate) * up pattern

// Layout: 0 = separate buffers, 1 = interleaved, 2 = rowsplit
override LAYOUT: u32 = 0u;

// For rowsplit layout: dimension size
override DIM_SIZE: u32 = 0u;

// =============================================================================
// Uniforms (per-dispatch)
// =============================================================================

struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;

// =============================================================================
// Helper Functions
// =============================================================================

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEF: f32 = 0.044715;

fn gelu_fn(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
    let inner_clamped = clamp(inner, -15.0, 15.0);
    return 0.5 * x * (1.0 + tanh(inner_clamped));
}

// =============================================================================
// Main Entry Point
// =============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    var result: f32;

    if (HAS_GATE) {
        if (LAYOUT == 0u) {
            // Separate buffers
            let g = gate[idx];
            let up = input[idx];
            result = gelu_fn(g) * up;
        } else if (LAYOUT == 1u) {
            // Interleaved: [g0, u0, g1, u1, ...]
            let gate_idx = idx * 2u;
            let up_idx = gate_idx + 1u;
            let g = input[gate_idx];
            let up = input[up_idx];
            result = gelu_fn(g) * up;
        } else if (LAYOUT == 2u) {
            // Rowsplit
            let dim = DIM_SIZE;
            let token_idx = idx / dim;
            let dim_idx = idx % dim;
            let row_base = token_idx * dim * 2u;
            let g = input[row_base + dim_idx];
            let up = input[row_base + dim + dim_idx];
            result = gelu_fn(g) * up;
        }
    } else {
        result = gelu_fn(input[idx]);
    }

    output[idx] = result;
}

// =============================================================================
// Batched Entry Point
// =============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_batched(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // GELU is element-wise, same as main
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    var result: f32;

    if (HAS_GATE && LAYOUT == 2u) {
        let dim = DIM_SIZE;
        let token_idx = idx / dim;
        let dim_idx = idx % dim;
        let row_base = token_idx * dim * 2u;
        let g = input[row_base + dim_idx];
        let up = input[row_base + dim + dim_idx];
        result = gelu_fn(g) * up;
    } else if (HAS_GATE) {
        let g = gate[idx];
        result = gelu_fn(g) * input[idx];
    } else {
        result = gelu_fn(input[idx]);
    }

    output[idx] = result;
}
