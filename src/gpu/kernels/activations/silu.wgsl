// SiLU (Swish) Activation Kernel
//
// SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
//
// Uses override constants for compile-time feature selection:
// - HAS_GATE: Enable gated variant (SwiGLU pattern)
// - HAS_BIAS: Add bias before activation
// - LAYOUT: Input layout (0=separate, 1=interleaved, 2=rowsplit)
// - USE_VEC4: Process 4 elements per thread
// - IN_PLACE: Use output buffer as read-write

// =============================================================================
// Override Constants (compile-time configuration)
// =============================================================================

override WORKGROUP_SIZE: u32 = 256u;

// Feature flags - compiler eliminates dead branches
override HAS_GATE: bool = false;      // SiLU(gate) * up pattern
override HAS_BIAS: bool = false;      // Add bias before activation
override IN_PLACE: bool = false;      // Modify output buffer in-place

// Layout: 0 = separate buffers, 1 = interleaved [g,u,g,u], 2 = rowsplit [g...,u...]
override LAYOUT: u32 = 0u;

// Vectorization
override USE_VEC4: bool = false;

// For rowsplit layout: dimension size
override DIM_SIZE: u32 = 0u;

// =============================================================================
// Uniforms (per-dispatch)
// =============================================================================

struct Uniforms {
    size: u32,          // Total number of output elements
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;   // For gated variant (LAYOUT=0)
@group(0) @binding(4) var<storage, read> bias: array<f32>;   // For bias variant

// =============================================================================
// Helper Functions
// =============================================================================

fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    return 1.0 / (1.0 + exp(-clamped));
}

fn silu_fn(x: f32) -> f32 {
    return x * sigmoid(x);
}

// =============================================================================
// Main Entry Point
// =============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Handle vectorization
    let idx = select(global_id.x, global_id.x * 4u, USE_VEC4);

    if (idx >= u.size) {
        return;
    }

    // Determine iteration count for vec4
    let count = select(1u, min(4u, u.size - idx), USE_VEC4);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let elem_idx = idx + i;

        // Load input value
        var x: f32;
        if (IN_PLACE) {
            x = output[elem_idx];
        } else {
            x = input[elem_idx];
        }

        // Apply bias if enabled
        if (HAS_BIAS) {
            x = x + bias[elem_idx];
        }

        // Apply SiLU
        var result = silu_fn(x);

        // Apply gating if enabled
        if (HAS_GATE) {
            var up: f32;

            if (LAYOUT == 0u) {
                // Separate buffers: gate from gate[], up from input[]
                let g = gate[elem_idx];
                up = x;  // input is 'up'
                result = silu_fn(g) * up;
            } else if (LAYOUT == 1u) {
                // Interleaved: [g0, u0, g1, u1, ...]
                let gate_idx = elem_idx * 2u;
                let up_idx = gate_idx + 1u;
                let g = input[gate_idx];
                up = input[up_idx];
                result = silu_fn(g) * up;
            } else if (LAYOUT == 2u) {
                // Rowsplit: [g0..gN, u0..uN] per row
                let dim = DIM_SIZE;
                let token_idx = elem_idx / dim;
                let dim_idx = elem_idx % dim;
                let row_base = token_idx * dim * 2u;
                let g = input[row_base + dim_idx];
                up = input[row_base + dim + dim_idx];
                result = silu_fn(g) * up;
            }
        }

        output[elem_idx] = result;
    }
}

// =============================================================================
// Batched Entry Point (different workgroup layout for M>1)
// =============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_batched(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // SiLU is element-wise, so batched is same as main
    // This entry point exists for consistency with other kernels
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    var x = input[idx];

    if (HAS_BIAS) {
        x = x + bias[idx];
    }

    var result = silu_fn(x);

    if (HAS_GATE) {
        if (LAYOUT == 0u) {
            let g = gate[idx];
            result = silu_fn(g) * x;
        } else if (LAYOUT == 2u) {
            let dim = DIM_SIZE;
            let token_idx = idx / dim;
            let dim_idx = idx % dim;
            let row_base = token_idx * dim * 2u;
            let g = input[row_base + dim_idx];
            let up = input[row_base + dim + dim_idx];
            result = silu_fn(g) * up;
        }
    }

    output[idx] = result;
}
