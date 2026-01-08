// ReLU Activation Kernel
//
// ReLU(x) = max(0, x)
// Leaky ReLU(x) = x if x >= 0, else alpha * x
//
// Uses override constants for compile-time feature selection:
// - LEAKY: Enable leaky ReLU
// - LEAKY_ALPHA: Alpha coefficient for leaky ReLU

// =============================================================================
// Override Constants (compile-time configuration)
// =============================================================================

override WORKGROUP_SIZE: u32 = 256u;

// Feature flags
override LEAKY: bool = false;
override LEAKY_ALPHA: f32 = 0.01;

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

// =============================================================================
// Main Entry Point
// =============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let x = input[idx];

    if (LEAKY) {
        output[idx] = select(LEAKY_ALPHA * x, x, x >= 0.0);
    } else {
        output[idx] = max(0.0, x);
    }
}

// =============================================================================
// Batched Entry Point
// =============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_batched(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // ReLU is element-wise, same as main
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let x = input[idx];

    if (LEAKY) {
        output[idx] = select(LEAKY_ALPHA * x, x, x >= 0.0);
    } else {
        output[idx] = max(0.0, x);
    }
}
