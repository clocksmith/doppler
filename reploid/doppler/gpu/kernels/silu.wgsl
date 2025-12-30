// SiLU (Swish) Activation Kernel
//
// SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
//
// Also known as Swish activation, used in LLaMA and other modern LLMs.
//
// Includes fused variants:
// - SiLU(gate) * up (LLaMA SwiGLU FFN pattern)
// - SiLU with optional bias add

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,          // Total number of elements
    has_bias: u32,      // 1 if bias should be added before activation
    has_gate: u32,      // 1 if using gated variant (SiLU(gate) * up)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;   // For gated variant
@group(0) @binding(4) var<storage, read> bias: array<f32>;   // Optional bias

// Sigmoid helper
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// SiLU helper
fn silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

// Basic SiLU activation
// output = x * sigmoid(x)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let x = input[idx];
    output[idx] = silu(x);
}

// SiLU with bias: output = silu(x + bias)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_bias(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let x = input[idx] + bias[idx];
    output[idx] = silu(x);
}

// Gated SiLU (SwiGLU): output = SiLU(gate) * up
// This is the pattern used in LLaMA FFN:
//   up = input @ W_up
//   gate = input @ W_gate
//   output = SiLU(gate) * up
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let up = input[idx];
    let g = gate[idx];

    // SiLU(gate) * up
    output[idx] = silu(g) * up;
}

// Fused gated SiLU with interleaved input
// Input format: [gate0, up0, gate1, up1, ...]
// Useful when gate and up are stored interleaved
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_interleaved(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let half_size = u.size / 2u;

    if (idx >= half_size) {
        return;
    }

    let gate_idx = idx * 2u;
    let up_idx = gateIdx + 1u;

    let g = input[gate_idx];
    let up = input[up_idx];

    output[idx] = silu(g) * up;
}

// Fused gated SiLU with split input
// First half of input is gate, second half is up
// Input format: [gate0, gate1, ..., gateN, up0, up1, ..., upN]
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_split(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let half_size = u.size / 2u;

    if (idx >= half_size) {
        return;
    }

    let g = input[idx];           // First half: gate
    let up = input[idx + half_size];  // Second half: up

    output[idx] = silu(g) * up;
}

// Vectorized SiLU (process 4 elements per thread)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_vec4(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let base_idx = global_id.x * 4u;

    if (base_idx >= u.size) {
        return;
    }

    // Process up to 4 elements
    let remaining = min(4u, u.size - base_idx);

    for (var i: u32 = 0u; i < remaining; i = i + 1u) {
        let x = input[base_idx + i];
        output[base_idx + i] = silu(x);
    }
}

// Vectorized gated SiLU
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_vec4(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let base_idx = global_id.x * 4u;

    if (base_idx >= u.size) {
        return;
    }

    let remaining = min(4u, u.size - base_idx);

    for (var i: u32 = 0u; i < remaining; i = i + 1u) {
        let up = input[base_idx + i];
        let g = gate[base_idx + i];
        output[base_idx + i] = silu(g) * up;
    }
}

// In-place SiLU (modifies input buffer)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_inplace(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let x = output[idx];  // Using output as read-write buffer
    output[idx] = silu(x);
}

// GELU activation for comparison (used in some models)
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gelu(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let x = input[idx];

    // Approximate GELU
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let c: f32 = 0.044715;

    let inner = sqrt_2_over_pi * (x + c * x * x * x);
    // Clamp inner to avoid tanh overflow: tanh uses exp(2x) which overflows for |x| > ~88
    let inner_clamped = clamp(inner, -15.0, 15.0);
    output[idx] = 0.5 * x * (1.0 + tanh(inner_clamped));
}

// Gated GELU (GeGLU) - similar pattern to SwiGLU
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn geglu(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let up = input[idx];
    let g = gate[idx];

    // GELU(gate) * up
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let c: f32 = 0.044715;

    let inner = sqrt_2_over_pi * (g + c * g * g * g);
    // Clamp inner to avoid tanh overflow: tanh uses exp(2x) which overflows for |x| > ~88
    let inner_clamped = clamp(inner, -15.0, 15.0);
    let gelu_g = 0.5 * g * (1.0 + tanh(inner_clamped));

    output[idx] = gelu_g * up;
}

// ReLU for simple comparison/fallback
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn relu(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    output[idx] = max(0.0, input[idx]);
}

// Leaky ReLU
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn leaky_relu(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let x = input[idx];
    let alpha: f32 = 0.01;

    output[idx] = select(alpha * x, x, x >= 0.0);
}

// Fused SiLU + element-wise multiply (common pattern)
// output = SiLU(a) * b
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_mul(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let a = input[idx];
    let b = gate[idx];  // Using gate binding for second operand

    output[idx] = silu(a) * b;
}

// Row-split gated SiLU (for fused gate+up FFN)
// Input format: [numTokens, 2*dim] where each row is [gate[0..dim), up[0..dim)]
// Output format: [numTokens, dim]
// u.size = numTokens * dim (output size)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_rowsplit(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // size is the output size (numTokens * dim)
    if (idx >= u.size) {
        return;
    }

    // Calculate position in output
    // u.size = numTokens * dim, so dim = size / numTokens
    // But we need dim to be passed in uniforms. For now, assume hasBias encodes dim.
    // Actually, let's use: idx maps to output[token, d], input has [token, gate_d, up_d]
    // The input has 2x the elements of output, so:
    //   output_idx = token * dim + d
    //   gate_idx = token * (dim * 2) + d
    //   up_idx = token * (dim * 2) + dim + d
    // We can compute: halfInputSize = output size = numTokens * dim
    // So: token = idx / dim, d = idx % dim (but we don't know dim separately)
    //
    // Alternative: use gate buffer to pass dim as first element (hacky)
    // Better: use hasBias to encode dim (multiply by workgroup size)
    // Best: Just require the fused path to pre-calculate indices
    //
    // Simplest approach: input is [N, 2D], output is [N, D]
    // Total output elements = N * D = u.size
    // Total input elements = N * 2D = 2 * u.size
    // For output[i], we need input[2*row*D + col] and input[2*row*D + D + col]
    // where row = i / D and col = i % D
    // But we need D! Use hasGate to encode log2(D) or similar...
    //
    // Actually simplest: assume dim is passed via hasBias field (repurposed)
    let dim = u.has_bias;  // Repurposed: hasBias stores dim for rowsplit variant
    let token_idx = idx / dim;
    let dim_idx = idx % dim;

    let row_base = token_idx * dim * 2u;
    let g = input[row_base + dim_idx];
    let up = input[row_base + dim + dim_idx];

    output[idx] = silu(g) * up;
}

// Row-split gated GELU (GeGLU) variant for models using GELU activation
// Input format: [numTokens, 2*dim] where each row is [gate[0..dim), up[0..dim)]
// Output format: [numTokens, dim]
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn geglu_rowsplit(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let dim = u.has_bias;  // Repurposed: hasBias stores dim for rowsplit variant
    let token_idx = idx / dim;
    let dim_idx = idx % dim;

    let row_base = token_idx * dim * 2u;
    let g = input[row_base + dim_idx];
    let up = input[row_base + dim + dim_idx];

    // GELU(gate) * up
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let c: f32 = 0.044715;
    let inner = sqrt_2_over_pi * (g + c * g * g * g);
    let inner_clamped = clamp(inner, -15.0, 15.0);
    let gelu_g = 0.5 * g * (1.0 + tanh(inner_clamped));

    output[idx] = gelu_g * up;
}

// Batched SiLU with separate batch dimension
// input shape: [batchSize, hiddenSize]
// Each thread handles one element
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_batched(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    // SiLU is element-wise, so batching is automatic
    let x = input[idx];
    output[idx] = silu(x);
}
