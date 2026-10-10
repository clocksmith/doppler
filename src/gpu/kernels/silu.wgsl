// SiLU (Swish) Activation Kernel
//
// SiLU(x) = x * sigmoid(x)
// Variants:
// - silu_gate: SiLU(gate) * up (separate gate buffer)
// - silu_gate_split: input = [gate..., up...]
// - silu_vec4: 4 elements per thread
// - silu_gate_rowsplit: input = [numTokens, 2*dim] row-split

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
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;

fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    let z = exp_refined(-abs(clamped));
    let numerator = select(z, 1.0, clamped >= 0.0);
    let denominator = 1.0 + z;
    let inverse = reciprocal_refined(denominator);
    let quotient = numerator * inverse;
    return fma(fma(-quotient, denominator, numerator), inverse, quotient);
}

// Compensate the rounded numerator and quotient; retain the established gate clamp.
fn silu(x: f32) -> f32 {
    let z = exp_refined(-abs(clamp(x, -15.0, 15.0)));
 let numerator = select(x * z, x, x >= 0.0);
 let error = select(fma(x, z, -numerator), 0.0, x >= 0.0);
 let den = 1.0 + z;
 let inverse = reciprocal_refined(den);
 let q = numerator * inverse;
 return fma(fma(-q, den, numerator) + error, inverse, q);
}

// GELU activation (approximate)

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
            let x = input[base_idx + i];
            if (HAS_GATE) {
                let g = gate[base_idx + i];
                let gateAct = select(silu(g), sigmoid(g), GATE_USE_SIGMOID);
                output[base_idx + i] = clamp_swiglu(gateAct * apply_input_activation(x));
            } else {
                output[base_idx + i] = apply_input_activation(x);
            }
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
        let g = input[row_base + dim_idx];
        let up = input[row_base + dim + dim_idx];
        output[token_idx * dim + dim_idx] = clamp_swiglu(silu(g) * up);
        return;
    }

    if (HAS_GATE) {
        let up = input[idx];
        let g = gate[idx];
        let gateAct = select(silu(g), sigmoid(g), GATE_USE_SIGMOID);
        output[idx] = clamp_swiglu(gateAct * apply_input_activation(up));
        return;
    }

    if (USE_SPLIT) {
        let g = input[idx];
        let up = input[idx + u.size];
        output[idx] = clamp_swiglu(silu(g) * up);
        return;
    }

    let x = input[idx];
    output[idx] = apply_input_activation(x);
}

fn exp_refined(x: f32) -> f32 {
  if (abs(x) > 80.0) {
    return exp(x);
  }
  let n = round(x * 1.4426950408889634);
  var r = x;
  if (abs(x) >= 0.000000059604644775390625) {
    let bits = bitcast<u32>(abs(x));
    let mantissa = (bits & 0x007fffffu) | 0x00800000u;
    let shift = i32((bits >> 23u) & 255u) - 118;
    var input_fixed = 0u;
    if (shift >= 0) {
      input_fixed = mantissa << u32(shift);
    } else {
      let right = u32(-shift);
      input_fixed = (mantissa + (1u << (right - 1u))) >> right;
    }
    let power_fixed = u32(abs(n)) * 2977044472u;
    let residual = select(power_fixed - input_fixed, input_fixed - power_fixed, x >= 0.0);
    r = f32(bitcast<i32>(residual)) * 0.00000000023283064365386962890625;
  }
  var p = 0.0001984126984126984;
  p = fma(p, r, 0.001388888888888889);
  p = fma(p, r, 0.008333333333333333);
  p = fma(p, r, 0.041666666666666664);
  p = fma(p, r, 0.16666666666666666);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  return p * bitcast<f32>(u32(i32(n) + 127) << 23u);
}

fn reciprocal_refined(x: f32) -> f32 {
  let estimate = 1.0 / x;
  return fma(estimate, fma(-x, estimate, 1.0), estimate);
}
