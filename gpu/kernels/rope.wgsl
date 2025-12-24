// Rotary Position Embeddings (RoPE) Kernel
//
// Applies rotary position embeddings to Q and K tensors.
// RoPE rotates pairs of dimensions based on position and frequency.
//
// For each pair (x_i, x_{i+1}):
//   x'_i     = x_i * cos(θ) - x_{i+1} * sin(θ)
//   x'_{i+1} = x_i * sin(θ) + x_{i+1} * cos(θ)
//
// Where θ = pos * freq_i, freq_i = 1 / (base^(2i/d))
//
// Supports:
// - Original RoPE (base = 10000)
// - Scaled RoPE (for extended context)
// - NTK-aware scaling

const WORKGROUP_SIZE: u32 = 256u;

struct RoPEUniforms {
    seqLen: u32,           // Sequence length
    numHeads: u32,         // Number of heads
    headDim: u32,          // Dimension per head (must be even)
    startPos: u32,         // Starting position (for decode)
    ropeBase: f32,         // Base frequency (default 10000)
    ropeScale: f32,        // Scaling factor for extended context
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: RoPEUniforms;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;  // [seqLen, numHeads, headDim]
@group(0) @binding(2) var<storage, read_write> freqsCos: array<f32>;  // Precomputed cos [maxSeqLen, headDim/2]
@group(0) @binding(3) var<storage, read_write> freqsSin: array<f32>;  // Precomputed sin [maxSeqLen, headDim/2]

// Apply RoPE using precomputed frequencies
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let headDim = uniforms.headDim;
    let numHeads = uniforms.numHeads;
    let seqLen = uniforms.seqLen;
    let startPos = uniforms.startPos;

    // Global thread index (one thread per complex pair)
    let halfDim = headDim / 2u;
    let totalPairs = seqLen * numHeads * halfDim;
    let idx = global_id.x;

    if (idx >= totalPairs) {
        return;
    }

    // Decompose index
    let pos = idx / (numHeads * halfDim);
    let remainder = idx % (numHeads * halfDim);
    let headIdx = remainder / halfDim;
    let pairIdx = remainder % halfDim;
    let actualPos = startPos + pos;

    // Get precomputed cos/sin for this position and dimension
    let freqIdx = actualPos * halfDim + pairIdx;
    let cosVal = freqsCos[freqIdx];
    let sinVal = freqsSin[freqIdx];

    // Apply "rotate-half" layout: pair (x[i], x[i + halfDim])
    let baseIdx = pos * numHeads * headDim + headIdx * headDim;
    let x0 = input[baseIdx + pairIdx];
    let x1 = input[baseIdx + pairIdx + halfDim];

    // Apply rotation
    let y0 = x0 * cosVal - x1 * sinVal;
    let y1 = x0 * sinVal + x1 * cosVal;

    // Write back
    input[baseIdx + pairIdx] = y0;
    input[baseIdx + pairIdx + halfDim] = y1;
}

// Compute frequencies on-the-fly (no precomputation needed)
@compute @workgroup_size(256, 1, 1)
fn rope_compute_freqs(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let headDim = uniforms.headDim;
    let numHeads = uniforms.numHeads;
    let seqLen = uniforms.seqLen;
    let startPos = uniforms.startPos;
    let ropeBase = uniforms.ropeBase;
    let ropeScale = uniforms.ropeScale;

    let idx = global_id.x;
    let halfDim = headDim / 2u;
    let totalPairs = seqLen * numHeads * halfDim;

    if (idx >= totalPairs) {
        return;
    }

    // Decompose index
    let pos = idx / (numHeads * halfDim);
    let remainder = idx % (numHeads * halfDim);
    let headIdx = remainder / halfDim;
    let pairIdx = remainder % halfDim;

    let actualPos = f32(startPos + pos) / ropeScale;

    // Compute frequency: 1 / (base^(2*pairIdx/headDim))
    let exponent = f32(pairIdx * 2u) / f32(headDim);
    let freq = 1.0 / pow(ropeBase, exponent);
    let theta = actualPos * freq;

    let cosVal = cos(theta);
    let sinVal = sin(theta);

    // Apply "rotate-half" layout: pair (x[i], x[i + halfDim])
    let baseIdx = pos * numHeads * headDim + headIdx * headDim;
    let x0 = input[baseIdx + pairIdx];
    let x1 = input[baseIdx + pairIdx + halfDim];

    // Apply rotation
    input[baseIdx + pairIdx] = x0 * cosVal - x1 * sinVal;
    input[baseIdx + pairIdx + halfDim] = x0 * sinVal + x1 * cosVal;
}

// Apply RoPE to both Q and K in one pass
// Input layout: Q and K concatenated [seqLen, numHeads, headDim * 2]
@compute @workgroup_size(256, 1, 1)
fn rope_qk(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let headDim = uniforms.headDim;
    let numHeads = uniforms.numHeads;
    let seqLen = uniforms.seqLen;
    let startPos = uniforms.startPos;
    let ropeBase = uniforms.ropeBase;
    let ropeScale = uniforms.ropeScale;

    let idx = global_id.x;
    // Each thread handles one Q-K pair at one dimension pair
    let halfDim = headDim / 2u;
    let totalPairs = seqLen * numHeads * halfDim;

    if (idx >= totalPairs) {
        return;
    }

    let pos = idx / (numHeads * halfDim);
    let remainder = idx % (numHeads * halfDim);
    let headIdx = remainder / halfDim;
    let pairIdx = remainder % halfDim;

    let actualPos = f32(startPos + pos) / ropeScale;

    // Compute frequency
    let exponent = f32(pairIdx * 2u) / f32(headDim);
    let freq = 1.0 / pow(ropeBase, exponent);
    let theta = actualPos * freq;

    let cosVal = cos(theta);
    let sinVal = sin(theta);

    // Q is in first half, K in second half
    let qBaseIdx = pos * numHeads * headDim * 2u + headIdx * headDim;
    let kBaseIdx = qBaseIdx + headDim;  // K starts after Q

    // Process Q
    let q0 = input[qBaseIdx + pairIdx];
    let q1 = input[qBaseIdx + pairIdx + halfDim];
    input[qBaseIdx + pairIdx] = q0 * cosVal - q1 * sinVal;
    input[qBaseIdx + pairIdx + halfDim] = q0 * sinVal + q1 * cosVal;

    // Process K
    let k0 = input[kBaseIdx + pairIdx];
    let k1 = input[kBaseIdx + pairIdx + halfDim];
    input[kBaseIdx + pairIdx] = k0 * cosVal - k1 * sinVal;
    input[kBaseIdx + pairIdx + halfDim] = k0 * sinVal + k1 * cosVal;
}

// Precompute frequency table (run once at init)
// Output: freqsCos, freqsSin [maxSeqLen, headDim/2]
@compute @workgroup_size(256, 1, 1)
fn precompute_freqs(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let headDim = uniforms.headDim;
    let seqLen = uniforms.seqLen;  // maxSeqLen for precomputation
    let ropeBase = uniforms.ropeBase;
    let ropeScale = uniforms.ropeScale;

    let idx = global_id.x;
    let halfDim = headDim / 2u;
    let totalElements = seqLen * halfDim;

    if (idx >= totalElements) {
        return;
    }

    let pos = idx / halfDim;
    let dimIdx = idx % halfDim;

    let actualPos = f32(pos) / ropeScale;
    let exponent = f32(dimIdx * 2u) / f32(headDim);
    let freq = 1.0 / pow(ropeBase, exponent);
    let theta = actualPos * freq;

    freqsCos[idx] = cos(theta);
    freqsSin[idx] = sin(theta);
}

// NTK-aware scaled RoPE (for extended context without fine-tuning)
// Uses dynamic scaling based on sequence length
@compute @workgroup_size(256, 1, 1)
fn rope_ntk_scaled(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let headDim = uniforms.headDim;
    let numHeads = uniforms.numHeads;
    let seqLen = uniforms.seqLen;
    let startPos = uniforms.startPos;
    var ropeBase = uniforms.ropeBase;
    let ropeScale = uniforms.ropeScale;

    let idx = global_id.x;
    let halfDim = headDim / 2u;
    let totalPairs = seqLen * numHeads * halfDim;

    if (idx >= totalPairs) {
        return;
    }

    // NTK scaling: increase base proportionally to scale factor
    // This preserves high-frequency components better than linear interpolation
    ropeBase = ropeBase * pow(ropeScale, f32(headDim) / (f32(headDim) - 2.0));

    let pos = idx / (numHeads * halfDim);
    let remainder = idx % (numHeads * halfDim);
    let headIdx = remainder / halfDim;
    let pairIdx = remainder % halfDim;

    let actualPos = f32(startPos + pos);

    let exponent = f32(pairIdx * 2u) / f32(headDim);
    let freq = 1.0 / pow(ropeBase, exponent);
    let theta = actualPos * freq;

    let cosVal = cos(theta);
    let sinVal = sin(theta);

    let baseIdx = pos * numHeads * headDim + headIdx * headDim;
    let x0 = input[baseIdx + pairIdx];
    let x1 = input[baseIdx + pairIdx + halfDim];

    input[baseIdx + pairIdx] = x0 * cosVal - x1 * sinVal;
    input[baseIdx + pairIdx + halfDim] = x0 * sinVal + x1 * cosVal;
}

// YaRN-style RoPE with attention scaling
// Combines NTK interpolation with linear interpolation based on frequency
@compute @workgroup_size(256, 1, 1)
fn rope_yarn(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let headDim = uniforms.headDim;
    let numHeads = uniforms.numHeads;
    let seqLen = uniforms.seqLen;
    let startPos = uniforms.startPos;
    let ropeBase = uniforms.ropeBase;
    let ropeScale = uniforms.ropeScale;

    let idx = global_id.x;
    let halfDim = headDim / 2u;
    let totalPairs = seqLen * numHeads * halfDim;

    if (idx >= totalPairs) {
        return;
    }

    let pos = idx / (numHeads * halfDim);
    let remainder = idx % (numHeads * halfDim);
    let headIdx = remainder / halfDim;
    let pairIdx = remainder % halfDim;

    let actualPos = f32(startPos + pos);

    // YaRN parameters
    let beta_fast: f32 = 32.0;
    let beta_slow: f32 = 1.0;
    let alpha: f32 = 1.0;

    // Compute original frequency
    let exponent = f32(pairIdx * 2u) / f32(headDim);
    let origFreq = 1.0 / pow(ropeBase, exponent);

    // Compute wavelength
    let wavelength = 2.0 * 3.14159265359 / origFreq;

    // Interpolation factor based on wavelength
    var ramp: f32;
    let lowWavelength = f32(headDim) / beta_fast;
    let highWavelength = f32(headDim) / beta_slow;

    if (wavelength < lowWavelength) {
        ramp = 0.0;  // No interpolation for high frequencies
    } else if (wavelength > highWavelength) {
        ramp = 1.0;  // Full interpolation for low frequencies
    } else {
        ramp = (wavelength - lowWavelength) / (highWavelength - lowWavelength);
    }

    // Combine original and scaled position
    let scaledPos = actualPos / ropeScale;
    let interpPos = (1.0 - ramp) * actualPos + ramp * scaledPos;

    let theta = interpPos * origFreq;
    let cosVal = cos(theta);
    let sinVal = sin(theta);

    let baseIdx = pos * numHeads * headDim + headIdx * headDim;
    let x0 = input[baseIdx + pairIdx];
    let x1 = input[baseIdx + pairIdx + halfDim];

    input[baseIdx + pairIdx] = x0 * cosVal - x1 * sinVal;
    input[baseIdx + pairIdx + halfDim] = x0 * sinVal + x1 * cosVal;
}
