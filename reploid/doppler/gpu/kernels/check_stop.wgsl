// Stop Condition Check Kernel
//
// Checks if generation should stop based on:
// - Token is EOS (end of sequence)
// - Position reached max_tokens
//
// Used for GPU-only decode loop to avoid CPU roundtrips

struct StopUniforms {
    eosTokenId: u32,
    maxTokens: u32,
    currentPos: u32,
}

@group(0) @binding(0) var<uniform> uniforms: StopUniforms;
@group(0) @binding(1) var<storage, read> sampledToken: u32;  // Token just sampled
@group(0) @binding(2) var<storage, read_write> shouldStop: u32;  // Output: 1 if should stop, 0 otherwise

@compute @workgroup_size(1, 1, 1)
fn main() {
    let token = sampledToken;
    let isEOS = (token == uniforms.eosTokenId);
    let reachedMax = (uniforms.currentPos >= uniforms.maxTokens);

    if (isEOS || reachedMax) {
        shouldStop = 1u;
    } else {
        shouldStop = 0u;
    }
}
