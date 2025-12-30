// Stop Condition Check Kernel
//
// Checks if generation should stop based on:
// - Token is EOS (end of sequence)
// - Position reached max_tokens
//
// Used for GPU-only decode loop to avoid CPU roundtrips

struct Uniforms {
    eos_token_id: u32,
    max_tokens: u32,
    current_pos: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> sampled_token: u32;  // Token just sampled
@group(0) @binding(2) var<storage, read_write> should_stop: u32;  // Output: 1 if should stop, 0 otherwise

@compute @workgroup_size(1, 1, 1)
fn main() {
    let token = sampled_token;
    let is_eos = (token == u.eos_token_id);
    let reached_max = (u.current_pos >= u.max_tokens);

    if (is_eos || reached_max) {
        should_stop = 1u;
    } else {
        should_stop = 0u;
    }
}
