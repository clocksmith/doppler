override WORKGROUP_SIZE: u32 = 64u;
override INPUT_ROWS: u32 = 32u;
override HIDDEN_SIZE: u32 = 256u;
override EMBEDDING_ID: u32 = 1u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> embeddings: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= (INPUT_ROWS + 1u) * HIDDEN_SIZE) { return; }
    if (i < INPUT_ROWS * HIDDEN_SIZE) { output[i] = input[i]; }
    else { output[i] = embeddings[EMBEDDING_ID * HIDDEN_SIZE + i % HIDDEN_SIZE]; }
}
