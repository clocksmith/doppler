enable f16;

struct Uniforms {
  canvas_length: u32,
  hidden_size: u32,
  vocab_size: u32,
  row_start: u32,
  row_count: u32,
  output_mode: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> softmax_rows: array<f32>;
@group(0) @binding(2) var<storage, read> embedding_section: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let hidden_idx = gid.x;
  let token_idx = gid.y;
  if (hidden_idx >= u.hidden_size || token_idx >= u.canvas_length) {
    return;
  }

  var sum = 0.0;
  var row = 0u;
  loop {
    if (row >= u.row_count) {
      break;
    }
    let vocab_idx = u.row_start + row;
    let probability = softmax_rows[token_idx * u.vocab_size + vocab_idx];
    let embedding = f32(embedding_section[row * u.hidden_size + hidden_idx]);
    sum += probability * embedding;
    row += 1u;
  }

  let out_idx = token_idx * u.hidden_size + hidden_idx;
  if (u.output_mode == 0u) {
    output[out_idx] = sum;
  } else {
    output[out_idx] = output[out_idx] + sum;
  }
}
