// Greedy prefix verification of GPU-resident draft and target token IDs.
// Header: emitted count, accepted drafts, status (0 success, 1 invalid token), last input row, next token.
struct SpeculativeAcceptParams {
  draft_count: u32,
  output_limit: u32,
  vocab_size: u32,
  _pad: u32,
}
@group(0) @binding(0) var<uniform> params: SpeculativeAcceptParams;
@group(0) @binding(1) var<storage, read> drafts: array<u32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  result[0] = 0u;
  result[1] = 0u;
  result[2] = 0u;
  result[3] = 0u;
  result[4] = 0u;
  var accepted = 0u;
  var matching = true;
  for (var i = 0u; i <= params.draft_count; i++) {
    if (targets[i] >= params.vocab_size) { result[2] = 1u; }
    result[5u + i] = targets[i];
    if (i < params.draft_count) {
      if (drafts[i] >= params.vocab_size) { result[2] = 1u; }
      matching = matching && drafts[i] == targets[i];
      if (matching) { accepted++; }
    }
  }
  if (result[2] == 0u) {
    result[0] = min(accepted + 1u, params.output_limit);
    result[1] = min(accepted, params.output_limit);
    result[3] = result[0] - 1u;
    result[4] = targets[result[3]];
  }
}
