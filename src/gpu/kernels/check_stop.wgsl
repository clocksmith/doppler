override WORKGROUP_SIZE: u32 = 1u;

struct StopUniforms {
  eosTokenId: u32,
  maxTokens: u32,
  currentPos: u32,
  tokenIndex: u32,
}

@group(0) @binding(0) var<uniform> uniforms: StopUniforms;
@group(0) @binding(1) var<storage, read> sampledToken: array<u32>;
@group(0) @binding(2) var<storage, read_write> shouldStop: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main() {
  let token = sampledToken[uniforms.tokenIndex];
  let is_eos = token == uniforms.eosTokenId;
  let reached_max = uniforms.currentPos >= uniforms.maxTokens;
  shouldStop[uniforms.tokenIndex] = select(0u, 1u, is_eos || reached_max);
}
