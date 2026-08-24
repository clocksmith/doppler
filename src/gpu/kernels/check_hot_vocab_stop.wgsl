override WORKGROUP_SIZE: u32 = 1u;

struct HotStopUniforms {
  eosTokenId: u32,
  maxTokens: u32,
  currentPos: u32,
  tokenIndex: u32,
  hotTokenSentinel: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: HotStopUniforms;
@group(0) @binding(1) var<storage, read> sampledToken: array<u32>;
@group(0) @binding(2) var<storage, read_write> shouldStop: array<u32>;
@group(0) @binding(3) var<storage, read> hotTokenIndexMap: array<u32>;
@group(0) @binding(4) var<storage, read_write> nextInputToken: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main() {
  let token = sampledToken[uniforms.tokenIndex];
  let hot_index = hotTokenIndexMap[token];
  let hot_miss = hot_index == uniforms.hotTokenSentinel;
  let is_eos = token == uniforms.eosTokenId;
  let reached_max = uniforms.currentPos >= uniforms.maxTokens;
  shouldStop[uniforms.tokenIndex] = select(0u, 1u, hot_miss || is_eos || reached_max);
  nextInputToken[uniforms.tokenIndex] = hot_index;
}
