// One attention algorithm owns both cache representations; all arithmetic stays F32.
export const ATTENTION_PREFILL_QUERY_TILE_F16KV_VARIANT = {
  id: 'attention-prefill-query-tile-f16kv',
  source: 'src/gpu/kernels/attention_prefill_query_tile_f32.wgsl',
  target: 'src/gpu/kernels/attention_prefill_query_tile_f16kv.wgsl',
  rules: [
    { type: 'literal', count: 1,
      from: '// F32 prefill query tile. Existing attention uniform/cache/mask contract.',
      to: '// F32 query/output and F16 KV prefill query tile. Existing attention uniform/cache/mask contract.' },
    { type: 'literal', count: 1, from: 'enable subgroups;', to: 'enable f16;\nenable subgroups;' },
    { type: 'literal', count: 1,
      from: 'var<storage, read> K: array<f32>;', to: 'var<storage, read> K: array<f16>;' },
    { type: 'literal', count: 1,
      from: 'var<storage, read> V: array<f32>;', to: 'var<storage, read> V: array<f16>;' },
    { type: 'literal', count: 1,
      from: 'value = K[base + i % HEAD_DIM];', to: 'value = f32(K[base + i % HEAD_DIM]);' },
    { type: 'literal', count: 1,
      from: 'value = V[base + i % HEAD_DIM];', to: 'value = f32(V[base + i % HEAD_DIM]);' },
  ],
};
