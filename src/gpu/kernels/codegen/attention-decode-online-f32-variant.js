export const ATTENTION_DECODE_ONLINE_F32_VARIANT = {
  id: 'attention-decode-online-f32',
  source: 'src/gpu/kernels/attention_decode_online_f16kv.wgsl',
  target: 'src/gpu/kernels/attention_decode_online.wgsl',
  rules: [
    { type: 'literal', count: 1, from: 'enable f16;\n', to: '' },
    { type: 'literal', count: 1, from: 'f32 Q + f16 KV + f32 output', to: 'f32 Q/K/V + f32 output' },
    { type: 'literal', count: 2, from: 'array<f16>', to: 'array<f32>' },
  ],
};
