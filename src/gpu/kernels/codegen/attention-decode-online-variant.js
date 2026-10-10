export const ATTENTION_DECODE_ONLINE_F16KV_VARIANT = {
  id: 'attention-decode-online-f16kv',
  source: 'src/gpu/kernels/attention_decode_online_f16.wgsl',
  target: 'src/gpu/kernels/attention_decode_online_f16kv.wgsl',
  patch: 'src/gpu/kernels/codegen/patches/attention_decode_online_f16kv.from.attention_decode_online_f16.diff',
};
