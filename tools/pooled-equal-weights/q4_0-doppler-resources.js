// Actual Doppler bindings, read only outside measurement. No source-file reads.
const fields = {
 attn_norm: 'inputNorm', post_attention_norm: 'postAttentionNorm', attn_q: 'qProj', attn_k: 'kProj',
 attn_v: 'vProj', attn_output: 'oProj', attn_q_norm: 'qNorm', attn_k_norm: 'kNorm',
 attn_qkv: 'qkvProj', attn_gate: 'linearInProjZ', ssm_out: 'oProj', ssm_alpha: 'linearInProjA',
 ssm_beta: 'linearInProjB', ssm_conv1d: 'linearConv1D', ssm_dt: 'linearDtBias',
 ssm_a: 'linearALog', ssm_norm: 'linearNorm', ffn_gate_inp: 'routerWeight',
 ffn_gate_inp_shexp: 'sharedExpertRouter', ffn_gate_shexp: 'sharedExpertGate',
 ffn_up_shexp: 'sharedExpertUp', ffn_down_shexp: 'sharedExpertDown',
};
const dtypeNames = { f32: 'F32', f16: 'F16', bf16: 'BF16', q4_0: 'Q4_0', q4_1: 'Q4_1', q8_0: 'Q8_0', q6k: 'Q6_K' };
export async function resolveDopplerQ40Resources(pipeline, inventory, readBufferSlice) {
 const state = pipeline.generator._state, resources = new Map(), banks = new Map();
 for (const tensor of inventory.tensors) {
  const match = /^blk\.(\d+)\.(.+?)(?:\.weight|\.bias)?$/u.exec(tensor.name);
  let value;
  if (!match) value = state.weights.get({ 'token_embd.weight': 'embed', 'output.weight': 'lm_head', 'output_norm.weight': 'final_norm' }[tensor.name]);
  else {
   const layer = Number(match[1]), field = match[2];
   if (['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'].includes(field)) {
    if (!banks.has(layer)) banks.set(layer, await state.dopplerLoader.loadExpert(layer, 0));
    value = banks.get(layer)?.[{ ffn_gate_exps: 'gate', ffn_up_exps: 'up', ffn_down_exps: 'down' }[field]];
   } else value = state.weights.get(`layer_${layer}`)?.[fields[field]];
  }
  const buffer = value?.buffer?.mapAsync ? value.buffer : value;
  if (!buffer?.mapAsync || !Number.isSafeInteger(buffer.size) || buffer.size < tensor.bytes) throw new Error(`Missing actual native Doppler binding: ${tensor.name}`);
  const dtype = value.dtype ? dtypeNames[value.dtype] : tensor.dtype;
  resources.set(tensor.name, { dtype, shape: value.shape ?? tensor.shape, bytes: tensor.bytes, location: 'gpu-bound',
   readRange: async (offset, bytes) => (await readBufferSlice(buffer, offset, Math.ceil(bytes / 4) * 4)).slice(0, bytes) });
 }
 return resources;
}
