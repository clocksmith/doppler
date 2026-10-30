// Shared by untimed observation snapshots and resident setup. Values are never
// fed back into execution; the buffer-pool owner performs actual GPU readback.
const layerFields = {
  'input_layernorm.weight': 'inputNorm', 'post_attention_layernorm.weight': 'postAttentionNorm',
  'mlp.gate_proj.weight': 'gate', 'mlp.up_proj.weight': 'up', 'mlp.down_proj.weight': 'down',
  'self_attn.q_proj.weight': 'qProj', 'self_attn.k_proj.weight': 'kProj', 'self_attn.v_proj.weight': 'vProj',
  'self_attn.o_proj.weight': 'oProj', 'self_attn.q_norm.weight': 'qNorm', 'self_attn.k_norm.weight': 'kNorm',
  'linear_attn.in_proj_qkv.weight': 'qkvProj', 'linear_attn.in_proj_z.weight': 'linearInProjZ',
  'linear_attn.in_proj_a.weight': 'linearInProjA', 'linear_attn.in_proj_b.weight': 'linearInProjB',
  'linear_attn.dt_bias': 'linearDtBias', 'linear_attn.A_log': 'linearALog', 'linear_attn.conv1d.weight': 'linearConv1D',
  'linear_attn.norm.weight': 'linearNorm', 'linear_attn.out_proj.weight': 'oProj',
};
export async function verifyDopplerGpu(pipeline, canonical, readBuffer) {
  if (canonical.schema !== 'doppler.equal-weight-canonical.v1' || canonical.completed !== true
      || !Array.isArray(canonical.tensors) || !canonical.tensors.length
      || new Set(canonical.tensors.map(tensor => tensor.name)).size !== canonical.tensors.length) throw new Error('Incomplete canonical GPU proof inventory');
  const weights = pipeline.generator._state.weights;
  const checks = [];
  for (const expected of canonical.tensors) {
    if (!Number.isSafeInteger(expected.bytes) || expected.bytes <= 0 || !/^[a-f0-9]{64}$/.test(expected.canonicalBytesSha256)) throw new Error('Invalid canonical GPU resource extent/hash');
    const local = expected.name.replace(/^model\.language_model\./, '');
    const layer = local.match(/^layers\.(\d+)\.(.+)$/);
    let slot, value;
    if (layer) { slot = layerFields[layer[2]]; value = weights.get(`layer_${layer[1]}`)?.[slot]; }
    else { slot = { 'embed_tokens.weight': 'embed', 'norm.weight': 'final_norm', 'lm_head.weight': 'lm_head' }[local]; value = weights.get(slot); }
    const buffer = value?.buffer?.mapAsync ? value.buffer : value;
    if (!slot || !buffer?.mapAsync || !Number.isSafeInteger(buffer.size) || buffer.size < expected.bytes) {
      throw new Error(`Missing actual canonical GPU resource: ${expected.name}`);
    }
    const bytes = await readBuffer(buffer, expected.bytes);
    const actual = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
    const check = { name: expected.name, slot, resourceLabel: buffer.label, bytes: expected.bytes,
      expected: expected.canonicalBytesSha256, actual, pass: actual === expected.canonicalBytesSha256 };
    checks.push(check);
    if (!check.pass) { const error = new Error(`Actual GPU tensor mismatch: ${expected.name}`); error.completedChecks = checks; throw error; }
  }
  return { checks, pass: true };
}
