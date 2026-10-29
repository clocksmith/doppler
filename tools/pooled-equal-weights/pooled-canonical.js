// Benchmark adapter only. The Pooled engine and its native F32 kernels are unchanged.
export async function loadCanonicalWeights(baseUrl, layerTypes) {
  const proof = await (await fetch(`${baseUrl}/canonical.json`)).json();
  if (!proof.completed || proof.schema !== 'doppler.equal-weight-canonical.v1') throw new Error('Incomplete canonical model');
  const range = async (start, count) => {
    const r = await fetch(`${baseUrl}/model.safetensors`, { headers: { Range: `bytes=${start}-${start + count - 1}` } });
    if (r.status !== 206) throw new Error(`Canonical tensor request requires HTTP206: ${r.status}`);
    const bytes = await r.arrayBuffer();
    if (bytes.byteLength !== count) throw new Error('Truncated canonical range');
    return bytes;
  };
  const prefix = await range(0, 8), length = Number(new DataView(prefix).getBigUint64(0, true));
  if (!Number.isSafeInteger(length) || length <= 0 || length > 8 << 20) throw new Error('Invalid canonical header length');
  const header = JSON.parse(new TextDecoder().decode(await range(8, length)));
  const inventory = new Map(proof.tensors.map(t => [t.name, t]));
  const ledger = [], bindings = [], consumed = new Set();
  const digest = async data => [...new Uint8Array(await crypto.subtle.digest('SHA-256', data))].map(x => x.toString(16).padStart(2, '0')).join('');
  async function entry(name, transform = 'identity') {
    if (consumed.has(name)) throw new Error(`Duplicate logical tensor binding: ${name}`);
    const tensor = header[name], expected = inventory.get(name);
    if (!tensor || !expected || tensor.dtype !== 'F32') throw new Error(`Missing canonical tensor ${name}`);
    const bytes = await range(8 + length + tensor.data_offsets[0], tensor.data_offsets[1] - tensor.data_offsets[0]);
    const original = await digest(bytes);
    if (original !== expected.canonicalBytesSha256) throw new Error(`Canonical source hash mismatch: ${name}`);
    const data = new Float32Array(bytes);
    if (transform === 'rms-plus-one') for (let i = 0; i < data.length; i++) data[i] = data[i] + 1;
    else if (transform === 'negative-exp') for (let i = 0; i < data.length; i++) data[i] = -Math.exp(data[i]);
    else if (transform !== 'identity') throw new Error('Unknown declared preparation transform');
    const record = { name, shape: tensor.shape, sourceSha256: original, transform, preparedSha256: await digest(data) };
    ledger.push(record);
    consumed.add(name);
    const value = { kind: 'f32', data, shape: tensor.shape };
    bindings.push({ record, value });
    return value;
  }
  const root = 'model.language_model.', layers = [];
  for (const [i, type] of layerTypes.entries()) {
    const p = `${root}layers.${i}.`;
    const layer = { isFull: type === 'full_attention',
      attnNorm: await entry(p + 'input_layernorm.weight', 'rms-plus-one'),
      postNorm: await entry(p + 'post_attention_layernorm.weight', 'rms-plus-one'),
      ffnGate: await entry(p + 'mlp.gate_proj.weight'), ffnUp: await entry(p + 'mlp.up_proj.weight'),
      ffnDown: await entry(p + 'mlp.down_proj.weight') };
    if (layer.isFull) {
      for (const [field, suffix] of Object.entries({ wq: 'q_proj', wk: 'k_proj', wv: 'v_proj', wo: 'o_proj' })) {
        layer[field] = await entry(p + `self_attn.${suffix}.weight`);
      }
      layer.qNorm = await entry(p + 'self_attn.q_norm.weight', 'rms-plus-one');
      layer.kNorm = await entry(p + 'self_attn.k_norm.weight', 'rms-plus-one');
    } else {
      const fields = { wqkv: 'in_proj_qkv.weight', wz: 'in_proj_z.weight', wBeta: 'in_proj_b.weight',
        wAlpha: 'in_proj_a.weight', dtBias: 'dt_bias', conv: 'conv1d.weight', ssmNorm: 'norm.weight', wOut: 'out_proj.weight' };
      for (const [field, suffix] of Object.entries(fields)) layer[field] = await entry(p + `linear_attn.${suffix}`);
      layer.ssmA = await entry(p + 'linear_attn.A_log', 'negative-exp');
    }
    layers.push(layer); console.log(`Canonical logical layer ${i + 1}/${layerTypes.length} verified`);
  }
  const result = { layers, embed: await entry(root + 'embed_tokens.weight'),
    finalNorm: await entry(root + 'norm.weight', 'rms-plus-one'), head: await entry(root + 'lm_head.weight') };
  const unused = proof.tensors.filter(x => !consumed.has(x.name)).map(x => x.name);
  if (unused.length) throw new Error(`Unbound canonical tensors: ${unused.join(', ')}`);
  result.canonicalProof = { schema: 'pooled.canonical-input-check.v1', tensorCount: ledger.length, ledger,
    claim: 'Host input tensors and declared preparation transforms only; GPU materialization proof and output parity remain separate gates.' };
  Object.defineProperty(result, 'materializationBindings', { value: bindings });
  return result;
}
