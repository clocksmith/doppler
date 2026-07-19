import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../../src/inference/pipelines/text/linear-attention.js', import.meta.url),
  'utf8'
);
const opsSource = await readFile(
  new URL('../../src/inference/pipelines/text/ops.js', import.meta.url),
  'utf8'
);

for (const target of ['in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj']) {
  assert.match(source, new RegExp(`getLoRAModule\\(lora, layerIdx, '${target}'\\)`));
}
assert.match(source, /qkvLoRA \|\| zLoRA \? null : resolveLinearAttentionQKVZProjection/);
assert.match(source, /aLoRA \|\| bLoRA \? null : resolveLinearAttentionABProjection/);
assert.match(source, /async function applyProjectionLoRA/);
assert.match(source, /catch \(error\) \{\s*releaseOrTrackBuffer\(recorder, baseOutput\.buffer\);\s*throw error;/);
assert.match(opsSource, /lora: lora \?\? null/);

console.log('linear-attention-lora-contract.test: ok');
