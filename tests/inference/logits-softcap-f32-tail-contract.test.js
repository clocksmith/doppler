import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const { shouldForceStableF32Logits, createStableF32LogitsKernelPath } = await import(
  '../../src/inference/pipelines/text/logits/precision-policy.js'
);

const precisionPolicySource = await fs.readFile(
  path.join(repoRoot, 'src/inference/pipelines/text/logits/precision-policy.js'),
  'utf8'
);
assert.match(
  precisionPolicySource,
  /from '..\/..\/..\/..\/config\/transforms\/execution-graph-transforms\.js'/,
  'logits precision policy must reuse the execution-graph kernel remap contract.'
);
assert.doesNotMatch(
  precisionPolicySource,
  /STABLE_F32_LOGITS_KERNEL_MAP|new Map\(\[/,
  'logits precision policy must not own a local kernel remap table.'
);

for (const relativePath of [
  'src/inference/pipelines/text/logits/index.js',
  'src/inference/pipelines/text/logits/gpu.js',
]) {
  const source = await fs.readFile(path.join(repoRoot, relativePath), 'utf8');
  assert.match(
    source,
    /from '\.\/precision-policy\.js'/,
    `${relativePath} must use the shared logits precision policy helper.`
  );
  assert.doesNotMatch(
    source,
    /finalLogitSoftcapping\)\s*&&\s*config\.finalLogitSoftcapping\s*>\s*0/,
    `${relativePath} must not own the stable-f32 logits condition inline.`
  );
}

const configRules = JSON.parse(
  await fs.readFile(path.join(repoRoot, 'src/rules/inference/config.rules.json'), 'utf8')
);
assert.deepEqual(configRules.stableF32Logits, [
  {
    match: {
      inputDtype: 'f16',
      finalLogitSoftcapping: { gt: 0 },
    },
    value: true,
  },
  {
    match: {
      inputDtype: 'f16',
      rmsNormWeightOffset: true,
      hiddenSize: { lte: 768 },
    },
    value: true,
  },
  { match: {}, value: false },
]);

assert.equal(shouldForceStableF32Logits({
  finalLogitSoftcapping: 30,
  rmsNormWeightOffset: false,
  hiddenSize: 2048,
}, 'f16'), true);
assert.equal(shouldForceStableF32Logits({
  finalLogitSoftcapping: null,
  rmsNormWeightOffset: true,
  hiddenSize: 768,
}, 'f16'), true);
assert.equal(shouldForceStableF32Logits({
  finalLogitSoftcapping: 30,
  rmsNormWeightOffset: false,
  hiddenSize: 2048,
}, 'f32'), false);
assert.throws(
  () => shouldForceStableF32Logits({
    rmsNormWeightOffset: false,
    hiddenSize: 2048,
  }, 'f16'),
  /config\.finalLogitSoftcapping/
);

const stableKernelPath = createStableF32LogitsKernelPath({
  id: 'logits-test',
  postLayer: [
    { op: 'final_norm', kernel: 'rmsnorm.wgsl', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
    { op: 'lm_head', kernel: 'matmul_gemv_subgroup_f16a.wgsl', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
  ],
});
assert.deepEqual(stableKernelPath.postLayer, [
  { op: 'final_norm', kernel: 'rmsnorm.wgsl', precision: { inputDtype: 'f32', outputDtype: 'f32' } },
  { op: 'lm_head', kernel: 'matmul_gemv_subgroup.wgsl', precision: { inputDtype: 'f32', outputDtype: 'f32' } },
]);
assert.throws(
  () => createStableF32LogitsKernelPath({
    id: 'logits-test',
    postLayer: [
      { op: 'lm_head', kernel: 'custom_f16_lm_head.wgsl', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
    ],
  }),
  /cannot map LM-head kernel/
);

console.log('logits-softcap-f32-tail-contract.test: ok');
