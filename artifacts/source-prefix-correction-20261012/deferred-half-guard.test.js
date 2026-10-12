// Execute the actual generated source guard, not a duplicate policy function.
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
const directory = await mkdtemp(join(tmpdir(), 'doppler-deferred-guard-'));
try {
  const root = join(directory, 'reports/local/source-prefix-correction-20261012/build');
  for (const file of ['src/inference/pipelines/text/attention/interpreter.js',
    'src/inference/pipelines/text/attention/projection-plan.js',
    'src/gpu/kernels/rmsnorm-qk.js', 'src/gpu/kernels/rmsnorm-qk.d.ts']) {
    await mkdir(join(root, file, '..'), { recursive: true });
    await writeFile(join(root, file), await readFile(new URL(`../../${file}`, import.meta.url)));
  }
  execFileSync(process.execPath, [new URL('./prepare-deferred-half-candidate.js', import.meta.url).pathname, root]);
  const source = await readFile(join(root, 'src/inference/pipelines/text/attention/interpreter.js'), 'utf8');
  const expression = source.match(/kHalfRoundingStart: ([\s\S]*?) : null,/)[1] + ' : null';
  const fields = ['skipKVCacheWrites', 'retainSharedKvBuffers', 'valueAliasesKey', 'diffusionGemmaDecoder',
    'state', 'qTensor', 'config', 'disableRoPE'];
  const evaluate = new Function(...fields, `return (${expression});`);
  const base = { skipKVCacheWrites: false, retainSharedKvBuffers: false, valueAliasesKey: false,
    diffusionGemmaDecoder: false, state: { kvCache: { kvDtype: 'f16', hasGPUCache: () => true } },
    qTensor: { dtype: 'f32' }, config: { queryKeyNormAxis: 'head', ropeRotaryDim: 64, ropeFrequencyBaseDim: 64 }, disableRoPE: false };
  const run = values => evaluate(...fields.map(name => ({ ...base, ...values })[name]));
  assert.equal(run({}), 64);
  assert.equal(run({ disableRoPE: true }), 0);
  for (const state of [{}, { kvCache: { kvDtype: 'f16' } },
    { kvCache: { kvDtype: 'f16', hasGPUCache: () => false } },
    { kvCache: { kvDtype: 'f32', hasGPUCache: () => true } }]) assert.equal(run({ state }), null);
  for (const flag of fields.slice(0, 4)) assert.equal(run({ [flag]: true }), null);
  assert.equal(run({ qTensor: { dtype: 'f16' } }), null);
  assert.equal(run({ config: { ...base.config, ropeFrequencyBaseDim: 256 } }), null);
  assert.equal(run({ config: { ...base.config, queryKeyNormAxis: 'full' } }), null);
  console.log('Deferred half-store guard: live GPU cache, exact geometry, dtype and ownership exclusions passed.');
} finally { await rm(directory, { recursive: true, force: true }); }
