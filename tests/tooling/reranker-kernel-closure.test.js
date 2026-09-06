import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildWgslClosure } from '../../src/tooling/program-bundle/wgsl-closure.js';
import { createShaderSourceScope, getScopedShaderSource, runWithShaderSourceScope } from '../../src/gpu/kernels/shader-source-scope.js';
import { assertRerankerObservedShaderClosure } from '../../tools/build-reranker-evaluation-capsule.js';

const repoRoot = fileURLToPath(new URL('../../', import.meta.url));
const recipe = JSON.parse(await fs.readFile(path.join(repoRoot,
  'src/config/conversion/qwen3/qwen-3-reranker-0-6b-q4k-ehf16-af32.json')));
assert.equal(recipe.manifest.artifactIdentity.sourceRevision, 'e61197ed45024b0ed8a2d74b80b4d909f1255473');

// Frozen mechanism requirements from the retained Electron reranker episode.
// This checks real packaged WGSL bytes, not physical tensor computation.
const required = ['dequant_f16_out_vec4.wgsl', 'rope_precompute.wgsl', 'gather_f16_vec4.wgsl',
  'rmsnorm_qk.wgsl', 'rope_qk.wgsl', 'kv_cache_write_f32_to_f16.wgsl',
  'residual_vec4.wgsl', 'lm_head_select_logits.wgsl'];
const f16Recipe = JSON.parse(await fs.readFile(path.join(repoRoot,
  'src/config/conversion/qwen3/qwen-3-reranker-0-6b-f16-true-logit-af32.json')));
assert.deepEqual(f16Recipe.inference.rerank, recipe.inference.rerank, 'precision variants preserve the scoring contract');
const f16Closure = await buildWgslClosure(f16Recipe.execution, [], { repoRoot });
for (const file of [...required.slice(1), 'split_qkv.wgsl']) {
  assert(f16Closure.modules.some(module => module.file === file), `F16 Capsule must seal ${file}`);
}
assert(!f16Closure.modules.some(module => module.file === required[0]), 'F16 closure must not borrow Q4K dequantization');
const f16Requests = { requests: [...required.slice(1), 'split_qkv.wgsl', 'probe_shader_f16.wgsl', 'submit_probe.wgsl']
  .map(file => `http://qualification.invalid/src/gpu/kernels/${file}`) };
const observed = assertRerankerObservedShaderClosure(f16Requests, f16Closure.modules);
assert(observed.modelShaders.includes('split_qkv.wgsl'));
assert.deepEqual(observed.deviceProbes, ['probe_shader_f16.wgsl', 'submit_probe.wgsl']);
assert.throws(() => assertRerankerObservedShaderClosure(f16Requests,
  f16Closure.modules.filter(module => module.file !== 'split_qkv.wgsl')), /omits observed model shaders: split_qkv.wgsl/);
assert.throws(() => assertRerankerObservedShaderClosure({}, f16Closure.modules), /requires observed shader requests/);
assert.throws(() => assertRerankerObservedShaderClosure({ requests: [] }, f16Closure.modules), /requires observed model shader requests/);
const closure = await buildWgslClosure(recipe.execution, [], { repoRoot });
const sources = new Map(closure.modules.map(module => [module.file,
  closure.packageFiles.find(file => file.path === module.sourcePath).contents]));
const scope = createShaderSourceScope(sources);
await runWithShaderSourceScope(scope, async () => {
  for (const file of required) {
    assert.ok(getScopedShaderSource(file).source.length > 0, `${file} must be sealed before loading`);
  }
});

const incomplete = structuredClone(recipe.execution);
incomplete.mechanismKernels = incomplete.mechanismKernels.filter(id => id !== 'dequant_q4_f16');
const missing = await buildWgslClosure(incomplete, [], { repoRoot });
assert.equal(missing.modules.some(module => module.file === required[0]), false);
const missingScope = createShaderSourceScope(new Map(missing.modules.map(module => [module.file,
  missing.packageFiles.find(file => file.path === module.sourcePath).contents])));
await assert.rejects(runWithShaderSourceScope(missingScope, async () => getScopedShaderSource(required[0])),
  /outside the verified Capsule source closure/);
const stale = structuredClone(recipe.execution);
stale.kernels.dequant_q4_f16.digest = `sha256:${'0'.repeat(64)}`;
await assert.rejects(buildWgslClosure(stale, [], { repoRoot }), /kernel digest mismatch/);
console.log('reranker-kernel-closure.test: passed (source closure, not hardware evidence)');
