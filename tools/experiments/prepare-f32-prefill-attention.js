#!/usr/bin/env node
// Offline, non-promoting preparation only. No model load, device creation or Git.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { parseManifest } from '../../src/formats/rdrr/index.js';
import { fetchManifestPayloadFromBaseUrl } from '../../src/client/runtime/model-source.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { applyExecutionPatch } from '../../src/inference/pipelines/text/execution-v1/validation.js';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { getKernelPathAttentionVariant, getLayerSteps } from '../../src/config/kernel-path-loader.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const [inputPath, outputDir] = process.argv.slice(2);
assert(inputPath && outputDir, 'Usage: node tools/experiments/prepare-f32-prefill-attention.js input-config.json output-directory');
const inputText = await readFile(inputPath, 'utf8');
const input = JSON.parse(inputText);
const manifestText = await readFile(join(input.rdrrDir, 'manifest.json'), 'utf8');
const manifest = parseManifest(manifestText);
const fetched = await fetchManifestPayloadFromBaseUrl(pathToFileURL(resolve(input.rdrrDir)).href);
assert.equal(fetched.text, manifestText, 'Exact source manifest fetch differs');
assert.equal(input.runtimeConfig.inference.session.kvcache.kvDtype, 'f32');
assert.equal(input.runtimeConfig.inference.session.compute.defaults.activationDtype, 'f32');
assert.equal(input.runtimeConfig.inference.session.compute.defaults.accumDtype, 'f32');
assert(!input.runtimeConfig.inference.executionPatch || Object.keys(input.runtimeConfig.inference.executionPatch).length === 0,
  'Use an unpatched attention baseline; this helper must not merge another owner\'s experiment');
const shader = 'attention_prefill_query_tile_f32.wgsl';
const digest = `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${shader}#main`]}`;
assert.match(digest, /^sha256:[0-9a-f]{64}$/);
const declaration = { kernel: shader, entry: 'main', digest, precision: { activationDtype: 'f32', kvDtype: 'f32' } };
const patch = { addKernels: [{ key: 'attn_prefill_query_tile_f32', kernel: declaration }],
  set: [{ section: 'prefill', op: 'attention', kernelKey: 'attn_prefill_query_tile_f32' }] };
const patched = applyExecutionPatch(manifest.inference.execution, patch);
const semantic = graph => ['preLayer', 'decode', 'prefill', 'postLayer'].map(section => [section,
  graph[section].map(entry => Array.isArray(entry) ? [entry[0], ...entry.slice(2)]
    : { ...entry, steps: entry.steps.map(step => [step[0], ...step.slice(2)]) })]);
assert.deepEqual(semantic(patched), semantic(manifest.inference.execution), 'Semantic ops, weights or layer scope changed');
assert.deepEqual(patched.decode, manifest.inference.execution.decode);
assert.deepEqual(patched.mechanismKernels, manifest.inference.execution.mechanismKernels);
expandExecutionV1(patched);
// Validate the exact effective graph at the manifest parser boundary as well.
parseManifest(JSON.stringify({ ...manifest, inference: { ...manifest.inference, execution: patched } }));
const candidate = structuredClone(input.runtimeConfig);
candidate.inference.executionPatch = patch;
candidate.inference.kernelPathPolicy = resolveRuntimeConfig(input.runtimeConfig).inference.kernelPathPolicy;
const path = candidate.inference.kernelPath;
const baselinePath = input.runtimeConfig.inference.kernelPath;
assert(path && path.activationDtype === 'f32' && path.kvDtype === 'f32');
const fullLayers = manifest.inference.layerPattern.layerTypes.flatMap((type, i) => type === 'full_attention' ? [i] : []);
assert.equal(fullLayers.length, 10, 'This preparation is scoped to the supplied ten-attention-layer program');
function replace(steps) {
  let count = 0;
  for (const step of steps ?? []) if (step.op === 'attention') {
    assert.equal(step.kernel, 'attention_small.wgsl', 'Unexpected baseline attention');
    Object.assign(step, declaration); count++;
  }
  assert.equal(count, 1, 'Expected one semantic attention op');
}
replace(path.prefill.steps);
for (const layer of fullLayers) {
  const override = path.layerOverrides.find(o => o.layers.includes(layer));
  assert(override?.prefill?.steps?.length, `Missing explicit prefill program for layer ${layer}`);
  assert(override.layers.every(i => fullLayers.includes(i)), 'Mixed full/linear override is outside this preparation');
  // Overrides may name more than one full layer; do not rewrite one twice.
  if (override.prefill.steps.some(s => s.op === 'attention' && s.kernel === 'attention_small.wgsl')) replace(override.prefill.steps);
}
path.id = `${baselinePath.id}-f32-prefill-query-tile-review`;
const routing = fullLayers.map(layer => {
  assert.deepEqual(getLayerSteps(path, layer, 'decode'), getLayerSteps(baselinePath, layer, 'decode'));
  const variant = getKernelPathAttentionVariant('prefill', layer, path);
  assert.equal(variant, 'prefill_query_tile_f32');
  return { layer, variant };
});
for (let layer = 0; layer < manifest.inference.layerPattern.layerTypes.length; layer++) {
  if (!fullLayers.includes(layer)) assert.deepEqual(getLayerSteps(path, layer, 'prefill'), getLayerSteps(baselinePath, layer, 'prefill'));
}
const compileOptions = { manifestInference: manifest.inference, modelId: manifest.modelId,
  numLayers: manifest.architecture.numLayers, headDim: 256, weightDtype: manifest.quantizationInfo?.weights ?? null, runtimeSession: candidate.inference.session,
  runtimeCompute: candidate.inference.compute, kernelPathPolicy: candidate.inference.kernelPathPolicy,
  capabilities: { hasF16: true, hasSubgroups: true, maxWorkgroupStorageSize: 32768 }, useGPU: true };
// Preparation must fail before writing a candidate if production compilation
// rejects either graph. An identical baseline error is not successful intake.
compileExecutionV1(compileOptions);
const compiled = compileExecutionV1({ ...compileOptions, executionPatch: patch });
for (const layer of fullLayers) assert.equal(getKernelPathAttentionVariant('prefill', layer, compiled.runtimeInferencePatch.kernelPath), 'prefill_query_tile_f32');
for (const key of ['kvDtype', 'activationDtype', 'accumDtype']) assert.equal(compiled.laneIntegrity.executed[key], 'f32');
await mkdir(outputDir, { recursive: true });
await writeFile(join(outputDir, 'runtime-candidate.json'), JSON.stringify(candidate, null, 2) + '\n');
await writeFile(join(outputDir, 'runtime-baseline.json'), JSON.stringify(input.runtimeConfig, null, 2) + '\n');
const receipt = { scope: 'Offline preparation, not physical qualification or activation', inputPath: resolve(inputPath),
  inputSha256: createHash('sha256').update(inputText).digest('hex'), sourceManifestSha256: fetched.manifestHash,
  sourceManifest: join(input.rdrrDir, 'manifest.json'), shader, digest, routing, laneIntegrity: compiled.laneIntegrity, compileBlocker: null,
  gates: { parseManifestSource: true, fetchManifestExactSourceBytes: true, parseManifestEffectiveCandidateGraph: true,
    expandExecutionV1: true, compileExecutionV1: compiled !== null, semanticOpsPreserved: true, decodePreserved: true, linearLayersPreserved: true },
  qualification: { physicalComponent: 'pending', fullModelAbs: 0.001, fullModelRel: 0, exactTokens: 'pending', quality: 'pending', speed: 'unmeasured' } };
await writeFile(join(outputDir, 'dry-run.json'), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
