import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs/promises';

import {
  buildLocalGpuChallengerReport,
  validateLocalGpuChallengerSchema,
  validateLocalGpuChallengerMatrix,
} from '../../tools/local-gpu-challengers.js';

const matrix = JSON.parse(
  await fs.readFile(new URL('../../benchmarks/vendors/local-gpu-challenger-matrix.json', import.meta.url), 'utf8')
);
const schema = JSON.parse(
  await fs.readFile(new URL('../../benchmarks/vendors/schema/local-gpu-challenger-matrix.schema.json', import.meta.url), 'utf8')
);
const catalog = JSON.parse(
  await fs.readFile(new URL('../../models/catalog.json', import.meta.url), 'utf8')
);

const validation = validateLocalGpuChallengerMatrix(matrix, catalog, schema);
assert.deepEqual(validation.errors, []);
assert.equal(validation.ok, true);

const schemaProbe = structuredClone(matrix);
schemaProbe.unexpected = true;
assert.match(
  validateLocalGpuChallengerSchema(schemaProbe, schema).join('\n'),
  /unexpected property "unexpected"/,
  'schema validator must enforce additionalProperties'
);

const check = spawnSync(process.execPath, ['tools/local-gpu-challengers.js', '--check'], {
  cwd: process.cwd(),
  encoding: 'utf8',
  stdio: ['ignore', 'pipe', 'pipe'],
});
assert.equal(check.status, 0, `local GPU challenger check failed\nstdout:\n${check.stdout}\nstderr:\n${check.stderr}`);

const report = buildLocalGpuChallengerReport(matrix, catalog, { schema });
assert.equal(report.ok, true);
assert.equal(report.summary.models, 8);
assert.equal(report.hostClass, 'multi-platform-local-gpu');
assert.equal(report.probeHostClass, 'linux-amd-vulkan-rocm-local');
assert.equal(report.summary.platformTargets, 8);
assert.equal(report.summary.tierCounts['tier-0'], 3);
assert.equal(report.summary.tierCounts['tier-1'], 3);
assert.equal(report.summary.tierCounts['tier-2'], 2);

const expectedModelIds = [
  'gemma-3-1b-it-q4k-ehf16-af32',
  'gemma-3-270m-it-q4k-ehf16-af32',
  'gemma-4-e2b-it-q4k-ehf16-af16-int4ple',
  'google-embeddinggemma-300m-q4k-ehf16-af32',
  'qwen-3-5-0-8b-q4k-ehaf16',
  'qwen-3-5-2b-q4k-ehaf16',
  'qwen-3-embedding-0-6b-q4k-ehf16-af32',
  'qwen-3-reranker-0-6b-q4k-ehf16-af32',
];
assert.deepEqual(report.rows.map((row) => row.modelId).sort(), expectedModelIds);

const requiredFairnessGates = [
  'artifact-identity',
  'format-disclosure',
  'runtime-surface',
  'hardware-identity',
  'fallback-status',
  'cache-mode',
  'timing-scope',
  'correctness-first',
  'work-accounting',
  'sample-statistics',
  'claim-grade',
];
for (const harness of matrix.harnesses) {
  for (const gateId of requiredFairnessGates) {
    assert.equal(harness.fairnessGates.includes(gateId), true, `${harness.id}: missing ${gateId}`);
  }
  assert.deepEqual(harness.claimGrades, matrix.selectionPolicy.claimGradeOrder, `${harness.id}: claim grades`);
  assert.equal(
    harness.engineOverlayPolicy.forbiddenSharedFields.length > 0,
    true,
    `${harness.id}: forbidden shared fields must stay explicit`
  );
}

const generationHarness = matrix.harnesses.find((entry) => entry.id === 'generation-local-gpu-v1');
assert.ok(generationHarness, 'generation harness must exist');
for (const field of ['kernelPath', 'readbackInterval', 'batchSize', 'providerSessionOptions']) {
  assert.equal(
    generationHarness.engineOverlayPolicy.forbiddenSharedFields.includes(field),
    true,
    `generation harness must keep ${field} out of the shared contract`
  );
}

const embeddingHarness = matrix.harnesses.find((entry) => entry.id === 'embedding-local-gpu-v1');
const rerankHarness = matrix.harnesses.find((entry) => entry.id === 'rerank-local-gpu-v1');
assert.equal(embeddingHarness.sharedContract.sampling, null, 'embedding harness must not fake a generation sampling contract');
assert.equal(rerankHarness.sharedContract.sampling, null, 'rerank harness must not fake a generation sampling contract');
assert.equal(embeddingHarness.metrics.includes('modelLoadMs'), true, 'embedding timing must include model load');
assert.equal(embeddingHarness.metrics.includes('readbackMs'), true, 'embedding timing must include readback');
assert.equal(rerankHarness.metrics.includes('modelLoadMs'), true, 'rerank timing must include model load');
assert.equal(rerankHarness.metrics.includes('readbackMs'), true, 'rerank timing must include readback');

const byModelId = new Map(report.rows.map((row) => [row.modelId, row]));
const byCompetitorId = new Map(matrix.competitors.map((entry) => [entry.id, entry]));
const platformTargetIds = new Set(report.platformTargets.map((entry) => entry.id));

for (const platformId of [
  'apple-metal',
  'linux-amd-vulkan-rocm',
  'linux-nvidia-vulkan-cuda',
  'linux-intel-vulkan',
  'windows-amd-webgpu-directml',
  'windows-nvidia-webgpu-cuda',
  'windows-intel-webgpu',
  'nvidia-orin-spark-linux',
]) {
  assert.equal(platformTargetIds.has(platformId), true, `platform target missing: ${platformId}`);
}

assert.equal(
  byCompetitorId.get('hf-transformers-rocm')?.availability,
  'blocked-pytorch-rocm-unavailable',
  'HF Transformers ROCm must stay blocked until the local torch build is ROCm-enabled'
);

for (const row of report.rows) {
  assert.equal(row.anchorComparator.competitorId, 'transformersjs-webgpu', `${row.modelId}: TJS anchor`);
  assert.equal(row.localChallengers.length >= matrix.selectionPolicy.minAdditionalLocalChallengers, true);
  assert.equal(
    row.localChallengers.some((challenger) => challenger.competitorId === 'transformersjs-webgpu'),
    false,
    `${row.modelId}: local challenger list must not duplicate the TJS anchor`
  );
  assert.equal(row.minimumClaimGrade, 'local-gpu-comparable', `${row.modelId}: claim floor`);
}

assert.deepEqual(
  byModelId.get('gemma-3-270m-it-q4k-ehf16-af32').localChallengers.map((entry) => entry.competitorId),
  ['onnxruntime-webgpu-direct', 'llamacpp-vulkan-gguf']
);
assert.deepEqual(
  byModelId.get('gemma-3-1b-it-q4k-ehf16-af32').localChallengers.map((entry) => entry.competitorId),
  ['onnxruntime-webgpu-direct', 'llamacpp-vulkan-gguf']
);
assert.deepEqual(
  byModelId.get('google-embeddinggemma-300m-q4k-ehf16-af32').localChallengers.map((entry) => entry.competitorId),
  ['onnxruntime-webgpu-direct', 'hf-transformers-rocm']
);
assert.deepEqual(
  byModelId.get('qwen-3-5-0-8b-q4k-ehaf16').localChallengers.map((entry) => entry.competitorId),
  ['onnxruntime-webgpu-direct', 'llamacpp-vulkan-gguf']
);
assert.deepEqual(
  byModelId.get('qwen-3-5-2b-q4k-ehaf16').localChallengers.map((entry) => entry.competitorId),
  ['onnxruntime-webgpu-direct', 'llamacpp-vulkan-gguf']
);
assert.deepEqual(
  byModelId.get('qwen-3-embedding-0-6b-q4k-ehf16-af32').localChallengers.map((entry) => entry.competitorId),
  ['onnxruntime-webgpu-direct', 'hf-transformers-rocm']
);
assert.deepEqual(
  byModelId.get('qwen-3-reranker-0-6b-q4k-ehf16-af32').localChallengers.map((entry) => entry.competitorId),
  ['hf-transformers-rocm', 'onnxruntime-webgpu-direct']
);
assert.equal(
  byModelId.get('qwen-3-reranker-0-6b-q4k-ehf16-af32').anchorComparator.status,
  'configured-artifact',
  'reranker TJS anchor must not imply performance evidence'
);
for (const modelId of [
  'google-embeddinggemma-300m-q4k-ehf16-af32',
  'qwen-3-embedding-0-6b-q4k-ehf16-af32',
  'qwen-3-reranker-0-6b-q4k-ehf16-af32',
]) {
  const challenger = byModelId
    .get(modelId)
    .localChallengers.find((entry) => entry.competitorId === 'hf-transformers-rocm');
  assert.equal(challenger.status, 'blocked-pytorch-rocm-unavailable', `${modelId}: ROCm torch status`);
  assert.equal(challenger.nextGate, 'install-rocm-enabled-torch', `${modelId}: ROCm torch next gate`);
}

const gemmaE2b = byModelId.get('gemma-4-e2b-it-q4k-ehf16-af16-int4ple');
assert.ok(gemmaE2b, 'Gemma E2B selected artifact must be AF16 INT4 PLE');
assert.deepEqual(gemmaE2b.localChallengers.map((entry) => entry.competitorId), ['litert-gpu', 'hf-transformers-rocm']);
assert.deepEqual(gemmaE2b.alternateDopplerArtifactIds, ['gemma-4-e2b-it-q4k-ehf16-af32-int4ple']);
assert.equal(byModelId.has('gemma-4-e2b-it-q4k-ehf16-af32-int4ple'), false);
const gemmaRocmChallenger = gemmaE2b.localChallengers.find((entry) => entry.competitorId === 'hf-transformers-rocm');
assert.equal(gemmaRocmChallenger.status, 'blocked-pytorch-rocm-unavailable', 'Gemma E2B ROCm torch status');
assert.equal(gemmaRocmChallenger.nextGate, 'install-rocm-enabled-torch', 'Gemma E2B ROCm torch next gate');

console.log('local-gpu-challenger-matrix-contract.test: ok');
