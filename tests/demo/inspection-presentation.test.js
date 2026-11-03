import assert from 'node:assert/strict';
import { readGpuTimings } from '../../demo/ui/xray/gpu-timings.js';
import { listQuickstartModels } from '../../src/client/doppler-registry.js';
import { isChatModel, buildModelCardDetail } from '../../demo/models.js';

const rows = await listQuickstartModels();
const displayed = rows.find((entry) => entry.label && Number.isFinite(entry.sizeBytes) && entry.quickstartRequirements);
assert.ok(displayed, 'The public registry must retain human labels, byte sizes and model requirements');
assert.ok(buildModelCardDetail(displayed).includes('cached in this browser when available'));
displayed.quickstartRequirements.paramCount = 'mutated by consumer';
assert.notEqual((await listQuickstartModels()).find((entry) => entry.modelId === displayed.modelId).quickstartRequirements.paramCount, 'mutated by consumer');

assert.equal(isChatModel({ modes: ['text'], classification: { tasks: ['generation'] } }), true);
assert.equal(isChatModel({ modes: ['embedding'], classification: { tasks: ['embedding'] } }), false);
assert.equal(isChatModel({ modes: ['text'], classification: { tasks: ['embedding'] } }), false);
assert.equal(isChatModel({ modes: ['image'], classification: { tasks: ['generation'] } }), false);
assert.equal(isChatModel({}), false, 'An unknown capability cannot be offered as a chat model');
assert.equal(buildModelCardDetail({ modelId: 'fixture', label: 'Fixture', sizeBytes: 418518521 }),
  'fixture · First run downloads about 419 MB; cached in this browser when available.');

const receipt = { generationEvidence: { stats: {
  prefillTimeMs: 50, gpuTimeDecodeMs: 4,
  prefillProfileSteps: [{ recorderStats: { opLabelCounts: { matmul: 100 } } }],
  decodeProfileSteps: [
    { step: 1, timings: { fused_ffn: 1.96, rmsnorm: .04 } },
    { step: 2, timings: { fused_ffn: 1, rmsnorm: 1, measured_zero: 0, invalid: -1, unknown: NaN } },
  ],
} } };
const before = JSON.stringify(receipt), result = readGpuTimings(receipt);
assert.equal(result.phases[0].totalMs, null, 'Wall time and operation counts cannot become GPU measurements');
assert.equal(result.phases[0].missingSteps, 1);
assert.equal(result.phases[1].totalMs, 4);
assert.equal(result.phases[1].steps[0].totalMs, 2);
assert.deepEqual(result.kernels.find((kernel) => kernel.label === 'fused_ffn'), { label: 'fused_ffn', prefill: null, decode: 2.96 });
assert.equal(result.kernels.find((kernel) => kernel.label === 'measured_zero').decode, 0, 'Measured zero remains distinct from unavailable');
assert.equal(result.kernels.some((kernel) => kernel.label === 'invalid' || kernel.label === 'unknown'), false);
assert.equal(JSON.stringify(receipt), before, 'Presentation must not rewrite a recorded receipt');
assert.equal(readGpuTimings({}).phases[1].totalMs, null);
console.log('inspection-presentation.test: ok');
