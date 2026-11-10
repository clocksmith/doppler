import assert from 'node:assert/strict';
import { readGpuTimings } from '../../demo/ui/xray/gpu-timings.js';
import { listQuickstartModels } from '../../src/client/doppler-registry.js';
import { isChatModel, buildModelDetail, modelFit } from '../../demo/models.js';
import { fieldText } from '../../demo/settings.js';

const rows = await listQuickstartModels();
const displayed = rows.find((entry) => entry.label && Number.isFinite(entry.sizeBytes) && entry.quickstartRequirements);
assert.ok(displayed, 'The public registry must retain human labels, byte sizes and model requirements');
assert.ok(buildModelDetail(displayed).includes('Your browser keeps a copy if it can.'));
displayed.quickstartRequirements.paramCount = 'mutated by consumer';
assert.notEqual((await listQuickstartModels()).find((entry) => entry.modelId === displayed.modelId).quickstartRequirements.paramCount, 'mutated by consumer');

assert.equal(isChatModel({ modes: ['text'], classification: { tasks: ['generation'] } }), true);
assert.equal(isChatModel({ modes: ['embedding'], classification: { tasks: ['embedding'] } }), false);
assert.equal(isChatModel({ modes: ['text'], classification: { tasks: ['embedding'] } }), false);
assert.equal(isChatModel({ modes: ['image'], classification: { tasks: ['generation'] } }), false);
assert.equal(isChatModel({}), false, 'An unknown capability cannot be offered as a chat model');
assert.equal(buildModelDetail({ modelId: 'fixture', label: 'Fixture', sizeBytes: 418518521 }),
  'First use downloads about 419 MB. Your browser keeps a copy if it can.');
assert.equal(buildModelDetail({ modelId: 'fixture', sizeBytes: 418518521 }, 'stored'),
  'Saved in this browser, so it loads without downloading.');
assert.equal(buildModelDetail({ modelId: 'fixture', sizeBytes: 418518521 }, 'loaded'), '', 'The status chip alone reports a loaded model');

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



// The picker compares the catalog's stated memory need with what the device reports, before any download.
const GIB = 2 ** 30;
const large = { modelId: 'large', sizeBytes: 7094076955, quickstartRequirements: { vramRequiredBytes: 12 * GIB } };
assert.equal(modelFit(large, 4 * GIB), 'too-large');
assert.equal(modelFit(large, 8 * GIB), 'unknown', 'Browsers cap the reported figure at 8 GiB, so more may be present');
assert.equal(modelFit(large, null), 'unknown', 'A browser that reports nothing gets no verdict');
assert.equal(modelFit({ modelId: 'small', quickstartRequirements: { vramRequiredBytes: GIB } }, 4 * GIB), 'fits');
assert.equal(modelFit({ modelId: 'unstated' }, 4 * GIB), 'unknown');
assert.equal(buildModelDetail(large, 'available', 4 * GIB),
  'Needs about 12 GB of GPU memory, and this device reports 4 GB in total, so it will probably fail to load. Loading it downloads about 7.1 GB first.');
assert.equal(buildModelDetail(large, 'available', null),
  'First use downloads about 7.1 GB. Your browser keeps a copy if it can. Needs about 12 GB of GPU memory.');

// A setting stored as a 32-bit float is shown as the number that was entered.
assert.equal(fieldText({ type: 'number', step: '0.05' }, Math.fround(0.95)), '0.95');
assert.equal(String(Math.fround(0.95)), '0.949999988079071', 'This is the value the field used to show');
assert.equal(fieldText({ type: 'number', step: '0.1' }, Math.fround(0.7)), '0.7');
assert.equal(fieldText({ type: 'number', step: '1' }, 50), '50');
assert.equal(fieldText({ type: 'number', step: 'any' }, 0.123456), '0.123456');
assert.equal(fieldText({ type: 'select-one', step: undefined }, 'profiles/default'), 'profiles/default');
console.log('inspection-presentation.test: ok');
