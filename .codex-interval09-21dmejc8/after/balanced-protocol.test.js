import assert from 'node:assert/strict';
import { PROFILE_PREFIX, acceptanceContract, requestMetrics, runWorkload, validatePair,
  validateVariantIntake, validateResolvedInference, validateDistinctResolvedVariants } from './balanced-protocol.js';

const contract = acceptanceContract({ shared: { qualityTasks: ['a', 'b', 'c', 'd'].map(id => ({ id, expected: 'ok' })),
  throughputTasks: ['short', 'long'].map(id => ({ id })) } });
const variants = [{ id: 'a' }, { id: 'b' }];
const gates = { a: { numericalPassed: true }, b: { numericalPassed: true } };
const calls = [];
let active = 0;
const fakeResult = (task, timed) => ({ promptIds: [1, 2, 3], tokenIds: Array(timed ? 128 : 1).fill(4),
  deliveryMs: Array.from({ length: timed ? 128 : 1 }, (_, index) => 10 + index * 2),
  chunks: ['ok'], textDeliveryMs: [12], completeMs: 300, firstTokenMs: 10,
  firstTextMs: 12, prefillMs: 8, gpuErrors: [], qualityPass: true, profiled: false,
  apiSnapshots: null, apiFinal: null, stopReason: timed ? 'max-tokens' : 'eos' });
const request = async (variant, task, metadata) => {
  assert.equal(active++, 0, 'Exactly one request may execute at once');
  await new Promise(resolve => setImmediate(resolve));
  calls.push({ id: variant.id, task: task.id, ...metadata });
  active--;
  return fakeResult(task, metadata.phase !== 'quality');
};
const phase = async () => {};
await assert.rejects(runWorkload({ variants, contract, numericalGates: { a: gates.a }, request, phase }), /Numerical gate/);
assert.equal(calls.length, 0, 'Missing second numerical gate prevents even quality requests');
await runWorkload({ variants, contract, numericalGates: gates, request, phase });
assert.equal(calls.length, 52, 'Eight quality requests plus two tasks with 22 warm/timed requests');
assert(calls.slice(0, 8).every(value => value.phase === 'quality'));
for (const task of ['short', 'long']) {
  const taskCalls = calls.filter(value => value.task === task);
  assert.equal(taskCalls.filter(value => value.warmup).length, 2);
  const timed = taskCalls.filter(value => value.phase === 'timing');
  assert.equal(timed.length, 20);
  assert.equal(timed.filter(value => value.id === 'a').length, 10);
  for (let pair = 0; pair < 10; pair++) {
    assert.deepEqual(timed.slice(pair * 2, pair * 2 + 2).map(value => value.id), pair % 2 ? ['b', 'a'] : ['a', 'b']);
  }
}
const metrics = requestMetrics(fakeResult({}, true), true);
assert.equal(metrics.totalRequestOutputTokensPerSecond, 128000 / 300);
assert.equal(metrics.postFirstTokenDeliveredTokensPerSecond, 127000 / 254);
let afterRejectedQuality = 0;
await assert.rejects(runWorkload({ variants, contract, numericalGates: gates, phase,
  request: async (_variant, task) => { afterRejectedQuality++; return { ...fakeResult(task, false), qualityPass: false }; } }),
  /quality rejected/);
assert.equal(afterRejectedQuality, 1, 'Quality rejection prevents timing');
assert.throws(() => requestMetrics({ ...fakeResult({}, true), tokenIds: [1] }, true));
assert.throws(() => requestMetrics({ ...fakeResult({}, true), profiled: true }, true), /Profiling forbidden/);
const shared = { contextTokens: 4096, maxNewTokens: 128, speculation: false, sampling: { temperature: 0 },
  promptFormat: {}, eosTokenIds: [0], requiredVendor: 'amd', qualityTasks: contract.shared.qualityTasks,
  throughputTasks: contract.shared.throughputTasks };
const config = { kinds: ['doppler'], profileRoot: PROFILE_PREFIX, requestTimeoutMs: 1000,
  referenceProbe: { contract: { shared: { ...shared, maxNewTokens: 2 } }, observation: { referenceDecodeSteps: 1 } },
  runtimeConfig: { inference: { generation: { disableMultiTokenDecode: true } } } };
validatePair([config, structuredClone(config)], [{ shared }, { shared }]);
assert.throws(() => validatePair([config, { ...config, countGpuApi: true }], [{ shared }, { shared }]), /Acceptance disallows/);
assert.throws(() => validatePair([config, { ...config, browserArgs: ['different'] }], [{ shared }, { shared }]), /Variant mismatch/);
const baseVariant = { id: 'a', configPath: '/a.json', runtimeRoot: '/same-source', config };
const headVariant = { id: 'b', configPath: '/b.json', runtimeRoot: '/same-source',
  config: { ...config, runtimeConfig: { ...config.runtimeConfig, inference: {
    ...config.runtimeConfig.inference, kernelPath: { postLayer: [{ op: 'lm_head', kernel: 'gemv' }] } } } } };
validateVariantIntake([baseVariant, headVariant]);
assert.throws(() => validateVariantIntake([baseVariant, { ...baseVariant, id: 'b', configPath: '/b.json' }]),
  /Semantically identical/);
const inference = { session: { speculation: { mode: 'none', tokens: 1 } },
  generation: { disableMultiTokenDecode: true }, batching: { batchSize: 1 },
  executionPatch: { set: [] }, kernelPath: { id: 'a', postLayer: [{ kernel: 'tiled' }] }, kernelPathSource: 'config' };
validateResolvedInference([inference, { ...inference, kernelPath: { id: 'b', postLayer: [{ kernel: 'gemv' }] } }]);
assert.throws(() => validateResolvedInference([inference, { ...inference,
  session: { speculation: { mode: 'draft', tokens: 5 } } }]), /Actual speculation/);
assert.throws(() => validateResolvedInference([inference, { ...inference, batching: { batchSize: 2 } }]), /Resolved inference mismatch/);
assert.throws(() => validateResolvedInference([inference, { ...inference, unknownFutureExecutionFlag: true }]), /Resolved inference mismatch/);
const resolved = { sourceHashes: { 'src/a.js': 'same' }, ready: { resolvedRuntime: { inference } } };
assert.throws(() => validateDistinctResolvedVariants([resolved, structuredClone(resolved)]), /Semantically identical resolved/);
validateDistinctResolvedVariants([resolved, { ...resolved, ready: { resolvedRuntime: {
  inference: { ...inference, kernelPath: { id: 'b', postLayer: [{ kernel: 'gemv' }] } } } } }]);
assert.throws(() => validateDistinctResolvedVariants([resolved, { ...resolved, ready: { resolvedRuntime: {
  inference: { ...inference, kernelPath: { ...inference.kernelPath, id: 'only-a-new-label' } } } } }]),
  /Semantically identical resolved/);
let throughputCalls = 0;
await assert.rejects(runWorkload({ variants, contract, numericalGates: gates, phase,
  request: async (_variant, task, metadata) => {
    const result = fakeResult(task, metadata.phase !== 'quality');
    if (metadata.phase !== 'quality' && ++throughputCalls === 2) result.tokenIds[0] = 99;
    return result;
  } }), /Accepted IDs changed/);
console.log('balanced protocol: both numerical gates, all quality before timing, sequential AB/BA, exact128 and distinct denominators passed');
const intervalMode = 'layer-submit-interval';
const intervalInference = interval => ({ ...structuredClone(inference), session: {
  ...structuredClone(inference.session), decodeLoop: { layerSubmitInterval: interval, batchSize: 1 } } });
const intervalConfig = interval => ({ ...structuredClone(config), runtimeConfig: { inference: {
  ...structuredClone(config.runtimeConfig.inference), session: { decodeLoop: { layerSubmitInterval: interval, batchSize: 1 } } } } });
const intervalConfigs = [intervalConfig(null), intervalConfig(1)];
validatePair(intervalConfigs, [{ shared }, { shared }], intervalMode);
validateResolvedInference([intervalInference(null), intervalInference(1)], intervalMode);
validateVariantIntake(intervalConfigs.map((config, index) => ({ ...baseVariant, id: String(index), configPath: '/' + index, config })));
for (const wrong of [undefined, null, 0, 2, 1.5, '1']) {
  assert.throws(() => validateResolvedInference([intervalInference(null), intervalInference(wrong)], intervalMode), /explicit ordered/);
  assert.throws(() => validatePair([intervalConfig(null), intervalConfig(wrong)], [{ shared }, { shared }], intervalMode), /explicit ordered/);
}
assert.throws(() => validateResolvedInference([intervalInference(1), intervalInference(null)], intervalMode), /explicit ordered/);
const missingInterval = intervalInference(null);
delete missingInterval.session.decodeLoop.layerSubmitInterval;
assert.throws(() => validateResolvedInference([missingInterval, intervalInference(1)], intervalMode), /explicit ordered/);
for (const mutate of [
  value => { value.batching.batchSize = 2; },
  value => { value.session.kvcache = { kvDtype: 'f16' }; },
  value => { value.session.compute = { defaults: { activationDtype: 'f16' } }; },
  value => { value.executionPatch.set.push({ op: 'change' }); },
  value => { value.session.speculation = { mode: 'draft', tokens: 2 }; },
  value => { value.kernelPath.postLayer[0].kernel = 'different'; },
  value => { value.session.decodeLoop.batchSize = 2; },
  value => { value.unknownFutureExecutionFlag = true; },
]) {
  const changed = intervalInference(1);
  mutate(changed);
  assert.throws(() => validateResolvedInference([intervalInference(null), changed], intervalMode));
}
for (const mutate of [
  value => { value.runtimeConfig.inference.session.kvcache = { kvDtype: 'f16' }; },
  value => { value.runtimeConfig.inference.kernelPath = { id: 'different' }; },
  value => { value.runtimeConfig.shared = { debug: true }; },
  value => { value.runtimeConfigByKind = { doppler: { inference: { generation: { maxTokens: 2 } } } }; },
  value => { value.runtimeProfileByKind = { doppler: 'other' }; },
]) {
  const changed = intervalConfig(1);
  mutate(changed);
  assert.throws(() => validatePair([intervalConfig(null), changed], [{ shared }, { shared }], intervalMode));
}
const intervalResolved = interval => ({ sourceHashes: { 'src/a.js': 'same' }, ready: {
  resolvedRuntime: { inference: intervalInference(interval) } } });
validateDistinctResolvedVariants([intervalResolved(null), intervalResolved(1)], intervalMode);
assert.throws(() => validateDistinctResolvedVariants([intervalResolved(null), { ...intervalResolved(1),
  sourceHashes: { 'src/a.js': 'different' } }], intervalMode), /identical source/);
assert.throws(() => validatePair(intervalConfigs, [{ shared }, { shared }], 'typo'), /Unknown comparisonMode/);
assert.throws(() => validateResolvedInference([intervalInference(null), intervalInference(1)]), /Resolved inference mismatch/);
console.log('interval isolation: explicit null/1, source equality, strict remaining config and READY semantics passed');
