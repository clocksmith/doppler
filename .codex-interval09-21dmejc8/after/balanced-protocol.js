import assert from 'node:assert/strict';
import { isDeepStrictEqual } from 'node:util';

export const PROFILE_PREFIX = '/dev/shm/doppler-remediation-22-profiles';
export const same = (left, right, label) => assert.deepEqual(left, right, label);

function withoutKernelPathLabels(inference) {
  if (!inference?.kernelPath) return inference;
  const kernelPath = { ...inference.kernelPath };
  for (const key of ['id', 'name', 'description']) delete kernelPath[key];
  return { ...inference, kernelPath };
}

function runtimeIdentity(runtime) {
  return runtime ? { ...runtime, inference: withoutKernelPathLabels(runtime.inference) } : runtime;
}

export function validateComparisonMode(mode) {
  assert(mode === undefined || mode === 'layer-submit-interval', 'Unknown comparisonMode');
}

function withoutLayerSubmitInterval(inference) {
  const copy = structuredClone(inference);
  delete copy.session.decodeLoop.layerSubmitInterval;
  return copy;
}

function validateIntervalValues(inferences) {
  same(inferences.map(value => value?.session?.decodeLoop?.layerSubmitInterval), [null, 1],
    'Interval comparison requires explicit ordered layerSubmitInterval null then 1');
}

export function validateVariantIntake(variants) {
  assert.equal(new Set(variants.map(value => value.id)).size, 2, 'Distinct variant IDs required');
  assert.equal(new Set(variants.map(value => value.configPath)).size, 2, 'Distinct config paths required');
  const identity = variant => ({ runtimeRoot: variant.runtimeRoot,
    runtimeConfig: runtimeIdentity(variant.config.runtimeConfig),
    runtimeConfigByKind: variant.config.runtimeConfigByKind
      ? Object.fromEntries(Object.entries(variant.config.runtimeConfigByKind).map(([kind, value]) => [kind, runtimeIdentity(value)])) : undefined,
    runtimeProfileByKind: variant.config.runtimeProfileByKind });
  assert(!isDeepStrictEqual(identity(variants[0]), identity(variants[1])), 'Semantically identical variants rejected');
}

export function validateResolvedInference(inferences, comparisonMode) {
  validateComparisonMode(comparisonMode);
  for (const inference of inferences) {
    assert.equal(inference?.session?.speculation?.mode, 'none', 'Actual speculation must be disabled');
    assert.equal(inference.session.speculation.tokens, 1, 'Actual speculative token budget must be one');
    assert.equal(inference.generation?.disableMultiTokenDecode, true, 'Actual multi-token decode must be disabled');
  }
  if (comparisonMode === 'layer-submit-interval') {
    validateIntervalValues(inferences);
    same(...inferences.map(withoutLayerSubmitInterval),
      'Resolved inference mismatch outside declared layerSubmitInterval');
    return;
  }
  const withoutKernelPath = inference => {
    const semantics = { ...inference };
    delete semantics.kernelPath;
    return semantics;
  };
  same(withoutKernelPath(inferences[0]), withoutKernelPath(inferences[1]),
    'Resolved inference mismatch outside declared kernelPath specialization');
}

export function validateDistinctResolvedVariants(variants, comparisonMode) {
  validateComparisonMode(comparisonMode);
  if (comparisonMode === 'layer-submit-interval') {
    same(variants[0].sourceHashes, variants[1].sourceHashes, 'Interval isolation requires identical source bytes');
  }
  assert(!isDeepStrictEqual(variants[0].sourceHashes, variants[1].sourceHashes)
    || !isDeepStrictEqual(withoutKernelPathLabels(variants[0].ready.resolvedRuntime.inference),
      withoutKernelPathLabels(variants[1].ready.resolvedRuntime.inference)),
  'Semantically identical resolved variants rejected');
}

export function validatePair(configs, contracts, comparisonMode) {
  validateComparisonMode(comparisonMode);
  assert.equal(configs.length, 2, 'Exactly two resident variants required');
  same(contracts[0], contracts[1], 'Variant input contracts must be identical');
  for (const key of ['modelId', 'rdrrDir', 'ggufPath', 'browserExecutable', 'browserArgs',
    'loadTimeoutMs', 'requestTimeoutMs', 'profileRoot', 'referenceProbe', 'recordWeightLayers']) {
    same(configs[0][key], configs[1][key], 'Variant mismatch: ' + key);
  }
  if (comparisonMode === 'layer-submit-interval') {
    validateIntervalValues(configs.map(config => config.runtimeConfig?.inference));
    const runtimeWithoutInterval = config => ({ ...config.runtimeConfig,
      inference: withoutLayerSubmitInterval(config.runtimeConfig.inference) });
    same(...configs.map(runtimeWithoutInterval), 'Runtime config mismatch outside declared layerSubmitInterval');
    same(configs[0].runtimeConfigByKind, configs[1].runtimeConfigByKind, 'Per-kind runtime overrides must match');
    same(configs[0].runtimeProfileByKind, configs[1].runtimeProfileByKind, 'Runtime profiles must match');
  }
  for (const config of configs) {
    same(config.kinds, ['doppler'], 'Each browser must remain kind=doppler');
    assert.equal(config.profileRoot, PROFILE_PREFIX, 'Owning lock wrapper requires the existing profile prefix');
    assert(Number.isFinite(config.requestTimeoutMs) && config.requestTimeoutMs > 0);
    for (const key of ['countGpuApi', 'profileGeneration', 'profileRepeats', 'captureLogitPositions',
      'referenceStateLayers', 'prefixDiagnostic', 'pageGlobals']) {
      assert(!config[key], 'Acceptance disallows observation/overrides: ' + key);
    }
    assert(config.referenceProbe, 'Both variants require the existing reference probe');
    assert.equal(config.runtimeConfig?.inference?.generation?.disableMultiTokenDecode, true);
    const probe = config.referenceProbe;
    assert.equal(probe.contract.shared.maxNewTokens, 2);
    assert.equal(probe.observation.referenceDecodeSteps, 1);
    for (const key of ['contextTokens', 'sampling', 'promptFormat', 'eosTokenIds', 'requiredVendor']) {
      same(probe.contract.shared[key], contracts[0].shared[key], 'Reference changes shared field: ' + key);
    }
    for (const section of ['compute', 'kvcache', 'generation', 'sampling', 'moe']) {
      same(config.runtimeConfig?.inference?.[section], configs[0].runtimeConfig?.inference?.[section],
        'Runtime semantic contract mismatch: ' + section);
    }
    const session = value => comparisonMode === 'layer-submit-interval'
      ? withoutLayerSubmitInterval(value.runtimeConfig.inference).session : value.runtimeConfig?.inference?.session;
    same(session(config), session(configs[0]), 'Session precision/cache contract mismatch');
  }
  const shared = contracts[0].shared;
  assert.equal(shared.maxNewTokens, 128);
  assert.equal(shared.speculation, false);
  assert.equal(shared.qualityTasks.length, 4);
  same(shared.throughputTasks.map(task => task.id), ['short', 'long']);
  for (const task of [...shared.qualityTasks, ...shared.throughputTasks]) assert(!task.profile);
}

export function acceptanceContract(source) {
  const contract = structuredClone(source);
  contract.scope = 'Two resident Doppler variants; unchanged weights; full numerical and 4/4 quality gates precede balanced acceptance timing';
  contract.shared.warmupRuns = 1;
  contract.shared.pairsPerPrompt = 10;
  return contract;
}

export function requestMetrics(result, timed) {
  assert(!result.error, result.error);
  assert.equal(result.profiled, false, 'Profiling forbidden during workload');
  assert.equal(result.apiSnapshots, null, 'API counters forbidden during workload');
  assert.equal(result.apiFinal, null);
  assert.equal(result.gpuErrors.length, 0);
  assert(result.completeMs > 0 && Number.isFinite(result.completeMs));
  assert.equal(result.tokenIds.length, result.deliveryMs.length);
  assert.equal(result.chunks.length, result.textDeliveryMs.length);
  assert(result.deliveryMs.every((time, index, values) => Number.isFinite(time) && time >= 0
    && (index === 0 || time >= values[index - 1])));
  assert(result.textDeliveryMs.every((time, index, values) => Number.isFinite(time) && time >= 0
    && (index === 0 || time >= values[index - 1])));
  if (timed) {
    assert.equal(result.tokenIds.length, 128, 'Require 128 accepted output IDs, not draft tokens');
    assert.equal(result.stopReason, 'max-tokens');
  }
  const deliveredSpanMs = result.deliveryMs.at(-1) - result.deliveryMs[0];
  return { acceptedOutputTokens: result.tokenIds.length, promptTokens: result.promptIds.length,
    totalRequestOutputTokensPerSecond: result.tokenIds.length * 1000 / result.completeMs,
    postFirstTokenDeliveredTokensPerSecond: deliveredSpanMs > 0
      ? (result.tokenIds.length - 1) * 1000 / deliveredSpanMs : null,
    firstTokenMs: result.firstTokenMs, firstTextMs: result.firstTextMs,
    reportedPrefillMs: result.prefillMs, totalRequestMs: result.completeMs, deliveredSpanMs };
}

export async function runWorkload({ variants, contract, numericalGates, request, phase }) {
  for (const variant of variants) assert.equal(numericalGates?.[variant.id]?.numericalPassed, true,
    'Numerical gate required before quality: ' + variant.id);
  for (const task of contract.shared.qualityTasks) {
    const results = [];
    for (const variant of variants) {
      await phase('quality', { variant: variant.id, task: task.id });
      const result = await request(variant, task, { phase: 'quality', repeat: 0, warmup: false });
      requestMetrics(result, false);
      assert.equal(result.qualityPass, true, variant.id + ' quality rejected: ' + task.id);
      results.push(result);
    }
    same(results[0].promptIds, results[1].promptIds, 'Quality prompt ID mismatch: ' + task.id);
    same(results[0].tokenIds, results[1].tokenIds, 'Quality accepted output ID mismatch: ' + task.id);
  }
  await phase('quality-passed', { tasksPerVariant: 4 });
  for (const task of contract.shared.throughputTasks) {
    let frozenIds = null;
    for (let repeat = 0; repeat <= 10; repeat++) {
      const warmup = repeat === 0;
      const order = warmup || repeat % 2 === 1 ? variants : [...variants].reverse();
      const results = new Map();
      for (const variant of order) {
        await phase(warmup ? 'warmup' : 'timing', { variant: variant.id, task: task.id,
          repeat, pair: warmup ? null : repeat - 1, order: order.map(value => value.id) });
        const result = await request(variant, task, { phase: warmup ? 'warmup' : 'timing', repeat, warmup });
        requestMetrics(result, true);
        if (frozenIds) {
          same(result.promptIds, frozenIds.promptIds, 'Prompt IDs changed across resident requests: ' + task.id);
          same(result.tokenIds, frozenIds.tokenIds, 'Accepted IDs changed across resident requests: ' + task.id);
        } else frozenIds = { promptIds: [...result.promptIds], tokenIds: [...result.tokenIds] };
        results.set(variant.id, result);
      }
      same(results.get(variants[0].id).promptIds, results.get(variants[1].id).promptIds,
        'Paired prompt ID mismatch: ' + task.id + '/' + repeat);
      same(results.get(variants[0].id).tokenIds, results.get(variants[1].id).tokenIds,
        'Paired accepted output ID mismatch: ' + task.id + '/' + repeat);
    }
  }
}
