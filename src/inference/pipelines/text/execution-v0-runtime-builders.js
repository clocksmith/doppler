import { selectRuleValue } from '../../../rules/rule-registry.js';
import { cloneJson, isPhaseMatch, normalizeDtype, requireSessionActivationDtype, stepHasLayer } from './execution-v0-contract-helpers.js';

const PIPELINE_COMPATIBLE_OPS = new Set([
  'save',
  'load',
  'conv',
  'attention',
  'rmsnorm',
  'ffn',
  'residual_add',
  'cast',
  'noop',
]);

function toKernelPathStep(step) {
  if (step.op === 'cast') return null;
  if (!step.kernel) return null;
  return {
    op: step.op,
    kernel: step.kernel,
    entry: step.entry ?? 'main',
    ...(step.weights ? { weights: step.weights } : {}),
    ...(step.constants ? { constants: step.constants } : {}),
  };
}

function getSectionSteps(steps, section, phase = null) {
  return steps
    .filter((step) => step.section === section)
    .filter((step) => (phase ? isPhaseMatch(step.phase, phase) : true))
    .map(toKernelPathStep)
    .filter((step) => step != null);
}

function buildLayerPhaseSteps(steps, phase, layerIdx) {
  return steps
    .filter((step) => step.section === 'layer' && isPhaseMatch(step.phase, phase))
    .filter((step) => stepHasLayer(step, layerIdx))
    .map(toKernelPathStep)
    .filter((step) => step != null);
}

function getInlineKernelPathSteps(path) {
  return [
    ...(path?.preLayer ?? []),
    ...(path?.decode?.steps ?? []),
    ...(path?.prefill?.steps ?? []),
    ...(path?.postLayer ?? []),
    ...(path?.sampling ?? []),
    ...(path?.layerOverrides?.flatMap((override) => override.steps ?? []) ?? []),
  ];
}

function assertInlineKernelPathSessionCompatibility(path, sessionDefaults) {
  if (!path) {
    return;
  }
  const activationDtype = normalizeDtype(
    path.activationDtype ?? requireSessionActivationDtype(sessionDefaults),
    'inlineKernelPath.activationDtype'
  );
  const kvDtype = normalizeDtype(
    path.kvDtype ?? sessionDefaults?.kvcache?.kvDtype ?? activationDtype,
    'inlineKernelPath.kvDtype'
  );

  for (const step of getInlineKernelPathSteps(path)) {
    const kernel = String(step?.kernel ?? '').trim();
    if (!kernel.startsWith('attention')) {
      continue;
    }
    if (kernel.includes('_f16kv')) {
      if (activationDtype !== 'f32' || kvDtype !== 'f16') {
        throw new Error(
          `[ExecutionV0] Inline kernelPath attention kernel "${kernel}" requires ` +
          `activationDtype="f32" and kvcache.kvDtype="f16", but resolved ` +
          `activationDtype="${activationDtype}" and kvcache.kvDtype="${kvDtype}".`
        );
      }
      continue;
    }
    if (kernel.includes('_f16')) {
      if (activationDtype !== 'f16' || kvDtype !== 'f16') {
        throw new Error(
          `[ExecutionV0] Inline kernelPath attention kernel "${kernel}" requires ` +
          `activationDtype="f16" and kvcache.kvDtype="f16", but resolved ` +
          `activationDtype="${activationDtype}" and kvcache.kvDtype="${kvDtype}".`
        );
      }
      continue;
    }
    if (activationDtype !== 'f32' || kvDtype !== 'f32') {
      throw new Error(
        `[ExecutionV0] Inline kernelPath attention kernel "${kernel}" requires ` +
        `activationDtype="f32" and kvcache.kvDtype="f32", but resolved ` +
        `activationDtype="${activationDtype}" and kvcache.kvDtype="${kvDtype}".`
      );
    }
  }
}

export function resolveFinitenessFallbackKernelPathId(defaultKernelPathId) {
  return defaultKernelPathId
    ? selectRuleValue(
      'inference',
      'kernelPath',
      'finitenessFallback',
      { kernelPathId: defaultKernelPathId }
    )
    : null;
}

export function buildInlineKernelPath(
  steps,
  sessionDefaults,
  modelId,
  numLayers,
  finitenessFallbackKernelPathId = null
) {
  const activationDtype = requireSessionActivationDtype(sessionDefaults);
  const kvDtype = normalizeDtype(
    sessionDefaults?.kvcache?.kvDtype ?? activationDtype,
    'sessionDefaults.kvcache.kvDtype'
  );
  const decodeSteps = buildLayerPhaseSteps(steps, 'decode', 0);
  const prefillSteps = buildLayerPhaseSteps(steps, 'prefill', 0);
  if (decodeSteps.length === 0 && prefillSteps.length === 0) {
    return null;
  }

  const path = {
    id: `${modelId || 'model'}-execution-v0`,
    name: 'Execution v0 inline kernel path',
    description: 'Generated from manifest.inference.execution.steps',
    activationDtype,
    kvDtype,
    ...(typeof finitenessFallbackKernelPathId === 'string' && finitenessFallbackKernelPathId.length > 0
      ? { finitenessFallbackKernelPathId }
      : {}),
    decode: {
      steps: decodeSteps.length > 0 ? decodeSteps : prefillSteps,
    },
    prefill: {
      steps: prefillSteps.length > 0 ? prefillSteps : decodeSteps,
    },
  };

  if (numLayers > 0) {
    const overrides = [];
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      const decodeLayerSteps = buildLayerPhaseSteps(steps, 'decode', layerIdx);
      const prefillLayerSteps = buildLayerPhaseSteps(steps, 'prefill', layerIdx);
      const hasCustomDecode = JSON.stringify(decodeLayerSteps) !== JSON.stringify(path.decode.steps);
      const hasCustomPrefill = JSON.stringify(prefillLayerSteps) !== JSON.stringify(path.prefill.steps);
      if (!hasCustomDecode && !hasCustomPrefill) continue;
      const mergedLayerSteps = decodeLayerSteps.length > 0
        ? decodeLayerSteps
        : prefillLayerSteps;
      if (mergedLayerSteps.length > 0) {
        overrides.push({
          layers: [layerIdx],
          steps: mergedLayerSteps,
        });
      }
    }
    if (overrides.length > 0) {
      path.layerOverrides = overrides;
    }
  }

  const preLayer = getSectionSteps(steps, 'preLayer');
  if (preLayer.length > 0) {
    path.preLayer = preLayer;
  }
  const postLayer = getSectionSteps(steps, 'postLayer');
  if (postLayer.length > 0) {
    path.postLayer = postLayer;
  }
  const sampling = getSectionSteps(steps, 'sampling', 'decode');
  if (sampling.length > 0) {
    path.sampling = sampling;
  }

  assertInlineKernelPathSessionCompatibility(path, sessionDefaults);
  return path;
}

export function buildLayerPipelineFromExecution(steps) {
  const layerSectionSteps = steps.filter((step) => step.section === 'layer');
  if (layerSectionSteps.length === 0) {
    return null;
  }
  if (layerSectionSteps.some((step) => !PIPELINE_COMPATIBLE_OPS.has(step.op))) {
    return null;
  }

  const layerSteps = layerSectionSteps
    .map((step) => ({
      op: step.op,
      phase: step.phase,
      src: step.src ?? 'state',
      dst: step.dst ?? 'state',
      ...(step.residual !== undefined ? { residual: step.residual } : {}),
      ...(step.a !== undefined ? { a: step.a } : {}),
      ...(step.b !== undefined ? { b: step.b } : {}),
      ...(step.variant !== undefined ? { variant: step.variant } : {}),
      ...(step.skipInputNorm !== undefined ? { skipInputNorm: step.skipInputNorm } : {}),
      ...(step.precision?.inputDtype ? { inputDtype: step.precision.inputDtype } : {}),
      ...(step.precision?.outputDtype ? { outputDtype: step.precision.outputDtype } : {}),
      ...(step.fromDtype ? { fromDtype: step.fromDtype } : {}),
      ...(step.toDtype ? { toDtype: step.toDtype } : {}),
      ...(step.probeStage ? { probeStage: step.probeStage } : {}),
      ...(step.name ? { name: step.name } : {}),
      ...(step.weight ? { weight: step.weight } : {}),
    }));

  return {
    steps: layerSteps,
    overrides: [],
  };
}

export function buildSessionRuntimePatch(sessionDefaults) {
  const patch = {};
  const computeDefaults = sessionDefaults?.compute?.defaults ?? null;
  const computePatch = {};
  const activationDtype = computeDefaults?.activationDtype;
  if (activationDtype) {
    computePatch.activationDtype = activationDtype;
  }
  if (computeDefaults && (computeDefaults.mathDtype || computeDefaults.accumDtype || computeDefaults.outputDtype)) {
    computePatch.defaults = {
      ...(computeDefaults.mathDtype ? { mathDtype: computeDefaults.mathDtype } : {}),
      ...(computeDefaults.accumDtype ? { accumDtype: computeDefaults.accumDtype } : {}),
      ...(computeDefaults.outputDtype ? { outputDtype: computeDefaults.outputDtype } : {}),
    };
  }
  if (Object.keys(computePatch).length > 0) {
    patch.compute = computePatch;
  }
  if (sessionDefaults?.kvcache) {
    patch.kvcache = sessionDefaults.kvcache;
  }
  if (sessionDefaults?.decodeLoop) {
    patch.batching = {
      batchSize: sessionDefaults.decodeLoop.batchSize,
      stopCheckMode: sessionDefaults.decodeLoop.stopCheckMode,
      readbackInterval: sessionDefaults.decodeLoop.readbackInterval,
      ringTokens: sessionDefaults.decodeLoop.ringTokens,
      ringStop: sessionDefaults.decodeLoop.ringStop,
      ringStaging: sessionDefaults.decodeLoop.ringStaging,
    };
    if (sessionDefaults.decodeLoop.disableCommandBatching !== undefined) {
      patch.generation = {
        disableCommandBatching: sessionDefaults.decodeLoop.disableCommandBatching === true,
      };
    }
  }
  return patch;
}

export function buildModelRuntimeOverrides(manifestInference) {
  const model = manifestInference?.model;
  if (!model || typeof model !== 'object') {
    return null;
  }
  return cloneJson(model);
}
