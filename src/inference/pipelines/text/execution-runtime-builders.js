import { selectRuleValue } from '../../../rules/rule-registry.js';
import { log } from '../../../debug/index.js';

// =============================================================================
// Shared execution helpers used by the v1 execution runtime.
// =============================================================================

export const PIPELINE_COMPATIBLE_OPS = new Set([
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

export function normalizeDtype(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(`[Execution] ${label} must be "f16" or "f32"; got "${value}"`);
  }
  return normalized;
}

export function isPhaseMatch(phase, targetPhase) {
  return phase === 'both' || phase === targetPhase;
}

export function stepHasLayer(step, layerIdx) {
  if (step.layers === 'all') return true;
  if (!Array.isArray(step.layers)) return false;
  return step.layers.includes(layerIdx);
}

export function requireSessionActivationDtype(
  sessionDefaults,
  label = 'sessionDefaults.compute.defaults.activationDtype'
) {
  const activationDtype = sessionDefaults?.compute?.defaults?.activationDtype;
  if (activationDtype == null) {
    throw new Error(`[Execution] ${label} is required.`);
  }
  return normalizeDtype(activationDtype, label);
}

function toKernelPathStep(step) {
  if (step.op === 'cast') return null;
  if (!step.kernel) {
    log.warn(
      'ExecutionRuntime',
      `toKernelPathStep: dropping step with op="${step.op}" — no kernel assigned. ` +
      `Section: ${step.section ?? 'unknown'}, phase: ${step.phase ?? 'unknown'}.`
    );
    return null;
  }
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

export function assertKernelPathSessionCompatibility(path, sessionDefaults) {
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
          `[Execution] Inline kernelPath attention kernel "${kernel}" requires ` +
          `activationDtype="f32" and kvcache.kvDtype="f16", but resolved ` +
          `activationDtype="${activationDtype}" and kvcache.kvDtype="${kvDtype}".`
        );
      }
      continue;
    }
    if (kernel.includes('_f16')) {
      if (activationDtype !== 'f16' || kvDtype !== 'f16') {
        throw new Error(
          `[Execution] Inline kernelPath attention kernel "${kernel}" requires ` +
          `activationDtype="f16" and kvcache.kvDtype="f16", but resolved ` +
          `activationDtype="${activationDtype}" and kvcache.kvDtype="${kvDtype}".`
        );
      }
      continue;
    }
    if (activationDtype !== 'f32' || kvDtype !== 'f32') {
      throw new Error(
        `[Execution] Inline kernelPath attention kernel "${kernel}" requires ` +
        `activationDtype="f32" and kvcache.kvDtype="f32", but resolved ` +
        `activationDtype="${activationDtype}" and kvcache.kvDtype="${kvDtype}".`
      );
    }
  }
}

export function resolveFinitenessFallbackKernelPathId(kernelPathId) {
  return kernelPathId
    ? selectRuleValue(
      'inference',
      'kernelPath',
      'finitenessFallback',
      { kernelPathId }
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
    id: `${modelId || 'model'}-execution-inline`,
    name: 'Execution inline kernel path',
    description: 'Generated from manifest.inference.execution',
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

  assertKernelPathSessionCompatibility(path, sessionDefaults);
  return path;
}

/**
 * Build a layer pipeline from execution-v1 resolved steps.
 *
 * @param {readonly Record<string, unknown>[]} steps - Resolved execution steps.
 * @param {{ strict?: boolean }} [options] - When strict is true, throws on
 *   incompatible ops instead of returning a degraded result.
 * @returns {{ steps: Record<string, unknown>[]; overrides: unknown[]; hasIncompatibleOps: boolean }
 *   | { incompatibleOps: string[]; hasIncompatibleOps: true } | null}
 */
export function buildLayerPipelineFromExecution(steps, options = {}) {
  const { strict = false } = options;
  const layerSectionSteps = steps.filter((step) => step.section === 'layer');
  if (layerSectionSteps.length === 0) {
    return null;
  }
  const incompatibleOps = [
    ...new Set(
      layerSectionSteps
        .filter((step) => !PIPELINE_COMPATIBLE_OPS.has(step.op))
        .map((step) => step.op)
    ),
  ];
  if (incompatibleOps.length > 0) {
    const message =
      `[Execution] Layer pipeline contains ops not in PIPELINE_COMPATIBLE_OPS: ` +
      `${incompatibleOps.join(', ')}. Pipeline will be degraded.`;
    if (strict) {
      throw new Error(message);
    }
    log.error('ExecutionRuntime', message);
    return { incompatibleOps, hasIncompatibleOps: true };
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
    hasIncompatibleOps: false,
  };
}

/**
 * Build a runtime config patch from manifest sessionDefaults.
 *
 * Field consumption status after merge into runtimeConfig.inference:
 *
 * CONSUMED (read by layers/logits/generator via runtimeConfig.inference.compute):
 *   - patch.compute.activationDtype  -- read by execution plan compilation,
 *     logits fallback (getRuntimeConfig().inference.compute.activationDtype),
 *     and layer context builder.
 *
 * CONSUMED (read by KV cache and batching subsystems):
 *   - patch.kvcache.*
 *   - patch.batching.*
 *   - patch.generation.disableCommandBatching
 *
 * DEAD / NOT CONSUMED at runtime (merged into runtimeConfig but never read back):
 *   - patch.session.compute.defaults.mathDtype
 *   - patch.session.compute.defaults.accumDtype
 *   - patch.session.compute.defaults.outputDtype
 *
 * The dead fields are retained for manifest round-trip fidelity and potential
 * future consumption. They should NOT be removed (non-breaking), but new code
 * should not rely on reading them from runtimeConfig.inference.session.
 */
export function buildSessionRuntimePatch(sessionDefaults) {
  const patch = {};
  const computeDefaults = sessionDefaults?.compute?.defaults ?? null;
  const computePatch = {};
  const sessionComputeDefaultsPatch = {};
  const activationDtype = computeDefaults?.activationDtype;
  if (activationDtype) {
    // CONSUMED: merged into patch.compute and read by execution plan + logits
    computePatch.activationDtype = activationDtype;
  }
  if (computeDefaults?.mathDtype) {
    // DEPRECATED / DEAD: merged into patch.session.compute.defaults but never
    // read back by any runtime subsystem. Retained for manifest round-trip.
    sessionComputeDefaultsPatch.mathDtype = computeDefaults.mathDtype;
  }
  if (computeDefaults?.accumDtype) {
    // DEPRECATED / DEAD: see mathDtype note above.
    sessionComputeDefaultsPatch.accumDtype = computeDefaults.accumDtype;
  }
  if (computeDefaults?.outputDtype) {
    // DEPRECATED / DEAD: see mathDtype note above.
    sessionComputeDefaultsPatch.outputDtype = computeDefaults.outputDtype;
  }
  if (Object.keys(computePatch).length > 0) {
    patch.compute = computePatch;
  }
  if (Object.keys(sessionComputeDefaultsPatch).length > 0) {
    // Log a deprecation notice listing the dead fields that are merged but never consumed.
    const deadFields = Object.keys(sessionComputeDefaultsPatch);
    log.debug(
      'ExecutionRuntime',
      `Session compute defaults contain fields that are merged but not consumed at runtime ` +
      `(deprecated): ${deadFields.join(', ')}. ` +
      'These are retained for manifest round-trip fidelity only.'
    );
    patch.session = {
      compute: {
        defaults: sessionComputeDefaultsPatch,
      },
    };
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
