import {
  inferConversionConfigModelId,
  resolveMaterializedManifestFromConversionConfig,
} from '../conversion-config-materializer.js';
import { mergeRuntimeValues } from '../../config/runtime-merge.js';
import { extractExecutionContractFacts } from '../../config/execution-contract-check.js';
import { validateKernelPath } from '../../config/kernel-path-loader.js';
import { DEFAULT_RUNTIME_CONFIG, expandExecutionV1 } from '../../config/schema/index.js';
import { compileExecutionPlanState } from '../../inference/pipelines/text/execution-plan.js';
import { compileExecutionV1, hasExecutionV1 } from '../../inference/pipelines/text/execution-v1.js';
import {
  assertKernelPathSessionCompatibility,
  buildInlineKernelPath,
  buildSessionRuntimePatch,
  isPhaseMatch,
  normalizeDtype,
  requireSessionActivationDtype,
  stepHasLayer,
} from '../../inference/pipelines/text/execution-runtime-builders.js';
import { buildCustomRuntimeFacts } from './custom-runtime-facts.js';
import {
  aggregateTopDecodeTimers,
  buildKernelPathBuilderRuntimeOverlay,
} from './runtime-overlay.js';
import { isPlainObject } from '../../formats/plain-object.js';

export function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

export function normalizeInteger(value) {
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric >= 0 ? numeric : null;
}

export function cloneJsonLike(value) {
  if (typeof globalThis.structuredClone === 'function') {
    return globalThis.structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

export function stableValue(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableValue(entry));
  }
  if (!isPlainObject(value)) {
    return value;
  }
  const sorted = {};
  for (const key of Object.keys(value).sort((left, right) => left.localeCompare(right))) {
    sorted[key] = stableValue(value[key]);
  }
  return sorted;
}

export function stableStringify(value) {
  return JSON.stringify(stableValue(value));
}

export function normalizeLayers(layers) {
  if (layers === 'all') {
    return 'all';
  }
  if (!Array.isArray(layers)) {
    return null;
  }
  const normalized = layers
    .map((value) => normalizeInteger(value))
    .filter((value) => value != null);
  normalized.sort((left, right) => left - right);
  return normalized;
}

export function normalizeStep(step) {
  const normalized = {
    op: normalizeText(step?.op),
    kernel: normalizeText(step?.kernel),
    entry: normalizeText(step?.entry || 'main') || 'main',
  };
  const weights = normalizeText(step?.weights);
  if (weights) {
    normalized.weights = weights;
  }
  if (step?.constants != null) {
    normalized.constants = stableValue(step.constants);
  }
  return normalized;
}

export function isSamplingStep(step) {
  const op = normalizeText(step?.op).toLowerCase();
  const kernel = normalizeText(step?.kernel).toLowerCase();
  return op === 'sample' || kernel.startsWith('sample');
}

export function splitKernelPathSampling(steps) {
  const postLayer = [];
  const sampling = [];
  for (const step of Array.isArray(steps) ? steps : []) {
    if (isSamplingStep(step)) {
      sampling.push(step);
      continue;
    }
    postLayer.push(step);
  }
  return { postLayer, sampling };
}

export function normalizeKernelPathShape(path) {
  if (!isPlainObject(path)) {
    return null;
  }
  const normalizedPostLayer = Array.isArray(path.postLayer) ? path.postLayer.map(normalizeStep) : [];
  const normalizedSampling = Array.isArray(path.sampling) ? path.sampling.map(normalizeStep) : [];
  const canonicalizedSampling = splitKernelPathSampling([
    ...normalizedPostLayer,
    ...normalizedSampling,
  ]);
  return stableValue({
    activationDtype: normalizeText(path.activationDtype) || null,
    kvDtype: normalizeText(path.kvDtype || path.activationDtype) || null,
    outputDtype: normalizeText(path.outputDtype) || null,
    preLayer: Array.isArray(path.preLayer) ? path.preLayer.map(normalizeStep) : [],
    decode: Array.isArray(path?.decode?.steps) ? path.decode.steps.map(normalizeStep) : [],
    prefill: Array.isArray(path?.prefill?.steps) ? path.prefill.steps.map(normalizeStep) : [],
    postLayer: canonicalizedSampling.postLayer,
    sampling: canonicalizedSampling.sampling,
    layerOverrides: Array.isArray(path.layerOverrides)
      ? path.layerOverrides.map((override) => ({
        layers: normalizeLayers(override?.layers) || [],
        ...(Array.isArray(override?.steps) && override.steps.length > 0
          ? { steps: override.steps.map(normalizeStep) }
          : {}),
        ...(Array.isArray(override?.decode?.steps) && override.decode.steps.length > 0
          ? { decode: { steps: override.decode.steps.map(normalizeStep) } }
          : {}),
        ...(Array.isArray(override?.prefill?.steps) && override.prefill.steps.length > 0
          ? { prefill: { steps: override.prefill.steps.map(normalizeStep) } }
          : {}),
      }))
      : [],
  });
}

export function toComparableKernelPathStep(step) {
  if (!step?.kernel) {
    return null;
  }
  return normalizeStep(step);
}

export function buildComparableBoundarySteps(resolvedSteps, section) {
  return resolvedSteps
    .filter((step) => step?.section === section)
    .map(toComparableKernelPathStep)
    .filter((step) => step != null);
}

export function buildComparableLayerPhaseSteps(resolvedSteps, phase, layerIdx) {
  return resolvedSteps
    .filter((step) => step?.section === 'layer' && isPhaseMatch(step.phase, phase))
    .filter((step) => stepHasLayer(step, layerIdx))
    .map(toComparableKernelPathStep)
    .filter((step) => step != null);
}

export function buildComparableKernelPathFromResolvedSteps(
  resolvedSteps,
  session,
  modelId,
  numLayers
) {
  const activationDtype = requireSessionActivationDtype(session);
  const kvDtype = normalizeDtype(
    session?.kvcache?.kvDtype ?? activationDtype,
    'session.kvcache.kvDtype'
  );
  const decodeSteps = buildComparableLayerPhaseSteps(resolvedSteps, 'decode', 0);
  const prefillSteps = buildComparableLayerPhaseSteps(resolvedSteps, 'prefill', 0);
  if (decodeSteps.length === 0 && prefillSteps.length === 0) {
    return null;
  }

  const path = {
    id: `${modelId || 'model'}-execution-inline`,
    name: 'Execution inline kernel path',
    description: 'Generated from manifest.inference.execution for structural comparison',
    activationDtype,
    kvDtype,
    decode: {
      steps: decodeSteps.length > 0 ? decodeSteps : prefillSteps,
    },
    prefill: {
      steps: prefillSteps.length > 0 ? prefillSteps : decodeSteps,
    },
  };

  if (numLayers > 0) {
    const overrides = [];
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
      const decodeLayerSteps = buildComparableLayerPhaseSteps(resolvedSteps, 'decode', layerIdx);
      const prefillLayerSteps = buildComparableLayerPhaseSteps(resolvedSteps, 'prefill', layerIdx);
      const hasCustomDecode = stableStringify(decodeLayerSteps) !== stableStringify(path.decode.steps);
      const hasCustomPrefill = stableStringify(prefillLayerSteps) !== stableStringify(path.prefill.steps);
      if (!hasCustomDecode && !hasCustomPrefill) {
        continue;
      }
      const mergedLayerSteps = decodeLayerSteps.length > 0 ? decodeLayerSteps : prefillLayerSteps;
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

  const preLayer = buildComparableBoundarySteps(resolvedSteps, 'preLayer');
  if (preLayer.length > 0) {
    path.preLayer = preLayer;
  }
  const postLayer = buildComparableBoundarySteps(resolvedSteps, 'postLayer');
  if (postLayer.length > 0) {
    path.postLayer = postLayer;
  }
  return path;
}

export function compareField(candidate, existing) {
  return stableStringify(candidate) === stableStringify(existing);
}

export function createMismatchDetail(options) {
  return {
    code: options.code,
    category: options.category,
    label: options.label,
    repairHint: options.repairHint,
    phase: options.phase ?? null,
  };
}

export function kernelCapabilityFingerprint(steps) {
  const normalizedSteps = Array.isArray(steps) ? steps : [];
  return stableValue({
    usesSubgroups: normalizedSteps.some((step) => {
      const kernel = normalizeText(step?.kernel).toLowerCase();
      const entry = normalizeText(step?.entry).toLowerCase();
      return kernel.includes('subgroup') || entry.includes('vec4') || entry.includes('multicol');
    }),
    attentionKinds: normalizedSteps
      .filter((step) => normalizeText(step?.op).toLowerCase() === 'attention')
      .map((step) => normalizeText(step?.kernel).toLowerCase())
      .sort((left, right) => left.localeCompare(right)),
  });
}

export function describePhaseMismatch(phase, candidateSteps, existingSteps) {
  if (phase === 'sampling' && candidateSteps.length > 0 && existingSteps.length === 0) {
    return createMismatchDetail({
      code: 'missing_sampling',
      category: 'sampling',
      label: 'Sampling block is missing from the registry path.',
      repairHint: 'Copy the sampling block from the execution graph into the kernel-path proposal.',
      phase,
    });
  }
  if (phase === 'layerOverrides') {
    return createMismatchDetail({
      code: 'layer_override_drift',
      category: 'layer-overrides',
      label: 'Layer override coverage differs.',
      repairHint: 'Emit explicit layerOverrides for the layers whose decode/prefill steps diverge from the default path.',
      phase,
    });
  }
  if (!compareField(kernelCapabilityFingerprint(candidateSteps), kernelCapabilityFingerprint(existingSteps))) {
    return createMismatchDetail({
      code: 'capability_drift',
      category: 'capability',
      label: `${phase} kernels assume different device capabilities.`,
      repairHint: 'Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.',
      phase,
    });
  }
  return createMismatchDetail({
    code: `${phase}_drift`,
    category: 'shape',
    label: `${phase} steps differ.`,
    repairHint: `Update the ${phase} steps to match the resolved execution graph step-for-step.`,
    phase,
  });
}

export function diffKernelPathShape(candidate, existing, modelRecord) {
  if (!candidate || !existing) {
    return [
      createMismatchDetail({
        code: 'candidate_unavailable',
        category: 'shape',
        label: 'Candidate kernel path is unavailable.',
        repairHint: 'Fix the inline kernel-path synthesis error before comparing against registry paths.',
      }),
    ];
  }
  const details = [];
  if (!compareField(candidate.activationDtype, existing.activationDtype)) {
    details.push(createMismatchDetail({
      code: 'activation_dtype_drift',
      category: 'dtype',
      label: 'Activation dtype differs.',
      repairHint: `Set activationDtype to "${candidate.activationDtype}" or choose kernels compatible with "${existing.activationDtype}".`,
    }));
  }
  if (!compareField(candidate.kvDtype, existing.kvDtype)) {
    details.push(createMismatchDetail({
      code: 'kv_dtype_drift',
      category: 'dtype',
      label: 'KV dtype differs.',
      repairHint: `Set kvDtype to "${candidate.kvDtype}" or swap attention kernels to ones compatible with "${existing.kvDtype}".`,
    }));
  }
  if (!compareField(candidate.outputDtype, existing.outputDtype)) {
    details.push(createMismatchDetail({
      code: 'output_dtype_drift',
      category: 'dtype',
      label: 'Output dtype differs.',
      repairHint: 'Align outputDtype with the execution graph or remove it when the path inherits activation dtype.',
    }));
  }
  for (const phase of ['preLayer', 'decode', 'prefill', 'postLayer', 'sampling']) {
    if (!compareField(candidate[phase], existing[phase])) {
      details.push(describePhaseMismatch(phase, candidate[phase], existing[phase]));
    }
  }
  if (!compareField(candidate.layerOverrides, existing.layerOverrides)) {
    details.push(describePhaseMismatch('layerOverrides', candidate.layerOverrides, existing.layerOverrides));
  }
  if (
    Array.isArray(modelRecord?.customRuntimeFacts)
    && modelRecord.customRuntimeFacts.some((fact) => fact?.assumptions?.registryBypass === true)
  ) {
    details.push(createMismatchDetail({
      code: 'custom_runtime_bypass',
      category: 'custom-runtime',
      label: 'Custom runtime layers bypass raw kernel-path lowering.',
      repairHint: 'Keep the proposal partial and preserve the custom runtime facts for the bypassed layers instead of forcing a registry path to claim ownership of them.',
    }));
  }
  return details;
}

export function buildRuntimeConfigFromSession(session) {
  const runtimeConfig = cloneJsonLike(DEFAULT_RUNTIME_CONFIG);
  const runtimeInferencePatch = buildSessionRuntimePatch(session);
  return {
    ...runtimeConfig,
    inference: mergeRuntimeValues(runtimeConfig.inference ?? {}, runtimeInferencePatch),
  };
}

export function summarizeCompiledPlan(planState) {
  return {
    primaryPlanId: planState?.primaryPlan?.id ?? null,
    primaryKernelPathId: planState?.primaryPlan?.kernelPathId ?? null,
    primaryActivationDtype: planState?.primaryPlan?.activationDtype ?? null,
    fallbackPlanId: planState?.fallbackPlan?.id ?? null,
    fallbackKernelPathId: planState?.fallbackPlan?.kernelPathId ?? null,
  };
}

export function verifyKernelPathProposal(path, candidateShape, session) {
  if (!path) {
    return null;
  }
  const checks = [];
  const errors = [];

  const validationErrors = validateKernelPath(path);
  checks.push({
    id: 'kernelPathContract',
    ok: validationErrors.length === 0,
  });
  if (validationErrors.length > 0) {
    errors.push(...validationErrors.map((entry) => `[KernelPath] ${entry}`));
  }

  try {
    assertKernelPathSessionCompatibility(path, session);
    checks.push({
      id: 'sessionCompatibility',
      ok: true,
    });
  } catch (error) {
    checks.push({
      id: 'sessionCompatibility',
      ok: false,
    });
    errors.push(error instanceof Error ? error.message : String(error));
  }

  const roundTripShape = normalizeKernelPathShape(path);
  const roundTripShapeMatches = compareField(roundTripShape, candidateShape);
  checks.push({
    id: 'roundTripShape',
    ok: roundTripShapeMatches,
  });
  if (!roundTripShapeMatches) {
    errors.push('[KernelPath] proposal does not round-trip to the same normalized execution shape.');
  }

  try {
    const runtimeConfig = buildRuntimeConfigFromSession(session);
    const planState = compileExecutionPlanState({
      runtimeConfig,
      resolvedKernelPath: path,
      kernelPathSource: 'self',
    });
    checks.push({
      id: 'executionPlanCompile',
      ok: true,
    });
    return {
      ok: errors.length === 0,
      checks,
      errors,
      roundTripShapeMatches,
      compiledPlan: summarizeCompiledPlan(planState),
    };
  } catch (error) {
    checks.push({
      id: 'executionPlanCompile',
      ok: false,
    });
    errors.push(error instanceof Error ? error.message : String(error));
    return {
      ok: false,
      checks,
      errors,
      roundTripShapeMatches,
      compiledPlan: null,
    };
  }
}
