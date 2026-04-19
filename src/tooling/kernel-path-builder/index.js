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
import { isPlainObject } from '../../utils/plain-object.js';

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeInteger(value) {
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric >= 0 ? numeric : null;
}

function normalizeObject(value) {
  return isPlainObject(value) ? value : {};
}

function cloneJsonLike(value) {
  if (typeof globalThis.structuredClone === 'function') {
    return globalThis.structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function stableValue(value) {
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

function stableStringify(value) {
  return JSON.stringify(stableValue(value));
}

function fnv1aHex(value) {
  const input = String(value ?? '');
  let hash = 0x811c9dc5;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

function normalizeLayers(layers) {
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

function normalizeStep(step) {
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

function cloneSteps(steps) {
  return (Array.isArray(steps) ? steps : []).map((step) => stableValue(step));
}

function normalizeResolvedStep(step, index) {
  const normalized = {
    id: normalizeText(step?.id) || `step_${index}`,
    section: normalizeText(step?.section),
    phase: normalizeText(step?.phase),
    op: normalizeText(step?.op),
    kernel: normalizeText(step?.kernel),
    entry: normalizeText(step?.entry || 'main') || 'main',
    digest: normalizeText(step?.kernelRef?.digest ?? step?.digest),
    layers: normalizeLayers(step?.layers),
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

function isSamplingStep(step) {
  const op = normalizeText(step?.op).toLowerCase();
  const kernel = normalizeText(step?.kernel).toLowerCase();
  return op === 'sample' || kernel.startsWith('sample');
}

function splitKernelPathSampling(steps) {
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

function resolveExecutionSteps(execution) {
  const expanded = expandExecutionV1(execution);
  return expanded.map((step, index) => ({
    id: `${step.section}_${step.phase}_${index}_${step.op}`,
    phase: step.phase,
    section: step.section,
    op: step.op,
    kernel: step.kernel,
    entry: step.entry,
    weights: step.weights ?? null,
    constants: step.constants ?? null,
    layers: step.layers,
    kernelRef: {
      id: `${step.kernel.replace('.wgsl', '')}.${step.entry}`,
      version: '1.0.0',
      digest: step.digest,
    },
  }));
}

function normalizeKernelPathShape(path) {
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

function materializeKernelPathFromShape(modelId, shape, meta = {}) {
  if (!shape) {
    return null;
  }
  return {
    id: normalizeText(meta.id) || `${modelId || 'model'}-execution-inline`,
    name: normalizeText(meta.name) || `Derived kernel path for ${modelId}`,
    description: normalizeText(meta.description) || `Generated from ${modelId} execution-v1 graph.`,
    activationDtype: shape.activationDtype,
    ...(shape.outputDtype ? { outputDtype: shape.outputDtype } : {}),
    ...(shape.kvDtype ? { kvDtype: shape.kvDtype } : {}),
    decode: {
      steps: cloneSteps(shape.decode),
    },
    prefill: {
      steps: cloneSteps(shape.prefill),
    },
    ...(Array.isArray(shape.preLayer) && shape.preLayer.length > 0 ? { preLayer: cloneSteps(shape.preLayer) } : {}),
    ...(Array.isArray(shape.postLayer) && shape.postLayer.length > 0 ? { postLayer: cloneSteps(shape.postLayer) } : {}),
    ...(Array.isArray(shape.sampling) && shape.sampling.length > 0 ? { sampling: cloneSteps(shape.sampling) } : {}),
    ...(Array.isArray(shape.layerOverrides) && shape.layerOverrides.length > 0
      ? {
        layerOverrides: shape.layerOverrides.map((override) => ({
          layers: normalizeLayers(override?.layers) || [],
          ...(Array.isArray(override?.steps) && override.steps.length > 0
            ? { steps: cloneSteps(override.steps) }
            : {}),
          ...(Array.isArray(override?.decode?.steps) && override.decode.steps.length > 0
            ? { decode: { steps: cloneSteps(override.decode.steps) } }
            : {}),
          ...(Array.isArray(override?.prefill?.steps) && override.prefill.steps.length > 0
            ? { prefill: { steps: cloneSteps(override.prefill.steps) } }
            : {}),
        })),
      }
      : {}),
  };
}

function toComparableKernelPathStep(step) {
  if (!step?.kernel) {
    return null;
  }
  return normalizeStep(step);
}

function buildComparableBoundarySteps(resolvedSteps, section) {
  return resolvedSteps
    .filter((step) => step?.section === section)
    .map(toComparableKernelPathStep)
    .filter((step) => step != null);
}

function buildComparableLayerPhaseSteps(resolvedSteps, phase, layerIdx) {
  return resolvedSteps
    .filter((step) => step?.section === 'layer' && isPhaseMatch(step.phase, phase))
    .filter((step) => stepHasLayer(step, layerIdx))
    .map(toComparableKernelPathStep)
    .filter((step) => step != null);
}

function buildComparableKernelPathFromResolvedSteps(
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

function compareField(candidate, existing) {
  return stableStringify(candidate) === stableStringify(existing);
}

function createMismatchDetail(options) {
  return {
    code: options.code,
    category: options.category,
    label: options.label,
    repairHint: options.repairHint,
    phase: options.phase ?? null,
  };
}

function kernelCapabilityFingerprint(steps) {
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

function describePhaseMismatch(phase, candidateSteps, existingSteps) {
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

function diffKernelPathShape(candidate, existing, modelRecord) {
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

function countLayerTypes(layerTypes) {
  const counts = {};
  if (!Array.isArray(layerTypes)) {
    return counts;
  }
  for (const layerType of layerTypes) {
    const key = normalizeText(layerType).toLowerCase();
    if (!key) continue;
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function inferLayerCount(manifestInference, resolvedSteps) {
  const layerTypes = manifestInference?.layerPattern?.layerTypes;
  if (Array.isArray(layerTypes) && layerTypes.length > 0) {
    return layerTypes.length;
  }
  let maxLayer = -1;
  for (const step of resolvedSteps) {
    if (!Array.isArray(step?.layers)) continue;
    for (const layer of step.layers) {
      const normalized = normalizeInteger(layer);
      if (normalized != null && normalized > maxLayer) {
        maxLayer = normalized;
      }
    }
  }
  return maxLayer >= 0 ? maxLayer + 1 : null;
}

function splitResolvedStepsBySection(resolvedSteps) {
  const grouped = {
    preLayer: [],
    decode: [],
    prefill: [],
    postLayer: [],
  };
  for (let index = 0; index < resolvedSteps.length; index += 1) {
    const step = normalizeResolvedStep(resolvedSteps[index], index);
    if (step.section === 'preLayer') {
      grouped.preLayer.push(step);
      continue;
    }
    if (step.section === 'postLayer') {
      grouped.postLayer.push(step);
      continue;
    }
    if (step.section === 'layer' && (step.phase === 'decode' || step.phase === 'both')) {
      grouped.decode.push(step);
    }
    if (step.section === 'layer' && (step.phase === 'prefill' || step.phase === 'both')) {
      grouped.prefill.push(step);
    }
  }
  return grouped;
}

function buildExecutionGraphSignaturePayload(modelRecord) {
  return stableValue({
    session: {
      activationDtype: modelRecord.session.compute.activationDtype,
      mathDtype: modelRecord.session.compute.mathDtype,
      accumDtype: modelRecord.session.compute.accumDtype,
      outputDtype: modelRecord.session.compute.outputDtype,
      kvDtype: modelRecord.session.kvDtype,
      kvLayout: modelRecord.session.kvLayout,
      decodeLoop: modelRecord.session.decodeLoop,
    },
    runtime: {
      inlineKernelPathEnabled: modelRecord.runtime.inlineKernelPathEnabled,
    },
    execution: modelRecord.execution.sections,
  });
}

function buildKernelRefKey(step) {
  return `${normalizeText(step.kernel)}#${normalizeText(step.entry || 'main')}#${normalizeText(step.digest)}`;
}

function buildKernelRefList(resolvedSteps) {
  const seen = new Set();
  const refs = [];
  for (const step of resolvedSteps) {
    const key = buildKernelRefKey(step);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    refs.push({
      key,
      kernel: normalizeText(step.kernel),
      entry: normalizeText(step.entry || 'main') || 'main',
      digest: normalizeText(step.digest),
    });
  }
  return refs;
}

function normalizeQuantizationTag(rawConfig) {
  const weights = normalizeText(rawConfig?.quantization?.weights).toLowerCase();
  if (!weights) return 'weights';
  if (weights === 'q4k') return 'q4k-dequant';
  return weights;
}

function resolveFamily(configPath, manifest) {
  const modelType = normalizeText(manifest?.modelType).toLowerCase();
  const segments = String(configPath ?? '').split('/');
  if (segments.length >= 2) {
    return normalizeText(segments[segments.length - 2]).toLowerCase() || modelType || 'model';
  }
  return modelType || 'model';
}

function getRegistryAffinity(entry, modelRecord) {
  const id = normalizeText(entry?.id).toLowerCase();
  const family = normalizeText(modelRecord?.family).toLowerCase();
  const quantTag = normalizeQuantizationTag(modelRecord?.rawConfig);
  return {
    familyMatch: family ? id.startsWith(family) : false,
    quantizationMatch: quantTag ? id.includes(quantTag) : false,
  };
}

function rankKernelPathMatches(candidateShape, registryEntries, modelRecord) {
  if (!candidateShape) {
    return [];
  }
  const ranked = registryEntries.map((entry) => {
    const mismatchDetails = diffKernelPathShape(candidateShape, entry.pathShape, modelRecord);
    const affinity = getRegistryAffinity(entry, modelRecord);
    return {
      id: entry.id,
      status: entry.status,
      statusReason: entry.statusReason,
      notes: entry.notes,
      mismatchCount: mismatchDetails.length,
      mismatchReasons: mismatchDetails.map((detail) => detail.label),
      mismatchDetails,
      exact: mismatchDetails.length === 0,
      activationDtype: entry.pathShape?.activationDtype ?? null,
      kvDtype: entry.pathShape?.kvDtype ?? null,
      familyMatch: affinity.familyMatch,
      quantizationMatch: affinity.quantizationMatch,
    };
  });
  ranked.sort((left, right) => {
    if (left.mismatchCount !== right.mismatchCount) {
      return left.mismatchCount - right.mismatchCount;
    }
    if (left.familyMatch !== right.familyMatch) {
      return left.familyMatch ? -1 : 1;
    }
    if (left.quantizationMatch !== right.quantizationMatch) {
      return left.quantizationMatch ? -1 : 1;
    }
    return left.id.localeCompare(right.id);
  });
  return ranked;
}

function deriveAttentionTag(candidateShape) {
  const attentionStep = candidateShape?.decode?.find((step) => step.op === 'attention')
    || candidateShape?.prefill?.find((step) => step.op === 'attention')
    || null;
  const kernel = normalizeText(attentionStep?.kernel).toLowerCase();
  if (kernel.includes('online')) return 'online';
  if (kernel.includes('stream')) return 'streaming';
  if (kernel.includes('small')) return 'small';
  if (kernel.includes('decode')) return 'decode';
  return 'path';
}

function buildSuggestedKernelPathId(modelRecord, candidateShape) {
  const family = resolveFamily(modelRecord.configPath, { modelType: modelRecord.modelType });
  const quantTag = normalizeQuantizationTag(modelRecord.rawConfig);
  const activation = normalizeText(candidateShape?.activationDtype || modelRecord.session.compute.activationDtype || 'f32').toLowerCase();
  const attentionTag = deriveAttentionTag(candidateShape);
  return [family, quantTag, `${activation}a`, attentionTag]
    .map((part) => part.replace(/[^a-z0-9-]+/g, '-'))
    .filter(Boolean)
    .join('-');
}

function buildRuntimeConfigFromSession(session) {
  const runtimeConfig = cloneJsonLike(DEFAULT_RUNTIME_CONFIG);
  const runtimeInferencePatch = buildSessionRuntimePatch(session);
  return {
    ...runtimeConfig,
    inference: mergeRuntimeValues(runtimeConfig.inference ?? {}, runtimeInferencePatch),
  };
}

function summarizeCompiledPlan(planState) {
  return {
    primaryPlanId: planState?.primaryPlan?.id ?? null,
    primaryKernelPathId: planState?.primaryPlan?.kernelPathId ?? null,
    primaryActivationDtype: planState?.primaryPlan?.activationDtype ?? null,
    fallbackPlanId: planState?.fallbackPlan?.id ?? null,
    fallbackKernelPathId: planState?.fallbackPlan?.kernelPathId ?? null,
  };
}

function verifyKernelPathProposal(path, candidateShape, session) {
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

function buildProposedKernelPath(modelRecord, candidateShape, exactMatchEntry, session) {
  if (!candidateShape) {
    return null;
  }
  const proposalPath = exactMatchEntry?.path
    ? cloneJsonLike(exactMatchEntry.path)
    : materializeKernelPathFromShape(modelRecord.modelId, candidateShape, {
      id: buildSuggestedKernelPathId(modelRecord, candidateShape),
      name: `Derived kernel path for ${modelRecord.modelId}`,
      description: `Generated from ${modelRecord.modelId} execution-v1 graph.`,
    });
  const verification = verifyKernelPathProposal(proposalPath, candidateShape, session);
  return {
    kind: exactMatchEntry?.path ? 'existing' : 'proposed',
    selectedKernelPathId: exactMatchEntry?.id ?? null,
    path: proposalPath,
    verification,
  };
}

function buildFallbackExecutionContractSessionFacts(manifest) {
  const session = normalizeObject(manifest?.inference?.session);
  const decodeLoop = normalizeObject(session.decodeLoop);
  return {
    source: 'session',
    facts: {
      layout: normalizeText(session?.kvcache?.layout) || null,
      disableCommandBatching: decodeLoop.disableCommandBatching ?? null,
      decodeBatchSize: normalizeInteger(decodeLoop.batchSize),
      headDim: normalizeInteger(manifest?.architecture?.headDim),
      kvLen: normalizeInteger(manifest?.architecture?.maxSeqLen ?? session?.kvcache?.maxSeqLen),
      coldQuantMode: normalizeText(session?.kvcache?.tiering?.mode) || null,
    },
    error: null,
  };
}

function buildExecutionContractSessionFacts(manifest, resolvedSteps) {
  const syntheticManifest = {
    modelId: manifest?.modelId ?? 'model',
    architecture: manifest?.architecture ?? null,
    inference: {
      session: manifest?.inference?.session ?? null,
      execution: {
        steps: resolvedSteps.map((step) => ({
          id: step.id,
          phase: step.phase,
          op: step.op,
        })),
      },
    },
  };
  try {
    const facts = extractExecutionContractFacts(syntheticManifest);
    return {
      source: 'execution-contract',
      facts: facts.session,
      error: null,
    };
  } catch (error) {
    return {
      ...buildFallbackExecutionContractSessionFacts(manifest),
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function buildSourceSignature(materialization) {
  const manifest = materialization.manifest;
  const resolvedSteps = resolveExecutionSteps(manifest.inference.execution);
  const contractFacts = buildExecutionContractSessionFacts(manifest, resolvedSteps);
  return stableValue({
    modelId: materialization.modelId,
    session: contractFacts.facts,
    execution: splitResolvedStepsBySection(resolvedSteps),
  });
}

function buildSourceConsistency(materializations) {
  if (!Array.isArray(materializations) || materializations.length <= 1) {
    return {
      ok: true,
      comparedSources: Array.isArray(materializations) ? materializations.length : 0,
      mismatchedSources: [],
    };
  }
  const preferred = buildSourceSignature(materializations[0]);
  const mismatchedSources = [];
  for (const materialization of materializations.slice(1)) {
    const signature = buildSourceSignature(materialization);
    if (!compareField(signature, preferred)) {
      mismatchedSources.push({
        sourceKind: materialization.sourceKind,
        sourcePath: materialization.sourcePath,
      });
    }
  }
  return {
    ok: mismatchedSources.length === 0,
    comparedSources: materializations.length,
    mismatchedSources,
  };
}

function buildModelSources(materializations) {
  return materializations.map((materialization) => ({
    sourceKind: materialization.sourceKind,
    sourcePath: materialization.sourcePath,
    sourceLabel: materialization.sourceKind === 'artifact-manifest'
      ? 'Artifact manifest'
      : 'Conversion config',
  }));
}

function pickPrimaryMaterialization(materializations) {
  const artifactManifest = materializations.find((entry) => entry.sourceKind === 'artifact-manifest');
  return artifactManifest ?? materializations[0] ?? null;
}

function buildMaterializationFromConfigEntry(configEntry) {
  const rawConfig = normalizeObject(configEntry?.rawConfig);
  const configPath = normalizeText(configEntry?.configPath);
  const modelId = inferConversionConfigModelId(configPath, rawConfig);
  const stubManifest = {
    modelId,
    modelType: rawConfig?.modelType ?? null,
  };
  const materialized = resolveMaterializedManifestFromConversionConfig(rawConfig, stubManifest);
  return {
    modelId,
    sourceKind: 'conversion-config',
    sourcePath: configPath,
    configPath,
    rawConfig,
    manifest: {
      modelId,
      modelType: materialized.modelType,
      architecture: materialized.architecture ?? null,
      inference: materialized.inference,
      metadata: null,
    },
  };
}

function buildMaterializationFromManifestEntry(manifestEntry) {
  const manifest = normalizeObject(manifestEntry?.manifest);
  const manifestPath = normalizeText(manifestEntry?.manifestPath);
  const modelId = normalizeText(manifest?.modelId)
    || normalizeText(manifestEntry?.modelId)
    || normalizeText(manifestPath.split('/').slice(-2, -1)[0])
    || 'model';
  return {
    modelId,
    sourceKind: 'artifact-manifest',
    sourcePath: manifestPath,
    configPath: manifestPath,
    rawConfig: {},
    manifest,
  };
}

function buildSkippedRecord(materialization, reason) {
  return {
    modelId: materialization.modelId,
    configPath: materialization.configPath,
    reason,
    sources: buildModelSources([materialization]),
  };
}

function buildModelRecord(materializations, registryEntries) {
  const primary = pickPrimaryMaterialization(materializations);
  if (!primary) {
    return null;
  }
  const manifest = primary.manifest;
  const manifestInference = manifest?.inference;
  if (
    manifest?.modelType === 'diffusion'
    || manifest?.modelType === 'energy'
    || normalizeText(manifestInference?.pipeline).toLowerCase() === 'diffusion'
    || normalizeText(manifestInference?.pipeline).toLowerCase() === 'energy'
  ) {
    return {
      skipped: true,
      modelId: primary.modelId,
      configPath: primary.configPath,
      reason: `unsupported pipeline scope "${normalizeText(manifestInference?.pipeline) || manifest?.modelType}" for kernel-path builder`,
      sources: buildModelSources(materializations),
    };
  }
  if (!hasExecutionV1(manifestInference)) {
    return {
      skipped: true,
      modelId: primary.modelId,
      configPath: primary.configPath,
      reason: 'source does not materialize an execution-v1 graph',
      sources: buildModelSources(materializations),
    };
  }

  const modelId = primary.modelId;
  const rawConfig = primary.rawConfig ?? {};
  const resolvedSteps = resolveExecutionSteps(manifestInference.execution);
  const layerCount = inferLayerCount(manifestInference, resolvedSteps);
  const sourceConsistency = buildSourceConsistency(materializations);
  const customRuntimeFacts = buildCustomRuntimeFacts({
    modelId,
    manifestInference,
  });

  let compileResult = null;
  let runtimeCompileError = null;
  try {
    compileResult = compileExecutionV1({
      manifestInference,
      modelId,
      numLayers: layerCount ?? 0,
    });
  } catch (error) {
    runtimeCompileError = error instanceof Error ? error.message : String(error);
  }

  let candidateKernelPath = null;
  let candidateError = null;
  const comparableKernelPath = buildComparableKernelPathFromResolvedSteps(
    resolvedSteps,
    manifestInference.session,
    modelId,
    layerCount ?? 0
  );
  try {
    candidateKernelPath = buildInlineKernelPath(
      resolvedSteps,
      manifestInference.session,
      modelId,
      layerCount ?? 0,
      null
    );
  } catch (error) {
    candidateError = error instanceof Error ? error.message : String(error);
  }

  const contractSession = buildExecutionContractSessionFacts(manifest, resolvedSteps);
  const candidateShape = normalizeKernelPathShape(candidateKernelPath ?? comparableKernelPath);

  const baseModelRecord = {
    modelId,
    configPath: primary.configPath,
    family: resolveFamily(primary.configPath, manifest),
    modelType: manifest?.modelType ?? null,
    rawConfig,
    customRuntimeFacts,
  };
  const rankedMatches = rankKernelPathMatches(candidateShape, registryEntries, baseModelRecord);
  const exactMatchIds = rankedMatches.filter((entry) => entry.exact).map((entry) => entry.id);
  const exactMatchEntry = exactMatchIds.length > 0
    ? registryEntries.find((entry) => entry.id === exactMatchIds[0]) || null
    : null;

  const session = normalizeObject(manifestInference.session);
  const computeDefaults = normalizeObject(session.compute?.defaults);
  const layerTypes = Array.isArray(manifestInference?.layerPattern?.layerTypes)
    ? [...manifestInference.layerPattern.layerTypes]
    : null;

  const modelRecord = {
    modelId,
    configPath: primary.configPath,
    family: resolveFamily(primary.configPath, manifest),
    modelType: manifest?.modelType ?? null,
    rawConfig,
    sources: buildModelSources(materializations),
    sourceConsistency,
    session: {
      compute: {
        activationDtype: normalizeText(computeDefaults.activationDtype) || null,
        mathDtype: normalizeText(computeDefaults.mathDtype) || null,
        accumDtype: normalizeText(computeDefaults.accumDtype) || null,
        outputDtype: normalizeText(computeDefaults.outputDtype) || null,
      },
      kvDtype: normalizeText(session?.kvcache?.kvDtype) || null,
      kvLayout: normalizeText(contractSession.facts?.layout) || null,
      decodeLoop: stableValue(normalizeObject(session.decodeLoop)),
      batching: {
        disableCommandBatching: contractSession.facts?.disableCommandBatching ?? null,
        batchSize: contractSession.facts?.decodeBatchSize ?? null,
        readbackInterval: session?.decodeLoop?.readbackInterval ?? null,
        ringTokens: session?.decodeLoop?.ringTokens ?? null,
        ringStop: session?.decodeLoop?.ringStop ?? null,
        ringStaging: session?.decodeLoop?.ringStaging ?? null,
        stopCheckMode: normalizeText(session?.decodeLoop?.stopCheckMode) || null,
      },
      executionContract: {
        source: contractSession.source,
        facts: stableValue(contractSession.facts),
        error: contractSession.error,
      },
      loading: {
        sourceKind: primary.sourceKind,
        sourceRuntimeMode: normalizeText(manifest?.metadata?.sourceRuntime?.mode) || null,
      },
    },
    layerPattern: {
      type: normalizeText(manifestInference?.layerPattern?.type) || null,
      layerCount,
      layerTypes,
      layerTypeCounts: countLayerTypes(layerTypes),
    },
    runtime: {
      inlineKernelPathEnabled: manifestInference?.execution?.inlineKernelPath !== false,
      actualLowering: compileResult?.runtimeInferencePatch?.kernelPath
        ? 'inline-kernel-path'
        : (runtimeCompileError ? 'inline-kernel-path-error' : 'execution-graph-only'),
      hasGeneratedInlineKernelPath: candidateKernelPath != null,
      kernelPathIds: [],
      exactKernelPathIds: exactMatchIds,
      compileError: runtimeCompileError,
    },
    execution: {
      schema: normalizeText(manifestInference?.schema) || null,
      sections: splitResolvedStepsBySection(resolvedSteps),
      kernelRefs: buildKernelRefList(resolvedSteps),
    },
    candidate: {
      available: candidateKernelPath != null,
      error: candidateError,
      closestMatches: rankedMatches.slice(0, 5),
      exactMatchIds,
      proposal: null,
      normalizedShape: candidateShape,
    },
    customRuntimeFacts,
  };

  modelRecord.candidate.proposal = buildProposedKernelPath(
    modelRecord,
    candidateShape,
    exactMatchEntry,
    manifestInference.session
  );

  if (
    modelRecord.runtime.actualLowering === 'inline-kernel-path'
    && modelRecord.candidate.proposal?.verification?.ok === true
  ) {
    const proposalKernelPathId = modelRecord.candidate.proposal.selectedKernelPathId
      || modelRecord.candidate.proposal.path?.id
      || null;
    modelRecord.runtime.kernelPathIds = exactMatchIds.length > 0
      ? [...exactMatchIds]
      : (proposalKernelPathId ? [proposalKernelPathId] : []);
  }

  const signaturePayload = buildExecutionGraphSignaturePayload(modelRecord);
  modelRecord.execution.signature = `execv1:${fnv1aHex(stableStringify(signaturePayload))}`;
  return modelRecord;
}

function buildReverseIndex(records, keyFn) {
  const map = new Map();
  for (const record of records) {
    const keys = keyFn(record);
    for (const key of keys) {
      if (!key) continue;
      const current = map.get(key) || [];
      current.push(record.modelId);
      map.set(key, current);
    }
  }
  const output = {};
  for (const [key, modelIds] of [...map.entries()].sort((left, right) => left[0].localeCompare(right[0]))) {
    output[key] = [...new Set(modelIds)].sort((left, right) => left.localeCompare(right));
  }
  return output;
}

function summarizeRegistryEntries(registryEntries, reverseIndexes) {
  return registryEntries.map((entry) => ({
    id: entry.id,
    status: entry.status,
    statusReason: entry.statusReason,
    notes: entry.notes,
    activationDtype: entry.pathShape?.activationDtype ?? null,
    kvDtype: entry.pathShape?.kvDtype ?? null,
    exactMatchModels: reverseIndexes.exactKernelPaths?.[entry.id] ?? [],
    loweredModels: reverseIndexes.kernelPaths?.[entry.id] ?? [],
  }));
}

function buildGroupedMaterializations(options = {}) {
  const materializations = [];
  const configEntries = Array.isArray(options.configEntries) ? options.configEntries : [];
  const manifestEntries = Array.isArray(options.manifestEntries) ? options.manifestEntries : [];

  for (const configEntry of configEntries) {
    materializations.push(buildMaterializationFromConfigEntry(configEntry));
  }
  for (const manifestEntry of manifestEntries) {
    materializations.push(buildMaterializationFromManifestEntry(manifestEntry));
  }

  const grouped = new Map();
  for (const materialization of materializations) {
    const key = materialization.modelId;
    const current = grouped.get(key) || [];
    current.push(materialization);
    grouped.set(key, current);
  }
  return [...grouped.entries()]
    .sort((left, right) => left[0].localeCompare(right[0]))
    .map(([, entries]) => entries);
}

export function buildKernelPathBuilderIndex(options = {}) {
  const registryEntries = Array.isArray(options.registryEntries)
    ? options.registryEntries
      .map((entry) => ({
        id: normalizeText(entry?.id),
        status: normalizeText(entry?.status) || 'canonical',
        statusReason: normalizeText(entry?.statusReason) || '',
        notes: normalizeText(entry?.notes) || '',
        path: entry?.path ?? null,
        pathShape: normalizeKernelPathShape(entry?.path),
      }))
      .filter((entry) => entry.id && entry.pathShape)
    : [];

  const supportedModels = [];
  const skipped = [];
  for (const materializations of buildGroupedMaterializations(options)) {
    const record = buildModelRecord(materializations, registryEntries);
    if (!record) {
      continue;
    }
    if (record?.skipped) {
      skipped.push(record);
      continue;
    }
    supportedModels.push(record);
  }

  supportedModels.sort((left, right) => left.modelId.localeCompare(right.modelId));

  const reverseIndexes = {
    kernelPaths: buildReverseIndex(supportedModels, (record) => record.runtime.kernelPathIds),
    exactKernelPaths: buildReverseIndex(supportedModels, (record) => record.runtime.exactKernelPathIds),
    executionGraphs: buildReverseIndex(supportedModels, (record) => [record.execution.signature]),
    kernels: buildReverseIndex(supportedModels, (record) => record.execution.kernelRefs.map((entry) => entry.key)),
  };

  return {
    schemaVersion: 2,
    artifactKind: 'execution-index',
    source: 'kernel-path-builder',
    stats: {
      models: supportedModels.length,
      skipped: skipped.length,
      registryKernelPaths: registryEntries.length,
      configSources: Array.isArray(options.configEntries) ? options.configEntries.length : 0,
      manifestSources: Array.isArray(options.manifestEntries) ? options.manifestEntries.length : 0,
      exactKernelPathMatches: supportedModels.filter((record) => record.runtime.exactKernelPathIds.length > 0).length,
      loweredThroughKernelPaths: supportedModels.filter((record) => record.runtime.kernelPathIds.length > 0).length,
      proposalOnlyModels: supportedModels.filter((record) => (
        record.runtime.actualLowering === 'inline-kernel-path'
        && record.runtime.exactKernelPathIds.length === 0
        && record.candidate.proposal?.verification?.ok === true
      )).length,
      verifiedProposals: supportedModels.filter((record) => record.candidate.proposal?.verification?.ok === true).length,
      unverifiedProposals: supportedModels.filter((record) => record.candidate.proposal && record.candidate.proposal.verification?.ok !== true).length,
      modelsWithArtifactManifest: supportedModels.filter((record) => record.sources.some((source) => source.sourceKind === 'artifact-manifest')).length,
      modelsWithSourceDrift: supportedModels.filter((record) => record.sourceConsistency?.ok === false).length,
      executionGraphOnlyModels: supportedModels.filter((record) => record.runtime.actualLowering === 'execution-graph-only').length,
    },
    skipped,
    kernelPaths: summarizeRegistryEntries(registryEntries, reverseIndexes),
    reverseIndexes,
    models: supportedModels.map((record) => {
      const { rawConfig, ...publicRecord } = record;
      return publicRecord;
    }),
  };
}

export function buildKernelPathBuilderProposals(indexPayload) {
  const models = Array.isArray(indexPayload?.models) ? indexPayload.models : [];
  const proposals = models
    .filter((model) => model?.candidate?.proposal)
    .map((model) => ({
      modelId: model.modelId,
      actualLowering: model.runtime?.actualLowering ?? null,
      sources: model.sources ?? [],
      exactKernelPathIds: model.runtime?.exactKernelPathIds ?? [],
      proposal: model.candidate.proposal,
      closestMatches: Array.isArray(model.candidate?.closestMatches)
        ? model.candidate.closestMatches.slice(0, 3)
        : [],
      customRuntimeFacts: model.customRuntimeFacts ?? [],
    }));
  return {
    schemaVersion: 1,
    artifactKind: 'kernel-path-proposals',
    source: indexPayload?.source ?? 'kernel-path-builder',
    stats: {
      proposals: proposals.length,
      verified: proposals.filter((entry) => entry.proposal?.verification?.ok === true).length,
      unverified: proposals.filter((entry) => entry.proposal?.verification?.ok !== true).length,
      newKernelPaths: proposals.filter((entry) => entry.proposal?.kind === 'proposed').length,
      existingKernelPaths: proposals.filter((entry) => entry.proposal?.kind === 'existing').length,
    },
    proposals,
  };
}

function formatLayers(value) {
  if (value === 'all') return 'all';
  if (!Array.isArray(value) || value.length === 0) return '-';
  return value.join(', ');
}

function renderResolvedStepTableRows(model) {
  const rows = [];
  for (const [section, steps] of Object.entries(model.execution?.sections ?? {})) {
    for (const step of Array.isArray(steps) ? steps : []) {
      rows.push(
        `| ${section} | ${step.phase || '-'} | ${step.op || '-'} | ${step.kernel || '-'} | ` +
        `${step.entry || 'main'} | ${step.digest || '-'} | ${step.weights || '-'} | ${formatLayers(step.layers)} |`
      );
    }
  }
  if (rows.length === 0) {
    return ['| - | - | - | - | - | - | - | - |'];
  }
  return rows;
}

export function renderKernelPathBuilderReportMarkdown(indexPayload, proposalsPayload = null) {
  const payload = indexPayload ?? { stats: {}, models: [], skipped: [] };
  const models = Array.isArray(payload.models) ? payload.models : [];
  const lines = [];
  lines.push('# Kernel-Path Builder Report');
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Models indexed: ${payload.stats?.models ?? 0}`);
  lines.push(`- Registry kernel paths: ${payload.stats?.registryKernelPaths ?? 0}`);
  lines.push(`- Config sources: ${payload.stats?.configSources ?? 0}`);
  lines.push(`- Artifact manifest sources: ${payload.stats?.manifestSources ?? 0}`);
  lines.push(`- Exact matches: ${payload.stats?.exactKernelPathMatches ?? 0}`);
  lines.push(`- Verified proposals: ${payload.stats?.verifiedProposals ?? 0}`);
  lines.push(`- Unverified proposals: ${payload.stats?.unverifiedProposals ?? 0}`);
  if (Array.isArray(payload.skipped) && payload.skipped.length > 0) {
    lines.push(`- Skipped: ${payload.skipped.length}`);
  }
  lines.push('');

  if (proposalsPayload?.stats) {
    lines.push('## Proposal Stats');
    lines.push('');
    lines.push(`- Proposal records: ${proposalsPayload.stats.proposals ?? 0}`);
    lines.push(`- New kernel-path ids: ${proposalsPayload.stats.newKernelPaths ?? 0}`);
    lines.push(`- Existing kernel-path ids reused: ${proposalsPayload.stats.existingKernelPaths ?? 0}`);
    lines.push('');
  }

  if (Array.isArray(payload.skipped) && payload.skipped.length > 0) {
    lines.push('## Skipped Sources');
    lines.push('');
    for (const skipped of payload.skipped) {
      lines.push(`- \`${skipped.modelId}\` (${skipped.reason})`);
    }
    lines.push('');
  }

  lines.push('## Models');
  lines.push('');
  for (const model of models) {
    lines.push(`### ${model.modelId}`);
    lines.push('');
    lines.push(`- Sources: ${(model.sources || []).map((source) => `\`${source.sourcePath}\``).join(', ') || 'none'}`);
    lines.push(`- Lowering: \`${model.runtime?.actualLowering || 'unknown'}\``);
    lines.push(`- Session: activation=\`${model.session?.compute?.activationDtype || '--'}\`, kv=\`${model.session?.kvDtype || '--'}\`, layout=\`${model.session?.kvLayout || '--'}\`, batch=\`${model.session?.batching?.batchSize ?? '--'}\`, readback=\`${model.session?.batching?.readbackInterval ?? '--'}\``);
    lines.push(`- Execution signature: \`${model.execution?.signature || '--'}\``);
    if (model.sourceConsistency?.ok === false) {
      lines.push(`- Source consistency: mismatch across ${model.sourceConsistency.comparedSources} sources`);
    }
    lines.push('');
    lines.push('| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |');
    lines.push('| --- | --- | --- | --- | --- | --- | --- | --- |');
    lines.push(...renderResolvedStepTableRows(model));
    lines.push('');

    const matches = Array.isArray(model.candidate?.closestMatches) ? model.candidate.closestMatches : [];
    if (matches.length > 0) {
      lines.push('Closest matches:');
      for (const match of matches.slice(0, 3)) {
        lines.push(`- \`${match.id}\` (${match.mismatchCount} mismatches)`);
        for (const detail of match.mismatchDetails || []) {
          lines.push(`  - ${detail.label} Hint: ${detail.repairHint}`);
        }
      }
      lines.push('');
    }

    if (model.candidate?.proposal) {
      lines.push('Proposal:');
      lines.push('');
      lines.push('```json');
      lines.push(JSON.stringify(model.candidate.proposal, null, 2));
      lines.push('```');
      lines.push('');
    }
  }
  return `${lines.join('\n')}\n`;
}

export {
  aggregateTopDecodeTimers,
  buildKernelPathBuilderRuntimeOverlay,
};

export function buildKernelPathBuilderArtifacts(options = {}) {
  const index = buildKernelPathBuilderIndex(options);
  const proposals = buildKernelPathBuilderProposals(index);
  const reportMarkdown = renderKernelPathBuilderReportMarkdown(index, proposals);
  return {
    index,
    proposals,
    reportMarkdown,
  };
}
