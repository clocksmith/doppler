import { expandExecutionV1, EXECUTION_V1_SCHEMA_ID } from '../../../config/schema/index.js';
import {
  buildInlineKernelPath,
  buildLayerPipelineFromExecution,
  buildSessionRuntimePatch,
  PIPELINE_COMPATIBLE_OPS,
  requireSessionActivationDtype,
  requireSessionKVDtype,
} from './execution-runtime-builders.js';
import { mergeKernelPathPolicy } from '../../../config/merge-helpers.js';
import { mergeRuntimeValues } from '../../../config/runtime-merge.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { buildOpIdFromExecutionStep } from './operator-identity.js';
import {
  resolveCapabilityTransforms,
  resolveFinitenessFallbackTransform,
} from '../../../config/transforms/capability-transform-resolver.js';
import { composeTransforms } from '../../../config/transforms/execution-graph-transforms.js';
import { log } from '../../../debug/index.js';
import { resolveRangeAwareSelectiveWideningConfig } from './finiteness-policy.js';

const SESSION_CAPABILITY_TRANSFORMS = new Set([
  'disableRetainQ4KMaterialization',
]);

export function hasExecutionV1(manifestInference) {
  return manifestInference?.schema === EXECUTION_V1_SCHEMA_ID
    && manifestInference?.execution
    && typeof manifestInference.execution.kernels === 'object';
}

function isSessionCapabilityTransform(name) {
  return SESSION_CAPABILITY_TRANSFORMS.has(name);
}

function resolveExecutionCapabilities(capabilities) {
  if (capabilities && typeof capabilities === 'object') {
    return capabilities;
  }
  return {
    hasSubgroups: false,
    hasSubgroupsF16: false,
    hasF16: false,
    hasTimestampQuery: false,
    maxBufferSize: 0,
    maxWorkgroupSize: 0,
    maxWorkgroupStorageSize: 0,
    adapterInfo: {
      vendor: 'unknown',
      architecture: 'unknown',
      device: 'unknown',
      description: 'execution-v1-no-f16-proof',
    },
  };
}

function resolveExecutionSessionKVDtype(session, manifestInference, capabilities, useGPU) {
  const runtimeKV = session?.kvcache ?? null;
  if (!runtimeKV) {
    return session;
  }
  if (useGPU == null && !capabilities) {
    return session;
  }
  const attnSoftcap = manifestInference?.attention?.attnLogitSoftcapping;
  const forceF32Softcap = runtimeKV.forceF32Softcap === true;
  const forceF32KV = attnSoftcap != null && attnSoftcap > 0 && forceF32Softcap;
  const kvDtype = selectRuleValue('inference', 'dtype', 'kvCacheDtype', {
    requested: runtimeKV.kvDtype,
    useGPU: useGPU === true,
    hasF16: capabilities?.hasF16 === true,
    forceF32: forceF32KV,
  });
  if (kvDtype === runtimeKV.kvDtype) {
    return session;
  }
  return {
    ...session,
    kvcache: {
      ...runtimeKV,
      kvDtype,
    },
  };
}

function mergeExecutionV1Session(manifestSession, runtimeSession) {
  return mergeRuntimeValues(
    manifestSession ?? {},
    runtimeSession ?? {}
  );
}

function hasOwnProperty(value, key) {
  return value != null && Object.prototype.hasOwnProperty.call(value, key);
}

function hasPatchEntries(value, label) {
  if (value == null) {
    return false;
  }
  if (!Array.isArray(value)) {
    throw new Error(`[ExecutionV1] ${label} must be an array when provided.`);
  }
  return value.length > 0;
}

function cloneExecutionTuple(tuple) {
  return Array.isArray(tuple) ? [...tuple] : tuple;
}

function cloneExecutionEntry(entry) {
  if (Array.isArray(entry)) {
    return cloneExecutionTuple(entry);
  }
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    return entry;
  }
  return {
    ...entry,
    layers: Array.isArray(entry.layers) ? [...entry.layers] : entry.layers,
    steps: Array.isArray(entry.steps) ? entry.steps.map((step) => cloneExecutionTuple(step)) : entry.steps,
  };
}

function cloneKernelDeclaration(kernel) {
  if (!kernel || typeof kernel !== 'object' || Array.isArray(kernel)) {
    return kernel;
  }
  return {
    ...kernel,
    ...(kernel.constants ? { constants: { ...kernel.constants } } : {}),
    ...(kernel.precision ? { precision: { ...kernel.precision } } : {}),
  };
}

function cloneExecutionGraph(execution) {
  return {
    ...execution,
    kernels: Object.fromEntries(
      Object.entries(execution?.kernels ?? {}).map(
        ([key, kernel]) => [key, cloneKernelDeclaration(kernel)]
      )
    ),
    preLayer: (execution?.preLayer ?? []).map((entry) => cloneExecutionEntry(entry)),
    decode: (execution?.decode ?? []).map((entry) => cloneExecutionEntry(entry)),
    prefill: (execution?.prefill ?? []).map((entry) => cloneExecutionEntry(entry)),
    postLayer: (execution?.postLayer ?? []).map((entry) => cloneExecutionEntry(entry)),
    policies: execution?.policies ? { ...execution.policies } : execution?.policies,
  };
}

function normalizePatchSection(section, label) {
  if (section == null) {
    return null;
  }
  if (typeof section !== 'string') {
    throw new Error(`[ExecutionV1] ${label}.section must be a string when provided.`);
  }
  const normalized = section.trim();
  if (normalized === 'decode' || normalized === 'prefill' || normalized === 'preLayer' || normalized === 'postLayer') {
    return normalized;
  }
  const lower = normalized.toLowerCase();
  if (lower === 'prelayer') {
    return 'preLayer';
  }
  if (lower === 'postlayer') {
    return 'postLayer';
  }
  throw new Error(
    `[ExecutionV1] ${label}.section must be one of decode|prefill|preLayer|postLayer.`
  );
}

function assertPatchLayersUnsupported(layers, label) {
  if (layers != null) {
    throw new Error(`[ExecutionV1] ${label}.layers is not supported yet.`);
  }
}

function applyPatchKernelAdditions(execution, addKernels) {
  for (const [index, addition] of addKernels.entries()) {
    const label = `runtime.inference.executionPatch.addKernels[${index}]`;
    if (!addition || typeof addition !== 'object' || Array.isArray(addition)) {
      throw new Error(`[ExecutionV1] ${label} must be an object.`);
    }
    const key = typeof addition.key === 'string' ? addition.key.trim() : '';
    if (!key) {
      throw new Error(`[ExecutionV1] ${label}.key must be a non-empty string.`);
    }
    if (hasOwnProperty(execution.kernels, key)) {
      throw new Error(`[ExecutionV1] ${label}.key "${key}" already exists in execution.kernels.`);
    }
    if (!addition.kernel || typeof addition.kernel !== 'object' || Array.isArray(addition.kernel)) {
      throw new Error(`[ExecutionV1] ${label}.kernel must be an object.`);
    }
    execution.kernels[key] = cloneKernelDeclaration(addition.kernel);
  }
}

function patchTuple(tuple, patch, label) {
  const patched = [...tuple];
  if (hasOwnProperty(patch, 'kernelKey') && patch.kernelKey !== undefined) {
    const kernelKey = typeof patch.kernelKey === 'string' ? patch.kernelKey.trim() : '';
    if (!kernelKey) {
      throw new Error(`[ExecutionV1] ${label}.kernelKey must be a non-empty string.`);
    }
    patched[1] = kernelKey;
  }
  if (hasOwnProperty(patch, 'weights') && patch.weights !== undefined) {
    const weights = typeof patch.weights === 'string' ? patch.weights.trim() : '';
    if (!weights) {
      throw new Error(`[ExecutionV1] ${label}.weights must be a non-empty string.`);
    }
    if (patched.length > 2) {
      patched[2] = weights;
    } else {
      patched.push(weights);
    }
  }
  return patched;
}

function applyPatchSetToEntries(entries, patch, label) {
  let matchCount = 0;
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    if (Array.isArray(entry)) {
      if (entry[0] !== patch.op) {
        continue;
      }
      entries[index] = patchTuple(entry, patch, label);
      matchCount += 1;
      continue;
    }
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.steps)) {
      continue;
    }
    for (let stepIndex = 0; stepIndex < entry.steps.length; stepIndex += 1) {
      const step = entry.steps[stepIndex];
      if (!Array.isArray(step) || step[0] !== patch.op) {
        continue;
      }
      entry.steps[stepIndex] = patchTuple(step, patch, label);
      matchCount += 1;
    }
  }
  return matchCount;
}

function applyPatchSets(execution, setPatches) {
  const defaultSections = ['preLayer', 'decode', 'prefill', 'postLayer'];
  for (const [index, patch] of setPatches.entries()) {
    const label = `runtime.inference.executionPatch.set[${index}]`;
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) {
      throw new Error(`[ExecutionV1] ${label} must be an object.`);
    }
    const op = typeof patch.op === 'string' ? patch.op.trim() : '';
    if (!op) {
      throw new Error(`[ExecutionV1] ${label}.op must be a non-empty string.`);
    }
    assertPatchLayersUnsupported(patch.layers, label);
    if (!hasOwnProperty(patch, 'kernelKey') && !hasOwnProperty(patch, 'weights')) {
      throw new Error(`[ExecutionV1] ${label} must provide kernelKey or weights.`);
    }
    const normalizedPatch = {
      ...patch,
      op,
      section: normalizePatchSection(patch.section, label),
    };
    const sections = normalizedPatch.section ? [normalizedPatch.section] : defaultSections;
    let matchCount = 0;
    for (const section of sections) {
      matchCount += applyPatchSetToEntries(execution[section] ?? [], normalizedPatch, label);
    }
    if (matchCount === 0) {
      throw new Error(`[ExecutionV1] ${label} did not match any execution step.`);
    }
  }
}

function assertUnsupportedPatchLists(executionPatch) {
  if (hasPatchEntries(executionPatch.remove, 'runtime.inference.executionPatch.remove')) {
    throw new Error('[ExecutionV1] runtime.inference.executionPatch.remove is not supported yet.');
  }
  if (hasPatchEntries(executionPatch.add, 'runtime.inference.executionPatch.add')) {
    throw new Error('[ExecutionV1] runtime.inference.executionPatch.add is not supported yet.');
  }
}

function applyExecutionPatch(execution, executionPatch) {
  if (executionPatch == null) {
    return execution;
  }
  if (typeof executionPatch !== 'object' || Array.isArray(executionPatch)) {
    throw new Error('[ExecutionV1] runtime.inference.executionPatch must be an object.');
  }
  assertUnsupportedPatchLists(executionPatch);
  const hasKernelAdditions = hasPatchEntries(
    executionPatch.addKernels,
    'runtime.inference.executionPatch.addKernels'
  );
  const hasSets = hasPatchEntries(executionPatch.set, 'runtime.inference.executionPatch.set');
  if (!hasKernelAdditions && !hasSets) {
    return execution;
  }
  const patchedExecution = cloneExecutionGraph(execution);
  if (hasKernelAdditions) {
    applyPatchKernelAdditions(patchedExecution, executionPatch.addKernels);
  }
  if (hasSets) {
    applyPatchSets(patchedExecution, executionPatch.set);
  }
  return patchedExecution;
}

function resolveRuntimeInferenceOverrideSection(runtimeOverrides, key) {
  const inferenceOverrides = runtimeOverrides?.inference;
  if (!inferenceOverrides || typeof inferenceOverrides !== 'object' || Array.isArray(inferenceOverrides)) {
    return null;
  }
  if (!hasOwnProperty(inferenceOverrides, key)) {
    return null;
  }
  return inferenceOverrides[key] ?? null;
}

function preserveRuntimeDecodeLoop(updatedInference, runtimeConfig) {
  const runtimeSession = runtimeConfig?.inference?.session;
  if (!hasOwnProperty(runtimeSession, 'decodeLoop')) {
    return updatedInference;
  }
  const updatedSession = updatedInference?.session;
  if (!updatedSession || typeof updatedSession !== 'object' || Array.isArray(updatedSession)) {
    return {
      ...updatedInference,
      session: {
        decodeLoop: runtimeSession.decodeLoop,
      },
    };
  }
  return {
    ...updatedInference,
    session: {
      ...updatedSession,
      decodeLoop: runtimeSession.decodeLoop,
    },
  };
}

function preserveConfiguredKernelPath(updatedInference, runtimeConfig) {
  const configuredInference = runtimeConfig?.inference;
  const configuredKernelPath = configuredInference?.kernelPath;
  if (configuredKernelPath == null) {
    return updatedInference;
  }
  const configuredSession = configuredInference?.session;
  const hasConfiguredSessionCompute = hasOwnProperty(configuredSession, 'compute');
  const hasConfiguredSessionKVCache = hasOwnProperty(configuredSession, 'kvcache');
  return {
    ...updatedInference,
    kernelPath: configuredKernelPath,
    kernelPathSource: 'config',
    ...(hasOwnProperty(configuredInference, 'compute')
      ? { compute: configuredInference.compute }
      : {}),
    ...(hasConfiguredSessionCompute || hasConfiguredSessionKVCache
      ? {
          session: {
            ...updatedInference.session,
            ...(hasConfiguredSessionCompute ? { compute: configuredSession.compute } : {}),
            ...(hasConfiguredSessionKVCache ? { kvcache: configuredSession.kvcache } : {}),
          },
        }
      : {}),
  };
}

const EXECUTION_V1_PROJECTION_OPS = new Set([
  'q_proj', 'k_proj', 'v_proj', 'o_proj',
  'gate_proj', 'up_proj', 'down_proj',
]);

const EXECUTION_V1_DENSE_Q4_PREFILL_FILES = new Set([
  'matmul_f16w_f32a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'matmul_f16.wgsl',
  'matmul_f16_tiled.wgsl',
]);

const EXECUTION_V1_F32_ACTIVATION_NARROWING_FILES = new Set([
  'rmsnorm.wgsl',
  'rope.wgsl',
  'residual.wgsl',
  'gelu.wgsl',
  'sample.wgsl',
  'gather.wgsl',
  'matmul_gemv_subgroup.wgsl',
  'matmul_f16w_f32a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'attention_decode_online_f16kv.wgsl',
  'attention_decode_chunked_f16kv.wgsl',
  'attention_small_f16kv.wgsl',
  'attention_streaming_f16kv.wgsl',
  'attention_decode.wgsl',
  'attention_small.wgsl',
  'attention_streaming.wgsl',
  'silu.wgsl',
]);

function summarizeExecutionGraphContext(execution) {
  const summary = {
    hasDensePrefillProjectionKernel: false,
    hasQ4DecodeProjectionKernel: false,
    hasQ4PrefillProjectionKernel: false,
    hasAvailableQ4PrefillProjectionKernel: false,
    requiresF16ActivationNarrowing: false,
  };
  const phases = [
    ['decode', execution?.decode ?? []],
    ['prefill', execution?.prefill ?? []],
  ];

  for (const [phase, steps] of phases) {
    for (const step of steps) {
      if (!EXECUTION_V1_PROJECTION_OPS.has(step[0])) {
        continue;
      }
      const kernelEntry = execution?.kernels?.[step[1]];
      if (!kernelEntry) {
        continue;
      }
      if (phase === 'decode' && kernelEntry.kernel === 'fused_matmul_q4.wgsl') {
        summary.hasQ4DecodeProjectionKernel = true;
      }
      if (phase === 'prefill' && EXECUTION_V1_DENSE_Q4_PREFILL_FILES.has(kernelEntry.kernel)) {
        summary.hasDensePrefillProjectionKernel = true;
      }
      if (phase === 'prefill' && kernelEntry.kernel.startsWith('fused_matmul_q4')) {
        summary.hasQ4PrefillProjectionKernel = true;
      }
    }
  }

  summary.hasAvailableQ4PrefillProjectionKernel = Object.values(execution?.kernels ?? {}).some(
    (entry) => entry?.kernel === 'fused_matmul_q4_batched_multicol_shared.wgsl'
      || entry?.kernel === 'fused_matmul_q4_batched.wgsl'
  );
  summary.requiresF16ActivationNarrowing = Object.values(execution?.kernels ?? {}).some(
    (entry) => EXECUTION_V1_F32_ACTIVATION_NARROWING_FILES.has(entry?.kernel)
  );

  return summary;
}

function normalizeExecutionLayerType(layerType) {
  return typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
}

function isLinearExecutionLayerType(layerType) {
  const normalized = normalizeExecutionLayerType(layerType);
  return normalized === 'linear_attention'
    || normalized === 'linear'
    || normalized === 'gated_delta'
    || normalized === 'gated_delta_net';
}

function isFullAttentionExecutionLayerType(layerType) {
  const normalized = normalizeExecutionLayerType(layerType);
  return normalized === 'full_attention'
    || normalized === 'full'
    || normalized === 'global'
    || normalized === 'standard';
}

function isFusedQ4ProjectionKernel(kernelEntry) {
  return typeof kernelEntry?.kernel === 'string'
    && kernelEntry.kernel.startsWith('fused_matmul_q4');
}

function collectPhaseOpEntries(phaseEntries, kernels, ops) {
  const entries = [];
  for (const entry of phaseEntries ?? []) {
    if (Array.isArray(entry)) {
      if (!ops.has(entry[0])) {
        continue;
      }
      entries.push({
        op: entry[0],
        layers: 'all',
        kernelEntry: kernels?.[entry[1]] ?? null,
      });
      continue;
    }
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.layers) || !Array.isArray(entry.steps)) {
      continue;
    }
    for (const step of entry.steps) {
      if (!Array.isArray(step) || !ops.has(step[0])) {
        continue;
      }
      entries.push({
        op: step[0],
        layers: entry.layers,
        kernelEntry: kernels?.[step[1]] ?? null,
      });
    }
  }
  return entries;
}

function collectCoveredLayers(entries, targetLayers, predicate) {
  const covered = new Set();
  for (const entry of entries) {
    if (!predicate(entry.kernelEntry)) {
      continue;
    }
    if (entry.layers === 'all') {
      for (const layerIdx of targetLayers) {
        covered.add(layerIdx);
      }
      continue;
    }
    for (const layerIdx of entry.layers ?? []) {
      if (targetLayers.includes(layerIdx)) {
        covered.add(layerIdx);
      }
    }
  }
  return covered;
}

function assertHybridLinearProjectionIsolation(execution, layerTypes, modelId, weightDtype) {
  if (!Array.isArray(layerTypes) || layerTypes.length === 0) {
    return;
  }
  const normalizedWeightDtype = String(weightDtype || '').trim().toLowerCase();
  if (normalizedWeightDtype === 'f16' || normalizedWeightDtype === 'f32') {
    return;
  }

  const linearLayers = layerTypes
    .map((layerType, layerIdx) => ({ layerType, layerIdx }))
    .filter(({ layerType }) => isLinearExecutionLayerType(layerType))
    .map(({ layerIdx }) => layerIdx);
  const fullAttentionLayers = layerTypes
    .map((layerType, layerIdx) => ({ layerType, layerIdx }))
    .filter(({ layerType }) => isFullAttentionExecutionLayerType(layerType))
    .map(({ layerIdx }) => layerIdx);

  if (linearLayers.length === 0 || fullAttentionLayers.length === 0) {
    return;
  }

  const guardedOps = [
    {
      phase: 'decode',
      label: 'q_proj/linear_qkv_proj',
      ops: new Set(['q_proj', 'qkv_proj', 'linear_qkv_proj']),
    },
    {
      phase: 'decode',
      label: 'o_proj/linear_out_proj',
      ops: new Set(['o_proj', 'linear_out_proj']),
    },
    {
      phase: 'prefill',
      label: 'q_proj/linear_qkv_proj',
      ops: new Set(['q_proj', 'qkv_proj', 'linear_qkv_proj']),
    },
    {
      phase: 'prefill',
      label: 'o_proj/linear_out_proj',
      ops: new Set(['o_proj', 'linear_out_proj']),
    },
  ];

  for (const guard of guardedOps) {
    const entries = collectPhaseOpEntries(execution?.[guard.phase], execution?.kernels, guard.ops);
    if (entries.length === 0) {
      continue;
    }

    const hasUnsafeGlobalEntry = entries.some(
      (entry) => entry.layers === 'all' && !isFusedQ4ProjectionKernel(entry.kernelEntry)
    );
    if (!hasUnsafeGlobalEntry) {
      continue;
    }

    const coveredLinearLayers = collectCoveredLayers(
      entries,
      linearLayers,
      (kernelEntry) => isFusedQ4ProjectionKernel(kernelEntry)
    );
    const missingLinearLayers = linearLayers.filter((layerIdx) => !coveredLinearLayers.has(layerIdx));
    if (missingLinearLayers.length === 0) {
      continue;
    }

    throw new Error(
      `[ExecutionV1] Hybrid linear-attention model "${modelId}" cannot apply a global non-Q4 ` +
      `${guard.phase} ${guard.label} kernel because linear-attention layers alias that role. ` +
      `Missing fused Q4 coverage for linear layers [${missingLinearLayers.join(', ')}]. ` +
      'Keep the linear-attention path on fused_matmul_q4* or isolate those layers with explicit execution-v1 overrides.'
    );
  }
}


function expandV1ToResolvedSteps(execution, options = {}) {
  const expanded = expandExecutionV1(execution, options);
  return expanded.map((step, index) => {
    const resolved = {
      id: `${step.section}_${step.phase}_${index}_${step.op}`,
      phase: step.phase,
      section: step.section,
      op: step.op,
      src: step.src,
      dst: step.dst,
      kernel: step.kernel,
      entry: step.entry,
      ...(step.weights ? { weights: step.weights } : {}),
      ...(step.constants ? { constants: step.constants } : {}),
      ...(step.precision ? { precision: step.precision } : {}),
      layers: step.layers,
      kernelRef: {
        id: `${step.kernel.replace('.wgsl', '')}.${step.entry}`,
        version: '1.0.0',
        digest: step.digest,
      },
    };
    resolved.canonicalOpId = buildOpIdFromExecutionStep(resolved);
    return resolved;
  });
}

function buildFinitenessFallbackSession(session, fallbackKvDtype) {
  const kvDtype = fallbackKvDtype === 'f16' ? 'f16' : 'f32';
  const computeDefaults = session?.compute?.defaults ?? {};
  const fallbackDefaults = {
    ...computeDefaults,
    activationDtype: 'f32',
    outputDtype: 'f32',
    ...(kvDtype === 'f32' ? {
      mathDtype: 'f32',
      accumDtype: 'f32',
    } : {}),
  };
  return {
    ...session,
    compute: {
      ...(session.compute ?? {}),
      defaults: fallbackDefaults,
    },
    kvcache: {
      ...(session.kvcache ?? {}),
      kvDtype,
    },
  };
}

function buildFinitenessFallbackKernelPath({
  execution,
  effectiveSession,
  modelId,
  numLayers,
  headDim,
  layerTypes,
  capabilities,
  platform,
}) {
  const activationDtype = effectiveSession?.compute?.defaults?.activationDtype ?? null;
  const mathDtype = effectiveSession?.compute?.defaults?.mathDtype ?? null;
  const accumDtype = effectiveSession?.compute?.defaults?.accumDtype ?? null;
  const kvDtype = effectiveSession?.kvcache?.kvDtype ?? null;
  if (activationDtype !== 'f16' || !execution) {
    return null;
  }

  const fallback = resolveFinitenessFallbackTransform({
    activationDtype,
    mathDtype,
    accumDtype,
    kvDtype,
    headDim,
    modelId,
    layerTypes,
  });
  if (!fallback) {
    return null;
  }

  const transformed = fallback.transform(execution, {
    capabilities,
    platform: platform ?? { id: 'unknown', vendor: 'unknown', architecture: 'unknown' },
    activationDtype,
    mathDtype,
    accumDtype,
    kvDtype,
    headDim,
    modelId,
    layerTypes,
  });
  if (!transformed) {
    return null;
  }

  const fallbackSession = buildFinitenessFallbackSession(
    effectiveSession,
    fallback.fallbackKvDtype
  );
  const fallbackSteps = expandV1ToResolvedSteps(transformed, { skipDigestValidation: true });
  const fallbackKernelPath = buildInlineKernelPath(
    fallbackSteps,
    fallbackSession,
    `${modelId || 'model'}-${fallback.name}-finiteness-fallback`,
    numLayers
  );
  if (!fallbackKernelPath) {
    return null;
  }
  return {
    ...fallbackKernelPath,
    name: 'Execution inline finiteness fallback kernel path',
    description: `Generated from manifest.inference.execution using ${fallback.name}`,
    finitenessFallbackTransform: fallback.name,
  };
}


export function compileExecutionV1(options = {}) {
  const manifestInference = options.manifestInference;
  const modelId = options.modelId ?? 'model';
  const numLayers = options.numLayers ?? 0;
  const headDim = Number.isFinite(options.headDim) ? Math.floor(options.headDim) : null;
  const weightDtype = options.weightDtype ?? null;
  const hasCapabilityProof = options.capabilities && typeof options.capabilities === 'object';
  const hasUseGPUOption = hasOwnProperty(options, 'useGPU');
  const useGPU = hasUseGPUOption
    ? options.useGPU === true
    : hasCapabilityProof
      ? true
      : null;
  const capabilities = hasCapabilityProof || hasUseGPUOption
    ? resolveExecutionCapabilities(options.capabilities ?? null)
    : null;
  const platform = options.platform ?? null;
  const runtimeSession = options.runtimeSession ?? null;
  const runtimeCompute = options.runtimeCompute ?? null;
  const kernelPathPolicy = mergeKernelPathPolicy(undefined, options.kernelPathPolicy ?? undefined);
  const finitenessPolicy = resolveRangeAwareSelectiveWideningConfig(runtimeCompute);

  if (!hasExecutionV1(manifestInference)) {
    throw new Error(`[ExecutionV1] manifest.inference.schema must be "${EXECUTION_V1_SCHEMA_ID}".`);
  }

  const manifestExecution = manifestInference.execution;
  expandExecutionV1(manifestExecution);
  let execution = applyExecutionPatch(manifestExecution, options.executionPatch ?? null);
  const mergedSession = mergeExecutionV1Session(
    manifestInference.session ?? {},
    runtimeSession ?? {}
  );
  const declaredSession = mergedSession;
  const session = resolveExecutionSessionKVDtype(
    mergedSession,
    manifestInference,
    capabilities,
    useGPU
  );

  if (!session?.compute?.defaults?.activationDtype) {
    throw new Error('[ExecutionV1] session.compute.defaults.activationDtype is required.');
  }

  const activationDtype = session.compute.defaults.activationDtype;
  const mathDtype = session.compute.defaults.mathDtype ?? null;
  const accumDtype = session.compute.defaults.accumDtype ?? null;
  const requestedActivationDtype = runtimeCompute?.activationDtype ?? activationDtype;
  const kvDtype = requireSessionKVDtype(session);
  const declaredActivationDtype = declaredSession.compute?.defaults?.activationDtype ?? activationDtype;
  const declaredMathDtype = declaredSession.compute?.defaults?.mathDtype ?? mathDtype;
  const declaredAccumDtype = declaredSession.compute?.defaults?.accumDtype ?? accumDtype;
  const declaredKvDtype = declaredSession.kvcache?.kvDtype ?? kvDtype;
  const layerTypes = manifestInference?.layerPattern?.layerTypes ?? null;

  if (execution !== manifestExecution) {
    expandExecutionV1(execution);
  }

  // -------------------------------------------------------------------------
  // Phase 2: Apply capability transforms to the execution graph
  // -------------------------------------------------------------------------
  let appliedTransformNames = [];
  let graphWasTransformed = false;
  let capabilityTransformPolicy = null;
  const graphContext = {
    requestedActivationDtype,
    activationDtype,
    mathDtype,
    accumDtype,
    kvDtype,
    headDim,
    modelId,
    layerTypes,
    retainQ4KMaterialization: session?.retainQ4KMaterialization === true,
    ...summarizeExecutionGraphContext(execution),
  };

  if (capabilities) {
    const resolved = resolveCapabilityTransforms(capabilities, platform, graphContext);
    capabilityTransformPolicy = {
      kind: resolved.kind ?? null,
      dtypeEffect: resolved.dtypeEffect ?? null,
      reason: resolved.reason ?? null,
      evidence: Array.isArray(resolved.evidence) ? [...resolved.evidence] : [],
    };
    const sourceScope = kernelPathPolicy.sourceScope ?? kernelPathPolicy.allowSources ?? [];
    const remapAllowed = kernelPathPolicy.mode === 'capability-aware'
      && kernelPathPolicy.onIncompatible === 'remap'
      && sourceScope.includes('manifest');

    if (resolved.transforms.length > 0) {
      if (!remapAllowed) {
        throw new Error(
          `[ExecutionV1] capability transforms required for "${modelId}" (${resolved.reason}) ` +
          `but runtime.inference.kernelPathPolicy is ${JSON.stringify(kernelPathPolicy)}. ` +
          'Use capability-aware remap for manifest-owned execution graphs or choose a compatible runtime.'
        );
      }
      const graphTransforms = resolved.transforms.filter((_, index) => {
        return !isSessionCapabilityTransform(resolved.names[index]);
      });
      const sessionTransformNames = resolved.names.filter(isSessionCapabilityTransform);
      if (graphTransforms.length > 0) {
        const composed = composeTransforms(...graphTransforms);
        const transformed = composed(execution, {
          capabilities,
          platform: platform ?? { id: 'unknown', vendor: 'unknown', architecture: 'unknown' },
          activationDtype,
          mathDtype,
          accumDtype,
          kvDtype,
          modelId,
          layerTypes,
        });
        if (transformed !== execution) {
          execution = transformed;
          graphWasTransformed = true;
          appliedTransformNames.push(...resolved.names.filter((name) => !isSessionCapabilityTransform(name)));
        }
      }
      if (sessionTransformNames.length > 0) {
        appliedTransformNames.push(...sessionTransformNames);
      }
      if (appliedTransformNames.length > 0) {
        log.info(
          'ExecutionV1',
          `Capability transforms applied: [${appliedTransformNames.join(', ')}] (${resolved.reason})`
        );
      }
    } else {
      log.debug('ExecutionV1', `No capability transforms needed (${resolved.reason})`);
    }

  }

  assertHybridLinearProjectionIsolation(execution, layerTypes, modelId, weightDtype);

  // Transformed graphs have null digests for derived kernels — skip digest
  // validation since the original graph was already validated above.
  const expandOptions = graphWasTransformed ? { skipDigestValidation: true } : {};
  const resolvedSteps = expandV1ToResolvedSteps(execution, expandOptions);

  const prefillSteps = resolvedSteps.filter((s) => s.phase === 'prefill' || s.phase === 'both');
  const decodeSteps = resolvedSteps.filter((s) => s.phase === 'decode' || s.phase === 'both');

  // When widenToF32Activations was applied, the graph's kernels now expect f32
  // activations. The resolved session must reflect this for kernel path building
  // and the runtime session patch. When the GPU has no f16 at all (full f32),
  // KV cache and all compute dtypes must also be f32.
  const activationWidened = appliedTransformNames.includes('widenToF32Activations');
  const fullF32 = activationWidened && capabilities?.hasF16 === false;
  const useSelectiveF16Primary =
    (
      appliedTransformNames.includes('useQwenF16PrimaryMatmuls')
      || appliedTransformNames.includes('useGemma4Int4PleSelectiveF16Decode')
    )
    && !appliedTransformNames.includes('narrowToF16Activations');
  const retainQ4KMaterializationDisabled =
    appliedTransformNames.includes('disableRetainQ4KMaterialization');
  let effectiveSession = session;
  if (activationWidened || useSelectiveF16Primary || retainQ4KMaterializationDisabled) {
    const f32Defaults = fullF32
      ? { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' }
      : useSelectiveF16Primary
        ? { activationDtype: 'f32', outputDtype: 'f32' }
        : { activationDtype: 'f32' };
    effectiveSession = {
      ...session,
      ...(retainQ4KMaterializationDisabled ? { retainQ4KMaterialization: false } : {}),
      ...(activationWidened || useSelectiveF16Primary ? {
        compute: {
          ...session.compute,
          defaults: {
            ...session.compute.defaults,
            ...f32Defaults,
          },
        },
      } : {}),
      ...(fullF32 && session?.kvcache ? {
        kvcache: {
          ...session.kvcache,
          kvDtype: 'f32',
        },
      } : {}),
    };
  }
  const inlineKernelPathEnabled = execution.inlineKernelPath !== false;
  const finitenessFallback = typeof execution.finitenessFallbackKernelPathId === 'string'
    && execution.finitenessFallbackKernelPathId.length > 0
    ? execution.finitenessFallbackKernelPathId
    : null;
  const kernelPath = inlineKernelPathEnabled
    ? buildInlineKernelPath(
      resolvedSteps,
      effectiveSession,
      modelId,
      numLayers,
      finitenessFallback
    )
    : null;

  const fallbackKernelPath = inlineKernelPathEnabled
    && finitenessPolicy.enabled
    && finitenessPolicy.onTrigger === 'fallback-plan'
    ? buildFinitenessFallbackKernelPath({
      execution,
      effectiveSession,
      modelId,
      numLayers,
      headDim,
      layerTypes,
      capabilities,
      platform,
    })
    : null;
  if (kernelPath && fallbackKernelPath?.id) {
    kernelPath.finitenessFallbackKernelPathId = fallbackKernelPath.id;
  }

  // Lane integrity: capture the pre-transform (manifest+profile-agreed) lane
  // and the post-transform (actually dispatched) lane so receipts can honestly
  // surface a capability-driven widening. The manifest-binding gate ensures
  // the pre-transform session already matches manifest quantizationInfo.compute;
  // any drift here is owned by capability transforms (e.g., widenToF32Activations
  // on hasF16=false GPUs).
  const declaredLane = {
    activationDtype: declaredActivationDtype,
    mathDtype: declaredMathDtype,
    accumDtype: declaredAccumDtype,
    kvDtype: declaredKvDtype,
  };
  const executedLane = {
    activationDtype: effectiveSession.compute?.defaults?.activationDtype ?? null,
    mathDtype: effectiveSession.compute?.defaults?.mathDtype ?? null,
    accumDtype: effectiveSession.compute?.defaults?.accumDtype ?? null,
    kvDtype: effectiveSession.kvcache?.kvDtype ?? effectiveSession.compute?.defaults?.activationDtype ?? null,
  };
  const laneFieldDelta = (
    declaredLane.activationDtype !== executedLane.activationDtype
    || declaredLane.mathDtype !== executedLane.mathDtype
    || declaredLane.accumDtype !== executedLane.accumDtype
    || declaredLane.kvDtype !== executedLane.kvDtype
  );
  const laneIntegrity = {
    declared: declaredLane,
    executed: executedLane,
    status: laneFieldDelta ? 'transformed' : 'matches',
    transforms: [...appliedTransformNames],
    policy: capabilityTransformPolicy,
  };

  const layerPipelineResult = buildLayerPipelineFromExecution(resolvedSteps, {
    logIncompatibleOps: !(kernelPath && inlineKernelPathEnabled),
    ffnDtypeFallback: requireSessionActivationDtype(effectiveSession),
  });
  if (layerPipelineResult?.incompatibleOps && !kernelPath && inlineKernelPathEnabled) {
    throw new Error(
      `[ExecutionV1] execution contains layer ops not compatible with the JS layer pipeline ` +
      `and no inline kernelPath was built. ` +
      `Unsupported ops: ${layerPipelineResult.incompatibleOps.join(', ')}.`
    );
  }
  const layerPipeline = layerPipelineResult?.incompatibleOps ? null : layerPipelineResult;
  const sessionPatch = buildSessionRuntimePatch(effectiveSession, {
    includeDecodeLoop: false,
  });

  return {
    session: effectiveSession,
    policies: execution.policies,
    resolvedSteps: {
      prefill: prefillSteps,
      decode: decodeSteps,
      all: resolvedSteps,
    },
    runtimeInferencePatch: {
      ...sessionPatch,
      ...(kernelPath ? { kernelPath, kernelPathSource: 'execution-v1' } : {}),
      ...(layerPipeline ? { pipeline: layerPipeline } : {}),
    },
    appliedTransforms: appliedTransformNames,
    laneIntegrity,
    fallbackKernelPath,
  };
}


// Patch order contract for applyExecutionV1RuntimeConfig:
//   1. compileExecutionV1 — resolves execution graph, applies capability transforms,
//      builds inline kernel path and layer pipeline from the execution graph.
//   2. mergeRuntimeValues — merges the runtimeInferencePatch into runtimeConfig.inference.
//      This writes kernelPath, kernelPathSource, pipeline, compute, and session
//      into the runtime config. decodeLoop stays manifest-owned
//      until applyModelBatchingRuntimeDefaults in phase 2. If runtime batching was
//      already explicitly configured, manifest decodeLoop is skipped and runtime
//      values take precedence.
//   3. preserveConfiguredKernelPath — restores a non-null runtime kernelPath and
//      its dtype-bearing config after compilation so explicit runtime config
//      remains the highest-precedence path contract.
//
// This function must be called exactly once per model load. Calling it again with
// an already-patched runtimeConfig would double-apply the execution-v1 merge and
// produce incorrect results.
export function applyExecutionV1RuntimeConfig(options = {}) {
  const runtimeConfig = options.runtimeConfig ?? null;
  const runtimeOverrides = options.runtimeOverrides;
  const hasExplicitRuntimeOverrides = runtimeOverrides != null;
  const manifest = options.manifest ?? null;
  if (!runtimeConfig || !manifest?.inference) {
    return { runtimeConfig, executionV1State: null };
  }
  if (!hasExecutionV1(manifest.inference)) {
    return { runtimeConfig, executionV1State: null };
  }

  // Assert that execution-v1 patches have not already been applied.
  if (runtimeConfig.inference?.kernelPathSource === 'execution-v1') {
    throw new Error(
      '[ExecutionV1] applyExecutionV1RuntimeConfig called on a runtimeConfig that already has ' +
      'kernelPathSource="execution-v1". Patches must not be applied twice.'
    );
  }

  const executionV1State = compileExecutionV1({
    manifestInference: manifest.inference,
    modelId: manifest.modelId ?? options.modelId,
    numLayers: options.numLayers ?? manifest.architecture?.numLayers ?? 0,
    headDim: options.headDim ?? manifest.architecture?.headDim ?? null,
    weightDtype: manifest.quantizationInfo?.weights ?? null,
    capabilities: options.capabilities ?? null,
    platform: options.platform ?? null,
    runtimeSession: hasExplicitRuntimeOverrides
      ? resolveRuntimeInferenceOverrideSection(runtimeOverrides, 'session')
      : (runtimeConfig.inference?.session ?? null),
    runtimeCompute: runtimeConfig.inference?.compute ?? null,
    kernelPathPolicy: runtimeConfig.inference?.kernelPathPolicy ?? null,
    executionPatch: runtimeConfig.inference?.executionPatch ?? null,
  });

  const runtimeInferencePatch = executionV1State.runtimeInferencePatch;
  const updatedInference = preserveConfiguredKernelPath(
    preserveRuntimeDecodeLoop(
      mergeRuntimeValues(runtimeConfig.inference ?? {}, runtimeInferencePatch),
      runtimeConfig
    ),
    runtimeConfig
  );

  return {
    runtimeConfig: {
      ...runtimeConfig,
      inference: updatedInference,
    },
    executionV1State,
  };
}
