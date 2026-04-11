import { expandExecutionV1, EXECUTION_V1_SCHEMA_ID } from '../../../config/schema/index.js';
import {
  buildInlineKernelPath,
  buildLayerPipelineFromExecution,
  buildSessionRuntimePatch,
  PIPELINE_COMPATIBLE_OPS,
  normalizeDtype,
  requireSessionActivationDtype,
} from './execution-runtime-builders.js';
import { mergeKernelPathPolicy } from '../../../config/merge-helpers.js';
import { mergeRuntimeValues } from '../../../config/runtime-merge.js';
import { buildOpIdFromExecutionStep } from './operator-identity.js';
import {
  resolveCapabilityTransforms,
  resolveFinitenessFallbackTransform,
} from '../../../config/transforms/capability-transform-resolver.js';
import { composeTransforms } from '../../../config/transforms/execution-graph-transforms.js';
import { resolveRangeAwareSelectiveWideningConfig } from './finiteness-policy.js';
import { log } from '../../../debug/index.js';

export function hasExecutionV1(manifestInference) {
  return manifestInference?.schema === EXECUTION_V1_SCHEMA_ID
    && manifestInference?.execution
    && typeof manifestInference.execution.kernels === 'object';
}

function mergeExecutionV1Session(manifestSession, runtimeSession) {
  return mergeRuntimeValues(
    manifestSession ?? {},
    runtimeSession ?? {}
  );
}

function applyExecutionV1RuntimeComputeOverrides(session, runtimeSession, runtimeCompute) {
  const activationOverrideRaw = runtimeCompute?.activationDtype;
  if (activationOverrideRaw == null) {
    return session;
  }

  const activationOverride = normalizeDtype(
    activationOverrideRaw,
    'runtime.inference.compute.activationDtype'
  );
  const runtimeSessionActivation = runtimeSession?.compute?.defaults?.activationDtype;
  if (
    runtimeSessionActivation != null
    && normalizeDtype(
      runtimeSessionActivation,
      'runtime.inference.session.compute.defaults.activationDtype'
    ) !== activationOverride
  ) {
    throw new Error(
      '[ExecutionV1] runtime.inference.compute.activationDtype conflicts with ' +
      'runtime.inference.session.compute.defaults.activationDtype. ' +
      'Set both to the same dtype or remove one override.'
    );
  }

  const runtimeSessionOutput = runtimeSession?.compute?.defaults?.outputDtype;
  if (
    runtimeSessionOutput != null
    && normalizeDtype(
      runtimeSessionOutput,
      'runtime.inference.session.compute.defaults.outputDtype'
    ) !== activationOverride
  ) {
    throw new Error(
      '[ExecutionV1] runtime.inference.compute.activationDtype conflicts with ' +
      'runtime.inference.session.compute.defaults.outputDtype. ' +
      'Set both to the same dtype or remove one override.'
    );
  }

  return {
    ...session,
    compute: {
      ...(session?.compute ?? {}),
      defaults: {
        ...(session?.compute?.defaults ?? {}),
        activationDtype: activationOverride,
        outputDtype: activationOverride,
      },
    },
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

function assertHybridLinearProjectionIsolation(execution, layerTypes, modelId) {
  if (!Array.isArray(layerTypes) || layerTypes.length === 0) {
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
      src: 'state',
      dst: 'state',
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


export function compileExecutionV1(options = {}) {
  const manifestInference = options.manifestInference;
  const modelId = options.modelId ?? 'model';
  const numLayers = options.numLayers ?? 0;
  const headDim = Number.isFinite(options.headDim) ? Math.floor(options.headDim) : null;
  const capabilities = options.capabilities ?? null;
  const platform = options.platform ?? null;
  const runtimeSession = options.runtimeSession ?? null;
  const runtimeCompute = options.runtimeCompute ?? null;
  const kernelPathPolicy = mergeKernelPathPolicy(undefined, options.kernelPathPolicy ?? undefined);

  if (!hasExecutionV1(manifestInference)) {
    throw new Error(`[ExecutionV1] manifest.inference.schema must be "${EXECUTION_V1_SCHEMA_ID}".`);
  }

  let execution = manifestInference.execution;
  const mergedSession = mergeExecutionV1Session(
    manifestInference.session ?? {},
    runtimeSession ?? {}
  );
  const session = applyExecutionV1RuntimeComputeOverrides(
    mergedSession,
    runtimeSession,
    runtimeCompute
  );

  if (!session?.compute?.defaults?.activationDtype) {
    throw new Error('[ExecutionV1] session.compute.defaults.activationDtype is required.');
  }

  const activationDtype = session.compute.defaults.activationDtype;
  const kvDtype = session?.kvcache?.kvDtype ?? activationDtype;
  const layerTypes = manifestInference?.layerPattern?.layerTypes ?? null;
  const finitenessPolicy = resolveRangeAwareSelectiveWideningConfig(runtimeCompute);

  // Validate the original manifest graph (full digest checks).
  expandExecutionV1(execution);

  // -------------------------------------------------------------------------
  // Phase 2: Apply capability transforms to the execution graph
  // -------------------------------------------------------------------------
  let appliedTransformNames = [];
  let fallbackExecution = null;
  let fallbackTransformResult = null;
  let graphWasTransformed = false;

  if (capabilities) {
    const graphContext = {
      activationDtype,
      kvDtype,
      headDim,
      modelId,
      layerTypes,
      ...summarizeExecutionGraphContext(execution),
    };
    const resolved = resolveCapabilityTransforms(capabilities, platform, graphContext);
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
      const composed = composeTransforms(...resolved.transforms);
      const transformed = composed(execution, {
        capabilities,
        platform: platform ?? { id: 'unknown', vendor: 'unknown', architecture: 'unknown' },
        activationDtype,
        kvDtype,
        modelId,
        layerTypes,
      });
      if (transformed !== execution) {
        execution = transformed;
        graphWasTransformed = true;
        appliedTransformNames = resolved.names;
        log.info(
          'ExecutionV1',
          `Capability transforms applied: [${resolved.names.join(', ')}] (${resolved.reason})`
        );
      }
    } else {
      log.debug('ExecutionV1', `No capability transforms needed (${resolved.reason})`);
    }

    // Build explicit finiteness fallback only when the runtime opted into
    // alternate-plan recovery. The default policy is fail-fast.
    fallbackTransformResult = finitenessPolicy.onTrigger === 'fallback-plan'
      ? resolveFinitenessFallbackTransform(graphContext)
      : null;
    if (fallbackTransformResult) {
      const fallbackGraph = fallbackTransformResult.transform(execution, {
        capabilities,
        platform: platform ?? { id: 'unknown', vendor: 'unknown', architecture: 'unknown' },
        activationDtype,
        kvDtype,
        headDim,
        modelId,
      });
      if (fallbackGraph) {
        fallbackExecution = fallbackGraph;
        log.info(
          'ExecutionV1',
          `Finiteness fallback transform resolved: ${fallbackTransformResult.name}`
        );
      }
    }
  }

  assertHybridLinearProjectionIsolation(execution, layerTypes, modelId);

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
  const useQwenSelectiveF16Primary =
    appliedTransformNames.includes('useQwenF16PrimaryMatmuls')
    && !appliedTransformNames.includes('narrowToF16Activations');
  let effectiveSession = session;
  if (activationWidened || useQwenSelectiveF16Primary) {
    const f32Defaults = fullF32
      ? { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' }
      : useQwenSelectiveF16Primary
        ? { activationDtype: 'f32', outputDtype: 'f32' }
        : { activationDtype: 'f32' };
    effectiveSession = {
      ...session,
      compute: {
        ...session.compute,
        defaults: {
          ...session.compute.defaults,
          ...f32Defaults,
        },
      },
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

  // Build fallback inline kernel path from the fallback execution graph
  let fallbackKernelPath = null;
  if (fallbackExecution && inlineKernelPathEnabled) {
    const fallbackSteps = expandV1ToResolvedSteps(fallbackExecution, { skipDigestValidation: true });
    const fallbackKvDtype = fallbackTransformResult?.fallbackKvDtype ?? 'f32';
    const fallbackSession = {
      ...session,
      compute: {
        ...session.compute,
        defaults: {
          ...session.compute.defaults,
          activationDtype: 'f32',
          mathDtype: 'f32',
          accumDtype: 'f32',
          outputDtype: 'f32',
        },
      },
      kvcache: {
        ...(session.kvcache ?? {}),
        kvDtype: fallbackKvDtype,
      },
    };
    fallbackKernelPath = buildInlineKernelPath(
      fallbackSteps,
      fallbackSession,
      modelId,
      numLayers,
      null
    );
    if (fallbackKernelPath) {
      fallbackKernelPath = {
        ...fallbackKernelPath,
        id: `${fallbackKernelPath.id}-finiteness-fallback`,
        name: 'Execution inline kernel path (finiteness fallback)',
      };
      log.info(
        'ExecutionV1',
        `Finiteness fallback inline kernel path built (${fallbackKernelPath.id})`
      );
    }
  }

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
//
// This function must be called exactly once per model load. Calling it again with
// an already-patched runtimeConfig would double-apply the execution-v1 merge and
// produce incorrect results.
export function applyExecutionV1RuntimeConfig(options = {}) {
  const runtimeConfig = options.runtimeConfig ?? null;
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
    capabilities: options.capabilities ?? null,
    platform: options.platform ?? null,
    runtimeSession: runtimeConfig.inference?.session ?? null,
    runtimeCompute: runtimeConfig.inference?.compute ?? null,
    kernelPathPolicy: runtimeConfig.inference?.kernelPathPolicy ?? null,
  });

  const runtimeInferencePatch = executionV1State.runtimeInferencePatch;
  const updatedInference = mergeRuntimeValues(runtimeConfig.inference ?? {}, runtimeInferencePatch);

  return {
    runtimeConfig: {
      ...runtimeConfig,
      inference: updatedInference,
    },
    executionV1State,
  };
}
