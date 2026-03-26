import { expandExecutionV1, EXECUTION_V1_SCHEMA_ID } from '../../../config/schema/index.js';
import {
  buildInlineKernelPath,
  buildLayerPipelineFromExecution,
  buildSessionRuntimePatch,
  PIPELINE_COMPATIBLE_OPS,
} from './execution-runtime-builders.js';
import { mergeRuntimeValues } from '../../../config/runtime-merge.js';
import { buildOpIdFromExecutionStep } from './operator-identity.js';
import {
  resolveCapabilityTransforms,
  resolveFinitenessFallbackTransform,
} from '../../../config/transforms/capability-transform-resolver.js';
import { composeTransforms } from '../../../config/transforms/execution-graph-transforms.js';
import { log } from '../../../debug/index.js';

export function hasExecutionV1(manifestInference) {
  return manifestInference?.schema === EXECUTION_V1_SCHEMA_ID
    && manifestInference?.execution
    && typeof manifestInference.execution.kernels === 'object';
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
  const capabilities = options.capabilities ?? null;
  const platform = options.platform ?? null;
  const runtimeSession = options.runtimeSession ?? null;

  if (!hasExecutionV1(manifestInference)) {
    throw new Error(`[ExecutionV1] manifest.inference.schema must be "${EXECUTION_V1_SCHEMA_ID}".`);
  }

  let execution = manifestInference.execution;
  const session = mergeRuntimeValues(
    manifestInference.session ?? {},
    runtimeSession ?? {}
  );

  if (!session?.compute?.defaults?.activationDtype) {
    throw new Error('[ExecutionV1] session.compute.defaults.activationDtype is required.');
  }

  const activationDtype = session.compute.defaults.activationDtype;
  const kvDtype = session?.kvcache?.kvDtype ?? activationDtype;

  // Validate the original manifest graph (full digest checks).
  expandExecutionV1(execution);

  // -------------------------------------------------------------------------
  // Phase 2: Apply capability transforms to the execution graph
  // -------------------------------------------------------------------------
  let appliedTransformNames = [];
  let fallbackExecution = null;
  let graphWasTransformed = false;

  if (capabilities) {
    const graphContext = { activationDtype, kvDtype };
    const resolved = resolveCapabilityTransforms(capabilities, platform, graphContext);

    if (resolved.transforms.length > 0) {
      const composed = composeTransforms(...resolved.transforms);
      const transformed = composed(execution, {
        capabilities,
        platform: platform ?? { id: 'unknown', vendor: 'unknown', architecture: 'unknown' },
        activationDtype,
        kvDtype,
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

    // Build finiteness fallback from the (possibly transformed) graph
    const fallbackResult = resolveFinitenessFallbackTransform(graphContext);
    if (fallbackResult) {
      const fallbackGraph = fallbackResult.transform(execution, {
        capabilities,
        platform: platform ?? { id: 'unknown', vendor: 'unknown', architecture: 'unknown' },
        activationDtype,
        kvDtype,
      });
      if (fallbackGraph) {
        fallbackExecution = fallbackGraph;
        log.info(
          'ExecutionV1',
          `Finiteness fallback transform resolved: ${fallbackResult.name}`
        );
      }
    }
  }

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
  let effectiveSession = session;
  if (activationWidened) {
    const f32Defaults = fullF32
      ? { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' }
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
    const fallbackSession = {
      ...session,
      compute: {
        ...session.compute,
        defaults: {
          ...session.compute.defaults,
          activationDtype: 'f32',
        },
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
      log.info(
        'ExecutionV1',
        `Finiteness fallback inline kernel path built (${fallbackKernelPath.id})`
      );
    }
  }

  const layerPipelineResult = buildLayerPipelineFromExecution(resolvedSteps);
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
    capabilities: options.capabilities ?? null,
    platform: options.platform ?? null,
    runtimeSession: runtimeConfig.inference?.session ?? null,
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
