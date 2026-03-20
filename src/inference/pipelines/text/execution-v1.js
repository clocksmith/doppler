import { expandExecutionV1, EXECUTION_V1_SCHEMA_ID } from '../../../config/schema/index.js';
import {
  buildInlineKernelPath,
  buildLayerPipelineFromExecution,
  buildSessionRuntimePatch,
  PIPELINE_COMPATIBLE_OPS,
} from './execution-runtime-builders.js';
import { mergeRuntimeValues } from '../../../config/runtime-merge.js';

export function hasExecutionV1(manifestInference) {
  return manifestInference?.schema === EXECUTION_V1_SCHEMA_ID
    && manifestInference?.execution
    && typeof manifestInference.execution.kernels === 'object';
}


function expandV1ToResolvedSteps(execution) {
  const expanded = expandExecutionV1(execution);
  return expanded.map((step, index) => ({
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
  }));
}


export function compileExecutionV1(options = {}) {
  const manifestInference = options.manifestInference;
  const modelId = options.modelId ?? 'model';
  const numLayers = options.numLayers ?? 0;

  if (!hasExecutionV1(manifestInference)) {
    throw new Error(`[ExecutionV1] manifest.inference.schema must be "${EXECUTION_V1_SCHEMA_ID}".`);
  }

  const execution = manifestInference.execution;
  const sessionDefaults = manifestInference.sessionDefaults;

  if (!sessionDefaults?.compute?.defaults?.activationDtype) {
    throw new Error('[ExecutionV1] sessionDefaults.compute.defaults.activationDtype is required.');
  }

  const resolvedSteps = expandV1ToResolvedSteps(execution);

  const prefillSteps = resolvedSteps.filter((s) => s.phase === 'prefill' || s.phase === 'both');
  const decodeSteps = resolvedSteps.filter((s) => s.phase === 'decode' || s.phase === 'both');

  const kernelPath = buildInlineKernelPath(
    resolvedSteps,
    sessionDefaults,
    modelId,
    numLayers,
    null
  );

  const layerPipelineResult = buildLayerPipelineFromExecution(resolvedSteps);
  if (layerPipelineResult?.incompatibleOps && !kernelPath) {
    throw new Error(
      `[ExecutionV1] execution contains layer ops not compatible with the JS layer pipeline ` +
      `and no inline kernelPath was built. ` +
      `Unsupported ops: ${layerPipelineResult.incompatibleOps.join(', ')}.`
    );
  }
  const layerPipeline = layerPipelineResult?.incompatibleOps ? null : layerPipelineResult;
  const sessionPatch = buildSessionRuntimePatch(sessionDefaults);

  return {
    sessionDefaults,
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
  };
}


export function applyExecutionV1RuntimeConfig(options = {}) {
  const runtimeConfig = options.runtimeConfig ?? null;
  const manifest = options.manifest ?? null;
  if (!runtimeConfig || !manifest?.inference) {
    return { runtimeConfig, executionV1State: null };
  }
  if (!hasExecutionV1(manifest.inference)) {
    return { runtimeConfig, executionV1State: null };
  }

  const executionV1State = compileExecutionV1({
    manifestInference: manifest.inference,
    modelId: manifest.modelId ?? options.modelId,
    numLayers: options.numLayers ?? manifest.architecture?.numLayers ?? 0,
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
