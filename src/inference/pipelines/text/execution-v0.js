import { mergeRuntimeValues } from '../../../config/runtime-merge.js';
import { buildExecutionV0FromKernelPath } from '../../../converter/execution-v0-manifest.js';
import {
  DEFAULT_EXECUTION_V0_COMPUTE_DEFAULTS,
  DEFAULT_EXECUTION_V0_POLICIES,
  DEFAULT_EXECUTION_V0_SESSION_DEFAULTS,
} from '../../../config/schema/execution-v0.schema.js';
import {
  applyExecutionPatchAtomic,
  assertExecutionRuntimeOverlay,
  assertExecutionV0Schema,
  assertKVLayoutExecutionCompatibility,
  collectLeafPaths,
  createInitialSlotDtypes,
  createSourceTrace,
  hasDefinedPath,
  indexKernelProfiles,
  indexRuntimePatchMeta,
  normalizeRuntimeSessionForExecutionV0,
  resolvePhaseSteps,
  setSourceTrace,
  validateManifestSessionDefaultsContract,
  validatePhaseBoundaryCompatibility,
  validateStepShape,
  validateUniqueStepIds,
  cloneJson,
} from './execution-v0-contract-helpers.js';
import {
  buildInlineKernelPath,
  buildLayerPipelineFromExecution,
  buildModelRuntimeOverrides,
  buildSessionRuntimePatch,
  resolveFinitenessFallbackKernelPathId,
} from './execution-v0-runtime-builders.js';

export function hasExecutionV0(manifestInference) {
  return !!manifestInference?.execution && Array.isArray(manifestInference.execution.steps);
}

export function compileExecutionV0(options = {}) {
  const manifestInference = options.manifestInference ?? null;
  if (!hasExecutionV0(manifestInference)) {
    return null;
  }
  assertExecutionV0Schema(manifestInference);
  validateManifestSessionDefaultsContract(manifestInference);

  const modelId = options.modelId ?? 'model';
  const numLayers = Number.isInteger(options.numLayers) ? options.numLayers : 0;
  const runtimeInference = options.runtimeInference ?? {};
  assertExecutionRuntimeOverlay(runtimeInference);
  const policies = {
    ...DEFAULT_EXECUTION_V0_POLICIES,
    ...(manifestInference.execution?.policies ?? {}),
  };
  const normalizedRuntimeSession = normalizeRuntimeSessionForExecutionV0(
    runtimeInference.session ?? {},
    manifestInference,
    DEFAULT_EXECUTION_V0_COMPUTE_DEFAULTS
  );
  const sessionDefaults = mergeRuntimeValues(
    DEFAULT_EXECUTION_V0_SESSION_DEFAULTS,
    manifestInference.sessionDefaults ?? {}
  );
  const resolvedSession = mergeRuntimeValues(
    sessionDefaults,
    normalizedRuntimeSession ?? {}
  );

  const baseSteps = cloneJson(manifestInference.execution.steps ?? []);
  baseSteps.forEach(validateStepShape);
  validateUniqueStepIds(baseSteps);

  const patchedSteps = applyExecutionPatchAtomic(baseSteps, runtimeInference.executionPatch ?? null);
  patchedSteps.forEach(validateStepShape);
  validateUniqueStepIds(patchedSteps);
  assertKVLayoutExecutionCompatibility(patchedSteps, resolvedSession);
  const runtimePatchMeta = indexRuntimePatchMeta(runtimeInference.executionPatch ?? null);

  const manifestSessionDefaults = manifestInference.sessionDefaults ?? {};
  const sessionSourceByPath = new Map();
  for (const pathSegments of collectLeafPaths(resolvedSession)) {
    const path = pathSegments.join('.');
    const source = hasDefinedPath(runtimeInference.session ?? {}, pathSegments)
      ? 'runtime.session'
      : hasDefinedPath(manifestSessionDefaults, pathSegments)
        ? 'manifest'
        : 'derived';
    sessionSourceByPath.set(path, source);
  }

  const profileIndex = indexKernelProfiles(resolvedSession);
  const sourceTrace = createSourceTrace();
  const sessionDefaultSources = {
    mathDtype: sessionSourceByPath.get('compute.defaults.mathDtype') ?? 'derived',
    accumDtype: sessionSourceByPath.get('compute.defaults.accumDtype') ?? 'derived',
    outputDtype: sessionSourceByPath.get('compute.defaults.outputDtype') ?? 'derived',
    kvDtype: sessionSourceByPath.get('kvcache.kvDtype') ?? 'derived',
  };
  const resolvedPrefill = resolvePhaseSteps(
    'prefill',
    patchedSteps,
    resolvedSession,
    profileIndex,
    policies,
    {
      initialSlotDtypes: createInitialSlotDtypes(resolvedSession),
      sourceTrace,
      sessionDefaultSources,
      runtimePatchMeta,
    }
  );
  const decodeInitialSlotDtypes = createInitialSlotDtypes(resolvedSession);
  validatePhaseBoundaryCompatibility({
    steps: patchedSteps,
    prefillFinalSlotDtypes: resolvedPrefill.finalSlotDtypes,
    decodeInitialSlotDtypes,
    sessionDefaults: resolvedSession,
    profileIndex,
    policies,
  });
  const resolvedDecode = resolvePhaseSteps(
    'decode',
    patchedSteps,
    resolvedSession,
    profileIndex,
    policies,
    {
      initialSlotDtypes: decodeInitialSlotDtypes,
      sourceTrace,
      sessionDefaultSources,
      runtimePatchMeta,
    }
  );
  const resolvedPrefillSteps = resolvedPrefill.steps;
  const resolvedDecodeSteps = resolvedDecode.steps;
  const resolvedSteps = [
    ...resolvedPrefillSteps,
    ...resolvedDecodeSteps.filter((step) => step.phase === 'decode'),
  ];

  const defaultKernelPathId = typeof manifestInference.defaultKernelPath === 'string'
    && manifestInference.defaultKernelPath.trim().length > 0
    ? manifestInference.defaultKernelPath.trim()
    : null;
  const finitenessFallbackKernelPathId = resolveFinitenessFallbackKernelPathId(defaultKernelPathId);

  const kernelPath = buildInlineKernelPath(
    patchedSteps,
    resolvedSession,
    modelId,
    numLayers,
    finitenessFallbackKernelPathId
  );
  const layerPipeline = buildLayerPipelineFromExecution(resolvedSteps);
  const sessionPatch = buildSessionRuntimePatch(resolvedSession);
  const modelOverrides = buildModelRuntimeOverrides(manifestInference);
  for (const [path, source] of sessionSourceByPath.entries()) {
    setSourceTrace(sourceTrace.session, `sessionDefaults.${path}`, source);
  }

  return {
    sessionDefaults: resolvedSession,
    policies,
    resolvedSteps: {
      prefill: resolvedPrefillSteps,
      decode: resolvedDecodeSteps,
      all: resolvedSteps,
    },
    runtimeInferencePatch: {
      ...sessionPatch,
      ...(kernelPath ? { kernelPath, kernelPathSource: 'execution-v0' } : {}),
      ...(layerPipeline ? { pipeline: layerPipeline } : {}),
      ...(modelOverrides ? { modelOverrides } : {}),
    },
    resolvedSources: sourceTrace,
  };
}

export function applyExecutionV0RuntimeConfig(options = {}) {
  const runtimeConfig = options.runtimeConfig ?? null;
  const manifest = options.manifest ?? null;
  if (!runtimeConfig || !manifest?.inference) {
    return { runtimeConfig, executionV0State: null };
  }
  if (!hasExecutionV0(manifest.inference)) {
    return { runtimeConfig, executionV0State: null };
  }

  const runtimeInference = runtimeConfig.inference ?? {};
  const kernelPathExecution = runtimeInference.kernelPath !== undefined
    ? buildExecutionV0FromKernelPath(runtimeInference.kernelPath)
    : null;
  const manifestInference = kernelPathExecution
    ? {
      ...manifest.inference,
      ...kernelPathExecution,
      defaultKernelPath: runtimeInference.kernelPath,
    }
    : manifest.inference;
  const runtimeExecutionOverlay = {
    ...(runtimeInference.session ? { session: runtimeInference.session } : {}),
    ...(runtimeInference.executionPatch ? { executionPatch: runtimeInference.executionPatch } : {}),
  };

  const executionV0State = compileExecutionV0({
    manifestInference,
    runtimeInference: runtimeExecutionOverlay,
    modelId: options.modelId ?? manifest.modelId ?? 'model',
    numLayers: Number.isInteger(options.numLayers)
      ? options.numLayers
      : Number(manifest.architecture?.numLayers ?? 0),
  });
  if (!executionV0State) {
    return { runtimeConfig, executionV0State: null };
  }

  const compiledKernelPathSource = runtimeInference.kernelPath !== undefined
    ? 'config'
    : 'manifest';
  const runtimeInferencePatch = { ...executionV0State.runtimeInferencePatch };
  if (runtimeInferencePatch.kernelPathSource) {
    runtimeInferencePatch.kernelPathSource = compiledKernelPathSource;
  }
  if (runtimeInference.kernelPath !== undefined) {
    delete runtimeInferencePatch.kernelPath;
    delete runtimeInferencePatch.kernelPathSource;
  }
  if (runtimeInferencePatch.modelOverrides) {
    runtimeInferencePatch.modelOverrides = mergeRuntimeValues(
      runtimeInferencePatch.modelOverrides,
      runtimeInference.modelOverrides ?? {}
    );
  }
  if (runtimeInference.kernelPath !== undefined && runtimeInference.compute) {
    runtimeInferencePatch.compute = mergeRuntimeValues(
      runtimeInferencePatch.compute ?? {},
      runtimeInference.compute
    );
  }
  if (runtimeInference.kernelPath !== undefined && runtimeInference.kvcache) {
    runtimeInferencePatch.kvcache = mergeRuntimeValues(
      runtimeInferencePatch.kvcache ?? {},
      runtimeInference.kvcache
    );
  }

  return {
    runtimeConfig: {
      ...runtimeConfig,
      inference: mergeRuntimeValues(runtimeConfig.inference, runtimeInferencePatch),
    },
    executionV0State,
  };
}
