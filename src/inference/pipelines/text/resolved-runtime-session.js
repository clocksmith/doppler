import { hashRefactorReceiptValue } from '../../refactor-receipt.js';

export const RESOLVED_RUNTIME_SESSION_SCHEMA = 'doppler.resolved-runtime-session/v1';

function cloneJsonRecord(value, label) {
  if (value == null) {
    return null;
  }
  let encoded;
  try {
    encoded = JSON.stringify(value);
  } catch (error) {
    throw new Error(
      `[ResolvedRuntimeSession] ${label} must be JSON-safe: ${error?.message ?? String(error)}`
    );
  }
  if (encoded === undefined) {
    throw new Error(`[ResolvedRuntimeSession] ${label} must be JSON-safe.`);
  }
  return JSON.parse(encoded);
}

function deepFreeze(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) {
    return value;
  }
  Object.freeze(value);
  for (const entry of Object.values(value)) {
    deepFreeze(entry);
  }
  return value;
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`[ResolvedRuntimeSession] ${label} must be a positive integer.`);
  }
  return value;
}

function requireModelId(manifest) {
  const modelId = String(manifest?.modelId ?? '').trim();
  if (!modelId) {
    throw new Error('[ResolvedRuntimeSession] manifest.modelId is required.');
  }
  return modelId;
}

function summarizeExecutionPlan(plan) {
  if (!plan) {
    return null;
  }
  return {
    id: plan.id,
    source: plan.source,
    activationDtype: plan.activationDtype,
    kernelPathId: plan.kernelPathId ?? null,
    kernelPathSource: plan.kernelPathSource ?? 'none',
    kernelPathHash: plan.kernelPath
      ? hashRefactorReceiptValue(cloneJsonRecord(plan.kernelPath, 'executionPlan.kernelPath'))
      : null,
    finitenessGuardEnabled: plan.finitenessGuardEnabled === true,
    finitenessOnTrigger: plan.finitenessOnTrigger ?? null,
    defaultDisableCommandBatching: plan.defaultDisableCommandBatching === true,
    defaultDisableMultiTokenDecode: plan.defaultDisableMultiTokenDecode === true,
    defaultBatchSize: plan.defaultBatchSize,
    defaultStopCheckMode: plan.defaultStopCheckMode,
    readbackInterval: plan.readbackInterval ?? null,
    readbackMode: plan.readbackMode ?? null,
    maxBatchDecodeTokens: plan.maxBatchDecodeTokens ?? null,
  };
}

export function createResolvedRuntimeSession(options = {}) {
  const {
    manifest,
    modelConfig,
    runtimeConfig,
    resolvedKernelPath = null,
    kernelPathSource = 'none',
    executionV1State = null,
    executionPlanState,
  } = options;
  if (!runtimeConfig?.inference?.session) {
    throw new Error('[ResolvedRuntimeSession] runtimeConfig.inference.session is required.');
  }
  if (!runtimeConfig.inference.compute) {
    throw new Error('[ResolvedRuntimeSession] runtimeConfig.inference.compute is required.');
  }
  if (!executionPlanState?.primaryPlan) {
    throw new Error('[ResolvedRuntimeSession] a compiled primary execution plan is required.');
  }

  const session = cloneJsonRecord(runtimeConfig.inference.session, 'runtime session');
  const compute = cloneJsonRecord(runtimeConfig.inference.compute, 'compute policy');
  const manifestInference = cloneJsonRecord(manifest?.inference ?? null, 'manifest inference contract');
  const kernelPath = cloneJsonRecord(resolvedKernelPath, 'resolved kernel path');
  const resolvedSteps = cloneJsonRecord(
    executionV1State?.resolvedSteps ?? null,
    'execution-v1 resolved steps'
  );
  const laneIntegrity = cloneJsonRecord(
    executionV1State?.laneIntegrity ?? null,
    'execution-v1 lane integrity'
  );

  const core = {
    model: {
      id: requireModelId(manifest),
      type: String(manifest?.modelType ?? ''),
      architecture: typeof manifest?.architecture === 'string'
        ? manifest.architecture
        : String(manifest?.architecture?.type ?? manifest?.modelType ?? ''),
      numLayers: requirePositiveInteger(modelConfig?.numLayers, 'modelConfig.numLayers'),
      hiddenSize: requirePositiveInteger(modelConfig?.hiddenSize, 'modelConfig.hiddenSize'),
      numHeads: requirePositiveInteger(modelConfig?.numHeads, 'modelConfig.numHeads'),
      numKVHeads: requirePositiveInteger(modelConfig?.numKVHeads, 'modelConfig.numKVHeads'),
      headDim: requirePositiveInteger(modelConfig?.headDim, 'modelConfig.headDim'),
    },
    manifestInference,
    runtime: {
      session,
      compute,
    },
    kernelPath: {
      id: kernelPath?.id ?? null,
      source: kernelPathSource,
      hash: kernelPath ? hashRefactorReceiptValue(kernelPath) : null,
      definition: kernelPath,
    },
    capabilityPolicy: laneIntegrity?.policy ?? null,
    laneIntegrity,
    execution: {
      primary: summarizeExecutionPlan(executionPlanState.primaryPlan),
      fallback: summarizeExecutionPlan(executionPlanState.fallbackPlan),
      resolvedSteps,
      resolvedStepsHash: resolvedSteps ? hashRefactorReceiptValue(resolvedSteps) : null,
      appliedTransforms: cloneJsonRecord(
        executionV1State?.appliedTransforms ?? [],
        'execution-v1 applied transforms'
      ),
    },
    dtypes: {
      activation: executionPlanState.primaryPlan.activationDtype,
      output: session?.compute?.defaults?.outputDtype
        ?? compute?.outputDtype
        ?? executionPlanState.primaryPlan.activationDtype,
      kv: session?.kvcache?.kvDtype
        ?? session?.compute?.defaults?.activationDtype
        ?? executionPlanState.primaryPlan.activationDtype,
      math: session?.compute?.defaults?.mathDtype ?? null,
      accumulation: session?.compute?.defaults?.accumDtype ?? null,
    },
  };
  const resolved = {
    schema: RESOLVED_RUNTIME_SESSION_SCHEMA,
    id: hashRefactorReceiptValue(core),
    ...core,
  };
  return deepFreeze(resolved);
}

export function resolveAttentionRuntimeSession(state) {
  const resolved = state?.resolvedRuntimeSession;
  if (!resolved || resolved.schema !== RESOLVED_RUNTIME_SESSION_SCHEMA) {
    throw new Error(
      '[ResolvedRuntimeSession] attention requires a resolved runtime session. ' +
      'Construct the pipeline session before executing layers.'
    );
  }
  if (!resolved.runtime?.session) {
    throw new Error('[ResolvedRuntimeSession] resolved runtime session is missing runtime.session.');
  }
  return resolved.runtime.session;
}
