import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { hashRefactorReceiptValue } from '../../../refactor-receipt.js';

export const ATTENTION_PLAN_SCHEMA = 'doppler.attention-plan/v1';

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
    throw new Error(`[AttentionPlan] ${label} must be a positive integer.`);
  }
  return value;
}

function requireDtype(value, label) {
  if (value !== 'f16' && value !== 'f32') {
    throw new Error(`[AttentionPlan] ${label} must be "f16" or "f32".`);
  }
  return value;
}

function requirePositiveFinite(value, label, fallback) {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved <= 0) {
    throw new Error(`[AttentionPlan] ${label} must be a positive finite number.`);
  }
  return resolved;
}

function resolveAttentionImplementation(input) {
  return selectRuleValue('inference', 'attention', 'attentionKernelVariant', {
    kvLayout: input.kv.layout,
    numTokens: input.geometry.numTokens,
    coldQuantMode: input.kv.coldQuantMode,
  });
}

function resolveOutputResidualMode(input) {
  const hasResidual = input.outputProjection.hasResidual === true;
  const residualMatches = input.outputProjection.residualDtype === input.dtypes.outputProjectionInput;
  const standardFused = selectRuleValue('inference', 'attention', 'useFusedOProjResidual', {
    allowFusedResidual: input.geometry.numTokens === 1,
    hasResidual,
    residualMatches,
    attnIsF32: input.dtypes.outputProjectionInput === 'f32',
    attnIsF16: input.dtypes.outputProjectionInput === 'f16',
    hasLoRA: input.outputProjection.hasLoRA === true,
    oProjIsF16: input.outputProjection.weightDtype === 'f16',
  });
  if (standardFused) {
    return 'matmul-residual';
  }
  const wideTileFused = hasResidual
    && residualMatches
    && input.outputProjection.weightDtype === 'q4k'
    && input.dtypes.outputProjectionInput === 'f32'
    && input.dtypes.output === 'f32'
    && input.outputProjection.hasLoRA !== true
    && input.capabilities.hasF16 === true
    && input.capabilities.hasSubgroups === true
    && input.session.useWideTileResidualFusion === true
    && input.session.retainQ4KMaterialization === true
    && (
      input.phase === 'decode'
        ? input.session.useWideTileQ4KDecode === true
        : input.session.useWideTileQ4KPrefill === true
    );
  return wideTileFused ? 'wide-tile-epilogue' : 'separate';
}

function resolveOutputGateMode(input) {
  if (input.outputGate.enabled !== true) {
    return 'none';
  }
  const kernelPathSupportsFusion = input.outputGate.kernelPathSupportsFusion === true;
  const fused = input.phase === 'decode'
    && input.geometry.numTokens === 1
    && input.geometry.headDim === 256
    && input.dtypes.cachedK === 'f16'
    && input.dtypes.cachedV === 'f16'
    && input.outputGate.tensorDtype === 'f32'
    && input.outputGate.tensorElementCount >= (
      input.geometry.numTokens * input.geometry.numHeads * input.geometry.headDim
    )
    && input.session.attentionDecodeOnline?.useOutputGateFusion === true
    && kernelPathSupportsFusion
    && input.diffusionGemmaDecoder !== true;
  return fused ? 'attention-epilogue' : 'sigmoid-post-attention';
}

export function kernelPathSupportsOutputGateFusion(kernelPath) {
  const steps = kernelPath?.decode?.steps;
  if (!Array.isArray(steps)) {
    return false;
  }
  return steps.some((step) => (
    step?.op === 'attention'
    && step?.kernel === 'attention_decode_online_head256_f16kv_output_gate.wgsl'
    && (step?.entry === undefined || step.entry === 'main')
  ));
}

export function resolveAttentionPlan(input = {}) {
  if (input.phase !== 'prefill' && input.phase !== 'decode') {
    throw new Error('[AttentionPlan] phase must be "prefill" or "decode".');
  }
  const geometry = {
    layerIdx: Number.isInteger(input.geometry?.layerIdx) ? input.geometry.layerIdx : 0,
    numTokens: requirePositiveInteger(input.geometry?.numTokens, 'geometry.numTokens'),
    numHeads: requirePositiveInteger(input.geometry?.numHeads, 'geometry.numHeads'),
    numKVHeads: requirePositiveInteger(input.geometry?.numKVHeads, 'geometry.numKVHeads'),
    headDim: requirePositiveInteger(input.geometry?.headDim, 'geometry.headDim'),
    hiddenSize: requirePositiveInteger(input.geometry?.hiddenSize, 'geometry.hiddenSize'),
    currentSeqLen: Number.isInteger(input.geometry?.currentSeqLen) && input.geometry.currentSeqLen >= 0
      ? input.geometry.currentSeqLen
      : 0,
  };
  const normalized = {
    phase: input.phase,
    geometry,
    dtypes: {
      input: requireDtype(input.dtypes?.input, 'dtypes.input'),
      activation: requireDtype(input.dtypes?.activation, 'dtypes.activation'),
      projection: requireDtype(input.dtypes?.projection, 'dtypes.projection'),
      kv: requireDtype(input.dtypes?.kv, 'dtypes.kv'),
      cachedK: requireDtype(input.dtypes?.cachedK, 'dtypes.cachedK'),
      cachedV: requireDtype(input.dtypes?.cachedV, 'dtypes.cachedV'),
      outputProjectionInput: requireDtype(
        input.dtypes?.outputProjectionInput,
        'dtypes.outputProjectionInput'
      ),
      output: requireDtype(input.dtypes?.output, 'dtypes.output'),
    },
    kv: {
      layout: String(input.kv?.layout ?? 'contiguous'),
      coldQuantMode: String(input.kv?.coldQuantMode ?? 'none'),
      length: requirePositiveInteger(input.kv?.length, 'kv.length'),
      pageSize: Number.isInteger(input.kv?.pageSize) ? input.kv.pageSize : 0,
    },
    session: input.session ?? {},
    capabilities: {
      hasF16: input.capabilities?.hasF16 === true,
      hasSubgroups: input.capabilities?.hasSubgroups === true,
    },
    fusion: {
      inputNormProjection: input.fusion?.inputNormProjection === true,
      qkvProjection: input.fusion?.qkvProjection === true,
      qkNorm: input.fusion?.qkNorm === true,
      qkNormRoPE: input.fusion?.qkNormRoPE === true,
    },
    queryTransform: {
      scale: requirePositiveFinite(input.queryTransform?.scale, 'queryTransform.scale', 1),
      rope: input.queryTransform?.rope === 'disabled' ? 'disabled' : 'enabled',
    },
    outputGate: {
      enabled: input.outputGate?.enabled === true,
      tensorDtype: input.outputGate?.tensorDtype ?? null,
      tensorElementCount: Number.isInteger(input.outputGate?.tensorElementCount)
        ? input.outputGate.tensorElementCount
        : 0,
      kernelPathSupportsFusion: input.outputGate?.kernelPathSupportsFusion === true,
    },
    outputProjection: {
      weightDtype: input.outputProjection?.weightDtype ?? null,
      hasResidual: input.outputProjection?.hasResidual === true,
      residualDtype: input.outputProjection?.residualDtype ?? null,
      hasLoRA: input.outputProjection?.hasLoRA === true,
    },
    diagnosticsEligible: input.diagnosticsEligible === true,
    sharedKV: {
      reuses: input.sharedKV?.reuses === true,
      stores: input.sharedKV?.stores === true,
    },
    diffusionGemmaDecoder: input.diffusionGemmaDecoder === true,
  };
  const implementation = resolveAttentionImplementation(normalized);
  const planCore = {
    phase: normalized.phase,
    geometry: normalized.geometry,
    dtypes: normalized.dtypes,
    transitions: {
      inputCast: normalized.dtypes.input === normalized.dtypes.activation
        ? null
        : `${normalized.dtypes.input}->${normalized.dtypes.activation}`,
      projectionCast: normalized.dtypes.projection === normalized.dtypes.outputProjectionInput
        ? null
        : `${normalized.dtypes.projection}->${normalized.dtypes.outputProjectionInput}`,
    },
    fusion: normalized.fusion,
    queryTransform: normalized.queryTransform,
    kv: {
      ...normalized.kv,
      implementation,
    },
    attention: {
      implementation,
      flashPrefill: normalized.phase === 'prefill'
        && normalized.diffusionGemmaDecoder !== true
        && normalized.session.useFlashPrefillAttention === true,
      ortFlashPrefill: normalized.phase === 'prefill'
        && normalized.diffusionGemmaDecoder !== true
        && normalized.session.useOrtFlashPrefillAttention === true,
    },
    outputGate: {
      enabled: normalized.outputGate.enabled,
      semantics: normalized.outputGate.enabled ? 'sigmoid' : 'none',
      mode: resolveOutputGateMode(normalized),
    },
    outputProjection: {
      inputDtype: normalized.dtypes.outputProjectionInput,
      outputDtype: normalized.dtypes.output,
      residualMode: resolveOutputResidualMode(normalized),
      hasLoRA: normalized.outputProjection.hasLoRA,
    },
    observation: {
      diagnosticsEligible: normalized.diagnosticsEligible,
    },
    lifetimes: {
      retainSharedKV: normalized.sharedKV.reuses || normalized.sharedKV.stores,
      valueMayAliasKey: true,
      submitRetentionRequired: true,
    },
    stages: [
      'input-normalization',
      'qkv-projection',
      'kv-update',
      'attention',
      'output-gate',
      'output-projection',
    ],
  };
  return deepFreeze({
    schema: ATTENTION_PLAN_SCHEMA,
    id: hashRefactorReceiptValue(planCore),
    ...planCore,
  });
}

function tensorElementCount(tensor) {
  if (!Array.isArray(tensor?.shape)) {
    return 0;
  }
  return tensor.shape.reduce((total, value) => total * value, 1);
}

export function resolveAttentionPlanForDispatch(options = {}) {
  const config = options.config ?? {};
  const dispatchParams = options.dispatchParams ?? {};
  const kvState = options.kvState ?? {};
  return resolveAttentionPlan({
    phase: config.isPrefill ? 'prefill' : 'decode',
    geometry: {
      layerIdx: config.layerIdx,
      numTokens: config.numTokens,
      numHeads: config.numHeads,
      numKVHeads: config.numKVHeads,
      headDim: config.headDim,
      hiddenSize: config.hiddenSize,
      currentSeqLen: config.currentSeqLen,
    },
    dtypes: {
      input: options.inputDtype,
      activation: options.activationDtype,
      projection: options.projectionDtype,
      kv: options.kvDtype,
      cachedK: dispatchParams.cachedKDtype,
      cachedV: dispatchParams.cachedVDtype,
      outputProjectionInput: options.outputProjectionInputDtype,
      output: options.outputDtype,
    },
    kv: {
      layout: kvState.kvLayout,
      coldQuantMode: kvState.coldQuantMode,
      length: kvState.kvLenForAttention,
      pageSize: kvState.kvPageSize ?? kvState.coldPageSize ?? 0,
    },
    session: options.session,
    capabilities: options.capabilities,
    fusion: options.fusion,
    queryTransform: {
      scale: config.queryScale ?? 1,
      rope: config.disableRoPE === true ? 'disabled' : 'enabled',
    },
    outputGate: {
      enabled: config.attentionOutputGate === true,
      tensorDtype: options.qGateTensor?.dtype ?? null,
      tensorElementCount: tensorElementCount(options.qGateTensor),
      kernelPathSupportsFusion: kernelPathSupportsOutputGateFusion(config.kernelPath),
    },
    outputProjection: {
      weightDtype: options.outputProjectionWeightDtype ?? null,
      hasResidual: config.residualTensor != null,
      residualDtype: config.residualTensor?.dtype ?? null,
      hasLoRA: options.outputProjectionHasLoRA === true,
    },
    diagnosticsEligible: options.diagnosticsEligible === true,
    sharedKV: options.sharedKV,
    diffusionGemmaDecoder: config.diffusionGemmaDecoder === true,
  });
}

export function bindAttentionPlan(plan, resources = {}) {
  if (!plan || plan.schema !== ATTENTION_PLAN_SCHEMA) {
    throw new Error('[AttentionPlan] bindAttentionPlan requires a resolved semantic plan.');
  }
  return Object.freeze({
    plan,
    resources,
  });
}

export async function executeBoundAttentionPlan(boundPlan, executor) {
  if (!boundPlan?.plan || !executor || typeof executor.executeStage !== 'function') {
    throw new Error('[AttentionPlan] executeBoundAttentionPlan requires a bound plan and executor.');
  }
  let result;
  for (const stage of boundPlan.plan.stages) {
    result = await executor.executeStage(stage, boundPlan, result);
  }
  return result;
}

export function createAttentionExecutor(mode, stageRunners = {}) {
  if (mode !== 'immediate' && mode !== 'recorded') {
    throw new Error('[AttentionPlan] executor mode must be "immediate" or "recorded".');
  }
  return Object.freeze({
    mode,
    async executeStage(stage, boundPlan, previous) {
      const runner = stageRunners[stage];
      return typeof runner === 'function'
        ? runner(boundPlan, previous)
        : previous;
    },
  });
}
