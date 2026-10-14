import { canonicalizeJson } from '../../../formats/canonical-hash.js';
import { ERROR_CODES, createDopplerError } from '../../../errors/doppler-error.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { getKernelPathRMSNormSpec } from '../../../config/kernel-path-loader.js';
import { getPipelineFast, resolvePipelineIdentity } from '../../../gpu/kernels/pipeline-cache.js';
import { resolveGatherPipeline, resolveSplitGatherPipeline } from '../../../gpu/kernels/gather.js';
import { resolvePairedCastPipeline } from '../../../gpu/kernels/paired-cast.js';
import { resolvePairedRMSNormPipeline, resolvePairedRMSNormVariant } from '../../../gpu/kernels/paired-rmsnorm.js';
import { resolvePairedPointwisePipeline } from '../../../gpu/kernels/paired-activation.js';
import { resolvePairedSplitPipeline } from '../../../gpu/kernels/paired-split.js';
import { resolvePairedAttentionPipeline, resolvePairedAttentionVariant } from '../../../gpu/kernels/paired-attention.js';
import { listPairedLinearAttentionPipelines } from '../../../gpu/kernels/paired-linear-attention.js';
import { resolvePairedProjectionPipeline } from '../../../gpu/kernels/matmul.js';
import { resolveRoPEQKPipeline } from '../../../gpu/kernels/rope-qk.js';
import { resolveLogitFinalizePipeline } from '../../../gpu/kernels/logit-finalize.js';
import { resolveLogitScatterPipeline } from '../../../gpu/kernels/logit-scatter.js';
import {
  createWeightBuffer, getWeightDtype, getWeightMetadata, isGpuBufferInstance, isSplitWeightBuffer, isWeightBuffer,
} from '../../../gpu/weight-buffer.js';
import { resolveActiveExecutionPlan } from './execution-plan.js';
import { detectSandwichNorm } from './layer.js';
import { resolveLayerScalarValue } from './layer-execution-contract.js';
import { resolveAttentionQKNormState } from './attention/projections.js';
import { isRoPEDisabledForLayer, resolveQueryScale } from './attention/heterogeneous-contract.js';
import { ATTENTION_OUTPUT_GATE_ACTIVATION } from './attention/plan.js';
import { getLogitsConfig, getLogitsWeights } from './generator/logits-config.js';
import { resolveLmHeadMatmulRole } from './logits/gpu-executor.js';
import { resolveLogitInputScale } from './logits/scale-policy.js';

export const RESIDENT_PIPELINE_PREPARATION_SCHEMA = 'doppler.resident-pipeline-preparation/v1';

/** @type {readonly ('prefill' | 'decode')[]} */
const PHASES = Object.freeze(['prefill', 'decode']);
const UNFUSED = Object.freeze({ preResidual: false, residual: false, outputPrenorm: false });
const UNGATED = Object.freeze({ inplace: false, gatedInputActivation: null });

// Raised only while enumerating: the stage program leaves the declared paired
// structure this module enumerates, so its pipelines remain lazy and reported.
class NotEnumerable extends Error {}

/**
 * @param {unknown} weight
 * @param {string} role
 * @returns {GPUBuffer | import('../../../gpu/weight-buffer.js').WeightBuffer}
 */
function requireGpuWeight(weight, role) {
  if (isWeightBuffer(weight) || isGpuBufferInstance(weight)) return weight;
  throw new NotEnumerable(`${role} weight is not GPU-resident.`);
}

/** RMSNorm call sites bind the raw GPU buffer, whose dtype tag equals the weight dtype.
 * @param {unknown} weight @param {string} role */
function normWeightDtype(weight, role) {
  return getWeightDtype(requireGpuWeight(weight, role));
}

/**
 * Ordered, de-duplicated pipeline requests. Two requests share an entry when
 * they compile the same shader, entry point and resolved constants.
 */
function createRequestSet() {
  /** @type {Map<string, import('./resident-pipeline-preparation.js').ResidentPipelineEntry>} */
  const entries = new Map();
  return {
    /** @param {string} role @param {import('../../../gpu/kernels/pipeline-cache.js').KernelPipelineRequest} request */
    add(role, request) {
      const identity = resolvePipelineIdentity(request);
      const key = `${identity.shaderFile}#${identity.entryPoint}${identity.constants ? ` ${canonicalizeJson(identity.constants)}` : ''}`;
      const entry = entries.get(key);
      if (entry) {
        if (!entry.roles.includes(role)) entry.roles.push(role);
        return;
      }
      entries.set(key, { key, request, identity, roles: [role] });
    },
    list() { return [...entries.values()]; },
  };
}

/**
 * Declared paired resident program. Each role names the request its call site
 * makes; the end-to-end execution comparison test keeps this table aligned
 * with layer orchestration.
 * @param {import('./partition-execution.js').PartitionExecutionState} owner
 * @param {import('./resident-partition-contract.js').ResidentPartitionAllocation} allocation
 * @param {ReturnType<typeof createRequestSet>} requests
 */
function enumeratePairedStage(owner, allocation, requests) {
  const config = owner.modelConfig;
  if (!config || !owner.executionPlanState) throw new Error('Resident pipeline preparation requires a resolved model and execution plan.');
  const session = owner.runtimeConfig.inference.session;
  if (owner.layerPipelinePlan) throw new NotEnumerable('A layer pipeline plan replaces the declared layer program.');
  if (detectSandwichNorm(config).useSandwichNorm) throw new NotEnumerable('Sandwich normalization uses another post-attention program.');
  if (Number(config.residualBranchScale) !== 1) throw new NotEnumerable('Scaled residual branches use another residual program.');
  if (session.useLinearAttentionQKVZProjectionFusion === true || session.useLinearAttentionABProjectionFusion === true) {
    throw new NotEnumerable('Fused linear-attention decode projections are outside the declared paired program.');
  }
  const kernelPath = resolveActiveExecutionPlan(owner.executionPlanState).kernelPath ?? null;
  const offset = config.rmsNormWeightOffset === true;
  const partition = allocation.plan.partitions[allocation.index];
  /** Each phase's declared paired RMSNorm program for a role.
   * @param {string} role @param {'layer' | 'postLayer'} section @param {number} layerIdx
   * @param {import('../../../config/schema/kernel-path.schema.js').KernelPathSchema | null} path
   * @param {string | null} weightDtype @param {import('../../../gpu/kernels/paired-rmsnorm.js').PairedRMSNormFusion} fusion */
  const norm = (role, section, layerIdx, path, weightDtype, fusion) => {
    for (const phase of PHASES) {
      const declared = getKernelPathRMSNormSpec(role, section, phase, layerIdx, path);
      requests.add(role, resolvePairedRMSNormPipeline(weightDtype, fusion, resolvePairedRMSNormVariant(declared?.variant ?? null)));
    }
  };
  /** @param {string} role @param {unknown} weight @param {number} layerIdx */
  const projection = (role, weight, layerIdx) => {
    for (const phase of PHASES) {
      requests.add(role, resolvePairedProjectionPipeline(requireGpuWeight(weight, role),
        { role, phase, layerIdx, kernelPath, transposeB: 'auto', outputPrecision: 'two-f32' }));
    }
  };

  if (partition.hasEmbedding) {
    const embedding = owner.weights.get('embed');
    if (isSplitWeightBuffer(embedding)) {
      requests.add('embed', resolveSplitGatherPipeline(embedding, 'f32', null));
    } else {
      const weight = requireGpuWeight(embedding, 'embed');
      requests.add('embed', resolveGatherPipeline({
        embeddingDtype: selectRuleValue('inference', 'dtype', 'embeddingDtype', { dtype: getWeightDtype(weight) }),
        outputDtype: 'f32', hiddenSize: config.hiddenSize, inputHiddenSize: config.hiddenSize, hiddenOffset: 0,
        storageEncoding: getWeightMetadata(weight)?.storageEncoding ?? null,
      }));
    }
    requests.add('embed', resolvePairedCastPipeline(true));
  }

  const [firstLayer, lastLayer] = partition.layerRange;
  for (let layerIdx = firstLayer; layerIdx <= lastLayer; layerIdx++) {
    const weights = /** @type {Record<string, unknown> | undefined} */ (/** @type {unknown} */ (owner.weights.get(`layer_${layerIdx}`)));
    if (!weights) throw new Error(`Resident pipeline preparation found no loaded weights for layer ${layerIdx}.`);
    if (resolveLayerScalarValue(/** @type {Float32Array | null | undefined} */ (weights.layerScalar)) !== 1) {
      throw new NotEnumerable(`Layer ${layerIdx} applies a layer scalar outside the declared paired program.`);
    }
    if (!weights.postAttnNorm) throw new NotEnumerable(`Layer ${layerIdx} has no fused post-attention normalization.`);
    norm('input_norm', 'layer', layerIdx, kernelPath, normWeightDtype(weights.inputNorm, 'input_norm'),
      { rmsNormWeightOffset: offset, ...UNFUSED });
    const layerType = config.layerTypes?.[layerIdx];
    if (layerType === 'full_attention') {
      enumeratePairedFullAttention(owner, layerIdx, weights, requests, projection, kernelPath);
    } else if (layerType === 'linear_attention') {
      for (const [role, field] of [['linear_qkv_proj', 'qkvProj'], ['linear_z_proj', 'linearInProjZ'],
        ['linear_a_proj', 'linearInProjA'], ['linear_b_proj', 'linearInProjB']]) {
        projection(role, weights[field], layerIdx);
      }
      for (const request of listPairedLinearAttentionPipelines(config.linearNormGateActivation, config.pairedMechanismReplacements)) {
        requests.add('linear_attention', request);
      }
      projection('linear_out_proj', weights.oProj, layerIdx);
    } else {
      throw new NotEnumerable(`Layer ${layerIdx} type "${String(layerType)}" is outside the declared paired program.`);
    }
    // The attention residual is fused into the FFN input normalization.
    norm('post_attn_norm', 'layer', layerIdx, kernelPath, normWeightDtype(weights.postAttnNorm, 'post_attn_norm'),
      { rmsNormWeightOffset: offset, preResidual: true, residual: false, outputPrenorm: true });
    projection('ffn_gate', weights.gate, layerIdx);
    projection('ffn_up', weights.up, layerIdx);
    requests.add('activation', resolvePairedPointwisePipeline('silu', { inplace: false, gatedInputActivation: 'identity' }));
    projection('ffn_down', weights.down, layerIdx);
    requests.add('ffn_residual', resolvePairedPointwisePipeline('add', UNGATED));
  }

  if (partition.hasLmHead) {
    const logitsConfig = getLogitsConfig(owner);
    const { finalNorm, lmHead } = getLogitsWeights(owner);
    norm('final_norm', 'postLayer', 0, logitsConfig.kernelPath ?? null, normWeightDtype(finalNorm, 'final_norm'),
      { rmsNormWeightOffset: logitsConfig.rmsNormWeightOffset === true, ...UNFUSED });
    if (resolveLogitInputScale(logitsConfig) !== 1) requests.add('logit_input_scale', resolvePairedPointwisePipeline('scale', UNGATED));
    for (const phase of PHASES) {
      const role = resolveLmHeadMatmulRole(phase);
      const weight = isSplitWeightBuffer(lmHead)
        ? createWeightBuffer(lmHead.sections[0].buffer, lmHead.dtype, lmHead.layout, [...lmHead.shape], 'lm_head:split:0')
        : requireGpuWeight(lmHead, role);
      requests.add(role, resolvePairedProjectionPipeline(weight,
        { role, phase, layerIdx: 0, kernelPath: logitsConfig.kernelPath ?? null, transposeB: 'auto', outputPrecision: 'f32' }));
    }
    if (isSplitWeightBuffer(lmHead)) requests.add('lm_head_scatter', resolveLogitScatterPipeline('f32'));
    requests.add('logits_finalize', resolveLogitFinalizePipeline('f32'));
  }
}

/**
 * @param {import('./partition-execution.js').PartitionExecutionState} owner
 * @param {number} layerIdx
 * @param {Record<string, unknown>} weights
 * @param {ReturnType<typeof createRequestSet>} requests
 * @param {(role: string, weight: unknown, layerIdx: number) => void} projection
 * @param {import('../../../config/schema/kernel-path.schema.js').KernelPathSchema | null} kernelPath
 */
function enumeratePairedFullAttention(owner, layerIdx, weights, requests, projection, kernelPath) {
  const config = /** @type {NonNullable<typeof owner.modelConfig>} */ (owner.modelConfig);
  if (weights.qkvProj && weights.qkvSizes) throw new NotEnumerable(`Layer ${layerIdx} fuses Q/K/V projections outside the declared paired program.`);
  projection('q_proj', weights.qProj, layerIdx);
  projection('k_proj', weights.kProj, layerIdx);
  projection('v_proj', weights.vProj, layerIdx);
  if (config.attentionOutputGate === true) requests.add('q_gate_split', resolvePairedSplitPipeline('qg'));
  const queryKeyNorm = config.queryKeyNorm === true
    && (!Array.isArray(config.queryKeyNormLayers) || config.queryKeyNormLayers.includes(layerIdx));
  const qkNorm = resolveAttentionQKNormState({
    config: { queryKeyNorm, queryKeyNormWeightLayers: config.queryKeyNormWeightLayers,
      rmsNormWeightOffset: config.rmsNormWeightOffset },
    layerWeights: /** @type {import('./weights.js').LayerWeights} */ (/** @type {unknown} */ (weights)),
    layerIdx, reusesSharedKV: false,
  });
  if (qkNorm.wantsQKNorm) {
    if (qkNorm.allowUnitQKNorm || config.queryKeyNormType !== 'rmsnorm') {
      throw new NotEnumerable(`Layer ${layerIdx} Q/K normalization is outside the declared paired program.`);
    }
    // Head normalization has no declared step; it runs the unpinned paired program.
    for (const [role, field] of [['q_norm', 'qNorm'], ['k_norm', 'kNorm']]) {
      requests.add(role, resolvePairedRMSNormPipeline(normWeightDtype(weights[field], role),
        { rmsNormWeightOffset: qkNorm.rmsNormWeightOffset, ...UNFUSED }, resolvePairedRMSNormVariant(null)));
    }
  }
  if (resolveQueryScale(config.queryScale) !== 1) requests.add('query_scale', resolvePairedPointwisePipeline('scale', UNGATED));
  if (!isRoPEDisabledForLayer(config, layerIdx) && owner.ropeFreqsCos && owner.ropeFreqsSin) {
    requests.add('rope_qk', resolveRoPEQKPipeline('f32', 'two-f32'));
  }
  // Paired K/V store directly into the F16 contiguous cache.
  requests.add('kv_cache_store', resolvePairedCastPipeline(false));
  for (const phase of PHASES) {
    requests.add('attention', resolvePairedAttentionPipeline(resolvePairedAttentionVariant(kernelPath, phase, layerIdx)));
  }
  if (config.attentionOutputGate === true) {
    requests.add('attention_output_gate', resolvePairedPointwisePipeline(ATTENTION_OUTPUT_GATE_ACTIVATION,
      { inplace: false, gatedInputActivation: 'identity' }));
  }
  projection('o_proj', weights.oProj, layerIdx);
}

/** @type {import('./resident-pipeline-preparation.js').resolveResidentPipelineProgram} */
export function resolveResidentPipelineProgram(owner, allocation) {
  if (allocation.plan.activationPrecision !== 'two-f32') {
    return { pipelines: [], lazy: [{ scope: 'program', reason: 'Scalar resident programs select matmul, attention and '
      + 'normalization variants from token and KV lengths and fused-kernel policy; their pipelines are created on first use.' }] };
  }
  const requests = createRequestSet();
  try {
    enumeratePairedStage(owner, allocation, requests);
  } catch (error) {
    if (!(error instanceof NotEnumerable)) throw error;
    return { pipelines: [], lazy: [{ scope: 'stage', reason: `${error.message} Its pipelines are created on first use.` }] };
  }
  return { pipelines: requests.list(), lazy: [] };
}

/**
 * @param {import('./resident-partition-contract.js').ResidentPartitionAllocation} allocation
 * @param {import('./resident-pipeline-preparation.js').ResidentPipelineEntry} entry
 * @param {number} maxDurationMs @param {number} prepared @param {number} total
 */
function preparationTimeout(allocation, entry, maxDurationMs, prepared, total) {
  return createDopplerError(ERROR_CODES.RESIDENT_PREPARATION_TIMEOUT, `Resident stage ${allocation.index} exceeded its `
    + `pipeline preparation bound (limits.maxPreparationMs=${maxDurationMs}) while preparing GPU pipeline `
    + `${entry.request.operation}/${entry.request.variant} (${entry.key}) after ${prepared} of ${total} pipelines `
    + 'and was not advertised ready. Inspect the driver shader cache for this program, place the stage on another '
    + 'device, or raise the bound in the host allocation policy only if this device is expected to compile it that slowly.');
}

/** @type {import('./resident-pipeline-preparation.js').prepareResidentPartitionPipelines} */
export async function prepareResidentPartitionPipelines(owner, allocation, signal, onProgress) {
  signal?.throwIfAborted();
  const program = resolveResidentPipelineProgram(owner, allocation);
  const total = program.pipelines.length;
  const device = owner.gpuContext?.device;
  const maxDurationMs = allocation.limits?.maxPreparationMs;
  if (total && !device) throw new Error('Resident pipeline preparation requires the stage GPU device.');
  if (total && (!Number.isSafeInteger(maxDurationMs) || maxDurationMs <= 0)) {
    throw new Error('Resident pipeline preparation requires a positive limits.maxPreparationMs declared by the host allocation policy.');
  }
  const startedAt = Date.now(), started = performance.now(), deadline = started + maxDurationMs;
  let prepared = 0;
  /** @param {import('./resident-partition-contract.js').ResidentPipelinePreparationProgress['status']} status
   * @param {import('./resident-pipeline-preparation.js').ResidentPipelineEntry | null} entry
   * @param {boolean} outstanding @param {Error | null} error */
  const report = (status, entry, outstanding, error) => onProgress?.(Object.freeze({ phase: 'pipelines', status, prepared,
    total, key: entry?.key ?? null, elapsedMs: performance.now() - started, startedAt,
    maxDurationMs: total ? maxDurationMs : null, outstanding, complete: program.lazy.length === 0,
    ...(error ? { error: error.message } : {}) }));
  for (const entry of program.pipelines) {
    const { key, request } = entry;
    if (signal?.aborted) { report('cancelled', entry, false, null); throw signal.reason; }
    if (performance.now() >= deadline) {
      const timeout = preparationTimeout(allocation, entry, maxDurationMs, prepared, total);
      report('timed-out', entry, false, timeout); throw timeout;
    }
    report('preparing', entry, true, null);
    // The same cache key execution uses: device, registry, shader source, entry, constants, auto layout.
    const creation = getPipelineFast(request.operation, request.variant, null, request.constants, device)
      .then(() => null, /** @param {unknown} error */ error => ({ error }));
    // Cancellation and the bound end preparation, but a started provider creation cannot
    // be interrupted: the opening keeps ownership and does not settle until it does.
    /** @type {null | 'cancelling' | 'timed-out'} */
    let interruption = null;
    const timeout = preparationTimeout(allocation, entry, maxDurationMs, prepared, total);
    const cancel = () => { if (!interruption) { interruption = 'cancelling'; report('cancelling', entry, true, null); } };
    const timer = setTimeout(() => {
      if (!interruption) { interruption = 'timed-out'; report('timed-out', entry, true, timeout); }
    }, Math.max(0, deadline - performance.now()));
    signal?.addEventListener('abort', cancel, { once: true });
    let outcome;
    try { outcome = await creation; }
    finally { clearTimeout(timer); signal?.removeEventListener('abort', cancel); }
    if (!outcome) prepared++;
    if (interruption === 'cancelling' || (!interruption && signal?.aborted)) {
      report('cancelled', entry, false, null); throw /** @type {AbortSignal} */ (signal).reason;
    }
    if (interruption === 'timed-out') { report('timed-out', entry, false, timeout); throw timeout; }
    if (outcome) {
      const cause = outcome.error;
      const failure = new Error(`Resident stage ${allocation.index} could not create GPU pipeline ${request.operation}/${request.variant} `
        + `(${key}) and was not advertised ready: ${cause instanceof Error ? cause.message : String(cause)}. `
        + 'Inspect the GPU provider and driver shader cache for this program, or place the stage on another device.',
      { cause });
      report('failed', entry, false, failure); throw failure;
    }
  }
  report('prepared', null, false, null);
  return {
    schema: RESIDENT_PIPELINE_PREPARATION_SCHEMA,
    complete: program.lazy.length === 0,
    count: total,
    durationMs: performance.now() - started,
    pipelines: program.pipelines.map(({ key, identity, roles }) => ({ key, operation: identity.operation,
      variant: identity.variant, shaderFile: identity.shaderFile, entryPoint: identity.entryPoint,
      constants: identity.constants, roles: [...roles] })),
    lazy: program.lazy.map(entry => ({ ...entry })),
  };
}
