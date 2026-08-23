import { validateModelIR } from '../config/model-ir.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../config/kernels/kernel-ref-digests.js';
import { expandExecutionV1 } from '../config/schema/index.js';
import { validateRequiredInferenceFields } from '../inference/pipelines/text/config.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';
import { sanitizeModelId } from './core.js';

export const SEMANTIC_MANIFEST_LOWERING_SCHEMA_ID = 'doppler.semantic-manifest-lowering/v1';

const TEXT_PHASES = Object.freeze(['prefill', 'decode']);
const LOGIT_OPERATION_ORDER = Object.freeze([
  'language-model-head',
  'multiply',
  'divide-by-softcap',
  'tanh',
  'multiply-by-softcap',
]);
const BLOCK_KIND_TO_LAYER_TYPE = Object.freeze({
  'local-attention': 'sliding_attention',
  'full-attention': 'full_attention',
});

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function clone(value) {
  return structuredClone(value);
}

function digest(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function requireAuthor(author) {
  if (!isObject(author) || !['human', 'ai', 'tool'].includes(author.kind)
    || typeof author.actor !== 'string' || !author.actor.trim()) {
    throw new Error('Semantic manifest lowering requires attributable authorship.');
  }
}

function requireObject(value, label) {
  if (!isObject(value)) throw new Error(`${label} must be an object.`);
  return value;
}

function requireEqual(observed, expected, label) {
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error(`${label} is unsupported: expected ${JSON.stringify(expected)}, got ${JSON.stringify(observed)}.`);
  }
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer.`);
  return value;
}

function requireNonNegativeInteger(value, label) {
  if (!Number.isInteger(value) || value < 0) throw new Error(`${label} must be a non-negative integer.`);
  return value;
}

function requirePositiveNumber(value, label) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be a positive finite number.`);
  return value;
}

function requireBoolean(value, label) {
  if (typeof value !== 'boolean') throw new Error(`${label} must be boolean.`);
  return value;
}

function requireExactKeys(value, keys, label) {
  const observed = Object.keys(requireObject(value, label)).sort();
  const expected = [...keys].sort();
  requireEqual(observed, expected, `${label} fields`);
}

function requireNode(nodes, predicate, label) {
  const node = nodes.find(predicate);
  if (!node) throw new Error(`ModelIR is missing ${label}.`);
  return node;
}

function requireAcceptedFacts(modelIR, factRefs, label) {
  const factsById = new Map(modelIR.provenance.facts.map((fact) => [fact.id, fact]));
  const refs = [...new Set(factRefs)].sort();
  const evidence = [];
  for (const factId of refs) {
    const fact = factsById.get(factId);
    if (!fact || !['direct', 'derived'].includes(fact.confidence)
      || fact.disposition !== 'accepted' || fact.validation?.status !== 'passed') {
      throw new Error(`${label} requires accepted, validated fact "${factId}".`);
    }
    evidence.push({
      factId,
      confidence: fact.confidence,
      evidence: clone(fact.evidence),
      validationReceipt: fact.validation.receipt,
    });
  }
  return { factRefs: refs, evidence };
}

function requireSharedSection(blockClasses, sectionName) {
  const first = blockClasses[0][sectionName];
  for (const blockClass of blockClasses.slice(1)) {
    requireEqual(blockClass[sectionName], first, `${sectionName} contract across text block classes`);
  }
  return first;
}

function requireSharedGeometry(blockClasses) {
  const geometryKeys = [
    'numHeads', 'numKvHeads', 'headDim', 'attentionScale', 'causal', 'attentionBias',
    'queryKeyNorm', 'queryScale', 'outputGate', 'outputGateType', 'outputGatePosition',
  ];
  const localKeys = [...geometryKeys, 'window'];
  for (const blockClass of blockClasses) {
    requireExactKeys(
      blockClass.geometry,
      blockClass.kind === 'local-attention' ? localKeys : geometryKeys,
      `block class "${blockClass.id}" geometry`
    );
  }
  const common = Object.fromEntries(geometryKeys.map((key) => [key, blockClasses[0].geometry[key]]));
  for (const blockClass of blockClasses.slice(1)) {
    const observed = Object.fromEntries(geometryKeys.map((key) => [key, blockClass.geometry[key]]));
    requireEqual(observed, common, 'attention geometry across text block classes');
  }
  return common;
}

function validateBlockContracts(blockClasses) {
  if (!blockClasses.some((blockClass) => blockClass.kind === 'local-attention')
    || !blockClasses.some((blockClass) => blockClass.kind === 'full-attention')) {
    throw new Error('Semantic text lowering requires both local-attention and full-attention block classes.');
  }
  for (const blockClass of blockClasses) {
    if (!Object.hasOwn(BLOCK_KIND_TO_LAYER_TYPE, blockClass.kind)) {
      throw new Error(`Block kind "${blockClass.kind}" has no conservative text-manifest lowering.`);
    }
    requireEqual(blockClass.phaseBehavior, { prefill: true, decode: true }, `block class "${blockClass.id}" phase behavior`);
  }

  const geometry = requireSharedGeometry(blockClasses);
  requirePositiveInteger(geometry.numHeads, 'attention numHeads');
  requirePositiveInteger(geometry.numKvHeads, 'attention numKvHeads');
  const headDim = requirePositiveInteger(geometry.headDim, 'attention headDim');
  requireEqual(geometry.attentionScale, 'inverse-sqrt-head-dim', 'attention scale');
  requireBoolean(geometry.causal, 'attention causal');
  requireBoolean(geometry.attentionBias, 'attention bias');
  requireEqual(geometry.queryKeyNorm, { type: 'rmsnorm', withScale: false }, 'query/key normalization');
  const queryScale = requirePositiveNumber(geometry.queryScale, 'attention queryScale');
  requireEqual(geometry.outputGate, true, 'attention output gate');
  requireEqual(geometry.outputGateType, 'sigmoid', 'attention output gate type');
  requireEqual(
    geometry.outputGatePosition,
    'attention-output-before-output-projection',
    'attention output gate position'
  );

  const normalization = requireSharedSection(blockClasses, 'normalization');
  requireExactKeys(normalization, [
    'type', 'eps', 'weightOffset', 'postNormEps', 'postNormWeightOffset', 'postNormPosition',
  ], 'text block normalization');
  requireEqual(normalization.type, 'centered-rmsnorm-with-postnorm', 'text normalization type');
  const rmsNormEps = requirePositiveNumber(normalization.eps, 'text normalization eps');
  requireEqual(normalization.weightOffset, 1, 'text normalization weight offset');
  const postNormEps = requirePositiveNumber(normalization.postNormEps, 'text post-normalization eps');
  requireEqual(normalization.postNormWeightOffset, 1, 'text post-normalization weight offset');
  requireEqual(
    normalization.postNormPosition,
    'sublayer-output-before-residual',
    'text post-normalization position'
  );

  const feedForward = requireSharedSection(blockClasses, 'feedForward');
  requireExactKeys(feedForward, ['type', 'gated', 'activation', 'operation', 'intermediateSize'], 'text feed-forward');
  requireEqual(feedForward.type, 'gated-dense', 'text feed-forward type');
  requireEqual(feedForward.gated, true, 'text feed-forward gate');
  requireEqual(feedForward.activation, 'silu', 'text feed-forward activation');
  requireEqual(feedForward.operation, 'down(silu(gate(x))*up(x))', 'text feed-forward operation');
  requirePositiveInteger(feedForward.intermediateSize, 'text feed-forward intermediateSize');

  const local = requireNode(blockClasses, (blockClass) => blockClass.kind === 'local-attention', 'local-attention block');
  requireExactKeys(local.positional, ['type', 'theta', 'ropeType'], 'local-attention positional contract');
  requireEqual(local.positional.type, 'rope', 'local-attention positional type');
  requireEqual(local.positional.ropeType, 'default', 'local-attention RoPE type');
  const ropeTheta = requirePositiveNumber(local.positional.theta, 'local-attention RoPE theta');
  const slidingWindow = requirePositiveInteger(local.geometry.window, 'local-attention window');

  const full = requireNode(blockClasses, (blockClass) => blockClass.kind === 'full-attention', 'full-attention block');
  requireExactKeys(full.positional, ['type', 'theta'], 'full-attention positional contract');
  requireEqual(full.positional, { type: 'no-rope', theta: 0 }, 'full-attention positional contract');

  return {
    geometry,
    headDim,
    queryScale,
    rmsNormEps,
    postNormEps,
    feedForward,
    ropeTheta,
    slidingWindow,
  };
}

function buildInference({ component, schedule, classById, blockContract, outputHead, chatTemplate }) {
  const layerTypes = schedule.blocks.map((block) => BLOCK_KIND_TO_LAYER_TYPE[classById.get(block.blockClassId).kind]);
  const disabledLayers = schedule.blocks
    .filter((block) => classById.get(block.blockClassId).kind === 'full-attention')
    .map((block) => block.index);
  const embeddingNormalization = component.properties.embeddingNormalization;
  requireEqual(embeddingNormalization, { type: 'rmsnorm', withScale: false }, 'embedding normalization');
  requireEqual(component.properties.finalNormalization, { type: 'rmsnorm', withScale: true }, 'final normalization');
  requireEqual(outputHead.properties.operationOrder, LOGIT_OPERATION_ORDER, 'logit operation order');
  requireEqual(outputHead.properties.tieWordEmbeddings, component.properties.tieWordEmbeddings, 'output-head tying contract');

  return {
    attention: {
      queryPreAttnScalar: blockContract.headDim,
      queryScale: blockContract.queryScale,
      attnLogitSoftcapping: null,
      slidingWindow: blockContract.slidingWindow,
      queryKeyNorm: true,
      queryKeyNormType: 'rmsnorm',
      queryKeyNormAxis: 'head',
      queryKeyNormLayers: null,
      queryKeyNormWeightLayers: [],
      valueNorm: false,
      causal: blockContract.geometry.causal,
      attentionBias: blockContract.geometry.attentionBias,
      attentionOutputGate: true,
      outputGateType: 'sigmoid',
    },
    normalization: {
      type: 'rmsnorm',
      rmsNormEps: blockContract.rmsNormEps,
      rmsNormWeightOffset: true,
      postNormEps: blockContract.postNormEps,
      postNormWeightOffset: true,
      postAttentionNorm: true,
      preFeedforwardNorm: true,
      postFeedforwardNorm: true,
      finalNormBiasTensor: null,
    },
    ffn: {
      activation: blockContract.feedForward.activation,
      gatedActivation: blockContract.feedForward.gated,
      branchMode: 'dense',
      useDoubleWideMlp: false,
      swigluLimit: null,
    },
    rope: {
      ropeTheta: blockContract.ropeTheta,
      ropeLocalTheta: null,
      disabledLayers,
      ropeInterleaved: false,
      mropeInterleaved: false,
      mropeSection: null,
      partialRotaryFactor: null,
      ropeLocalPartialRotaryFactor: null,
      ropeFrequencyBaseDim: null,
      ropeLocalFrequencyBaseDim: null,
      ropeScalingType: null,
      ropeScalingFactor: 1,
      ropeLocalScalingType: null,
      ropeLocalScalingFactor: 1,
      yarnBetaFast: null,
      yarnBetaSlow: null,
      yarnOriginalMaxPos: null,
      ropeLocalYarnBetaFast: null,
      ropeLocalYarnBetaSlow: null,
      ropeLocalYarnOriginalMaxPos: null,
      longropeShortFactor: null,
      longropeLongFactor: null,
      longropeOriginalMaxPos: null,
    },
    output: {
      finalLogitSoftcapping: requirePositiveNumber(
        outputHead.properties.finalLogitSoftcap,
        'final logit softcap'
      ),
      tieWordEmbeddings: requireBoolean(component.properties.tieWordEmbeddings, 'tieWordEmbeddings'),
      scaleEmbeddings: false,
      embeddingScale: 1,
      embeddingNormalization: {
        ...clone(embeddingNormalization),
        eps: blockContract.rmsNormEps,
        position: 'after-scale',
      },
      logitInputScale: 1,
      logitOutputScale: requirePositiveNumber(
        outputHead.properties.preSoftcapMultiplier,
        'pre-softcap multiplier'
      ),
      embeddingTranspose: false,
      embeddingVocabSize: null,
      embeddingPostprocessor: null,
      lmHeadBiasTensor: null,
    },
    layerPattern: {
      type: 'custom',
      globalPattern: null,
      period: null,
      offset: null,
      layerTypes,
      residualBranchScale: 1,
    },
    chatTemplate: clone(chatTemplate),
  };
}

function bindKernelDigests(value, dispositions) {
  if (!value || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    value.forEach((entry) => bindKernelDigests(entry, dispositions));
    return;
  }
  if (typeof value.kernel === 'string' && typeof value.entry === 'string') {
    const key = `${value.kernel}#${value.entry}`;
    const resolved = KERNEL_REF_CONTENT_DIGESTS[key];
    if (typeof resolved !== 'string') throw new Error(`Kernel reference "${key}" has no canonical digest.`);
    const bound = `sha256:${resolved}`;
    const templateDigest = value.digest ?? null;
    value.digest = bound;
    dispositions.push({
      kind: 'kernel-digest-binding',
      kernelRef: key,
      templateDigest,
      digest: bound,
      changed: templateDigest !== bound,
      disposition: 'accepted',
    });
  }
  Object.values(value).forEach((entry) => bindKernelDigests(entry, dispositions));
}

function bindPackModelIR(modelIR, entryPointId, modelId) {
  const packModelIR = clone(modelIR);
  packModelIR.modelId = modelId;
  const entryPoint = requireNode(packModelIR.entryPoints, (candidate) => candidate.id === entryPointId, `entry point "${entryPointId}"`);
  entryPoint.status = 'lowered';
  entryPoint.phases = [...TEXT_PHASES];
  delete entryPoint.reason;
  packModelIR.supportScope.loweredEntryPoints = [...new Set([
    ...packModelIR.supportScope.loweredEntryPoints,
    entryPointId,
  ])].sort();
  packModelIR.supportScope.unloweredEntryPoints = packModelIR.supportScope.unloweredEntryPoints
    .filter((id) => id !== entryPointId)
    .sort();
  const validation = validateModelIR(packModelIR);
  if (!validation.ok) {
    throw new Error(`Semantic lowering produced invalid Pack-bound ModelIR: ${validation.errors.join('; ')}`);
  }
  return packModelIR;
}

function validateMechanismTemplate(template) {
  requireObject(template, 'Semantic lowering template');
  requireObject(template.quantization, 'Semantic lowering template quantization');
  requireObject(template.session, 'Semantic lowering template session');
  requireObject(template.execution, 'Semantic lowering template execution');
  requireObject(template.execution.kernels, 'Semantic lowering template execution kernels');
  requireEqual(template.quantization, {
    weights: 'f16',
    embeddings: 'f16',
    lmHead: 'f16',
    computePrecision: 'f32',
  }, 'Semantic lowering template quantization');
  const steps = expandExecutionV1(template.execution);
  const activations = steps.filter((step) => step.op === 'activation');
  if (activations.length === 0 || activations.some((step) => step.kernel !== 'silu.wgsl')) {
    throw new Error('Semantic lowering template must bind every activation step to silu.wgsl.');
  }
}

function bindConservativeSessionPolicy(session, dispositions) {
  session.speculation = {
    mode: 'none',
    tokens: 1,
    verify: 'greedy',
    threshold: null,
    rollbackOnReject: true,
  };
  session.useSandwichRMSNormPairFusion = false;
  session.usePostFfnNextInputRMSNormPairFusion = false;
  session.usePostAttnNormFusedGateUp = false;
  dispositions.push({
    kind: 'conservative-session-policy',
    speculation: 'none',
    disabledFusions: [
      'usePostAttnNormFusedGateUp',
      'usePostFfnNextInputRMSNormPairFusion',
      'useSandwichRMSNormPairFusion',
    ],
    disposition: 'accepted',
    rationale: 'No speculative drafter or post-normalization fusion has qualification evidence for this entry point.',
  });
}

export function materializeSemanticManifestCandidate({ modelIR, template, recipe }) {
  const validation = validateModelIR(modelIR);
  if (!validation.ok || modelIR?.schema !== 'doppler.model-ir/v2') {
    throw new Error(`Semantic manifest lowering requires ModelIR v2: ${validation.errors.join('; ')}`);
  }
  if (!isObject(recipe) || recipe.schema !== SEMANTIC_MANIFEST_LOWERING_SCHEMA_ID) {
    throw new Error(`Semantic manifest lowering requires recipe "${SEMANTIC_MANIFEST_LOWERING_SCHEMA_ID}".`);
  }
  requireAuthor(recipe.author);
  validateMechanismTemplate(template);
  const requestedModelId = recipe.modelId;
  const modelId = sanitizeModelId(requestedModelId);
  if (!modelId) throw new Error(`Semantic manifest lowering cannot derive a model ID from "${String(requestedModelId)}".`);
  if (recipe.runtimeModelType !== 'transformer') {
    throw new Error('Semantic text lowering currently requires runtimeModelType="transformer".');
  }
  requireEqual(recipe.supportScope, { loweredEntryPoint: recipe.entryPointId }, 'Pack support scope');
  requireExactKeys(recipe.chatTemplate, ['type', 'enabled'], 'chat-template policy');
  if (recipe.chatTemplate.enabled !== false || recipe.chatTemplate.type !== null) {
    throw new Error('Unproven chat templates must remain explicitly disabled during semantic lowering.');
  }
  requireExactKeys(recipe.conversion, ['convertedAt', 'tool'], 'conversion identity policy');
  if (typeof recipe.conversion.convertedAt !== 'string'
    || Number.isNaN(Date.parse(recipe.conversion.convertedAt))
    || new Date(recipe.conversion.convertedAt).toISOString() !== recipe.conversion.convertedAt) {
    throw new Error('conversion identity policy convertedAt must be an ISO-8601 timestamp.');
  }
  requireEqual(recipe.conversion.tool, SEMANTIC_MANIFEST_LOWERING_SCHEMA_ID, 'conversion identity policy tool');

  const entryPoint = requireNode(modelIR.entryPoints, (candidate) => candidate.id === recipe.entryPointId, `entry point "${recipe.entryPointId}"`);
  requireEqual(entryPoint.kind, 'generate', 'entry-point kind');
  const component = requireNode(modelIR.components, (candidate) => candidate.id === entryPoint.componentId, 'entry-point component');
  requireEqual(component.type, 'text-decoder', 'entry-point component type');
  requireExactKeys(component.properties, [
    'modelType', 'hiddenSize', 'vocabSize', 'numLayers', 'maxPositionEmbeddings',
    'bosTokenId', 'eosTokenId', 'tieWordEmbeddings', 'embeddingNormalization', 'finalNormalization',
  ], 'text-decoder properties');
  requirePositiveInteger(component.properties.hiddenSize, 'text hiddenSize');
  requirePositiveInteger(component.properties.vocabSize, 'text vocabSize');
  const numLayers = requirePositiveInteger(component.properties.numLayers, 'text numLayers');
  const maxSequenceLength = requirePositiveInteger(
    component.properties.maxPositionEmbeddings,
    'text maxPositionEmbeddings'
  );
  requireNonNegativeInteger(component.properties.bosTokenId, 'text bosTokenId');
  requireNonNegativeInteger(component.properties.eosTokenId, 'text eosTokenId');

  const schedule = requireNode(
    modelIR.blockSchedules,
    (candidate) => candidate.componentId === component.id,
    'text block schedule'
  );
  if (schedule.blocks.length !== numLayers
    || schedule.blocks.some((block, index) => block.index !== index)) {
    throw new Error('Text block schedule must be contiguous and match component numLayers.');
  }
  const classById = new Map(modelIR.blockClasses.map((blockClass) => [blockClass.id, blockClass]));
  const blockClasses = [...new Set(schedule.blocks.map((block) => block.blockClassId))]
    .map((id) => classById.get(id));
  if (blockClasses.some((blockClass) => !blockClass)) throw new Error('Text block schedule references an absent block class.');
  const blockContract = validateBlockContracts(blockClasses);

  const stateSpaces = modelIR.stateSpaces.filter((state) => state.persistence === 'session');
  requireEqual(stateSpaces.map((state) => state.kind).sort(), ['kv'], 'text persistent state kinds');
  const kvState = stateSpaces[0];
  requireEqual(kvState.contract, {
    numKvHeads: blockContract.geometry.numKvHeads,
    headDim: blockContract.headDim,
    localWindow: blockContract.slidingWindow,
    maxSequenceLength,
  }, 'KV state contract');

  const outputHead = requireNode(
    modelIR.outputHeads,
    (candidate) => candidate.componentId === component.id && candidate.kind === 'causal-lm',
    'causal LM output head'
  );
  requireExactKeys(outputHead.properties, [
    'vocabSize', 'tieWordEmbeddings', 'preSoftcapMultiplier', 'finalLogitSoftcap', 'operationOrder',
  ], 'causal LM output-head properties');
  requireEqual(outputHead.properties.vocabSize, component.properties.vocabSize, 'output-head vocab size');

  const boundFacts = requireAcceptedFacts(modelIR, [
    ...component.factRefs,
    ...schedule.factRefs,
    ...entryPoint.factRefs,
    ...outputHead.factRefs,
    ...kvState.factRefs,
    ...blockClasses.flatMap((blockClass) => blockClass.factRefs),
  ], 'Semantic text lowering');
  const inference = buildInference({
    component,
    schedule,
    classById,
    blockContract,
    outputHead,
    chatTemplate: recipe.chatTemplate,
  });
  validateRequiredInferenceFields(clone(inference), modelId);

  const dispositions = [{
    kind: 'semantic-fact-closure',
    entryPointId: recipe.entryPointId,
    disposition: 'accepted',
    ...boundFacts,
  }, {
    kind: 'mechanism-template-policy',
    template: recipe.template,
    templateDigest: digest(template),
    disposition: 'accepted',
    author: clone(recipe.author),
    rationale: recipe.templateRationale,
  }, {
    kind: 'pack-support-scope',
    sourceTopology: modelIR.supportScope.sourceTopology,
    loweredEntryPoints: [recipe.entryPointId],
    unloweredEntryPoints: modelIR.entryPoints
      .filter((candidate) => candidate.id !== recipe.entryPointId)
      .map((candidate) => candidate.id)
      .sort(),
    disposition: 'accepted',
  }, {
    kind: 'reproducible-conversion-identity',
    conversion: clone(recipe.conversion),
    disposition: 'accepted',
    rationale: 'Forge freezes conversion identity before execution so rebuilds do not inherit wall-clock manifest drift.',
  }];
  if (typeof recipe.templateRationale !== 'string' || !recipe.templateRationale.trim()) {
    throw new Error('Semantic manifest lowering requires templateRationale.');
  }

  const config = {
    modelType: recipe.runtimeModelType,
    quantization: clone(template.quantization),
    largeWeights: clone(template.largeWeights),
    session: clone(template.session),
    execution: clone(template.execution),
    manifest: {
      hashAlgorithm: 'sha256',
      artifactIdentity: {
        sourceCheckpointId: modelIR.sourceIdentity.checkpointId,
        sourceRepo: modelIR.sourceIdentity.repository,
        sourceRevision: modelIR.sourceIdentity.revision,
        artifactCompleteness: 'complete',
      },
      conversion: clone(recipe.conversion),
      eosTokenId: component.properties.eosTokenId,
    },
    inference,
    output: {
      baseDir: recipe.outputBaseDir,
      modelBaseId: modelId,
      textOnly: true,
      fast: false,
    },
  };
  if (typeof config.output.baseDir !== 'string' || !config.output.baseDir.trim()) {
    throw new Error('Semantic manifest lowering requires outputBaseDir.');
  }
  config.session.kvcache.maxSeqLen = maxSequenceLength;
  bindConservativeSessionPolicy(config.session, dispositions);
  bindKernelDigests(config.execution, dispositions);
  expandExecutionV1(config.execution);

  const packModelIR = bindPackModelIR(modelIR, recipe.entryPointId, modelId);
  const sourceModelIRHash = digest(modelIR);
  const modelIRHash = digest(packModelIR);
  const conversionConfigDigest = digest(config);
  return Object.freeze({
    schema: 'doppler.semantic-manifest-lowering-receipt/v1',
    modelId,
    requestedModelId,
    entryPointId: recipe.entryPointId,
    sourceModelIRHash,
    modelIRHash,
    modelIR: packModelIR,
    author: clone(recipe.author),
    template: recipe.template,
    generatedCandidates: Number(recipe.candidateAudit?.generated || 1),
    rejectedCandidates: clone(recipe.candidateAudit?.rejected || []),
    acceptedCandidateId: recipe.candidateAudit?.acceptedCandidateId ?? 'conservative-text-manifest',
    dispositions,
    unresolvedFacts: [],
    conversionConfigDigest,
    conversionConfig: config,
  });
}
