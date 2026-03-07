import { resolveConfig, resolvePreset } from './loader.js';
import {
  chooseNullish,
  chooseDefinedWithSource,
  mergeExecutionPatchLists,
  mergeKernelPathPolicy,
  mergeLayeredShallowObjects,
  mergeShallowObject,
  replaceSubtree,
} from './merge-helpers.js';
import { mergeConfig } from './merge.js';
import { createDopplerConfig } from './schema/index.js';

function buildWitnessManifestForArchitecture() {
  return {
    modelId: 'merge-semantics-witness',
    modelType: 'transformer',
    architecture: {
      numLayers: 2,
      hiddenSize: 256,
      intermediateSize: 512,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      headDim: 64,
      vocabSize: 1024,
      maxSeqLen: 2048,
      ropeTheta: null,
      rmsNormEps: 1e-6,
    },
    config: {},
  };
}

function buildWitnessMergeManifest() {
  return {
    modelId: 'merge-overlay-witness',
    inference: {
      attention: {
        queryPreAttnScalar: 1,
        attentionBias: false,
        attnLogitSoftcapping: null,
        slidingWindow: 4096,
        queryKeyNorm: false,
        attentionOutputGate: false,
        causal: true,
      },
      normalization: {
        rmsNormEps: 1e-6,
        rmsNormWeightOffset: 0,
        postAttentionNorm: true,
        preFeedforwardNorm: true,
        postFeedforwardNorm: false,
      },
      ffn: {
        activation: 'gelu',
        gatedActivation: false,
        swigluLimit: null,
      },
      rope: {
        ropeTheta: 1000000,
        ropeLocalTheta: null,
        ropeScalingType: null,
        ropeScalingFactor: null,
        ropeLocalScalingType: null,
        ropeLocalScalingFactor: null,
        yarnBetaFast: null,
        yarnBetaSlow: null,
        yarnOriginalMaxPos: null,
        ropeLocalYarnBetaFast: null,
        ropeLocalYarnBetaSlow: null,
        ropeLocalYarnOriginalMaxPos: null,
      },
      output: {
        finalLogitSoftcapping: 30,
        tieWordEmbeddings: true,
        scaleEmbeddings: false,
        embeddingTranspose: false,
        embeddingVocabSize: 1024,
      },
      pipeline: 'decode-only',
      layerPattern: null,
      chatTemplate: {
        type: 'gemma',
        enabled: true,
      },
      defaultKernelPath: 'gemma3-f16-fused-f16a-online',
    },
    architecture: {
      headDim: 64,
      maxSeqLen: 2048,
    },
  };
}

function recordCheck(results, id, ok, detail, mode = 'actual') {
  results.push({ id, ok, detail, mode });
}

export function buildMergeContractArtifact() {
  const checks = [];
  const preset = resolvePreset('gemma3');
  const resolved = resolveConfig(buildWitnessManifestForArchitecture(), 'gemma3');
  recordCheck(
    checks,
    'loader.architecture.nullish_null_falls_through',
    resolved.architecture.ropeTheta === preset.architecture.ropeTheta,
    `resolved ropeTheta=${resolved.architecture.ropeTheta}, preset ropeTheta=${preset.architecture.ropeTheta}`
  );

  const mergedUndefined = mergeConfig(buildWitnessMergeManifest(), {});
  recordCheck(
    checks,
    'runtime.mergeConfig.defined_overlay_missing_falls_through',
    mergedUndefined.inference.defaultKernelPath === 'gemma3-f16-fused-f16a-online'
      && mergedUndefined._sources.get('inference.defaultKernelPath') === 'manifest',
    `value=${mergedUndefined.inference.defaultKernelPath}, source=${mergedUndefined._sources.get('inference.defaultKernelPath')}`
  );
  recordCheck(
    checks,
    'runtime.mergeConfig.pipeline_preserves_manifest_value',
    mergedUndefined.inference.pipeline === 'decode-only'
      && mergedUndefined._sources.get('inference.pipeline') === 'manifest',
    `value=${String(mergedUndefined.inference.pipeline)}, source=${mergedUndefined._sources.get('inference.pipeline')}`
  );

  const mergedNull = mergeConfig(buildWitnessMergeManifest(), {
    defaultKernelPath: null,
    chatTemplate: {
      enabled: null,
    },
  });
  recordCheck(
    checks,
    'runtime.mergeConfig.defined_overlay_preserves_null',
    mergedNull.inference.defaultKernelPath === null
      && mergedNull._sources.get('inference.defaultKernelPath') === 'runtime',
    `value=${mergedNull.inference.defaultKernelPath}, source=${mergedNull._sources.get('inference.defaultKernelPath')}`
  );
  recordCheck(
    checks,
    'runtime.inference.chatTemplate.spread_preserves_null',
    mergedNull.inference.chatTemplate.enabled === null
      && mergedNull._sources.get('inference.chatTemplate.enabled') === 'runtime',
    `value=${String(mergedNull.inference.chatTemplate.enabled)}`
  );

  const runtimeConfig = createDopplerConfig({
    runtime: {
      inference: {
        chatTemplate: {
          enabled: null,
        },
      },
    },
  });
  recordCheck(
    checks,
    'runtime.schema.chatTemplate.spread_preserves_null',
    runtimeConfig.runtime.inference.chatTemplate.enabled === null,
    `value=${String(runtimeConfig.runtime.inference.chatTemplate.enabled)}`
  );

  const isolatedConfigA = createDopplerConfig();
  isolatedConfigA.runtime.inference.compute.activationDtype = 'f32';
  const isolatedConfigB = createDopplerConfig();
  recordCheck(
    checks,
    'runtime.schema.defaults_are_isolated_per_instance',
    isolatedConfigB.runtime.inference.compute.activationDtype !== 'f32'
      && isolatedConfigA.runtime.inference.compute !== isolatedConfigB.runtime.inference.compute,
    `configA=${isolatedConfigA.runtime.inference.compute.activationDtype}, configB=${isolatedConfigB.runtime.inference.compute.activationDtype}`,
    'actual'
  );

  const calibrateConfig = createDopplerConfig({
    runtime: {
      shared: {
        tooling: {
          intent: 'calibrate',
        },
      },
    },
  });
  recordCheck(
    checks,
    'runtime.schema.calibrate_does_not_mutate_kernel_warmup_defaults',
    calibrateConfig.runtime.shared.kernelWarmup.prewarm === false,
    `prewarm=${String(calibrateConfig.runtime.shared.kernelWarmup.prewarm)}`,
    'actual'
  );

  const overlaySources = new Map();
  const chosenRuntimeValue = chooseDefinedWithSource(
    'inference.defaultKernelPath',
    null,
    'manifest-path',
    overlaySources
  );
  recordCheck(
    checks,
    'runtime.mergeHelpers.chooseDefinedWithSource.runtime_marks_source',
    chosenRuntimeValue === null && overlaySources.get('inference.defaultKernelPath') === 'runtime',
    `value=${String(chosenRuntimeValue)}, source=${overlaySources.get('inference.defaultKernelPath')}`,
    'actual'
  );

  const manifestSources = new Map();
  const chosenManifestValue = chooseDefinedWithSource(
    'inference.defaultKernelPath',
    undefined,
    'manifest-path',
    manifestSources
  );
  recordCheck(
    checks,
    'runtime.mergeHelpers.chooseDefinedWithSource.manifest_marks_source',
    chosenManifestValue === 'manifest-path' && manifestSources.get('inference.defaultKernelPath') === 'manifest',
    `value=${String(chosenManifestValue)}, source=${manifestSources.get('inference.defaultKernelPath')}`,
    'actual'
  );

  const executionPatchBase = {
    set: [{ op: 'seed' }],
    remove: ['old_step'],
    add: [{ id: 'new_step' }],
  };
  const executionPatchOverride = {
    set: null,
    remove: [],
    add: undefined,
  };
  const mergedExecutionPatch = {
    ...mergeExecutionPatchLists(executionPatchBase, executionPatchOverride),
  };
  recordCheck(
    checks,
    'runtime.inference.executionPatch.nullish_null_falls_through',
    mergedExecutionPatch.set === executionPatchBase.set
      && mergedExecutionPatch.add === executionPatchBase.add
      && Array.isArray(mergedExecutionPatch.remove)
      && mergedExecutionPatch.remove.length === 0,
    `setLength=${mergedExecutionPatch.set.length}, removeLength=${mergedExecutionPatch.remove.length}, addLength=${mergedExecutionPatch.add.length}`,
    'actual'
  );

  const sessionBase = {
    kvcache: {
      layout: 'paged',
      maxSeqLen: 8192,
      kvDtype: 'f16',
    },
    decodeLoop: {
      batchSize: 16,
      disableCommandBatching: false,
    },
  };
  const sessionOverride = {
    kvcache: {
      layout: 'tiered',
    },
    decodeLoop: {
      batchSize: 1,
    },
  };
  const mergedSession = {
    kvcache: replaceSubtree(sessionOverride.kvcache, sessionBase.kvcache),
    decodeLoop: replaceSubtree(sessionOverride.decodeLoop, sessionBase.decodeLoop),
  };
  recordCheck(
    checks,
    'runtime.inference.session.subtree_override_replaces_base',
    mergedSession.kvcache.layout === 'tiered'
      && mergedSession.kvcache.maxSeqLen === undefined
      && mergedSession.decodeLoop.batchSize === 1
      && mergedSession.decodeLoop.disableCommandBatching === undefined,
    `kvcacheKeys=${Object.keys(mergedSession.kvcache).join(',')}; decodeLoopKeys=${Object.keys(mergedSession.decodeLoop).join(',')}`,
    'actual'
  );

  const mergedChatTemplate = mergeShallowObject(
    { type: 'base', enabled: true },
    { enabled: null }
  );
  recordCheck(
    checks,
    'runtime.mergeShallowObject.spread_preserves_null',
    mergedChatTemplate.enabled === null && mergedChatTemplate.type === 'base',
    `type=${mergedChatTemplate.type}, enabled=${String(mergedChatTemplate.enabled)}`,
    'actual'
  );

  let invalidShallowOverrideError = null;
  try {
    mergeShallowObject(
      { type: 'base', enabled: true },
      null
    );
  } catch (error) {
    invalidShallowOverrideError = error;
  }
  recordCheck(
    checks,
    'runtime.mergeShallowObject.invalid_explicit_override_fails_closed',
    invalidShallowOverrideError instanceof Error
      && /shallow object overrides must be plain objects/.test(invalidShallowOverrideError.message),
    `error=${invalidShallowOverrideError?.message ?? 'none'}`,
    'actual'
  );

  const layeredAttention = mergeLayeredShallowObjects(
    { slidingWindow: 4096, attentionBias: false },
    { slidingWindow: 2048 },
    { slidingWindow: null }
  );
  recordCheck(
    checks,
    'loader.mergeLayeredShallowObjects.spread_preserves_null',
    layeredAttention.slidingWindow === null && layeredAttention.attentionBias === false,
    `slidingWindow=${String(layeredAttention.slidingWindow)}, attentionBias=${String(layeredAttention.attentionBias)}`,
    'actual'
  );

  const mergedKernelPathPolicy = mergeKernelPathPolicy(
    {
      mode: 'locked',
      sourceScope: ['model', 'manifest'],
      allowSources: ['model', 'manifest'],
      onIncompatible: 'error',
    },
    {
      allowSources: ['config', 'execution-v0'],
      onIncompatible: 'remap',
    }
  );
  recordCheck(
    checks,
    'runtime.kernelPathPolicy.source_scope_mirrors_allow_sources',
    Array.isArray(mergedKernelPathPolicy.sourceScope)
      && Array.isArray(mergedKernelPathPolicy.allowSources)
      && mergedKernelPathPolicy.sourceScope.length === 2
      && mergedKernelPathPolicy.sourceScope[0] === 'config'
      && mergedKernelPathPolicy.allowSources[1] === 'execution-v0'
      && mergedKernelPathPolicy.onIncompatible === 'remap',
    `sourceScope=${JSON.stringify(mergedKernelPathPolicy.sourceScope)}, allowSources=${JSON.stringify(mergedKernelPathPolicy.allowSources)}`,
    'actual'
  );

  const runtimeConfigWithKernelPathPolicy = createDopplerConfig({
    runtime: {
      inference: {
        kernelPathPolicy: {
          allowSources: ['config', 'execution-v0'],
        },
      },
    },
  });
  recordCheck(
    checks,
    'runtime.schema.kernelPathPolicy.helper_is_used',
    Array.isArray(runtimeConfigWithKernelPathPolicy.runtime.inference.kernelPathPolicy.sourceScope)
      && runtimeConfigWithKernelPathPolicy.runtime.inference.kernelPathPolicy.sourceScope[0] === 'config'
      && runtimeConfigWithKernelPathPolicy.runtime.inference.kernelPathPolicy.allowSources[1] === 'execution-v0',
    `policy=${JSON.stringify(runtimeConfigWithKernelPathPolicy.runtime.inference.kernelPathPolicy)}`,
    'actual'
  );

  const errors = checks.filter((entry) => !entry.ok).map((entry) => `[MergeContract] ${entry.id}: ${entry.detail}`);

  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: errors.length === 0,
    checks,
    errors,
  };
}
