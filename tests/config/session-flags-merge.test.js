import assert from 'node:assert/strict';
import { mergeConfig } from '../../src/config/merge.js';

// Minimal manifest with just enough inference structure to pass merge.
function buildManifest(sessionOverrides = {}, largeWeightsOverrides = undefined) {
  return {
    modelId: 'session-flags-witness',
    inference: {
      attention: {
        queryPreAttnScalar: 1,
        attentionBias: false,
        attnLogitSoftcapping: null,
        slidingWindow: 4096,
        queryKeyNorm: false,
        valueNorm: false,
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
      ffn: { activation: 'gelu', gatedActivation: false, branchMode: 'auto', swigluLimit: null },
      rope: {
        ropeTheta: 1000000, ropeLocalTheta: null, ropeFrequencyBaseDim: null,
        ropeLocalFrequencyBaseDim: null, ropeScalingType: null, ropeScalingFactor: null,
        ropeLocalScalingType: null, ropeLocalScalingFactor: null, yarnBetaFast: null,
        yarnBetaSlow: null, yarnOriginalMaxPos: null, ropeLocalYarnBetaFast: null,
        ropeLocalYarnBetaSlow: null, ropeLocalYarnOriginalMaxPos: null,
      },
      output: {
        finalLogitSoftcapping: null, tieWordEmbeddings: false, scaleEmbeddings: false,
        embeddingTranspose: false, embeddingVocabSize: 0, embeddingPostprocessor: null,
      },
      session: sessionOverrides,
      ...(largeWeightsOverrides !== undefined ? { largeWeights: largeWeightsOverrides } : {}),
      pipeline: null,
      layerPattern: null,
      chatTemplate: { type: 'gemma', enabled: true },
    },
    architecture: { headDim: 64, maxSeqLen: 2048 },
  };
}

// Case 1: manifest sets useFlashPrefillAttention=true, no runtime override → merged value is true, source=manifest.
{
  const merged = mergeConfig(
    buildManifest({ useFlashPrefillAttention: true }),
    {}
  );
  assert.equal(merged.inference.session.useFlashPrefillAttention, true);
  assert.equal(merged._sources.get('inference.session.useFlashPrefillAttention'), 'manifest');
}

// Case 2: manifest sets useFlashPrefillAttention=true, runtime overrides to false → runtime wins, source=runtime.
{
  const merged = mergeConfig(
    buildManifest({ useFlashPrefillAttention: true }),
    { session: { useFlashPrefillAttention: false } }
  );
  assert.equal(merged.inference.session.useFlashPrefillAttention, false);
  assert.equal(merged._sources.get('inference.session.useFlashPrefillAttention'), 'runtime');
}

// Case 3: manifest does not set prefill chunk fields, runtime sets them → runtime wins.
{
  const merged = mergeConfig(
    buildManifest({}),
    { session: { prefillChunkSubmitMode: 'async', prefillTokenChunkSize: 128 } }
  );
  assert.equal(merged.inference.session.prefillChunkSubmitMode, 'async');
  assert.equal(merged._sources.get('inference.session.prefillChunkSubmitMode'), 'runtime');
  assert.equal(merged.inference.session.prefillTokenChunkSize, 128);
  assert.equal(merged._sources.get('inference.session.prefillTokenChunkSize'), 'runtime');
}

// Case 4: per-field overlay — manifest sets decodeLoop, runtime sets different session flag, both survive.
{
  const merged = mergeConfig(
    buildManifest({
      decodeLoop: { batchSize: 8, readbackInterval: 8 },
      useFlashPrefillAttention: true,
    }),
    {
      session: {
        retainQ4KMaterialization: true,
        useSandwichRMSNormPairFusion: true,
        usePostFfnNextInputRMSNormPairFusion: true,
        usePostAttnNormFusedGateUp: true,
        fusedFfnQ4K: {
          decode: {
            pipelineConstants: {
              COLS_PER_WG: 64,
              THREADS_PER_COL: 4,
            },
          },
        },
        lmHeadArgmaxQ4K: {
          useFullBlockFastPath: true,
          colsPerWorkgroup: 128,
          threadsPerCol: 2,
        },
        attentionDecodeOnline: {
          workgroupSize: 128,
          useDirectContiguousKVLayout: true,
          useOutputGateFusion: true,
        },
        useLinearAttentionABProjectionFusion: true,
        useLinearAttentionQKVZProjectionFusion: true,
        useLinearAttentionFusedDecodeCore: true,
        useWideTileResidualFusion: true,
        useFusedRmsnormWideTile: true,
        useFusedQKVSplitQKNorm: true,
        useFusedQKVSplitQKNormRoPE: true,
        useLargeBatchF16F32FusedGateUp: true,
        skipEmbeddingKVCacheWrites: true,
      },
    }
  );
  assert.equal(merged.inference.session.decodeLoop.batchSize, 8, 'manifest decodeLoop preserved');
  assert.equal(merged.inference.session.useFlashPrefillAttention, true, 'manifest flag preserved');
  assert.equal(merged.inference.session.retainQ4KMaterialization, true, 'runtime flag applied');
  assert.equal(merged.inference.session.useSandwichRMSNormPairFusion, true, 'runtime sandwich RMSNorm pair flag applied');
  assert.equal(merged.inference.session.usePostFfnNextInputRMSNormPairFusion, true, 'runtime post-FFN next input RMSNorm pair flag applied');
  assert.equal(merged.inference.session.usePostAttnNormFusedGateUp, true, 'runtime post-attention norm fused gate/up flag applied');
  assert.deepEqual(
    merged.inference.session.fusedFfnQ4K,
    {
      decode: {
        pipelineConstants: {
          COLS_PER_WG: 64,
          THREADS_PER_COL: 4,
        },
      },
    },
    'runtime Q4K fused FFN constants applied'
  );
  assert.deepEqual(
    merged.inference.session.lmHeadArgmaxQ4K,
    {
      useFullBlockFastPath: true,
      colsPerWorkgroup: 128,
      threadsPerCol: 2,
    },
    'runtime Q4K LM-head argmax tuning applied'
  );
  assert.deepEqual(
    merged.inference.session.attentionDecodeOnline,
      {
        workgroupSize: 128,
        useDirectContiguousKVLayout: true,
        useOutputGateFusion: true,
      },
    'runtime online attention decode tuning applied'
  );
  assert.equal(merged.inference.session.useLinearAttentionABProjectionFusion, true, 'runtime linear-attention A/B projection fusion flag applied');
  assert.equal(merged.inference.session.useLinearAttentionQKVZProjectionFusion, true, 'runtime linear-attention QKV/Z projection fusion flag applied');
  assert.equal(merged.inference.session.useLinearAttentionFusedDecodeCore, true, 'runtime linear-attention fused decode core flag applied');
  assert.equal(merged.inference.session.useWideTileResidualFusion, true, 'runtime WideTile residual fusion flag applied');
  assert.equal(merged.inference.session.useFusedRmsnormWideTile, true, 'runtime fused RMSNorm WideTile flag applied');
  assert.equal(merged.inference.session.useFusedQKVSplitQKNorm, true, 'runtime fused split QK norm flag applied');
  assert.equal(merged.inference.session.useFusedQKVSplitQKNormRoPE, true, 'runtime fused split QK norm RoPE flag applied');
  assert.equal(merged.inference.session.useLargeBatchF16F32FusedGateUp, true, 'runtime large-batch f16/f32 fused gate/up flag applied');
  assert.equal(merged.inference.session.skipEmbeddingKVCacheWrites, true, 'runtime embedding KV skip flag applied');
  assert.equal(merged._sources.get('inference.session.decodeLoop'), 'manifest');
  assert.equal(merged._sources.get('inference.session.useFlashPrefillAttention'), 'manifest');
  assert.equal(merged._sources.get('inference.session.retainQ4KMaterialization'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useSandwichRMSNormPairFusion'), 'runtime');
  assert.equal(merged._sources.get('inference.session.usePostFfnNextInputRMSNormPairFusion'), 'runtime');
  assert.equal(merged._sources.get('inference.session.usePostAttnNormFusedGateUp'), 'runtime');
  assert.equal(merged._sources.get('inference.session.fusedFfnQ4K'), 'runtime');
  assert.equal(merged._sources.get('inference.session.lmHeadArgmaxQ4K'), 'runtime');
  assert.equal(merged._sources.get('inference.session.attentionDecodeOnline'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useLinearAttentionABProjectionFusion'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useLinearAttentionQKVZProjectionFusion'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useLinearAttentionFusedDecodeCore'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useWideTileResidualFusion'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useFusedRmsnormWideTile'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useFusedQKVSplitQKNorm'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useFusedQKVSplitQKNormRoPE'), 'runtime');
  assert.equal(merged._sources.get('inference.session.useLargeBatchF16F32FusedGateUp'), 'runtime');
  assert.equal(merged._sources.get('inference.session.skipEmbeddingKVCacheWrites'), 'runtime');
}

// Case 5: inference.largeWeights is merged as a sibling of session (not nested under session).
{
  const merged = mergeConfig(
    buildManifest({}, { gpuResidentOverrides: ['a.b.weight'] }),
    {}
  );
  assert.deepEqual(merged.inference.largeWeights.gpuResidentOverrides, ['a.b.weight']);
  assert.equal(merged._sources.get('inference.largeWeights.gpuResidentOverrides'), 'manifest');
}

// Case 6: runtime largeWeights override wins over manifest.
{
  const merged = mergeConfig(
    buildManifest({}, { gpuResidentOverrides: ['manifest.weight'] }),
    { largeWeights: { gpuResidentOverrides: ['runtime.weight'] } }
  );
  assert.deepEqual(merged.inference.largeWeights.gpuResidentOverrides, ['runtime.weight']);
  assert.equal(merged._sources.get('inference.largeWeights.gpuResidentOverrides'), 'runtime');
}

console.log('session-flags-merge.test: ok');
