import assert from 'node:assert/strict';

import {
  buildExecutionContractArtifact,
  extractExecutionContractFacts,
  validateExecutionContractFacts,
  validateManifestExecutionContract,
} from '../../src/config/execution-contract-check.js';
import { EXECUTION_V1_SCHEMA_ID } from '../../src/config/schema/index.js';
import { validateManifest } from '../../src/formats/rdrr/validation.js';

const D = (char) => `sha256:${char.repeat(64)}`;

function buildExecutionContractFixtureManifest() {
  return {
    version: 1,
    modelId: 'execution-contract-fixture',
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    quantizationInfo: {
      weights: 'q4k',
    },
    hashAlgorithm: 'sha256',
    eos_token_id: 1,
    totalSize: 1,
    shards: [
      {
        index: 0,
        size: 1,
        hash: '0'.repeat(64),
        filename: 'shard_00000.bin',
        offset: 0,
      },
    ],
    architecture: {
      numLayers: 2,
      hiddenSize: 256,
      intermediateSize: 512,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      headDim: 128,
      vocabSize: 1024,
      maxSeqLen: 4096,
      ropeTheta: 1000000,
    },
    tensors: {},
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      attention: {
        queryPreAttnScalar: 256,
        attnLogitSoftcapping: null,
        slidingWindow: null,
        queryKeyNorm: false,
        valueNorm: false,
        causal: true,
        attentionBias: false,
        attentionOutputGate: false,
      },
      normalization: {
        rmsNormEps: 1e-6,
        rmsNormWeightOffset: false,
        postAttentionNorm: false,
        preFeedforwardNorm: false,
        postFeedforwardNorm: false,
      },
      ffn: {
        activation: 'gelu',
        gatedActivation: true,
        useDoubleWideMlp: false,
        swigluLimit: null,
      },
      rope: {
        ropeTheta: 1000000,
        ropeLocalTheta: null,
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
      },
      output: {
        finalLogitSoftcapping: null,
        tieWordEmbeddings: true,
        scaleEmbeddings: false,
        embeddingTranspose: false,
        embeddingVocabSize: null,
        embeddingPostprocessor: null,
      },
      layerPattern: {
        type: 'every_n',
        globalPattern: null,
        period: 6,
        offset: 0,
        layerTypes: null,
      },
      chatTemplate: {
        type: 'gemma',
        enabled: true,
      },
      session: {
        compute: {
          defaults: {
            activationDtype: 'f32',
            mathDtype: 'f32',
            accumDtype: 'f32',
            outputDtype: 'f32',
          },
        },
        kvcache: {
          layout: 'paged',
          kvDtype: 'f16',
          tiering: {
            mode: 'off',
          },
        },
        decodeLoop: {
          batchSize: 1,
          stopCheckMode: 'batch',
          readbackInterval: 1,
          disableCommandBatching: true,
        },
      },
      execution: {
        kernels: {
          attn: {
            kernel: 'attention_streaming_f16kv.wgsl',
            entry: 'main',
            digest: D('a'),
          },
        },
        preLayer: [],
        decode: [
          ['attention', 'attn'],
        ],
        prefill: [
          ['attention', 'attn'],
        ],
        postLayer: [],
        policies: {
          unsupportedPrecision: 'error',
          dtypeTransition: 'require_cast_step',
          unresolvedKernel: 'error',
        },
      },
    },
  };
}

const translateGemmaManifest = buildExecutionContractFixtureManifest();

{
  const result = validateManifestExecutionContract(translateGemmaManifest);
  assert.equal(result.ok, true);
  assert.deepEqual(
    result.checks.map((entry) => entry.ok),
    [true, true]
  );
}

{
  const conflictingManifest = structuredClone(translateGemmaManifest);
  conflictingManifest.inference.session.kvcache.layout = 'bdpa';
  conflictingManifest.inference.session.decodeLoop = {
    ...(conflictingManifest.inference.session.decodeLoop ?? {}),
    disableCommandBatching: false,
  };
  conflictingManifest.inference.session.decodeLoop.batchSize = 16;

  const facts = extractExecutionContractFacts(conflictingManifest);
  assert.equal(facts.session.layout, 'bdpa');
  assert.equal(facts.session.decodeBatchSize, 16);

  const executionContract = validateExecutionContractFacts(facts);
  assert.equal(executionContract.ok, false);
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('decode-only') && message.includes('prefill attention')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('disableCommandBatching=true')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('batchSize <= 1')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('maxSeqLen <= 2048')
    )
  );

  const validation = validateManifest(conflictingManifest);
  assert.equal(validation.valid, false);
  assert.ok(
    validation.errors.some((message) =>
      message.includes('decode-only') && message.includes('prefill attention')
    )
  );
}

{
  const turboquantManifest = structuredClone(translateGemmaManifest);
  turboquantManifest.inference.session.kvcache = {
    layout: 'contiguous',
    kvDtype: 'f16',
    maxSeqLen: 2048,
    tiering: {
      mode: 'off',
    },
    quantization: {
      mode: 'turboquant',
      bitWidth: 4,
      prodMode: false,
    },
  };

  const facts = extractExecutionContractFacts(turboquantManifest);
  assert.equal(facts.session.kvLen, 2048, 'session maxSeqLen should clamp execution-contract KV len');
  assert.equal(facts.session.contiguousQuantMode, 'turboquant');

  const evaluation = validateExecutionContractFacts(facts);
  assert.equal(evaluation.ok, true);
}

{
  const invalidTurboquantManifest = structuredClone(translateGemmaManifest);
  invalidTurboquantManifest.inference.session.kvcache = {
    layout: 'contiguous',
    kvDtype: 'f16',
    maxSeqLen: 4096,
    tiering: {
      mode: 'off',
    },
    quantization: {
      mode: 'turboquant',
      bitWidth: 4,
      prodMode: false,
    },
  };

  const evaluation = validateManifestExecutionContract(invalidTurboquantManifest);
  assert.equal(evaluation.ok, false);
  assert.ok(
    evaluation.errors.some((message) =>
      message.includes('quantization.mode="turboquant"') && message.includes('effective maxSeqLen <= 2048')
    )
  );
}

{
  const unsupportedOutlierManifest = structuredClone(translateGemmaManifest);
  unsupportedOutlierManifest.inference.session.kvcache = {
    layout: 'contiguous',
    kvDtype: 'f16',
    maxSeqLen: 2048,
    tiering: {
      mode: 'off',
    },
    quantization: {
      mode: 'turboquant_outlier',
      bitWidth: 4,
      prodMode: false,
    },
  };

  assert.throws(
    () => validateManifestExecutionContract(unsupportedOutlierManifest),
    /turboquant_outlier.*not supported/,
    'turboquant_outlier must fail closed during execution-contract extraction'
  );
}

{
  const tieredTurboquantManifest = structuredClone(translateGemmaManifest);
  tieredTurboquantManifest.inference.session.kvcache = {
    layout: 'tiered',
    kvDtype: 'f16',
    maxSeqLen: 2048,
    tiering: {
      mode: 'turboquant',
      hotWindow: 256,
      coldPageSize: 64,
      coldDtype: 'f16',
      compression: {
        mode: 'turboquant',
        blockSize: 1,
        bitWidth: 4,
        prodMode: false,
      },
      gating: {
        mode: 'force_on',
      },
    },
  };

  const facts = extractExecutionContractFacts(tieredTurboquantManifest);
  assert.equal(facts.session.coldQuantMode, 'turboquant');

  const evaluation = validateExecutionContractFacts(facts);
  assert.equal(evaluation.ok, true);
}

{
  const executionContractManifest = buildExecutionContractFixtureManifest();
  executionContractManifest.modelId = 'execution-contract-artifact';
  executionContractManifest.inference.session.compute.defaults = {
    activationDtype: 'f16',
    mathDtype: 'f16',
    accumDtype: 'f32',
    outputDtype: 'f16',
  };
  executionContractManifest.inference.session.decodeLoop = {
    batchSize: 4,
    stopCheckMode: 'batch',
    readbackInterval: 1,
    disableCommandBatching: false,
  };
  executionContractManifest.inference.execution.kernels.attn = {
    kernel: 'attention_streaming_f16.wgsl',
    entry: 'main',
    digest: D('b'),
  };

  const artifact = buildExecutionContractArtifact(executionContractManifest);
  assert.equal(artifact?.ok, true);
  assert.ok(
    artifact?.checks.some((entry) => entry.id === 'execution-contract-artifact.steps' && entry.ok)
  );
  assert.ok(
    artifact?.checks.some((entry) => entry.id === 'execution-contract-artifact.session' && entry.ok)
  );
}

console.log('execution-contract-check.test: ok');
