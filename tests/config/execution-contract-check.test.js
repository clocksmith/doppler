import assert from 'node:assert/strict';

import {
  buildExecutionContractArtifact,
  extractExecutionContractFacts,
  validateExecutionContractFacts,
  validateManifestExecutionContract,
} from '../../src/config/execution-contract-check.js';
import { buildKernelRefFromKernelEntry } from '../../src/config/kernels/kernel-ref.js';
import { validateManifest } from '../../src/formats/rdrr/validation.js';

function kernelRef(kernel, entry = 'main') {
  return buildKernelRefFromKernelEntry(kernel, entry);
}

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
      schema: null,
      presetId: 'gemma3',
      defaultKernelPath: 'gemma3-q4k-dequant-f32a-online',
      layerPattern: {
        type: 'every_n',
        period: 6,
        offset: 0,
      },
      sessionDefaults: {
        compute: {
          defaults: {
            activationDtype: 'f32',
            mathDtype: 'f32',
            accumDtype: 'f32',
            outputDtype: 'f32',
          },
          kernelProfiles: [
            {
              kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
            },
          ],
        },
        kvcache: {
          layout: 'paged',
          kvDtype: 'f16',
        },
        decodeLoop: {
          batchSize: 1,
          stopCheckMode: 'batch',
          readbackInterval: 1,
          disableCommandBatching: true,
        },
      },
      execution: {
        steps: [
          {
            id: 'attn',
            phase: 'both',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16kv.wgsl',
            entry: 'main',
            kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
          },
        ],
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
  conflictingManifest.inference.sessionDefaults.kvcache.layout = 'bdpa';
  conflictingManifest.inference.sessionDefaults.decodeLoop = {
    ...(conflictingManifest.inference.sessionDefaults.decodeLoop ?? {}),
  };
  conflictingManifest.inference.sessionDefaults.decodeLoop.batchSize = 16;
  delete conflictingManifest.inference.sessionDefaults.decodeLoop.disableCommandBatching;

  const facts = extractExecutionContractFacts(conflictingManifest);
  assert.equal(facts.session.layout, 'bdpa');
  assert.equal(facts.session.decodeBatchSize, 16);

  const executionContract = validateExecutionContractFacts(facts);
  assert.equal(executionContract.ok, false);
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('decode-only') && message.includes('both attention')
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
      message.includes('decode-only') && message.includes('both attention')
    )
  );
}

{
  const executionContractManifest = {
    modelId: 'execution-contract-artifact',
    modelType: 'transformer',
    architecture: {
      headDim: 128,
      maxSeqLen: 4096,
      numLayers: 2,
    },
    inference: {
      sessionDefaults: {
        compute: {
          defaults: {
            activationDtype: 'f16',
            mathDtype: 'f16',
            accumDtype: 'f32',
            outputDtype: 'f16',
          },
          kernelProfiles: [
            {
              kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
            },
          ],
        },
        kvcache: {
          layout: 'paged',
          kvDtype: 'f16',
        },
        decodeLoop: {
          batchSize: 4,
          stopCheckMode: 'batch',
          readbackInterval: 1,
        },
      },
      execution: {
        steps: [
          {
            id: 'attn',
            phase: 'both',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            entry: 'main',
            kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
          },
        ],
      },
    },
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
