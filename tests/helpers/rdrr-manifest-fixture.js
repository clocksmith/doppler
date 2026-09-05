import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { createExecutionContractSession } from './execution-v1-fixtures.js';

// A format fixture, not a qualified model or a dependency on models/local.
export function createRDRRManifestFixture() {
  return {
    version: 1,
    modelId: 'synthetic-transformer',
    modelType: 'transformer',
    quantization: 'F16',
    hashAlgorithm: 'sha256',
    eos_token_id: 1,
    architecture: {
      numLayers: 1, hiddenSize: 4, intermediateSize: 8,
      numAttentionHeads: 1, numKeyValueHeads: 1, headDim: 4,
      vocabSize: 8, maxSeqLen: 16,
    },
    inference: {
      ...structuredClone(DEFAULT_MANIFEST_INFERENCE),
      session: createExecutionContractSession(),
    },
    shards: [{ index: 0, filename: 'shard_00000.bin', size: 16, offset: 0, hash: 'a'.repeat(64) }],
    totalSize: 16,
    tensors: {},
    groups: {},
    metadata: { source: 'synthetic-test-fixture' },
  };
}
