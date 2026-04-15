import assert from 'node:assert/strict';
import { buildGemma4LiteRTPackedFixture } from '../helpers/gemma4-litert-fixture.js';

const {
  inferLiteRTRowwiseLayout,
  resolveLiteRTPackageParsedArtifact,
  resolveGemma4AttentionHeadDim,
} = await import('../../src/tooling/litert-package-runtime.js');

assert.deepEqual(
  inferLiteRTRowwiseLayout({ dtypeId: 9, size: 8, name: 'int8-rowwise' }, [2, 4]),
  { sourceDtype: 'INT8', storageEncoding: 'signed' }
);

assert.deepEqual(
  inferLiteRTRowwiseLayout({ dtypeId: 17, size: 4, name: 'int4-rowwise' }, [2, 4]),
  { sourceDtype: 'INT4', storageEncoding: 'offset_binary' }
);

assert.deepEqual(
  inferLiteRTRowwiseLayout({ dtypeId: 9, size: 2, name: 'int2-packed-in-int8' }, [2, 4]),
  { sourceDtype: 'INT2', storageEncoding: 'offset_binary' }
);

assert.deepEqual(
  inferLiteRTRowwiseLayout({ dtypeId: 3, size: 8, name: 'uint8-packed-int4' }, [2, 8]),
  { sourceDtype: 'INT4', storageEncoding: 'offset_binary' }
);

assert.deepEqual(
  inferLiteRTRowwiseLayout(
    { dtypeId: 3, size: 8, name: 'uint8-packed-int4-signed' },
    [2, 8],
    'uint8-packed-int4-signed',
    { preferSignedPacked: true }
  ),
  { sourceDtype: 'INT4', storageEncoding: 'signed' }
);

assert.throws(
  () => inferLiteRTRowwiseLayout({ dtypeId: 9, size: 3, name: 'bad-packed-layout' }, [2, 4]),
  /does not match any supported packed layout/i
);

const gemma4RuntimeProfile = {
  architecture: {
    headDim: 256,
    globalHeadDim: 512,
  },
  manifestInference: {
    layerPattern: {
      type: 'every_n',
      period: 5,
      offset: 4,
    },
  },
};

assert.equal(resolveGemma4AttentionHeadDim(gemma4RuntimeProfile, 0), 256);
assert.equal(resolveGemma4AttentionHeadDim(gemma4RuntimeProfile, 4), 512);
assert.equal(resolveGemma4AttentionHeadDim(gemma4RuntimeProfile, 9), 512);
assert.equal(resolveGemma4AttentionHeadDim(gemma4RuntimeProfile, 10), 256);

const rawTaskBytes = buildGemma4LiteRTPackedFixture({ profileAligned: true });
const resolvedRawTask = await resolveLiteRTPackageParsedArtifact({
  sourceKind: 'litert-task',
  sourcePathForModelId: '/fixture/gemma-4-E2B-it-web.task',
  source: {
    name: 'gemma-4-E2B-it-web.task',
    size: rawTaskBytes.byteLength,
    async readRange(offset, length) {
      const start = Math.max(0, Math.floor(offset));
      const end = Math.min(rawTaskBytes.byteLength, start + Math.max(0, Math.floor(length)));
      return rawTaskBytes.slice(start, end);
    },
  },
});
assert.equal(
  resolvedRawTask.virtualFiles.some((entry) => entry.kind === 'tflite_model' && entry.path === 'gemma-4-E2B-it-web.task'),
  true
);
assert.equal(
  resolvedRawTask.virtualFiles.some((entry) => entry.kind === 'tokenizer_model' && entry.path === 'TOKENIZER_MODEL'),
  true
);
assert.equal(
  resolvedRawTask.virtualFiles.some((entry) => entry.kind === 'litert_metadata' && entry.path === 'METADATA'),
  true
);
assert.equal(resolvedRawTask.parsedArtifact.tokenizerModelPath, 'TOKENIZER_MODEL');
assert.equal(
  resolvedRawTask.parsedArtifact.auxiliaryFiles.some((entry) => entry.kind === 'tokenizer_model' && entry.path === 'TOKENIZER_MODEL'),
  true
);
assert.equal(
  resolvedRawTask.parsedArtifact.auxiliaryFiles.some((entry) => entry.kind === 'litert_metadata' && entry.path === 'METADATA'),
  true
);
const layer34EmbeddingTensor = resolvedRawTask.parsedArtifact.tensors.find(
  (tensor) => tensor.name === 'model.language_model.layers.34.embed_tokens_per_layer.weight'
);
assert.ok(layer34EmbeddingTensor, 'layer 34 per-layer embedding tensor should be present');
assert.equal(layer34EmbeddingTensor.sourceTransform.kind, 'litert_axis_dequant');
assert.equal(layer34EmbeddingTensor.sourceTransform.scaleCompanionDtype, 'UINT8');
assert.ok(
  Math.abs((layer34EmbeddingTensor.sourceTransform.scaleCompanionDequant?.scale ?? 0) - 0.01) < 1e-9
);
assert.equal(layer34EmbeddingTensor.sourceTransform.scaleCompanionDequant?.zeroPoint, 0);

console.log('litert-package-runtime.test: ok');
