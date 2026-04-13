import assert from 'node:assert/strict';

const {
  inferLiteRTRowwiseLayout,
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

console.log('litert-package-runtime.test: ok');
