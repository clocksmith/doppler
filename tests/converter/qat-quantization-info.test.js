import assert from 'node:assert/strict';

const {
  buildQuantizationInfo,
  resolveEffectiveQuantizationInfo,
  resolveManifestQuantization,
} = await import('../../src/converter/quantization-info.js');
const { transformTensorBytes } = await import('../../src/converter/core.js');
const { loadTensorToCPU } = await import('../../src/loader/tensors/tensor-loader.js');

{
  const info = buildQuantizationInfo(
    {
      quantization: {
        weights: 'q4_0',
        embeddings: 'q4_0',
        lmHead: 'q4_0',
        sourceTrainingQuantization: 'qat',
        sourceQuantizationTarget: 'q4_0',
      },
    },
    'F16',
    'F16',
    'F16'
  );

  assert.equal(info.weights, 'q4_0');
  assert.equal(info.embeddings, 'q4_0');
  assert.equal(info.lmHead, 'q4_0');
  assert.equal(info.sourceTrainingQuantization, 'qat');
  assert.equal(info.sourceQuantizationTarget, 'q4_0');
  assert.equal(info.variantTag, 'q4_0');
  assert.equal(resolveManifestQuantization('q4_0', 'F16'), 'Q4_0');
}

{
  const info = buildQuantizationInfo(
    {
      quantization: {
        weights: 'w4a16',
        embeddings: 'f16',
        lmHead: 'w4a16',
        sourceTrainingQuantization: 'quantization-aware-training',
        sourceQuantizationTarget: 'w4a16-ct',
      },
    },
    'F16',
    'F16',
    'F16'
  );

  assert.equal(info.weights, 'w4a16');
  assert.equal(info.embeddings, 'f16');
  assert.equal(info.lmHead, 'w4a16');
  assert.equal(info.sourceTrainingQuantization, 'qat');
  assert.equal(info.sourceQuantizationTarget, 'w4a16');
  assert.equal(resolveManifestQuantization('w4a16', 'F16'), 'W4A16');
}

{
  const info = buildQuantizationInfo(
    {
      quantization: {
        weights: 'w4a16',
        embeddings: 'f16',
        lmHead: 'f16',
        sourceTrainingQuantization: 'qat',
        sourceQuantizationTarget: 'w4a16',
      },
    },
    'F16',
    'F16',
    'F16'
  );

  assert.equal(info.weights, 'w4a16');
  assert.equal(info.embeddings, 'f16');
  assert.equal(info.lmHead, 'f16');
  assert.equal(info.sourceTrainingQuantization, 'qat');
  assert.equal(info.sourceQuantizationTarget, 'w4a16');
}

{
  const info = buildQuantizationInfo(
    {
      quantization: {
        weights: 'wna8o8',
        embeddings: 'f16',
        lmHead: 'wna8o8',
        sourceTrainingQuantization: 'qat',
        sourceQuantizationTarget: 'wna8o8',
      },
    },
    'F16',
    'F16',
    'F16'
  );

  assert.equal(info.weights, 'wna8o8');
  assert.equal(info.lmHead, 'wna8o8');
  assert.equal(info.sourceQuantizationTarget, 'wNa8o8');
  assert.equal(resolveManifestQuantization('wna8o8', 'F16'), 'WNA8O8');
}

{
  const info = buildQuantizationInfo(
    {
      output: {
        modelBaseId: 'google-gemma-4-12b-it-qat-q4-0',
      },
      quantization: {
        weights: 'f16',
        embeddings: 'f16',
        lmHead: 'f16',
      },
    },
    'F16',
    'F16',
    'F16'
  );

  assert.equal(info.sourceTrainingQuantization, undefined);
  assert.equal(info.sourceQuantizationTarget, undefined);
  assert.equal(info.weights, 'f16');
}

{
  assert.throws(
    () => buildQuantizationInfo(
      {
        quantization: {
          weights: 'q4_0',
          embeddings: 'q4_0',
          lmHead: 'f16',
          sourceTrainingQuantization: 'qat',
          sourceQuantizationTarget: 'q4_0',
        },
      },
      'F16',
      'F16',
      'F16'
    ),
    /requires quantizationInfo\.lmHead="q4_0"/
  );
}

{
  assert.throws(
    () => buildQuantizationInfo(
      {
        quantization: {
          weights: 'f16',
          embeddings: 'f16',
          lmHead: 'f16',
          sourceQuantizationTarget: 'q4_0',
        },
      },
      'F16',
      'F16',
      'F16'
    ),
    /sourceQuantizationTarget requires/
  );
}

{
  const resolved = resolveEffectiveQuantizationInfo(
    {
      weights: 'q4_0',
      embeddings: 'q4_0',
      sourceTrainingQuantization: 'qat',
      sourceQuantizationTarget: 'q4_0',
    },
    [
      { name: 'model.embed_tokens.weight', role: 'embedding', dtype: 'Q4_0' },
      { name: 'model.layers.0.self_attn.q_proj.weight', role: 'matmul', dtype: 'Q4_0' },
      { name: 'lm_head.weight', role: 'lm_head', dtype: 'Q4_0' },
    ]
  );

  assert.equal(resolved.weights, 'q4_0');
  assert.equal(resolved.embeddings, 'q4_0');
  assert.equal(resolved.lmHead, 'q4_0');
  assert.equal(resolved.sourceTrainingQuantization, 'qat');
  assert.equal(resolved.sourceQuantizationTarget, 'q4_0');
}

{
  const packed = new Uint8Array(18);
  const result = transformTensorBytes(
    {
      name: 'model.layers.0.self_attn.q_proj.weight',
      dtype: 'Q4_0',
      shape: [1, 32],
    },
    packed,
    {
      targetQuant: 'q4_0',
    }
  );

  assert.equal(result.tensorData, packed);
  assert.equal(result.outDtype, 'Q4_0');
  assert.equal(result.tensorTargetQuant, 'q4_0');
  assert.deepEqual(result.storage, {
    packing: 'q4_0',
    blockShape: [32],
    blockBytes: 18,
  });
}

{
  const packed = new Uint8Array(32);
  const result = transformTensorBytes(
    {
      name: 'model.layers.0.self_attn.q_proj.weight',
      dtype: 'W4A16',
      shape: [1, 64],
    },
    packed,
    {
      targetQuant: 'w4a16',
    }
  );

  assert.equal(result.tensorData, packed);
  assert.equal(result.outDtype, 'W4A16');
  assert.equal(result.tensorTargetQuant, 'w4a16');
  assert.deepEqual(result.storage, {
    packing: 'w4a16',
    blockShape: [32],
    blockBytes: 16,
  });
}

{
  assert.throws(
    () => transformTensorBytes(
      {
        name: 'model.layers.0.self_attn.q_proj.weight',
        dtype: 'F16',
        shape: [4, 256],
      },
      new Uint8Array(4 * 256 * 2),
      {
        targetQuant: 'w4a16',
      }
    ),
    /does not re-quantize tensors into this packed format/
  );
}

{
  assert.throws(
    () => transformTensorBytes(
      {
        name: 'model.layers.0.self_attn.q_proj.weight',
        dtype: 'F16',
        shape: [4, 256],
      },
      new Uint8Array(4 * 256 * 2),
      {
        targetQuant: 'q4_0',
      }
    ),
    /does not re-quantize tensors into this packed format/
  );
}

{
  const normResult = transformTensorBytes(
    {
      name: 'model.layers.0.input_layernorm.weight',
      dtype: 'F16',
      shape: [256],
    },
    new Uint8Array(512),
    {
      targetQuant: 'q4_0',
      quantizationInfo: {
        weights: 'q4_0',
        embeddings: 'q4_0',
        lmHead: 'q4_0',
      },
    }
  );

  assert.equal(normResult.outDtype, 'F16');
  assert.equal(normResult.storage, undefined);
}

{
  assert.throws(
    () => transformTensorBytes(
      {
        name: 'lm_head.weight',
        dtype: 'F16',
        shape: [4, 256],
      },
      new Uint8Array(4 * 256 * 2),
      {
        targetQuant: 'q4_0',
        quantizationInfo: {
          weights: 'q4_0',
          embeddings: 'q4_0',
          lmHead: 'q4_0',
        },
      }
    ),
    /does not re-quantize tensors into this packed format/
  );
}

{
  assert.throws(
    () => loadTensorToCPU(new Uint8Array(18), { dtype: 'Q4_0' }),
    /Unsupported packed quantization dtype "Q4_0"/
  );
}

{
  assert.throws(
    () => loadTensorToCPU(new Uint8Array(32), { dtype: 'W4A16' }),
    /Unsupported packed quantization dtype "W4A16"/
  );
}

console.log('qat-quantization-info.test: ok');
