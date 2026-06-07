import assert from 'node:assert/strict';

const {
  buildQuantizationInfo,
  resolveEffectiveQuantizationInfo,
  resolveManifestQuantization,
} = await import('../../src/converter/quantization-info.js');
const { convertModel, transformTensorBytes } = await import('../../src/converter/core.js');
const { loadTensorToCPU } = await import('../../src/loader/tensors/tensor-loader.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

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
  assert.throws(
    () => buildQuantizationInfo(
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
    ),
    /requires quantizationInfo\.lmHead="w4a16"/
  );
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
  const tensorBytes = new Map([
    ['model.layers.0.self_attn.q_proj.weight', new Uint8Array(16)],
    ['model.layers.0.self_attn.q_proj.weight_scale', new Uint8Array(4)],
    ['model.layers.0.self_attn.q_proj.weight_shape', new Uint8Array(16)],
  ]);
  let manifest = null;
  const result = await convertModel(
    {
      name: 'gemma-4-12b-w4a16-ct-test',
      modelId: 'gemma-4-12b-w4a16-ct-test',
      modelType: 'transformer',
      quantization: 'W4A16',
      architecture: {
        numLayers: 1,
        hiddenSize: 64,
        intermediateSize: 128,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        headDim: 64,
        vocabSize: 16,
        maxSeqLen: 8,
      },
      config: { model_type: 'gemma4_text' },
      tensors: [
        {
          name: 'model.layers.0.self_attn.q_proj.weight_packed',
          shape: [1, 64],
          dtype: 'I32',
          size: 16,
          offset: 0,
        },
        {
          name: 'model.layers.0.self_attn.q_proj.weight_scale',
          shape: [1, 2],
          dtype: 'F16',
          size: 4,
          offset: 16,
        },
        {
          name: 'model.layers.0.self_attn.q_proj.weight_shape',
          shape: [2],
          dtype: 'I64',
          size: 16,
          offset: 20,
        },
      ],
    },
    {
      async readTensorData(tensor) {
        const bytes = tensorBytes.get(tensor.name);
        if (!bytes) {
          throw new Error(`missing test bytes for ${tensor.name}`);
        }
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      },
      async writeShard(index) {
        return `hash-${index}`;
      },
      async writeManifest(value) {
        manifest = value;
      },
    },
    {
      modelId: 'gemma-4-12b-w4a16-ct-test',
      modelType: 'transformer',
      quantization: 'W4A16',
      quantizationInfo: {
        weights: 'w4a16',
        embeddings: 'f16',
        lmHead: 'w4a16',
        sourceTrainingQuantization: 'qat',
        sourceQuantizationTarget: 'w4a16',
      },
      architecture: {
        numLayers: 1,
        hiddenSize: 64,
        intermediateSize: 128,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        headDim: 64,
        vocabSize: 16,
        maxSeqLen: 8,
      },
      inference: { ...DEFAULT_MANIFEST_INFERENCE },
      converterConfig: createConverterConfig({
        quantization: {
          weights: 'w4a16',
          embeddings: 'f16',
          lmHead: 'w4a16',
          sourceTrainingQuantization: 'qat',
          sourceQuantizationTarget: 'w4a16',
        },
      }),
      source: 'unit-test',
      sourceFormat: 'safetensors',
      eosTokenId: 1,
    }
  );

  assert.ok(manifest);
  assert.equal(result.tensorCount, 3);
  const primary = manifest.tensors['model.layers.0.self_attn.q_proj.weight'];
  assert.equal(primary.dtype, 'W4A16');
  assert.equal(primary.role, 'matmul');
  assert.equal(primary.size, 16);
  assert.deepEqual(primary.storage, {
    packing: 'w4a16',
    blockShape: [32],
    blockBytes: 16,
    companions: [
      { role: 'scales', tensorId: 'model.layers.0.self_attn.q_proj.weight_scale' },
      { role: 'shape', tensorId: 'model.layers.0.self_attn.q_proj.weight_shape' },
    ],
  });
  assert.equal(
    manifest.tensors['model.layers.0.self_attn.q_proj.weight_packed'],
    undefined
  );
  assert.ok(manifest.tensors['model.layers.0.self_attn.q_proj.weight_scale']);
  assert.ok(manifest.tensors['model.layers.0.self_attn.q_proj.weight_shape']);
  assert.equal(manifest.quantizationInfo.sourceTrainingQuantization, 'qat');
  assert.equal(manifest.quantizationInfo.sourceQuantizationTarget, 'w4a16');
  assert.equal(manifest.quantizationInfo.lmHead, 'w4a16');
}

{
  assert.throws(
    () => loadTensorToCPU(new Uint8Array(32), { dtype: 'W4A16' }),
    /Unsupported packed quantization dtype "W4A16"/
  );
}

console.log('qat-quantization-info.test: ok');
