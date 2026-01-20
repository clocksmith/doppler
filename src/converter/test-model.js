

import { RDRRWriter } from './writer.js';
import { DEFAULT_MANIFEST_INFERENCE, DEFAULT_RMS_NORM_EPS } from '../config/schema/index.js';


export async function createTestModel(outputDir) {
  const hiddenSize = 64;
  const vocabSize = 1000;
  const intermediateSize = 256;
  const numHeads = 2;
  const numLayers = 2;
  const headDim = hiddenSize / numHeads;
  const maxSeqLen = 128;

  const writer = new RDRRWriter(outputDir, {
    modelId: 'tiny-test',
    architecture: {
      numLayers,
      hiddenSize,
      intermediateSize,
      numAttentionHeads: numHeads,
      numKeyValueHeads: numHeads,
      headDim,
      vocabSize,
      maxSeqLen,
      ropeTheta: 10000,
      rmsNormEps: DEFAULT_RMS_NORM_EPS,
    },
    quantization: 'F32',
  });

  await writer.init();

  writer.setConfig({
    vocabSize,
    hiddenSize,
    numLayers,
    numHeads,
    contextLength: maxSeqLen,
  });

  writer.setTokenizer({
    model: 'bpe',
    vocabSize: 1000,
    bosTokenId: 1,
    eosTokenId: 2,
  });

  writer.setInference({
    ...DEFAULT_MANIFEST_INFERENCE,
    attention: {
      ...DEFAULT_MANIFEST_INFERENCE.attention,
      queryPreAttnScalar: Math.sqrt(headDim),
    },
  });

  // Embedding layer
  const embedData = new Float32Array(vocabSize * hiddenSize);
  for (let i = 0; i < embedData.length; i++) {
    embedData[i] = (Math.random() - 0.5) * 0.02;
  }
  await writer.writeTensor('embed_tokens.weight', new Uint8Array(embedData.buffer), {
    shape: [vocabSize, hiddenSize],
    dtype: 'F32',
  });

  // Transformer layers
  for (let layer = 0; layer < numLayers; layer++) {
    const qkvSize = hiddenSize * hiddenSize * 3;
    const qkvData = new Float32Array(qkvSize);
    for (let i = 0; i < qkvSize; i++) {
      qkvData[i] = (Math.random() - 0.5) * 0.02;
    }
    await writer.writeTensor(`layers.${layer}.attention.qkv.weight`, new Uint8Array(qkvData.buffer), {
      shape: [hiddenSize * 3, hiddenSize],
      dtype: 'F32',
    });

    const oData = new Float32Array(hiddenSize * hiddenSize);
    for (let i = 0; i < oData.length; i++) {
      oData[i] = (Math.random() - 0.5) * 0.02;
    }
    await writer.writeTensor(`layers.${layer}.attention.o.weight`, new Uint8Array(oData.buffer), {
      shape: [hiddenSize, hiddenSize],
      dtype: 'F32',
    });

    const upData = new Float32Array(intermediateSize * hiddenSize);
    for (let i = 0; i < upData.length; i++) {
      upData[i] = (Math.random() - 0.5) * 0.02;
    }
    await writer.writeTensor(`layers.${layer}.ffn.up.weight`, new Uint8Array(upData.buffer), {
      shape: [intermediateSize, hiddenSize],
      dtype: 'F32',
    });

    const downData = new Float32Array(hiddenSize * intermediateSize);
    for (let i = 0; i < downData.length; i++) {
      downData[i] = (Math.random() - 0.5) * 0.02;
    }
    await writer.writeTensor(`layers.${layer}.ffn.down.weight`, new Uint8Array(downData.buffer), {
      shape: [hiddenSize, intermediateSize],
      dtype: 'F32',
    });

    const normData = new Float32Array(hiddenSize).fill(1.0);
    await writer.writeTensor(`layers.${layer}.input_norm.weight`, new Uint8Array(normData.buffer), {
      shape: [hiddenSize],
      dtype: 'F32',
    });
    await writer.writeTensor(`layers.${layer}.post_norm.weight`, new Uint8Array(normData.buffer), {
      shape: [hiddenSize],
      dtype: 'F32',
    });
  }

  // LM head
  const lmHeadData = new Float32Array(vocabSize * hiddenSize);
  for (let i = 0; i < lmHeadData.length; i++) {
    lmHeadData[i] = (Math.random() - 0.5) * 0.02;
  }
  await writer.writeTensor('lm_head.weight', new Uint8Array(lmHeadData.buffer), {
    shape: [vocabSize, hiddenSize],
    dtype: 'F32',
  });

  // Final norm
  const finalNormData = new Float32Array(hiddenSize).fill(1.0);
  await writer.writeTensor('final_norm.weight', new Uint8Array(finalNormData.buffer), {
    shape: [hiddenSize],
    dtype: 'F32',
  });

  return writer.finalize();
}
