

import { log } from '../../debug/index.js';
import { extractArchitecture } from '../core.js';
import { RDRRWriter } from './writer.js';
import { resolveEosTokenId } from '../tokenizer-utils.js';

// Re-export all types (JS just re-exports the constants)
export * from './types.js';

// Re-export utility functions
export { computeHash, alignOffset, createPadding, getBytesPerElement, transpose2D } from './utils.js';

// Re-export constants
export { DEFAULT_SHARD_SIZE, ALIGNMENT } from './types.js';

// Re-export classes
export { RDRRWriter } from './writer.js';
export { ShardWriter } from './shard-writer.js';
export { ManifestWriter } from './manifest-writer.js';
export { TokenizerWriter } from './tokenizer-writer.js';

// Re-export test model helper
export { createTestModel } from '../test-model.js';

function resolveExpertFormat(config, modelType) {
  const rawType = (modelType ?? config?.model_type ?? config?.text_config?.model_type ?? '').toLowerCase();
  if (rawType.includes('gpt_oss') || rawType.includes('gpt-oss') || rawType.includes('gptoss')) {
    return 'gpt-oss';
  }
  return 'mixtral';
}

export async function writeRDRR(outputDir, modelInfo, getTensorData, options = {}) {
  const config = modelInfo.config;
  const architecture = modelInfo.architecture ?? (config ? extractArchitecture(config) : null);
  const resolvedQuantization = modelInfo.quantization ?? options.quantization;
  if (!resolvedQuantization) {
    throw new Error('Quantization must be specified when writing RDRR output.');
  }
  const writer = new RDRRWriter(outputDir, {
    modelId: modelInfo.modelName || config?.modelId || 'model',
    architecture,
    quantization: resolvedQuantization,
    quantizationInfo: modelInfo.quantizationInfo ?? options.quantizationInfo,
    ...options,
  });

  try {
    await writer.init();

    if (modelInfo.config) {
      writer.setConfig(modelInfo.config);
    }

    const tokenizer = modelInfo.tokenizer || modelInfo.tokenizerConfig;
    const hfTokenizer = modelInfo.tokenizerJson;

    const eosTokenId = resolveEosTokenId({
      config,
      tokenizer: tokenizer ?? null,
      tokenizerJson: hfTokenizer ?? null,
    });
    writer.setMetadata({ eos_token_id: eosTokenId });

    if (hfTokenizer && hfTokenizer.model?.vocab) {
      await writer.writeHuggingFaceTokenizer(hfTokenizer);
    } else if (tokenizer?.tokens && tokenizer.tokens.length > 0) {
      await writer.writeTokenizer(tokenizer);
    } else if (tokenizer) {
      writer.setTokenizer(tokenizer);
    }

    if (config?.expertCount || config?.num_local_experts || config?.num_experts) {
      const numExperts = config.expertCount || config.num_local_experts || config.num_experts;
      const numExpertsPerToken =
        config.expertUsedCount ||
        config.num_experts_per_tok ||
        config.num_experts_per_token ||
        config.experts_per_token ||
        2;
      const expertFormat = resolveExpertFormat(config, options.modelType);
      writer.setMoEConfig({ numExperts, numExpertsPerToken, expertFormat });
      log.verbose(
        'writeRDRR',
        `MoE config: ${numExperts} experts, ${numExpertsPerToken} active per token (${expertFormat})`
      );
    }

    if (options.conversion) {
      writer.setConversion(options.conversion);
    }

    if (options.optimizations) {
      writer.setOptimizations(options.optimizations);
    }

    if (options.inference) {
      writer.setInference(options.inference);
    }

    const progressCallback = options.onProgress || (() => {});
    const totalTensors = modelInfo.tensors.length;

    for (let i = 0; i < modelInfo.tensors.length; i++) {
      const tensor = modelInfo.tensors[i];
      const data = await getTensorData(tensor);

      await writer.writeTensor(tensor.name, new Uint8Array(data), {
        shape: tensor.shape,
        dtype: tensor.dtype,
      });

      progressCallback({
        stage: 'writing',
        current: i + 1,
        total: totalTensors,
        tensorName: tensor.name,
      });
    }

    const result = await writer.finalize();
    progressCallback({ stage: 'complete', ...result });

    return result;
  } catch (error) {
    await writer.cleanup();
    throw error;
  }
}
