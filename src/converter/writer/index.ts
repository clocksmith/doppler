/**
 * RDRR Writer Module
 *
 * Re-exports all writer components and provides the high-level writeRDRR function.
 *
 * @module converter/writer
 */

import { log } from '../../debug/index.js';
import { RDRRWriter } from './writer.js';
import type {
  ModelInfo,
  WriteRDRROptions,
  WriteResultSchema,
  TensorInfoSchema,
} from './types.js';

// Re-export all types
export * from './types.js';

// Re-export utility functions
export { computeHash, alignOffset, createPadding, getBytesPerElement, transpose2D } from './utils.js';

// Re-export constants
export { DEFAULT_SHARD_SIZE, ALIGNMENT } from './types.js';

// Re-export classes
export { RDRRWriter } from './writer.js';
export { ShardWriter } from './shard-writer.js';
export { ManifestWriter, type ManifestData } from './manifest-writer.js';
export { TokenizerWriter, type TokenizerManifestEntry } from './tokenizer-writer.js';

// Re-export test model helper (from separate module)
export { createTestModel } from '../test-model.js';

/**
 * High-level function to write a model in RDRR format.
 *
 * @param outputDir - Directory to write the RDRR files
 * @param modelInfo - Model metadata and tensor list
 * @param getTensorData - Callback to retrieve tensor data by tensor info
 * @param options - Writer options including progress callback
 * @returns Write result with manifest path and statistics
 */
export async function writeRDRR(
  outputDir: string,
  modelInfo: ModelInfo,
  getTensorData: (tensor: TensorInfoSchema) => Promise<ArrayBuffer>,
  options: WriteRDRROptions = {}
): Promise<WriteResultSchema> {
  const config = modelInfo.config as Record<string, unknown> | undefined;
  const writer = new RDRRWriter(outputDir, {
    modelId: modelInfo.modelName || (config?.modelId as string) || 'model',
    architecture: modelInfo.architecture || (config?.architectures as string[])?.[0] || 'llama',
    quantization: modelInfo.quantization || options.quantization || 'Q4_K_M',
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

    if (hfTokenizer && hfTokenizer.model?.vocab) {
      await writer.writeHuggingFaceTokenizer(hfTokenizer);
    } else if (tokenizer?.tokens && tokenizer.tokens.length > 0) {
      await writer.writeTokenizer(tokenizer);
    } else if (tokenizer) {
      writer.setTokenizer(tokenizer as Record<string, unknown>);
    }

    if (config?.expertCount || config?.num_local_experts) {
      const numExperts = (config.expertCount || config.num_local_experts) as number;
      const numExpertsPerToken = (
        config.expertUsedCount ||
        config.num_experts_per_tok ||
        config.experts_per_token ||
        2
      ) as number;
      writer.setMoEConfig({ numExperts, numExpertsPerToken });
      log.verbose('writeRDRR', `MoE config: ${numExperts} experts, ${numExpertsPerToken} active per token`);
    }

    // Set conversion metadata if provided
    if (options.conversion) {
      writer.setConversion(options.conversion);
    }

    // Set runtime optimizations (including kernel path) if provided
    if (options.optimizations) {
      writer.setOptimizations(options.optimizations);
    }

    // Set inference configuration if provided (from model preset)
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
