/**
 * Main conversion orchestration logic for the Node.js Model Converter.
 *
 * Handles conversion of SafeTensors and GGUF models to RDRR format.
 *
 * @module converter/node-converter/converter
 */

import { basename } from 'path';
import { open } from 'fs/promises';
import type { ConvertOptions, TensorInfo, ModelInfo } from './types.js';
import { createVerboseLogger } from './progress.js';
import {
  detectModelTypeFromPreset,
  isEmbeddingTensorName,
  isLmHeadTensorName,
  findTensorDtype,
} from './detection.js';
import {
  buildQuantizationInfo,
  resolveManifestQuantization,
  resolveModelId,
  toWebGPUDtype,
} from './quantization.js';
import {
  parseSafetensors,
  loadModelConfig,
  loadTokenizerConfig,
  readTensorData,
  type SafetensorsTensor,
} from '../../formats/safetensors/index.js';
import { parseGGUFFile } from '../../formats/gguf/index.js';
import {
  writeRDRR,
  type WriteResult,
  type WriteRDRROptions,
} from '../writer.js';
import { quantizeToQ4KM, float32ToFloat16 } from '../quantizer.js';
import { shouldQuantize as shouldQuantizeCore } from '../core.js';
import { buildManifestInference } from '../manifest-inference.js';
import { resolvePreset } from '../../config/index.js';

/**
 * Convert SafeTensors model to RDRR format.
 *
 * @param inputPath - Path to SafeTensors file or directory
 * @param outputPath - Output directory for RDRR model
 * @param opts - Conversion options
 * @returns Write result with manifest path, shard count, etc.
 */
export async function convertSafetensors(
  inputPath: string,
  outputPath: string,
  opts: ConvertOptions
): Promise<WriteResult> {
  const verboseLog = createVerboseLogger(opts.verbose, 'Convert');

  verboseLog(`Parsing safetensors from: ${inputPath}`);
  const parsed = await parseSafetensors(inputPath);
  verboseLog(`Found ${parsed.tensors.length} tensors`);

  // Load config and tokenizer
  const config = await loadModelConfig(inputPath).catch(() => ({}));
  const tokenizerConfig = await loadTokenizerConfig(inputPath).catch(() => undefined);

  // Detect architecture and model type using preset system
  const configRec = config as Record<string, unknown>;
  const arch = (configRec.architectures as string[] | undefined)?.[0] ||
    (configRec.model_type as string | undefined) ||
    'llama';
  const { presetId, modelType } = detectModelTypeFromPreset(arch, config);

  verboseLog(`Architecture: ${arch}`);
  verboseLog(`Detected preset: ${presetId}`);
  verboseLog(`Model type: ${modelType}`);
  verboseLog(`Config: ${JSON.stringify(config, null, 2).slice(0, 500)}...`);

  // Detect multimodal components
  const visionPatterns = ['vision_', 'vision_tower', 'vision_model', 'image_encoder'];
  const audioPatterns = ['audio_', 'audio_encoder', 'whisper', 'wav2vec'];
  const projectorPatterns = ['multi_modal_projector', 'mm_projector', 'projector'];

  const hasVision = parsed.tensors.some((t: SafetensorsTensor) =>
    visionPatterns.some(p => t.name.toLowerCase().includes(p)));
  const hasAudio = parsed.tensors.some((t: SafetensorsTensor) =>
    audioPatterns.some(p => t.name.toLowerCase().includes(p)));
  const hasProjector = parsed.tensors.some((t: SafetensorsTensor) =>
    projectorPatterns.some(p => t.name.toLowerCase().includes(p)));

  if (hasVision) verboseLog('Detected vision encoder');
  if (hasAudio) verboseLog('Detected audio encoder');
  if (hasProjector) verboseLog('Detected multimodal projector');

  // Filter text-only if requested
  let tensors = parsed.tensors;
  if (opts.textOnly) {
    const textOnlyPatterns = ['language_model', 'model.', 'lm_head', 'embed_tokens'];
    const excludePatterns = [...visionPatterns, ...audioPatterns, ...projectorPatterns, 'image_newline'];

    tensors = tensors.filter((t: SafetensorsTensor) => {
      const isMultimodal = excludePatterns.some(p => t.name.toLowerCase().includes(p));
      if (isMultimodal) {
        verboseLog(`Skipping multimodal tensor: ${t.name}`);
        return false;
      }
      return true;
    });
    verboseLog(`Filtered to ${tensors.length} text-only tensors`);
  }

  // Detect the original dtype from the first weight tensor (most are the same dtype)
  const firstWeight = tensors.find((t: SafetensorsTensor) => t.name.includes('.weight'));
  const originalDtype = firstWeight?.dtype?.toUpperCase() || 'F32';

  // Filter out tensors without valid shape (e.g., metadata entries)
  const validTensors = tensors.filter((t: SafetensorsTensor) => {
    if (!t.shape || !Array.isArray(t.shape)) {
      verboseLog(`Skipping tensor without shape: ${t.name}`);
      return false;
    }
    return true;
  });

  const embedDtypeRaw = findTensorDtype(validTensors, isEmbeddingTensorName);
  const lmHeadDtypeRaw = findTensorDtype(validTensors, isLmHeadTensorName);
  const quantizationInfo = buildQuantizationInfo(
    opts, originalDtype, embedDtypeRaw, lmHeadDtypeRaw,
    hasVision && !opts.textOnly,
    hasAudio,
    hasProjector && !opts.textOnly
  );
  const baseModelId = typeof configRec._name_or_path === 'string'
    ? configRec._name_or_path
    : basename(inputPath);
  const resolvedModelId = resolveModelId(opts.modelId, baseModelId, quantizationInfo.variantTag);
  const manifestQuantization = resolveManifestQuantization(opts.weightQuant, originalDtype);

  // Helper to determine output dtype for a tensor based on its type
  const getOutputDtype = (name: string, shape: number[], origDtype: string): string => {
    const isEmbedding = isEmbeddingTensorName(name);
    const isHead = isLmHeadTensorName(name);

    if (isEmbedding) {
      return toWebGPUDtype(quantizationInfo.embeddings);
    }
    if (isHead) {
      const headQuant = quantizationInfo.lmHead ?? quantizationInfo.embeddings;
      return toWebGPUDtype(headQuant);
    }
    // Regular weight tensor
    if (shouldQuantizeCore(name, shape)) {
      return toWebGPUDtype(quantizationInfo.weights);
    }
    // Non-quantized tensor - still need WebGPU-safe dtype
    if (origDtype === 'BF16') return 'F16';
    return origDtype;
  };

  // Create modelInfo with adjusted dtypes for quantized tensors
  const modelInfo: ModelInfo = {
    modelName: resolvedModelId,
    architecture: arch,
    quantization: manifestQuantization,
    quantizationInfo,
    config,
    tokenizerConfig,
    tokenizerJson: parsed.tokenizerJson,
    tensors: validTensors.map((t: SafetensorsTensor) => ({
      name: t.name,
      shape: t.shape,
      dtype: getOutputDtype(t.name, t.shape, t.dtype),
      size: t.size,
    })),
  };

  // Create tensor data getter
  const tensorMap = new Map(validTensors.map((t: SafetensorsTensor) => [t.name, t]));

  const getTensorData = async (info: TensorInfo): Promise<ArrayBuffer> => {
    const tensor = tensorMap.get(info.name);
    if (!tensor) {
      throw new Error(`Tensor not found: ${info.name}`);
    }

    // Read raw tensor data from safetensors file
    const tensorWithPath = { ...tensor, filePath: tensor.filePath || tensor.shardPath || inputPath };
    const data = await readTensorData(tensorWithPath);

    // Determine target dtype from quantizationInfo
    const targetDtype = getOutputDtype(info.name, info.shape, tensor.dtype);

    // Quantize to Q4_K_M if needed
    if (targetDtype === 'Q4_K_M') {
      verboseLog(`Quantizing ${info.name} to Q4_K_M`);

      // Convert to F32 based on source dtype
      let f32: Float32Array;
      if (tensor.dtype === 'BF16') {
        const bf16Data = new Uint16Array(data);
        f32 = new Float32Array(bf16Data.length);
        for (let i = 0; i < bf16Data.length; i++) {
          const bits = bf16Data[i] << 16;
          const f32View = new Float32Array(1);
          new Uint32Array(f32View.buffer)[0] = bits;
          f32[i] = f32View[0];
        }
      } else {
        f32 = new Float32Array(data);
      }

      const q4 = quantizeToQ4KM(f32, info.shape);
      return q4.quantized.buffer as ArrayBuffer;
    }

    // Convert to F16 if needed
    if (targetDtype === 'F16' && tensor.dtype !== 'F16') {
      verboseLog(`Converting ${info.name} from ${tensor.dtype} to F16`);

      let f32: Float32Array;
      if (tensor.dtype === 'BF16') {
        const bf16Data = new Uint16Array(data);
        f32 = new Float32Array(bf16Data.length);
        for (let i = 0; i < bf16Data.length; i++) {
          const bits = bf16Data[i] << 16;
          const f32View = new Float32Array(1);
          new Uint32Array(f32View.buffer)[0] = bits;
          f32[i] = f32View[0];
        }
      } else if (tensor.dtype === 'F32') {
        f32 = new Float32Array(data);
      } else {
        verboseLog(`Warning: Unknown dtype ${tensor.dtype}, treating as F32`);
        f32 = new Float32Array(data);
      }

      const f16 = new Uint16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        f16[i] = float32ToFloat16(f32[i]);
      }
      return f16.buffer;
    }

    return data;
  };

  // Write RDRR
  verboseLog(`Writing RDRR to: ${outputPath}`);

  // Build inference config from resolved preset
  const resolvedPreset = resolvePreset(presetId);
  // Compute headDim from HF config (hidden_size / num_attention_heads)
  const headDim = (configRec.head_dim as number) ??
    (((configRec.hidden_size as number) / (configRec.num_attention_heads as number)) || 64);
  const manifestInference = buildManifestInference(resolvedPreset, config, headDim);
  verboseLog(`Inference config: rmsNormWeightOffset=${manifestInference.normalization.rmsNormWeightOffset}, attnLogitSoftcapping=${manifestInference.attention.attnLogitSoftcapping}`);

  const writerOpts: WriteRDRROptions = {
    shardSize: opts.shardSize * 1024 * 1024,
    modelId: resolvedModelId,
    modelType,
    architecture: arch,
    quantization: manifestQuantization,
    quantizationInfo,
    inference: manifestInference,
  };

  return writeRDRR(outputPath, modelInfo, getTensorData, writerOpts);
}

/**
 * Convert GGUF model to RDRR format.
 *
 * @param inputPath - Path to GGUF file
 * @param outputPath - Output directory for RDRR model
 * @param opts - Conversion options
 * @returns Write result with manifest path, shard count, etc.
 */
export async function convertGGUF(
  inputPath: string,
  outputPath: string,
  opts: ConvertOptions
): Promise<WriteResult> {
  const verboseLog = createVerboseLogger(opts.verbose, 'Convert');

  verboseLog(`Parsing GGUF from: ${inputPath}`);
  const parsed = await parseGGUFFile(inputPath);
  verboseLog(`Found ${parsed.tensors.length} tensors`);

  // Use the already parsed GGUF config
  const ggufConfig = parsed.config;
  const arch = ggufConfig.architecture || parsed.architecture || 'llama';
  const modelName = parsed.modelName || basename(inputPath, '.gguf');

  // Map GGUF config to HuggingFace-style config that DOPPLER expects
  // (Build this first so we can detect model type from it)
  const config: Record<string, unknown> = {
    architectures: [arch + 'ForCausalLM'], // e.g., "gemma2" -> "gemma2ForCausalLM"
    model_type: arch,
    hidden_size: ggufConfig.embeddingLength,
    num_hidden_layers: ggufConfig.blockCount,
    num_attention_heads: ggufConfig.attentionHeadCount,
    num_key_value_heads: ggufConfig.attentionHeadCountKV,
    vocab_size: ggufConfig.vocabSize,
    intermediate_size: ggufConfig.feedForwardLength,
    rms_norm_eps: ggufConfig.attentionLayerNormRMSEpsilon || ggufConfig.attentionLayerNormEpsilon || 1e-5,
    rope_theta: ggufConfig.ropeFreqBase || 10000,
    max_position_embeddings: ggufConfig.contextLength || 8192,
    // MoE fields from GGUF
    num_local_experts: ggufConfig.expertCount,
  };

  // Detect model type using preset system
  const { presetId, modelType } = detectModelTypeFromPreset(arch, config);

  verboseLog(`Architecture: ${arch}`);
  verboseLog(`Detected preset: ${presetId}`);
  verboseLog(`Model type: ${modelType}`);
  verboseLog(`Model: ${modelName}`);

  const tensors = parsed.tensors;

  // Extract tokenizer from GGUF
  const ggufTokenizer = ggufConfig.tokenizer;
  verboseLog(`Tokenizer: ${ggufTokenizer?.model || 'unknown'}, ${ggufTokenizer?.tokens?.length || 0} tokens`);

  const embedDtypeRaw = findTensorDtype(tensors, isEmbeddingTensorName);
  const lmHeadDtypeRaw = findTensorDtype(tensors, isLmHeadTensorName);
  const quantizationInfo = buildQuantizationInfo(
    opts,
    parsed.quantization || 'F32',
    embedDtypeRaw,
    lmHeadDtypeRaw
  );
  const resolvedModelId = resolveModelId(opts.modelId, modelName, quantizationInfo.variantTag);
  const manifestQuantization = resolveManifestQuantization(opts.weightQuant, parsed.quantization || 'F32');

  const modelInfo: ModelInfo = {
    modelName: resolvedModelId,
    architecture: arch,
    quantization: manifestQuantization,
    quantizationInfo,
    config,
    // Pass GGUF tokenizer data for bundling
    tokenizer: ggufTokenizer ? {
      model: ggufTokenizer.model,
      tokens: ggufTokenizer.tokens,
      scores: ggufTokenizer.scores,
      tokenTypes: ggufTokenizer.tokenTypes,
      merges: ggufTokenizer.merges,
      bosTokenId: ggufTokenizer.bosTokenId,
      eosTokenId: ggufTokenizer.eosTokenId,
      padTokenId: ggufTokenizer.padTokenId,
      unkTokenId: ggufTokenizer.unkTokenId,
      sepTokenId: ggufTokenizer.sepTokenId,
      clsTokenId: ggufTokenizer.clsTokenId,
      maskTokenId: ggufTokenizer.maskTokenId,
      addBosToken: ggufTokenizer.addBosToken,
      addEosToken: ggufTokenizer.addEosToken,
      addSpacePrefix: ggufTokenizer.addSpacePrefix,
    } : undefined,
    tensors: tensors.map(t => ({
      name: t.name,
      shape: t.shape,
      dtype: t.dtype,
      size: t.size,
    })),
  };

  // Read tensor data
  const fileHandle = await open(inputPath, 'r');
  const getTensorData = async (info: TensorInfo): Promise<ArrayBuffer> => {
    const tensor = tensors.find(t => t.name === info.name);
    if (!tensor) {
      throw new Error(`Tensor not found: ${info.name}`);
    }

    const buffer = Buffer.alloc(tensor.size);
    await fileHandle.read(buffer, 0, tensor.size, tensor.offset);
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  };

  try {
    verboseLog(`Writing RDRR to: ${outputPath}`);

    // Build inference config from resolved preset
    const resolvedPreset = resolvePreset(presetId);
    // Compute headDim from config (hidden_size / num_attention_heads) or use GGUF's headDim
    const ggufHeadDim = ggufConfig.attentionKeyLength || ggufConfig.attentionValueLength;
    const headDim = (ggufHeadDim as number | undefined) ??
      (((config.hidden_size as number) / (config.num_attention_heads as number)) || 64);
    const manifestInference = buildManifestInference(resolvedPreset, config as Record<string, unknown>, headDim);
    verboseLog(`Inference config: rmsNormWeightOffset=${manifestInference.normalization.rmsNormWeightOffset}, attnLogitSoftcapping=${manifestInference.attention.attnLogitSoftcapping}`);

    const writerOpts: WriteRDRROptions = {
      shardSize: opts.shardSize * 1024 * 1024,
      modelId: resolvedModelId,
      modelType,
      architecture: arch,
      quantization: manifestQuantization,
      quantizationInfo,
      inference: manifestInference,
    };

    const result = await writeRDRR(outputPath, modelInfo, getTensorData, writerOpts);
    return result;
  } finally {
    await fileHandle.close();
  }
}
