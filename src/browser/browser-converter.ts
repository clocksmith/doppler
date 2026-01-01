/**
 * browser-converter.ts - Browser Model Converter
 *
 * Converts GGUF and safetensors models to RDRR format in the browser.
 * Uses ShardPacker for platform-agnostic shard packing logic.
 *
 * Supports:
 * - GGUF files
 * - Single safetensors files
 * - Sharded HuggingFace models (multiple safetensors + config)
 *
 * Output is written to OPFS (Origin Private File System).
 *
 * @module browser/browser-converter
 */

import { parseGGUFHeader } from './gguf-parser-browser.js';
import {
  parseSafetensorsFile,
  parseSafetensorsSharded,
  parseConfigJson,
  parseTokenizerJson,
  parseIndexJson,
  readTensorData,
  detectModelFormat,
  getAuxiliaryFiles,
  calculateTotalSize,
  SafetensorsParseResult,
  TensorInfo as BrowserTensorInfo,
  ModelConfig,
} from './safetensors-parser-browser.js';
import {
  initOPFS,
  openModelDirectory,
  saveManifest,
  deleteModel,
} from '../storage/shard-manager.js';

// Import shared shard packing logic
import {
  ShardPacker,
  sortTensorsByGroup,
  type PackerTensorInput,
  type TensorLocation,
  type ShardPackerResult,
} from '../converter/shard-packer.js';
import { BrowserShardIO, isOPFSSupported } from './shard-io-browser.js';

// Import shared types and functions from convert-core
import {
  ConvertStage,
  sanitizeModelId,
  formatBytes,
  extractArchitecture,
  createManifest,
  type ConvertStageType,
  type ConvertProgress,
  type ConvertOptions as CoreConvertOptions,
  type ShardInfo,
  type RDRRManifest,
  type ParsedModel,
  type TensorInfo,
} from '../converter/core.js';

import { detectPreset, resolvePreset } from '../config/index.js';

// Re-export types for consumers
export {
  ConvertStage,
  type ConvertStageType,
  type ConvertProgress,
  type ShardInfo,
  type TensorLocation,
  type RDRRManifest,
};

// Re-export OPFS support check
export { isOPFSSupported as isConversionSupported };

// ============================================================================
// Browser-specific Types
// ============================================================================

/**
 * Browser conversion options (extends core with File[] input)
 */
export interface ConvertOptions extends CoreConvertOptions {
  // Browser-specific: no additional options needed
}

/**
 * GGUF model config extracted from header
 */
interface GGUFModelConfig {
  blockCount?: number;
  embeddingLength?: number;
  feedForwardLength?: number;
  attentionHeadCount?: number;
  attentionHeadCountKV?: number;
  vocabSize?: number;
  contextLength?: number;
  [key: string]: unknown;
}

/**
 * Internal tensor info with browser File reference
 */
interface InternalTensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  dtypeOriginal?: string;
  dtypeId?: number;
  offset: number;
  size: number;
  elemSize?: number;
  file?: File;
  shardFile?: string;
}

/**
 * Extended model info for internal use
 */
interface ModelInfo {
  format?: string;
  tensors: InternalTensorInfo[];
  config?: ModelConfig | GGUFModelConfig;
  architecture?: string;
  quantization?: string;
  tensorDataOffset?: number;
  file?: File;
  tokenizerJson?: unknown;
}

/**
 * Write result
 */
interface WriteResult {
  totalSize: number;
  tensorLocations: Record<string, TensorLocation>;
}


// ============================================================================
// Main Convert Function
// ============================================================================

/**
 * Convert model files to RDRR format
 *
 * @param files - Selected model files
 * @param options - Conversion options
 * @returns Model ID
 */
export async function convertModel(files: File[], options: ConvertOptions = {}): Promise<string> {
  const { modelId: userModelId, onProgress, signal } = options;

  let modelId: string | null = null;
  let modelDir: FileSystemDirectoryHandle | null = null;
  const shardInfos: ShardInfo[] = [];

  try {
    // Initialize OPFS
    await initOPFS();

    // Detect format
    onProgress?.({
      stage: ConvertStage.DETECTING,
      message: 'Detecting model format...',
    });

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    const format = detectModelFormat(files);
    const auxiliary = getAuxiliaryFiles(files);

    onProgress?.({
      stage: ConvertStage.DETECTING,
      message: `Format: ${format.type}`,
      format: format.type,
    });

    // Parse based on format
    let modelInfo: ModelInfo;
    let config: ModelConfig | null = null;
    let tokenizerJson: unknown = null;

    if (format.type === 'gguf') {
      modelInfo = await parseGGUFModel(format.ggufFile!, onProgress, signal);
    } else if (format.type === 'single') {
      const parsed = await parseSafetensorsFile(format.safetensorsFile!);
      modelInfo = { tensors: parsed.tensors as InternalTensorInfo[], config: parsed.config };
      if (auxiliary.config) {
        config = await parseConfigJson(auxiliary.config);
        modelInfo.config = config;
      }
    } else if (format.type === 'sharded' || format.type === 'sharded-no-index') {
      let indexJson = null;
      if (format.indexFile) {
        indexJson = await parseIndexJson(format.indexFile);
      }
      const parsed = await parseSafetensorsSharded(format.safetensorsFiles!, indexJson);
      modelInfo = { tensors: parsed.tensors as InternalTensorInfo[], config: parsed.config };
      if (auxiliary.config) {
        config = await parseConfigJson(auxiliary.config);
        modelInfo.config = config;
      }
    } else {
      throw new Error(`Unsupported format: ${format.type}`);
    }

    // Parse tokenizer if available
    if (auxiliary.tokenizer) {
      tokenizerJson = await parseTokenizerJson(auxiliary.tokenizer);
      modelInfo.tokenizerJson = tokenizerJson;
    }

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    // Determine model ID
    modelId = userModelId || extractModelId(files, config) || 'converted-model';
    modelId = sanitizeModelId(modelId);

    onProgress?.({
      stage: ConvertStage.PARSING,
      message: `Model: ${modelId}`,
      modelId,
      tensorCount: modelInfo.tensors.length,
      totalSize: formatBytes(calculateTotalSize(modelInfo as SafetensorsParseResult)),
    });

    // Open model directory in OPFS
    modelDir = await openModelDirectory(modelId);

    // Detect model type using preset system
    const presetId = detectPreset(
      (config || modelInfo.config || {}) as Parameters<typeof detectPreset>[0],
      modelInfo.architecture
    );
    const preset = resolvePreset(presetId);
    const modelType = preset.modelType || 'transformer';

    // Create shard I/O adapter
    const shardIO = new BrowserShardIO(modelDir);

    // Create shard packer
    const packer = new ShardPacker(shardIO, { modelType });

    // Prepare tensors for packing
    const packerTensors: PackerTensorInput[] = modelInfo.tensors.map(tensor => ({
      name: tensor.name,
      shape: tensor.shape,
      dtype: tensor.dtype,
      size: tensor.size,
      getData: async () => {
        // Handle GGUF format (data is relative to tensorDataOffset)
        if (modelInfo.format === 'gguf' && modelInfo.file) {
          const blob = modelInfo.file.slice(tensor.offset, tensor.offset + tensor.size);
          return new Uint8Array(await blob.arrayBuffer());
        }
        // Safetensors format
        const data = await readTensorData(tensor as BrowserTensorInfo);
        return new Uint8Array(data);
      },
    }));

    // Pack tensors into shards
    onProgress?.({
      stage: ConvertStage.WRITING,
      message: 'Packing tensors...',
    });

    const packResult = await packer.pack(packerTensors, {
      onProgress: (current, total, tensorName) => {
        onProgress?.({
          stage: ConvertStage.WRITING,
          message: `Processing ${tensorName}`,
          current,
          total,
          percent: Math.round((current / total) * 100),
        });
      },
      signal,
    });

    // Convert pack result to expected format
    const result = {
      totalSize: packResult.totalSize,
      tensorLocations: packResult.tensors,
    };

    // Copy shard infos
    for (const shard of packResult.shards) {
      shardInfos.push(shard);
    }

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    // Create manifest using shared function
    onProgress?.({
      stage: ConvertStage.MANIFEST,
      message: 'Creating manifest...',
    });

    // Convert to ParsedModel format for createManifest
    const parsedModel: ParsedModel = {
      tensors: modelInfo.tensors.map(t => ({
        name: t.name,
        shape: t.shape,
        dtype: t.dtype,
        size: t.size,
        offset: t.offset,
      })),
      config: (config || modelInfo.config || {}) as ParsedModel['config'],
      architecture: modelInfo.architecture,
      quantization: modelInfo.quantization || 'F16',
      tokenizerJson,
    };

    const manifest = createManifest(
      modelId,
      parsedModel,
      shardInfos,
      result.tensorLocations,
      'browser-converter'
    );

    // Save manifest
    await saveManifest(JSON.stringify(manifest, null, 2));

    onProgress?.({
      stage: ConvertStage.COMPLETE,
      message: 'Conversion complete!',
      modelId,
      shardCount: shardInfos.length,
      totalSize: formatBytes(result.totalSize),
    });

    return modelId;
  } catch (error) {
    // Cleanup on error
    if (modelId) {
      try {
        await deleteModel(modelId);
      } catch {
        // Ignore cleanup errors
      }
    }

    onProgress?.({
      stage: ConvertStage.ERROR,
      message: (error as Error).message,
      error: error as Error,
    });

    throw error;
  }
}

/**
 * Parse GGUF model file
 */
async function parseGGUFModel(
  file: File,
  onProgress?: (progress: ConvertProgress) => void,
  signal?: AbortSignal
): Promise<ModelInfo> {
  onProgress?.({
    stage: ConvertStage.PARSING,
    message: 'Parsing GGUF header...',
  });

  const headerBlob = file.slice(0, 10 * 1024 * 1024);
  const headerBuffer = await headerBlob.arrayBuffer();
  const ggufInfo = parseGGUFHeader(headerBuffer);

  return {
    format: 'gguf',
    tensors: ggufInfo.tensors.map((t) => ({
      ...t,
      file,
      offset: t.offset,
    })),
    config: ggufInfo.config,
    architecture: ggufInfo.architecture,
    quantization: ggufInfo.quantization,
    tensorDataOffset: ggufInfo.tensorDataOffset,
    file,
  };
}

/**
 * Extract model ID from files or config
 */
function extractModelId(files: File[], config: ModelConfig | null): string | null {
  // Try config first
  if (config?._name_or_path) {
    const parts = config._name_or_path.split('/');
    return parts[parts.length - 1];
  }

  // Try first safetensors file name
  const stFile = files.find((f) => f.name.endsWith('.safetensors'));
  if (stFile) {
    return stFile.name.replace(/\.safetensors$/, '').replace(/model[-_.]?/, '');
  }

  // Try GGUF file name
  const ggufFile = files.find((f) => f.name.endsWith('.gguf'));
  if (ggufFile) {
    return ggufFile.name.replace(/\.gguf$/, '');
  }

  return null;
}

// ============================================================================
// File Picker Utilities
// ============================================================================

/**
 * Pick model files using File System Access API
 * @returns Selected files
 */
export async function pickModelFiles(): Promise<File[]> {
  // Try directory picker first (for HuggingFace models)
  if ('showDirectoryPicker' in window) {
    try {
      const dirHandle = await (window as Window & { showDirectoryPicker: (opts?: { mode?: string }) => Promise<FileSystemDirectoryHandle> }).showDirectoryPicker({
        mode: 'read',
      });
      return await collectFilesFromDirectory(dirHandle);
    } catch (e) {
      if ((e as Error).name === 'AbortError') throw e;
      // Fall back to file picker
    }
  }

  // Fall back to file picker
  if ('showOpenFilePicker' in window) {
    const handles = await (window as Window & {
      showOpenFilePicker: (opts?: {
        multiple?: boolean;
        types?: Array<{ description: string; accept: Record<string, string[]> }>;
      }) => Promise<FileSystemFileHandle[]>;
    }).showOpenFilePicker({
      multiple: true,
      types: [
        {
          description: 'Model files',
          accept: {
            'application/octet-stream': ['.gguf', '.safetensors', '.bin'],
            'application/json': ['.json'],
          },
        },
      ],
    });
    return Promise.all(handles.map((h) => h.getFile()));
  }

  // Ultimate fallback: input element
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.gguf,.safetensors,.json,.bin';
    input.onchange = () => {
      resolve(Array.from(input.files || []));
    };
    input.click();
  });
}

/**
 * Collect all files from a directory handle recursively
 */
async function collectFilesFromDirectory(
  dirHandle: FileSystemDirectoryHandle,
  files: File[] = []
): Promise<File[]> {
  const entries = (dirHandle as unknown as { values(): AsyncIterable<FileSystemHandle> }).values();
  for await (const entry of entries) {
    if (entry.kind === 'file') {
      const file = await (entry as FileSystemFileHandle).getFile();
      // Only include relevant files
      if (
        file.name.endsWith('.safetensors') ||
        file.name.endsWith('.gguf') ||
        file.name.endsWith('.json') ||
        file.name === 'tokenizer.model'
      ) {
        files.push(file);
      }
    }
  }
  return files;
}

export default convertModel;
