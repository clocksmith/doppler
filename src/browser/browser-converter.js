

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
} from '../converter/shard-packer.js';
import { BrowserShardIO, isOPFSSupported } from './shard-io-browser.js';

// Import shared types and functions from convert-core
import {
  ConvertStage,
  sanitizeModelId,
  formatBytes,
  extractArchitecture,
  createManifest,
} from '../converter/core.js';
import { buildManifestInference } from '../converter/manifest-inference.js';

import { detectPreset, resolvePreset } from '../config/index.js';

// Re-export types for consumers
export {
  ConvertStage,
};

// Re-export OPFS support check
export { isOPFSSupported as isConversionSupported };

// ============================================================================
// Main Convert Function
// ============================================================================


export async function convertModel(files, options = {}) {
  const { modelId: userModelId, onProgress, signal } = options;

  let modelId = null;
  let modelDir = null;
  const shardInfos = [];

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
    let modelInfo;
    let config = null;
    let tokenizerJson = null;

    if (format.type === 'gguf') {
      modelInfo = await parseGGUFModel(format.ggufFile, onProgress, signal);
    } else if (format.type === 'single') {
      const parsed = await parseSafetensorsFile(format.safetensorsFile);
      modelInfo = { tensors: parsed.tensors, config: parsed.config };
      if (auxiliary.config) {
        config = await parseConfigJson(auxiliary.config);
        modelInfo.config = config;
      }
    } else if (format.type === 'sharded' || format.type === 'sharded-no-index') {
      let indexJson = null;
      if (format.indexFile) {
        indexJson = await parseIndexJson(format.indexFile);
      }
      const parsed = await parseSafetensorsSharded(format.safetensorsFiles, indexJson);
      modelInfo = { tensors: parsed.tensors, config: parsed.config };
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
      totalSize: formatBytes(calculateTotalSize(modelInfo)),
    });

    // Open model directory in OPFS
    modelDir = await openModelDirectory(modelId);

    // Detect model type using preset system
    const rawConfig = (config || modelInfo.config || {});
    const presetId = detectPreset(rawConfig, modelInfo.architecture);
    if (presetId === 'transformer') {
      const modelType = rawConfig.model_type ?? 'unknown';
      throw new Error(
        `Unknown model family: architecture="${modelInfo.architecture || 'unknown'}", model_type="${modelType}"\n\n` +
        `DOPPLER requires a known model preset to generate correct inference config.\n` +
        `The manifest-first architecture does not support generic defaults.\n\n` +
        `Options:\n` +
        `  1. Wait for official support of this model family\n` +
        `  2. Create a custom preset in src/config/presets/models/\n` +
        `  3. File an issue at https://github.com/clocksmith/doppler/issues\n\n` +
        `Supported model families: gemma2, gemma3, llama3, qwen3, mixtral, deepseek, mamba`
      );
    }
    const preset = resolvePreset(presetId);
    const modelType = preset.modelType || 'transformer';
    const hfConfig = (config || (modelInfo.format === 'gguf' ? null : modelInfo.config));
    const ggufConfig = modelInfo.format === 'gguf' ? modelInfo.config : undefined;
    const architecture = extractArchitecture(hfConfig || {}, ggufConfig);
    const headDim = architecture.headDim || 64;
    const quantizationInfo = modelInfo.quantization
      ? { weights: modelInfo.quantization, compute: 'f16' }
      : null;
    const manifestInference = buildManifestInference(preset, rawConfig, headDim, quantizationInfo);

    // Create shard I/O adapter
    const shardIO = new BrowserShardIO(modelDir);

    // Create shard packer
    const packer = new ShardPacker(shardIO, { modelType });

    // Prepare tensors for packing
    const packerTensors = modelInfo.tensors.map(tensor => ({
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
        const data = await readTensorData(tensor);
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
    const parsedModel = {
      tensors: modelInfo.tensors.map(t => ({
        name: t.name,
        shape: t.shape,
        dtype: t.dtype,
        size: t.size,
        offset: t.offset,
      })),
      config: (config || modelInfo.config || {}),
      architecture: modelInfo.architecture,
      quantization: modelInfo.quantization || 'F16',
      tokenizerJson,
    };

    const manifest = createManifest(
      modelId,
      parsedModel,
      shardInfos,
      result.tensorLocations,
      {
        source: 'browser-converter',
        inference: manifestInference,
        modelType,
      }
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
      message: error.message,
      error: error,
    });

    throw error;
  }
}


async function parseGGUFModel(
  file,
  onProgress,
  signal
) {
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


function extractModelId(files, config) {
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


export async function pickModelFiles() {
  // Try directory picker first (for HuggingFace models)
  if ('showDirectoryPicker' in window) {
    try {
      const dirHandle = await window.showDirectoryPicker({
        mode: 'read',
      });
      return await collectFilesFromDirectory(dirHandle);
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      // Fall back to file picker
    }
  }

  // Fall back to file picker
  if ('showOpenFilePicker' in window) {
    const handles = await window.showOpenFilePicker({
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


async function collectFilesFromDirectory(
  dirHandle,
  files = []
) {
  const entries = dirHandle.values();
  for await (const entry of entries) {
    if (entry.kind === 'file') {
      const file = await entry.getFile();
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
