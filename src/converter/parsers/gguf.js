import { ConvertStage } from '../core.js';
import { extractArchitecture } from '../artifact-identity.js';
import { normalizeQwenGGUFSource } from '../../formats/gguf/qwen-source.js';
import { buildGGUFTokenizer } from '../../formats/gguf/tokenizer.js';

function toAbortError(message = 'Cancelled') {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

export async function parseGGUFModel(adapter) {
  const {
    file,
    parseGGUFHeaderFromSource,
    normalizeTensorSource,
    onProgress,
    signal,
  } = adapter;

  onProgress?.({
    stage: ConvertStage.PARSING,
    message: 'Parsing GGUF header...',
  });

  if (signal?.aborted) throw toAbortError();

  const source = normalizeTensorSource(file);
  const ggufInfo = await parseGGUFHeaderFromSource(source);
  const metadata = ggufInfo.metadata ?? {};
  const native = normalizeQwenGGUFSource(ggufInfo.architecture, metadata);
  const tokenizer = native ? buildGGUFTokenizer(metadata) : null;
  const tensors = ggufInfo.tensors.filter(tensor => {
    const layer = /^blk\.(\d+)\./u.exec(tensor.name);
    return !native || !layer || Number(layer[1]) < native.textLayers;
  });

  return {
    format: 'gguf',
    modelType: native?.modelType,
    normalizedArchitecture: native ? extractArchitecture(native.config, null) : null,
    excludedTensors: ggufInfo.tensors.filter(tensor => !tensors.includes(tensor)).map(tensor => ({ name: tensor.name, reason: 'next-token-prediction layer outside text decoder' })),
    tokenizerJson: tokenizer?.tokenizerJson ?? null,
    tokenizerConfig: tokenizer?.tokenizerConfig ?? null,
    tensors: tensors.map((tensor) => ({
      ...tensor,
      sourceShape: [...tensor.shape],
      shape: native && /ffn_gate_inp_shexp\.weight$/u.test(tensor.name) && tensor.shape.length === 1
        ? [1, ...tensor.shape] : [...tensor.shape].reverse(),
      ...(tensor.shape.length >= 2 || (native && /ffn_gate_inp_shexp\.weight$/u.test(tensor.name)) ? { layout: 'row' } : {}),
      file: source.file,
      source,
      offset: tensor.offset,
    })),
    config: native ? { ...ggufInfo.config, ...native.config } : ggufInfo.config,
    architecture: ggufInfo.architecture,
    quantization: ggufInfo.quantization,
    tensorDataOffset: ggufInfo.tensorDataOffset,
    file: source.file,
    source,
  };
}
