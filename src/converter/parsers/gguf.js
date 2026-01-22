import { ConvertStage } from '../core.js';

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

  return {
    format: 'gguf',
    tensors: ggufInfo.tensors.map((tensor) => ({
      ...tensor,
      file: source.file,
      source,
      offset: tensor.offset,
    })),
    config: ggufInfo.config,
    architecture: ggufInfo.architecture,
    quantization: ggufInfo.quantization,
    tensorDataOffset: ggufInfo.tensorDataOffset,
    file: source.file,
    source,
  };
}
