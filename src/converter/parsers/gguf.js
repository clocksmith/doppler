import { bindQwen4ExpGGUFTable } from '../../formats/gguf/qwen4-exp-source.js';
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

  const sources = (adapter.files ?? [file]).map(normalizeTensorSource);
  if (sources.length === 0) throw new Error('GGUF input requires at least one source.');
  const pieces = [];
  for (const source of sources) {
    if (signal?.aborted) throw toAbortError();
    pieces.push({ source, info: await parseGGUFHeaderFromSource(source) });
  }
  const split = pieces.length > 1 || Number(pieces[0].info.metadata?.['split.count'] ?? 1) > 1;
  if (split) {
    const count = pieces[0].info.metadata?.['split.count'];
    const total = pieces[0].info.metadata?.['split.tensors.count'];
    if (!Number.isSafeInteger(count) || count < 2 || count !== pieces.length
        || !Number.isSafeInteger(total) || total < 1) {
      throw new Error('GGUF split requires every declared part and a valid total tensor count.');
    }
    const indices = new Set();
    for (const { info } of pieces) {
      const index = info.metadata?.['split.no'];
      if (!Number.isSafeInteger(index) || index < 0 || index >= count || indices.has(index)
          || info.metadata?.['split.count'] !== count || info.metadata?.['split.tensors.count'] !== total) {
        throw new Error('GGUF split has duplicate, missing or inconsistent part metadata.');
      }
      indices.add(index);
    }
    pieces.sort((a, b) => a.info.metadata['split.no'] - b.info.metadata['split.no']);
    if (pieces.reduce((n, piece) => n + piece.info.tensors.length, 0) !== total) {
      throw new Error('GGUF split tensor count differs from its declared total.');
    }
    for (const { info } of pieces.slice(1)) {
      for (const [key, value] of Object.entries(info.metadata)) {
        if (key.startsWith('split.')) continue;
        if (JSON.stringify(value) !== JSON.stringify(pieces[0].info.metadata[key])) {
          throw new Error(`GGUF split metadata differs from part zero: ${key}.`);
        }
      }
    }
  }
  const { source, info: primary } = pieces[0];
  const names = new Set();
  const allTensors = pieces.flatMap(piece => piece.info.tensors.map(tensor => {
    if (names.has(tensor.name)) throw new Error(`GGUF split contains duplicate tensor ${tensor.name}.`);
    if (!Number.isSafeInteger(tensor.offset) || tensor.offset < 0 || !Number.isSafeInteger(tensor.size)
        || tensor.size < 1 || tensor.offset > piece.source.size - tensor.size) {
      throw new Error(`GGUF tensor ${tensor.name} exceeds its source byte extent.`);
    }
    names.add(tensor.name);
    return { ...tensor, source: piece.source };
  }));
  const ggufInfo = { ...primary, tensors: allTensors };
  const metadata = ggufInfo.metadata ?? {};
  const native = normalizeQwenGGUFSource(ggufInfo.architecture, metadata);
  const ngramTable = ggufInfo.architecture === 'qwen4exp' ? bindQwen4ExpGGUFTable(native.config, allTensors) : null;
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
      ...(tensor === ngramTable?.table ? { name: ngramTable.name, originalName: tensor.name, role: 'ngram_table', group: 'ngram_table' } : {}),
      sourceShape: [...tensor.shape],
      shape: native && /ffn_gate_inp_shexp\.weight$/u.test(tensor.name) && tensor.shape.length === 1
        ? [1, ...tensor.shape] : [...tensor.shape].reverse(),
      ...(tensor.shape.length >= 2 || (native && /ffn_gate_inp_shexp\.weight$/u.test(tensor.name)) ? { layout: 'row' } : {}),
      file: tensor.source.file,
      source: tensor.source,
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
