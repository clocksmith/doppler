import { LAYER_PREFIXES } from './layer-tensor-aliases.js';

/** @param {string} message @returns {never} */
function tableError(message) { throw new Error(`[NgramTable] ${message}`); }

/** @type {typeof import('./ngram-table-bindings.js').bindNgramTables} */
export function bindNgramTables(loader, contract, session) {
  /** @type {Map<number, import('../inference/pipelines/text/ngram/ngram-table.js').NgramTableReaderOptions>} */
  const tables = new Map();
  for (const layer of contract.layers) {
    const prefixes = LAYER_PREFIXES(layer.decoderLayerIndex).map(prefix => `${prefix}.ple.ple_embedding.ngram_embedding`);
    const matches = prefixes.filter(prefix => [...loader.tensorLocations.keys()].some(name => name.startsWith(`${prefix}.shard_`)));
    if (matches.length !== 1) tableError(`Missing or ambiguous table for layer ${layer.decoderLayerIndex}.`);
    const parts = ngramTablePartsFromTensorLocations(loader.tensorLocations,
      { tensorPrefix: matches[0], rows: layer.rows, rowWidth: contract.headDim });
    tables.set(layer.decoderLayerIndex, { rows: layer.rows, rowWidth: contract.headDim, dtype: contract.tableDtype,
      parts, session, rowBytesVerified: false,
      readRange: async (source, offset, length, signal) => {
        if (typeof source !== 'number' || !Number.isSafeInteger(source)) tableError('A bound table requires a numeric shard index.');
        signal?.throwIfAborted();
        const result = await loader.shardCache.loadRange(source, offset, length, { requireRange: true });
        signal?.throwIfAborted();
        return result;
      } });
  }
  return tables;
}

/** @type {typeof import('./ngram-table-bindings.js').ngramTablePartsFromTensorLocations} */
export function ngramTablePartsFromTensorLocations(tensorLocations, { tensorPrefix, rows, rowWidth }) {
  const rowBytes = rowWidth * 2;
  const parts = [];
  let rowBegin = 0;
  for (let k = 0; tensorLocations.has(`${tensorPrefix}.shard_${k}.weight`); k += 1) {
    const name = `${tensorPrefix}.shard_${k}.weight`;
    const location = /** @type {import('../inference/pipelines/text/ngram/ngram-table.js').NgramTableTensorLocation} */ (tensorLocations.get(name));
    if (!location || typeof location !== 'object') throw new Error(`[NgramTable] Invalid location for ${name}.`);
    if ('sourceTransform' in location && location.sourceTransform != null) throw new Error(`[NgramTable] ${name} must retain untransformed source rows.`);
    if (location.role !== 'ngram_table' || String(location.dtype).toUpperCase() !== 'BF16') {
      tableError(`${name} must be a retained BF16 ngram_table tensor; got role ${location.role}, dtype ${location.dtype}.`);
    }
    if ((location.spans?.length ?? 0) > 0 || !Number.isSafeInteger(location.shardIndex)) {
      tableError(`${name} must lie in one shard.`);
    }
    const [partRows, width] = location.shape ?? [];
    if (location.shape?.length !== 2 || width !== rowWidth || !Number.isSafeInteger(partRows) || partRows <= 0
      || location.size !== partRows * rowBytes) {
      tableError(`${name} has shape [${location.shape}] and ${location.size} bytes; rows of ${rowWidth} bf16 values are required.`);
    }
    parts.push({ rowBegin, rowEnd: rowBegin + partRows, source: location.shardIndex, byteOffset: location.offset, byteLength: location.size });
    rowBegin += partRows;
  }
  if (rowBegin !== rows) tableError(`${tensorPrefix} parts cover ${rowBegin} rows; the contract declares ${rows}.`);
  for (const name of tensorLocations.keys()) {
    if (!name.startsWith(`${tensorPrefix}.shard_`)) continue;
    const segment = name.slice(tensorPrefix.length + '.shard_'.length, -'.weight'.length);
    const k = /^(0|[1-9]\d*)$/.test(segment) ? Number(segment) : -1;
    if (!name.endsWith('.weight') || !(k >= 0 && k < parts.length)) tableError(`${name} is not part of the contiguous table shard sequence.`);
  }
  return parts;
}
