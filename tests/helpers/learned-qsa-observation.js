import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { QsaIndexerCache } from '../../src/inference/pipelines/text/indexed-attention/qsa-cache.js';
import { readBuffer, readBufferSlice } from '../../src/memory/buffer-pool.js';

// Test-only readback instrumentation. It returns the original selection object
// untouched, never feeds observations into execution, and restores the method.
export function observeLearnedQsa(layerIds, reference, outputPrefix) {
  const original = QsaIndexerCache.prototype.select;
  const records = [], errors = [];
  const hash = bytes => createHash('sha256').update(new Uint8Array(bytes)).digest('hex');
  QsaIndexerCache.prototype.select = async function(target, layer, query, call) {
    const result = await original.call(this, target, layer, query, call);
    if (!layerIds.includes(layer)) return result;
    try {
      assert.equal(target, null, 'QSA observations require the explicit immediate lane');
      const expected = reference[layer];
      const positions = Object.keys(expected).map(Number).filter(p => p >= call.startPos && p < call.startPos + call.numTokens);
      const meta = new Uint32Array(await readBuffer(result.meta, call.numTokens * 8));
      const rows = [];
      for (const position of positions) {
        const row = position - call.startPos, count = meta[row * 2], mode = meta[row * 2 + 1];
        assert.equal(mode, 1, 'the upstream sparse reference only records sparse rows');
        assert.ok(count <= result.maxSelected);
        const indices = new Uint32Array(await readBufferSlice(result.selection, row * result.maxSelected * 4, count * 4));
        const completeRows = Math.floor((position + 1) / this.contract.compressRatio) * this.contract.compressRatio;
        const blocks = [...new Set([...indices].filter(p => p < completeRows).map(p => Math.floor(p / this.contract.compressRatio)))];
        const source = expected[position];
        const stride = result.scores.shape[1];
        const scores = new Float32Array(await readBufferSlice(result.scores.buffer, row * stride * 4, stride * 4));
        const queryValues = new Float32Array(await readBufferSlice(query.buffer,
          row * query.shape[1] * 4, query.shape[1] * 4));
        rows.push({ position, count, selectedSha256: hash(indices), sourceSelectedSha256: source.selectedSha256,
          blocks, addedBlocks: blocks.filter(b => !source.selectedBlocks.includes(b)),
          removedBlocks: source.selectedBlocks.filter(b => !blocks.includes(b)),
          sourceBlockMargin: source.blockMargin, scores: Array.from(scores), query: Array.from(queryValues) });
      }
      const { rawKeys, blockKeys } = this.buffers(layer);
      const visible = call.startPos + call.numTokens, bytesPerRow = this.contract.headDim * 4;
      const raw = await readBuffer(rawKeys, visible * bytesPerRow);
      const pooled = await readBuffer(blockKeys, Math.floor(visible / this.contract.compressRatio) * bytesPerRow);
      const rawFile = `${outputPrefix}.qsa${layer}.raw.f32`, pooledFile = `${outputPrefix}.qsa${layer}.pooled.f32`;
      await fs.writeFile(rawFile, Buffer.from(raw)); await fs.writeFile(pooledFile, Buffer.from(pooled));
      records.push({ layer, ...call, headDim: this.contract.headDim, queryHeads: this.contract.queryHeads,
        rawFile, rawSha256: hash(raw), pooledFile, pooledSha256: hash(pooled), rows });
    } catch (error) {
      // The caller still receives and disposes its selection. The harness then
      // fails explicitly after normal execution has settled.
      errors.push(String(error));
    }
    return result;
  };
  return { records, errors, restore() { QsaIndexerCache.prototype.select = original; } };
}
