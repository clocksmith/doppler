export function createRowChunks({ rows, rowChunkRows, rowSourceBytes }) {
  const chunks = [];
  for (let rowStart = 0; rowStart < rows; rowStart += rowChunkRows) {
    const rowCount = Math.min(rowChunkRows, rows - rowStart);
    chunks.push({
      rowStart,
      rowCount,
      offset: rowStart * rowSourceBytes,
      length: rowCount * rowSourceBytes,
    });
  }
  return chunks;
}

export async function mapOrderedChunkBatches({ chunks, batchSize, transform, consume }) {
  for (let batchStart = 0; batchStart < chunks.length; batchStart += batchSize) {
    const batch = chunks.slice(batchStart, batchStart + batchSize);
    const results = await Promise.all(batch.map((chunk) => transform(chunk)));
    for (const result of results) await consume(result);
  }
}
