export async function withMappedReadBuffer(buffer, read) {
  let mapped = false;
  try {
    await buffer.mapAsync(GPUMapMode.READ);
    mapped = true;
    return await read(buffer.getMappedRange());
  } finally {
    if (mapped) {
      buffer.unmap();
    }
  }
}

export async function withMappedReadBuffers(entries, read) {
  const mappedEntries = [];
  try {
    await Promise.all(entries.map(async (entry) => {
      await entry.buffer.mapAsync(GPUMapMode.READ);
      mappedEntries.push(entry);
    }));
    const ranges = entries.map((entry) => {
      if (entry.offset != null || entry.size != null) {
        return entry.buffer.getMappedRange(
          entry.offset ?? 0,
          entry.size ?? (entry.buffer.size - (entry.offset ?? 0))
        );
      }
      return entry.buffer.getMappedRange();
    });
    return await read(ranges);
  } finally {
    for (let index = mappedEntries.length - 1; index >= 0; index -= 1) {
      mappedEntries[index].buffer.unmap();
    }
    for (const entry of entries) {
      if (entry.destroy === true) {
        entry.buffer.destroy();
      }
    }
  }
}
