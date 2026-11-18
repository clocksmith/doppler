self.onmessage = async (/** @type {MessageEvent<import('./recording-import-worker.js').RecordingImportRequest>} */ { data: { file, limit } }) => {
  try {
    const head = new Uint8Array(await file.slice(0, 2).arrayBuffer());
    const compressed = head[0] === 0x1f && head[1] === 0x8b;
    if (compressed && typeof DecompressionStream !== 'function') throw new Error('This browser cannot open compressed recordings.');
    const stream = compressed ? file.stream().pipeThrough(new DecompressionStream('gzip')) : file.stream();
    const reader = stream.getReader();
    const decoder = new TextDecoder('utf-8', { fatal: true });
    const parts = [];
    let byteLength = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      byteLength += value.byteLength;
      if (byteLength > limit) {
        await reader.cancel();
        throw new Error('Expanded recording exceeds the import limit.');
      }
      parts.push(decoder.decode(value, { stream: true }));
    }
    parts.push(decoder.decode());
    self.postMessage({ value: JSON.parse(parts.join('')), byteLength });
  } catch (error) {
    self.postMessage({ error: error.message || 'Could not open the recording.' });
  }
};
