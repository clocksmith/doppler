

export function isTensorSource(value) {
  return value && typeof value.readRange === 'function' && typeof value.size === 'number';
}

export function createFileTensorSource(file) {
  if (!file || typeof file.slice !== 'function') {
    throw new Error('File tensor source requires a File or Blob');
  }

  const name = typeof file.name === 'string' ? file.name : 'unknown';

  return {
    sourceType: 'file',
    name,
    size: file.size ?? 0,
    file,
    async readRange(offset, length) {
      if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
        return new ArrayBuffer(0);
      }
      const start = Math.max(0, offset);
      const end = Math.min(start + length, file.size ?? start + length);
      const blob = file.slice(start, end);
      return blob.arrayBuffer();
    },
  };
}

export function normalizeTensorSource(input) {
  if (isTensorSource(input)) {
    return input;
  }
  return createFileTensorSource(input);
}
