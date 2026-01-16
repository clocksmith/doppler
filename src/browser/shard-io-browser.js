

import { generateShardFilename } from '../storage/rdrr-format.js';


export class BrowserShardIO {
  constructor(modelDir) {
    this.modelDir = modelDir;
  }

  
  static async create(modelId) {
    const opfsRoot = await navigator.storage.getDirectory();
    const modelsDir = await opfsRoot.getDirectoryHandle('models', { create: true });
    const modelDir = await modelsDir.getDirectoryHandle(modelId, { create: true });
    return new BrowserShardIO(modelDir);
  }

  
  async writeShard(index, data) {
    const filename = generateShardFilename(index);
    const fileHandle = await this.modelDir.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    // Use ArrayBuffer for FileSystemWritableFileStream compatibility
    const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    await writable.write(buffer);
    await writable.close();
    return this.computeHash(data);
  }

  
  async computeHash(data) {
    // Use ArrayBuffer slice for SubtleCrypto compatibility
    const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = new Uint8Array(hashBuffer);
    return Array.from(hashArray)
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }

  
  async writeJson(filename, data) {
    const fileHandle = await this.modelDir.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(JSON.stringify(data, null, 2));
    await writable.close();
  }

  
  async writeFile(filename, data) {
    const fileHandle = await this.modelDir.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    if (typeof data === 'string') {
      await writable.write(data);
    } else {
      const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      await writable.write(buffer);
    }
    await writable.close();
  }

  
  getModelDir() {
    return this.modelDir;
  }

  
  async clear() {
    const entries = this.modelDir.values();
    for await (const entry of entries) {
      await this.modelDir.removeEntry(entry.name);
    }
  }
}


export function isOPFSSupported() {
  return (
    typeof navigator !== 'undefined' &&
    'storage' in navigator &&
    'getDirectory' in (navigator.storage)
  );
}
