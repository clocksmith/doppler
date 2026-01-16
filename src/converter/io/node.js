

import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { createHash } from 'crypto';
import { generateShardFilename } from '../../storage/rdrr-format.js';


export class NodeShardIO {
  #outputDir;
  #useBlake3;

  constructor(outputDir, options = {}) {
    this.#outputDir = outputDir;
    this.#useBlake3 = options.useBlake3 ?? false;
  }

  
  async init() {
    await mkdir(this.#outputDir, { recursive: true });
  }

  
  async writeShard(index, data) {
    const filename = generateShardFilename(index);
    const filepath = join(this.#outputDir, filename);
    await writeFile(filepath, data);
    return this.computeHash(data);
  }

  
  async computeHash(data) {
    if (this.#useBlake3) {
      try {
        // Try to use blake3 package if available
        const blake3 = await import('blake3');
        return blake3.hash(data).toString('hex');
      } catch {
        // Fall back to SHA-256
      }
    }
    return createHash('sha256').update(data).digest('hex');
  }

  
  async writeJson(filename, data) {
    const filepath = join(this.#outputDir, filename);
    await writeFile(filepath, JSON.stringify(data, null, 2));
  }

  
  async writeFile(filename, data) {
    const filepath = join(this.#outputDir, filename);
    await writeFile(filepath, data);
  }

  
  getOutputDir() {
    return this.#outputDir;
  }
}
