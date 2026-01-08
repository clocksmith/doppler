/**
 * node.ts - Node.js I/O Adapter for Shard Packer
 *
 * Implements ShardIO interface using Node.js fs APIs.
 *
 * @module converter/io/node
 */

import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { createHash } from 'crypto';
import { generateShardFilename } from '../../storage/rdrr-format.js';

/**
 * Node.js implementation of ShardIO interface.
 */
export class NodeShardIO {
  #outputDir;
  #useBlake3;

  constructor(outputDir, options = {}) {
    this.#outputDir = outputDir;
    this.#useBlake3 = options.useBlake3 ?? false;
  }

  /**
   * Ensure output directory exists.
   */
  async init() {
    await mkdir(this.#outputDir, { recursive: true });
  }

  /**
   * Write shard data to file, returns hash.
   */
  async writeShard(index, data) {
    const filename = generateShardFilename(index);
    const filepath = join(this.#outputDir, filename);
    await writeFile(filepath, data);
    return this.computeHash(data);
  }

  /**
   * Compute hash of data using SHA-256 or BLAKE3.
   */
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

  /**
   * Write a JSON file to the output directory.
   */
  async writeJson(filename, data) {
    const filepath = join(this.#outputDir, filename);
    await writeFile(filepath, JSON.stringify(data, null, 2));
  }

  /**
   * Write raw file to output directory.
   */
  async writeFile(filename, data) {
    const filepath = join(this.#outputDir, filename);
    await writeFile(filepath, data);
  }

  /**
   * Get output directory path.
   */
  getOutputDir() {
    return this.#outputDir;
  }
}
