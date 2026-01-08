/**
 * node.ts - Node.js I/O Adapter for Shard Packer
 *
 * Implements ShardIO interface using Node.js fs APIs.
 *
 * @module converter/io/node
 */

import type { ShardIO } from '../shard-packer.js';

export interface NodeShardIOOptions {
  useBlake3?: boolean;
}

/**
 * Node.js implementation of ShardIO interface.
 */
export declare class NodeShardIO implements ShardIO {
  constructor(outputDir: string, options?: NodeShardIOOptions);

  init(): Promise<void>;
  writeShard(index: number, data: Uint8Array): Promise<string>;
  computeHash(data: Uint8Array): Promise<string>;
  writeJson(filename: string, data: unknown): Promise<void>;
  writeFile(filename: string, data: string | Uint8Array): Promise<void>;
  getOutputDir(): string;
}
