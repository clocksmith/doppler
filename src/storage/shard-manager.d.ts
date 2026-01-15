import type { RDRRManifest, HashAlgorithm } from './rdrr-format.js';
import type { OpfsPathConfigSchema } from '../config/schema/loading.schema.js';

export { getManifest } from './rdrr-format.js';

export interface ShardWriteOptions {
  verify?: boolean;
}

export interface ShardWriteResult {
  success: boolean;
  hash: string | null;
}

export interface ShardReadOptions {
  verify?: boolean;
}

export interface IntegrityResult {
  valid: boolean;
  missingShards: number[];
  corruptShards: number[];
}

export interface ModelInfo {
  exists: boolean;
  shardCount: number;
  totalSize: number;
  hasManifest: boolean;
}

export interface StreamingHasher {
  update(data: Uint8Array): void;
  finalize(): Promise<Uint8Array>;
}

export function setOpfsPathConfig(config: OpfsPathConfigSchema): void;
export function getOpfsPathConfig(): OpfsPathConfigSchema;
export function getHashAlgorithm(): HashAlgorithm | null;
export function hexToBytes(hex: string): Uint8Array;
export function computeBlake3(data: Uint8Array | ArrayBuffer): Promise<string>;
export function computeSHA256(data: Uint8Array | ArrayBuffer): Promise<string>;
export function computeHash(data: Uint8Array | ArrayBuffer, algorithm?: HashAlgorithm): Promise<string>;
export function createStreamingHasher(): Promise<StreamingHasher>;

export function initOPFS(): Promise<void>;
export function openModelDirectory(modelId: string): Promise<FileSystemDirectoryHandle>;
export function getCurrentModelDirectory(): FileSystemDirectoryHandle | null;

export function writeShard(
  shardIndex: number,
  data: ArrayBuffer,
  options?: ShardWriteOptions
): Promise<ShardWriteResult>;

export function loadShard(
  shardIndex: number,
  options?: ShardReadOptions
): Promise<ArrayBuffer>;

export function loadShardSync(
  shardIndex: number,
  offset?: number,
  length?: number
): Promise<Uint8Array>;

export function shardExists(shardIndex: number): Promise<boolean>;
export function verifyIntegrity(): Promise<IntegrityResult>;
export function deleteShard(shardIndex: number): Promise<boolean>;
export function deleteModel(modelId: string): Promise<boolean>;
export function listModels(): Promise<string[]>;
export function getModelInfo(modelId: string): Promise<ModelInfo>;
export function modelExists(modelId: string): Promise<boolean>;

export function saveManifest(manifestJson: string): Promise<void>;
export function loadManifestFromOPFS(): Promise<string>;
export function loadTensorsFromOPFS(): Promise<string | null>;
export function saveTokenizer(tokenizerJson: string): Promise<void>;
export function loadTokenizerFromOPFS(): Promise<string | null>;

export function cleanup(): void;

export class OpfsShardStore {
  constructor(modelId: string);
  read(shardIndex: number, offset: number, length: number): Promise<Uint8Array>;
  write(shardIndex: number, data: Uint8Array): Promise<void>;
  exists(shardIndex: number): Promise<boolean>;
  delete(shardIndex: number): Promise<void>;
  list(): Promise<number[]>;
}
