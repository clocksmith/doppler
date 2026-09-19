import type { RuntimeModelContract } from '../inference/runtime-model.js';
import type { SourceRuntimeShardSource } from '../formats/source-runtime.js';

export interface CreateSourceStorageContextOptions {
  model?: RuntimeModelContract | null;
  manifest?: RuntimeModelContract | null;
  shardSources?: SourceRuntimeShardSource[] | null;
  readRange: (
    path: string,
    offset: number,
    length: number
  ) => Promise<ArrayBuffer | Uint8Array>;
  streamRange?: (
    path: string,
    offset: number,
    length: number,
    options?: { chunkBytes?: number }
  ) => AsyncIterable<ArrayBuffer | Uint8Array>;
  readText?: (path: string) => Promise<string | Record<string, unknown> | null | undefined>;
  readBinary?: (path: string) => Promise<ArrayBuffer | Uint8Array>;
  close?: (() => Promise<void>) | null;
  tokenizerJsonPath?: string | null;
  tokenizerConfigPath?: string | null;
  tokenizerModelPath?: string | null;
  verifyHashes?: boolean;
  sourceHashesTrusted?: boolean;
}

export interface SourceStorageContext {
  preflight?: () => Promise<void>;
  loadShard: (index: number) => Promise<ArrayBuffer | Uint8Array>;
  loadShardRange: ((
    index: number,
    offset?: number,
    length?: number | null
  ) => Promise<ArrayBuffer | Uint8Array>) | null;
  streamShardRange: ((
    index: number,
    offset?: number,
    length?: number | null,
    options?: { chunkBytes?: number }
  ) => AsyncIterable<Uint8Array>) | null;
  loadTokenizerJson: (() => Promise<Record<string, unknown> | null>) | null;
  loadTokenizerModel: ((pathHint?: string) => Promise<ArrayBuffer | null>) | null;
  loadAuxiliaryFile: ((path: string) => Promise<ArrayBuffer | null>) | null;
  loadTensorsJson?: (() => Promise<string | Record<string, unknown> | null>) | null;
  verifyHashes: boolean;
  close?: (() => Promise<void>) | null;
}

export declare function createSourceStorageContext(
  options: CreateSourceStorageContextOptions
): SourceStorageContext;
