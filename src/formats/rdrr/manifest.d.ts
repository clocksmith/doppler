/**
 * RDRR Manifest Creation and Serialization
 *
 * @module formats/rdrr/manifest
 */

import type {
  RDRRManifest,
  ShardInfo,
  TensorMap,
  CreateManifestOptions,
} from './types.js';

export declare function generateShardFilename(index: number): string;

export declare function calculateShardCount(totalSize: number, shardSize?: number): number;

export declare function createShardLayout(
  totalSize: number,
  hashes: string[],
  shardSize?: number
): ShardInfo[];

export declare function createManifest(options: CreateManifestOptions): RDRRManifest;

export declare function serializeTensorMap(tensorMap: TensorMap): string;

export declare function serializeManifest(manifest: RDRRManifest): string;

export declare function getShardUrl(baseUrl: string, shardIndex: number): string;

export declare function getManifestUrl(baseUrl: string): string;
