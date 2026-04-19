/**
 * RDRR Manifest Creation and Serialization
 *
 * @module formats/rdrr/manifest
 */

import type {
  RDRRManifest,
  CreateManifestOptions,
} from './types.js';

export declare function generateShardFilename(index: number): string;

export declare function calculateShardCount(totalSize: number, shardSize?: number): number;

export declare function createManifest(options: CreateManifestOptions): RDRRManifest;

export declare function serializeManifest(manifest: RDRRManifest): string;

export declare function getManifestUrl(baseUrl: string): string;
