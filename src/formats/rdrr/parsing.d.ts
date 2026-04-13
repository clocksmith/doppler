/**
 * RDRR Parsing Functions
 *
 * @module formats/rdrr/parsing
 */

import type { RDRRManifest, ShardInfo, TensorMap } from './types.js';
import type { RuntimeModelContract } from '../../inference/runtime-model.js';

export declare function parseManifest(jsonString: string): RDRRManifest;
export declare function getExpectedShardHash(
  shard: Partial<ShardInfo> | Record<string, unknown> | null | undefined,
  manifestHashAlgorithm?: string | null
): string;

export declare function parseTensorMap(jsonString: string): TensorMap;

export declare function getManifest(): (RDRRManifest | RuntimeModelContract) | null;

export declare function setManifest(manifest: RDRRManifest | RuntimeModelContract): void;

export declare function clearManifest(): void;

export declare function getShardInfo(index: number): ShardInfo | null;

export declare function getShardCount(): number;

export declare function isMoE(): boolean;
