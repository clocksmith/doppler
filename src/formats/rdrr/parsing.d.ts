/**
 * RDRR Parsing Functions
 *
 * @module formats/rdrr/parsing
 */

import type { RDRRManifest, ShardInfo, TensorMap } from './types.js';

export declare function parseManifest(jsonString: string): RDRRManifest;

export declare function parseTensorMap(jsonString: string): TensorMap;

export declare function getManifest(): RDRRManifest | null;

export declare function setManifest(manifest: RDRRManifest): void;

export declare function clearManifest(): void;

export declare function getShardInfo(index: number): ShardInfo | null;

export declare function getShardCount(): number;

export declare function isMoE(): boolean;
