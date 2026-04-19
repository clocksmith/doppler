/**
 * RDRR Group Accessors
 *
 * Functions for accessing component groups from the current manifest.
 *
 * @module formats/rdrr/groups
 */

import type { RDRRManifest } from './types.js';

export declare function getShardsForExpert(
  layerIdx: number,
  expertIdx: number,
  manifest?: RDRRManifest | null
): number[];

export declare function getTensorsForExpert(
  layerIdx: number,
  expertIdx: number,
  manifest?: RDRRManifest | null
): string[];

export declare function getExpertBytes(manifest?: RDRRManifest | null): number;
