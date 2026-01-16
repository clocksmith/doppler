/**
 * RDRR Group Accessors
 *
 * Functions for accessing component groups from the current manifest.
 *
 * @module formats/rdrr/groups
 */

import type { ComponentGroup } from './types.js';

export declare function getGroup(groupId: string): ComponentGroup | null;

export declare function getGroupIds(): string[];

export declare function getShardsForGroup(groupId: string): number[];

export declare function getTensorsForGroup(groupId: string): string[];

export declare function getShardsForExpert(layerIdx: number, expertIdx: number): number[];

export declare function getTensorsForExpert(layerIdx: number, expertIdx: number): string[];

export declare function getExpertBytes(): number;

export declare function getLayerGroupIds(): string[];

export declare function getExpertGroupIds(layerIdx: number): string[];
