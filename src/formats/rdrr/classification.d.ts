/**
 * RDRR Tensor Classification
 *
 * Functions for classifying tensors into component groups.
 *
 * @module formats/rdrr/classification
 */

import type { ModelType, ComponentGroupType } from './types.js';

/**
 * Classify a tensor into a component group based on its name and model type.
 */
export declare function classifyTensor(name: string, modelType: ModelType): string;

/**
 * Get the component group type from a group ID
 */
export declare function getGroupType(groupId: string, modelType: ModelType): ComponentGroupType;

/**
 * Parse layer index from group ID
 */
export declare function parseGroupLayerIndex(groupId: string): number | undefined;

/**
 * Parse expert index from group ID
 */
export declare function parseGroupExpertIndex(groupId: string): number | undefined;

/**
 * Sort group IDs in loading order: embed → layers (by index) → head
 */
export declare function sortGroupIds(groupIds: string[]): string[];
