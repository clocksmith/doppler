/**
 * RDRR Writer Utility Functions
 *
 * @module converter/writer/utils
 */

import type { HashAlgorithm } from './types.js';

export declare function computeHash(data: Uint8Array, algorithm?: HashAlgorithm): Promise<string>;

export declare function alignOffset(offset: number, alignment?: number): number;

export declare function createPadding(size: number): Uint8Array;

export declare function getBytesPerElement(dtype: string): number;

export declare function transpose2D(data: Uint8Array, rows: number, cols: number, dtype: string): Uint8Array;
