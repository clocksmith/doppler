/**
 * Test Model Generator
 *
 * Creates a tiny test model for validation purposes.
 *
 * @module converter/test-model
 */

import type { WriteResult } from './writer.js';

/**
 * Create a tiny test model with random weights.
 */
export declare function createTestModel(outputDir: string): Promise<WriteResult>;
