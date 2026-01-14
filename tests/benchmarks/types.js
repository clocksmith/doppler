/**
 * Benchmark Types
 *
 * JavaScript module for the DOPPLER benchmark harness.
 * Follows the JSON schema defined in docs/spec/BENCHMARK_SCHEMA.json
 *
 * @module tests/benchmark/types
 */

import { DEFAULT_BENCHMARK_RUN_CONFIG } from '../../src/config/schema/benchmark.schema.js';

/**
 * Default benchmark configuration
 */
export const DEFAULT_BENCHMARK_CONFIG = {
  ...DEFAULT_BENCHMARK_RUN_CONFIG,
};
