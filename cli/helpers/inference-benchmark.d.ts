/**
 * Inference benchmark runner (Playwright + browser harness).
 *
 * Uses a persistent Playwright profile to keep OPFS state between runs.
 */

import type { CLIOptions } from './types.js';

export function runFullInferenceBenchmark(opts: CLIOptions): Promise<any>;

export function formatBenchmarkResult(result: any): void;
