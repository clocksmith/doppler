#!/usr/bin/env node
/**
 * DOPPLER CLI - Unified testing, benchmarking, and debugging
 *
 * Usage:
 *   npx tsx cli/index.ts run                    # Serve demo page
 *   npx tsx cli/index.ts test <suite> [options] # Run tests
 *   npx tsx cli/index.ts bench <suite> [options] # Run benchmarks
 *   npx tsx cli/index.ts debug [options]        # Debug mode
 *   npx tsx cli/index.ts test simulation        # Simulation mode
 *
 * Examples:
 *   doppler run                              # Serve demo at :8080
 *   doppler test kernels --filter matmul     # Kernel correctness tests
 *   doppler bench inference                 # Full inference benchmark
 *   doppler debug --model gemma-1b --layer 5 # Inspect layer 5
 */

import type { Page } from 'playwright';
import type {
  CLIOptions,
  SuiteType,
  Command,
  TestResult,
  SuiteResult,
} from './helpers/types.js';

export type FlagHandler = (opts: CLIOptions, tokens: string[]) => void;
export interface FlagSpec {
  names: string[];
  handler: FlagHandler;
}

export declare const KERNEL_TESTS: readonly string[];
export declare const KERNEL_BENCHMARKS: readonly string[];
export declare const QUICK_TESTS: readonly string[];
export declare const KERNEL_PROFILE_PATHS: Record<string, string>;

export declare function parseArgs(argv: string[]): CLIOptions;
export declare function hasCliFlag(opts: CLIOptions, flags: string[]): boolean;
export declare function printHelp(): void;
export declare function printSummary(suites: SuiteResult[]): void;

export declare function runCorrectnessTests(
  page: Page,
  opts: CLIOptions,
  tests: readonly string[]
): Promise<SuiteResult>;

export declare function runKernelBenchmarks(
  page: Page,
  opts: CLIOptions
): Promise<SuiteResult>;

export declare function runInferenceTest(
  page: Page,
  opts: CLIOptions
): Promise<SuiteResult>;

export declare function runSimulationTest(
  page: Page,
  opts: CLIOptions
): Promise<SuiteResult>;

export declare function runDemoTest(
  page: Page,
  opts: CLIOptions
): Promise<SuiteResult>;

export declare function runConverterTest(
  page: Page,
  opts: CLIOptions
): Promise<SuiteResult>;

export declare function runPipelineBenchmark(
  page: Page,
  opts: CLIOptions
): Promise<SuiteResult>;

export declare function main(): Promise<void>;
