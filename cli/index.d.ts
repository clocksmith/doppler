#!/usr/bin/env node
/**
 * DOPPLER CLI - Unified testing, benchmarking, and debugging
 *
 * Usage:
 *   npx tsx cli/index.ts --config <ref>
 *
 * Examples:
 *   doppler --config ./tmp-bench.json        # Bench config file
 *   doppler --config ./tmp-gemma3-debug.json # Debug config file
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

export declare function parseArgs(argv: string[]): CLIOptions;
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
