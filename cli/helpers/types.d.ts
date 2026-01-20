/**
 * CLI Types - Shared type definitions for DOPPLER CLI
 */

import type { RuntimeConfigSchema } from '../../src/config/schema/index.js';

export type Command = 'run' | 'test' | 'bench' | 'debug' | 'convert' | 'tool';

export type TestSuite =
  | 'kernels'          // Kernel correctness tests
  | 'demo'             // Demo UI test (model load + generate via app)
  | 'converter'        // Converter UI test
  | 'inference'        // Quick inference validation
  | 'simulation'       // Pod simulation init and stats
  | 'training'         // Training correctness tests
  | 'quick'            // Quick validation (subset of kernels)
  | 'all';             // All tests

export type BenchSuite =
  | 'kernels'          // Kernel microbenchmarks
  | 'inference'        // Full inference benchmark (E2E generation)
  | 'loading'          // Model loading to GPU timing
  | 'system'           // Storage/OPFS benchmarks
  | 'all';             // All benchmarks

export type SuiteType = TestSuite | BenchSuite;

export interface CLIOptions {
  command: Command | null;
  suite: SuiteType | null;
  tool: string | null;
  model: string | null;
  baseUrl: string | null;
  /** Config preset or path (resolves: name -> path -> URL -> inline JSON) */
  config: string | null;
  /** Loaded runtime config (merged with defaults) */
  runtimeConfig: RuntimeConfigSchema | null;
  /** Config inheritance chain for debugging (e.g., ['debug', 'default']) */
  configChain: string[] | null;
  /** Serve files directly from disk via Playwright routing (no dev server). */
  noServer: boolean | null;
  /** Run headless (no browser window). */
  headless: boolean | null;
  /** Position browser window off-screen to avoid focus stealing. */
  minimized: boolean | null;
  /** Try to connect to existing Chrome via CDP before launching new instance.
   *  Avoids focus stealing by reusing already-open browser window.
   *  Start Chrome with: chrome --remote-debugging-port=9222 */
  reuseBrowser: boolean | null;
  /** CDP endpoint URL for reuseBrowser mode. */
  cdpEndpoint: string | null;
  verbose: boolean;
  filter: string | null;
  timeout: number | null;
  output: string | null;
  html: string | null;       // HTML report path (bench only)
  compare: string | null;    // Compare against baseline
  /** Playwright persistent profile directory.
   *  Controls browser storage persistence, including OPFS model cache. */
  profileDir: string | null;
  retries: number | null;    // Number of retries on failure
  quiet: boolean;            // Suppress JSON output
  help: boolean;
  /** Run in performance/benchmark mode (measure throughput instead of correctness). */
  perf: boolean;
}

// Alias for backward compatibility
export type TestOptions = CLIOptions;

export interface TestResult {
  name: string;
  passed: boolean;
  duration: number;
  error?: string;
}

export interface SuiteResult {
  suite: string;
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  results: TestResult[];
}

export interface ComparisonResult {
  metric: string;
  baseline: number;
  current: number;
  delta: number;
  deltaPercent: number;
  improved: boolean;
}

export interface RegressionSummary {
  thresholdPercent: number;
  regressions: ComparisonResult[];
  hasRegression: boolean;
}

export interface TTestResult {
  tStatistic: number;
  degreesOfFreedom: number;
  pValue: number;
  significant: boolean;
  meanA: number;
  meanB: number;
  stdA: number;
  stdB: number;
}
