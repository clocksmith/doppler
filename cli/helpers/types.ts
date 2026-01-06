/**
 * CLI Types - Shared type definitions for DOPPLER CLI
 */

import type { RuntimeConfigSchema, KernelPlanSchema } from '../../src/config/schema/index.js';
export type Command = 'run' | 'test' | 'bench' | 'debug';

export type TestSuite =
  | 'kernels'          // Kernel correctness tests (renamed from 'correctness')
  | 'correctness'      // DEPRECATED: alias for 'kernels'
  | 'demo'             // Demo UI test (model load + generate via app)
  | 'converter'        // Converter UI test
  | 'inference'        // Quick inference validation
  | 'quick'            // Quick validation (subset of kernels)
  | 'all';             // All tests

export type BenchSuite =
  | 'kernels'          // Kernel microbenchmarks
  | 'inference'        // Full inference benchmark (E2E generation)
  | 'loading'          // Model loading to GPU timing
  | 'system'           // Storage/OPFS benchmarks
  | 'all';             // All benchmarks

// Legacy suite types for backward compatibility
export type LegacySuite =
  | 'bench:kernels'
  | 'bench:pipeline'
  | 'bench:system';

export type SuiteType = TestSuite | BenchSuite | LegacySuite;

export interface CLIOptions {
  /** Raw CLI flags passed by the user (for config override precedence). */
  cliFlags: Set<string>;
  command: Command;
  suite: SuiteType;
  model: string;
  baseUrl: string;
  /** Config preset or path (resolves: name -> path -> URL -> inline JSON) */
  config: string | null;
  /** Loaded runtime config (merged with defaults) */
  runtimeConfig: RuntimeConfigSchema | null;
  /** Config inheritance chain for debugging (e.g., ['debug', 'default']) */
  configChain: string[] | null;
  /** Dump resolved config and exit */
  dumpConfig: boolean;
  /** List available presets and exit */
  listPresets: boolean;
  /** Serve files directly from disk via Playwright routing (no dev server). */
  noServer: boolean;
  /** Run headless (no browser window). Default is headless with real GPU. */
  headless: boolean;
  /** Position browser window off-screen to avoid focus stealing. */
  minimized: boolean;
  /** Try to connect to existing Chrome via CDP before launching new instance.
   *  Avoids focus stealing by reusing already-open browser window.
   *  Start Chrome with: chrome --remote-debugging-port=9222 */
  reuseBrowser: boolean;
  /** CDP endpoint URL for reuseBrowser mode. Default: http://localhost:9222 */
  cdpEndpoint: string;
  verbose: boolean;
  filter: string | null;
  timeout: number;
  output: string | null;
  html: string | null;       // HTML report path (bench only)
  warmup: number;
  runs: number;
  maxTokens: number;         // For inference benchmarks
  temperature: number;       // For inference benchmarks
  noChat: boolean;           // Disable chat template
  prompt: string;            // Prompt size preset: xs, short, medium, long
  promptProvided: boolean;   // Whether --prompt was explicitly set
  text: string | null;       // Custom prompt text (overrides prompt)
  file: string | null;       // Load prompt from file (overrides prompt)
  compare: string | null;    // Compare against baseline
  trace: string | null;      // Debug trace preset: quick, layers, attention, full
  /** Layer filter for debug trace categories (does NOT enable recorder batching). */
  traceLayers: number[] | null;
  debugLayers: number[] | null; // Specific layers to debug
  /** Playwright persistent profile directory.
   *  Controls browser storage persistence, including OPFS model cache. */
  profileDir: string | null;
  retries: number;           // Number of retries on failure
  quiet: boolean;            // Suppress JSON output
  help: boolean;
  /** Run in performance/benchmark mode (measure throughput instead of correctness). */
  perf: boolean;
  /** Enable GPU timestamp profiling for per-kernel timing.
   *  Requires 'timestamp-query' WebGPU feature. */
  gpuProfile: boolean;
  /** Kernel plan overrides (bench/debug). */
  kernelProfile: string | null;
  kernelPlan: KernelPlanSchema | null;

  // Debug mode options
  /** Enable verbose debug output during inference */
  debug: boolean;
  /** Stop at specific layer for inspection (debug mode) */
  layer: number | null;
  /** Number of tokens to encode before stopping (debug mode) */
  tokens: number | null;
  /** Specific kernel to trace (debug mode) */
  kernel: string | null;

  // Warm mode options (preserve model in GPU RAM)
  /** Skip model loading, assume pipeline already exists in window.pipeline */
  skipLoad: boolean;
  /** Keep browser open with model loaded for subsequent runs */
  warm: boolean;
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
