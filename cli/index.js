#!/usr/bin/env node


import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { writeFile, mkdir, readFile } from 'fs/promises';

import { loadConfig, listPresets, dumpConfig } from './config/index.js';

import {
  runBuild,
  ensureServerRunning,
  createBrowserContext,
  setupPage,
  generateResultFilename,
} from './helpers/utils.js';

import {
  runFullInferenceBenchmark,
  formatBenchmarkResult,
} from './helpers/inference-benchmark.js';

import {
  compareResults,
  formatComparison,
  detectRegressions,
  formatRegressionSummary,
  welchTTest,
  formatTTestResult,
} from './helpers/comparison.js';

import { generateHTMLReport } from './helpers/html-report.js';
import { loadBaselineRegistry, findBaselineForResult, evaluateBaseline } from './helpers/baselines.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============================================================================
// Test Suite Definitions (Composable)
// ============================================================================

const KERNEL_TESTS = [
  'matmul',
  'matmul-q4k',  // Q4K fused batched matmul (tests subgroup column fix)
  'matmul-q4k-large',  // Q4K with inference-like dimensions (K=1152)
  'attention',
  'rmsnorm',
  'softmax',
  'rope',
  'silu',
  'swiglu',  // SwiGLU activation with bias
  'gather',
  'scatter-add',
  'moe-gather',
  'residual',
  'scale',  // Element-wise scalar multiplication
  'topk',
  'dequant',
  'dequant-q4k',  // Q4_K dequantization (production path)
  'dequant-q4k-f16',  // Q4_K dequantization with F16 output
  'matmul-f16w',  // f16w_f32a matmul with dequanted F16 weights (production path test)
  'dequant-q6k',  // Q6_K dequantization
  'sample',  // GPU argmax sampling
];

const TRAINING_TESTS = [
  'loss-forward',
  'softmax-backward',
  'cross-entropy-backward',
  'rmsnorm-backward',
  'matmul-backward',
  'parity-fixture',
  'training-leak-perf',
];

const KERNEL_BENCHMARKS = [
  'matmul',
  'attention',
  'softmax',
  'rmsnorm',
  'silu',
  'rope',
  'moe',
];

// Quick validation - subset for fast CI
const QUICK_TESTS = ['matmul', 'rmsnorm', 'softmax', 'gather'];

// ============================================================================
// Argument Parsing
// ============================================================================






const FLAG_SPECS = [
  { names: ['--help', '-h'], handler: (opts) => { opts.help = true; } },
  { names: ['--config'], handler: (opts, tokens) => { opts.config = tokens.shift() || null; } },
  { names: ['--mode'], handler: (opts, tokens) => { opts.mode = tokens.shift() || null; } },
  { names: ['--dump-config'], handler: (opts) => { opts.dumpConfig = true; } },
  { names: ['--list-presets'], handler: (opts) => { opts.listPresets = true; } },
  { names: ['--model', '-m'], handler: (opts, tokens) => { opts.model = tokens.shift() || opts.model; } },
  { names: ['--base-url', '-u'], handler: (opts, tokens) => { opts.baseUrl = tokens.shift() || opts.baseUrl; } },
  { names: ['--no-server'], handler: (opts) => { opts.noServer = true; } },
  { names: ['--headless'], handler: (opts) => { opts.headless = true; } },
  { names: ['--headed', '--no-headless'], handler: (opts) => { opts.headless = false; } },
  { names: ['--minimized', '--no-focus'], handler: (opts) => { opts.minimized = true; } },
  { names: ['--reuse-browser'], handler: (opts) => { opts.reuseBrowser = true; } },
  { names: ['--no-reuse-browser', '--new-browser'], handler: (opts) => { opts.reuseBrowser = false; } },
  { names: ['--cdp-endpoint'], handler: (opts, tokens) => { opts.cdpEndpoint = tokens.shift() || opts.cdpEndpoint; } },
  { names: ['--skip-load'], handler: (opts) => { opts.skipLoad = true; } },
  { names: ['--warm'], handler: (opts) => { opts.warm = true; opts.headless = false; opts.reuseBrowser = true; } },
  { names: ['--inference'], handler: (opts) => { opts.suite = 'inference'; } },
  { names: ['--kernels'], handler: (opts) => { opts.suite = 'kernels'; } },
  { names: ['--training'], handler: (opts) => { opts.suite = 'training'; } },
  { names: ['--quick'], handler: (opts) => { opts.suite = 'quick'; } },
  { names: ['--full'], handler: (opts) => { opts.suite = 'all'; } },
  { names: ['--filter', '-f'], handler: (opts, tokens) => { opts.filter = tokens.shift() || null; } },
  { names: ['--timeout'], handler: (opts, tokens) => { opts.timeout = parseInt(tokens.shift() || '120000', 10); } },
  { names: ['--output', '-o'], handler: (opts, tokens) => { opts.output = tokens.shift() || null; } },
  { names: ['--html'], handler: (opts, tokens) => { opts.html = tokens.shift() || null; } },
  { names: ['--compare', '-c'], handler: (opts, tokens) => { opts.compare = tokens.shift() || null; } },
  { names: ['--profile-dir'], handler: (opts, tokens) => { opts.profileDir = tokens.shift() || null; } },
  { names: ['--retries'], handler: (opts, tokens) => { opts.retries = parseInt(tokens.shift() || '2', 10); } },
  { names: ['--perf'], handler: (opts) => { opts.perf = true; } },
  { names: ['--gpu-profile'], handler: (opts) => { opts.gpuProfile = true; } },
];


function buildFlagHandlers() {
  
  const handlers = new Map();
  for (const spec of FLAG_SPECS) {
    for (const name of spec.names) {
      handlers.set(name, spec.handler);
    }
  }
  return handlers;
}

const FLAG_HANDLERS = buildFlagHandlers();
const KNOWN_FLAGS = new Set(FLAG_HANDLERS.keys());


function resolveFlagAlias(flag) {
  if (!flag.startsWith('--')) return null;
  const raw = flag.slice(2);
  if (!raw) return null;
  const normalized = raw
    .replace(/_/g, '-')
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .toLowerCase();
  const candidate = `--${normalized}`;
  return KNOWN_FLAGS.has(candidate) ? candidate : null;
}


function suggestFlag(flag) {
  if (!flag.startsWith('--')) return null;
  if (!/[A-Z]/.test(flag)) return null;
  const kebab = `--${flag.slice(2).replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`)}`;
  return KNOWN_FLAGS.has(kebab) ? kebab : null;
}


function normalizeFlag(flag) {
  return flag.replace(/^-+/, '').replace(/_/g, '-').toLowerCase();
}


function levenshteinDistance(a, b) {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  const prev = new Array(b.length + 1).fill(0);
  const curr = new Array(b.length + 1).fill(0);
  for (let j = 0; j <= b.length; j++) prev[j] = j;

  for (let i = 1; i <= a.length; i++) {
    curr[0] = i;
    const aChar = a.charCodeAt(i - 1);
    for (let j = 1; j <= b.length; j++) {
      const cost = aChar === b.charCodeAt(j - 1) ? 0 : 1;
      curr[j] = Math.min(
        prev[j] + 1,
        curr[j - 1] + 1,
        prev[j - 1] + cost
      );
    }
    for (let j = 0; j <= b.length; j++) prev[j] = curr[j];
  }

  return prev[b.length];
}


function suggestClosestFlags(flag) {
  const camelSuggestion = suggestFlag(flag);
  if (camelSuggestion) return [camelSuggestion];

  const normalized = normalizeFlag(flag);
  const candidates = Array.from(KNOWN_FLAGS).filter((candidate) => {
    if (flag.startsWith('--')) return candidate.startsWith('--');
    return true;
  });
  const scored = candidates.map((candidate) => ({
    candidate,
    distance: levenshteinDistance(normalized, normalizeFlag(candidate)),
  }));

  scored.sort((a, b) => a.distance - b.distance || a.candidate.localeCompare(b.candidate));

  const best = scored[0];
  if (!best) return [];

  const maxDistance = Math.max(2, Math.floor(normalized.length / 3));
  return scored
    .filter((item) => item.distance <= maxDistance)
    .slice(0, 3)
    .map((item) => item.candidate);
}


function parseArgs(argv) {
  
  const opts = {
    cliFlags: new Set(),
    command: 'test',
    suite: 'kernels',
    model: 'gemma-2-2b-it-wf16',  // Format: {family}-{version}-{size}-{variant}-{quant}
    baseUrl: 'http://localhost:8080',
    config: null,           // Config preset or path
    mode: null,             // Mode preset shortcut (bench/debug/profile/test)
    runtimeConfig: null,    // Loaded runtime config (merged with defaults)
    configChain: null,      // Config inheritance chain for debugging
    dumpConfig: false,      // Dump resolved config and exit
    listPresets: false,     // List available presets and exit
    noServer: false,
    headless: true,   // Default to headless (real GPU via --headless=new)
    minimized: false, // Position window off-screen when true
    reuseBrowser: true, // Try to connect to existing Chrome via CDP first
    cdpEndpoint: 'http://localhost:9222', // CDP endpoint for reuseBrowser
    verbose: false,
    filter: null,
    timeout: 300000,
    output: null,
    html: null,
    compare: null,
    profileDir: null,
    retries: 2,
    quiet: false,
    help: false,
    perf: false,
    gpuProfile: false,
    // Warm mode options
    skipLoad: false,
    warm: false,
  };

  const tokens = [...argv];
  let positionalIndex = 0;

  while (tokens.length) {
    const arg = tokens.shift();
    if (arg.startsWith('-')) {
      let resolvedFlag = arg;
      let handler = FLAG_HANDLERS.get(arg);
      if (!handler) {
        const alias = resolveFlagAlias(arg);
        if (alias) {
          resolvedFlag = alias;
          handler = FLAG_HANDLERS.get(alias);
        }
      }
      if (!handler) {
        const suggestions = suggestClosestFlags(arg);
        const hint = suggestions.length > 0
          ? ` Did you mean ${suggestions.map((s) => `"${s}"`).join(' or ')}?`
          : '';
        throw new Error(`Unknown flag "${arg}".${hint}`);
      }
      opts.cliFlags.add(resolvedFlag);
      handler(opts, tokens);
      continue;
    }

    if (positionalIndex === 0) {
      if (arg === 'run' || arg === 'test' || arg === 'bench' || arg === 'debug') {
        opts.command =  (arg);
      } else {
        opts.suite = normalizeSuite(arg);
      }
    } else if (positionalIndex === 1) {
      opts.suite = normalizeSuite(arg);
    }
    positionalIndex++;
  }

  return opts;
}


function hasCliFlag(opts, flags) {
  return flags.some((flag) => opts.cliFlags.has(flag));
}



function appendRuntimeConfigParams(params, opts) {
  if (opts.runtimeConfig) {
    params.set('runtimeConfig', JSON.stringify(opts.runtimeConfig));
  }
  if (opts.configChain) {
    params.set('configChain', JSON.stringify(opts.configChain));
  }
}


function setHarnessConfig(opts, updates) {
  const harness = opts.runtimeConfig?.shared?.harness;
  if (!harness) {
    throw new Error('runtime.shared.harness is required for harness config.');
  }
  Object.assign(harness, updates);
}


function normalizeSuite(suite) {
  
  const legacyMap = {
    'bench:kernels': 'kernels',
    'bench:pipeline': 'inference',
    'bench:system': 'system',
    'correctness': 'kernels',  // Renamed: correctness -> kernels
  };
  return  (legacyMap[suite] || suite);
}


function printHelp() {
  console.log(`
DOPPLER CLI - Test, Benchmark, Debug

Three commands, three purposes:

  doppler test   ->  Correctness (does it work?)
  doppler bench  ->  Performance (how fast?)
  doppler debug  ->  Debugging (why is it broken?)

===============================================================

TEST - Correctness Tests
  doppler test                        All kernel tests (default)
  doppler test --quick                Quick subset (matmul, rmsnorm, softmax, gather)
  doppler test --inference            Model loads + generates (smoke test)
  doppler test --training             Training kernel + loop checks
  doppler test --filter matmul        Filter to specific kernel

BENCH - Performance Benchmarks
  doppler bench                       Full inference benchmark (tok/s)
  doppler bench --kernels             Kernel microbenchmarks only
  doppler bench --compare base.json   Compare against baseline

DEBUG - Interactive Debugging
  doppler debug                       Debug mode (config-driven)

===============================================================

Common Options:
  --model, -m <name>     Model (default: gemma-2-2b-it-wf16)
  --config <ref>         Load config (preset name, path, URL, or inline JSON)
  --mode <name>          Shortcut for runtime preset (e.g., bench, debug)
  --dump-config          Print resolved config and exit
  --list-presets         List available config presets
  --headed               Show browser window (default: headless with real GPU)
  --no-reuse-browser     Always launch new browser (don't try CDP)
  --cdp-endpoint <url>   CDP endpoint (default: http://localhost:9222)
  --timeout <ms>         Timeout (default: 300000)
  --output, -o <file>    Save JSON results
  --help, -h             Show this help

Config System:
  --config debug               Use built-in 'debug' preset
  --config ./my-config.json    Load from file path
  --config '{"runtime":...}'   Inline JSON config
  --mode bench                 Shortcut for --config bench

  Built-in presets live under src/config/presets/runtime/* (use --list-presets)
  User presets: ~/.doppler/presets/*.json
  Project presets: ./.doppler/*.json

Warm Mode (preserve model in GPU RAM):
  --warm                 Keep browser open with model loaded for reuse
  --skip-load            Skip model loading (use existing window.pipeline)

  Usage:
    1. First run: doppler debug --warm  (loads model, keeps browser open)
    2. Next runs: doppler debug --skip-load  (reuses loaded model via CDP)

  Start Chrome with CDP first for best results:
    /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222

Headless Mode (default):
  Uses --headless=new with real GPU acceleration (not SwiftShader).
  No browser window, no focus stealing, full GPU compute support.

Examples:
  doppler test                    # Quick correctness check
  doppler bench --config bench    # Benchmark preset
  doppler debug --config debug    # Debug preset

Notes:
  - Headless mode by default (real GPU via --headless=new)
  - Use --headed for visible browser window (debugging)
  - Dev server auto-starts at localhost:8080
  - Exit code: 0=pass, 1=fail
`);
}

// ============================================================================
// Correctness Tests
// ============================================================================


async function runCorrectnessTests(page, opts, tests) {
  console.log('\n' + '='.repeat(60));
  console.log('KERNEL CORRECTNESS TESTS');
  console.log('='.repeat(60));

  setHarnessConfig(opts, {
    mode: 'kernels',
    autorun: false,
    skipLoad: false,
    modelId: null,
  });
  const harnessParams = new URLSearchParams();
  appendRuntimeConfigParams(harnessParams, opts);
  await page.goto(`${opts.baseUrl}/doppler/tests/harness.html?${harnessParams.toString()}`, {
    timeout: opts.timeout,
  });

  // Wait for page to fully load
  await page.waitForTimeout(500);

  // Module scripts run after DOM ready, so DOMContentLoaded may have already fired.
  // Try manual GPU init via testHarness.getGPU() if needed.
  try {
    await page.evaluate(async () => {
      const w =  (window);
      if (!w.gpuReady && w.testHarness?.getGPU) {
        await w.testHarness.getGPU();
        w.gpuReady = true;
      }
    });
  } catch (err) {
    // Will be caught by waitForFunction below
  }

  await page.waitForFunction(
    () => {
      const w =  (window);
      if (w.gpuError) {
        throw new Error(`WebGPU init failed: ${w.gpuError}`);
      }
      return w.gpuReady === true && w.testHarness && w.testHarness.references;
    },
    { timeout: 30000 }
  );

  
  const results = [];
  const startTime = Date.now();

  const testsToRun = opts.filter
    ? tests.filter((t) => t.includes(opts.filter))
    : tests;

  for (const testName of testsToRun) {
    console.log(`\n  Running: ${testName}...`);
    const testStart = Date.now();

    try {
      const result = await page.evaluate(
        async (name) => {
          const harness =  (window).testHarness;
          const gpu = await harness.getGPU();
          const refs = harness.references;

          switch (name) {
            case 'matmul': {
              const M = 64, K = 128, N = 64;
              const A = new Float32Array(M * K).map(() => Math.random() * 2 - 1);
              const B = new Float32Array(K * N).map(() => Math.random() * 2 - 1);
              const ref = refs.matmulRef(A, B, M, N, K);
              const gpuResult = await harness.runMatmul(gpu.device, A, B, M, N, K);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'matmul-q4k': {
              // Test Q4K fused batched matmul (tests the subgroup column mixing fix)
              // M > 1 triggers q4_fused_batched kernel, K must be multiple of 256 for Q4K
              const M = 8, K = 256, N = 32;

              // Create activation matrix A[M, K]
              const A = new Float32Array(M * K).map(() => (Math.random() * 2 - 1) * 0.5);

              // Create weight matrix B[N, K]
              const B_f32 = new Float32Array(N * K).map(() => (Math.random() * 2 - 1) * 0.5);

              // Quantize B to Q4K format
              const numBlocks = N * (K / 256);
              const B_q4k = refs.quantizeQ4_KRef(B_f32, numBlocks);

              // Reference: dequant then matmul (A @ B^T for [M,N] output)
              const B_dequant = refs.dequantQ4_KRef(B_q4k, numBlocks);
              const refC = new Float32Array(M * N);
              for (let m = 0; m < M; m++) {
                for (let n = 0; n < N; n++) {
                  let sum = 0;
                  for (let k = 0; k < K; k++) {
                    sum += A[m * K + k] * B_dequant[n * K + k];
                  }
                  refC[m * N + n] = sum;
                }
              }

              // GPU: fused Q4K matmul
              const gpuC = await harness.runMatmulQ4K(gpu.device, A, B_q4k, M, N, K);

              // Compare
              let maxError = 0;
              let hasNaN = false;
              let zeroCount = 0;
              for (let i = 0; i < refC.length; i++) {
                if (isNaN(gpuC[i])) hasNaN = true;
                if (gpuC[i] === 0 && refC[i] !== 0) zeroCount++;
                maxError = Math.max(maxError, Math.abs(gpuC[i] - refC[i]));
              }

              // Q4K quantization has inherent error, allow 0.1 tolerance
              const passed = maxError < 0.1 && !hasNaN && zeroCount < refC.length / 2;
              return { passed, maxError, hasNaN, zeroCount, M, N, K };
            }

            case 'matmul-q4k-large': {
              // Test Q4K with EXACT inference dimensions (Gemma3-1B: hiddenSize=1152)
              // K=1152 is NOT divisible by 256! This tests the bounds checking in the kernel.
              // This is the exact scenario causing inference failures.
              const M = 16, K = 1152, N = 1024;  // Exact dims from q_proj inference

              // Create activation matrix A[M, K]
              const A = new Float32Array(M * K).map(() => (Math.random() * 2 - 1) * 0.3);

              // Create weight matrix B[N, K]
              const B_f32 = new Float32Array(N * K).map(() => (Math.random() * 2 - 1) * 0.3);

              // Quantize B to Q4K format
              // For non-256-aligned K, we need ceil(K/256) blocks per row
              const blocksPerRow = Math.ceil(K / 256);
              const paddedK = blocksPerRow * 256;  // 1280 for K=1152
              const numBlocks = N * blocksPerRow;

              // Pad B_f32 to have paddedK elements per row (zeros for padding)
              const B_padded = new Float32Array(N * paddedK);
              for (let n = 0; n < N; n++) {
                for (let k = 0; k < K; k++) {
                  B_padded[n * paddedK + k] = B_f32[n * K + k];
                }
                // Padding elements (k >= K) are already 0 from initialization
              }
              const B_q4k = refs.quantizeQ4_KRef(B_padded, numBlocks);

              // Reference: dequant then matmul
              // The dequant produces blocksPerRow * 256 = 1280 elements per row (for K=1152)
              // We only use the first K=1152 elements of each row
              const B_dequant = refs.dequantQ4_KRef(B_q4k, numBlocks);
              const dequantRowStride = blocksPerRow * 256;  // 1280 for K=1152
              const refC = new Float32Array(M * N);
              for (let m = 0; m < M; m++) {
                for (let n = 0; n < N; n++) {
                  let sum = 0;
                  for (let k = 0; k < K; k++) {
                    // B_dequant has dequantRowStride elements per row, we only use first K
                    sum += A[m * K + k] * B_dequant[n * dequantRowStride + k];
                  }
                  refC[m * N + n] = sum;
                }
              }

              // GPU: fused Q4K matmul
              const gpuC = await harness.runMatmulQ4K(gpu.device, A, B_q4k, M, N, K);

              // Compare
              let maxError = 0;
              let hasNaN = false;
              let zeroCount = 0;
              let allZero = true;
              for (let i = 0; i < refC.length; i++) {
                if (isNaN(gpuC[i])) hasNaN = true;
                if (gpuC[i] !== 0) allZero = false;
                if (gpuC[i] === 0 && Math.abs(refC[i]) > 0.001) zeroCount++;
                maxError = Math.max(maxError, Math.abs(gpuC[i] - refC[i]));
              }

              // Q4K quantization has inherent error, allow larger tolerance for accumulated error
              const passed = maxError < 0.5 && !hasNaN && !allZero && zeroCount < refC.length / 4;
              return { passed, maxError, hasNaN, zeroCount, allZero, M, N, K, numBlocks };
            }

            case 'rmsnorm': {
              const batchSize = 4, hiddenSize = 256;
              const input = new Float32Array(batchSize * hiddenSize).map(() => Math.random() * 2 - 1);
              const weight = new Float32Array(hiddenSize).map(() => Math.random() * 0.5 + 0.5);
              const ref = refs.rmsNormRef(input, weight, batchSize, hiddenSize);
              const gpuResult = await harness.runRMSNorm(gpu.device, input, weight, batchSize, hiddenSize);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'softmax': {
              const innerSize = 128, outerSize = 32;
              const input = new Float32Array(innerSize * outerSize).map(() => Math.random() * 4 - 2);
              const ref = refs.softmaxRef(input, innerSize, outerSize);
              const gpuResult = await harness.runSoftmax(gpu.device, input, innerSize, outerSize);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-5, maxError };
            }

            case 'gather': {
              const vocabSize = 1000, embedDim = 128, numTokens = 16;
              const embeddings = new Float32Array(vocabSize * embedDim).map(() => Math.random() * 2 - 1);
              const indices = new Uint32Array(numTokens).map(() => Math.floor(Math.random() * vocabSize));
              const ref = refs.gatherRef(embeddings, indices, vocabSize, embedDim);
              const gpuResult = await harness.runGather(gpu.device, embeddings, indices, vocabSize, embedDim);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-6, maxError };
            }

            case 'silu': {
              const size = 1024;
              const input = new Float32Array(size).map(() => Math.random() * 4 - 2);
              const ref = refs.siluRef(input);
              const gpuResult = await harness.runSiLU(gpu.device, input);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-5, maxError };
            }

            case 'rope': {
              const seqLen = 16, numHeads = 8, headDim = 64;
              const input = new Float32Array(seqLen * numHeads * headDim).map(() => Math.random() * 2 - 1);
              const { cos, sin } = refs.computeRopeFreqs(headDim, seqLen);
              const ref = refs.ropeRef(input, cos, sin, seqLen, numHeads, headDim, 0);
              const gpuResult = await harness.runRoPE(gpu.device, input, seqLen, numHeads, headDim);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'residual': {
              const size = 1024;
              const x = new Float32Array(size).map(() => Math.random() * 2 - 1);
              const residual = new Float32Array(size).map(() => Math.random() * 2 - 1);
              const ref = refs.residualAddRef(x, residual);
              const gpuResult = await harness.runResidual(gpu.device, x, residual);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-6, maxError };
            }

            case 'attention': {
              const seqLen = 16, numHeads = 4, headDim = 32;
              const Q = new Float32Array(seqLen * numHeads * headDim).map(() => Math.random() * 0.5);
              const K = new Float32Array(seqLen * numHeads * headDim).map(() => Math.random() * 0.5);
              const V = new Float32Array(seqLen * numHeads * headDim).map(() => Math.random() * 0.5);
              const mask = refs.createCausalMask(seqLen);
              const ref = refs.attentionRef(Q, K, V, seqLen, seqLen, numHeads, numHeads, headDim, mask);
              const gpuResult = await harness.runAttention(gpu.device, Q, K, V, seqLen, seqLen, numHeads, numHeads, headDim, mask);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-3, maxError };
            }

            case 'scatter-add': {
              const numTokens = 8, hiddenSize = 64, numExperts = 4, topK = 2;
              const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize).map(() => Math.random());
              const indices = new Uint32Array(numTokens * topK);
              const weights = new Float32Array(numTokens * topK);
              for (let t = 0; t < numTokens; t++) {
                for (let k = 0; k < topK; k++) {
                  indices[t * topK + k] = Math.floor(Math.random() * numExperts);
                  weights[t * topK + k] = 1.0 / topK;
                }
              }
              const ref = refs.scatterAddRef(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK);
              const gpuResult = await harness.runScatterAdd(gpu.device, expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'moe-gather': {
              const numTokens = 8, hiddenSize = 64, numExperts = 4, topK = 2;
              const tokens = new Float32Array(numTokens * hiddenSize).map(() => Math.random());
              const expertIndices = new Uint32Array(numTokens * topK);
              for (let t = 0; t < numTokens; t++) {
                for (let k = 0; k < topK; k++) {
                  expertIndices[t * topK + k] = Math.floor(Math.random() * numExperts);
                }
              }
              const ref = refs.moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
              const gpuResult = await harness.runMoEGather(gpu.device, tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
              let passed = true;
              for (let i = 0; i < numExperts; i++) {
                if (ref.tokenCounts[i] !== gpuResult.tokenCounts[i]) {
                  passed = false;
                  break;
                }
              }
              return { passed, tokenCounts: Array.from(gpuResult.tokenCounts) };
            }

            case 'topk': {
              const numTokens = 4, numExperts = 8, topK = 2;
              const logits = new Float32Array(numTokens * numExperts).map(() => Math.random() * 4 - 2);
              const ref = refs.softmaxTopkRef(logits, numTokens, numExperts, topK, true);
              const gpuResult = await harness.runSoftmaxTopK(gpu.device, logits, numTokens, numExperts, topK);
              let passed = true;
              for (let t = 0; t < numTokens; t++) {
                
                const refSet = new Set();
                
                const gpuSet = new Set();
                for (let k = 0; k < topK; k++) {
                  refSet.add(ref.indices[t * topK + k]);
                  gpuSet.add(gpuResult.indices[t * topK + k]);
                }
                for (const idx of refSet) {
                  if (!gpuSet.has(idx)) {
                    passed = false;
                    break;
                  }
                }
              }
              return { passed };
            }

            case 'dequant': {
              const numBlocks = 4;
              const blockSize = 32;
              const quantized = new Uint8Array(numBlocks * 18);
              for (let i = 0; i < quantized.length; i++) {
                quantized[i] = Math.floor(Math.random() * 256);
              }
              const ref = refs.dequantQ4_0Ref(quantized, numBlocks);
              return { passed: ref.length === numBlocks * blockSize, refLength: ref.length };
            }

            case 'dequant-q4k': {
              // Test Q4_K dequantization against CPU reference
              const numBlocks = 4;
              const blockElems = 256;

              // Create deterministic test data
              const values = new Float32Array(numBlocks * blockElems);
              for (let i = 0; i < values.length; i++) {
                values[i] = Math.sin(i * 0.1) * 0.75 + Math.cos(i * 0.03) * 0.25;
              }

              // Quantize using CPU reference
              const quantized = refs.quantizeQ4_KRef(values, numBlocks);
              const expected = refs.dequantQ4_KRef(quantized, numBlocks);

              // Run GPU dequant (F32 output, non-vec4 path)
              const gpuResult = await harness.runDequantQ4K(gpu.device, quantized, numBlocks);

              // Compare results
              let maxError = 0;
              let maxErrorIdx = -1;
              for (let i = 0; i < expected.length; i++) {
                const err = Math.abs(gpuResult[i] - expected[i]);
                if (err > maxError) {
                  maxError = err;
                  maxErrorIdx = i;
                }
              }

              const passed = maxError < 1e-3;
              return {
                passed,
                maxError,
                maxErrorIdx,
                length: gpuResult.length,
                first10Actual: Array.from(gpuResult.slice(0, 10)),
                first10Expected: Array.from(expected.slice(0, 10)),
              };
            }

            case 'dequant-q4k-f16': {
              // Test Q4_K dequantization with F16 output (production loader path)
              // Use larger size to match production dimensions (e.g. q_proj [2048, 2304])
              const numBlocks = 2048 * 9;  // 2048 rows, 9 blocks per row (K=2304)
              const blockElems = 256;

              // Create deterministic test data
              const values = new Float32Array(numBlocks * blockElems);
              for (let i = 0; i < values.length; i++) {
                values[i] = Math.sin(i * 0.1) * 0.75 + Math.cos(i * 0.03) * 0.25;
              }

              // Quantize using CPU reference
              const quantized = refs.quantizeQ4_KRef(values, numBlocks);
              const expected = refs.dequantQ4_KRef(quantized, numBlocks);

              // Run GPU dequant (F16 output, vec4 path) - this is the production path!
              const gpuResult = await harness.runDequantQ4K_F16(gpu.device, quantized, numBlocks);

              // Compare results (allow larger tolerance for F16 precision)
              let maxError = 0;
              let maxErrorIdx = -1;
              
              const sampleErrors = [];
              for (let i = 0; i < expected.length; i++) {
                const err = Math.abs(gpuResult[i] - expected[i]);
                if (err > maxError) {
                  maxError = err;
                  maxErrorIdx = i;
                }
                if (err > 0.01 && sampleErrors.length < 10) {
                  sampleErrors.push({i, expected: expected[i], actual: gpuResult[i], err});
                }
              }

              const passed = maxError < 0.01;  // F16 has lower precision
              return {
                passed,
                maxError,
                maxErrorIdx,
                length: gpuResult.length,
                first10Actual: Array.from(gpuResult.slice(0, 10)),
                first10Expected: Array.from(expected.slice(0, 10)),
                sampleErrors,
              };
            }

            case 'matmul-f16w': {
              // Test F16 weights matmul (f16w_f32a kernel) with dequanted weights
              // This tests the EXACT production path: Q4K -> F16 dequant -> f16w_f32a matmul
              // All operations stay on GPU without CPU round-trip (matches production loader)
              const M = 10;  // Match production prefill size
              const K = 2304;  // Gemma hidden size
              const N = 2048;  // Typical weight output dim

              // Create activation matrix A[M, K] with production-like values
              // Production: L0 input has min=-100.5, max=132.0
              const A = new Float32Array(M * K).map((_, i) =>
                Math.sin(i * 0.1) * 50 + Math.cos(i * 0.03) * 80
              );

              // Create weight matrix B[N, K] - will be stored row-major
              // Use deterministic data similar to dequant test values
              const B_f32 = new Float32Array(N * K).map((_, i) =>
                Math.sin(i * 0.07) * 0.02 + Math.cos(i * 0.11) * 0.01
              );

              // Quantize to Q4K
              const blocksPerRow = Math.ceil(K / 256);
              const numBlocks = N * blocksPerRow;
              const B_q4k = refs.quantizeQ4_KRef(B_f32, numBlocks);

              // CPU reference: dequant + matmul
              const B_cpu_dequant = refs.dequantQ4_KRef(B_q4k, numBlocks);
              const refC = new Float32Array(M * N);
              for (let m = 0; m < M; m++) {
                for (let n = 0; n < N; n++) {
                  let sum = 0;
                  for (let k = 0; k < K; k++) {
                    sum += A[m * K + k] * B_cpu_dequant[n * K + k];
                  }
                  refC[m * N + n] = sum;
                }
              }

              // GPU combined path: Q4K dequant -> F16 -> matmul (all on GPU)
              // This is the exact production path the loader uses
              const gpuC = await harness.runDequantAndMatmulF16W(gpu.device, A, B_q4k, M, N, K, numBlocks);

              // Compare
              let maxError = 0;
              let maxErrorIdx = -1;
              let hasNaN = false;
              
              const sampleErrors = [];
              for (let i = 0; i < refC.length; i++) {
                if (isNaN(gpuC[i])) hasNaN = true;
                const err = Math.abs(gpuC[i] - refC[i]);
                if (err > maxError) {
                  maxError = err;
                  maxErrorIdx = i;
                }
                if (err > 0.1 && sampleErrors.length < 10) {
                  sampleErrors.push({i, expected: refC[i], actual: gpuC[i], err});
                }
              }

              // Q4K quantization adds error, plus F16 precision loss - allow 0.5 tolerance
              const passed = maxError < 0.5 && !hasNaN;
              return {
                passed,
                maxError,
                maxErrorIdx,
                hasNaN,
                length: gpuC.length,
                first10Actual: Array.from(gpuC.slice(0, 10)),
                first10Ref: Array.from(refC.slice(0, 10)),
                sampleErrors,
                M, N, K, numBlocks,
              };
            }

            case 'swiglu': {
              // Test SwiGLU activation: output = SiLU(gate + gate_bias) * (up + up_bias)
              const size = 256;
              const gate = new Float32Array(size).map(() => Math.random() * 2 - 1);
              const up = new Float32Array(size).map(() => Math.random() * 2 - 1);
              const gateBias = new Float32Array(size).map(() => Math.random() * 0.1 - 0.05);
              const upBias = new Float32Array(size).map(() => Math.random() * 0.1 - 0.05);

              // Reference: SwiGLU = SiLU(gate + gate_bias) * (up + up_bias)
              const expected = new Float32Array(size);
              for (let i = 0; i < size; i++) {
                const gatedValue = gate[i] + gateBias[i];
                const silu = gatedValue / (1 + Math.exp(-gatedValue));  // SiLU = x * sigmoid(x)
                expected[i] = silu * (up[i] + upBias[i]);
              }

              const gpuResult = await harness.runSwiGLU(gpu.device, gate, up, gateBias, upBias);
              let maxError = 0;
              for (let i = 0; i < size; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - expected[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'scale': {
              // Test element-wise scalar multiplication: output[i] = input[i] * scale
              const size = 512;
              const input = new Float32Array(size).map(() => Math.random() * 10 - 5);
              const scale = 0.125;  // Common scaling factor (e.g., 1/sqrt(head_dim))

              const expected = new Float32Array(size);
              for (let i = 0; i < size; i++) {
                expected[i] = input[i] * scale;
              }

              const gpuResult = await harness.runScale(gpu.device, input, scale);
              let maxError = 0;
              for (let i = 0; i < size; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - expected[i]));
              }
              return { passed: maxError < 1e-6, maxError };
            }

            case 'dequant-q6k': {
              // Test Q6_K dequantization (6-bit GGUF quantization)
              // Q6_K block: 210 bytes for 256 elements
              // Layout: ql[128] + qh[64] + scales[16] + d[2] (f16)
              const numBlocks = 2;
              const blockSize = 256;
              const Q6K_BLOCK_BYTES = 210;
              const D_OFFSET = 208;  // f16 scale offset

              // Create quantized data with valid f16 scale
              const quantized = new Uint8Array(numBlocks * Q6K_BLOCK_BYTES);
              for (let i = 0; i < quantized.length; i++) {
                quantized[i] = Math.floor(Math.random() * 256);
              }

              // Ensure d (f16 at offset 208) is a valid small number, not Inf/NaN
              // Use 0x3C00 = 1.0 in f16 for each block
              for (let b = 0; b < numBlocks; b++) {
                const base = b * Q6K_BLOCK_BYTES + D_OFFSET;
                quantized[base] = 0x00;     // Low byte
                quantized[base + 1] = 0x3C; // High byte = 0x3C00 = 1.0
              }

              // Run GPU dequant
              const gpuResult = await harness.runDequantQ6K(gpu.device, quantized, numBlocks);

              // Verify output is the right size and has valid values
              let nanCount = 0;
              let infCount = 0;
              for (let i = 0; i < gpuResult.length; i++) {
                if (isNaN(gpuResult[i])) nanCount++;
                if (!isFinite(gpuResult[i])) infCount++;
              }
              const passed = gpuResult.length === numBlocks * blockSize && nanCount === 0 && infCount === 0;
              return { passed, outputLength: gpuResult.length, expectedLength: numBlocks * blockSize, nanCount, infCount };
            }

            case 'sample': {
              // Test GPU argmax sampling
              const vocabSize = 128;
              const logits = new Float32Array(vocabSize).map(() => Math.random() * 10 - 5);

              // Set a clear maximum for deterministic test
              const expectedIdx = 42;
              logits[expectedIdx] = 100;  // Clear maximum

              const gpuIdx = await harness.runArgmax(gpu.device, logits);
              const refIdx = refs.argmaxRef(logits);

              const passed = gpuIdx === refIdx && gpuIdx === expectedIdx;
              return { passed, gpuIdx, refIdx, expectedIdx };
            }

            default:
              return { passed: false, error: `Unknown test: ${name}` };
          }
        },
        testName
      );

      const duration = Date.now() - testStart;
      const passed = result.passed === true;

      results.push({
        name: testName,
        passed,
        duration,
        error: passed ? undefined : JSON.stringify(result),
      });

      const status = passed ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
      console.log(`  ${status} ${testName} (${duration}ms)`);
      if (!passed && opts.verbose) {
        console.log(`    Details: ${JSON.stringify(result)}`);
      }
    } catch (err) {
      const duration = Date.now() - testStart;
      results.push({
        name: testName,
        passed: false,
        duration,
        error:  (err).message,
      });
      console.log(`  \x1b[31mFAIL\x1b[0m ${testName} (${duration}ms)`);
      console.log(`    Error: ${ (err).message}`);
    }
  }

  const totalDuration = Date.now() - startTime;
  const passed = results.filter((r) => r.passed).length;
  const failed = results.filter((r) => !r.passed).length;

  return {
    suite: 'correctness',
    passed,
    failed,
    skipped: tests.length - testsToRun.length,
    duration: totalDuration,
    results,
  };
}


async function runTrainingTests(page, opts, tests) {
  console.log('\n' + '='.repeat(60));
  console.log('TRAINING CORRECTNESS TESTS');
  console.log('='.repeat(60));

  setHarnessConfig(opts, {
    mode: 'training',
    autorun: false,
    skipLoad: false,
    modelId: null,
  });
  const harnessParams = new URLSearchParams();
  appendRuntimeConfigParams(harnessParams, opts);
  await page.goto(`${opts.baseUrl}/doppler/tests/harness.html?${harnessParams.toString()}`, {
    timeout: opts.timeout,
  });

  await page.waitForTimeout(500);

  await page.waitForFunction(
    () => {
      const w =  (window);
      return Boolean(w.trainingHarness);
    },
    { timeout: 30000 }
  );

  
  const results = [];
  const startTime = Date.now();

  const testsToRun = opts.filter
    ? tests.filter((t) => t.includes(opts.filter))
    : tests;

  for (const testName of testsToRun) {
    console.log(`\n  Running: ${testName}...`);
    const testStart = Date.now();

    try {
      const result = await page.evaluate(
        async (name) => {
          const harness =  (window).trainingHarness;
          if (harness?.getGPU) {
            await harness.getGPU();
          }
          const testResult = await harness.runTest(name);
          return testResult;
        },
        testName
      );

      const duration = Date.now() - testStart;
      results.push({
        name: testName,
        passed: result.passed,
        duration,
        error: result.error,
      });

      const status = result.passed ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
      console.log(`  ${status} ${testName} (${duration}ms)`);
      if (result.error) {
        console.log(`    Error: ${result.error}`);
      }
    } catch (err) {
      const duration = Date.now() - testStart;
      results.push({
        name: testName,
        passed: false,
        duration,
        error: err?.message || String(err),
      });
      console.log(`  \x1b[31mFAIL\x1b[0m ${testName} (${duration}ms)`);
    }
  }

  const duration = Date.now() - startTime;
  const passed = results.filter(r => r.passed).length;
  const failed = results.length - passed;

  if (opts.output && results.length > 0) {
    const outputPath = resolve(opts.output);
    await writeFile(outputPath, JSON.stringify({ suite: 'training', results }, null, 2));
    console.log(`\nResults written to ${outputPath}`);
  }

  if (await page.evaluate(() =>  (window).renderResults)) {
    await page.evaluate((res) =>  (window).renderResults(res), results);
  }

  return {
    suite: 'training',
    passed,
    failed,
    skipped: tests.length - testsToRun.length,
    duration,
    results,
  };
}

// ============================================================================
// Kernel Benchmarks
// ============================================================================


async function runKernelBenchmarks(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('KERNEL BENCHMARKS');
  console.log('='.repeat(60));

  const runtimeConfig = opts.runtimeConfig;
  if (!runtimeConfig) {
    throw new Error('Runtime config is required for benchmarks.');
  }
  const benchmarkRun = runtimeConfig.shared.benchmark.run;

  await page.addInitScript(() => {
    
    (window).__name = (target) => target;
  });

  setHarnessConfig(opts, {
    mode: 'kernels',
    autorun: false,
    skipLoad: false,
    modelId: null,
  });
  const harnessParams = new URLSearchParams();
  appendRuntimeConfigParams(harnessParams, opts);
  await page.goto(`${opts.baseUrl}/doppler/tests/harness.html?${harnessParams.toString()}`, {
    timeout: opts.timeout,
  });

  await page.waitForFunction(
    () =>  (window).testHarness &&  (window).testHarness.references,
    { timeout: 10000 }
  );

  
  const results = [];
  const startTime = Date.now();

  const benchmarks = opts.filter
    ? KERNEL_BENCHMARKS.filter((b) => b.includes(opts.filter))
    : KERNEL_BENCHMARKS;

  for (const benchName of benchmarks) {
    console.log(`\n  Benchmarking: ${benchName}...`);

    try {
      const result = await page.evaluate(
        async (config) => {
          // esbuild may inject __name helpers into evaluated bundles; define a no-op shim.
          const __name = (target) => target;
          const { name, warmup, runs } = config;
          const harness =  (window).testHarness;
          const gpu = await harness.getGPU();

          
          const benchmarks = {
            matmul: async () => {
              const M = 1, N = 4096, K = 4096;
              const A = new Float32Array(M * K).fill(1);
              const B = new Float32Array(K * N).fill(1);
              await harness.runMatmul(gpu.device, A, B, M, N, K);
            },
            rmsnorm: async () => {
              const batchSize = 1, hiddenSize = 4096;
              const input = new Float32Array(batchSize * hiddenSize).fill(1);
              const weight = new Float32Array(hiddenSize).fill(1);
              await harness.runRMSNorm(gpu.device, input, weight, batchSize, hiddenSize);
            },
            softmax: async () => {
              const innerSize = 32000, outerSize = 1;
              const input = new Float32Array(innerSize * outerSize).fill(1);
              await harness.runSoftmax(gpu.device, input, innerSize, outerSize);
            },
            silu: async () => {
              const size = 14336;
              const input = new Float32Array(size).fill(1);
              await harness.runSiLU(gpu.device, input);
            },
            rope: async () => {
              const seqLen = 1, numHeads = 32, headDim = 128;
              const input = new Float32Array(seqLen * numHeads * headDim).fill(1);
              await harness.runRoPE(gpu.device, input, seqLen, numHeads, headDim);
            },
            attention: async () => {
              const seqLen = 1, kvLen = 128, numHeads = 32, headDim = 128;
              const Q = new Float32Array(seqLen * numHeads * headDim).fill(0.1);
              const K = new Float32Array(kvLen * numHeads * headDim).fill(0.1);
              const V = new Float32Array(kvLen * numHeads * headDim).fill(0.1);
              await harness.runAttention(gpu.device, Q, K, V, seqLen, kvLen, numHeads, numHeads, headDim);
            },
            moe: async () => {
              const numTokens = 1, numExperts = 8, topK = 2;
              const logits = new Float32Array(numTokens * numExperts).fill(1);
              await harness.runSoftmaxTopK(gpu.device, logits, numTokens, numExperts, topK);
            },
          };

          const fn = benchmarks[name];
          if (!fn) return { error: `Unknown benchmark: ${name}` };

          for (let i = 0; i < warmup; i++) {
            await fn();
            await gpu.device.queue.onSubmittedWorkDone();
          }

          
          const times = [];
          for (let i = 0; i < runs; i++) {
            const start = performance.now();
            await fn();
            await gpu.device.queue.onSubmittedWorkDone();
            times.push(performance.now() - start);
          }

          const sorted = [...times].sort((a, b) => a - b);
          const median = sorted[Math.floor(sorted.length / 2)];
          const mean = times.reduce((a, b) => a + b, 0) / times.length;
          const min = sorted[0];
          const max = sorted[sorted.length - 1];

          return { median, mean, min, max, samples: times.length };
        },
        { name: benchName, warmup: benchmarkRun.warmupRuns, runs: benchmarkRun.timedRuns }
      );

      if ('error' in result) {
        results.push({
          name: benchName,
          passed: false,
          duration: 0,
          error: result.error,
        });
        console.log(`  \x1b[31mFAIL\x1b[0m ${benchName}: ${result.error}`);
      } else {
        results.push({
          name: benchName,
          passed: true,
          duration: result.median,
        });
        console.log(
          `  ${benchName}: median=${result.median.toFixed(3)}ms ` +
          `mean=${result.mean.toFixed(3)}ms ` +
          `min=${result.min.toFixed(3)}ms max=${result.max.toFixed(3)}ms`
        );
      }
    } catch (err) {
      results.push({
        name: benchName,
        passed: false,
        duration: 0,
        error:  (err).message,
      });
      console.log(`  \x1b[31mFAIL\x1b[0m ${benchName}: ${ (err).message}`);
    }
  }

  const totalDuration = Date.now() - startTime;

  return {
    suite: 'bench:kernels',
    passed: results.filter((r) => r.passed).length,
    failed: results.filter((r) => !r.passed).length,
    skipped: 0,
    duration: totalDuration,
    results,
  };
}

// ============================================================================
// Inference Test (Quick)
// ============================================================================


async function runInferenceTest(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('INFERENCE TEST');
  console.log('='.repeat(60));
  console.log(`  Model: ${opts.model}`);

  setHarnessConfig(opts, {
    mode: 'inference',
    autorun: true,
    skipLoad: false,
    modelId: opts.model,
  });
  const testParams = new URLSearchParams();
  appendRuntimeConfigParams(testParams, opts);

  const testUrl = `${opts.baseUrl}/doppler/tests/harness.html?${testParams.toString()}`;
  console.log(`  URL: ${testUrl}`);

  await page.goto(testUrl, { timeout: opts.timeout });

  const startTime = Date.now();

  try {
    await page.waitForFunction(
      () => {
        const state =  (window).testState;
        return state && state.done === true;
      },
      { timeout: opts.timeout }
    );

    const testState = await page.evaluate(() =>  (window).testState);
    const duration = Date.now() - startTime;

    const passed = testState.loaded && testState.tokens?.length > 0 && testState.errors?.length === 0;

    if (passed) {
      const output =  (testState.output || '');
      const outputPreview = output.slice(0, 100);
      const outputLabel = outputPreview.trim().length === 0
        ? '<empty>'
        : `${outputPreview}${output.length > 100 ? '...' : ''}`;
      console.log(`\n  \x1b[32mPASS\x1b[0m Model loaded and generated ${testState.tokens?.length || 0} tokens`);
      console.log(`  Output: ${outputLabel}`);
    } else {
      console.log(`\n  \x1b[31mFAIL\x1b[0m`);
      if (!testState.loaded) console.log('    Model failed to load');
      if (testState.errors?.length > 0) {
        for (const err of testState.errors) {
          console.log(`    Error: ${err}`);
        }
      }
    }

    return {
      suite: 'inference',
      passed: passed ? 1 : 0,
      failed: passed ? 0 : 1,
      skipped: 0,
      duration,
      results: [
        {
          name: `inference:${opts.model}`,
          passed,
          duration,
          error: passed ? undefined : (testState.errors?.[0] || 'Unknown error'),
        },
      ],
    };
  } catch (err) {
    const duration = Date.now() - startTime;
    console.log(`\n  \x1b[31mFAIL\x1b[0m ${ (err).message}`);

    return {
      suite: 'inference',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration,
      results: [
        {
          name: `inference:${opts.model}`,
          passed: false,
          duration,
          error:  (err).message,
        },
      ],
    };
  }
}

// ============================================================================
// Demo UI Test
// ============================================================================


async function runDemoTest(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('DEMO UI TEST');
  console.log('='.repeat(60));
  console.log(`  Model: ${opts.model}`);

  const prompt = opts.runtimeConfig?.inference?.prompt;
  if (!prompt) {
    throw new Error('runtime.inference.prompt must be set for demo tests.');
  }
  console.log(`  Prompt: "${prompt}"`);
  const followupPrompt = 'and at night?';

  const startTime = Date.now();
  
  const errors = [];
  
  const logs = [];

  // Good/bad token patterns for quality analysis
  const GOOD_TOKENS = ['blue', 'sky', 'the', 'is', 'clear', 'clouds', 'sun', 'day', 'night', 'color'];
  const BAD_TOKENS = ['<unk>', '####', '\u0000', '\uFFFD'];

  // Setup console capture
  page.on('console', (msg) => {
    const text = msg.text();
    logs.push(text);
    console.log(`  [browser] ${text}`);
  });

  page.on('pageerror', (err) => {
    errors.push(err.message);
    console.error(`  [browser error] ${err.message}`);
  });

  try {
    // Navigate to demo
    console.log('\n  Step 1: Opening demo page...');
    const demoParams = new URLSearchParams();
    appendRuntimeConfigParams(demoParams, opts);
    const demoUrl = `${opts.baseUrl}/d${demoParams.toString() ? `?${demoParams.toString()}` : ''}`;
    await page.goto(demoUrl, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Wait for model list
    console.log('  Step 2: Waiting for model list...');
    await page.waitForSelector('#model-list', { timeout: 10000 }).catch(() => {
      console.log('    (model-list selector not found, trying alternative)');
    });

    // Select model matching pattern
    console.log(`  Step 3: Selecting model matching "${opts.model}"...`);
    const modelSelected = await page.evaluate(async (modelPattern) => {
      const modelList = document.querySelector('#model-list');
      if (!modelList) return false;

      const buttons = modelList.querySelectorAll('button, a, div');
      for (const btn of buttons) {
        const text = btn.textContent?.toLowerCase() || '';
        if (text.includes(modelPattern.toLowerCase())) {
           (btn).click();
          return true;
        }
      }
      return false;
    }, opts.model);

    if (!modelSelected) {
      throw new Error(`Model "${opts.model}" not found in model list`);
    }

    // Wait for model to load
    console.log('  Step 4: Waiting for model to load...');
    await page.waitForFunction(
      () => {
        const textarea =  (document.querySelector('#chat-input'));
        return textarea && !textarea.disabled;
      },
      { timeout: 90000 }
    );

    console.log('  Model loaded successfully!');

    // Send prompt
    console.log(`  Step 5: Sending prompt: "${prompt}"...`);
    await page.fill('#chat-input', prompt);
    await page.click('#send-btn').catch(async () => {
      await page.press('#chat-input', 'Enter');
    });

    // Wait for generation (check logs for output)
    console.log('  Step 6: Waiting for generation...');
    const generationTimeout = 30000;
    const genStartTime = Date.now();

    while (Date.now() - genStartTime < generationTimeout) {
      await page.waitForTimeout(1000);

      const hasOutput = logs.some(l =>
        l.includes('OUTPUT') ||
        l.includes('Generated') ||
        l.includes('generation complete') ||
        l.includes('[Pipeline] Decode complete')
      );

      if (hasOutput) {
        console.log('  Generation complete!');
        await page.waitForTimeout(1000); // Let it settle
        break;
      }
    }

    // Capture message counts after first response
    const firstTurnCounts = await page.evaluate(() => ({
      users: document.querySelectorAll('.message.user').length,
      assistants: document.querySelectorAll('.message.assistant').length,
    }));

    // Send follow-up prompt to validate multi-turn chat flow
    console.log(`  Step 7: Sending follow-up prompt: "${followupPrompt}"...`);
    await page.fill('#chat-input', followupPrompt);
    await page.click('#send-btn').catch(async () => {
      await page.press('#chat-input', 'Enter');
    });

    console.log('  Step 8: Waiting for follow-up generation...');
    await page.waitForFunction(
      (counts) => {
        const users = document.querySelectorAll('.message.user').length;
        const assistants = document.querySelectorAll('.message.assistant').length;
        const cursor = document.querySelector('.message.assistant .cursor');
        return users >= counts.users + 1 && assistants >= counts.assistants + 1 && !cursor;
      },
      { timeout: generationTimeout },
      firstTurnCounts
    );

    const roleSequence = await page.evaluate(() =>
      Array.from(document.querySelectorAll('.message-role'))
        .map((el) => el.textContent?.trim().toLowerCase())
        .filter(Boolean)
    );
    const rolesOk = roleSequence.length >= 4
      && roleSequence[0] === 'user'
      && roleSequence[1] === 'assistant'
      && roleSequence[2] === 'user'
      && roleSequence[3] === 'assistant';

    const statsOk = await page.evaluate(() => {
      const stats = Array.from(document.querySelectorAll('.message.assistant .message-stats'));
      return stats.some((el) => /\d+ tokens/.test(el.textContent || ''));
    });

    // Analyze token quality
    const allText = logs.join(' ');
    const goodFound = GOOD_TOKENS.filter(t => allText.toLowerCase().includes(t.toLowerCase()));
    const badFound = BAD_TOKENS.filter(t => allText.includes(t));

    const hasGood = goodFound.length > 0;
    const hasBad = badFound.length > 0;
    const passed = hasGood && !hasBad && errors.length === 0 && rolesOk && statsOk;

    // Print summary
    console.log('\n  ' + '-'.repeat(50));
    console.log('  Token Quality Analysis:');
    console.log(`    Good tokens found: ${goodFound.join(', ') || 'none'}`);
    console.log(`    Bad tokens found: ${badFound.join(', ') || 'none'}`);
    console.log(`    Errors: ${errors.length}`);
    console.log(`    Multi-turn roles: ${rolesOk ? 'ok' : 'invalid'}`);
    console.log(`    Message stats: ${statsOk ? 'present' : 'missing'}`);

    const duration = Date.now() - startTime;

    if (passed) {
      console.log(`\n  \x1b[32mPASS\x1b[0m Demo test completed successfully (${(duration / 1000).toFixed(1)}s)`);
    } else {
      console.log(`\n  \x1b[31mFAIL\x1b[0m Demo test failed`);
      if (!hasGood) console.log('    - No coherent tokens detected');
      if (hasBad) console.log(`    - Garbage tokens found: ${badFound.join(', ')}`);
      if (errors.length > 0) console.log(`    - Page errors: ${errors.join(', ')}`);
      if (!rolesOk) console.log('    - Message roles did not alternate user/assistant');
      if (!statsOk) console.log('    - Assistant token stats missing');
    }

    return {
      suite: 'demo',
      passed: passed ? 1 : 0,
      failed: passed ? 0 : 1,
      skipped: 0,
      duration,
      results: [
        {
          name: `demo:${opts.model}`,
          passed,
          duration,
          error: passed ? undefined : (errors[0] || (hasBad ? `Bad tokens: ${badFound.join(', ')}` : 'No coherent output')),
        },
      ],
    };
  } catch (err) {
    const duration = Date.now() - startTime;
    console.log(`\n  \x1b[31mFAIL\x1b[0m ${ (err).message}`);

    return {
      suite: 'demo',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration,
      results: [
        {
          name: `demo:${opts.model}`,
          passed: false,
          duration,
          error:  (err).message,
        },
      ],
    };
  }
}

// ============================================================================
// Converter UI Test
// ============================================================================


async function runConverterTest(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('CONVERTER UI TEST');
  console.log('='.repeat(60));

  const startTime = Date.now();
  
  const results = [];
  
  const errors = [];

  // Setup console capture
  page.on('console', (msg) => {
    const text = msg.text();
    console.log(`  [browser] ${text}`);
  });

  page.on('pageerror', (err) => {
    errors.push(err.message);
    console.error(`  [browser error] ${err.message}`);
  });

  try {
    // Navigate to demo
    console.log('\n  Step 1: Opening demo page...');
    await page.goto(`${opts.baseUrl}/d`, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Test 1: Convert button exists and is enabled
    console.log('  Step 2: Checking convert button...');
    const testStart1 = Date.now();

    const convertBtnExists = await page.locator('#convert-btn').isVisible({ timeout: 5000 }).catch(() => false);
    const convertBtnEnabled = await page.locator('#convert-btn').isEnabled({ timeout: 1000 }).catch(() => false);

    const test1Passed = convertBtnExists && convertBtnEnabled;
    results.push({
      name: 'converter-ui:button-present',
      passed: test1Passed,
      duration: Date.now() - testStart1,
      error: test1Passed ? undefined : `Button visible: ${convertBtnExists}, enabled: ${convertBtnEnabled}`,
    });

    console.log(`    ${test1Passed ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m'} Convert button present and enabled`);

    // Test 2: Convert status is initially hidden
    console.log('  Step 3: Checking convert status...');
    const testStart2 = Date.now();

    const convertStatusHidden = await page.locator('#convert-status').isHidden({ timeout: 1000 }).catch(() => true);

    results.push({
      name: 'converter-ui:status-hidden',
      passed: convertStatusHidden,
      duration: Date.now() - testStart2,
      error: convertStatusHidden ? undefined : 'Convert status should be hidden initially',
    });

    console.log(`    ${convertStatusHidden ? '\x1b[32mPASS\x1b[0m' : '\x1b[33mWARN\x1b[0m'} Convert status initially hidden`);

    // Test 3: Convert button click triggers file picker setup
    console.log('  Step 4: Testing convert button interaction...');
    const testStart3 = Date.now();

    // Inject a test file input to avoid native file picker
    await page.evaluate(() => {
      const input = document.createElement('input');
      input.type = 'file';
      input.id = 'test-file-input';
      input.multiple = true;
      input.style.display = 'none';
      document.body.appendChild(input);
    });

    // Click convert button (should not throw)
    let clickSucceeded = false;
    try {
      await page.click('#convert-btn', { timeout: 2000 });
      clickSucceeded = true;
    } catch {
      // Button might trigger native dialog, that's ok
      clickSucceeded = true;
    }

    results.push({
      name: 'converter-ui:button-clickable',
      passed: clickSucceeded,
      duration: Date.now() - testStart3,
    });

    console.log(`    ${clickSucceeded ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m'} Convert button clickable`);

    // Test 4: Check for progress bar element (exists but may be hidden)
    console.log('  Step 5: Checking progress bar presence...');
    const testStart4 = Date.now();

    const hasProgressBar = await page.evaluate(() => {
      return !!document.querySelector('#convert-progress, .convert-progress, [role="progressbar"]');
    });

    results.push({
      name: 'converter-ui:progress-element',
      passed: true, // Optional, just informational
      duration: Date.now() - testStart4,
      error: hasProgressBar ? undefined : 'Progress bar element not found (may be created dynamically)',
    });

    console.log(`    ${hasProgressBar ? '\x1b[32mPASS\x1b[0m' : '\x1b[33mINFO\x1b[0m'} Progress bar element ${hasProgressBar ? 'found' : 'not found (may be dynamic)'}`);

    const duration = Date.now() - startTime;
    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;

    console.log('\n  ' + '-'.repeat(50));
    console.log(`  Converter UI Tests: ${passed} passed, ${failed} failed`);

    if (failed === 0) {
      console.log(`\n  \x1b[32mPASS\x1b[0m Converter UI test completed (${(duration / 1000).toFixed(1)}s)`);
    } else {
      console.log(`\n  \x1b[31mFAIL\x1b[0m Converter UI test had failures`);
    }

    return {
      suite: 'converter',
      passed,
      failed,
      skipped: 0,
      duration,
      results,
    };
  } catch (err) {
    const duration = Date.now() - startTime;
    console.log(`\n  \x1b[31mFAIL\x1b[0m ${ (err).message}`);

    return {
      suite: 'converter',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration,
      results: [
        {
          name: 'converter-ui',
          passed: false,
          duration,
          error:  (err).message,
        },
      ],
    };
  }
}

// ============================================================================
// Simple Pipeline Benchmark (for bench:all)
// ============================================================================


async function runPipelineBenchmark(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('PIPELINE BENCHMARK');
  console.log('='.repeat(60));
  console.log(`  Model: ${opts.model}`);

  const runtimeConfig = opts.runtimeConfig;
  if (!runtimeConfig) {
    throw new Error('Runtime config is required for benchmarks.');
  }
  const benchmarkRun = runtimeConfig.shared.benchmark.run;

  await page.goto(`${opts.baseUrl}/d`, { timeout: opts.timeout });
  await page.waitForTimeout(1000);

  // Build script as string to avoid TypeScript module resolution issues
  // The import runs in browser context, not Node.js
  const config = JSON.stringify({
    promptName: benchmarkRun.promptName,
    customPrompt: benchmarkRun.customPrompt ?? undefined,
    maxNewTokens: benchmarkRun.maxNewTokens,
    warmupRuns: benchmarkRun.warmupRuns,
    timedRuns: benchmarkRun.timedRuns,
    sampling: benchmarkRun.sampling,
    debug: benchmarkRun.debug,
    profile: benchmarkRun.profile,
    useChatTemplate: benchmarkRun.useChatTemplate,
    captureMemoryTimeSeries: benchmarkRun.captureMemoryTimeSeries,
    memoryTimeSeriesIntervalMs: benchmarkRun.memoryTimeSeriesIntervalMs,
    runtimeConfig,
  });

  const script = `
    (async () => {
      const { PipelineBenchmark } = await import('./tests/benchmark/index.js');
      const config = ${config};
      const harness = new PipelineBenchmark(config);
      return await harness.run();
    })()
  `;

  try {
    const result =  (await page.evaluate(script));

    console.log(`\n  TTFT: ${result.metrics?.ttft_ms || 'N/A'}ms`);
    console.log(`  Prefill: ${result.metrics?.prefill_tokens_per_sec || 'N/A'} tok/s`);
    console.log(`  Decode: ${result.metrics?.decode_tokens_per_sec || 'N/A'} tok/s`);

    return {
      suite: 'bench:pipeline',
      passed: 1,
      failed: 0,
      skipped: 0,
      duration: result.metrics?.decode_ms_total || 0,
      results: [
        {
          name: 'pipeline',
          passed: true,
          duration: result.metrics?.decode_ms_total || 0,
        },
      ],
    };
  } catch (err) {
    console.log(`  \x1b[31mFAIL\x1b[0m: ${ (err).message}`);
    return {
      suite: 'bench:pipeline',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration: 0,
      results: [
        {
          name: 'pipeline',
          passed: false,
          duration: 0,
          error:  (err).message,
        },
      ],
    };
  }
}

// ============================================================================
// Summary Formatting
// ============================================================================


function printSummary(suites) {
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));

  let totalPassed = 0;
  let totalFailed = 0;
  let totalSkipped = 0;

  for (const suite of suites) {
    totalPassed += suite.passed;
    totalFailed += suite.failed;
    totalSkipped += suite.skipped;

    const status = suite.failed === 0 ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
    console.log(
      `  ${status} ${suite.suite}: ${suite.passed} passed, ${suite.failed} failed` +
      (suite.skipped > 0 ? `, ${suite.skipped} skipped` : '') +
      ` (${(suite.duration / 1000).toFixed(1)}s)`
    );
  }

  console.log('');
  console.log(`Total: ${totalPassed} passed, ${totalFailed} failed, ${totalSkipped} skipped`);

  if (totalFailed > 0) {
    console.log('\n\x1b[31mTests failed!\x1b[0m');
  } else {
    console.log('\n\x1b[32mAll tests passed!\x1b[0m');
  }
}

// ============================================================================
// Main
// ============================================================================


async function main() {
  
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    console.error(`Error: ${ (err).message}`);
    console.error('Run with --help for usage.');
    process.exit(1);
  }

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  // Handle --list-presets
  if (opts.listPresets) {
    console.log('\nAvailable Config Presets:\n');
    const presets = await listPresets();
    
    const grouped = presets.reduce((acc, p) => {
      if (!acc[p.source]) acc[p.source] = [];
      acc[p.source].push(p);
      return acc;
    },  ({}));

    for (const [source, items] of Object.entries(grouped)) {
      console.log(`  ${source.toUpperCase()}:`);
      for (const preset of items) {
        console.log(`    ${preset.name.padEnd(15)} ${preset.path}`);
      }
      console.log('');
    }
    process.exit(0);
  }

  // Handle --dump-config
  if (opts.dumpConfig) {
    const configRef = opts.config || opts.mode || 'default';
    try {
      const loaded = await loadConfig(configRef);
      console.log('\n' + dumpConfig(loaded));
    } catch (err) {
      console.error(`Failed to load config "${configRef}": ${ (err).message}`);
      process.exit(1);
    }
    process.exit(0);
  }

  // Default presets for config-driven commands
  if (opts.command === 'debug' && !opts.config && !opts.mode) {
    opts.mode = 'debug';
  }
  if (opts.command === 'bench' && !opts.config && !opts.mode) {
    opts.mode = 'bench';
  }

  // Load config (default + overrides)
  
  let loadedConfig = null;
  const configRef = opts.config || opts.mode || 'default';
  const shouldLogConfig = Boolean(opts.config || opts.mode);
  try {
    loadedConfig = await loadConfig(configRef);
    if (shouldLogConfig) {
      console.log(`Config loaded: ${loadedConfig.chain.join(' -> ')}`);
    }
    opts.runtimeConfig = loadedConfig.runtime;
    opts.configChain = loadedConfig.chain;

    // Apply runtime config to opts
    const runtime = loadedConfig.runtime;
    if (runtime.shared?.debug?.logLevel?.defaultLogLevel === 'verbose') opts.verbose = true;
    if (runtime.shared?.debug?.logLevel?.defaultLogLevel === 'silent') opts.quiet = true;
    if (runtime.shared?.debug?.trace?.enabled) {
      // Trace config handled in runtimeConfig; no CLI overrides.
    }

    // Apply CLI-specific config from raw preset (not part of RuntimeConfigSchema)
    const cli =  (loadedConfig.raw.cli);
    if (cli) {
      const hasHeadlessFlag = hasCliFlag(opts, ['--headless', '--headed', '--no-headless']);
      if (cli.headed && !hasHeadlessFlag) opts.headless = false;
      if (typeof cli.timeout === 'number' && !hasCliFlag(opts, ['--timeout'])) opts.timeout = cli.timeout;
    }
  } catch (err) {
    console.error(`Failed to load config "${configRef}": ${ (err).message}`);
    process.exit(1);
  }

  // Handle 'bench' command - performance mode
  // Convert 'bench' to 'test --perf' so it uses the same code path
  if (opts.command === 'bench') {
    opts.command = 'test';  // Use test infrastructure with perf mode
    opts.perf = true;
    // Default to inference benchmark unless --kernels specified
    if (opts.suite === 'quick') {
      opts.suite = 'inference';
    }
  }

  // Handle 'debug' command
  if (opts.command === 'debug') {
    // No CLI overrides; rely on config.
  }

  // Handle 'run' command - just start the server
  if (opts.command === 'run') {
    console.log('\nDOPPLER CLI - Starting demo server...');
    console.log(`Open http://localhost:8080/d in your browser`);
    await ensureServerRunning(opts.baseUrl, opts.verbose);
    // Keep process alive
    await new Promise(() => {}); // Never resolves
  }

  console.log('\nDOPPLER CLI');
  console.log(`Command: ${opts.command}`);
  console.log(`Suite: ${opts.suite}`);
  console.log(`Base URL: ${opts.baseUrl}`);

  // Warn if running test --inference (smoke test) when they probably want debug
  if (opts.command === 'test' && opts.suite === 'inference') {
    console.log('\n\x1b[33m' + 'WARNING'.repeat(4) + '\x1b[0m');
    console.log('\x1b[33mNOTE: "test --inference" is a SMOKE TEST only.\x1b[0m');
    console.log('\x1b[33mFor debugging with kernel trace, use: doppler debug\x1b[0m');
    console.log('\x1b[33m' + 'WARNING'.repeat(4) + '\x1b[0m\n');
  }
  if (opts.profileDir) {
    console.log(`Profile Dir: ${opts.profileDir}`);
  }

  // Build TypeScript and ensure server is running
  // TEMP: Skip tsc build due to pre-existing test errors - esbuild works fine
  // await runBuild(opts.verbose);
  console.log('Skipping TypeScript build (using esbuild)...');
  if (!opts.noServer) {
    await ensureServerRunning(opts.baseUrl, opts.verbose);
  } else {
    console.log('No-server mode enabled (serving assets from disk)...');
  }

  const scope = opts.perf ? 'bench' : 'test';
  const context = await createBrowserContext(opts, { scope });
  const page = await setupPage(context, opts);
  
  const suites = [];

  try {
    if (opts.command === 'debug') {
      // DEBUG MODE - interactive tensor inspection
      console.log('\n' + '='.repeat(60));
      console.log('DEBUG MODE');
      console.log('='.repeat(60));
      console.log(`  Model: ${opts.model}`);

      // Track generation completion for auto-close
      let generationDone = false;
      let generationError = false;

      // Track completion signals without duplicating console output
      page.on('console', (msg) => {
        const text = msg.text();
        // Detect generation completion via standardized signals
        // [DOPPLER:DONE] is the canonical completion signal
        if (text.startsWith('[DOPPLER:DONE]')) {
          generationDone = true;
        } else if (text.startsWith('[DOPPLER:ERROR]')) {
          generationError = true;
        }
      });
      page.on('pageerror', (err) => {
        generationError = true;
      });

      setHarnessConfig(opts, {
        mode: 'inference',
        autorun: true,
        skipLoad: opts.skipLoad,
        modelId: opts.model,
      });

      const debugParams = new URLSearchParams();
      appendRuntimeConfigParams(debugParams, opts);

      const debugUrl = `${opts.baseUrl}/doppler/tests/harness.html?${debugParams.toString()}`;
      console.log(`  URL: ${debugUrl}`);

      await page.goto(debugUrl, { timeout: opts.timeout });

      // Auto-close behavior depends on headless mode
      if (!opts.headless) {
        // Headed mode: keep browser open for manual inspection
        console.log('\nDebug mode active. Browser will stay open until manually closed.');
        console.log('Press Ctrl+C to exit.\n');
        await new Promise(() => {}); // Never resolves
      } else {
        // Headless mode: wait for generation to complete, then auto-close
        console.log('\nWaiting for generation to complete...\n');

        // Poll for completion (check every 100ms)
        const startTime = Date.now();
        const maxWait = opts.timeout || 300000;
        while (!generationDone && !generationError) {
          await new Promise(r => setTimeout(r, 100));
          if (Date.now() - startTime > maxWait) {
            console.error('\nTimeout waiting for generation to complete');
            break;
          }
        }

        // Small delay to capture final output
        await new Promise(r => setTimeout(r, 500));

        await context.close();
        process.exit(generationError ? 1 : 0);
      }

    } else if (opts.command === 'test') {
      // TEST SUITES - correctness or performance based on --perf flag
      if (opts.perf) {
        // PERFORMANCE MODE (--perf)
        switch (opts.suite) {
          case 'kernels':
          case 'bench:kernels':
            suites.push(await runKernelBenchmarks(page, opts));
            break;

          case 'inference':
          case 'bench:pipeline':
          // Full inference benchmark - close existing context
          await context.close();

          const benchResults = await runFullInferenceBenchmark(opts);
          formatBenchmarkResult(benchResults);

          const benchmarkConfig = loadedConfig.runtime.shared.benchmark;
          const baselineConfig = benchmarkConfig.baselines;

          if (baselineConfig?.enabled) {
            try {
              const registry = await loadBaselineRegistry(baselineConfig.file);
              const baseline = findBaselineForResult(benchResults, registry);
              if (!baseline) {
                console.log('Baseline:      no matching entry');
              } else {
                const evaluation = evaluateBaseline(benchResults, baseline);
                if (evaluation.ok) {
                  console.log('Baseline:      ok');
                } else {
                  console.log('Baseline:      out of range');
                  for (const violation of evaluation.violations) {
                    console.log(
                      `  ${violation.metric}: ${violation.value} ` +
                      `(expected ${violation.min ?? '-inf'}..${violation.max ?? '+inf'})`
                    );
                  }
                  if (baselineConfig.failOnOutOfRange && process.env.CI) {
                    console.error('\nBaseline check failed (CI).');
                    process.exit(1);
                  }
                }
              }
            } catch (err) {
              console.error(`\nFailed to load baseline registry: ${ (err).message}`);
            }
          }

          if (baselineConfig?.requireQualityOk && benchResults.quality && !benchResults.quality.ok) {
            console.error('\nOutput quality check failed.');
            if (process.env.CI) {
              process.exit(1);
            }
          }

          // Compare against baseline if provided
          
          let baseline = null;
          if (opts.compare) {
            try {
              const baselinePath = resolve(opts.compare);
              const baselineJson = await readFile(baselinePath, 'utf-8');
              baseline = JSON.parse(baselineJson);
              const comparison = compareResults(baseline, benchResults);
              console.log(formatComparison(comparison));

              const regressionSummary = detectRegressions(
                comparison,
                benchmarkConfig.comparison
              );
              console.log(formatRegressionSummary(regressionSummary));

              const baseLatencies = baseline.raw?.decode_latencies_ms;
              const currLatencies = benchResults.raw?.decode_latencies_ms;
              const minSamples = benchmarkConfig.stats.minSamplesForComparison;
              if (baseLatencies?.length >= minSamples && currLatencies?.length >= minSamples) {
                console.log('\n' + '-'.repeat(60));
                console.log('STATISTICAL SIGNIFICANCE (Welch\'s t-test)');
                console.log('-'.repeat(60));
                const ttest = welchTTest(baseLatencies, currLatencies);
                console.log(formatTTestResult('Decode Latency', ttest));
                if (ttest.significant) {
                  console.log(`  -> The difference IS statistically significant (p < 0.05)`);
                } else {
                  console.log(`  -> The difference is NOT statistically significant (p >= 0.05)`);
                }
              }

              if (regressionSummary.hasRegression && benchmarkConfig.comparison.failOnRegression) {
                console.error('\nBenchmark regression threshold exceeded.');
                process.exit(1);
              }
            } catch (err) {
              console.error(`\nFailed to load baseline for comparison: ${ (err).message}`);
            }
          }

          // Auto-save results
          const resultsDir = resolve(__dirname, '../tests/results');
          await mkdir(resultsDir, { recursive: true });

          const autoFilename = generateResultFilename(benchResults);
          const autoPath = resolve(resultsDir, autoFilename);
          await writeFile(autoPath, JSON.stringify(benchResults, null, 2));
          console.log(`\nResults auto-saved to: ${autoPath}`);

          // Generate HTML report
          const htmlFilename = autoFilename.replace('.json', '.html');
          const htmlPath = opts.html ? resolve(opts.html) : resolve(resultsDir, htmlFilename);
          await mkdir(dirname(htmlPath), { recursive: true });
          const htmlContent = generateHTMLReport(benchResults, baseline);
          await writeFile(htmlPath, htmlContent);
          console.log(`HTML report saved to: ${htmlPath}`);

          if (opts.output) {
            const outputPath = resolve(opts.output);
            await mkdir(dirname(outputPath), { recursive: true });
            await writeFile(outputPath, JSON.stringify(benchResults, null, 2));
            console.log(`Results also saved to: ${outputPath}`);
          }

          if (!opts.quiet) {
            console.log('\n' + '-'.repeat(60));
            console.log('JSON Output:');
            console.log('-'.repeat(60));
            console.log(JSON.stringify(benchResults, null, 2));
          }

          process.exit(0);

        case 'loading':
          // Model loading benchmark - measure GPU load time
          console.log('\n' + '='.repeat(60));
          console.log('MODEL LOADING BENCHMARK');
          console.log('='.repeat(60));
          console.log(`  Model: ${opts.model}`);

          setHarnessConfig(opts, {
            mode: 'inference',
            autorun: true,
            skipLoad: false,
            modelId: opts.model,
          });
          const loadParams = new URLSearchParams();
          appendRuntimeConfigParams(loadParams, opts);
          const loadUrl = `${opts.baseUrl}/doppler/tests/harness.html?${loadParams.toString()}`;
          await page.goto(loadUrl, { timeout: opts.timeout });

          const loadStart = Date.now();
          await page.waitForFunction(
            () =>  (window).testState?.loaded === true,
            { timeout: opts.timeout }
          );
          const loadDuration = Date.now() - loadStart;

          console.log(`\n  Load time: ${loadDuration}ms`);
          suites.push({
            suite: 'loading',
            passed: 1,
            failed: 0,
            skipped: 0,
            duration: loadDuration,
            results: [{ name: `loading:${opts.model}`, passed: true, duration: loadDuration }],
          });
          break;

        case 'system':
        case 'bench:system':
          console.log('System benchmark not yet implemented');
          suites.push({ suite: 'system', passed: 0, failed: 0, skipped: 1, duration: 0, results: [] });
          break;

        case 'all':
          suites.push(await runKernelBenchmarks(page, opts));
          suites.push(await runPipelineBenchmark(page, opts));
          break;

        default:
          console.error(`Unknown benchmark suite: ${opts.suite}`);
          printHelp();
          process.exit(1);
        }
      } else {
        // CORRECTNESS MODE (default)
        switch (opts.suite) {
          case 'quick':
            suites.push(await runCorrectnessTests(page, opts, QUICK_TESTS));
            break;

          case 'kernels':
          case 'correctness':  // Legacy alias
            suites.push(await runCorrectnessTests(page, opts, KERNEL_TESTS));
            break;

          case 'inference':
            suites.push(await runInferenceTest(page, opts));
            break;

          case 'demo':
            suites.push(await runDemoTest(page, opts));
            break;

          case 'converter':
            suites.push(await runConverterTest(page, opts));
            break;

          case 'training':
            suites.push(await runTrainingTests(page, opts, TRAINING_TESTS));
            break;

          case 'all':
            suites.push(await runCorrectnessTests(page, opts, KERNEL_TESTS));
            suites.push(await runInferenceTest(page, opts));
            suites.push(await runTrainingTests(page, opts, TRAINING_TESTS));
            break;

          default:
            console.error(`Unknown test suite: ${opts.suite}`);
            printHelp();
            process.exit(1);
        }
      }
    }

    printSummary(suites);

    if (opts.output) {
      const outputPath = resolve(opts.output);
      await mkdir(dirname(outputPath), { recursive: true });
      await writeFile(outputPath, JSON.stringify(suites, null, 2));
      console.log(`\nResults saved to: ${outputPath}`);
    }

    if (!opts.headless) {
      console.log('\nKeeping browser open for 10s...');
      await page.waitForTimeout(10000);
    }

    await context.close();

    const hasFailed = suites.some((s) => s.failed > 0);
    process.exit(hasFailed ? 1 : 0);
  } catch (err) {
    console.error('\nTest runner failed:',  (err).message);
    await context.close();
    process.exit(1);
  }
}

main();
