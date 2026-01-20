#!/usr/bin/env node


import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { rmSync } from 'fs';
import { open, writeFile, mkdir, readFile } from 'fs/promises';
import { tmpdir } from 'os';

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

import { parseArgs, hasCliFlag, setHarnessConfig, appendRuntimeConfigParams } from './args/index.js';
import { printHelp } from './help.js';
import { KERNEL_TESTS, TRAINING_TESTS, QUICK_TESTS } from './suites.js';
import { printSummary } from './output.js';

import {
  runCorrectnessTests,
  runTrainingTests,
  runKernelBenchmarks,
  runInferenceTest,
  runSimulationTest,
  runDemoTest,
  runConverterTest,
  runPipelineBenchmark,
} from './runners/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CLI_LOCK_FILENAME = 'doppler-cli.lock';

let cliLockPath = null;

function getCliLockPath() {
  return resolve(tmpdir(), CLI_LOCK_FILENAME);
}

function isProcessAlive(pid) {
  if (!Number.isFinite(pid)) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err?.code !== 'ESRCH';
  }
}

function registerCliLockCleanup(lockPath) {
  if (cliLockPath) return;
  cliLockPath = lockPath;
  const cleanup = () => {
    if (!cliLockPath) return;
    try {
      rmSync(cliLockPath, { force: true });
    } catch {}
  };
  process.on('exit', cleanup);
  process.on('SIGINT', () => {
    cleanup();
    process.exit(130);
  });
  process.on('SIGTERM', () => {
    cleanup();
    process.exit(143);
  });
}

async function acquireCliLock(command) {
  const lockPath = getCliLockPath();
  const payload = {
    pid: process.pid,
    command,
    cwd: process.cwd(),
    startedAt: new Date().toISOString(),
  };

  try {
    const handle = await open(lockPath, 'wx');
    await handle.writeFile(JSON.stringify(payload, null, 2));
    await handle.close();
    registerCliLockCleanup(lockPath);
    return;
  } catch (err) {
    if (err?.code !== 'EEXIST') {
      throw err;
    }
  }

  let existing = null;
  try {
    const raw = await readFile(lockPath, 'utf-8');
    existing = JSON.parse(raw);
  } catch {}

  const existingPid = existing?.pid;
  if (isProcessAlive(existingPid)) {
    const meta = existing?.startedAt ? ` started=${existing.startedAt}` : '';
    const cmd = existing?.command ? ` command=${existing.command}` : '';
    throw new Error(
      `Another DOPPLER CLI run is active (pid=${existingPid}${cmd}${meta}). ` +
      `If this is stale, remove ${lockPath}.`
    );
  }

  try {
    rmSync(lockPath, { force: true });
  } catch {}

  const handle = await open(lockPath, 'wx');
  await handle.writeFile(JSON.stringify(payload, null, 2));
  await handle.close();
  registerCliLockCleanup(lockPath);
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

  if (opts.command === 'debug' || opts.command === 'test' || opts.command === 'bench') {
    try {
      await acquireCliLock(opts.command);
    } catch (err) {
      console.error(`Error: ${ (err).message}`);
      process.exit(1);
    }
  }

  // Default presets for config-driven commands
  if (opts.command === 'debug' && !opts.config && !opts.mode) {
    opts.mode = 'debug';
  }
  if (opts.command === 'bench' && !opts.config && !opts.mode) {
    opts.mode = 'bench';
  }
  if (opts.command === 'test' && opts.suite === 'simulation' && !opts.config && !opts.mode) {
    opts.mode = 'simulation';
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
  if (opts.command === 'bench') {
    opts.perf = true;
    if (opts.suite === 'quick') {
      opts.suite = 'inference';
    }
  }

  // Handle 'run' command - just start the server
  if (opts.command === 'run') {
    console.log('\nDOPPLER CLI - Starting demo server...');
    console.log(`Open http://localhost:8080/d in your browser`);
    await ensureServerRunning(opts.baseUrl, opts.verbose);
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

  // Skip TypeScript build - using esbuild
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
      await runDebugMode(page, opts, context);
    } else if (opts.command === 'test' || opts.command === 'bench') {
      await runTestCommand(page, opts, suites, context, loadedConfig);
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

async function runDebugMode(page, opts, context) {
  console.log('\n' + '='.repeat(60));
  console.log('DEBUG MODE');
  console.log('='.repeat(60));
  console.log(`  Model: ${opts.model}`);

  let generationDone = false;
  let generationError = false;

  page.on('console', (msg) => {
    const text = msg.text();
    if (text.startsWith('[DOPPLER:DONE]')) {
      generationDone = true;
    } else if (text.startsWith('[DOPPLER:ERROR]')) {
      generationError = true;
    }
  });
  page.on('pageerror', () => {
    generationError = true;
  });

  // Strip 'models/' prefix if present - harness BASE_URL already includes /models
  const modelId = opts.model.replace(/^models\//, '');
  setHarnessConfig(opts, {
    mode: 'inference',
    autorun: true,
    skipLoad: opts.skipLoad,
    modelId,
  });

  const debugParams = new URLSearchParams();
  appendRuntimeConfigParams(debugParams, opts);

  const debugUrl = `${opts.baseUrl}/doppler/tests/harness.html?${debugParams.toString()}`;
  console.log(`  URL: ${debugUrl}`);

  await page.goto(debugUrl, { timeout: opts.timeout });

  if (!opts.headless) {
    console.log('\nDebug mode active. Browser will stay open until manually closed.');
    console.log('Press Ctrl+C to exit.\n');
    await new Promise(() => {});
  } else {
    console.log('\nWaiting for generation to complete...\n');

    const startTime = Date.now();
    const maxWait = opts.timeout || 300000;
    while (!generationDone && !generationError) {
      await new Promise(r => setTimeout(r, 100));
      if (Date.now() - startTime > maxWait) {
        console.error('\nTimeout waiting for generation to complete');
        break;
      }
    }

    await new Promise(r => setTimeout(r, 500));
    await context.close();
    process.exit(generationError ? 1 : 0);
  }
}

async function runTestCommand(page, opts, suites, context, loadedConfig) {
  const __dirname = dirname(fileURLToPath(import.meta.url));

  if (opts.perf) {
    // PERFORMANCE MODE
    switch (opts.suite) {
      case 'kernels':
      case 'bench:kernels':
        suites.push(await runKernelBenchmarks(page, opts));
        break;

      case 'inference':
      case 'bench:pipeline':
        await context.close();
        await runInferenceBenchmark(opts, loadedConfig);
        break;

      case 'loading':
        await runLoadingBenchmark(page, opts, suites);
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
    // CORRECTNESS MODE
    switch (opts.suite) {
      case 'quick':
        suites.push(await runCorrectnessTests(page, opts, QUICK_TESTS));
        break;

      case 'kernels':
      case 'correctness':
        suites.push(await runCorrectnessTests(page, opts, KERNEL_TESTS));
        break;

      case 'inference':
        suites.push(await runInferenceTest(page, opts));
        break;
      case 'simulation':
        suites.push(await runSimulationTest(page, opts));
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

async function runInferenceBenchmark(opts, loadedConfig) {
  const __dirname = dirname(fileURLToPath(import.meta.url));

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

      const regressionSummary = detectRegressions(comparison, benchmarkConfig.comparison);
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
}

async function runLoadingBenchmark(page, opts, suites) {
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
}

main();
