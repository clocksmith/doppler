#!/usr/bin/env node


import { resolve, dirname } from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { rmSync } from 'fs';
import { open, writeFile, mkdir, readFile } from 'fs/promises';
import { tmpdir } from 'os';

import { loadConfig } from './config/index.js';

import {
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

import { parseArgs, setHarnessConfig, appendRuntimeConfigParams } from './args/index.js';
import { printHelp } from './help.js';
import { KERNEL_TESTS, TRAINING_TESTS, QUICK_TESTS } from './suites.js';
import { printSummary } from './output.js';
import { getTool, listTools } from './tools/registry.js';

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

const COMMANDS = new Set(['run', 'test', 'bench', 'debug', 'convert', 'tool']);
const TEST_SUITES = new Set([
  'kernels',
  'inference',
  'demo',
  'converter',
  'simulation',
  'training',
  'quick',
  'all',
]);
const BENCH_SUITES = new Set([
  'kernels',
  'inference',
  'loading',
  'system',
  'all',
]);

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

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertStringOrNull(value, label) {
  if (value === null) return;
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string or null`);
  }
}

function assertBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be a boolean`);
  }
}

function assertNumber(value, label, { min = null } = {}) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`${label} must be a number`);
  }
  if (min !== null && value < min) {
    throw new Error(`${label} must be >= ${min}`);
  }
}

function assertConverterConfig(raw) {
  if (!raw?.converter) {
    throw new Error('converter is required in config for cli.command="convert"');
  }
  assertObject(raw.converter, 'converter');
}

function assertToolConfig(raw, tool) {
  if (!tool?.configKey) return;
  const tools = raw?.tools;
  if (!tools || typeof tools !== 'object') {
    throw new Error(`tools.${tool.configKey} is required for cli.tool="${tool.id}"`);
  }
  if (!Object.prototype.hasOwnProperty.call(tools, tool.configKey)) {
    throw new Error(`tools.${tool.configKey} is required for cli.tool="${tool.id}"`);
  }
}

function parseCliConfig(cliConfig) {
  assertObject(cliConfig, 'cli');

  const allowedKeys = new Set([
    'command',
    'suite',
    'tool',
    'baseUrl',
    'noServer',
    'headless',
    'minimized',
    'reuseBrowser',
    'cdpEndpoint',
    'timeout',
    'retries',
    'profileDir',
    'output',
    'html',
    'compare',
    'filter',
  ]);

  for (const key of Object.keys(cliConfig)) {
    if (!allowedKeys.has(key)) {
      throw new Error(`cli.${key} is not supported`);
    }
  }

  if (!('command' in cliConfig)) {
    throw new Error('cli.command is required');
  }
  assertString(cliConfig.command, 'cli.command');
  if (!COMMANDS.has(cliConfig.command)) {
    throw new Error(`cli.command must be one of: ${[...COMMANDS].join(', ')}`);
  }

  const suite = cliConfig.suite ?? null;
  if (suite !== null) {
    assertString(suite, 'cli.suite');
  }
  const tool = cliConfig.tool ?? null;
  if (tool !== null) {
    assertString(tool, 'cli.tool');
  }

  const baseUrl = cliConfig.baseUrl ?? null;
  const noServer = cliConfig.noServer ?? null;
  const headless = cliConfig.headless ?? null;
  const minimized = cliConfig.minimized ?? null;
  const reuseBrowser = cliConfig.reuseBrowser ?? null;
  const cdpEndpoint = cliConfig.cdpEndpoint ?? null;
  const timeout = cliConfig.timeout ?? null;
  const retries = cliConfig.retries ?? null;
  const profileDir = cliConfig.profileDir ?? null;
  const output = cliConfig.output ?? null;
  const html = cliConfig.html ?? null;
  const compare = cliConfig.compare ?? null;
  const filter = cliConfig.filter ?? null;

  if (baseUrl !== null) assertString(baseUrl, 'cli.baseUrl');
  if (noServer !== null) assertBoolean(noServer, 'cli.noServer');
  if (headless !== null) assertBoolean(headless, 'cli.headless');
  if (minimized !== null) assertBoolean(minimized, 'cli.minimized');
  if (reuseBrowser !== null) assertBoolean(reuseBrowser, 'cli.reuseBrowser');
  if (cdpEndpoint !== null) assertStringOrNull(cdpEndpoint, 'cli.cdpEndpoint');
  if (timeout !== null) assertNumber(timeout, 'cli.timeout', { min: 1 });
  if (retries !== null) assertNumber(retries, 'cli.retries', { min: 0 });
  if (profileDir !== null) assertStringOrNull(profileDir, 'cli.profileDir');
  if (output !== null) assertStringOrNull(output, 'cli.output');
  if (html !== null) assertStringOrNull(html, 'cli.html');
  if (compare !== null) assertStringOrNull(compare, 'cli.compare');
  if (filter !== null) assertStringOrNull(filter, 'cli.filter');

  return {
    command: cliConfig.command,
    suite,
    tool,
    baseUrl,
    noServer,
    headless,
    minimized,
    reuseBrowser,
    cdpEndpoint,
    timeout,
    retries,
    profileDir,
    output,
    html,
    compare,
    filter,
  };
}

function resolveCommandAndSuite(cliConfig) {
  const command = cliConfig.command;
  const suite = cliConfig.suite ?? null;

  if (command === 'run') {
    if (suite) {
      throw new Error('cli.suite must be null for command "run"');
    }
    return { command, suite: null };
  }

  if (command === 'debug') {
    if (suite) {
      throw new Error('cli.suite must be null for command "debug"');
    }
    return { command, suite: null };
  }

  if (command === 'convert' || command === 'tool') {
    if (suite) {
      throw new Error(`cli.suite must be null for command "${command}"`);
    }
    return { command, suite: null };
  }

  if (!suite) {
    throw new Error(`cli.suite is required for command "${command}"`);
  }

  if (command === 'test' && !TEST_SUITES.has(suite)) {
    throw new Error(`Unknown test suite "${suite}"`);
  }
  if (command === 'bench' && !BENCH_SUITES.has(suite)) {
    throw new Error(`Unknown benchmark suite "${suite}"`);
  }

  return { command, suite };
}

function resolveModel(raw) {
  if (raw === undefined) return null;
  assertString(raw, 'model');
  return raw;
}

function assertToolingIntent(command, runtime) {
  if (command === 'run' || command === 'convert' || command === 'tool') return;
  const intent = runtime?.shared?.tooling?.intent ?? null;
  if (!intent) {
    throw new Error('runtime.shared.tooling.intent is required for CLI runs.');
  }

  const allowed = {
    debug: new Set(['investigate']),
    test: new Set(['verify']),
    bench: new Set(['calibrate', 'investigate']),
  }[command];

  if (allowed && !allowed.has(intent)) {
    throw new Error(
      `cli.command="${command}" requires runtime.shared.tooling.intent to be ` +
      `${[...allowed].join(' or ')}.`
    );
  }
}

function assertCliRequirements(command, cliConfig) {
  if (command === 'convert' || command === 'tool') {
    return;
  }

  if (cliConfig.baseUrl === null) throw new Error('cli.baseUrl is required');
  if (cliConfig.noServer === null) throw new Error('cli.noServer is required');
  if (cliConfig.headless === null) throw new Error('cli.headless is required');
  if (cliConfig.minimized === null) throw new Error('cli.minimized is required');
  if (cliConfig.reuseBrowser === null) throw new Error('cli.reuseBrowser is required');
  if (cliConfig.cdpEndpoint === null) throw new Error('cli.cdpEndpoint is required');
  if (cliConfig.timeout === null) throw new Error('cli.timeout is required');
  if (cliConfig.retries === null) throw new Error('cli.retries is required');
  if (cliConfig.profileDir === null) throw new Error('cli.profileDir is required');
}

function resolveToolId(tool) {
  if (!tool) {
    throw new Error('cli.tool is required for command "tool"');
  }
  return tool;
}

async function runNodeScript(script, args) {
  await new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, [script, ...args], { stdio: 'inherit' });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) {
        resolvePromise();
        return;
      }
      reject(new Error(`Process exited with code ${code}`));
    });
  });
}

async function runConverterCommand(configRef) {
  const script = resolve(__dirname, '..', 'src', 'converter', 'node-converter.js');
  await runNodeScript(script, ['--config', configRef]);
}

async function runToolCommand(toolId, configRef) {
  const tool = getTool(toolId);
  if (!tool) {
    const available = listTools().join(', ');
    throw new Error(`Unknown tool "${toolId}". Available: ${available}`);
  }
  await runNodeScript(tool.script, ['--config', configRef]);
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

  if (!opts.config) {
    console.error('Error: --config is required.');
    console.error('Run with --help for usage.');
    process.exit(1);
  }

  // Load config
  let loadedConfig = null;
  const configRef = opts.config;
  try {
    loadedConfig = await loadConfig(configRef);
    console.log(`Config loaded: ${loadedConfig.chain.join(' -> ')}`);
    opts.runtimeConfig = loadedConfig.runtime;
    opts.configChain = loadedConfig.chain;

    // Apply runtime config to opts
    const runtime = loadedConfig.runtime;
    if (runtime.shared?.debug?.logLevel?.defaultLogLevel === 'verbose') opts.verbose = true;
    if (runtime.shared?.debug?.logLevel?.defaultLogLevel === 'silent') opts.quiet = true;

    const cliConfig = parseCliConfig(loadedConfig.raw?.cli);
    const resolved = resolveCommandAndSuite(cliConfig);
    assertCliRequirements(resolved.command, cliConfig);
    opts.command = resolved.command;
    opts.suite = resolved.suite;
    opts.tool = cliConfig.tool ?? null;
    opts.baseUrl = cliConfig.baseUrl;
    opts.noServer = cliConfig.noServer;
    opts.headless = cliConfig.headless;
    opts.minimized = cliConfig.minimized;
    opts.reuseBrowser = cliConfig.reuseBrowser;
    opts.cdpEndpoint = cliConfig.cdpEndpoint;
    opts.timeout = cliConfig.timeout;
    opts.retries = cliConfig.retries;
    opts.profileDir = cliConfig.profileDir;
    opts.output = cliConfig.output;
    opts.html = cliConfig.html;
    opts.compare = cliConfig.compare;
    opts.filter = cliConfig.filter;

    if (opts.command === 'convert') {
      assertConverterConfig(loadedConfig.raw);
    }

    if (opts.command === 'tool') {
      const toolEntry = getTool(opts.tool ?? '');
      if (!toolEntry) {
        const available = listTools().join(', ');
        throw new Error(`Unknown tool "${opts.tool}". Available: ${available}`);
      }
      assertToolConfig(loadedConfig.raw, toolEntry);
    }

    opts.model = resolveModel(loadedConfig.raw?.model);
    const harnessModel = runtime.shared?.harness?.modelId ?? null;
    if (harnessModel && opts.model && harnessModel !== opts.model) {
      throw new Error(
        `Model mismatch: config.model="${opts.model}" vs runtime.shared.harness.modelId="${harnessModel}"`
      );
    }
    const requiresModel = new Set(['run', 'test', 'bench', 'debug']);
    if (requiresModel.has(opts.command) && !opts.model) {
      throw new Error('config.model is required for CLI runs.');
    }

    assertToolingIntent(opts.command, runtime);
  } catch (err) {
    console.error(`Failed to load config "${configRef}": ${ (err).message}`);
    process.exit(1);
  }

  if (opts.command === 'debug' || opts.command === 'test' || opts.command === 'bench') {
    try {
      await acquireCliLock(opts.command);
    } catch (err) {
      console.error(`Error: ${ (err).message}`);
      process.exit(1);
    }
  }

  if (opts.command === 'convert') {
    try {
      await runConverterCommand(opts.config);
      process.exit(0);
    } catch (err) {
      console.error(`Error: ${ (err).message}`);
      process.exit(1);
    }
  }

  if (opts.command === 'tool') {
    const toolId = resolveToolId(opts.tool);
    try {
      await runToolCommand(toolId, opts.config);
      process.exit(0);
    } catch (err) {
      console.error(`Error: ${ (err).message}`);
      process.exit(1);
    }
  }

  // Handle 'bench' command - performance mode
  if (opts.command === 'bench') {
    opts.perf = true;
  }

  // Handle 'run' command - just start the server
  if (opts.command === 'run') {
    console.log('\nDOPPLER CLI - Starting demo server...');
    console.log(`Open ${opts.baseUrl}/d in your browser`);
    await ensureServerRunning(opts.baseUrl, opts.verbose);
    await new Promise(() => {}); // Never resolves
  }

  console.log('\nDOPPLER CLI');
  console.log(`Command: ${opts.command}`);
  console.log(`Suite: ${opts.suite ?? '-'}`);
  console.log(`Base URL: ${opts.baseUrl}`);

  // Warn if running test --inference (smoke test) when they probably want debug
  if (opts.command === 'test' && opts.suite === 'inference') {
    console.log('\n\x1b[33m' + 'WARNING'.repeat(4) + '\x1b[0m');
    console.log('\x1b[33mNOTE: "test inference" is a SMOKE TEST only.\x1b[0m');
    console.log('\x1b[33mFor debugging with kernel trace, use: doppler debug\x1b[0m');
    console.log('\x1b[33m' + 'WARNING'.repeat(4) + '\x1b[0m\n');
  }
  if (opts.profileDir) {
    console.log(`Profile Dir: ${opts.profileDir}`);
  }

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
  const skipLoad = opts.runtimeConfig?.shared?.harness?.skipLoad ?? false;
  setHarnessConfig(opts, {
    mode: 'inference',
    autorun: true,
    skipLoad,
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
  if (opts.perf) {
    // PERFORMANCE MODE
    switch (opts.suite) {
      case 'kernels':
        suites.push(await runKernelBenchmarks(page, opts));
        break;

      case 'inference':
        await context.close();
        await runInferenceBenchmark(opts, loadedConfig);
        break;

      case 'loading':
        await runLoadingBenchmark(page, opts, suites);
        break;

      case 'system':
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
