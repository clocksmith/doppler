#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { runNodeCommand } from '../src/tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../src/tooling/node-browser-command-runner.js';
import { TOOLING_COMMANDS } from '../src/tooling/command-api.js';

const NODE_WEBGPU_INCOMPLETE_MESSAGE = 'node command: WebGPU runtime is incomplete in Node';
const DEFAULT_BENCH_MODEL_ID = 'gemma-3-270m-it-wf16';
const DEFAULT_BENCH_SURFACE = 'browser';

function usage() {
  return [
    'Usage:',
    '  doppler convert <inputDir> <outputDir> [--model-id <id>] [--surface auto|node]',
    '  doppler debug --model-id <id> [--model-url <url>] [--runtime-preset <id>] [--runtime-config-url <url>] [--runtime-config-json <json>] [--surface auto|node|browser]',
    '  doppler bench [--model-id <id>] [--model-url <url>] [--runtime-preset <id>] [--runtime-config-url <url>] [--runtime-config-json <json>] [--surface auto|node|browser]',
    '  doppler test-model --suite <kernels|inference|diffusion|energy> [--model-id <id>] [--model-url <url>] [--runtime-preset <id>] [--runtime-config-url <url>] [--runtime-config-json <json>] [--surface auto|node|browser]',
    '',
    'Flags:',
    '  --surface <auto|node|browser>   Execution surface (default: auto)',
    '  --json                          Print machine-readable result JSON',
    '  --capture-output                Include captured output for supported suites',
    '  --keep-pipeline                 Keep loaded pipeline in result payload (node surface only)',
    '  --browser-channel <name>        Browser channel for Playwright launch (e.g. chrome)',
    '  --browser-executable <path>     Browser executable path for Playwright launch',
    '  --browser-headless <true|false> Headless browser mode (must be true)',
    '  --browser-port <port>           Static server port for browser relay (default: random)',
    '  --browser-timeout-ms <ms>       Browser command timeout (default: 180000)',
    '  --browser-arg <arg>            Extra launch arg (repeatable); WebGPU/Vulkan args are applied automatically',
    '  --browser-url-path <path>       Runner page path (default: /src/tooling/command-runner.html)',
    '  --browser-static-root <path>    Static server root directory (default: doppler root)',
    '  --browser-base-url <url>        Reuse an existing static server base URL',
    '  --browser-console               Stream browser console lines to stderr',
    '',
    `Bench Defaults: --model-id ${DEFAULT_BENCH_MODEL_ID}, --surface ${DEFAULT_BENCH_SURFACE}, --browser-console (browser channel auto-selected)`,
  ].join('\n');
}

function applyCommandDefaults(parsed) {
  if (!parsed || parsed.command !== 'bench') {
    return parsed;
  }

  const flags = { ...parsed.flags };
  if (!flags['model-id'] && !flags['model-url']) {
    flags['model-id'] = DEFAULT_BENCH_MODEL_ID;
  }
  if (!flags.surface) {
    flags.surface = DEFAULT_BENCH_SURFACE;
  }
  flags['browser-console'] = true;

  return {
    ...parsed,
    flags,
  };
}

function parseArgs(argv) {
  const out = {
    command: null,
    positional: [],
    flags: {},
  };

  if (!argv.length) return out;
  out.command = argv[0] ?? null;

  for (let i = 1; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) {
      out.positional.push(token);
      continue;
    }

    const key = token.slice(2);
    if (
      key === 'json'
      || key === 'capture-output'
      || key === 'keep-pipeline'
      || key === 'help'
      || key === 'h'
      || key === 'browser-console'
    ) {
      out.flags[key] = true;
      continue;
    }

    const value = argv[i + 1];
    if (value === undefined) {
      throw new Error(`Missing value for --${key}`);
    }

    if (key !== 'browser-arg' && value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }

    if (key === 'browser-arg') {
      const previous = out.flags[key];
      if (Array.isArray(previous)) {
        previous.push(value);
      } else {
        out.flags[key] = [value];
      }
      i += 1;
      continue;
    }

    out.flags[key] = value;
    i += 1;
  }

  return out;
}

function parseRuntimeConfigJson(value) {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('runtime config must be an object');
    }
    return parsed;
  } catch (error) {
    throw new Error(`Invalid --runtime-config-json: ${error.message}`);
  }
}

function parseBooleanFlag(value, label) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true') return true;
    if (normalized === 'false') return false;
  }
  throw new Error(`${label} must be true or false`);
}

function parseNumberFlag(value, label) {
  if (value === undefined || value === null || value === '') return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a number`);
  }
  return parsed;
}

function parseBrowserArgs(value) {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value.map((item) => String(item)) : [String(value)];
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function resolveStaticRootDir(parsed) {
  const configured = parsed.flags['browser-static-root'];
  if (configured) {
    return path.resolve(String(configured));
  }
  return process.cwd();
}

async function resolveBrowserModelUrl(request, parsed) {
  if (request.modelUrl || !request.modelId) {
    return request;
  }

  const modelId = String(request.modelId);
  const encodedModelId = encodeURIComponent(modelId);

  if (parsed.flags['browser-base-url']) {
    return {
      ...request,
      modelUrl: `/models/${encodedModelId}`,
    };
  }

  const staticRootDir = resolveStaticRootDir(parsed);
  const candidates = [
    {
      modelUrl: `/models/${encodedModelId}`,
      manifestPath: path.join(staticRootDir, 'models', modelId, 'manifest.json'),
    },
    {
      modelUrl: `/models/curated/${encodedModelId}`,
      manifestPath: path.join(staticRootDir, 'models', 'curated', modelId, 'manifest.json'),
    },
    {
      modelUrl: `/models/local/${encodedModelId}`,
      manifestPath: path.join(staticRootDir, 'models', 'local', modelId, 'manifest.json'),
    },
  ];

  for (const candidate of candidates) {
    if (await pathExists(candidate.manifestPath)) {
      return {
        ...request,
        modelUrl: candidate.modelUrl,
      };
    }
  }

  return {
    ...request,
    modelUrl: `/models/${encodedModelId}`,
  };
}

function parseSurface(value, command) {
  const normalized = String(value || 'auto').trim().toLowerCase();
  if (normalized !== 'auto' && normalized !== 'node' && normalized !== 'browser') {
    throw new Error('--surface must be one of auto, node, browser');
  }
  if (command === 'convert' && normalized === 'browser') {
    throw new Error('convert is not supported on browser relay. Use --surface node or --surface auto.');
  }
  return normalized;
}

function buildRequest(parsed) {
  const command = parsed.command;
  if (!command || !TOOLING_COMMANDS.includes(command)) {
    throw new Error(`Unsupported command "${command || ''}"`);
  }

  const common = {
    command,
    modelId: parsed.flags['model-id'] ?? null,
    modelUrl: parsed.flags['model-url'] ?? null,
    runtimePreset: parsed.flags['runtime-preset'] ?? null,
    runtimeConfigUrl: parsed.flags['runtime-config-url'] ?? null,
    runtimeConfig: parseRuntimeConfigJson(parsed.flags['runtime-config-json'] ?? null),
    captureOutput: parsed.flags['capture-output'] === true,
    keepPipeline: parsed.flags['keep-pipeline'] === true,
  };

  if (command === 'convert') {
    const inputDir = parsed.positional[0] ?? null;
    const outputDir = parsed.positional[1] ?? null;
    if (!inputDir || !outputDir) {
      throw new Error('convert requires <inputDir> <outputDir>');
    }
    return {
      ...common,
      inputDir,
      outputDir,
    };
  }

  if (command === 'test-model') {
    return {
      ...common,
      suite: parsed.flags.suite ?? null,
    };
  }

  return common;
}

function buildNodeRunOptions(jsonOutput) {
  return {
    onProgress(progress) {
      if (jsonOutput) return;
      if (!progress?.message) return;
      if (Number.isFinite(progress.current) && Number.isFinite(progress.total)) {
        console.error(`[progress] ${progress.current}/${progress.total} ${progress.message}`);
      } else {
        console.error(`[progress] ${progress.stage ?? 'run'} ${progress.message}`);
      }
    },
  };
}

function buildBrowserRunOptions(parsed, jsonOutput) {
  const headless = parseBooleanFlag(parsed.flags['browser-headless'], '--browser-headless');
  const port = parseNumberFlag(parsed.flags['browser-port'], '--browser-port');
  const timeoutMs = parseNumberFlag(parsed.flags['browser-timeout-ms'], '--browser-timeout-ms');

  if (headless === false) {
    throw new Error('--browser-headless=false is not supported. Browser relay always runs headless.');
  }

  const options = {
    channel: parsed.flags['browser-channel'] ?? null,
    executablePath: parsed.flags['browser-executable'] ?? null,
    runnerPath: parsed.flags['browser-url-path'] ?? null,
    staticRootDir: parsed.flags['browser-static-root'] ?? null,
    baseUrl: parsed.flags['browser-base-url'] ?? null,
    browserArgs: parseBrowserArgs(parsed.flags['browser-arg']),
  };

  options.headless = true;
  if (port !== null) {
    options.port = port;
  }
  if (timeoutMs !== null) {
    options.timeoutMs = timeoutMs;
  }

  if (parsed.flags['browser-console'] === true && !jsonOutput) {
    options.onConsole = ({ type, text }) => {
      console.error(`[browser:${type}] ${text}`);
    };
  }

  return options;
}

function isNodeWebGPUFallbackCandidate(error) {
  const message = error?.message || String(error || '');
  return message.includes(NODE_WEBGPU_INCOMPLETE_MESSAGE);
}

async function runCommandOnSurface(request, surface, parsed, jsonOutput) {
  if (surface === 'node') {
    return runNodeCommand(request, buildNodeRunOptions(jsonOutput));
  }

  const browserRequest = await resolveBrowserModelUrl(request, parsed);

  if (!jsonOutput) {
    console.error('[progress] browser launching WebGPU harness...');
    if (browserRequest.modelUrl && browserRequest.modelUrl !== request.modelUrl) {
      console.error(`[progress] browser resolved modelUrl=${browserRequest.modelUrl}`);
    }
  }

  return runBrowserCommandInNode(browserRequest, buildBrowserRunOptions(parsed, jsonOutput));
}

async function runWithAutoSurface(request, parsed, jsonOutput) {
  if (request.command === 'convert') {
    return runCommandOnSurface(request, 'node', parsed, jsonOutput);
  }

  try {
    return await runCommandOnSurface(request, 'node', parsed, jsonOutput);
  } catch (error) {
    if (!isNodeWebGPUFallbackCandidate(error)) {
      throw error;
    }
    return runCommandOnSurface(request, 'browser', parsed, jsonOutput);
  }
}

function toSummary(result) {
  if (!result || typeof result !== 'object') {
    return 'ok';
  }

  if (result.manifest?.modelId) {
    return `converted ${result.manifest.modelId} (${result.tensorCount} tensors, ${result.shardCount} shards)`;
  }

  const suite = result.suite || result.report?.suite || 'suite';
  const modelId = result.modelId || result.report?.modelId || 'unknown';
  const passed = Number.isFinite(result.passed) ? result.passed : null;
  const failed = Number.isFinite(result.failed) ? result.failed : null;
  const duration = Number.isFinite(result.duration) ? `${result.duration.toFixed(1)}ms` : 'n/a';
  if (passed !== null && failed !== null) {
    return `${suite} model=${modelId} passed=${passed} failed=${failed} duration=${duration}`;
  }
  return `${suite} model=${modelId}`;
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : 'n/a';
}

function formatMs(value) {
  return Number.isFinite(value) ? `${Number(value).toFixed(1)}ms` : 'n/a';
}

function quoteOneLine(value) {
  const s = String(value ?? '').replace(/\s+/g, ' ').trim();
  if (!s) return '""';
  const clipped = s.length > 120 ? `${s.slice(0, 117)}...` : s;
  return JSON.stringify(clipped);
}

function printMetricsSummary(result) {
  if (!result || typeof result !== 'object') return;
  const suite = String(result.suite || '');
  const metrics = result.metrics;
  if (!metrics || typeof metrics !== 'object') return;

  if (suite === 'inference' || suite === 'debug') {
    const prompt = quoteOneLine(metrics.prompt);
    console.log(`[metrics] prompt=${prompt}`);
    console.log(
      `[metrics] load=${formatMs(metrics.modelLoadMs)} ` +
      `prefillTokens=${Number.isFinite(metrics.prefillTokens) ? Math.round(metrics.prefillTokens) : 'n/a'} ` +
      `decodeTokens=${Number.isFinite(metrics.decodeTokens) ? Math.round(metrics.decodeTokens) : 'n/a'} ` +
      `maxTokens=${Number.isFinite(metrics.maxTokens) ? Math.round(metrics.maxTokens) : 'n/a'}`
    );
    console.log(
      `[metrics] ttft=${formatMs(metrics.ttftMs)} prefill=${formatMs(metrics.prefillMs)} ` +
      `decode=${formatMs(metrics.decodeMs)} total=${formatMs(metrics.totalMs)}`
    );
    console.log(
      `[metrics] tok/s total=${formatNumber(metrics.tokensPerSec)} ` +
      `prefill=${formatNumber(metrics.prefillTokensPerSec)} ` +
      `decode=${formatNumber(metrics.decodeTokensPerSec)}`
    );
    return;
  }

  if (suite === 'bench') {
    if (Number.isFinite(metrics.embeddingDim) || Number.isFinite(metrics.avgEmbeddingMs)) {
      console.log(`[metrics] prompt=${quoteOneLine(metrics.prompt)}`);
      console.log(
        `[metrics] load=${formatMs(metrics.modelLoadMs)} runs=${Number.isFinite(metrics.warmupRuns) ? metrics.warmupRuns : 'n/a'}+${Number.isFinite(metrics.timedRuns) ? metrics.timedRuns : 'n/a'}`
      );
      console.log(
        `[metrics] embedding dim=${Number.isFinite(metrics.embeddingDim) ? Math.round(metrics.embeddingDim) : 'n/a'} ` +
        `median=${formatMs(metrics.medianEmbeddingMs)} avg=${formatMs(metrics.avgEmbeddingMs)} ` +
        `eps=${formatNumber(metrics.avgEmbeddingsPerSec)}`
      );
      return;
    }

    console.log(`[metrics] prompt=${quoteOneLine(metrics.prompt)}`);
    console.log(
      `[metrics] load=${formatMs(metrics.modelLoadMs)} runs=${Number.isFinite(metrics.warmupRuns) ? metrics.warmupRuns : 'n/a'}+${Number.isFinite(metrics.timedRuns) ? metrics.timedRuns : 'n/a'} ` +
      `maxTokens=${Number.isFinite(metrics.maxTokens) ? Math.round(metrics.maxTokens) : 'n/a'}`
    );
    console.log(
      `[metrics] tokens prefill(avg)=${Number.isFinite(metrics.avgPrefillTokens) ? Math.round(metrics.avgPrefillTokens) : 'n/a'} ` +
      `decode(avg)=${Number.isFinite(metrics.avgDecodeTokens) ? Math.round(metrics.avgDecodeTokens) : 'n/a'} ` +
      `generated(avg)=${Number.isFinite(metrics.avgTokensGenerated) ? Math.round(metrics.avgTokensGenerated) : 'n/a'}`
    );
    console.log(
      `[metrics] tok/s median=${formatNumber(metrics.medianTokensPerSec)} avg=${formatNumber(metrics.avgTokensPerSec)} ` +
      `decode median=${formatNumber(metrics.medianDecodeTokensPerSec)} avg=${formatNumber(metrics.avgDecodeTokensPerSec)}`
    );
    console.log(
      `[metrics] latency ttft median=${formatMs(metrics.medianTtftMs)} ` +
      `prefill median=${formatMs(metrics.medianPrefillMs)} decode median=${formatMs(metrics.medianDecodeMs)}`
    );
  }
}

async function main() {
  const argv = process.argv.slice(2);
  if (!argv.length || argv[0] === '--help' || argv[0] === '-h') {
    console.log(usage());
    return;
  }

  const parsed = parseArgs(argv);
  const parsedWithDefaults = applyCommandDefaults(parsed);
  if (parsedWithDefaults.flags.help === true || parsedWithDefaults.flags.h === true) {
    console.log(usage());
    return;
  }

  const request = buildRequest(parsedWithDefaults);
  const jsonOutput = parsedWithDefaults.flags.json === true;
  const surface = parseSurface(parsedWithDefaults.flags.surface, request.command);

  let response;
  if (surface === 'auto') {
    response = await runWithAutoSurface(request, parsedWithDefaults, jsonOutput);
  } else {
    response = await runCommandOnSurface(request, surface, parsedWithDefaults, jsonOutput);
  }

  if (jsonOutput) {
    console.log(JSON.stringify(response, null, 2));
    return;
  }

  console.log(`[ok] ${toSummary(response.result)}`);
  printMetricsSummary(response.result);
}

main().catch((error) => {
  console.error(`[error] ${error?.message || String(error)}`);
  process.exit(1);
});
