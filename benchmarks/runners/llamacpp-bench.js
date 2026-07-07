#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';

const WORKLOADS = Object.freeze({
  'p064-d064-t0-k1': Object.freeze({ promptTokens: 64, decodeTokens: 64 }),
  'p064-d064-t1-k32': Object.freeze({ promptTokens: 64, decodeTokens: 64 }),
  'p256-d128-t0-k1': Object.freeze({ promptTokens: 256, decodeTokens: 128 }),
  'p256-d128-t1-k32': Object.freeze({ promptTokens: 256, decodeTokens: 128 }),
  'p512-d128-t0-k1': Object.freeze({ promptTokens: 512, decodeTokens: 128 }),
});

function usage() {
  return [
    'Usage:',
    '  node benchmarks/runners/llamacpp-bench.js --model <model.gguf> --workload <id> --llama-bench <path> [--runs <n>] [--json]',
    '',
    'Examples:',
    '  node benchmarks/runners/llamacpp-bench.js --model /models/qwen.gguf --workload p512-d128-t0-k1 --llama-bench /home/x/src/llama.cpp/build-vulkan/bin/llama-bench --runs 15 --json',
    '',
    'Options:',
    '  --model <path>            GGUF model path. Required.',
    '  --workload <id>           Shared workload id. Required unless prompt/decode tokens are provided.',
    '  --prompt-tokens <n>       Prompt token count when not using --workload.',
    '  --decode-tokens <n>       Decode token count when not using --workload.',
    '  --llama-bench <path>      llama-bench executable. Falls back to LLAMACPP_BENCH.',
    '  --runs <n>                llama-bench repetitions. Default: 5.',
    '  --gpu-layers <n>          GPU layers passed to -ngl. Default: 99.',
    '  --batch-size <n>          Batch size passed to -b. Default: 2048.',
    '  --ubatch-size <n>         Microbatch size passed to -ub. Default: 512.',
    '  --threads <n>             Optional CPU thread count passed to -t.',
    '  --cache-type-k <type>     KV cache K dtype. Default: f16.',
    '  --cache-type-v <type>     KV cache V dtype. Default: f16.',
    '  --flash-attn <0|1>        Flash attention flag passed to -fa. Default: 0.',
    '  --mmap <0|1>              Memory map flag passed to -mmp. Default: 1.',
    '  --no-warmup              Pass --no-warmup to llama-bench.',
    '  --json                   Accepted for runner parity; stdout is always JSON.',
  ].join('\n');
}

function parseArgs(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (token === '--json' || token === '--no-warmup') {
      flags[token.slice(2)] = true;
      continue;
    }
    if (!token.startsWith('--')) {
      throw new Error(`Unexpected positional argument: ${token}`);
    }
    const key = token.slice(2);
    const value = argv[i + 1];
    if (value == null || value.startsWith('--')) {
      throw new Error(`Missing value for ${token}`);
    }
    flags[key] = value;
    i += 1;
  }
  return flags;
}

function parsePositiveInteger(value, label, defaultValue = null) {
  if (value == null || value === '') {
    if (defaultValue != null) return defaultValue;
    throw new Error(`${label} is required`);
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseZeroOne(value, label, defaultValue) {
  if (value == null || value === '') return defaultValue;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'off') return '0';
  if (normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on') return '1';
  throw new Error(`${label} must be 0 or 1`);
}

function resolveWorkload(flags) {
  const workloadId = typeof flags.workload === 'string' && flags.workload.trim() !== ''
    ? flags.workload.trim()
    : null;
  if (workloadId) {
    const workload = WORKLOADS[workloadId];
    if (!workload) {
      throw new Error(`Unsupported workload "${workloadId}"`);
    }
    return { workloadId, ...workload };
  }
  return {
    workloadId: null,
    promptTokens: parsePositiveInteger(flags['prompt-tokens'], '--prompt-tokens'),
    decodeTokens: parsePositiveInteger(flags['decode-tokens'], '--decode-tokens'),
  };
}

function ensureExecutable(filePath, label) {
  const resolved = typeof filePath === 'string' && filePath.trim() !== '' ? filePath.trim() : null;
  if (!resolved) {
    throw new Error(`${label} is required`);
  }
  if (!fs.existsSync(resolved)) {
    throw new Error(`${label} does not exist: ${resolved}`);
  }
  return resolved;
}

function ensureFile(filePath, label) {
  const resolved = typeof filePath === 'string' && filePath.trim() !== '' ? path.resolve(filePath.trim()) : null;
  if (!resolved) {
    throw new Error(`${label} is required`);
  }
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`${label} must be an existing file: ${resolved}`);
  }
  return resolved;
}

function parseJsonOutput(stdout) {
  const text = String(stdout || '').trim();
  if (text === '') {
    throw new Error('llama-bench produced empty stdout');
  }
  const parsed = JSON.parse(text);
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error('llama-bench JSON output must be a non-empty array');
  }
  return parsed;
}

function finiteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function findPhaseRow(rows, phase, tokenCount) {
  const promptPhase = phase === 'prefill';
  const row = rows.find((item) => {
    if (!item || typeof item !== 'object') return false;
    const nPrompt = Number(item.n_prompt);
    const nGen = Number(item.n_gen);
    if (promptPhase) return nPrompt === tokenCount && nGen === 0;
    return nPrompt === 0 && nGen === tokenCount;
  });
  if (!row) {
    throw new Error(`Missing ${phase} row for ${tokenCount} tokens in llama-bench output`);
  }
  return row;
}

function percentile(values, ratio) {
  if (!Array.isArray(values) || values.length === 0) return null;
  const sorted = values
    .filter((value) => typeof value === 'number' && Number.isFinite(value))
    .sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(ratio * sorted.length) - 1));
  return sorted[index];
}

function msPerTokenSamples(row) {
  const samples = Array.isArray(row.samples_ts) && row.samples_ts.length > 0
    ? row.samples_ts
    : [row.avg_ts];
  return samples
    .filter((value) => typeof value === 'number' && Number.isFinite(value) && value > 0)
    .map((tokensPerSecond) => 1000 / tokensPerSecond);
}

function inferGpuVendor(description) {
  const value = String(description || '').toLowerCase();
  if (value.includes('radeon') || value.includes('amd')) return 'amd';
  if (value.includes('nvidia') || value.includes('geforce') || value.includes('rtx')) return 'nvidia';
  if (value.includes('intel')) return 'intel';
  return null;
}

function buildLlamaBenchCommand(options) {
  const command = [
    options.llamaBench,
    '-m', options.modelPath,
    '-p', String(options.promptTokens),
    '-n', String(options.decodeTokens),
    '-r', String(options.runs),
    '-o', 'json',
    '-ngl', String(options.gpuLayers),
    '-b', String(options.batchSize),
    '-ub', String(options.ubatchSize),
    '-ctk', options.cacheTypeK,
    '-ctv', options.cacheTypeV,
    '-fa', options.flashAttn,
    '-mmp', options.mmap,
  ];
  if (options.threads !== null) {
    command.push('-t', String(options.threads));
  }
  if (options.noWarmup) {
    command.push('--no-warmup');
  }
  return command;
}

function runLlamaBench(command) {
  const [binary, ...args] = command;
  const result = spawnSync(binary, args, {
    cwd: process.cwd(),
    env: process.env,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 64,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    const stderr = String(result.stderr || '').trim();
    throw new Error(`llama-bench failed with status ${result.status}: ${stderr || command.join(' ')}`);
  }
  return parseJsonOutput(result.stdout);
}

function buildOutput(options, rows, command) {
  const prefill = findPhaseRow(rows, 'prefill', options.promptTokens);
  const decode = findPhaseRow(rows, 'decode', options.decodeTokens);
  const decodeMsSamples = msPerTokenSamples(decode);
  const runtimeVersion = [
    prefill.build_commit || decode.build_commit,
    prefill.build_number || decode.build_number,
  ].filter((value) => value != null && String(value).trim() !== '').join('+') || null;
  const gpuDescription = prefill.gpu_info || decode.gpu_info || null;
  const cpuInfo = prefill.cpu_info || decode.cpu_info || null;
  const prefillMs = finiteNumber(prefill.avg_ns, 'prefill.avg_ns') / 1_000_000;
  const decodeMs = finiteNumber(decode.avg_ns, 'decode.avg_ns') / 1_000_000;

  return {
    schemaVersion: 1,
    kind: 'llamacpp-bench',
    workload: {
      id: options.workloadId,
      promptTokens: options.promptTokens,
      decodeTokens: options.decodeTokens,
    },
    model: {
      id: options.modelId,
      path: options.modelPath,
      filename: prefill.model_filename || decode.model_filename || path.basename(options.modelPath),
      type: prefill.model_type || decode.model_type || null,
      sizeBytes: prefill.model_size || decode.model_size || null,
      parameters: prefill.model_n_params || decode.model_n_params || null,
    },
    metrics: {
      prefillTokensPerSec: finiteNumber(prefill.avg_ts, 'prefill.avg_ts'),
      decodeTokensPerSec: finiteNumber(decode.avg_ts, 'decode.avg_ts'),
      prefillMs,
      decodeMs,
      totalRunMs: prefillMs + decodeMs,
      decodeMsPerTokenP50: percentile(decodeMsSamples, 0.50),
      decodeMsPerTokenP95: percentile(decodeMsSamples, 0.95),
      decodeMsPerTokenP99: percentile(decodeMsSamples, 0.99),
    },
    environment: {
      host: {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
        osRelease: typeof os.release === 'function' ? os.release() : null,
        cpuModel: cpuInfo || (Array.isArray(os.cpus()) && os.cpus()[0] ? os.cpus()[0].model : null),
      },
      gpu: {
        api: 'vulkan',
        backend: 'vulkan',
        vendor: inferGpuVendor(gpuDescription),
        architecture: null,
        device: gpuDescription,
        description: gpuDescription,
        hasF16: null,
        hasSubgroups: null,
        hasTimestampQuery: null,
      },
      runtime: {
        library: 'llama.cpp',
        version: runtimeVersion,
        surface: 'native-vulkan',
        device: prefill.backends || decode.backends || 'Vulkan',
        dtype: prefill.model_type || decode.model_type || null,
        requestedDtype: null,
        executionProviderMode: 'vulkan',
        cacheMode: null,
        loadMode: 'local-gguf',
      },
    },
    metadata: {
      runner: {
        command,
        runs: options.runs,
        gpuLayers: options.gpuLayers,
        batchSize: options.batchSize,
        ubatchSize: options.ubatchSize,
        threads: options.threads,
        cacheTypeK: options.cacheTypeK,
        cacheTypeV: options.cacheTypeV,
        flashAttn: options.flashAttn,
        mmap: options.mmap,
        noWarmup: options.noWarmup,
      },
      prefill,
      decode,
    },
  };
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  if (flags.help) {
    console.log(usage());
    return;
  }

  const workload = resolveWorkload(flags);
  const options = {
    ...workload,
    modelId: flags['model-id'] || null,
    modelPath: ensureFile(flags.model, '--model'),
    llamaBench: ensureExecutable(flags['llama-bench'] || process.env.LLAMACPP_BENCH, '--llama-bench or LLAMACPP_BENCH'),
    runs: parsePositiveInteger(flags.runs, '--runs', 5),
    gpuLayers: parsePositiveInteger(flags['gpu-layers'], '--gpu-layers', 99),
    batchSize: parsePositiveInteger(flags['batch-size'], '--batch-size', 2048),
    ubatchSize: parsePositiveInteger(flags['ubatch-size'], '--ubatch-size', 512),
    threads: flags.threads == null ? null : parsePositiveInteger(flags.threads, '--threads'),
    cacheTypeK: flags['cache-type-k'] || 'f16',
    cacheTypeV: flags['cache-type-v'] || 'f16',
    flashAttn: parseZeroOne(flags['flash-attn'], '--flash-attn', '0'),
    mmap: parseZeroOne(flags.mmap, '--mmap', '1'),
    noWarmup: flags['no-warmup'] === true,
  };

  const command = buildLlamaBenchCommand(options);
  const rows = runLlamaBench(command);
  const output = buildOutput(options, rows, command);
  console.log(JSON.stringify(output, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
