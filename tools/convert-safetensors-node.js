#!/usr/bin/env node

import fs from 'node:fs/promises';
import { runNodeCommand } from '../src/tooling/node-command-runner.js';

function parseArgs(argv) {
  const out = {
    inputDir: null,
    outputDir: null,
    configPath: null,
    execution: null,
  };
  const execution = {};
  const positional = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--output-dir') {
      out.outputDir = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--config' || arg === '--converter-config') {
      out.configPath = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--workers') {
      execution.workers = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--worker-policy') {
      execution.workerCountPolicy = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--row-chunk-rows') {
      execution.rowChunkRows = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--row-chunk-min-tensor-bytes') {
      execution.rowChunkMinTensorBytes = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--max-in-flight-jobs') {
      execution.maxInFlightJobs = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    positional.push(arg);
  }
  out.inputDir = positional[0] ?? null;
  out.execution = Object.keys(execution).length > 0 ? execution : null;
  return out;
}

function parseOptionalPositiveInteger(value, label) {
  if (value == null || value === '') return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function parseWorkerPolicy(value, label) {
  if (value == null || value === '') return null;
  const normalized = String(value).trim().toLowerCase();
  if (normalized !== 'cap' && normalized !== 'error') {
    throw new Error(`${label} must be "cap" or "error".`);
  }
  return normalized;
}

function normalizeExecutionConfig(rawExecution) {
  if (!rawExecution || typeof rawExecution !== 'object') return null;
  const workers = parseOptionalPositiveInteger(rawExecution.workers, '--workers');
  const workerCountPolicy = parseWorkerPolicy(rawExecution.workerCountPolicy, '--worker-policy');
  const rowChunkRows = parseOptionalPositiveInteger(rawExecution.rowChunkRows, '--row-chunk-rows');
  const rowChunkMinTensorBytes = parseOptionalPositiveInteger(
    rawExecution.rowChunkMinTensorBytes,
    '--row-chunk-min-tensor-bytes'
  );
  const maxInFlightJobs = parseOptionalPositiveInteger(
    rawExecution.maxInFlightJobs,
    '--max-in-flight-jobs'
  );
  if (
    workers == null
    && workerCountPolicy == null
    && rowChunkRows == null
    && rowChunkMinTensorBytes == null
    && maxInFlightJobs == null
  ) {
    return null;
  }
  return {
    ...(workers != null ? { workers } : {}),
    ...(workerCountPolicy != null ? { workerCountPolicy } : {}),
    ...(rowChunkRows != null ? { rowChunkRows } : {}),
    ...(rowChunkMinTensorBytes != null ? { rowChunkMinTensorBytes } : {}),
    ...(maxInFlightJobs != null ? { maxInFlightJobs } : {}),
  };
}

async function readJsonFile(filePath) {
  if (!filePath) return null;
  const raw = await fs.readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('--config must point to a JSON object.');
  }
  return parsed;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.inputDir || !args.configPath) {
    console.error(
      'Usage: node tools/convert-safetensors-node.js <inputPath> --config <path.json> [--output-dir <path>] [--workers <n>] [--worker-policy <cap|error>] [--row-chunk-rows <n>] [--row-chunk-min-tensor-bytes <n>] [--max-in-flight-jobs <n>]'
    );
    process.exit(2);
  }
  const converterConfig = await readJsonFile(args.configPath);
  const execution = normalizeExecutionConfig(args.execution);

  const response = await runNodeCommand(
    {
      command: 'convert',
      inputDir: args.inputDir,
      outputDir: args.outputDir,
      convertPayload: converterConfig ? { converterConfig, execution } : null,
    },
    {
      onProgress(progress) {
        if (!progress) return;
        if (Number.isFinite(progress.current) && Number.isFinite(progress.total)) {
          console.log(`[convert] ${progress.current}/${progress.total} ${progress.message ?? ''}`.trim());
          return;
        }
        if (progress.message) {
          console.log(`[convert] ${progress.stage ?? 'progress'}: ${progress.message}`);
        }
      },
    }
  );

  const result = response.result;
  console.log(
    `[done] modelId=${result.manifest?.modelId ?? 'unknown'} preset=${result.presetId} modelType=${result.modelType} shards=${result.shardCount} tensors=${result.tensorCount}`
  );
}

main().catch((err) => {
  console.error(`[error] ${err?.stack || err?.message || String(err)}`);
  process.exit(1);
});
