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
    const nextValue = () => {
      const value = argv[i + 1];
      if (value == null || String(value).startsWith('--')) {
        throw new Error(`Missing value for ${arg}.`);
      }
      i += 1;
      return value;
    };
    if (arg === '--output-dir') {
      out.outputDir = nextValue();
      continue;
    }
    if (arg === '--config') {
      out.configPath = nextValue();
      continue;
    }
    if (arg === '--converter-config') {
      throw new Error('--converter-config has been removed. Use --config <path.json>.');
    }
    if (arg === '--workers') {
      execution.workers = nextValue();
      continue;
    }
    if (arg === '--worker-policy') {
      execution.workerCountPolicy = nextValue();
      continue;
    }
    if (arg === '--row-chunk-rows') {
      execution.rowChunkRows = nextValue();
      continue;
    }
    if (arg === '--row-chunk-min-tensor-bytes') {
      execution.rowChunkMinTensorBytes = nextValue();
      continue;
    }
    if (arg === '--max-in-flight-jobs') {
      execution.maxInFlightJobs = nextValue();
      continue;
    }
    if (arg === '--use-gpu-cast') {
      execution.useGpuCast = true;
      continue;
    }
    if (arg === '--gpu-cast-min-tensor-bytes') {
      execution.gpuCastMinTensorBytes = nextValue();
      continue;
    }
    if (arg.startsWith('--')) {
      throw new Error(`Unknown flag: ${arg}`);
    }
    positional.push(arg);
  }
  if (positional.length > 1) {
    throw new Error(`Unexpected positional arguments: ${positional.slice(1).join(', ')}`);
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
  const useGpuCast = rawExecution.useGpuCast === true;
  const gpuCastMinTensorBytes = parseOptionalPositiveInteger(
    rawExecution.gpuCastMinTensorBytes,
    '--gpu-cast-min-tensor-bytes'
  );
  if (
    workers == null
    && workerCountPolicy == null
    && rowChunkRows == null
    && rowChunkMinTensorBytes == null
    && maxInFlightJobs == null
    && !useGpuCast
    && gpuCastMinTensorBytes == null
  ) {
    return null;
  }
  return {
    ...(workers != null ? { workers } : {}),
    ...(workerCountPolicy != null ? { workerCountPolicy } : {}),
    ...(rowChunkRows != null ? { rowChunkRows } : {}),
    ...(rowChunkMinTensorBytes != null ? { rowChunkMinTensorBytes } : {}),
    ...(maxInFlightJobs != null ? { maxInFlightJobs } : {}),
    ...(useGpuCast ? { useGpuCast: true } : {}),
    ...(gpuCastMinTensorBytes != null ? { gpuCastMinTensorBytes } : {}),
  };
}

function printConvertContractSummary(result) {
  const artifact = result?.executionContractArtifact;
  if (!artifact || typeof artifact !== 'object') return;
  const checks = Array.isArray(artifact.checks) ? artifact.checks : [];
  const passedChecks = checks.filter((entry) => entry?.ok === true).length;
  const layout = artifact.session?.layout ?? 'n/a';
  console.log(
    `[contract] status=${artifact.ok === true ? 'pass' : 'fail'} ` +
    `checks=${checks.length > 0 ? `${passedChecks}/${checks.length}` : 'n/a'} layout=${layout}`
  );
  if (artifact.ok !== true && Array.isArray(artifact.errors)) {
    for (const error of artifact.errors.slice(0, 3)) {
      console.log(`[contract] error=${String(error)}`);
    }
  }
  const graphArtifact = result?.executionV0GraphContractArtifact;
  if (graphArtifact && typeof graphArtifact === 'object') {
    const checks = Array.isArray(graphArtifact.checks) ? graphArtifact.checks : [];
    const passedChecks = checks.filter((entry) => entry?.ok === true).length;
    console.log(
      `[graph] status=${graphArtifact.ok === true ? 'pass' : 'fail'} ` +
      `checks=${checks.length > 0 ? `${passedChecks}/${checks.length}` : 'n/a'}`
    );
  }
  for (const [label, extraArtifact] of [
    ['layer-pattern', result?.layerPatternContractArtifact],
    ['required-inference', result?.requiredInferenceFieldsArtifact],
  ]) {
    if (!extraArtifact || typeof extraArtifact !== 'object') continue;
    const checks = Array.isArray(extraArtifact.checks) ? extraArtifact.checks : [];
    const passedChecks = checks.filter((entry) => entry?.ok === true).length;
    console.log(
      `[${label}] status=${extraArtifact.ok === true ? 'pass' : 'fail'} ` +
      `checks=${checks.length > 0 ? `${passedChecks}/${checks.length}` : 'n/a'}`
    );
  }
}

function printConvertReportSummary(result) {
  const reportInfo = result?.reportInfo;
  if (!reportInfo || typeof reportInfo !== 'object') return;
  if (typeof reportInfo.path !== 'string' || reportInfo.path.length === 0) return;
  console.log(`[report] ${reportInfo.path}`);
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
      'Usage: node tools/convert-safetensors-node.js <inputPath> --config <path.json> [--output-dir <path>] [--workers <n>] [--worker-policy <cap|error>] [--row-chunk-rows <n>] [--row-chunk-min-tensor-bytes <n>] [--max-in-flight-jobs <n>] [--use-gpu-cast] [--gpu-cast-min-tensor-bytes <n>]'
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
  printConvertContractSummary(result);
  printConvertReportSummary(result);
}

main().catch((err) => {
  console.error(`[error] ${err?.stack || err?.message || String(err)}`);
  process.exit(1);
});
