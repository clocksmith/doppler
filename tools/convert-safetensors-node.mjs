#!/usr/bin/env node

import fs from 'node:fs/promises';
import { runNodeCommand } from '../src/tooling/node-command-runner.js';

function parseArgs(argv) {
  const out = {
    inputDir: null,
    outputDir: null,
    modelId: null,
    converterConfigPath: null,
  };
  const positional = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model-id') {
      out.modelId = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--converter-config') {
      out.converterConfigPath = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    positional.push(arg);
  }
  out.inputDir = positional[0] ?? null;
  out.outputDir = positional[1] ?? null;
  return out;
}

async function readJsonFile(filePath) {
  if (!filePath) return null;
  const raw = await fs.readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('--converter-config must point to a JSON object.');
  }
  return parsed;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.inputDir || !args.outputDir) {
    console.error(
      'Usage: node tools/convert-safetensors-node.mjs <inputDir> <outputDir> [--model-id <id>] [--converter-config <path.json>]'
    );
    process.exit(2);
  }
  const converterConfig = await readJsonFile(args.converterConfigPath);

  const response = await runNodeCommand(
    {
      command: 'convert',
      inputDir: args.inputDir,
      outputDir: args.outputDir,
      modelId: args.modelId,
      convertPayload: converterConfig ? { converterConfig } : null,
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
