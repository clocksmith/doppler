#!/usr/bin/env node

import { runNodeCommand } from '../src/tooling/node-command-runner.js';

function parseArgs(argv) {
  const out = {
    inputDir: null,
    outputDir: null,
    modelId: null,
  };
  const positional = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model-id') {
      out.modelId = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    positional.push(arg);
  }
  out.inputDir = positional[0] ?? null;
  out.outputDir = positional[1] ?? null;
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.inputDir || !args.outputDir) {
    console.error('Usage: node tools/convert-safetensors-node.mjs <inputDir> <outputDir> [--model-id <id>]');
    process.exit(2);
  }

  const response = await runNodeCommand(
    {
      command: 'convert',
      inputDir: args.inputDir,
      outputDir: args.outputDir,
      modelId: args.modelId,
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
