#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { copyFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const INPUT_ROOT = 'reports/training/native-parity/qwen35-9b-peft-gpu-upload-input';
const DEFAULT_OUTPUT = 'reports/training/native-parity/qwen35-9b-peft-adapter-gpu-upload-oracle.json';
const LAYER_TYPES = Object.freeze(
  Array.from({ length: 32 }, (_, index) => (
    (index + 1) % 4 === 0 ? 'full_attention' : 'linear_attention'
  ))
);

function parseArgs(argv) {
  const args = { adapterDir: null, output: DEFAULT_OUTPUT };
  for (let index = 0; index < argv.length; index += 2) {
    const token = argv[index];
    const value = argv[index + 1];
    if (token === '--adapter-dir' && value) args.adapterDir = value;
    else if (token === '--output' && value) args.output = value;
    else throw new Error(`${token} requires a value.`);
  }
  if (!args.adapterDir) throw new Error('--adapter-dir is required.');
  return args;
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const adapterDir = path.resolve(args.adapterDir);
  const sourceWeights = path.join(adapterDir, 'adapter_model.safetensors');
  const sourceConfig = path.join(adapterDir, 'adapter_config.json');
  const stagedRoot = path.resolve(ROOT, INPUT_ROOT);
  const stagedWeights = path.join(stagedRoot, 'adapter_model.safetensors');
  const stagedConfig = path.join(stagedRoot, 'adapter_config.json');
  await mkdir(stagedRoot, { recursive: true });
  await Promise.all([
    copyFile(sourceWeights, stagedWeights),
    copyFile(sourceConfig, stagedConfig),
  ]);
  const [sourceSha256, stagedSha256, sourceConfigSha256, stagedConfigSha256] = await Promise.all([
    sha256File(sourceWeights),
    sha256File(stagedWeights),
    sha256File(sourceConfig),
    sha256File(stagedConfig),
  ]);
  if (sourceSha256 !== stagedSha256 || sourceConfigSha256 !== stagedConfigSha256) {
    throw new Error('Staged Qwen PEFT adapter differs from the source artifact.');
  }
  await runBrowserOracle({
    argv: ['--output', args.output],
    root: ROOT,
    defaultOutput: DEFAULT_OUTPUT,
    modulePath: 'tests/training/browser/qwen-peft-adapter-gpu-upload-oracle.js',
    exportName: 'runQwenPeftAdapterGpuUploadOracle',
    oracleArgs: {
      weightsUrl: `/${INPUT_ROOT}/adapter_model.safetensors`,
      configUrl: `/${INPUT_ROOT}/adapter_config.json`,
      weightsSha256: sourceSha256,
      configSha256: sourceConfigSha256,
      baseModel: 'Qwen/Qwen3.5-9B',
      layerTypes: LAYER_TYPES,
    },
    sourcePaths: {
      importer: 'src/experimental/training/qwen-peft-adapter-import.js',
      exporter: 'src/experimental/training/qwen-peft-adapter-export.js',
      oracle: 'tests/training/browser/qwen-peft-adapter-gpu-upload-oracle.js',
      runner: 'tools/run-qwen-peft-adapter-gpu-upload-oracle.js',
    },
  });
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
