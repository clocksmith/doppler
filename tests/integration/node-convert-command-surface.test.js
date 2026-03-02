import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { runNodeCommand } from '../../src/tooling/node-command-runner.js';

await assert.rejects(
  () => runNodeCommand({
    command: 'convert',
    inputDir: '/tmp/in',
  }),
  /tooling command: convert requires convertPayload\.converterConfig\./
);

await assert.rejects(
  () => runNodeCommand({
    command: 'convert',
    convertPayload: {
      converterConfig: {},
    },
  }),
  /tooling command: convert requires inputDir\./
);

await assert.rejects(
  () => runNodeCommand({
    command: 'convert',
    inputDir: '/tmp/in',
    modelId: 'not-allowed',
    convertPayload: {
      converterConfig: {},
    },
  }),
  /tooling command: convert does not accept modelId\./
);

await assert.rejects(
  () => runNodeCommand({
    command: 'convert',
    inputDir: '/tmp/in',
    convertPayload: {
      converterConfig: {},
      execution: {
        workerCountPolicy: 'invalid',
      },
    },
  }),
  /tooling command: convertPayload\.execution\.workerCountPolicy must be "cap" or "error" when provided\./
);

await assert.rejects(
  () => runNodeCommand({
    command: 'convert',
    inputDir: '/tmp/in',
    convertPayload: {
      converterConfig: {},
      execution: {
        rowChunkRows: 0,
      },
    },
  }),
  /tooling command: convertPayload\.execution\.rowChunkRows must be a positive integer when provided\./
);

await assert.rejects(
  () => runNodeCommand({
    command: 'convert',
    inputDir: '/tmp/in',
    convertPayload: {
      converterConfig: [],
    },
  }),
  /tooling command: convertPayload\.converterConfig must be an object when provided\./
);

{
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-convert-file-input-'));
  const inputPath = path.join(tempDir, 'weights.bin');
  writeFileSync(inputPath, 'not-a-gguf', 'utf8');
  try {
    await assert.rejects(
      () => runNodeCommand({
        command: 'convert',
        inputDir: inputPath,
        convertPayload: {
          converterConfig: {
            output: {
              dir: path.join(tempDir, 'out'),
            },
          },
        },
      }),
      /node convert: inputDir must be a directory containing safetensors files or a \.gguf file path\./
    );
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

{
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-convert-multi-gguf-'));
  writeFileSync(path.join(tempDir, 'a.gguf'), '', 'utf8');
  writeFileSync(path.join(tempDir, 'b.gguf'), '', 'utf8');
  try {
    await assert.rejects(
      () => runNodeCommand({
        command: 'convert',
        inputDir: tempDir,
        convertPayload: {
          converterConfig: {
            output: {
              dir: path.join(tempDir, 'out'),
            },
          },
        },
      }),
      /node convert: multiple GGUF files found/
    );
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

{
  const missingInputDir = path.join(tmpdir(), `doppler-missing-convert-input-${Date.now()}`);
  await assert.rejects(
    () => runNodeCommand({
      command: 'convert',
      inputDir: missingInputDir,
      convertPayload: {
        converterConfig: {},
      },
    }),
    /node convert: inputDir does not exist:/
  );
}

{
  const inputDir = mkdtempSync(path.join(tmpdir(), 'doppler-convert-empty-input-'));
  const outputDir = mkdtempSync(path.join(tmpdir(), 'doppler-convert-empty-output-'));
  try {
    await assert.rejects(
      () => runNodeCommand({
        command: 'convert',
        inputDir,
        convertPayload: {
          converterConfig: {
            output: {
              dir: outputDir,
            },
          },
          execution: {
            workers: 100_000,
            workerCountPolicy: 'error',
          },
        },
      }),
      /node convert: requested workers \(\d+\) exceed available CPU parallelism \(\d+\)\./
    );
  } finally {
    rmSync(inputDir, { recursive: true, force: true });
    rmSync(outputDir, { recursive: true, force: true });
  }
}

console.log('node-convert-command-surface.test: ok');
