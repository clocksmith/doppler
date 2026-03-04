import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { runNodeCommand } = await import('../../src/tooling/node-command-runner.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');

const MODEL_DIR = process.env.DOPPLER_QWEN35_LINEAR_MODEL_DIR
  ? path.resolve(process.env.DOPPLER_QWEN35_LINEAR_MODEL_DIR)
  : null;
const MANIFEST_PATH = MODEL_DIR ? path.join(MODEL_DIR, 'manifest.json') : null;
const PROMPT = process.env.DOPPLER_QWEN35_LINEAR_PROMPT || 'What color is the sky?';

function toModelUrl(dirPath) {
  const asUrl = pathToFileURL(dirPath).toString();
  return asUrl.endsWith('/') ? asUrl : `${asUrl}/`;
}

function normalizeOutput(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

if (!MODEL_DIR) {
  console.log(
    'qwen-linear-attention-regression.test: skipped (set DOPPLER_QWEN35_LINEAR_MODEL_DIR)'
  );
} else if (!existsSync(MANIFEST_PATH)) {
  console.log(
    `qwen-linear-attention-regression.test: skipped (missing ${MANIFEST_PATH})`
  );
} else {
  let webgpuReady = false;
  try {
    await bootstrapNodeWebGPU();
    webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
  } catch {
    webgpuReady = false;
  }

  if (!webgpuReady) {
    console.log('qwen-linear-attention-regression.test: skipped (no WebGPU runtime)');
  } else {
    const response = await runNodeCommand({
      command: 'debug',
      modelId: 'qwen-linear-attention-regression',
      modelUrl: toModelUrl(MODEL_DIR),
      loadMode: 'http',
      captureOutput: true,
      runtimeConfig: {
        inference: {
          prompt: PROMPT,
          batching: {
            maxTokens: 24,
          },
          sampling: {
            temperature: 0,
            topP: 1,
            topK: 1,
            repetitionPenalty: 1,
            greedyThreshold: 0,
          },
        },
      },
    });

    const result = response?.result ?? null;
    assert.ok(result, 'Debug result is required.');

    const output = normalizeOutput(result.output);
    assert.ok(output.length > 0, 'Qwen linear-attention regression run produced empty output.');
    assert.equal(
      output.includes('\uFFFD'),
      false,
      'Output contains Unicode replacement characters, indicating decode/runtime regression.'
    );

    const metrics = result.metrics ?? {};
    const generated = Number(metrics.tokensGenerated ?? 0);
    const decodeTps = Number(metrics.decodeTokensPerSec ?? 0);
    assert.ok(generated > 0, 'Qwen linear-attention regression run generated zero tokens.');
    assert.ok(Number.isFinite(decodeTps) && decodeTps > 0, 'Qwen linear-attention decode throughput must be > 0.');
  }
}

console.log('qwen-linear-attention-regression.test: ok');
