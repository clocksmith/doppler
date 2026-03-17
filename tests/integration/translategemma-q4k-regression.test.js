import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { runNodeCommand } = await import('../../src/tooling/node-command-runner.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');

const MODEL_DIR = path.resolve('models/local/translategemma-4b-it-q4k-ehf16-af32');
const MANIFEST_PATH = path.join(MODEL_DIR, 'manifest.json');
const PROMPT = 'Translate English to French: Hello world.';

function toModelUrl(dirPath) {
  const asUrl = pathToFileURL(dirPath).toString();
  return asUrl.endsWith('/') ? asUrl : `${asUrl}/`;
}

function normalizeOutput(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

if (!existsSync(MANIFEST_PATH)) {
  console.log(
    `translategemma-q4k-regression.test: skipped (missing ${MANIFEST_PATH})`
  );
} else {
  let webgpuReady = false;
  try {
    await bootstrapNodeWebGPU();
    const adapter = typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu
      ? await globalThis.navigator.gpu.requestAdapter()
      : null;
    webgpuReady = !!adapter;
  } catch {
    webgpuReady = false;
  }

  if (!webgpuReady) {
    console.log('translategemma-q4k-regression.test: skipped (no WebGPU runtime)');
  } else {
    const response = await runNodeCommand({
      command: 'debug',
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      modelUrl: toModelUrl(MODEL_DIR),
      loadMode: 'http',
      captureOutput: true,
      runtimeConfig: {
        shared: {
          tooling: {
            intent: 'investigate',
          },
        },
        inference: {
          prompt: PROMPT,
          batching: {
            maxTokens: 12,
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
    assert.ok(output.length > 0, 'TranslateGemma Q4K regression run produced empty output.');
    assert.equal(
      output.includes('\uFFFD'),
      false,
      'Output contains Unicode replacement characters, indicating decode/runtime regression.'
    );
    assert.match(
      output.toLowerCase(),
      /\bbonjour\b/u,
      'Expected a coherent French translation containing "bonjour".'
    );

    const metrics = result.metrics ?? {};
    const generated = Number(metrics.tokensGenerated ?? 0);
    const decodeTps = Number(metrics.decodeTokensPerSec ?? 0);
    assert.ok(generated > 0, 'TranslateGemma Q4K regression run generated zero tokens.');
    assert.ok(Number.isFinite(decodeTps) && decodeTps > 0, 'TranslateGemma Q4K decode throughput must be > 0.');
  }
}

console.log('translategemma-q4k-regression.test: ok');
