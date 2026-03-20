import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { runNodeCommand } = await import('../../src/tooling/node-command-runner.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');

const MODEL_DIR = path.resolve('models/local/translategemma-4b-it-q4k-ehf16-af32');
const MANIFEST_PATH = path.join(MODEL_DIR, 'manifest.json');
const PROMPT = Object.freeze({
  messages: Object.freeze([
    Object.freeze({
      role: 'user',
      content: Object.freeze([
        Object.freeze({
          type: 'text',
          source_lang_code: 'en',
          target_lang_code: 'fr',
          text: 'Hello world.',
        }),
      ]),
    }),
  ]),
});
const RUNTIME_CONFIG = Object.freeze({
  shared: {
    tooling: {
      intent: 'investigate',
    },
  },
  inference: {
    batching: {
      maxTokens: 16,
    },
    sampling: {
      temperature: 0,
      topP: 1,
      topK: 1,
      repetitionPenalty: 1,
      greedyThreshold: 0,
    },
    compute: {
      activationDtype: 'f32',
    },
    kvcache: {
      kvDtype: 'f16',
    },
    session: {
      compute: {
        defaults: {
          outputDtype: 'f32',
        },
      },
    },
    kernelPath: 'gemma3-q4k-dequant-f32w-f32a-online',
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['config', 'model', 'manifest'],
      onIncompatible: 'remap',
    },
  },
});

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
  let hasSubgroups = false;
  try {
    await bootstrapNodeWebGPU();
    const adapter = typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu
      ? await globalThis.navigator.gpu.requestAdapter()
      : null;
    webgpuReady = !!adapter;
    hasSubgroups = !!adapter?.features?.has?.('subgroups');
  } catch {
    webgpuReady = false;
    hasSubgroups = false;
  }

  if (!webgpuReady) {
    console.log('translategemma-q4k-regression.test: skipped (no WebGPU runtime)');
  } else if (!hasSubgroups) {
    console.log('translategemma-q4k-regression.test: skipped (node WebGPU has no subgroups)');
  } else {
    const response = await runNodeCommand({
      command: 'debug',
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      modelUrl: toModelUrl(MODEL_DIR),
      loadMode: 'http',
      captureOutput: true,
      runtimeConfig: {
        ...RUNTIME_CONFIG,
        inference: {
          ...RUNTIME_CONFIG.inference,
          prompt: PROMPT,
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
