import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { runNodeCommand } = await import('../../src/tooling/node-command-runner.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');

const MODEL_DIR = path.resolve('models/local/translategemma-4b-it-q4k-ehf16-af32');
const MANIFEST_PATH = path.join(MODEL_DIR, 'manifest.json');
const RUN_TRANSLATEGEMMA_LEGACY_REGRESSION = false;
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
    sampling: {
      temperature: 0,
      topP: 1,
      topK: 1,
      repetitionPenalty: 1,
      greedyThreshold: 0,
    },
    compute: {
      activationDtype: 'f16',
    },
    kvcache: {
      kvDtype: 'f16',
    },
    session: {
      compute: {
        defaults: {
          activationDtype: 'f16',
          mathDtype: 'f16',
          accumDtype: 'f16',
          outputDtype: 'f16',
        },
      },
    },
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
} else if (!RUN_TRANSLATEGEMMA_LEGACY_REGRESSION) {
  console.log(
    'translategemma-q4k-regression.test: skipped (legacy TranslateGemma Q4K is outside the current Gemma 4/Qwen 3.5 release focus)'
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
      inferenceInput: {
        prompt: PROMPT,
        maxTokens: 16,
      },
      runtimeConfig: RUNTIME_CONFIG,
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

    const metrics = result.metrics ?? {};
    assert.equal(
      metrics.prompt,
      'en -> fr: Hello world.',
      'Structured TranslateGemma prompt must reach the node command surface unchanged.'
    );
    const generated = Number(metrics.tokensGenerated ?? 0);
    const decodeTps = Number(metrics.decodeTokensPerSec ?? 0);
    assert.ok(generated > 0, 'TranslateGemma Q4K regression run generated zero tokens.');
    assert.ok(Number.isFinite(decodeTps) && decodeTps > 0, 'TranslateGemma Q4K decode throughput must be > 0.');
  }
}

console.log('translategemma-q4k-regression.test: ok');
