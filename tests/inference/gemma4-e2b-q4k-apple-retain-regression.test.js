import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { runNodeCommand } from '../../src/tooling/node-command-runner.js';

const PROMPT = 'The color of the sky is clear because sunlight scatters through the atmosphere. The color of the sky is clear because sunlight scatters through the atmosphere. The color of the sky is clear because sunlight scatters through the atmosphere.';

const MODELS = [
  {
    id: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    description: 'af32-int4ple with retain=true forced on (formerly broken on Apple Metal pre-2026-05-07)',
    runtimeConfigOverrides: {
      inference: {
        compute: { activationDtype: 'f32' },
        session: { retainQ4KMaterialization: true },
      },
    },
    expectedAppleError: /platform unsupported|non-finite decode logits|non-finite logits/i,
  },
  {
    id: 'gemma-4-e2b-it-q4k-ehf16-af16-int4ple',
    description: 'af16-int4ple manifest default must fail closed on Apple Metal',
    runtimeConfigOverrides: {},
    expectedAppleError: /lane mismatch/,
  },
];

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`gemma4-e2b-q4k-apple-retain-regression.test: skipped (${gpu.reason})`);
  process.exit(0);
}

const adapter = getDevice()?.adapterInfo || {};
const isApple = (adapter.vendor || '').toLowerCase().includes('apple') || /metal/i.test(adapter.architecture || '');

destroyDevice();

if (!isApple) {
  console.log(`gemma4-e2b-q4k-apple-retain-regression.test: skipped (vendor=${adapter.vendor || 'unknown'} not apple)`);
  process.exit(0);
}

for (const model of MODELS) {
  const localPath = resolve(`models/local/${model.id}`);
  if (!existsSync(localPath)) {
    console.log(`gemma4-e2b-q4k-apple-retain-regression.test: skipped ${model.id} (no local artifact)`);
    continue;
  }

  let result = null;
  try {
    result = await runNodeCommand({
      command: 'debug',
      workload: 'inference',
      modelId: model.id,
      modelUrl: `file://${localPath}`,
      cacheMode: 'warm',
      runtimeConfig: {
        shared: { tooling: { intent: 'investigate' } },
        inference: {
          ...(model.runtimeConfigOverrides.inference || {}),
          generation: { maxTokens: 24 },
          sampling: { temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, greedyThreshold: 0 },
          prompt: PROMPT,
        },
      },
    });
  } catch (error) {
    if (!model.expectedAppleError) {
      throw error;
    }
    assert.match(
      error?.message ?? '',
      model.expectedAppleError,
      `${model.id}: Apple rejection must explain the lane mismatch`
    );
    console.log(
      `gemma4-e2b-q4k-apple-retain-regression: ${model.id} correctly rejected (${error?.message})`
    );
    continue;
  }

  if (model.expectedAppleError) {
    assert.equal(
      result.ok,
      false,
      `${model.id}: command unexpectedly succeeded; Apple Metal must reject this lane instead of running the intermittent fused-q4k/f16 path`
    );
    assert.match(
      result.error?.message ?? '',
      model.expectedAppleError,
      `${model.id}: Apple rejection must explain the lane mismatch`
    );
    console.log(
      `gemma4-e2b-q4k-apple-retain-regression: ${model.id} correctly rejected (${result.error?.message})`
    );
    continue;
  }

  assert.equal(result.ok, true, `${model.id}: command failed: ${result.error?.message}`);
  const output = result.result?.output ?? '';
  assert.ok(output.length > 0, `${model.id}: no generated output`);

  const printable = [...output].filter(c => /[\x20-\x7E]/.test(c)).length;
  const ratio = printable / output.length;
  assert.ok(ratio > 0.85, `${model.id}: output is largely non-printable (ratio=${ratio.toFixed(2)}) — possible NaN/garbage regression: ${JSON.stringify(output.slice(0, 100))}`);
  assert.ok(!/(.)\1{12,}/.test(output), `${model.id}: output contains a long single-character run — possible degenerate decode regression: ${JSON.stringify(output.slice(0, 100))}`);

  const tokensGenerated = result.result?.metrics?.tokensGenerated ?? 0;
  assert.ok(tokensGenerated >= 16, `${model.id}: only generated ${tokensGenerated} tokens — premature stop suggests numerical failure`);

  console.log(`gemma4-e2b-q4k-apple-retain-regression: ${model.id} ok (${tokensGenerated} tok, sample=${JSON.stringify(output.slice(0, 60))})`);
}

destroyDevice();

console.log('gemma4-e2b-q4k-apple-retain-regression.test: ok');
