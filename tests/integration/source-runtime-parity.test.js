import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { runNodeCommand } = await import('../../src/tooling/node-command-runner.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');

const SOURCE_PATH = process.env.DOPPLER_SOURCE_PARITY_SOURCE_PATH || null;
const RDRR_URL = process.env.DOPPLER_SOURCE_PARITY_RDRR_URL || null;
const PROMPT = process.env.DOPPLER_SOURCE_PARITY_PROMPT || 'Summarize the Doppler runtime in one sentence.';
const MAX_TOKEN_DELTA = Number.isFinite(Number(process.env.DOPPLER_SOURCE_PARITY_MAX_TOKEN_DELTA))
  ? Math.max(0, Math.floor(Number(process.env.DOPPLER_SOURCE_PARITY_MAX_TOKEN_DELTA)))
  : 2;
const MAX_PREFIX_MISMATCH_RATE = Number.isFinite(Number(process.env.DOPPLER_SOURCE_PARITY_MAX_PREFIX_MISMATCH_RATE))
  ? Math.max(0, Math.min(1, Number(process.env.DOPPLER_SOURCE_PARITY_MAX_PREFIX_MISMATCH_RATE)))
  : 0.25;

function toModelUrl(value) {
  const raw = String(value || '').trim();
  if (!raw) return raw;
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(raw)) {
    return raw;
  }
  const resolved = path.resolve(raw);
  const asUrl = pathToFileURL(resolved).toString();
  return asUrl.endsWith('/') ? asUrl : `${asUrl}/`;
}

function normalizeOutput(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function prefixMismatchRate(left, right) {
  const compareLength = Math.min(left.length, right.length, 96);
  if (compareLength <= 0) return 1;
  let mismatches = 0;
  for (let i = 0; i < compareLength; i++) {
    if (left[i] !== right[i]) mismatches++;
  }
  return mismatches / compareLength;
}

if (!SOURCE_PATH || !RDRR_URL) {
  console.log(
    'source-runtime-parity.test: skipped (set DOPPLER_SOURCE_PARITY_SOURCE_PATH and DOPPLER_SOURCE_PARITY_RDRR_URL)'
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
    console.log('source-runtime-parity.test: skipped (no WebGPU runtime)');
  } else {
    const runtimeConfig = {
      inference: {
        prompt: PROMPT,
        batching: {
          maxTokens: 32,
        },
        sampling: {
          temperature: 0,
          topP: 1,
          topK: 1,
          repetitionPenalty: 1,
          greedyThreshold: 0,
        },
      },
    };

    const common = {
      command: 'debug',
      modelId: 'source-runtime-parity',
      runtimeConfig,
      captureOutput: true,
    };

    const rdrr = await runNodeCommand({
      ...common,
      modelUrl: toModelUrl(RDRR_URL),
      loadMode: 'http',
    });
    const source = await runNodeCommand({
      ...common,
      modelUrl: SOURCE_PATH,
      loadMode: 'memory',
    });

    const rdrrResult = rdrr?.result ?? null;
    const sourceResult = source?.result ?? null;
    assert.ok(rdrrResult, 'RDRR run result is required.');
    assert.ok(sourceResult, 'Direct-source run result is required.');

    const rdrrOutput = normalizeOutput(rdrrResult.output);
    const sourceOutput = normalizeOutput(sourceResult.output);
    assert.ok(rdrrOutput.length > 0, 'RDRR run did not produce output.');
    assert.ok(sourceOutput.length > 0, 'Direct-source run did not produce output.');

    const rdrrTokens = Number(rdrrResult.metrics?.tokensGenerated || 0);
    const sourceTokens = Number(sourceResult.metrics?.tokensGenerated || 0);
    assert.ok(rdrrTokens > 0, 'RDRR run produced zero tokens.');
    assert.ok(sourceTokens > 0, 'Direct-source run produced zero tokens.');

    const tokenDelta = Math.abs(rdrrTokens - sourceTokens);
    assert.ok(
      tokenDelta <= MAX_TOKEN_DELTA,
      `Token delta too high: ${tokenDelta} (max ${MAX_TOKEN_DELTA}).`
    );

    const mismatchRate = prefixMismatchRate(rdrrOutput, sourceOutput);
    assert.ok(
      mismatchRate <= MAX_PREFIX_MISMATCH_RATE,
      `Output prefix mismatch rate too high: ${mismatchRate.toFixed(3)} (max ${MAX_PREFIX_MISMATCH_RATE}).`
    );
  }
}

console.log('source-runtime-parity.test: ok');
