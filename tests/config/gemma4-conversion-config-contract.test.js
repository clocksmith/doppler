import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

const configPath = path.join(
  repoRoot,
  'src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json'
);
const config = JSON.parse(await fs.readFile(configPath, 'utf8'));

// === Conversion config loads and has required structure ===

{
  assert.ok(config.output, 'must have output');
  assert.equal(config.output.baseDir, 'models/local');
  assert.ok(config.output.modelBaseId, 'must have modelBaseId');
}

// === No legacy family-indirection field (v1 is self-contained) ===

{
  assert.equal(config.presets, undefined, 'v1 config must not have a legacy family-indirection field');
}

// === Quantization config is complete ===

{
  assert.equal(config.quantization.weights, 'q4k');
  assert.equal(config.quantization.embeddings, 'f16');
  assert.equal(config.quantization.lmHead, 'f16');
  assert.equal(config.quantization.computePrecision, 'f32');
  assert.equal(config.quantization.q4kLayout, 'row');
}

// === Inference config is explicit (no defaultKernelPath) ===

{
  assert.ok(config.inference, 'must have inference');
  assert.ok(config.inference.attention, 'must have inference.attention');
  assert.ok(config.inference.normalization, 'must have inference.normalization');
  assert.ok(config.inference.ffn, 'must have inference.ffn');
  assert.ok(config.inference.rope, 'must have inference.rope');
  assert.ok(config.inference.output, 'must have inference.output');
}

// === Execution graph is v1 format ===

{
  assert.ok(config.execution, 'must have execution');
  assert.ok(config.execution.kernels, 'must have execution.kernels');
  assert.ok(Array.isArray(config.execution.decode), 'must have execution.decode array');
  assert.ok(Array.isArray(config.execution.prefill), 'must have execution.prefill array');
  assert.ok(config.execution.policies, 'must have execution.policies');

  const expanded = expandExecutionV1(config.execution);
  assert.ok(expanded.length > 0, 'execution must expand to at least one step');
}

// === Session present ===

{
  assert.ok(config.session, 'must have session');
  assert.ok(config.session.compute?.defaults, 'must have compute defaults');
  assert.equal(config.session.compute.defaults.activationDtype, 'f32');
}

// === Gemma 4 specific: sliding window + YARN scaling ===

{
  assert.equal(config.inference.attention.slidingWindow, 1024);
  assert.equal(config.inference.normalization.rmsNormWeightOffset, false);
  assert.equal(config.inference.chatTemplate.type, 'gemma4');
  assert.equal(config.inference.rope.ropeScalingType, 'yarn');
  assert.equal(config.inference.rope.ropeScalingFactor, 8);
}

console.log('gemma4-conversion-config-contract.test: ok');
