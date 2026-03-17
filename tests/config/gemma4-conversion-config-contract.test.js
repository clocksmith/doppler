import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { resolvePreset } from '../../src/config/loader.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

const configPath = path.join(
  repoRoot,
  'tools/configs/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json'
);
const config = JSON.parse(await fs.readFile(configPath, 'utf8'));

// === Conversion config loads and has required structure ===

{
  assert.ok(config.output, 'must have output');
  assert.equal(config.output.baseDir, 'models/local');
  assert.ok(config.output.modelBaseId, 'must have modelBaseId');
}

// === Preset reference is valid ===

{
  assert.equal(config.presets.model, 'gemma4');
  const preset = resolvePreset(config.presets.model);
  assert.ok(preset, 'gemma4 preset must resolve');
  assert.equal(preset.id, 'gemma4');
}

// === Quantization config is complete ===

{
  assert.equal(config.quantization.weights, 'q4k');
  assert.equal(config.quantization.embeddings, 'f16');
  assert.equal(config.quantization.lmHead, 'f16');
  assert.equal(config.quantization.computePrecision, 'f32');
  assert.equal(config.quantization.q4kLayout, 'row');
}

// === Inference config references valid kernel path ===

{
  assert.equal(config.inference.schema, 'doppler.execution/v0');
  assert.ok(config.inference.defaultKernelPath, 'must have defaultKernelPath');

  // Kernel path must exist in registry
  const registryPath = path.join(
    repoRoot,
    'src/config/presets/kernel-paths/registry.json'
  );
  const registry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
  const registryIds = new Set(registry.entries.map((e) => e.id));

  assert.ok(
    registryIds.has(config.inference.defaultKernelPath),
    `defaultKernelPath "${config.inference.defaultKernelPath}" must exist in kernel-path registry`
  );
}

// === Kernel path is also reachable from the model preset ===

{
  const preset = resolvePreset('gemma4');
  const presetPaths = preset.inference.kernelPaths?.q4k;
  assert.ok(presetPaths, 'gemma4 preset must define kernelPaths.q4k');

  // The conversion config's kernel path should match one of the preset's Q4K paths
  const presetPathValues = new Set(Object.values(presetPaths));
  assert.ok(
    presetPathValues.has(config.inference.defaultKernelPath),
    `conversion defaultKernelPath "${config.inference.defaultKernelPath}" must be reachable from gemma4 preset kernelPaths.q4k`
  );
}

// === Kernel path inheritance rationale is documented ===

{
  assert.ok(
    config.inference.defaultKernelPathRationale,
    'conversion config must document kernel path inheritance rationale'
  );
  assert.ok(
    config.inference.defaultKernelPathRationale.includes('MoE'),
    'rationale must explain MoE dispatch separation'
  );
}

console.log('gemma4-conversion-config-contract.test: ok');
