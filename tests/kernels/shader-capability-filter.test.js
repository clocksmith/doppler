import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { filterShaderForCapabilities, shaderEnablesSubgroups } from '../../src/gpu/kernels/shader-capability-filter.js';

const kernelDir = new URL('../../src/gpu/kernels/', import.meta.url);
const read = (name) => fs.readFileSync(new URL(name, kernelDir), 'utf8');
const registry = JSON.parse(fs.readFileSync(new URL('../../src/config/kernels/registry.json', import.meta.url), 'utf8'));

test('a device with subgroups receives every shader unchanged', () => {
  const source = read('rmsnorm.wgsl');
  assert.equal(filterShaderForCapabilities(source, { hasSubgroups: true }).source, source);
});

test('rmsnorm without subgroups keeps its plain entry points and loses the directive', () => {
  const result = filterShaderForCapabilities(read('rmsnorm.wgsl'), { hasSubgroups: false, label: 'rmsnorm.wgsl' });
  assert.deepEqual(result.removedFunctions.sort(), ['main_small_subgroup', 'main_subgroup']);
  assert.equal(shaderEnablesSubgroups(result.source), false);
  for (const entry of ['main', 'main_small', 'main_cached']) assert.match(result.source, new RegExp(`fn ${entry}\\(`));
  assert.doesNotMatch(result.source.replace(/\/\/[^\n]*/g, ''), /subgroupAdd|subgroup_invocation_id/);
});

test('a shader with no plain entry point reports the missing feature', () => {
  const source = 'enable subgroups;\n@compute @workgroup_size(1)\nfn main(@builtin(subgroup_size) n: u32) { _ = subgroupAdd(1.0); }\n';
  assert.throws(() => filterShaderForCapabilities(source, { hasSubgroups: false, label: 'only_subgroup.wgsl' }),
    /only_subgroup\.wgsl requires the WebGPU "subgroups" feature/);
});

// The registry is the contract: an entry point that survives without subgroups
// is exactly one whose variant can run on a device that lacks them.
test('every registry variant that declares no subgroup requirement survives the filter', () => {
  for (const [operation, definition] of Object.entries(registry.operations)) {
    for (const [variant, config] of Object.entries(definition.variants ?? {})) {
      if (!config.wgsl || (config.requires ?? []).includes('subgroups')) continue;
      let source;
      try { source = read(config.wgsl); } catch { continue; }
      if (!shaderEnablesSubgroups(source)) continue;
      const result = filterShaderForCapabilities(source, { hasSubgroups: false, label: config.wgsl });
      assert.ok(!result.removedFunctions.includes(config.entryPoint), `${operation}/${variant} lost ${config.entryPoint}`);
      assert.match(result.source, new RegExp(`fn ${config.entryPoint}\\(`), `${operation}/${variant}`);
    }
  }
});
