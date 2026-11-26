// The 0.2.0 compatibility reads: inputs written before the kebab-case rename
// still load, and every reader hands kebab-case values to the code behind it.
// CPU only; no WebGPU device is opened.
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  LEGACY_ENUM_VALUE_ALIASES,
  canonicalizeLegacyEnumValue,
  canonicalizeLegacyEnumValues,
} from '../../src/config/legacy-enum-values.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { mergeRuntimeValues } from '../../src/config/runtime-merge.js';
import { normalizeFailureClass, FAILURE_CLASSES } from '../../src/client/failure-taxonomy.js';
import { normalizeModelType } from '../../src/inference/pipelines/registry.js';
import { assertSupportedDiffusionSchedulerType } from '../../src/inference/pipelines/diffusion/scheduler-contract.js';
import { parseManifest } from '../../src/formats/rdrr/parsing.js';

const KEBAB = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

describe('legacy enum alias table', () => {
  it('maps every legacy spelling to a kebab-case value and nothing back to snake_case', () => {
    for (const [legacy, current] of Object.entries(LEGACY_ENUM_VALUE_ALIASES)) {
      assert.match(legacy, /_/, `${legacy} is not a legacy spelling`);
      assert.match(current, KEBAB, `${legacy} maps to non-kebab ${current}`);
      assert.equal(canonicalizeLegacyEnumValue(current), current, `${current} must be stable`);
    }
  });

  it('leaves unknown values, op ids and non-enum fields untouched', () => {
    assert.equal(canonicalizeLegacyEnumValue('not_a_known_value'), 'not_a_known_value');
    const input = { ops: [{ id: 'q_proj', layout: 'contiguous_quantized' }], note: 'contiguous_quantized' };
    const output = canonicalizeLegacyEnumValues(input);
    assert.deepEqual(output, { ops: [{ id: 'q_proj', layout: 'contiguous-quantized' }], note: 'contiguous_quantized' });
    assert.equal(input.ops[0].layout, 'contiguous_quantized', 'input must not be mutated');
    const unchanged = { layout: 'contiguous-quantized' };
    assert.equal(canonicalizeLegacyEnumValues(unchanged), unchanged, 'kebab-case input is returned as-is');
  });
});

describe('runtime config overrides', () => {
  it('reads legacy KV, tiering and diagnostics spellings through resolveRuntimeConfig', () => {
    const resolved = resolveRuntimeConfig({
      inference: { kvcache: { layout: 'contiguous_quantized', tiering: { gating: { mode: 'force_on' } } } },
      shared: { tooling: { diagnostics: 'on_failure' } },
    });
    assert.equal(resolved.inference.kvcache.layout, 'contiguous-quantized');
    assert.equal(resolved.inference.kvcache.tiering.gating.mode, 'force-on');
    assert.equal(resolved.shared.tooling.diagnostics, 'on-failure');
  });

  it('resolves legacy and kebab-case overrides to the same config', () => {
    const legacy = resolveRuntimeConfig({ inference: { kvcache: { layout: 'bdpa_paged' } } });
    const current = resolveRuntimeConfig({ inference: { kvcache: { layout: 'bdpa-paged' } } });
    assert.deepEqual(legacy, current);
  });

  it('still rejects values that were never valid', () => {
    assert.throws(
      () => resolveRuntimeConfig({ shared: { tooling: { diagnostics: 'sometimes' } } }),
      { code: 'DOPPLER_CONFIG_INVALID' }
    );
  });

  it('canonicalizes the override in mergeRuntimeValues without touching the base', () => {
    const base = { kvcache: { layout: 'contiguous' } };
    const merged = mergeRuntimeValues(base, { kvcache: { layout: 'contiguous_quantized' } });
    assert.equal(merged.kvcache.layout, 'contiguous-quantized');
    assert.equal(base.kvcache.layout, 'contiguous');
  });
});

describe('manifests', () => {
  it('parseManifest reads a manifest published with legacy spellings as kebab-case', () => {
    const text = fs.readFileSync('models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json', 'utf8');
    assert.ok(text.includes('"per_step_slices"'), 'fixture must still carry the legacy spelling');
    const manifest = parseManifest(text);
    assert.equal(manifest.inference.session.perLayerInputs.gpuUpload.mode, 'per-step-slices');
    assert.equal(manifest.inference.execution.policies.dtypeTransition, 'require-cast-step');
    assert.equal(manifest.inference.layerPattern.type, 'every-n');
  });
});

describe('provider, pipeline and scheduler readers', () => {
  it('normalizes legacy failure classes for policy.fallbackOn', () => {
    assert.equal(normalizeFailureClass('gpu_oom'), FAILURE_CLASSES.GPU_OOM);
    assert.equal(normalizeFailureClass('gpu-oom'), FAILURE_CLASSES.GPU_OOM);
    assert.equal(FAILURE_CLASSES.GPU_OOM, 'gpu-oom');
  });

  it('normalizes legacy structured-head model types', () => {
    assert.equal(normalizeModelType('structured_json_head'), 'structured-json-head');
    assert.equal(normalizeModelType('Dream_Energy_Head'), 'dream-energy-head');
    assert.equal(normalizeModelType('transformer'), 'transformer');
  });

  it('accepts legacy diffusion scheduler types', () => {
    assert.doesNotThrow(() => assertSupportedDiffusionSchedulerType('flowmatch_euler'));
    assert.doesNotThrow(() => assertSupportedDiffusionSchedulerType('flowmatch-euler'));
  });
});
