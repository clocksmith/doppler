// The reading half of the 0.2.0 rule: "Legacy spellings are read throughout 0.x." Every
// value converted by the emitted-value audit (tests/fixtures/emitted-enum-values.json)
// is still accepted in its snake_case spelling wherever Doppler takes it as input, and is
// canonicalized to kebab-case on read. CPU only; no WebGPU device is opened.
// tests/integration/legacy-enum-reads.test.js covers the earlier manifest, runtime-config,
// provider, pipeline and scheduler reads.
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  LEGACY_ENUM_VALUE_ALIASES,
  LEGACY_ENUM_VALUE_FIELDS,
  canonicalizeLegacyEnumValue,
  canonicalizeLegacyEnumValues,
} from '../../src/config/legacy-enum-values.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { mergeRuntimeValues } from '../../src/config/runtime-merge.js';
import { assertTrainingBackend } from '../../src/training.js';
import { loadTrainingWorkloadCapsule } from '../../src/experimental/training/workloads.js';
import { validateVerifierGuidedArtifact } from '../../src/experimental/training/wgsl-repair.js';
import { resolveUlScheduledLambda } from '../../src/experimental/training/ul_schedule.js';
import { evaluateAgentHeldoutRows } from '../../src/experimental/training/operator-agent-eval.js';
import { parseDiffusionGemmaConfig } from '../../src/inference/pipelines/diffusion-gemma/config.js';
import { normalizeToolingCommandRequest } from '../../src/tooling/command-api.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { registerPipeline } from '../../src/inference/pipelines/registry.js';
import { resolveModelSource } from '../../src/client/runtime/model-source.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests/fixtures/emitted-enum-values.json'), 'utf8'));
const KEBAB = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

describe('converted outcome values: shared config reader', () => {
  it('accepts every converted legacy spelling under every canonicalized field and returns kebab-case', () => {
    const aliased = fixture.converted.filter((entry) => entry.readAlias);
    assert.ok(aliased.length > 300, `expected the full converted set, got ${aliased.length}`);
    for (const { legacy, kebab } of aliased) {
      assert.equal(canonicalizeLegacyEnumValue(legacy), kebab);
      for (const field of LEGACY_ENUM_VALUE_FIELDS) {
        const input = { outer: [{ [field]: legacy }, { [field]: [legacy, kebab] }] };
        const output = canonicalizeLegacyEnumValues(input);
        assert.equal(output.outer[0][field], kebab, `${field}: ${legacy}`);
        assert.deepEqual(output.outer[1][field], [kebab, kebab], `${field}[]: ${legacy}`);
        assert.equal(input.outer[0][field], legacy, 'input must not be mutated');
      }
    }
    for (const [legacy, current] of Object.entries(LEGACY_ENUM_VALUE_ALIASES)) {
      assert.match(current, KEBAB, legacy);
    }
  });

  it('runtime config overrides and merges read a converted legacy value and hold kebab-case', () => {
    const merged = mergeRuntimeValues(
      { shared: { tooling: { diagnostics: 'on-failure' } } },
      { shared: { tooling: { diagnostics: 'on_failure' } }, loading: { decoderCacheMode: 'encoder_kv_readonly_canvas_concat' } }
    );
    assert.equal(merged.shared.tooling.diagnostics, 'on-failure');
    assert.equal(merged.loading.decoderCacheMode, 'encoder-kv-readonly-canvas-concat');
    const resolved = resolveRuntimeConfig({ shared: { tooling: { diagnostics: 'on_failure' } } });
    assert.equal(resolved.shared.tooling.diagnostics, 'on-failure');
  });
});

describe('manifest objects written before 0.2.0', () => {
  const legacyManifest = () => JSON.parse(fs.readFileSync(
    path.join(ROOT, 'models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json'), 'utf8'
  ));

  it('createPipeline canonicalizes a manifest object, not only manifest text', async () => {
    const seen = [];
    registerPipeline('legacy-read-probe', async (manifest) => { seen.push(manifest); return { manifest }; });
    const input = { ...legacyManifest(), modelType: 'legacy-read-probe' };
    assert.equal(input.inference.session.perLayerInputs.gpuUpload.mode, 'per_step_slices');
    await createPipeline(input, {});
    assert.equal(seen[0].inference.session.perLayerInputs.gpuUpload.mode, 'per-step-slices');
    assert.equal(seen[0].inference.layerPattern.type, 'every-n');
    assert.equal(input.inference.session.perLayerInputs.gpuUpload.mode, 'per_step_slices', 'input must not be mutated');
  });

  it('an inline { manifest } model source is read like manifest text', async () => {
    const resolved = await resolveModelSource({ manifest: legacyManifest(), baseUrl: 'file:///tmp/legacy-model/' });
    assert.equal(resolved.manifest.inference.session.perLayerInputs.gpuUpload.mode, 'per-step-slices');
    assert.equal(resolved.manifest.inference.execution.policies.dtypeTransition, 'require-cast-step');
  });
});

describe('converted outcome values: readers that take them as input', () => {
  it('command requests read the 0.1.x sequence alphabet and abort spellings', () => {
    const request = (sequenceAlphabet, sequenceQualificationAbort) => normalizeToolingCommandRequest({
      command: 'verify',
      workload: 'inference',
      modelId: 'esm2-t12-35m-ur50d-f32-af32',
      inferenceInput: { sequence: 'MKTIIALSYIFCLVFADYKDDDDK', sequenceAlphabet, sequenceQualificationAbort },
    }).inferenceInput;
    assert.deepEqual(request('amino_acid', 'after_start'), request('amino-acid', 'after-start'));
    assert.equal(request('amino_acid', 'before_execution').sequenceQualificationAbort, 'before-execution');
    assert.equal(request('amino_acid', 'after_start').sequenceAlphabet, 'amino-acid');
  });

  it('training reads the legacy backend, workload and artifact spellings', async () => {
    const loaded = await loadTrainingWorkloadCapsule(
      path.join(ROOT, 'src/experimental/training/workload-capsules/lora-toy-tiny.json')
    );
    // The checked-in capsule is sha256-pinned and still spells the dataset format in snake_case.
    assert.match(loaded.raw, /"toy_linear_classification_jsonl"/);
    assert.equal(loaded.workload.pipeline?.datasetFormat ?? loaded.workload.lora?.datasetFormat ?? 'toy-linear-classification-jsonl',
      'toy-linear-classification-jsonl');
    assert.doesNotMatch(JSON.stringify(loaded.workload), /toy_linear_classification_jsonl/);
    for (const backend of ['webgpu_native', 'webgpu-native']) {
      let message = '';
      try { assertTrainingBackend(loaded.workload, backend); } catch (error) { message = error.message; }
      assert.doesNotMatch(message, /training-backend-invalid/, `${backend} must be accepted`);
      if (message) assert.match(message, /^training-backend-not-supported: backend=webgpu-native /);
    }

    for (const artifactType of ['training_reward_vector', 'training-reward-vector']) {
      assert.throws(
        () => validateVerifierGuidedArtifact({ artifactType, schemaVersion: 1 }),
        (error) => !/Unsupported verifier-guided artifact type/.test(error.message),
        `${artifactType} must reach its validator`
      );
    }

    const schedule = (type) => resolveUlScheduledLambda({
      lambda0: 5,
      noiseSchedule: { type, steps: 5, minLogSNR: -2, maxLogSNR: 4 },
    }, 2);
    assert.equal(schedule('log_snr_cosine'), schedule('log-snr-cosine'));
    assert.equal(schedule('log_snr_linear'), schedule('log-snr-linear'));
    assert.notEqual(schedule('log-snr-cosine'), schedule('log-snr-linear'));
  });

  it('agent evaluation reads legacy category spellings from datasets and policies', () => {
    const rows = [{ id: 'row-1', prompt: 'p', expected: 'x', agentEval: { categories: ['patch_applies'] } }];
    const candidates = [{ id: 'row-1', output: 'x' }];
    for (const category of ['patch_applies', 'patch-applies']) {
      const result = evaluateAgentHeldoutRows(rows, candidates, { policy: { categories: [category] } });
      assert.deepEqual(Object.keys(result.categorySummary), ['patch-applies'], category);
      assert.deepEqual(result.requiredCategories, ['patch-applies']);
    }
  });

  it('diffusion-gemma manifests written before 0.2.0 keep loading', () => {
    const conversion = JSON.parse(fs.readFileSync(
      path.join(ROOT, 'src/config/conversion/diffusiongemma/diffusiongemma-26b-a4b-it-q4k-ehf16-af16.json'),
      'utf8'
    ));
    const legacyContract = findKey(conversion, 'diffusionGemma');
    assert.equal(legacyContract.decoderCacheMode, 'encoder_kv_readonly_canvas_concat');
    let parsed = null;
    try {
      parsed = parseDiffusionGemmaConfig({
        modelId: 'legacy-diffusion-gemma',
        modelType: 'diffusion_gemma',
        architecture: { vocabSize: 262144 },
        inference: { diffusionGemma: legacyContract },
      });
    } catch (error) {
      assert.doesNotMatch(error.message, /decoderCacheMode/, 'the legacy decoder cache mode must be read');
    }
    if (parsed) assert.equal(parsed.decoderCacheMode, 'encoder-kv-readonly-canvas-concat');
  });
});

function findKey(value, key) {
  if (!value || typeof value !== 'object') return null;
  if (Object.prototype.hasOwnProperty.call(value, key)) return value[key];
  for (const entry of Object.values(value)) {
    const found = findKey(entry, key);
    if (found) return found;
  }
  return null;
}
