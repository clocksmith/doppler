// The 0.2.0 compatibility rule: "Legacy spellings are read throughout 0.x. Doppler emits
// only kebab-case. Aliases are removed no earlier than 1.0, and only with at least one minor
// release of notice." This file checks the emitting half: no snake_case value at an outcome
// field in source (driven by tests/fixtures/emitted-enum-values.json and
// tools/list-emitted-enum-values.js), and no snake_case value in the results, errors,
// failure classes and receipts that public entry points return on CPU. No WebGPU device is
// opened; the decode-stats variant runs in the GPU tier
// (tests/integration/emitted-decode-stats-kebab-physical.test.js).
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { classifyEmittedSnakeValues } from '../../tools/list-emitted-enum-values.js';
import { LEGACY_ENUM_VALUE_ALIASES } from '../../src/config/legacy-enum-values.js';
import { createDopplerProvider } from '../../src/client/provider.js';
import { classifyProviderFailure, FAILURE_CLASSES } from '../../src/client/failure-taxonomy.js';
import { activateLoRAFromTrainingOutput } from '../../src/client/runtime/model-manager.js';
import { TRAINING_BACKENDS, assertTrainingBackend } from '../../src/training.js';
import { checkDrift } from '../../src/inference/pipelines/text/policy/drift-policy.js';
import {
  evaluateTranslationArtifactCompetition,
  readTranslationArtifactCompetitionReceipt,
} from '../../src/tooling/translation-artifact-competition.js';
import { verifyTrainerArtifactParityEvidence } from '../../src/experimental/bridge/trainer-artifact-bridge.js';
import { TRAINING_ROW_ORDERS, TRAINING_EVAL_KINDS } from '../../src/experimental/training/workloads.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests/fixtures/emitted-enum-values.json'), 'utf8'));
const policy = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/policies/emitted-enum-values.json'), 'utf8'));
const SNAKE = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?::.*)?$/;
const KEBAB = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const RETAINED = new Set(Object.values(policy.retainedValues).flatMap((entry) => entry.values));
const CONVERTED_LEGACY = new Map(fixture.converted.map((entry) => [entry.legacy, entry.kebab]));

// Keys whose values are identifiers rather than Doppler enum output: model, file and
// tensor names, hashes, ids chosen by the caller, and free text.
const IDENTIFIER_KEYS = new Set([
  'message', 'modelId', 'model', 'logicalModelId', 'contractId', 'bridgeId', 'sourceContractId',
  'checkpointSha256', 'receiptHash', 'artifactIdentitySha256', 'identityReceiptHash', 'handoffSha256',
  'stack', 'prompt', 'outputText', 'name', 'baseUrl', 'provider', 'schema', 'claimBoundary', 'lane', 'id',
]);

function collectSnakeValues(value, trail = '$', out = []) {
  if (typeof value === 'string') {
    const head = value.split(':')[0];
    if (SNAKE.test(value) && !RETAINED.has(head)) out.push(`${trail}=${value}`);
    return out;
  }
  if (Array.isArray(value)) {
    value.forEach((entry, index) => collectSnakeValues(entry, `${trail}[${index}]`, out));
    return out;
  }
  if (value instanceof Error) {
    for (const key of ['code', 'failureClass', 'reason', 'details', 'receipt', 'cause']) {
      if (value[key] !== undefined) collectSnakeValues(value[key], `${trail}.${key}`, out);
    }
    const code = /^([a-z][a-z0-9]*(?:[_-][a-z0-9]+)+):/.exec(value.message ?? '');
    if (code && code[1].includes('_')) out.push(`${trail}.message code=${code[1]}`);
    return out;
  }
  if (value && typeof value === 'object') {
    for (const [key, entry] of Object.entries(value)) {
      if (IDENTIFIER_KEYS.has(key) && typeof entry === 'string') continue;
      collectSnakeValues(entry, `${trail}.${key}`, out);
    }
  }
  return out;
}

function assertKebabOutput(label, value) {
  const found = collectSnakeValues(value);
  assert.deepEqual(found, [], `${label} emitted snake_case values: ${found.join(', ')}`);
}

describe('emitted enum values: source', () => {
  it('has no snake_case value at an outcome field outside the retained identifier categories', () => {
    const { unclassified, retained } = classifyEmittedSnakeValues({ root: ROOT, policy });
    assert.deepEqual(unclassified.map((entry) => `${entry.value} (${entry.fields.join(',')}) ${entry.files.join(' ')}`), []);
    assert.deepEqual(
      retained.map((entry) => entry.value).sort(),
      fixture.retained.map((entry) => entry.value).sort(),
      'retained values changed; regenerate tests/fixtures/emitted-enum-values.json and review each new value'
    );
  });

  it('reads every converted value in its legacy spelling and maps it to the kebab-case value it emits', () => {
    assert.equal(fixture.convertedCount, fixture.converted.length);
    for (const entry of fixture.converted) {
      assert.equal(entry.kebab, entry.legacy.replace(/_/g, '-'));
      assert.match(entry.kebab, KEBAB);
      if (entry.readAlias) {
        assert.equal(LEGACY_ENUM_VALUE_ALIASES[entry.legacy], entry.kebab, `${entry.legacy} must stay readable`);
      }
    }
    for (const [legacy, current] of Object.entries(LEGACY_ENUM_VALUE_ALIASES)) {
      assert.match(current, KEBAB, `${legacy} maps to ${current}`);
    }
  });
});

describe('emitted enum values: public entry points on CPU', () => {
  it('provider failures, policy denials and receipts carry kebab-case failure classes and reasons', async () => {
    for (const value of Object.values(FAILURE_CLASSES)) assert.match(value, KEBAB);
    const oomHandle = {
      loaded: true,
      modelId: 'cpu-probe',
      manifest: null,
      deviceInfo: null,
      async generateText() { throw new Error('WebGPU out of memory during prefill'); },
      async unload() {},
    };
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      choices: [{ message: { content: 'from-fallback' } }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    try {
      const fallback = createDopplerProvider({
        local: { handle: oomHandle },
        policy: { mode: 'prefer-local', fallbackOn: ['gpu_oom'] },
        fallback: { provider: 'openai', model: 'gpt-4o', baseUrl: 'http://127.0.0.1:9' },
        diagnostics: { receipts: true },
      });
      const result = await fallback.generate('hello');
      assert.equal(result.receipt.failure.failureClass, 'gpu-oom');
      assertKebabOutput('provider fallback result', result);

      const denied = createDopplerProvider({
        local: { handle: oomHandle },
        policy: { mode: 'prefer-local', fallbackOn: ['gpu_timeout'] },
        fallback: { provider: 'openai', model: 'gpt-4o', baseUrl: 'http://127.0.0.1:9' },
      });
      const error = await denied.generate('hello').then(() => null, (caught) => caught);
      assert.ok(error instanceof Error);
      assertKebabOutput('provider policy denial', error);

      const localOnly = createDopplerProvider({
        local: { handle: oomHandle },
        policy: { mode: 'local-only' },
      });
      const localError = await localOnly.generate('hello').then(() => null, (caught) => caught);
      assert.ok(localError instanceof Error);
      assertKebabOutput('provider local-only failure', localError);
    } finally {
      globalThis.fetch = originalFetch;
    }
    for (const message of ['device lost', 'out of memory', 'WebGPU is unavailable', 'timeout', 'unsupported feature']) {
      assertKebabOutput(`classifyProviderFailure(${message})`, classifyProviderFailure(new Error(message)));
    }
  });

  it('model-manager and training report kebab-case reasons, backends and error codes', async () => {
    assert.deepEqual(await activateLoRAFromTrainingOutput(null), {
      activated: false, adapterName: null, source: null, reason: 'no-model-loaded',
    });
    assert.deepEqual([...TRAINING_BACKENDS], ['webgpu-native', 'external']);
    for (const list of [TRAINING_ROW_ORDERS, TRAINING_EVAL_KINDS]) assertKebabOutput('training vocabulary', [...list]);
    const error = (() => { try { assertTrainingBackend({}, 'bogus'); } catch (caught) { return caught; } return null; })();
    assert.match(error.message, /^training-backend-invalid: expected one of webgpu-native, external, got bogus\.$/);
    assertKebabOutput('assertTrainingBackend error', error);
  });

  it('drift checks, translation competition receipts and parity verification emit kebab-case', () => {
    assertKebabOutput('checkDrift', checkDrift('not-a-class', 'f32', {}));
    const competitionPolicy = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/policies/translation-artifact-competition.json'), 'utf8'));
    const receipt = evaluateTranslationArtifactCompetition({ policy: competitionPolicy });
    assert.ok(receipt.blockers.length > 0);
    assertKebabOutput('translation competition receipt', receipt);
    const recordedPath = path.join(ROOT, 'reports/translation-artifact-competition');
    if (fs.existsSync(recordedPath)) {
      for (const file of fs.readdirSync(recordedPath).filter((name) => name.endsWith('.json'))) {
        const recorded = JSON.parse(fs.readFileSync(path.join(recordedPath, file), 'utf8'));
        if (recorded?.schema !== receipt.schema) continue;
        assertKebabOutput(`recorded receipt ${file} as read`, readTranslationArtifactCompetitionReceipt(recorded));
      }
    }
    const parityError = (() => {
      try { return verifyTrainerArtifactParityEvidence(null, { ok: true }, {}); } catch (caught) { return caught; }
    })();
    assertKebabOutput('parity verification', parityError);
  });

  it('every converted legacy value is absent from what these entry points returned', () => {
    // Guard against the probe silently accepting a converted spelling under an identifier key.
    for (const legacy of CONVERTED_LEGACY.keys()) assert.ok(!RETAINED.has(legacy), `${legacy} is both converted and retained`);
  });
});
