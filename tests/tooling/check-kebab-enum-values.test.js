import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { checkKebabEnumValues } from '../../tools/check-kebab-enum-values.js';
import {
  canonicalizeLegacyEnumValue,
  canonicalizeLegacyEnumValues,
} from '../../src/config/legacy-enum-values.js';
import { parseManifest } from '../../src/formats/rdrr/parsing.js';

const policy = {
  jsonScopes: ['src/config'],
  jsScopes: ['src/config'],
  excludeFiles: [],
  externalValues: { ggufDtypes: ['q4_0'] },
  publishedValues: {},
};

function makeRepo(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kebab-enum-'));
  for (const [file, text] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(root, file)), { recursive: true });
    fs.writeFileSync(path.join(root, file), text);
  }
  execFileSync('git', ['init', '-q'], { cwd: root });
  execFileSync('git', ['add', '.'], { cwd: root });
  return root;
}

{
  const root = makeRepo({
    'src/config/ok.json': JSON.stringify({ gpuUpload: { mode: 'per-step-slices' }, weights: 'q4_0' }),
    'src/config/ok.js': "export const UPLOAD_MODES = Object.freeze(['per-step-slices', 'off']);\n",
  });
  assert.deepEqual(checkKebabEnumValues({ root, policy }), []);
}

{
  const root = makeRepo({
    'src/config/bad.json': JSON.stringify({ gpuUpload: { mode: 'per_step_slices' }, policies: { diagnostics: 'onFailure' } }),
    'src/config/bad.js': "export const UPLOAD_MODES = Object.freeze(['per-step-slices', 'new_mode']);\n",
  });
  const violations = checkKebabEnumValues({ root, policy });
  assert.ok(violations.some((entry) => entry.includes('legacy value "per_step_slices"')), violations.join('\n'));
  assert.ok(violations.some((entry) => entry.includes('diagnostics="onFailure"')), violations.join('\n'));
  assert.ok(violations.some((entry) => entry.includes('UPLOAD_MODES member "new_mode"')), violations.join('\n'));
}

assert.equal(canonicalizeLegacyEnumValue('stage_a'), 'stage-a');
assert.equal(canonicalizeLegacyEnumValue('stage-a'), 'stage-a');
assert.equal(canonicalizeLegacyEnumValue('not_a_known_value'), 'not_a_known_value');

{
  const legacy = {
    inference: {
      layerPattern: { type: 'every_n', period: 6, offset: 5 },
      session: { perLayerInputs: { prefetch: { mode: 'next_token' }, hotCache: { mode: 'prepared_tokens' } } },
    },
    tensors: { a: { storageEncoding: 'offset_binary', kernel: 'next_token' } },
  };
  const canonical = canonicalizeLegacyEnumValues(legacy);
  assert.equal(canonical.inference.layerPattern.type, 'every-n');
  assert.equal(canonical.inference.session.perLayerInputs.prefetch.mode, 'next-token');
  assert.equal(canonical.inference.session.perLayerInputs.hotCache.mode, 'prepared-tokens');
  assert.equal(canonical.tensors.a.storageEncoding, 'offset-binary');
  assert.equal(canonical.tensors.a.kernel, 'next_token', 'non-enum fields stay untouched');
  assert.equal(legacy.inference.layerPattern.type, 'every_n', 'input is not mutated');
}

{
  const manifestPath = 'models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json';
  const text = fs.readFileSync(manifestPath, 'utf8');
  assert.ok(text.includes('"per_step_slices"'), 'published fixture still uses the legacy spelling');
  const manifest = parseManifest(text);
  assert.equal(manifest.inference.session.perLayerInputs.gpuUpload.mode, 'per-step-slices');
  assert.equal(manifest.inference.execution.policies.dtypeTransition, 'require-cast-step');
  assert.equal(manifest.inference.layerPattern.type, 'every-n');
}

console.log('check-kebab-enum-values.test: ok');
