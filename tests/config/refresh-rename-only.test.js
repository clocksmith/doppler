// tools/refresh-converted-manifest.js --rename-only reissues a manifest under a new name without the weights.
import { canonicalizeLegacyEnumValues } from '../../src/config/legacy-enum-values.js';
// A manifest's identity is its hash, so the tool must change the name and the provenance and nothing else,
// and must refuse a config or a name that does not describe the manifest (docs/style/model-naming.md).
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const tool = path.join(root, 'tools', 'refresh-converted-manifest.js');
const earlier = JSON.parse(fs.readFileSync(path.join(root, 'models', 'local', 'qwen-3-5-2b-q4k-ehaf16', 'manifest.json'), 'utf8'));
const newName = 'qwen-3-5-2b-q4k-ef16-af32';
const newConfig = path.join(root, 'src', 'config', 'conversion', 'qwen3', `${newName}.json`);
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'doppler-rename-only-'));

function reissue(manifest, config, name) {
  const directory = fs.mkdtempSync(path.join(work, 'model-'));
  fs.writeFileSync(path.join(directory, 'manifest.json'), JSON.stringify(manifest, null, 2));
  const result = spawnSync(process.execPath, [tool, directory, '--config', config, '--model-id', name, '--rename-only'], { cwd: root, encoding: 'utf8' });
  return { status: result.status, output: `${result.stdout}${result.stderr}`, written: JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json'), 'utf8')) };
}

function differingLeaves(a, b, prefix = '', out = []) {
  if (a && b && typeof a === 'object' && typeof b === 'object' && Array.isArray(a) === Array.isArray(b)) {
    for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) differingLeaves(a[key], b[key], `${prefix}/${key}`, out);
  } else if (a !== b) out.push(prefix);
  return out;
}

try {
  // The committed manifest lists each shard hash twice; the parser keeps one. Start from that normal form.
  const normal = structuredClone(earlier);
  for (const shard of normal.shards) delete shard.sha256;

  const renamed = reissue(normal, newConfig, newName);
  assert.equal(renamed.status, 0, renamed.output);
  assert.equal(renamed.written.modelId, newName);
  // Only the name and the provenance change. How the model is stored and run is carried over untouched.
  // Legacy snake_case enum spellings are rewritten to kebab-case on refresh.
  assert.deepEqual(differingLeaves(canonicalizeLegacyEnumValues(normal), renamed.written).sort(), [
    '/artifactIdentity/conversionConfigDigest',
    '/artifactIdentity/conversionConfigPath',
    '/artifactIdentity/manifestVariantId',
    '/metadata/manifestRefresh/at',
    '/metadata/manifestRefresh/config',
    '/modelId',
  ]);
  for (const key of ['shards', 'tensors', 'integrityExtensions', 'tokenizer', 'quantizationInfo', 'inference', 'architecture']) {
    assert.deepEqual(renamed.written[key], canonicalizeLegacyEnumValues(normal)[key], `${key} changed`);
  }
  assert.equal(renamed.written.artifactIdentity.conversionConfigPath, `src/config/conversion/qwen3/${newName}.json`);
  assert.equal(renamed.written.artifactIdentity.weightCapsuleId, normal.artifactIdentity.weightCapsuleId, 'the weights are the same weights');
  assert.ok(renamed.written.artifactIdentity.manifestVariantId.startsWith(`${newName}-mv-`));

  // A name whose suffix is not the one the manifest's format generates.
  const wrongSuffix = reissue(normal, newConfig, 'qwen-3-5-2b-q4k-ehaf16-v2');
  assert.notEqual(wrongSuffix.status, 0);
  assert.match(wrongSuffix.output, /does not describe its model: it must end with "-q4k-ef16-af32"/);
  assert.equal(wrongSuffix.written.modelId, earlier.modelId, 'a refused reissue writes nothing');

  // A name with a character that is not allowed.
  const badCharacter = reissue(normal, newConfig, 'qwen_3_5_2b-q4k-ef16-af32');
  assert.notEqual(badCharacter.status, 0);
  assert.match(badCharacter.output, /not a valid model name/);

  // A config that claims an F16 head over this manifest's Q4_K_M head would give the model another name and
  // another execution graph. Rename-only refuses it instead of rewriting the manifest to match.
  const drifted = JSON.parse(fs.readFileSync(newConfig, 'utf8'));
  drifted.quantization.lmHead = 'f16';
  const driftedConfig = path.join(work, 'qwen-3-5-2b-q4k-ehf16-af32.json');
  fs.writeFileSync(driftedConfig, JSON.stringify(drifted, null, 2));
  const wrongConfig = reissue(normal, driftedConfig, 'qwen-3-5-2b-q4k-ehf16-af32');
  assert.notEqual(wrongConfig.status, 0);
  assert.match(wrongConfig.output, /the conversion config does not describe this manifest: it generates the format q4k-ehf16-af32, the manifest records q4k-ef16-af32/);

  // Without the weights the tool cannot justify a change to the shard list, so a manifest that is not in
  // normal form is refused and has to be refreshed from the weights.
  const notNormal = reissue(earlier, newConfig, newName);
  assert.notEqual(notNormal.status, 0);
  assert.match(notNormal.output, /--rename-only changed the shard or tensor list/);
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}

console.log('refresh-rename-only.test: ok');
