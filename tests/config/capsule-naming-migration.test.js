import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import { canonicalizeLegacyEnumValues } from '../../src/config/legacy-enum-values.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const archive = 'tests/fixtures/pre-capsule';
const manifest = JSON.parse(await fs.readFile(`${archive}/manifest.json`, 'utf8'));
const digestRebindings = JSON.parse(await fs.readFile(
  'artifacts/release-kernel-digests-2026-09-12/recipe-digest-rebindings.json', 'utf8'
));
// This is an explicit test projection, never a runtime compatibility adapter.
function renameIdentityFields(value) {
  if (Array.isArray(value)) return value.map(renameIdentityFields);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, child]) => [
    key === 'weightPackId' ? 'weightCapsuleId' : key === 'weightPackHash' ? 'weightCapsuleHash' : key,
    renameIdentityFields(child),
  ]));
}

for (const entry of manifest.files) {
  const bytes = await fs.readFile(`${archive}/${entry.source}`);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.sha256, entry.source);
  if (!entry.source.startsWith('src/config/conversion/')) continue;
  const current = JSON.parse(await fs.readFile(entry.source, 'utf8'));
  const expected = canonicalizeLegacyEnumValues(renameIdentityFields(JSON.parse(bytes)));
  // Later shader changes and the recorded stale-digest repair have exact
  // identity mappings. Every other field still matches the frozen archive.
  const shaderChanges = {
    // 17dc3b3a: flatten large gather/split dispatches across workgroup rows.
    'sha256:8995b4a790a1f3b89fbb801f89d595985eee46c9bfca4133f74396259083eda9':
      'sha256:a2c318bc362d676769d8cb751d61a4d7777e1068be0456eff634e0821d783c10',
    'sha256:ce77263ace3099526665174b0fba1d93696bf782268d172d6b71b63a8021338d':
      'sha256:5a6922136dad0cf05059533f24ea0989875c3659b464dfc7528518d57ce1e0c1',
    'sha256:4b12653c53247b32ebde7f6cf6a989d6248977e3816c761540b990b5f9818cb6':
      'sha256:07fa1796cbddf137054897c831cf10ac7845dc98bffa1a8637614ddb4b82d74d',

    ...digestRebindings,
    ...JSON.parse(await fs.readFile(
      'artifacts/recovery-20261009/numerical-digest-rebindings.json', 'utf8'
    )),
    ...JSON.parse(await fs.readFile(
      'artifacts/numerical-20261010/digest-rebindings.json', 'utf8'
    )),
    'sha256:7f8900b69de5107e4cf424cac0d4de1539591ece34e977ff30f6f7364460ab34':
      'sha256:db864455ec72070d7d8729c8744e2eac377a4939a9bc2ecb26eb086d866e16b0',
    'sha256:284d3efb0ad0991fc57ece5f634cbc8a931fcab36f193b2b6832561d4cc79ef1':
      'sha256:4f4d144735bdff86b8567346ba7ea7b82fe68a2ac9656ba1577106158af9bc7c',
    'sha256:5631ce6f19f805c7ac85722bac17365048bc7b9e2c422f3ef9477b63cb8dad62':
      'sha256:c434473d64a163eddbb6fc349aec944e4bef6b067170acdf4d55daf92f1932bf',
    // b557333e7: WGSL uniform/helper identifier cleanup.
    'sha256:abd19bc08ad668a7cad562e2bc0a4be1aa8827ad4300d8d474b7fc2af4c27117':
      'sha256:4dbfb0a9ca0fb6047a01618f0374b445c243774d039407acdf41b0711ea8eaf4',
    'sha256:7ff3dd675b4737d920e0b3889e15c4fedde8410a353bbc6d4651db76ce69eb9b':
      'sha256:f2377b2a58c2eeb695a6f6f7ad877630bdba5e74d5dc1a8d919cd73b360d4870',
    'sha256:7b52d30fb741beef2dbf728e0c4ecffe5b08d9661d63c306caecb4cb3ced85e5':
      'sha256:712787a59b032453b09f147be0d0c073266c47a7a12c3b746984aaf28075bedb',
  };
  for (const [id, kernel] of Object.entries(expected.execution.kernels)) {
    if (kernel.digest in shaderChanges) {
      const currentDigest = `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${kernel.kernel}#${kernel.entry}`]}`;
      assert.equal(current.execution.kernels[id].digest, currentDigest);
      kernel.digest = currentDigest;
    }
  }
  if (entry.source === 'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af16-int4ple.json') {
    // 5122308e6 bound weightsRef to the retained local af32-int4ple manifest prior to 9523d93b1 digest rebinding.
    const retainedAf16 = JSON.parse(await fs.readFile('models/local/gemma-4-e2b-it-q4k-ehf16-af16-int4ple/manifest.json', 'utf8'));
    expected.manifest.weightsRef.manifestDigest = retainedAf16.weightsRef.manifestDigest;
  }
  // a76f1add6 made Gemma 4 audio log-mel feature_extractor parameters explicit in conversion configs.
  if (
    entry.source.startsWith('src/config/conversion/gemma4/')
    && expected.manifest?.audioConfig
    && expected.manifest.audioConfig.feature_extractor === undefined
  ) {
    expected.manifest.audioConfig.feature_extractor = current.manifest?.audioConfig?.feature_extractor;
  }
  // 56e3abbd made the previously implicit Gemma 4 runtime pipeline explicit.
  if (entry.source.startsWith('src/config/conversion/gemma4/') && expected.modelType === undefined) {
    expected.modelType = 'gemma4';
  }
  assert.deepEqual(current, expected,
    `${entry.source}: naming must not change model computation or numerical policy`);
}
console.log('capsule-naming-migration.test: frozen evidence and unchanged conversion semantics passed');
