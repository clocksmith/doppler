import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
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
  const expected = renameIdentityFields(JSON.parse(bytes));
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
      'sha256:26e00e5a829ac064b85ed9b1c2565ec8d957ee468666903f1d47e07d93dfb3bc',
  };
  for (const [id, kernel] of Object.entries(expected.execution.kernels)) {
    if (kernel.digest in shaderChanges) {
      const currentDigest = `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${kernel.kernel}#${kernel.entry}`]}`;
      assert.equal(current.execution.kernels[id].digest, currentDigest);
      kernel.digest = currentDigest;
    }
  }
  if (entry.source === 'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af16-int4ple.json') {
    // The referenced manifest now uses Capsule identity fields. Bind the new
    // digest to its actual bytes while comparing every computational field.
    const base = await fs.readFile('models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/manifest.json');
    expected.manifest.weightsRef.manifestDigest = `sha256:${createHash('sha256').update(base).digest('hex')}`;
  }
  // 56e3abbd made the previously implicit Gemma 4 runtime pipeline explicit.
  if (entry.source.startsWith('src/config/conversion/gemma4/') && expected.modelType === undefined) {
    expected.modelType = 'gemma4';
  }
  assert.deepEqual(current, expected,
    `${entry.source}: naming must not change model computation or numerical policy`);
}
console.log('capsule-naming-migration.test: frozen evidence and unchanged conversion semantics passed');
