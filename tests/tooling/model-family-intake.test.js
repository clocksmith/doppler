import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { findNewModelFamilies, validateModelFamilyAuthorization, checkModelFamilyIntake } from '../../tools/check-model-family-intake.js';

assert.deepEqual(findNewModelFamilies(['gemma3', 'qwen3'], ['gemma3']), ['qwen3']);
const content = new Map([
  ['src/config/conversion/new-family/model.json', '{"output":{"modelBaseId":"test-model"}}\n'],
  ['tests/converter/new-family.test.js', 'import assert from "node:assert/strict"; assert.equal(1, 1);\n'],
  ['docs/source-license.txt', 'Synthetic test license evidence; not model redistribution authority.\n'],
]);
const input = (file) => ({
  path: file, digest: `sha256:${createHash('sha256').update(content.get(file)).digest('hex')}`,
});
const authorization = {
  schema: 'doppler.model-family-authorization/v2',
  family: 'new-family', authority: 'maintainer', maintainerId: 'independent-developer',
  purpose: 'Investigate related model dimensions without a customer or publication.',
  sourceRepository: 'https://example.invalid/publisher/model',
  sourceRevision: 'a'.repeat(40),
  conversionConfigs: [input('src/config/conversion/new-family/model.json')],
  referenceTest: input('tests/converter/new-family.test.js'),
  licenseEvidence: input('docs/source-license.txt'),
  publicationAllowed: false,
};
assert.deepEqual(validateModelFamilyAuthorization(authorization, 'new-family'), []);
for (const mutation of [
  { authority: 'customer' }, { schema: 'doppler.model-family-authorization/v1' },
  { sourceRevision: 'main' }, { sourceRepository: 'http://example.invalid/model' },
  { sourceRepository: 'https://user:password@example.invalid/model' },
  { publicationAllowed: true }, { customerId: 'buyer' }, { maintainerId: '' },
  { purpose: '' }, { conversionConfigs: [] }, { referenceTest: null },
  { licenseEvidence: { path: '../license', digest: authorization.licenseEvidence.digest } },
  { licenseEvidence: { path: '/license', digest: authorization.licenseEvidence.digest } },
  { licenseEvidence: { ...authorization.licenseEvidence, extra: true } },
  { licenseEvidence: { ...authorization.licenseEvidence, digest: [authorization.licenseEvidence.digest] } },
  { conversionConfigs: [authorization.referenceTest] },
  { conversionConfigs: [authorization.conversionConfigs[0], authorization.conversionConfigs[0]] },
]) {
  assert.ok(validateModelFamilyAuthorization({ ...authorization, ...mutation }, 'new-family').length, JSON.stringify(mutation));
}
assert.ok(validateModelFamilyAuthorization(authorization, 'other-family').length);
for (const field of Object.keys(authorization)) {
  const incomplete = structuredClone(authorization);
  delete incomplete[field];
  assert.ok(validateModelFamilyAuthorization(incomplete, 'new-family').length, field);
}

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-family-intake-'));
async function write(file, value) {
  await fs.mkdir(path.dirname(path.join(root, file)), { recursive: true });
  await fs.writeFile(path.join(root, file), value);
}
const git = (...args) => execFileSync('git', args, { cwd: root, stdio: 'pipe' }).toString();
const authorizationPath = 'tools/policies/model-family-authorizations/new-family.json';
try {
  git('init');
  await write('src/config/conversion/existing/model.json', '{}\n');
  git('add', 'src/config/conversion/existing/model.json');
  git('-c', 'user.name=Test Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'Freeze family intake fixture');
  for (const [file, value] of content) await write(file, value);
  let report = await checkModelFamilyIntake(root, 'HEAD');
  assert.equal(report.ok, false, 'unapproved scope must fail');
  await write(authorizationPath, JSON.stringify(authorization));
  report = await checkModelFamilyIntake(root, 'HEAD');
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.deepEqual(report.newFamilies, ['new-family']);
  // This fixture contains no customer, release, network, goal matrix or secret.
  await write('src/config/conversion/new-family/extra.json', '{}');
  report = await checkModelFamilyIntake(root, 'HEAD');
  assert.match(report.errors.join('\n'), /exact maintainer-approved scope/);
  await fs.unlink(path.join(root, 'src/config/conversion/new-family/extra.json'));
  await write(authorization.referenceTest.path, 'changed acceptance');
  report = await checkModelFamilyIntake(root, 'HEAD');
  assert.match(report.errors.join('\n'), /approved input changed/);
  await write(authorization.referenceTest.path, content.get(authorization.referenceTest.path));
  await fs.unlink(path.join(root, authorization.licenseEvidence.path));
  await fs.symlink(path.join(os.tmpdir(), 'outside-doppler-intake'), path.join(root, authorization.licenseEvidence.path));
  report = await checkModelFamilyIntake(root, 'HEAD');
  assert.equal(report.ok, false, 'missing/escaping references cannot authorize scope');
  await fs.unlink(path.join(root, authorization.licenseEvidence.path));
  await write(authorization.licenseEvidence.path, content.get(authorization.licenseEvidence.path));
  await fs.symlink('model.json', path.join(root, 'src/config/conversion/new-family/alias.json'));
  report = await checkModelFamilyIntake(root, 'HEAD');
  assert.match(report.errors.join('\n'), /may not contain symlinks/);
  await assert.rejects(checkModelFamilyIntake(root, 'missing-ref'), /Unable to list/);
} finally {
  await fs.rm(root, { recursive: true, force: true });
}
console.log('model-family-intake.test: ok (maintainer engineering scope; not support or publication)');
