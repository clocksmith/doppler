import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chooseRuntime, describeRuntime, validateRuntimes } from '../public/runtimes.js';
import { evaluateFlags } from '../public/flags.js';

const file = async (name) => JSON.parse(await readFile(new URL(`../config/${name}`, import.meta.url), 'utf8'));
const on = { 'runtime.preview': { enabled: true } };
const off = { 'runtime.preview': { enabled: false } };

test('the checked-in runtimes file is valid and defaults to the hosted runtime', async () => {
  const config = validateRuntimes(await file('runtimes.json'));
  assert.equal(config.default, 'hosted');
  assert.equal(config.runtimes.hosted.preview, false);
});

test('the preview is not offered, and cannot be asked for, while its switch is off', async () => {
  const choice = chooseRuntime(await file('runtimes.json'), off, '?runtime=preview');
  assert.equal(choice.id, 'hosted');
  assert.deepEqual(choice.offered.map((item) => item.id), ['hosted']);
});

test('with the switch on the visitor still has to ask for the preview', async () => {
  const config = await file('runtimes.json');
  assert.equal(chooseRuntime(config, on, '').id, 'hosted');
  const preview = chooseRuntime(config, on, '?runtime=preview');
  assert.deepEqual([preview.id, preview.preview, preview.offered.length], ['preview', true, 2]);
  assert.match(preview.notice, /not qualified/);
  assert.equal(chooseRuntime(config, on, '?runtime=https://elsewhere.example/x.js').id, 'hosted', 'only a listed runtime can be chosen');
});

test('the switch is a visitor preview in the flags file, not on by default', async () => {
  const flags = await file('flags.json');
  assert.equal(evaluateFlags(flags)['runtime.preview'].enabled, false);
  assert.equal(evaluateFlags(flags, { preview: ['runtime.preview'] })['runtime.preview'].enabled, true);
});

test('a runtimes file cannot make a preview the default or point off this site', async () => {
  const config = await file('runtimes.json');
  assert.throws(() => validateRuntimes({ ...config, default: 'preview' }), /default runtime must exist, be unconditional and not be a preview/);
  const elsewhere = structuredClone(config);
  elsewhere.runtimes.preview.module = 'https://elsewhere.example/runtime.js';
  assert.throws(() => validateRuntimes(elsewhere), /same-site module address/);
  const unlabelled = structuredClone(config);
  unlabelled.runtimes.preview.notice = '';
  assert.throws(() => validateRuntimes(unlabelled), /needs a switch and a notice/);
});

test('a runtime is described with its commit and its preview status', () => {
  assert.equal(describeRuntime({ preview: true, release: { version: '0.7.0', label: '0.7.0 plus 9 commits', sourceCommit: 'b9116524d9a2' } }), 'Doppler 0.7.0 plus 9 commits (b9116524) · preview, not qualified');
  assert.equal(describeRuntime({ preview: false, release: { version: '0.6.36', sourceCommit: '04f9eae389aa' } }), 'Doppler 0.6.36 (04f9eae3)');
  assert.equal(describeRuntime({ preview: false, release: null }), 'Doppler, version not reported');
});
