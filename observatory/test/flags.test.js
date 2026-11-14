import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { evaluateFlags, previewNames, validateFlags } from '../public/flags.js';

const flagFile = async () => JSON.parse(await readFile(new URL('../config/flags.json', import.meta.url), 'utf8'));
const config = (flags) => ({ schema: 'doppler.observatory.flags/v1', flags });
const flag = (overrides) => ({ enabled: false, enforcedBy: 'client', previewOverride: false, description: 'A switch.', ...overrides });

test('the checked-in flags file is valid', async () => {
  validateFlags(await flagFile());
});

test('a visitor can preview only a client switch that allows it', () => {
  const flags = evaluateFlags(config({
    open: flag({ previewOverride: true }),
    closed: flag({}),
  }), { preview: ['open', 'closed', 'missing'] });
  assert.deepEqual(flags.open, { enabled: true, source: 'preview', enforcedBy: 'client' });
  assert.equal(flags.closed.enabled, false);
  assert.equal(flags.missing, undefined);
});

test('a page cannot turn on publishing, rankings or voting', async () => {
  const file = await flagFile();
  const server = Object.entries(file.flags).filter(([, value]) => value.enforcedBy === 'server').map(([name]) => name);
  assert.deepEqual(server.sort(), ['rankings.community', 'results.upload', 'voting']);
  const flags = evaluateFlags(file, { preview: server });
  for (const name of server) assert.equal(flags[name].enabled, false, name);
});

test('the file itself cannot enable or preview a server-enforced switch', () => {
  assert.throws(() => validateFlags(config({ upload: flag({ enforcedBy: 'server', enabled: true }) })), /server-enforced/);
  assert.throws(() => validateFlags(config({ upload: flag({ enforcedBy: 'server', previewOverride: true }) })), /server-enforced/);
});

test('malformed flags are refused rather than read as off', () => {
  assert.throws(() => validateFlags({ schema: 'other', flags: {} }), /schema/);
  assert.throws(() => validateFlags(config({ a: flag({ enabled: 'yes' }) })), /boolean/);
  assert.throws(() => validateFlags(config({ a: flag({ enforcedBy: 'nobody' }) })), /enforcedBy/);
  assert.throws(() => validateFlags(config({ a: flag({ description: ' ' }) })), /description/);
});

test('preview names come from one query parameter', () => {
  assert.deepEqual(previewNames('?preview=views.experimental,%20voting&x=1'), ['views.experimental', 'voting']);
  assert.deepEqual(previewNames(''), []);
});
