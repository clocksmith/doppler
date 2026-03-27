import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(HERE, '..', '..');
const QUICKSTART_CLI_PATH = path.join(ROOT_DIR, 'src', 'cli', 'doppler-quickstart.js');
const PACKAGE_JSON_PATH = path.join(ROOT_DIR, 'package.json');

const {
  parseQuickstartArgs,
  readQuickstartConfig,
  resolveQuickstartSettings,
} = await import('../../src/cli/doppler-quickstart.js');

{
  const parsed = parseQuickstartArgs(['--model', 'gemma3-270m', 'Write a haiku']);
  assert.equal(parsed.model, 'gemma3-270m');
  assert.equal(parsed.positionalPrompt, 'Write a haiku');
}

{
  const config = await readQuickstartConfig();
  assert.equal(config.defaults.model, 'gemma3-270m');
  assert.equal(config.defaults.topK, 1);
}

{
  const settings = await resolveQuickstartSettings([]);
  assert.equal(settings.action, 'run');
  assert.equal(settings.model, 'gemma3-270m');
  assert.equal(typeof settings.prompt, 'string');
  assert.ok(settings.prompt.length > 0);
}

{
  const settings = await resolveQuickstartSettings(['--list-models', '--json']);
  assert.deepEqual(settings, { action: 'list-models', json: true });
}

{
  const result = spawnSync(process.execPath, [QUICKSTART_CLI_PATH, '--help'], {
    cwd: ROOT_DIR,
    encoding: 'utf8',
  });
  assert.equal(result.status, 0);
  assert.match(result.stdout, /npx doppler-gpu/);
  assert.match(result.stdout, /--list-models/);
}

{
  const pkg = JSON.parse(readFileSync(PACKAGE_JSON_PATH, 'utf8'));
  assert.equal(pkg.bin.doppler, 'src/cli/doppler-cli.js');
  assert.equal(pkg.bin['doppler-gpu'], 'src/cli/doppler-quickstart.js');
}

console.log('doppler-quickstart-cli.test: ok');
