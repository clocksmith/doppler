import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, symlinkSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(HERE, '..', '..');
const QUICKSTART_CLI_PATH = path.join(ROOT_DIR, 'src', 'cli', 'doppler-quickstart.js');
const PACKAGE_JSON_PATH = path.join(ROOT_DIR, 'package.json');

const {
  parseQuickstartArgs,
  requireQuickstartContent,
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
  assert.equal(
    requireQuickstartContent({ modelId: 'gemma-3-270m-it-q4k-ehf16-af32', content: 'WebGPU exposes GPU compute in the browser and modern JS runtimes.' }),
    'WebGPU exposes GPU compute in the browser and modern JS runtimes.'
  );
  assert.throws(
    () => requireQuickstartContent({ modelId: 'gemma-3-270m-it-q4k-ehf16-af32', content: '' }),
    /returned empty output/
  );
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
  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'doppler-quickstart-link-'));
  const linkedPath = path.join(tempDir, 'doppler-gpu');
  symlinkSync(QUICKSTART_CLI_PATH, linkedPath);
  try {
    const result = spawnSync(process.execPath, [linkedPath, '--help'], {
      cwd: ROOT_DIR,
      encoding: 'utf8',
    });
    assert.equal(result.status, 0);
    assert.match(result.stdout, /npx doppler-gpu/);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

{
  const pkg = JSON.parse(readFileSync(PACKAGE_JSON_PATH, 'utf8'));
  assert.equal(pkg.bin.doppler, 'src/cli/doppler-cli.js');
  assert.equal(pkg.bin['doppler-gpu'], 'src/cli/doppler-quickstart.js');
}

console.log('doppler-quickstart-cli.test: ok');
