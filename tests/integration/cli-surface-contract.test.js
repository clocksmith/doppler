import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { closeSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(HERE, '..', '..');
const CLI_PATH = path.join(ROOT_DIR, 'tools', 'doppler-cli.js');

function runCli(args) {
  const logDir = mkdtempSync(path.join(tmpdir(), 'doppler-cli-test-'));
  const stdoutPath = path.join(logDir, 'stdout.log');
  const stderrPath = path.join(logDir, 'stderr.log');
  const stdoutFd = openSync(stdoutPath, 'w');
  const stderrFd = openSync(stderrPath, 'w');

  const result = spawnSync(
    process.execPath,
    [CLI_PATH, ...args],
    {
      cwd: ROOT_DIR,
      stdio: ['ignore', stdoutFd, stderrFd],
    }
  );
  closeSync(stdoutFd);
  closeSync(stderrFd);

  const output = {
    code: result.status ?? 1,
    stdout: readFileSync(stdoutPath, 'utf8'),
    stderr: readFileSync(stderrPath, 'utf8'),
  };
  rmSync(logDir, { recursive: true, force: true });
  return output;
}

function makeTempDir() {
  return mkdtempSync(path.join(tmpdir(), 'doppler-cli-fixture-'));
}

function writeJsonFixture(content) {
  const fixtureDir = makeTempDir();
  const filePath = path.join(fixtureDir, 'config.json');
  writeFileSync(filePath, JSON.stringify(content), 'utf8');
  return { fixtureDir, filePath };
}

{
  const result = runCli([]);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /Usage:/);
}

{
  const result = runCli(['bench', '--help']);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /Usage:/);
}

{
  const result = runCli(['unknown-command']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Unsupported command "unknown-command"/);
}

{
  const result = runCli(['verify']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] command requires --config <path\.json\|json>\./);
}

{
  const result = runCli(['verify', '--json']);
  assert.equal(result.code, 1);
  const payload = JSON.parse(result.stdout);
  assert.equal(payload.ok, false);
  assert.equal(payload.schemaVersion, 1);
  assert.equal(payload.surface, null);
  assert.equal(payload.request, null);
  assert.equal(payload.error.code, 'tooling_error');
  assert.match(payload.error.message, /command requires --config <path\.json\|json>\./);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '{}']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Positional arguments are not supported\./);
}

{
  const result = runCli(['bench', '--manifset', '/tmp/run-manifest.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Unknown flag --manifset for "bench"\./);
}

{
  const result = runCli(['debug', '--runtime-config-json', '{}']);
  assert.equal(result.code, 1);
  assert.match(
    result.stderr,
    /\[error\] Unknown flag --runtime-config-json for "debug"\.( Did you mean --runtime-config\?)?/
  );
}

{
  const result = runCli(['bench', '--surface']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Missing value for --surface/);
}

{
  const result = runCli(['bench', '--surface', 'invalid', '--config', '{}']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --surface must be one of auto, node, browser/);
}

{
  const convertConfig = JSON.stringify({
    request: {
      inputDir: '/tmp/input',
      convertPayload: {
        converterConfig: {
          output: {
            modelBaseId: 'toy-model',
          },
        },
      },
    },
  });
  const result = runCli(['convert', '--surface', 'browser', '--config', convertConfig]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert is not supported on browser relay/);
}

{
  const result = runCli(['verify', '--config', '/tmp/does-not-exist-config.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config not found or unreadable:/);
}

{
  const fixtureDir = makeTempDir();
  const invalidJsonPath = path.join(fixtureDir, 'invalid.json');
  writeFileSync(invalidJsonPath, '{not-json', 'utf8');
  const result = runCli(['verify', '--config', invalidJsonPath]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config must contain valid JSON:/);
  rmSync(fixtureDir, { recursive: true, force: true });
}

{
  const fixtureDir = makeTempDir();
  const nonObjectJsonPath = path.join(fixtureDir, 'array.json');
  writeFileSync(nonObjectJsonPath, '[]', 'utf8');
  const result = runCli(['verify', '--config', nonObjectJsonPath]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config must be a JSON object\./);
  rmSync(fixtureDir, { recursive: true, force: true });
}

{
  const result = runCli([
    'verify',
    '--config',
    JSON.stringify({
      request: [],
    }),
  ]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config field "request" must be a JSON object when provided\./);
}

{
  const result = runCli([
    'verify',
    '--config',
    JSON.stringify({
      request: {
        suite: 'inference',
      },
      run: [],
    }),
  ]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config field "run" must be a JSON object when provided\./);
}

{
  const result = runCli([
    'debug',
    '--config',
    JSON.stringify({
      request: {
        command: 'bench',
      },
    }),
  ]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config request command mismatch:/);
}

{
  const result = runCli([
    'debug',
    '--config',
    JSON.stringify({
      request: {
        modelId: 'toy-model',
        runtimePreset: 'modes/debug',
      },
    }),
    '--runtime-config',
    '{"shared":{"tooling":{"intent":"investigate"}}}',
  ]);
  assert.equal(result.code, 1);
  assert.match(
    result.stderr,
    /\[error\] --runtime-config cannot be combined with runtimePreset\/runtimeConfigUrl\/runtimeConfig values inside --config request payload\./
  );
}

{
  const result = runCli([
    'verify',
    '--config',
    JSON.stringify({
      request: {
        suite: 'inference',
        modelId: 'toy-model',
        runtimePreset: 'modes/debug',
      },
    }),
  ]);
  assert.equal(result.code, 1);
  assert.doesNotMatch(
    result.stderr,
    /\[error\] --runtime-config cannot be combined with runtimePreset\/runtimeConfigUrl\/runtimeConfig values inside --config request payload\./
  );
}

{
  const result = runCli([
    'debug',
    '--config',
    JSON.stringify({
      request: {
        modelId: 'toy-model',
      },
    }),
    '--runtime-config',
    '{not-json',
  ]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Invalid --runtime-config:/);
}

{
  const result = runCli([
    'bench',
    '--config',
    JSON.stringify({
      request: {
        modelId: 'toy-model',
      },
      run: {
        bench: {
          manifest: '/tmp/does-not-exist-manifest.json',
        },
      },
    }),
  ]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] ENOENT: no such file or directory/);
}

{
  const fixtureDir = makeTempDir();
  const manifestPath = path.join(fixtureDir, 'manifest.json');
  writeFileSync(
    manifestPath,
    JSON.stringify({
      runs: [
        {
          label: 'bad-run',
          request: {
            command: 'verify',
            suite: 'unknown',
          },
        },
      ],
    }),
    'utf8'
  );
  const result = runCli([
    'verify',
    '--json',
    '--config',
    JSON.stringify({
      request: {
        suite: 'kernels',
      },
      run: {
        bench: {
          manifest: manifestPath,
        },
      },
    }),
  ]);
  assert.equal(result.code, 0);
  const payload = JSON.parse(result.stdout);
  assert.equal(Array.isArray(payload), true);
  assert.equal(payload.length, 1);
  assert.equal(payload[0].ok, false);
  assert.equal(payload[0].schemaVersion, 1);
  assert.equal(payload[0].surface, null);
  assert.equal(payload[0].request, null);
  assert.match(payload[0].error?.message, /unsupported suite "unknown"/);
  rmSync(fixtureDir, { recursive: true, force: true });
}

{
  const { fixtureDir, filePath } = writeJsonFixture({
    request: {
      modelId: 'toy-model',
    },
  });
  const result = runCli(['verify', '--config', filePath, '--surface', 'node']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] tooling command: suite is required for "verify"\./);
  rmSync(fixtureDir, { recursive: true, force: true });
}

console.log('cli-surface-contract.test: ok');
