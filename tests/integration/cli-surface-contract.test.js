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

{
  const result = runCli([]);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /Usage:/);
  assert.match(result.stdout, /doppler convert/);
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
  const result = runCli(['bench', '--surface']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Missing value for --surface/);
}

{
  const result = runCli(['bench', '--surface', 'invalid']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --surface must be one of auto, node, browser/);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '/tmp/cfg.json', '--surface', 'browser']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert is not supported on browser relay/);
}

{
  const result = runCli(['convert', '--config', '/tmp/cfg.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert requires <inputPath>/);
}

{
  const result = runCli(['convert', '/tmp/in']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert requires --config <path\.json>/);
}

{
  const result = runCli(['convert', '/tmp/in', '/tmp/out', '--config', '/tmp/cfg.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert accepts only one positional argument/);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '/tmp/cfg-a.json', '--converter-config', '/tmp/cfg-b.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --converter-config has been removed\. Use --config <path\.json>\./);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '/tmp/cfg.json', '--row-chunk-rows', '0']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --row-chunk-rows must be a positive integer/);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '/tmp/cfg.json', '--use-gpu-cast', 'maybe']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --use-gpu-cast must be true or false/);
}

{
  const result = runCli([
    'convert',
    '/tmp/in',
    '--config',
    '/tmp/cfg.json',
    '--gpu-cast-min-tensor-bytes',
    '1024',
  ]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --gpu-cast-min-tensor-bytes requires --use-gpu-cast true\./);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '/tmp/cfg.json', '--model-id', 'not-allowed']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert does not accept --model-id/);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '/tmp/cfg.json', '--workers', '0']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --workers must be a positive integer/);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '/tmp/cfg.json', '--worker-policy', 'explode']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --worker-policy must be one of: cap, error/);
}

{
  const fixtureDir = makeTempDir();
  const converterConfigPath = path.join(fixtureDir, 'config.json');
  writeFileSync(converterConfigPath, JSON.stringify({ output: { dir: '/tmp/out' } }), 'utf8');
  const result = runCli(['convert', '/tmp/does-not-exist', '--converter-config', converterConfigPath]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --converter-config has been removed\. Use --config <path\.json>\./);
  rmSync(fixtureDir, { recursive: true, force: true });
}

{
  const result = runCli(['debug', '--surface', 'browser', '--headless', 'maybe']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --headless\/--browser-headless must be true or false/);
}

{
  const result = runCli(['debug', '--surface', 'browser', '--headed', '--headless', 'true']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --headed is mutually exclusive with --headless/);
}

{
  const result = runCli(['debug', '--surface', 'browser', '--headed', '--browser-headless', 'false']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --headed is mutually exclusive with --headless/);
}

{
  const result = runCli(['debug', '--surface', 'browser', '--browser-headless', 'maybe']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --headless\/--browser-headless must be true or false/);
}

{
  const result = runCli(['test-model', '--suite', 'inference', '--training-tests', ',']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --training-tests must include at least one non-empty value/);
}

{
  const result = runCli(['bench', '--cache-mode', 'invalid']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --cache-mode must be one of: cold, warm/);
}

{
  const result = runCli(['bench', '--load-mode', 'invalid']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --load-mode must be one of: opfs, http, memory/);
}

{
  const result = runCli(['debug', '--surface', 'browser', '--model-id', 'toy-model', '--browser-port', 'abc']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --browser-port must be a number/);
}

{
  const result = runCli(['debug', '--surface', 'browser', '--model-id', 'toy-model', '--browser-timeout-ms', 'abc']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --browser-timeout-ms must be a number/);
}

{
  const result = runCli([
    'debug',
    '--surface',
    'browser',
    '--model-id',
    'toy-model',
    '--browser-base-url',
    'http://127.0.0.1:1',
    '--browser-arg',
    '--disable-gpu',
    '--browser-timeout-ms',
    'abc',
  ]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --browser-timeout-ms must be a number/);
}

{
  const result = runCli(['bench', '--runtime-config-json', '[]']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Invalid --runtime-config-json: value must be a JSON object/);
}

{
  const result = runCli([
    'debug',
    '--model-id',
    'toy-model',
    '--runtime-config',
    '{"runtime":{"shared":{"harness":{"mode":"debug"}}}}',
    '--runtime-config-json',
    '{"runtime":{"shared":{"harness":{"mode":"inference"}}}}',
  ]);
  assert.equal(result.code, 1);
  assert.match(
    result.stderr,
    /\[error\] --runtime-config cannot be combined with --runtime-preset, --runtime-config-url, or --runtime-config-json\./
  );
}

{
  const result = runCli([
    'debug',
    '--model-id',
    'toy-model',
    '--runtime-config',
    'tools/configs/runtime/modes/debug.json',
    '--runtime-preset',
    'modes/debug',
  ]);
  assert.equal(result.code, 1);
  assert.match(
    result.stderr,
    /\[error\] --runtime-config cannot be combined with --runtime-preset, --runtime-config-url, or --runtime-config-json\./
  );
}

{
  const result = runCli(['test-model', '--suite', 'training', '--training-config-json', '[1,2,3]']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Invalid --training-config-json: value must be a JSON object/);
}

{
  const result = runCli(['convert', '/tmp/in', '--config', '/tmp/does-not-exist-config.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config not found or unreadable:/);
}

{
  const fixtureDir = makeTempDir();
  const invalidJsonPath = path.join(fixtureDir, 'invalid.json');
  writeFileSync(invalidJsonPath, '{not-json', 'utf8');
  const result = runCli(['convert', '/tmp/in', '--config', invalidJsonPath]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config must contain valid JSON:/);
  rmSync(fixtureDir, { recursive: true, force: true });
}

{
  const fixtureDir = makeTempDir();
  const nonObjectJsonPath = path.join(fixtureDir, 'array.json');
  writeFileSync(nonObjectJsonPath, '[]', 'utf8');
  const result = runCli(['convert', '/tmp/in', '--config', nonObjectJsonPath]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --config must be a JSON object\./);
  rmSync(fixtureDir, { recursive: true, force: true });
}

{
  const fixtureDir = makeTempDir();
  const invalidManifestPath = path.join(fixtureDir, 'manifest.json');
  writeFileSync(invalidManifestPath, JSON.stringify({ runs: [] }), 'utf8');
  const result = runCli(['bench', '--manifest', invalidManifestPath]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] manifest must have a non-empty "runs" array/);
  rmSync(fixtureDir, { recursive: true, force: true });
}

console.log('cli-surface-contract.test: ok');
