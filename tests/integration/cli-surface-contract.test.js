import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(HERE, '..', '..');
const CLI_PATH = path.join(ROOT_DIR, 'tools', 'doppler-cli.js');

async function runCli(args) {
  try {
    const { stdout, stderr } = await execFileAsync(
      process.execPath,
      [CLI_PATH, ...args],
      {
        cwd: ROOT_DIR,
        maxBuffer: 2 * 1024 * 1024,
      }
    );
    return { code: 0, stdout, stderr };
  } catch (error) {
    return {
      code: Number.isInteger(error?.code) ? error.code : 1,
      stdout: error?.stdout || '',
      stderr: error?.stderr || '',
    };
  }
}

{
  const result = await runCli([]);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /Usage:/);
  assert.match(result.stdout, /doppler convert/);
}

{
  const result = await runCli(['bench', '--help']);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /Usage:/);
}

{
  const result = await runCli(['unknown-command']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Unsupported command "unknown-command"/);
}

{
  const result = await runCli(['bench', '--surface']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] Missing value for --surface/);
}

{
  const result = await runCli(['bench', '--surface', 'invalid']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --surface must be one of auto, node, browser/);
}

{
  const result = await runCli(['convert', '/tmp/in', '--config', '/tmp/cfg.json', '--surface', 'browser']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert is not supported on browser relay/);
}

{
  const result = await runCli(['convert', '--config', '/tmp/cfg.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert requires <inputPath>/);
}

{
  const result = await runCli(['convert', '/tmp/in']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert requires --config <path\.json>/);
}

{
  const result = await runCli(['convert', '/tmp/in', '/tmp/out', '--config', '/tmp/cfg.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert accepts only one positional argument/);
}

{
  const result = await runCli(['convert', '/tmp/in', '--config', '/tmp/cfg-a.json', '--converter-config', '/tmp/cfg-b.json']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] convert accepts one config flag\. Use --config only\./);
}

{
  const result = await runCli(['debug', '--surface', 'browser', '--headless', 'maybe']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --headless\/--browser-headless must be true or false/);
}

{
  const result = await runCli(['debug', '--surface', 'browser', '--headed', '--headless', 'true']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --headed is mutually exclusive with --headless/);
}

{
  const result = await runCli(['test-model', '--suite', 'inference', '--training-tests', ',']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /\[error\] --training-tests must include at least one non-empty value/);
}

console.log('cli-surface-contract.test: ok');
