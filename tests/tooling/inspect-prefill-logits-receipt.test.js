import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '../..');
const tool = path.join(root, 'tools/inspect-prefill-logits.js');

const help = spawnSync(process.execPath, [tool, '--help'], {
  cwd: root,
  encoding: 'utf8',
});
assert.equal(help.status, 0, help.stderr);
assert.match(help.stdout, /--logits-out <path>/);
assert.match(help.stdout, /--adapter-manifest <path>/);
assert.match(help.stdout, /--diagnostics-level <level>/);

const tempDir = mkdtempSync(path.join(os.tmpdir(), 'doppler-prefill-receipt-'));
try {
  const outputPath = path.join(tempDir, 'negative.json');
  const missingModel = path.join(tempDir, 'missing-model');
  const result = spawnSync(process.execPath, [
    tool,
    '--model-dir', missingModel,
    '--runtime-profile', 'profiles/definitely-missing-prefill-receipt-test',
    '--out', outputPath,
  ], {
    cwd: root,
    encoding: 'utf8',
  });
  assert.notEqual(result.status, 0, 'missing runtime profile must fail closed');

  const receipt = JSON.parse(readFileSync(outputPath, 'utf8'));
  assert.equal(receipt.ok, false);
  assert.equal(receipt.failurePhase, 'runtime-profile');
  assert.equal(receipt.artifactKind, 'first_token_inference_receipt');
  assert.equal(receipt.dopplerCommit?.length, 40);
  assert.equal(receipt.invocation?.workingDirectory, 'repository root');
  assert.equal(receipt.invocation?.executable, 'node');
  assert.equal(receipt.invocation?.script, 'tools/inspect-prefill-logits.js');
  assert.deepEqual(
    receipt.invocation?.command,
    ['node', 'tools/inspect-prefill-logits.js', ...receipt.invocation.argv]
  );
  assert.equal(receipt.artifactFiles?.['manifest.json'], null);
  assert.equal(receipt.error?.code, 'runtime_config_not_found');
  assert.equal(Object.hasOwn(receipt.host ?? {}, 'hostname'), false);
  assert.equal(Object.hasOwn(receipt.host ?? {}, 'serialNumber'), false);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('inspect-prefill-logits-receipt.test: ok');
