// Integration test for the `parity` stage the bundle CLI now writes after
// receipt. Uses the committed Gemma 3 270M example program bundle to exercise
// the same call the CLI makes (checkProgramBundleParity in contract mode) and
// asserts the artifact shape + file serialization round-trip.

import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  PROGRAM_BUNDLE_PARITY_SCHEMA_ID,
  checkProgramBundleParity,
} from '../../src/tooling/program-bundle-parity.js';

const bundlePath = path.join(
  process.cwd(),
  'examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json'
);

const outDir = mkdtempSync(path.join(tmpdir(), 'doppler-bundle-parity-emit-'));

try {
  const parityReport = await checkProgramBundleParity({
    bundlePath,
    mode: 'contract',
    repoRoot: process.cwd(),
  });

  // First-class artifact contract: the CLI writes the same shape as the return
  // value, so round-tripping through JSON must preserve every field the summary
  // stage advertises.
  const parityOutputPath = path.join(outDir, 'program-bundle-parity.json');
  writeFileSync(parityOutputPath, `${JSON.stringify(parityReport, null, 2)}\n`, 'utf8');
  const roundTripped = JSON.parse(readFileSync(parityOutputPath, 'utf8'));

  assert.equal(roundTripped.schema, PROGRAM_BUNDLE_PARITY_SCHEMA_ID);
  assert.equal(roundTripped.mode, 'contract');
  assert.equal(typeof roundTripped.ok, 'boolean');
  assert.equal(typeof roundTripped.bundleId, 'string');
  assert.equal(typeof roundTripped.modelId, 'string');
  assert.equal(typeof roundTripped.executionGraphHash, 'string');
  assert.equal(typeof roundTripped.parityHash, 'string');
  assert.ok(Array.isArray(roundTripped.providers), 'providers must be an array');
  assert.ok(roundTripped.providers.length >= 1, 'providers must not be empty');

  // The per-provider shape the CLI flattens into its stage summary.
  for (const providerResult of roundTripped.providers) {
    assert.equal(typeof providerResult.provider, 'string');
    assert.equal(typeof providerResult.status, 'string');
    assert.equal(typeof providerResult.ok, 'boolean');
  }

  // Reference summary must carry the fields the F parity binder compares
  // against (tokenHash, textHash, kvCacheStateHash, executionGraphHash).
  assert.equal(typeof roundTripped.reference.executionGraphHash, 'string');
  assert.equal(typeof roundTripped.reference.tokenHash, 'string');
  assert.equal(typeof roundTripped.reference.textHash, 'string');
  assert.equal(typeof roundTripped.reference.tokensGenerated, 'number');
  assert.equal(typeof roundTripped.reference.stopReason, 'string');
  assert.equal(typeof roundTripped.reference.kvCacheStateHash, 'string');

  // Parity hash must deterministically reflect {bundleId, mode, reference,
  // provider-status summary} — rerunning under the same inputs must produce
  // byte-identical output.
  const rerun = await checkProgramBundleParity({
    bundlePath,
    mode: 'contract',
    repoRoot: process.cwd(),
  });
  assert.equal(rerun.parityHash, roundTripped.parityHash);
} finally {
  rmSync(outDir, { recursive: true, force: true });
}

console.log('doppler-bundle-parity-emit.test: ok');
