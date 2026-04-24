import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const { performIntake } = await import('../../src/cli/doppler-cli.js');

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(HERE, '..', '..');
const CLI_PATH = path.join(ROOT_DIR, 'src', 'cli', 'doppler-cli.js');
const FIXTURE_MANIFEST = path.join(
  ROOT_DIR,
  'models',
  'local',
  'gemma-3-270m-it-q4k-ehf16-af32',
  'manifest.json'
);

function runCli(args, options = {}) {
  return spawnSync('node', [CLI_PATH, ...args], {
    cwd: ROOT_DIR,
    encoding: 'utf8',
    env: { ...process.env },
    ...options,
  });
}

// Usage line must advertise the bundle subcommand.
{
  const result = runCli(['--help']);
  assert.equal(result.status, 0, 'doppler --help should exit 0');
  assert.match(result.stdout, /doppler bundle --manifest/, 'help must advertise doppler bundle');
  assert.match(result.stdout, /doppler bundle --convert-config/, 'help must list convert-config variant');
}

// Unknown flag surfaces an actionable error keyed to "bundle".
{
  const result = runCli(['bundle', '--out', '/tmp/ignored', '--not-a-flag', 'x']);
  assert.notEqual(result.status, 0, 'unknown flag should fail');
  assert.match(result.stdout, /Unknown flag --not-a-flag for \\?"bundle\\?"\./);
}

// Bundle without --out fails fast with an actionable message.
{
  const result = runCli(['bundle', '--manifest', FIXTURE_MANIFEST]);
  assert.notEqual(result.status, 0, 'missing --out should fail');
  assert.match(result.stdout, /bundle: --out <dir> is required\./);
  // Error envelope shape still asserted via stdout content above.
}

// --skip-capture without pre-existing artifacts records a structured blocker.
{
  const outDir = mkdtempSync(path.join(tmpdir(), 'doppler-bundle-skip-'));
  try {
    const result = runCli([
      'bundle',
      '--manifest', FIXTURE_MANIFEST,
      '--out', outDir,
      '--skip-capture',
    ]);
    assert.notEqual(result.status, 0, 'skip-capture without artifacts must fail');
    const summaryPath = path.join(outDir, 'bundle-summary.json');
    const summary = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.equal(summary.schema, 'doppler.bundle-summary/v1');
    assert.equal(summary.ok, false);
    const intakeStage = summary.stages.find((s) => s.stage === 'intake');
    assert.equal(intakeStage?.status, 'succeeded', 'intake stage must succeed on fixture manifest');
    assert.equal(intakeStage.modelId, 'gemma-3-270m-it-q4k-ehf16-af32');
    const captureStage = summary.stages.find((s) => s.stage === 'capture');
    assert.equal(captureStage?.status, 'skipped');
    const blocker = summary.blockers.find((b) => b.code === 'skip_capture_requires_artifacts');
    assert.ok(blocker, 'must record skip_capture_requires_artifacts blocker');
    // intake-report must be emitted even on bundle failure.
    const intakeReport = JSON.parse(readFileSync(path.join(outDir, 'intake-report.json'), 'utf8'));
    assert.equal(intakeReport.schema, 'doppler.intake-report/v1');
    assert.equal(intakeReport.ok, true);
  } finally {
    rmSync(outDir, { recursive: true, force: true });
  }
}

// performIntake can be called programmatically and returns a structured result.
{
  const result = await performIntake({ manifestFlag: FIXTURE_MANIFEST });
  assert.equal(result.report.schema, 'doppler.intake-report/v1');
  assert.equal(result.report.ok, true, 'fixture manifest must pass intake');
  assert.ok(Array.isArray(result.report.stages));
  const stageNames = result.report.stages.map((s) => s.stage);
  assert.deepEqual(stageNames, ['convert', 'manifest_load', 'execution_contract_check']);
  assert.equal(result.manifest?.modelId, 'gemma-3-270m-it-q4k-ehf16-af32');
  assert.equal(path.resolve(result.manifestPath), path.resolve(FIXTURE_MANIFEST));
}

// performIntake fails structurally when no manifest path is resolvable.
{
  const result = await performIntake({});
  assert.equal(result.report.ok, false);
  const blocker = result.report.blockers.find((b) => b.code === 'no_manifest_path');
  assert.ok(blocker, 'must raise no_manifest_path blocker when no manifest is provided');
}

console.log('doppler-bundle-cli: ok');
