#!/usr/bin/env node

import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { createTrainingConfig } from '../src/config/training-defaults.js';
import { createUlArtifactSession } from '../src/training/artifacts.js';
import { validateTrainingMetricsReport } from '../src/config/schema/training-metrics.schema.js';

function parseArgs(argv) {
  const parsed = {
    manifest: null,
    stage1Manifest: null,
    report: null,
    checkpoint: null,
    selfTest: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--self-test') {
      parsed.selfTest = true;
      continue;
    }
    if (arg === '--manifest') {
      parsed.manifest = argv[i + 1] || null;
      i += 1;
      continue;
    }
    if (arg === '--stage1-manifest') {
      parsed.stage1Manifest = argv[i + 1] || null;
      i += 1;
      continue;
    }
    if (arg === '--report') {
      parsed.report = argv[i + 1] || null;
      i += 1;
      continue;
    }
    if (arg === '--checkpoint') {
      parsed.checkpoint = argv[i + 1] || null;
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }
  return parsed;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertNonEmptyString(value, label) {
  assert(typeof value === 'string' && value.trim().length > 0, `${label} must be a non-empty string.`);
}

function assertFiniteNumber(value, label) {
  assert(typeof value === 'number' && Number.isFinite(value), `${label} must be a finite number.`);
}

function assertNullableString(value, label) {
  if (value === null || value === undefined) return;
  assertNonEmptyString(value, label);
}

function assertArray(value, label) {
  assert(Array.isArray(value), `${label} must be an array.`);
}

async function readManifest(pathValue) {
  const absolute = resolve(String(pathValue));
  const raw = await readFile(absolute, 'utf8');
  return { absolute, raw, parsed: JSON.parse(raw) };
}

function verifyBuildProvenance(manifest) {
  const provenance = manifest.buildProvenance;
  assert(provenance && typeof provenance === 'object' && !Array.isArray(provenance), 'buildProvenance must be an object.');
  assertNonEmptyString(provenance.runtime, 'buildProvenance.runtime');
  assert(provenance.schemaVersions && typeof provenance.schemaVersions === 'object', 'buildProvenance.schemaVersions must be an object.');
  assertFiniteNumber(provenance.schemaVersions.trainingMetrics, 'buildProvenance.schemaVersions.trainingMetrics');
  if (provenance.schemaVersions.ulManifest != null) {
    assertFiniteNumber(provenance.schemaVersions.ulManifest, 'buildProvenance.schemaVersions.ulManifest');
  }
  if (provenance.schemaVersions.ulTraining != null) {
    assertFiniteNumber(provenance.schemaVersions.ulTraining, 'buildProvenance.schemaVersions.ulTraining');
  }
  if (provenance.schemaVersions.distillManifest != null) {
    assertFiniteNumber(provenance.schemaVersions.distillManifest, 'buildProvenance.schemaVersions.distillManifest');
  }
  if (provenance.schemaVersions.distillTraining != null) {
    assertFiniteNumber(provenance.schemaVersions.distillTraining, 'buildProvenance.schemaVersions.distillTraining');
  }
}

function verifyMetricsBlock(manifest) {
  assert(manifest.metrics && typeof manifest.metrics === 'object', 'metrics must be an object.');
  assertFiniteNumber(manifest.metrics.count, 'metrics.count');
  assertNonEmptyString(manifest.metrics.stepMetricsPath, 'metrics.stepMetricsPath');
}

async function readNdjson(pathValue) {
  const absolute = resolve(String(pathValue));
  const raw = await readFile(absolute, 'utf8');
  const lines = raw.split('\n').map((line) => line.trim()).filter(Boolean);
  return {
    absolute,
    entries: lines.map((line) => JSON.parse(line)),
  };
}

async function verifyManifestMetricsPayload(manifest) {
  const stepMetricsPath = resolve(process.cwd(), String(manifest.metrics.stepMetricsPath));
  const metrics = await readNdjson(stepMetricsPath);
  assertArray(metrics.entries, 'metrics entries');
  assert(
    metrics.entries.length === Number(manifest.metrics.count),
    `metrics count mismatch: manifest=${manifest.metrics.count}, file=${metrics.entries.length}`
  );
  validateTrainingMetricsReport(metrics.entries);
}

function verifyUlRuntimeDump(manifest) {
  const dump = manifest.runtimeDump;
  assert(dump && typeof dump === 'object' && !Array.isArray(dump), 'runtimeDump must be an object.');
  assertNonEmptyString(dump.stage, 'runtimeDump.stage');
  assertFiniteNumber(dump.lambda0, 'runtimeDump.lambda0');
  assert(dump.noiseSchedule && typeof dump.noiseSchedule === 'object', 'runtimeDump.noiseSchedule must be an object.');
}

function isDistillManifest(manifest) {
  const stage = String(manifest?.stage || '').trim();
  if (stage === 'stage_a' || stage === 'stage_b') return true;
  return typeof manifest?.distillContractHash === 'string';
}

function verifyDistillRuntimeDump(manifest) {
  const dump = manifest.runtimeDump;
  assert(dump && typeof dump === 'object' && !Array.isArray(dump), 'runtimeDump must be an object.');
  assertNonEmptyString(dump.stage, 'runtimeDump.stage');
  if (dump.temperature != null) {
    assertFiniteNumber(dump.temperature, 'runtimeDump.temperature');
  }
  if (dump.alphaKd != null) {
    assertFiniteNumber(dump.alphaKd, 'runtimeDump.alphaKd');
  }
  if (dump.alphaCe != null) {
    assertFiniteNumber(dump.alphaCe, 'runtimeDump.alphaCe');
  }
  if (dump.tripletMargin != null) {
    assertFiniteNumber(dump.tripletMargin, 'runtimeDump.tripletMargin');
  }
}

async function verifyUlManifestShape(manifest) {
  assertFiniteNumber(manifest.schemaVersion, 'schemaVersion');
  assertNonEmptyString(manifest.stage, 'stage');
  assertNonEmptyString(manifest.configHash, 'configHash');
  assertNonEmptyString(manifest.modelHash, 'modelHash');
  assertNonEmptyString(manifest.datasetHash, 'datasetHash');
  assertNonEmptyString(manifest.ulContractHash, 'ulContractHash');
  assertNonEmptyString(manifest.manifestHash, 'manifestHash');
  assertNonEmptyString(manifest.manifestContentHash, 'manifestContentHash');
  assert(manifest.manifestHash === manifest.manifestContentHash, 'manifestHash and manifestContentHash must match.');
  verifyBuildProvenance(manifest);
  verifyUlRuntimeDump(manifest);
  verifyMetricsBlock(manifest);
  await verifyManifestMetricsPayload(manifest);
  if (manifest.stage === 'stage1_joint') {
    assert(manifest.latentDataset && typeof manifest.latentDataset === 'object', 'stage1 latentDataset must exist.');
    assertNonEmptyString(manifest.latentDataset.path, 'latentDataset.path');
    assertNonEmptyString(manifest.latentDataset.hash, 'latentDataset.hash');
    assertFiniteNumber(manifest.latentDataset.count, 'latentDataset.count');
    assertFiniteNumber(manifest.latentDataset.summary?.vectorCount, 'latentDataset.summary.vectorCount');
    assert(manifest.latentDataset.summary.vectorCount >= 1, 'latentDataset.summary.vectorCount must be >= 1.');
  }
}

async function verifyDistillManifestShape(manifest) {
  assertFiniteNumber(manifest.schemaVersion, 'schemaVersion');
  assertNonEmptyString(manifest.stage, 'stage');
  assertNonEmptyString(manifest.configHash, 'configHash');
  assertNonEmptyString(manifest.modelHash, 'modelHash');
  assertNonEmptyString(manifest.datasetHash, 'datasetHash');
  assertNonEmptyString(manifest.distillContractHash, 'distillContractHash');
  assertNonEmptyString(manifest.manifestHash, 'manifestHash');
  assertNonEmptyString(manifest.manifestContentHash, 'manifestContentHash');
  assert(manifest.manifestHash === manifest.manifestContentHash, 'manifestHash and manifestContentHash must match.');
  verifyBuildProvenance(manifest);
  verifyDistillRuntimeDump(manifest);
  verifyMetricsBlock(manifest);
  await verifyManifestMetricsPayload(manifest);

  if (manifest.stage === 'stage_b') {
    assert(manifest.stageADependency && typeof manifest.stageADependency === 'object', 'stage_b stageADependency must exist.');
    assertNonEmptyString(manifest.stageADependency.hash, 'stageADependency.hash');
    assertNonEmptyString(manifest.lineage?.parentManifestHash, 'lineage.parentManifestHash');
    assertNonEmptyString(manifest.lineage?.parentContractHash, 'lineage.parentContractHash');
  }
}

async function verifyManifestShape(manifest) {
  if (isDistillManifest(manifest)) {
    await verifyDistillManifestShape(manifest);
    return;
  }
  await verifyUlManifestShape(manifest);
}

function verifyCheckpointShape(checkpoint) {
  assert(checkpoint && typeof checkpoint === 'object', 'checkpoint must be an object.');
  assert(checkpoint.metadata && typeof checkpoint.metadata === 'object', 'checkpoint.metadata must be an object.');
  const metadata = checkpoint.metadata;
  assertNonEmptyString(metadata.configHash, 'checkpoint.metadata.configHash');
  assertNonEmptyString(metadata.datasetHash, 'checkpoint.metadata.datasetHash');
  assertNullableString(metadata.tokenizerHash, 'checkpoint.metadata.tokenizerHash');
  assertNonEmptyString(metadata.optimizerHash, 'checkpoint.metadata.optimizerHash');
  assertNullableString(metadata.runtimePresetId, 'checkpoint.metadata.runtimePresetId');
  assertNullableString(metadata.kernelPathId, 'checkpoint.metadata.kernelPathId');
  assertFiniteNumber(metadata.timestamp, 'checkpoint.metadata.timestamp');
  assert(
    metadata.environmentMetadata && typeof metadata.environmentMetadata === 'object' && !Array.isArray(metadata.environmentMetadata),
    'checkpoint.metadata.environmentMetadata must be an object.'
  );
  assertNonEmptyString(metadata.environmentMetadata.runtime, 'checkpoint.metadata.environmentMetadata.runtime');
  assertNullableString(metadata.environmentMetadata.command, 'checkpoint.metadata.environmentMetadata.command');
  assertNullableString(metadata.environmentMetadata.surface, 'checkpoint.metadata.environmentMetadata.surface');
  if (metadata.environmentMetadata.gpuAdapter != null) {
    assert(
      metadata.environmentMetadata.gpuAdapter && typeof metadata.environmentMetadata.gpuAdapter === 'object' && !Array.isArray(metadata.environmentMetadata.gpuAdapter),
      'checkpoint.metadata.environmentMetadata.gpuAdapter must be an object.'
    );
  }
  if (metadata.buildProvenance != null) {
    assert(
      metadata.buildProvenance && typeof metadata.buildProvenance === 'object' && !Array.isArray(metadata.buildProvenance),
      'checkpoint.metadata.buildProvenance must be an object.'
    );
    assertNullableString(metadata.buildProvenance.commitHash, 'checkpoint.metadata.buildProvenance.commitHash');
    assertNullableString(metadata.buildProvenance.buildId, 'checkpoint.metadata.buildProvenance.buildId');
    assertNullableString(metadata.buildProvenance.buildTimestamp, 'checkpoint.metadata.buildProvenance.buildTimestamp');
  }
  const lineage = checkpoint.metadata.lineage;
  assert(lineage && typeof lineage === 'object', 'checkpoint.metadata.lineage must be an object.');
  assertNonEmptyString(lineage.checkpointKey, 'checkpoint.metadata.lineage.checkpointKey');
  assertFiniteNumber(lineage.sequence, 'checkpoint.metadata.lineage.sequence');
  assertNonEmptyString(checkpoint.metadata.checkpointHash, 'checkpoint.metadata.checkpointHash');
}

async function verifyReportShape(report, options = {}) {
  assert(report && typeof report === 'object', 'report must be an object.');
  assertNonEmptyString(report.suite, 'report.suite');
  assertNonEmptyString(report.modelId, 'report.modelId');
  assert(report.metrics == null || typeof report.metrics === 'object', 'report.metrics must be object/null.');

  const trainingReport = report.metrics?.trainingMetricsReport;
  if (Array.isArray(trainingReport) && trainingReport.length > 0) {
    validateTrainingMetricsReport(trainingReport);
  }

  const artifactEntries = [];
  const metricsArtifacts = report.metrics?.ulArtifacts;
  if (Array.isArray(metricsArtifacts)) {
    artifactEntries.push(...metricsArtifacts);
  }
  const metricsDistillArtifacts = report.metrics?.distillArtifacts;
  if (Array.isArray(metricsDistillArtifacts)) {
    artifactEntries.push(...metricsDistillArtifacts);
  }
  const lineageArtifacts = report.lineage?.training?.ulArtifacts;
  if (Array.isArray(lineageArtifacts)) {
    artifactEntries.push(...lineageArtifacts);
  }
  const lineageDistillArtifacts = report.lineage?.training?.distillArtifacts;
  if (Array.isArray(lineageDistillArtifacts)) {
    artifactEntries.push(...lineageDistillArtifacts);
  }

  for (const artifact of artifactEntries) {
    if (!artifact || typeof artifact !== 'object') continue;
    if (!artifact.manifestPath) continue;
    const manifestRaw = await readManifest(artifact.manifestPath);
    await verifyManifestShape(manifestRaw.parsed);
    if (typeof artifact.manifestHash === 'string' && artifact.manifestHash.trim()) {
      assert(
        artifact.manifestHash === manifestRaw.parsed.manifestHash,
        'report artifact manifestHash must match linked manifest.manifestHash'
      );
    }
  }

  const checkpointResumeTimeline = report.lineage?.training?.checkpointResumeTimeline;
  if (checkpointResumeTimeline != null) {
    assertArray(checkpointResumeTimeline, 'report.lineage.training.checkpointResumeTimeline');
    for (let i = 0; i < checkpointResumeTimeline.length; i += 1) {
      const event = checkpointResumeTimeline[i];
      assert(event && typeof event === 'object', `report.lineage.training.checkpointResumeTimeline[${i}] must be an object.`);
      assertNonEmptyString(event.type, `report.lineage.training.checkpointResumeTimeline[${i}].type`);
      assertFiniteNumber(event.timestamp, `report.lineage.training.checkpointResumeTimeline[${i}].timestamp`);
      if (event.type === 'resume_override_applied') {
        assertArray(event.resumeAudits, `report.lineage.training.checkpointResumeTimeline[${i}].resumeAudits`);
        for (let j = 0; j < event.resumeAudits.length; j += 1) {
          const audit = event.resumeAudits[j];
          assert(
            audit && typeof audit === 'object' && !Array.isArray(audit),
            `report.lineage.training.checkpointResumeTimeline[${i}].resumeAudits[${j}] must be an object.`
          );
          assertFiniteNumber(
            audit.timestamp,
            `report.lineage.training.checkpointResumeTimeline[${i}].resumeAudits[${j}].timestamp`
          );
          assertArray(
            audit.mismatchedFields,
            `report.lineage.training.checkpointResumeTimeline[${i}].resumeAudits[${j}].mismatchedFields`
          );
          assertNonEmptyString(
            audit.source,
            `report.lineage.training.checkpointResumeTimeline[${i}].resumeAudits[${j}].source`
          );
          assertNonEmptyString(
            audit.reason,
            `report.lineage.training.checkpointResumeTimeline[${i}].resumeAudits[${j}].reason`
          );
          assertNullableString(
            audit.operator,
            `report.lineage.training.checkpointResumeTimeline[${i}].resumeAudits[${j}].operator`
          );
          assertNullableString(
            audit.priorCheckpointMetadataHash,
            `report.lineage.training.checkpointResumeTimeline[${i}].resumeAudits[${j}].priorCheckpointMetadataHash`
          );
        }
      }
    }
  }

  const checkpoint = options.checkpoint;
  if (checkpoint) {
    verifyCheckpointShape(checkpoint);
    const reportCheckpoint = report.lineage?.checkpoint;
    if (reportCheckpoint && typeof reportCheckpoint === 'object') {
      if (reportCheckpoint.checkpointHash) {
        assert(
          reportCheckpoint.checkpointHash === checkpoint.metadata.checkpointHash,
          'report checkpoint hash must match checkpoint metadata hash.'
        );
      }
      if (reportCheckpoint.previousCheckpointHash != null) {
        assert(
          reportCheckpoint.previousCheckpointHash === checkpoint.metadata.lineage.previousCheckpointHash,
          'report previous checkpoint hash must match checkpoint lineage.'
        );
      }
    }
  }
}

function verifyStage2Dependency(stage2Manifest, stage1Manifest) {
  assert(stage2Manifest.stage === 'stage2_base', 'stage2 manifest stage must be stage2_base.');
  assert(stage1Manifest.stage === 'stage1_joint', 'stage1 manifest stage must be stage1_joint.');
  const dependency = stage2Manifest.stage1Dependency;
  assert(dependency && typeof dependency === 'object', 'stage2Dependency must exist for stage2.');
  assertNonEmptyString(dependency.hash, 'stage2Dependency.hash');
  assert(
    dependency.manifestHash === stage1Manifest.manifestHash,
    'stage2Dependency.manifestHash must match stage1 manifestHash.'
  );
  assert(
    stage2Manifest.lineage?.parentManifestHash === stage1Manifest.manifestHash,
    'lineage.parentManifestHash must match stage1 manifestHash.'
  );
  assert(
    stage2Manifest.lineage?.parentContractHash === stage1Manifest.ulContractHash,
    'lineage.parentContractHash must match stage1 ulContractHash.'
  );
}

async function runSelfTest() {
  const tempDir = await mkdtemp(join(tmpdir(), 'doppler-provenance-selftest-'));
  try {
    const stage1Config = createTrainingConfig({
      training: {
        enabled: true,
        ul: {
          enabled: true,
          stage: 'stage1_joint',
          artifactDir: tempDir,
        },
      },
    });
    const stage1Session = await createUlArtifactSession({
      config: stage1Config,
      stage: 'stage1_joint',
      runOptions: {
        ulArtifactDir: tempDir,
        modelId: 'selftest-model',
        timestamp: '2026-03-01T00:00:00.000Z',
        batchSize: 1,
        epochs: 1,
        maxSteps: 1,
      },
    });
    assert(stage1Session, 'failed to create stage1 session in self-test');
    const stage1Entry = {
      schemaVersion: 1,
      step: 1,
      epoch: 0,
      batch: 1,
      objective: 'ul_stage1_joint',
      total_loss: 0.1,
      step_time_ms: 1,
      forward_ms: 0.5,
      backward_ms: 0.5,
      lr: 0.0001,
      seed: 1337,
      model_id: 'selftest-model',
      runtime_preset: null,
      kernel_path: null,
      environment_metadata: { runtime: 'node' },
      memory_stats: null,
      build_provenance: null,
      ul_stage: 'stage1_joint',
      lambda: 5,
      loss_total: 0.1,
      loss_prior: 0.01,
      loss_decoder: 0.05,
      loss_recon: 0.04,
      latent_bitrate_proxy: 1.1,
      coeff_ce: 1,
      coeff_prior: 1,
      coeff_decoder: 1,
      coeff_recon: 1,
      schedule_step_index: 0,
      latent_clean_mean: 0.1,
      latent_clean_std: 0.2,
      latent_noise_mean: 0.01,
      latent_noise_std: 0.3,
      latent_noisy_mean: 0.11,
      latent_noisy_std: 0.22,
      latent_shape: [2, 3],
      latent_clean_values: [0.5, 0.1, -0.3, 0.2, 0.4, -0.1],
      latent_noise_values: [0.01, -0.02, 0.03, -0.04, 0.02, -0.01],
      latent_noisy_values: [0.51, 0.08, -0.27, 0.16, 0.42, -0.11],
    };
    await stage1Session.appendStep(stage1Entry);
    const stage1Result = await stage1Session.finalize([stage1Entry]);

    const stage1Path = resolve(process.cwd(), stage1Result.manifestPath);
    const stage1Manifest = (await readManifest(stage1Path)).parsed;
    await verifyManifestShape(stage1Manifest);

    const stage2Config = createTrainingConfig({
      training: {
        enabled: true,
        ul: {
          enabled: true,
          stage: 'stage2_base',
          artifactDir: tempDir,
          stage1Artifact: stage1Path,
          stage1ArtifactHash: stage1Result.manifestHash,
        },
      },
    });
    const stage2Session = await createUlArtifactSession({
      config: stage2Config,
      stage: 'stage2_base',
      runOptions: {
        ulArtifactDir: tempDir,
        modelId: 'selftest-model',
        timestamp: '2026-03-01T00:00:01.000Z',
        batchSize: 1,
        epochs: 1,
        maxSteps: 1,
      },
    });
    assert(stage2Session, 'failed to create stage2 session in self-test');
    await stage2Session.appendStep({
      ...stage1Entry,
      objective: 'ul_stage2_base',
      ul_stage: 'stage2_base',
      step: 1,
      stage1_latent_count: 1,
    });
    const stage2Result = await stage2Session.finalize([{
      ...stage1Entry,
      objective: 'ul_stage2_base',
      ul_stage: 'stage2_base',
      step: 1,
      stage1_latent_count: 1,
    }]);
    const stage2Path = resolve(process.cwd(), stage2Result.manifestPath);
    const stage2Manifest = (await readManifest(stage2Path)).parsed;
    await verifyManifestShape(stage2Manifest);
    verifyStage2Dependency(stage2Manifest, stage1Manifest);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function main() {
  const args = parseArgs(process.argv);
  let checkpoint = null;
  if (args.checkpoint) {
    checkpoint = (await readManifest(args.checkpoint)).parsed;
    verifyCheckpointShape(checkpoint);
  }

  if (args.report) {
    const report = (await readManifest(args.report)).parsed;
    await verifyReportShape(report, { checkpoint });
    console.log('verify-training-provenance: report ok');
    return;
  }

  if (args.selfTest) {
    await runSelfTest();
    console.log('verify-training-provenance: self-test ok');
    return;
  }

  if (!args.manifest) {
    throw new Error(
      'Usage: node tools/verify-training-provenance.mjs --manifest <path> [--stage1-manifest <path>] [--report <path>] [--checkpoint <path>]'
    );
  }

  const manifest = (await readManifest(args.manifest)).parsed;
  await verifyManifestShape(manifest);
  if (manifest.stage === 'stage2_base') {
    if (!args.stage1Manifest) {
      throw new Error('stage2 verification requires --stage1-manifest <path>.');
    }
    const stage1 = (await readManifest(args.stage1Manifest)).parsed;
    await verifyManifestShape(stage1);
    verifyStage2Dependency(manifest, stage1);
  }

  console.log('verify-training-provenance: ok');
}

await main();
