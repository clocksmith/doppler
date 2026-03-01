import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { sha256Hex } from '../../src/utils/sha256.js';
import { createTrainingConfig } from '../../src/config/training-defaults.js';
import { createUlArtifactSession, resolveStage1ArtifactContext } from '../../src/training/artifacts.js';

const TEMP_ROOT = await mkdtemp(join(tmpdir(), 'doppler-ul-artifacts-'));

function buildUlConfig(overrides = {}) {
  return createTrainingConfig({
    training: {
      enabled: true,
      ul: {
        enabled: true,
        stage: 'stage1_joint',
        ...overrides,
      },
    },
  });
}

try {
  const stage1Config = buildUlConfig({
    stage: 'stage1_joint',
    artifactDir: TEMP_ROOT,
  });
  const stage1Session = await createUlArtifactSession({
    config: stage1Config,
    stage: 'stage1_joint',
    runOptions: {
      ulArtifactDir: TEMP_ROOT,
      modelId: 'toy-model',
      timestamp: '2026-03-01T00:00:00.000Z',
      batchSize: 1,
      epochs: 1,
      maxSteps: 1,
    },
  });
  assert(stage1Session, 'stage1 session should be created');

  const stage1Entry = {
    schemaVersion: 1,
    step: 1,
    epoch: 0,
    batch: 1,
    objective: 'ul_stage1_joint',
    total_loss: 0.1,
    step_time_ms: 1,
    forward_ms: 0.5,
    backward_ms: 0.4,
    ul_stage: 'stage1_joint',
    lambda: 5,
    loss_total: 0.1,
    loss_prior: 0.01,
    loss_decoder: 0.05,
    loss_recon: 0.04,
    latent_bitrate_proxy: 1,
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
  const stage1Artifact = await stage1Session.finalize([stage1Entry]);
  const stage1ManifestPath = resolve(process.cwd(), stage1Artifact.manifestPath);
  const stage1ManifestRaw = await readFile(stage1ManifestPath, 'utf8');
  const stage1Manifest = JSON.parse(stage1ManifestRaw);
  const stage1FileHash = sha256Hex(stage1ManifestRaw);

  {
    const stage1SessionRepeat = await createUlArtifactSession({
      config: stage1Config,
      stage: 'stage1_joint',
      runOptions: {
        ulArtifactDir: TEMP_ROOT,
        modelId: 'toy-model',
        timestamp: '2026-03-01T00:00:10.000Z',
        batchSize: 1,
        epochs: 1,
        maxSteps: 1,
      },
    });
    assert(stage1SessionRepeat, 'stage1 repeat session should be created');
    await stage1SessionRepeat.appendStep(stage1Entry);
    const stage1ArtifactRepeat = await stage1SessionRepeat.finalize([stage1Entry]);
    assert.equal(
      stage1Artifact.manifestHash,
      stage1ArtifactRepeat.manifestHash,
      'deterministic manifestHash should be stable across timestamps'
    );
  }

  {
    const noLatentsPath = join(TEMP_ROOT, 'no-latents-manifest.json');
    const { latentDataset: _latentDataset, ...withoutLatents } = stage1Manifest;
    await writeFile(noLatentsPath, JSON.stringify(withoutLatents, null, 2), 'utf8');
    const noLatentsHash = sha256Hex(await readFile(noLatentsPath, 'utf8'));
    const stage2Config = buildUlConfig({
      stage: 'stage2_base',
      artifactDir: TEMP_ROOT,
      stage1Artifact: noLatentsPath,
      stage1ArtifactHash: noLatentsHash,
    });
    await assert.rejects(
      () => createUlArtifactSession({
        config: stage2Config,
        stage: 'stage2_base',
        runOptions: { ulArtifactDir: TEMP_ROOT, timestamp: '2026-03-01T00:00:00.500Z' },
      }),
      /requires stage1 latentDataset metadata/
    );
  }

  {
    const stage2Config = buildUlConfig({
      stage: 'stage2_base',
      artifactDir: TEMP_ROOT,
      stage1Artifact: stage1ManifestPath,
      stage1ArtifactHash: stage1FileHash,
    });
    const stage2Context = await resolveStage1ArtifactContext(stage2Config);
    assert(stage2Context, 'stage2 context should resolve');
    assert.equal(stage2Context.latentDataset.count, 1);
    const stage2Session = await createUlArtifactSession({
      config: stage2Config,
      stage: 'stage2_base',
      runOptions: {
        ulArtifactDir: TEMP_ROOT,
        timestamp: '2026-03-01T00:00:01.000Z',
      },
    });
    assert(stage2Session, 'stage2 session should accept file hash gate');
  }

  {
    const stage2Config = buildUlConfig({
      stage: 'stage2_base',
      artifactDir: TEMP_ROOT,
      stage1Artifact: stage1ManifestPath,
      stage1ArtifactHash: stage1Manifest.manifestHash,
    });
    const stage2Session = await createUlArtifactSession({
      config: stage2Config,
      stage: 'stage2_base',
      runOptions: {
        ulArtifactDir: TEMP_ROOT,
        timestamp: '2026-03-01T00:00:02.000Z',
      },
    });
    assert(stage2Session, 'stage2 session should accept deterministic manifest hash gate');
  }

  {
    const stage2Config = buildUlConfig({
      stage: 'stage2_base',
      artifactDir: TEMP_ROOT,
      stage1Artifact: stage1ManifestPath,
      stage1ArtifactHash: 'deadbeef',
    });
    await assert.rejects(
      () => createUlArtifactSession({
        config: stage2Config,
        stage: 'stage2_base',
        runOptions: { ulArtifactDir: TEMP_ROOT, timestamp: '2026-03-01T00:00:03.000Z' },
      }),
      /artifact hash mismatch/
    );
  }

  {
    const stageMismatchPath = join(TEMP_ROOT, 'stage-mismatch-manifest.json');
    await writeFile(stageMismatchPath, JSON.stringify({
      ...stage1Manifest,
      stage: 'stage2_base',
    }, null, 2), 'utf8');
    const stageMismatchHash = sha256Hex(await readFile(stageMismatchPath, 'utf8'));
    const stage2Config = buildUlConfig({
      stage: 'stage2_base',
      artifactDir: TEMP_ROOT,
      stage1Artifact: stageMismatchPath,
      stage1ArtifactHash: stageMismatchHash,
    });
    await assert.rejects(
      () => createUlArtifactSession({
        config: stage2Config,
        stage: 'stage2_base',
        runOptions: { ulArtifactDir: TEMP_ROOT, timestamp: '2026-03-01T00:00:04.000Z' },
      }),
      /requires stage1_joint artifact/
    );
  }

  {
    const contractMismatchPath = join(TEMP_ROOT, 'contract-mismatch-manifest.json');
    await writeFile(contractMismatchPath, JSON.stringify({
      ...stage1Manifest,
      ulContractHash: 'bad-contract-hash',
    }, null, 2), 'utf8');
    const contractMismatchHash = sha256Hex(await readFile(contractMismatchPath, 'utf8'));
    const stage2Config = buildUlConfig({
      stage: 'stage2_base',
      artifactDir: TEMP_ROOT,
      stage1Artifact: contractMismatchPath,
      stage1ArtifactHash: contractMismatchHash,
    });
    await assert.rejects(
      () => createUlArtifactSession({
        config: stage2Config,
        stage: 'stage2_base',
        runOptions: { ulArtifactDir: TEMP_ROOT, timestamp: '2026-03-01T00:00:05.000Z' },
      }),
      /contract mismatch/
    );
  }
} finally {
  await rm(TEMP_ROOT, { recursive: true, force: true });
}

console.log('ul-artifacts-contract.test: ok');
