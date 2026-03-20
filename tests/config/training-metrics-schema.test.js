import assert from 'node:assert/strict';
import {
  validateTrainingMetricsEntry,
  validateTrainingMetricsReport,
} from '../../src/config/schema/training-metrics.schema.js';

function makeBaseEntry(overrides = {}) {
  return {
    schemaVersion: 1,
    step: 1,
    epoch: 0,
    batch: 1,
    objective: 'cross_entropy',
    total_loss: 0.42,
    step_time_ms: 1.2,
    forward_ms: 0.6,
    backward_ms: 0.5,
    lr: 0.0001,
    seed: 1337,
    model_id: 'training-test-model',
    runtime_profile: null,
    kernel_path: null,
    environment_metadata: { runtime: 'node' },
    memory_stats: null,
    build_provenance: null,
    ...overrides,
  };
}

{
  const entry = validateTrainingMetricsEntry(makeBaseEntry());
  assert.equal(entry.objective, 'cross_entropy');
}

{
  const legacyRuntimeField = 'runtime' + '_preset';
  const legacyOnlyEntry = makeBaseEntry({ [legacyRuntimeField]: null });
  delete legacyOnlyEntry.runtime_profile;
  assert.throws(
    () => validateTrainingMetricsEntry(legacyOnlyEntry),
    /runtime_profile is required/
  );
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({ ul_stage: 'stage1_joint' })),
    /cross_entropy objective must not set ul_stage/
  );
}

{
  const ulStage1 = validateTrainingMetricsEntry(makeBaseEntry({
    objective: 'ul_stage1_joint',
    ul_stage: 'stage1_joint',
    lambda: 5,
    loss_total: 0.31,
    loss_prior: 0.11,
    loss_decoder: 0.12,
    loss_recon: 0.08,
    latent_bitrate_proxy: 1.6,
    coeff_ce: 1,
    coeff_prior: 1,
    coeff_decoder: 1,
    coeff_recon: 1,
    schedule_step_index: 0,
    latent_clean_mean: 0.1,
    latent_clean_std: 0.2,
    latent_noise_mean: 0,
    latent_noise_std: 0.3,
    latent_noisy_mean: 0.05,
    latent_noisy_std: 0.22,
  }));
  assert.equal(ulStage1.objective, 'ul_stage1_joint');
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({
      objective: 'ul_stage2_base',
      ul_stage: 'stage2_base',
      lambda: 5,
      loss_total: 0.2,
      loss_prior: 0,
      loss_recon: 0,
      latent_bitrate_proxy: 0.9,
      coeff_ce: 1,
      coeff_prior: 1,
      coeff_decoder: 1,
      coeff_recon: 1,
    })),
    /loss_decoder must be a finite number/
  );
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({
      objective: 'ul_stage2_base',
      ul_stage: 'stage2_base',
      lambda: 5,
      loss_total: 0.2,
      loss_prior: 0,
      loss_decoder: 0.2,
      loss_recon: 0,
      latent_bitrate_proxy: 0.9,
      coeff_ce: 1,
      coeff_prior: 1,
      coeff_decoder: 1,
      coeff_recon: 1,
      stage1_latent_count: 0,
    })),
    /stage1_latent_count must be an integer >= 1/
  );
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({
      objective: 'kd',
      loss_kd: 0.12,
      distill_stage: 'stage_a',
    })),
    /distill_temperature must be a finite number/
  );
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({
      objective: 'triplet',
      loss_triplet: 0.21,
      distill_stage: 'stage_b',
    })),
    /distill_triplet_margin must be a finite number/
  );
}

{
  const kd = validateTrainingMetricsEntry(makeBaseEntry({
    objective: 'kd',
    loss_kd: 0.12,
    distill_stage: 'stage_a',
    distill_temperature: 1,
    distill_alpha_kd: 1,
    distill_alpha_ce: 0,
    distill_loss_total: 0.12,
  }));
  assert.equal(kd.distill_stage, 'stage_a');
}

{
  const triplet = validateTrainingMetricsEntry(makeBaseEntry({
    objective: 'triplet',
    loss_triplet: 0.21,
    distill_stage: 'stage_b',
    distill_triplet_margin: 0.2,
    distill_triplet_active_count: 1,
  }));
  assert.equal(triplet.distill_stage, 'stage_b');
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({
      telemetry_mode: 'invalid',
    })),
    /telemetry_mode must be "step", "window", or "epoch"/
  );
}

{
  const entry = validateTrainingMetricsEntry(makeBaseEntry({
    telemetry_alerts: ['max_step_time_ms_exceeded'],
  }));
  assert.deepEqual(entry.telemetry_alerts, ['max_step_time_ms_exceeded']);
}

{
  const entry = validateTrainingMetricsEntry(makeBaseEntry({
    progress_shard_index: 2,
    progress_shard_count: 5,
    progress_step_in_shard: 1,
    progress_steps_in_shard: 2,
    progress_global_step: 3,
    progress_global_steps: 10,
    progress_percent_complete: 30,
    progress_elapsed_ms: 1000,
    progress_eta_ms: 2000,
    progress_eta_iso: '2026-03-02T17:00:00.000Z',
  }));
  assert.equal(entry.progress_shard_index, 2);
  assert.equal(entry.progress_percent_complete, 30);
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({
      progress_shard_index: 6,
      progress_shard_count: 5,
    })),
    /progress_shard_index must be <= progress_shard_count/
  );
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({
      telemetry_alerts: [123],
    })),
    /telemetry_alerts\[0\] must be a string/
  );
}

{
  const report = validateTrainingMetricsReport([
    makeBaseEntry(),
    makeBaseEntry({
      step: 2,
      objective: 'ul_stage2_base',
      ul_stage: 'stage2_base',
      lambda: 4.5,
      loss_total: 0.15,
      loss_prior: 0,
      loss_decoder: 0.15,
      loss_recon: 0,
      latent_bitrate_proxy: 1.2,
      coeff_ce: 1,
      coeff_prior: 1,
      coeff_decoder: 1,
      coeff_recon: 1,
      stage1_latent_count: 2,
    }),
  ]);
  assert.equal(report.length, 2);
}

console.log('training-metrics-schema.test: ok');
