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
    ...overrides,
  };
}

{
  const entry = validateTrainingMetricsEntry(makeBaseEntry());
  assert.equal(entry.objective, 'cross_entropy');
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
    })),
    /loss_kd must be a finite number/
  );
}

{
  assert.throws(
    () => validateTrainingMetricsEntry(makeBaseEntry({
      objective: 'triplet',
    })),
    /loss_triplet must be a finite number/
  );
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
