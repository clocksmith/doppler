const UL_STAGE_SET = new Set(['stage1_joint', 'stage2_base']);
const DISTILL_STAGE_SET = new Set(['stage_a', 'stage_b']);
const OBJECTIVE_SET = new Set([
  'cross_entropy',
  'ul_stage1_joint',
  'ul_stage2_base',
  'kd',
  'triplet',
]);

function assertFiniteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`training metrics: ${label} must be a finite number.`);
  }
}

function assertNullableFiniteNumber(value, label) {
  if (value === null || value === undefined) return;
  assertFiniteNumber(value, label);
}

function assertIntegerGte(value, minValue, label) {
  if (!Number.isInteger(value) || value < minValue) {
    throw new Error(`training metrics: ${label} must be an integer >= ${minValue}.`);
  }
}

function assertOptionalIntegerGte(value, minValue, label) {
  if (value === undefined || value === null) return;
  assertIntegerGte(value, minValue, label);
}

function assertString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`training metrics: ${label} must be a non-empty string.`);
  }
}

function assertOptionalString(value, label) {
  if (value === undefined || value === null) return;
  assertString(value, label);
}

function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`training metrics: ${label} must be an object.`);
  }
}

function assertNullablePlainObject(value, label) {
  if (value === undefined || value === null) return;
  assertPlainObject(value, label);
}

function assertStringArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`training metrics: ${label} must be an array of strings.`);
  }
  for (let i = 0; i < value.length; i += 1) {
    if (typeof value[i] !== 'string') {
      throw new Error(`training metrics: ${label}[${i}] must be a string.`);
    }
  }
}

export const DEFAULT_TRAINING_METRICS_REPORT = Object.freeze({
  schemaVersion: 1,
  step: 0,
  epoch: 0,
  batch: 0,
  objective: 'cross_entropy',
  total_loss: 0,
  step_time_ms: 0,
  forward_ms: 0,
  backward_ms: 0,
  optimizer_ms: 0,
  gradient_norm_unclipped: 0,
  gradient_norm_clipped: 0,
  clipped_event_count: 0,
  total_param_count: 0,
  trainable_param_count: 0,
  lr: null,
  seed: 1337,
  model_id: 'training',
  runtime_preset: null,
  kernel_path: null,
  environment_metadata: {
    runtime: 'unknown',
  },
  memory_stats: null,
  build_provenance: null,
  trainable_groups: [],
  frozen_groups: [],
  effective_lr: null,
  scheduler_index: null,
  scheduler_phase: null,
  nan_count: 0,
  inf_count: 0,
  saturation_count: 0,
  telemetry_mode: 'step',
  telemetry_window_size: 1,
  telemetry_alerts: [],
  window_loss_avg: null,
  window_step_time_ms_avg: null,
  loss_kd: null,
  loss_triplet: null,
  distill_stage: null,
  distill_temperature: null,
  distill_alpha_kd: null,
  distill_alpha_ce: null,
  distill_loss_ce_aux: null,
  distill_loss_total: null,
  distill_triplet_margin: null,
  distill_triplet_active_count: null,
  distill_stage_a_step_count: null,
  distill_stage_a_kd_mean: null,
  ul_stage: null,
  lambda: null,
  progress_shard_index: null,
  progress_shard_count: null,
  progress_step_in_shard: null,
  progress_steps_in_shard: null,
  progress_global_step: null,
  progress_global_steps: null,
  progress_percent_complete: null,
  progress_elapsed_ms: null,
  progress_eta_ms: null,
  progress_eta_iso: null,
  loss_total: null,
  loss_prior: null,
  loss_decoder: null,
  loss_recon: null,
  latent_bitrate_proxy: null,
  coeff_ce: null,
  coeff_prior: null,
  coeff_decoder: null,
  coeff_recon: null,
  schedule_step_index: null,
  latent_clean_mean: null,
  latent_clean_std: null,
  latent_noise_mean: null,
  latent_noise_std: null,
  latent_noisy_mean: null,
  latent_noisy_std: null,
  stage1_latent_count: null,
});

export function validateTrainingMetricsEntry(entry) {
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    throw new Error('training metrics: entry must be an object.');
  }

  assertIntegerGte(entry.schemaVersion, 1, 'schemaVersion');
  assertIntegerGte(entry.step, 1, 'step');
  assertIntegerGte(entry.epoch, 0, 'epoch');
  assertIntegerGte(entry.batch, 1, 'batch');
  assertString(entry.objective, 'objective');
  if (!OBJECTIVE_SET.has(entry.objective)) {
    throw new Error(
      `training metrics: objective must be one of ${Array.from(OBJECTIVE_SET).join(', ')}.`
    );
  }
  assertFiniteNumber(entry.total_loss, 'total_loss');
  assertFiniteNumber(entry.step_time_ms, 'step_time_ms');
  assertFiniteNumber(entry.forward_ms, 'forward_ms');
  assertFiniteNumber(entry.backward_ms, 'backward_ms');

  assertNullableFiniteNumber(entry.optimizer_ms, 'optimizer_ms');
  assertNullableFiniteNumber(entry.gradient_norm_unclipped, 'gradient_norm_unclipped');
  assertNullableFiniteNumber(entry.gradient_norm_clipped, 'gradient_norm_clipped');
  assertOptionalIntegerGte(entry.clipped_event_count, 0, 'clipped_event_count');
  assertOptionalIntegerGte(entry.total_param_count, 0, 'total_param_count');
  assertOptionalIntegerGte(entry.trainable_param_count, 0, 'trainable_param_count');
  if (!('lr' in entry)) {
    throw new Error('training metrics: lr is required (nullable).');
  }
  assertNullableFiniteNumber(entry.lr, 'lr');
  if (!('seed' in entry)) {
    throw new Error('training metrics: seed is required.');
  }
  assertFiniteNumber(entry.seed, 'seed');
  if (!('model_id' in entry)) {
    throw new Error('training metrics: model_id is required.');
  }
  assertString(entry.model_id, 'model_id');
  if (!('runtime_preset' in entry)) {
    throw new Error('training metrics: runtime_preset is required (nullable).');
  }
  assertOptionalString(entry.runtime_preset, 'runtime_preset');
  if (!('kernel_path' in entry)) {
    throw new Error('training metrics: kernel_path is required (nullable).');
  }
  assertOptionalString(entry.kernel_path, 'kernel_path');
  if (!('environment_metadata' in entry)) {
    throw new Error('training metrics: environment_metadata is required.');
  }
  assertPlainObject(entry.environment_metadata, 'environment_metadata');
  if (!('memory_stats' in entry)) {
    throw new Error('training metrics: memory_stats is required (nullable).');
  }
  assertNullablePlainObject(entry.memory_stats, 'memory_stats');
  if (!('build_provenance' in entry)) {
    throw new Error('training metrics: build_provenance is required (nullable).');
  }
  assertNullablePlainObject(entry.build_provenance, 'build_provenance');
  assertNullableFiniteNumber(entry.effective_lr, 'effective_lr');
  assertOptionalIntegerGte(entry.scheduler_index, 0, 'scheduler_index');
  if (entry.scheduler_phase !== undefined && entry.scheduler_phase !== null) {
    assertString(entry.scheduler_phase, 'scheduler_phase');
  }
  assertOptionalIntegerGte(entry.nan_count, 0, 'nan_count');
  assertOptionalIntegerGte(entry.inf_count, 0, 'inf_count');
  assertOptionalIntegerGte(entry.saturation_count, 0, 'saturation_count');
  if (entry.telemetry_mode !== undefined && entry.telemetry_mode !== null) {
    if (entry.telemetry_mode !== 'step' && entry.telemetry_mode !== 'window' && entry.telemetry_mode !== 'epoch') {
      throw new Error('training metrics: telemetry_mode must be "step", "window", or "epoch".');
    }
  }
  assertOptionalIntegerGte(entry.telemetry_window_size, 1, 'telemetry_window_size');
  if (entry.telemetry_alerts !== undefined && entry.telemetry_alerts !== null) {
    assertStringArray(entry.telemetry_alerts, 'telemetry_alerts');
  }
  assertNullableFiniteNumber(entry.window_loss_avg, 'window_loss_avg');
  assertNullableFiniteNumber(entry.window_step_time_ms_avg, 'window_step_time_ms_avg');
  assertNullableFiniteNumber(entry.loss_kd, 'loss_kd');
  assertNullableFiniteNumber(entry.loss_triplet, 'loss_triplet');
  assertNullableFiniteNumber(entry.distill_temperature, 'distill_temperature');
  assertNullableFiniteNumber(entry.distill_alpha_kd, 'distill_alpha_kd');
  assertNullableFiniteNumber(entry.distill_alpha_ce, 'distill_alpha_ce');
  assertNullableFiniteNumber(entry.distill_loss_ce_aux, 'distill_loss_ce_aux');
  assertNullableFiniteNumber(entry.distill_loss_total, 'distill_loss_total');
  assertNullableFiniteNumber(entry.distill_triplet_margin, 'distill_triplet_margin');
  assertOptionalIntegerGte(entry.distill_triplet_active_count, 0, 'distill_triplet_active_count');
  assertNullableFiniteNumber(entry.distill_stage_a_step_count, 'distill_stage_a_step_count');
  assertNullableFiniteNumber(entry.distill_stage_a_kd_mean, 'distill_stage_a_kd_mean');
  if (entry.distill_stage !== undefined && entry.distill_stage !== null && !DISTILL_STAGE_SET.has(entry.distill_stage)) {
    throw new Error('training metrics: distill_stage must be "stage_a", "stage_b", or null.');
  }
  if (entry.trainable_groups !== undefined) {
    assertStringArray(entry.trainable_groups, 'trainable_groups');
  }
  if (entry.frozen_groups !== undefined) {
    assertStringArray(entry.frozen_groups, 'frozen_groups');
  }
  assertOptionalIntegerGte(entry.progress_shard_index, 1, 'progress_shard_index');
  assertOptionalIntegerGte(entry.progress_shard_count, 1, 'progress_shard_count');
  assertOptionalIntegerGte(entry.progress_step_in_shard, 1, 'progress_step_in_shard');
  assertOptionalIntegerGte(entry.progress_steps_in_shard, 1, 'progress_steps_in_shard');
  assertOptionalIntegerGte(entry.progress_global_step, 1, 'progress_global_step');
  assertOptionalIntegerGte(entry.progress_global_steps, 1, 'progress_global_steps');
  assertNullableFiniteNumber(entry.progress_percent_complete, 'progress_percent_complete');
  if (
    Number.isFinite(entry.progress_percent_complete)
    && (entry.progress_percent_complete < 0 || entry.progress_percent_complete > 100)
  ) {
    throw new Error('training metrics: progress_percent_complete must be between 0 and 100.');
  }
  assertNullableFiniteNumber(entry.progress_elapsed_ms, 'progress_elapsed_ms');
  if (Number.isFinite(entry.progress_elapsed_ms) && entry.progress_elapsed_ms < 0) {
    throw new Error('training metrics: progress_elapsed_ms must be >= 0.');
  }
  assertNullableFiniteNumber(entry.progress_eta_ms, 'progress_eta_ms');
  if (Number.isFinite(entry.progress_eta_ms) && entry.progress_eta_ms < 0) {
    throw new Error('training metrics: progress_eta_ms must be >= 0.');
  }
  if (entry.progress_eta_iso !== undefined && entry.progress_eta_iso !== null) {
    assertString(entry.progress_eta_iso, 'progress_eta_iso');
  }
  if (
    Number.isInteger(entry.progress_shard_index)
    && Number.isInteger(entry.progress_shard_count)
    && entry.progress_shard_index > entry.progress_shard_count
  ) {
    throw new Error('training metrics: progress_shard_index must be <= progress_shard_count.');
  }
  if (
    Number.isInteger(entry.progress_step_in_shard)
    && Number.isInteger(entry.progress_steps_in_shard)
    && entry.progress_step_in_shard > entry.progress_steps_in_shard
  ) {
    throw new Error('training metrics: progress_step_in_shard must be <= progress_steps_in_shard.');
  }
  if (
    Number.isInteger(entry.progress_global_step)
    && Number.isInteger(entry.progress_global_steps)
    && entry.progress_global_step > entry.progress_global_steps
  ) {
    throw new Error('training metrics: progress_global_step must be <= progress_global_steps.');
  }
  if (entry.ul_stage !== undefined && entry.ul_stage !== null && !UL_STAGE_SET.has(entry.ul_stage)) {
    throw new Error('training metrics: ul_stage must be "stage1_joint", "stage2_base", or null.');
  }

  if (entry.objective === 'cross_entropy') {
    if (entry.ul_stage != null) {
      throw new Error('training metrics: cross_entropy objective must not set ul_stage.');
    }
    if (entry.distill_stage != null) {
      throw new Error('training metrics: cross_entropy objective must not set distill_stage.');
    }
  }

  if (entry.objective === 'kd') {
    if (entry.ul_stage != null) {
      throw new Error('training metrics: kd objective must not set ul_stage.');
    }
    assertFiniteNumber(entry.loss_kd, 'loss_kd');
    if (entry.distill_stage !== 'stage_a') {
      throw new Error('training metrics: kd objective requires distill_stage="stage_a".');
    }
  }

  if (entry.objective === 'triplet') {
    if (entry.ul_stage != null) {
      throw new Error('training metrics: triplet objective must not set ul_stage.');
    }
    assertFiniteNumber(entry.loss_triplet, 'loss_triplet');
    if (entry.distill_stage !== 'stage_b') {
      throw new Error('training metrics: triplet objective requires distill_stage="stage_b".');
    }
  }

  if (entry.objective === 'ul_stage1_joint' || entry.objective === 'ul_stage2_base') {
    if (entry.distill_stage != null) {
      throw new Error('training metrics: UL objectives must not set distill_stage.');
    }
    if (entry.objective === 'ul_stage1_joint' && entry.ul_stage !== 'stage1_joint') {
      throw new Error('training metrics: ul_stage1_joint objective requires ul_stage="stage1_joint".');
    }
    if (entry.objective === 'ul_stage2_base' && entry.ul_stage !== 'stage2_base') {
      throw new Error('training metrics: ul_stage2_base objective requires ul_stage="stage2_base".');
    }
    assertFiniteNumber(entry.lambda, 'lambda');
    assertFiniteNumber(entry.loss_total, 'loss_total');
    assertFiniteNumber(entry.loss_prior, 'loss_prior');
    assertFiniteNumber(entry.loss_decoder, 'loss_decoder');
    assertFiniteNumber(entry.loss_recon, 'loss_recon');
    assertFiniteNumber(entry.latent_bitrate_proxy, 'latent_bitrate_proxy');
    assertFiniteNumber(entry.coeff_ce, 'coeff_ce');
    assertFiniteNumber(entry.coeff_prior, 'coeff_prior');
    assertFiniteNumber(entry.coeff_decoder, 'coeff_decoder');
    assertFiniteNumber(entry.coeff_recon, 'coeff_recon');
    if (entry.objective === 'ul_stage1_joint') {
      assertIntegerGte(entry.schedule_step_index, 0, 'schedule_step_index');
      assertFiniteNumber(entry.latent_clean_mean, 'latent_clean_mean');
      assertFiniteNumber(entry.latent_clean_std, 'latent_clean_std');
      assertFiniteNumber(entry.latent_noise_mean, 'latent_noise_mean');
      assertFiniteNumber(entry.latent_noise_std, 'latent_noise_std');
      assertFiniteNumber(entry.latent_noisy_mean, 'latent_noisy_mean');
      assertFiniteNumber(entry.latent_noisy_std, 'latent_noisy_std');
    }
    if (entry.objective === 'ul_stage2_base') {
      assertIntegerGte(entry.stage1_latent_count, 1, 'stage1_latent_count');
    }
  }

  return entry;
}

export function validateTrainingMetricsReport(report) {
  if (!Array.isArray(report)) {
    throw new Error('training metrics report: expected an array of entries.');
  }
  for (let i = 0; i < report.length; i += 1) {
    validateTrainingMetricsEntry(report[i]);
  }
  return report;
}
