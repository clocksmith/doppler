export interface TrainingMetricsReportSchema {
    schemaVersion: number;
    step: number;
    epoch: number;
    batch: number;
    objective: 'cross_entropy' | 'ul_stage1_joint' | 'ul_stage2_base' | 'kd' | 'triplet';
    total_loss: number;
    step_time_ms: number;
    forward_ms?: number;
    backward_ms?: number;
    optimizer_ms?: number;
    effective_lr?: number | null;
    scheduler_index?: number | null;
    scheduler_phase?: string | null;
    loss_kd?: number | null;
    loss_triplet?: number | null;
    gradient_norm_unclipped?: number;
    gradient_norm_clipped?: number;
    clipped_event_count?: number;
    total_param_count?: number;
    trainable_param_count?: number;
    trainable_groups?: string[];
    frozen_groups?: string[];
    nan_count?: number;
    inf_count?: number;
    saturation_count?: number;
    telemetry_mode?: 'step' | 'window' | 'epoch';
    telemetry_window_size?: number;
    telemetry_alerts?: string[];
    window_loss_avg?: number | null;
    window_step_time_ms_avg?: number | null;
    ul_stage?: 'stage1_joint' | 'stage2_base' | null;
    lambda?: number | null;
    loss_total?: number | null;
    loss_prior?: number | null;
    loss_decoder?: number | null;
    loss_recon?: number | null;
    latent_bitrate_proxy?: number | null;
    coeff_ce?: number | null;
    coeff_prior?: number | null;
    coeff_decoder?: number | null;
    coeff_recon?: number | null;
    schedule_step_index?: number | null;
    latent_clean_mean?: number | null;
    latent_clean_std?: number | null;
    latent_noise_mean?: number | null;
    latent_noise_std?: number | null;
    latent_noisy_mean?: number | null;
    latent_noisy_std?: number | null;
    stage1_latent_count?: number | null;
}

export declare const DEFAULT_TRAINING_METRICS_REPORT: TrainingMetricsReportSchema;

export declare function validateTrainingMetricsEntry(
  entry: TrainingMetricsReportSchema
): TrainingMetricsReportSchema;

export declare function validateTrainingMetricsReport(
  report: TrainingMetricsReportSchema[]
): TrainingMetricsReportSchema[];
