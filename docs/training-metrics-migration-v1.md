# Training Metrics Schema v1 Migration Notes

This note documents the training metrics schema contract tightening.

## Summary

`training-metrics.schema` now enforces objective-aware unions:

- `cross_entropy`
- `ul_stage1_joint`
- `ul_stage2_base`
- `kd`
- `triplet`

## New/Clarified Fields

- UL objective required fields:
  - `ul_stage`
  - `lambda`
  - `loss_total`
  - `loss_prior`
  - `loss_decoder`
  - `loss_recon`
  - `latent_bitrate_proxy`
  - `coeff_ce`, `coeff_prior`, `coeff_decoder`, `coeff_recon`
  - Stage1-only: `schedule_step_index`, `latent_clean_*`, `latent_noise_*`, `latent_noisy_*`
  - Stage2-only: `stage1_latent_count`
- Telemetry fields:
  - `effective_lr`, `scheduler_index`, `scheduler_phase`
  - `nan_count`, `inf_count`, `saturation_count`
  - `telemetry_mode`, `telemetry_window_size`
  - `window_loss_avg`, `window_step_time_ms_avg`

## Behavioral Changes

- `cross_entropy` entries must not set `ul_stage`.
- `cross_entropy` entries must not set `distill_stage`.
- `ul_stage1_joint` requires `ul_stage="stage1_joint"`.
- `ul_stage2_base` requires `ul_stage="stage2_base"`.
- `kd` requires `loss_kd` and `distill_stage="stage_a"`.
- `triplet` requires `loss_triplet` and `distill_stage="stage_b"`.

## Compatibility Guidance

- Producers should emit all required base fields for every step.
- Consumers should treat missing objective-required fields as invalid payloads.
- For archived payloads, migrate by backfilling explicit `null`/default fields
  where semantically valid before validation.
