# Migrating from DopplerX 0.1.7 to 0.2.0

DopplerX 0.2.0 is a breaking release for one reason: every enum value Doppler
owns, including every value it emits in results, errors, failure classes,
reasons, statuses, modes, decode stats, receipts, diagnostics and training
artifacts, is now spelled in kebab-case. Nothing else in the public API, the Capsule
formats or the defaults changed between 0.1.7 and 0.2.0. Error codes
(`DOPPLER_*`), export names, function signatures, object keys and schema ids
are unchanged.

## Compatibility rule

Legacy spellings are read throughout 0.x. Doppler emits only kebab-case. Aliases are removed no earlier than 1.0, and only with at least one minor release of notice.

## What a consumer must change

Doppler now **emits** only kebab-case. Code that compares a value Doppler
returns, or narrows on its TypeScript type, must use the new spelling.

| Where the value appears | 0.1.7 | 0.2.0 |
| --- | --- | --- |
| Provider `failureClass`, receipts, `fallbackDecision.reason` | `gpu_device_lost`, `gpu_oom`, `gpu_timeout`, `gpu_unsupported`, `gpu_unavailable`, `model_load_failed`, `policy_denied`, `runtime_internal`, `fallback_failed` | `gpu-device-lost`, `gpu-oom`, `gpu-timeout`, `gpu-unsupported`, `gpu-unavailable`, `model-load-failed`, `policy-denied`, `runtime-internal`, `fallback-failed` |
| Provider fallback surface and policy reasons | `openai_compat`, `policy_local_only`, `policy_cloud_only`, `policy_prefer_cloud`, `no_fallback_configured` | `openai-compat`, `policy-local-only`, `policy-cloud-only`, `policy-prefer-cloud`, `no-fallback-configured` |
| `encodeSequence()` result `alphabet`, manifest `inference.sequence.alphabet`, `sequenceAlphabet` | `amino_acid` | `amino-acid` (`nucleotide` is unchanged) |
| Sequence qualification abort (`sequenceQualificationAbort`) | `before_execution`, `after_start` | `before-execution`, `after-start` |
| Structured-head `modelType` | `structured_json_head`, `dream_structured`, `dream_intent_posterior_head`, `dream_d1_to2_bridge`, `dream_synthesis`, `dream_energy_compose`, `energy_row_head`, `dream_energy_head` | the same words joined with `-` |
| Diffusion scheduler `type` | `flowmatch_euler`, `euler_a`, `dpmpp_2m` | `flowmatch-euler`, `euler-a`, `dpmpp-2m` |
| KV cache `layout`, tiering and quant modes, gating | `contiguous_quantized`, `bdpa_paged`, `turboquant_prod`, `force_on`, `force_off` | `contiguous-quantized`, `bdpa-paged`, `turboquant-prod`, `force-on`, `force-off` |
| Tooling `diagnostics`, `receiptPolicy` | `on_failure` | `on-failure` |
| Decode stats `decodeMode`, `decodeStrategy` | `single_token`, `batched_gpu`, `batched_gpu_stepwise_ple`, `self_speculation`, `replay_prefill`, `target_verified_lookup` | `single-token`, `batched-gpu`, `batched-gpu-stepwise-ple`, `self-speculation`, `replay-prefill`, `target-verified-lookup` |
| Decode stats `batchGuardReason` | `replay_prefill_decode`, `cpu_weights`, `no_gpu`, `no_gpu_sampling`, `command_batching_disabled`, `sampling_suppression_requires_cpu_logits`, `token_constraint_requires_per_token_logits`, `batch_size_1`, `bdpa_paged_layout`, `finiteness_fallback_window`, `linear_attention_layers` | the same words joined with `-` (`multi-token-decode-disabled` was already kebab-case) |
| Training backends (`TRAINING_BACKENDS`, `backend`) and the Capsule adapter format | `webgpu_native`, `peft_safetensors` | `webgpu-native`, `peft-safetensors` |
| Execution plan id | `finiteness_fallback` | `finiteness-fallback` |
| Shard loader `lastSource.fallback` (output only) | `custom_range_unavailable`, `custom_range_not_supported`, `custom_stream_not_supported[_resume]`, `custom_stream_interrupted[_resume]`, `custom_stream_partial_resume`, `custom_range_partial_retry` | the same words joined with `-` |
| Manifest per-layer inputs, dtype transitions, layer patterns, FFN branch mode, INT4 encodings, lowering classes, training stages and objectives, bench lanes, precision replay modes, operator events, support status | snake_case | kebab-case; the complete list is the table in [`src/config/legacy-enum-values.js`](../../src/config/legacy-enum-values.js) |
| Trainer handoff format | `huggingface_safetensors` | `huggingface-safetensors` |
| Translation artifact lane | `selective_f16` | `selective-f16` |

The TypeScript declarations changed the same way: for example `FailureClass`,
`KVLayout`, `KVTieringMode`, `ToolingDiagnosticsMode`, `DiffusionSchedulerType`
and `StructuredJsonHeadModelType` now list only the kebab-case values, so a
comparison against an old literal fails to type-check.

### Every other emitted value

The emitted-value audit (`node tools/list-emitted-enum-values.js`) lists every
snake_case string that packaged source writes into an outcome field. Each value
below was converted to kebab-case (the same words joined with `-`) and is still
read in its 0.1.7 spelling. The complete list, with the fields and files of each
value, is the checked fixture `tests/fixtures/emitted-enum-values.json`
(355 values); `tests/integration/emitted-enum-values.test.js` fails if
source or a public entry point emits a snake_case value again.

| Where the value appears | 0.1.7 spelling (still read) | 0.2.0 |
| --- | --- | --- |
| Runtime results, errors and policies (reasons, codes, kinds, types, modes) | `abs_exceeded`, `accumulated_drift_exceeded`, `amplification_detected`, `both_exceeded`, `device_lost_handler_failed`, `device_unavailable_during_weight_load`, `encoder_kv_readonly_canvas_concat`, `gpu_resident_embedding_exceeds_device_limit`, `incomplete_or_nonfinite_output`, `missing_attempt`, `no_data`, `no_policy`, `path_override_adaptive_prefill`, `peft_safetensors`, `rel_exceeded`, `within_bound` | the same words joined with `-` |
| Tooling, CLI, harness and diagnostics codes, stages, ids and types | `activation_dtype_drift`, `baseline_verification_failed`, `benchmark_baseline_load_failed`, `bootstrap_check_in`, `both_failed`, `bundle_export_failed`, `candidate_parity_failed`, `candidate_unavailable`, `candidate_variance_above_threshold`, `candidate_verification_failed`, `capability_drift`, `capture_failed`, `capture_precondition_node_webgpu_unavailable`, `capture_precondition_playwright_chromium_missing`, `capture_precondition_playwright_missing`, `capture_precondition_unsupported_surface`, `command_record`, `confidence_interval_below_threshold`, `convert_failed`, `custom_runtime_bypass`, `diffusion_gemma_canvas_denoising_phase`, `diffusion_gemma_encoder_prefill_phase`, `dtype_mismatch`, `execution_contract_check`, `f16_or_f32`, `full_tensor_digest_mismatch`, `gpu_compute`, `image_to_text`, `improvement_below_threshold`, `insufficient_valid_pairs`, `kv_dtype_drift`, `layer_override_drift`, `manifest_load`, `manifest_unreadable`, `missing_model_id`, `missing_runtime_boundary`, `missing_sample`, `missing_sampling`, `neighbor_parity_failed`, `neighbor_regression_exceeded`, `neighbor_verification_failed`, `neighboring_workload_guard_failed`, `no_manifest_path`, `not_applicable_training_workload`, `not_requested`, `output_dtype_drift`, `parity_check_failed`, `performance_evidence_failed`, `primary_candidate_rejected`, `readback_cleanup`, `readback_copy`, `readback_map_wait`, `runtime_config_fetch_failed`, `runtime_config_not_found`, `sample_tolerance_exceeded`, `sequential_evidence_below_threshold`, `shape_mismatch`, `skip_capture_requires_artifacts`, `submit_readback_unattributed`, `submit_readback_wait`, `tooling_error`, `training_surface_downgrade_blocked`, `unsupported_op_or_contract_violation`, `unsupported_workload` | the same words joined with `-` |
| Training, evaluation and adapter vocabularies (backends, runner and dataset kinds, artifact types, metrics alerts, categories, check names, reward ids, error codes) | `address_space`, `agent_heldout_eval`, `attribute_name`, `base_model`, `base_model_not_registered_for_current_lora_runner`, `baseline_quality_claims`, `builtin_name`, `candidate_present`, `causal_lm_lora`, `causal_lm_text_generation`, `causal_lm_text_pairs`, `causal_lm_trainer_not_configured`, `compile_pass`, `compiler_qualified_reference_vs_modal_all_fail_rollout_v1`, `compiler_qualified_task_reference`, `completion_masked_sft_lora`, `completion_only_causal_lm`, `contract_pass`, `dataset_format_must_be_text_pairs`, `dataset_format_not_supported_by_current_lora_runner`, `dataset_order`, `declared_type`, `distill_kd`, `distill_stage_a`, `distill_stage_manifest`, `distill_triplet`, `doppler_teacher_trace_v1`, `empty_response`, `eval_reports`, `exact_reference_match`, `external_training_backend_not_configured`, `f16_rounded_values_promoted_to_f32_compute`, `forbidden_term`, `function_name`, `identifier_reference`, `injected_causal_lm_lora_trainer`, `js_patching`, `log_snr_cosine`, `log_snr_linear`, `lora_adapter_manifest`, `manifest_config_review`, `markdown_fence`, `max_gradient_norm_exceeded`, `max_inf_count_exceeded`, `max_nan_count_exceeded`, `max_saturation_count_exceeded`, `max_step_time_ms_exceeded`, `min_effective_lr_below_threshold`, `missing_semicolon`, `missing_timeline`, `modal_all_fail_on_policy_completion`, `native_full_graph_runner_unavailable_for_base_model`, `native_lora_target_not_supported`, `native_qwen_completion_masked_sft`, `native_qwen_lora_checkpoint_tensor_mismatch`, `native_qwen_lora_checkpoint_tensor_size_mismatch`, `native_qwen_lora_missing_weight`, `native_qwen_lora_model_not_supported`, `native_qwen_lora_pipeline_required`, `native_qwen_lora_supervised_token_count_mismatch`, `native_training_webgpu_bootstrap_failed`, `no_adapter_candidate`, `no_adapter_payload`, `no_hallucinated_files`, `no_hallucinated_files_tools`, `no_hallucinated_tools`, `no_model_loaded`, `numeric_drift`, `numeric_literal_suffix`, `operator_diff`, `patch_applies`, `policy_pass`, `preserve_completion_head_tail_prompt`, `prose_or_diff_wrapper`, `qwen_adapter_training_state`, `qwen_sft_backend_parity_fixture`, `rdrr_lora_adapter`, `regression_pass`, `replacement_only_wgsl_span_v1`, `reploid_vfs_status_tool_loop`, `required_improvement_claims`, `required_term`, `response_not_string`, `runner_combination_not_supported_by_current_lora_runner`, `seed_hash_sorted_v1`, `storage_access`, `subset_manifest`, `task_type_must_be_text_generation`, `teacher_top_token`, `teacher_trace`, `text_generation`, `topk_class`, `toy_linear_classification`, `toy_linear_classification_jsonl`, `training_backend_invalid`, `training_backend_mismatch`, `training_backend_not_supported`, `training_checkpoint`, `training_checkpoint_watch_manifest`, `training_compare_report`, `training_eval_report`, `training_fixture`, `training_policy_checkpoint`, `training_policy_update`, `training_promotion_decision`, `training_quality_gate`, `training_reward_vector`, `training_rollout_group`, `training_run_contract`, `training_scoreboard`, `training_verifier_report`, `training_workload_lock`, `ul_stage1`, `webgpu_native`, `wgsl_lexicographic_compile_regression_v1`, `wgsl_review`, `within_tolerance`, `workload_kind_must_be_lora`, `zero_advantages` | the same words joined with `-` |
| Trainer handoff, parity and translation-competition blockers, check ids, profiles, statuses and decisions | `architecture_and_conversion_lineage`, `architecture_mismatch`, `artifact_comparison_allowed`, `artifact_generation_allowed`, `artifact_lane_evidence_incomplete`, `artifact_lane_selection_not_unique`, `blocked_awaiting_gamma_selection`, `browser_capability_evaluation`, `byte_length_mismatch`, `columbo_qwen_adapter`, `competition_policy_not_bound_to_gamma_selected_source`, `config_file_identity_failed`, `config_file_role_missing`, `conversion_source_checkpoint_hash_mismatch`, `doppler_f16_vs_quantized_browser`, `duplicate_parity_check`, `evaluation_input_identity`, `exact_browser_artifact_capability`, `first_token_logits`, `full_checkpoint`, `gamma_bf16_selection_receipt_absent`, `gamma_campaign_blocked`, `gamma_retention_floor`, `gamma_sealed_task_gain`, `gamma_selected_checkpoint_binding_mismatch`, `gamma_selected_handoff_absent`, `gamma_selected_handoff_sha256_absent`, `gamma_selected_handoff_sha256_mismatch`, `gamma_selection_receipt_binding_mismatch`, `gamma_source_not_selected_candidate`, `gamma_source_selection_contract_invalid`, `invalid_parity_evidence_hash`, `invalid_parity_evidence_schema`, `missing_parity_check`, `not_selected`, `parity_artifact_identity_mismatch`, `parity_bridge_id_mismatch`, `parity_check_identity_mismatch`, `parity_check_missing_id`, `parity_check_not_passed`, `parity_profile_mismatch`, `parity_upstream_not_passed`, `peft_adapter`, `peft_reference_vs_doppler_f16`, `promotion_submission_allowed`, `recorded_import_receipt_candidate_gate_mismatch`, `recorded_import_receipt_file_identity_failed`, `recorded_import_receipt_hash_binding_mismatch`, `recorded_import_receipt_identity`, `recorded_import_receipt_promotion_gate_mismatch`, `reference_converted_adapter_vs_doppler_adapter`, `reference_converted_base_vs_doppler_base`, `selected_source_identity_bridge_mismatch`, `selected_source_identity_receipt_binding_mismatch`, `selected_source_identity_role_mismatch`, `selected_source_identity_verification_absent`, `selected_source_identity_verification_absent_or_invalid`, `selected_source_identity_verification_failed`, `selected_tokens`, `sha256_mismatch`, `source_adapter_vs_converted_adapter`, `source_artifact_byte_identity`, `source_base_vs_converted_base`, `tinker_checkpoint_vs_peft_reference`, `tinker_peft_browser_adapter`, `tokenizer_and_prompt_identity`, `translation_full_checkpoint`, `unexpected_parity_check` | the same words joined with `-` |
| Storage, distribution, P2P and hot-swap codes, reasons, policy reasons and breach ids | `availability_breach`, `cache_disabled`, `cache_miss`, `canary_excluded`, `canary_selected`, `control_plane_policy_evaluator`, `control_plane_token_provider`, `default_enabled`, `descriptor_hash_mismatch`, `descriptor_shards_missing`, `file_done`, `file_progress`, `file_start`, `force_enable`, `hash_mismatch`, `hash_missing`, `hotswap_disabled`, `http_content_length_invalid`, `http_content_length_mismatch`, `http_content_range_invalid`, `http_content_range_invalid_span`, `http_content_range_invalid_total`, `http_content_range_length_mismatch`, `http_content_range_missing`, `http_content_range_total_mismatch`, `http_content_range_unexpected_status`, `http_disabled`, `http_fallback_rate_breach`, `http_header_length_mismatch`, `http_resume_offset_mismatch`, `http_resume_response_mismatch`, `insufficient_vram`, `latency_p95_breach`, `manifest_version_set_mismatch`, `manifest_version_set_missing`, `not_cached`, `opt_in_required`, `opt_in_selected`, `p2p_disabled`, `p2p_hit_rate_breach`, `p2p_payload_exceeds_total`, `p2p_payload_range_start_invalid`, `p2p_payload_size_invalid`, `p2p_payload_size_mismatch`, `p2p_range_start_invalid`, `p2p_resume_offset_mismatch`, `p2p_total_size_mismatch`, `p2p_transport_unconfigured`, `peer_unavailable`, `policy_denied`, `policy_denied_control_plane`, `rate_limited`, `resume_state_oversize`, `resume_state_prefix_mismatch`, `session_token_expired`, `session_token_missing`, `session_token_missing_after_refresh`, `shadow_mode`, `shard_hash_mismatch`, `size_mismatch`, `stream_read_failed`, `transport_quarantined`, `unknown_breach` | the same words joined with `-` |
| Decode stats (`decodeMode`, `batchGuardReason`, `stopReason`) | `batch_size_1`, `bdpa_paged_layout`, `command_batching_disabled`, `cpu_weights`, `finiteness_fallback_window`, `linear_attention_layers`, `max_new_tokens`, `no_gpu`, `no_gpu_sampling`, `replay_prefill_decode`, `sampling_suppression_requires_cpu_logits`, `target_verified_lookup`, `token_constraint_requires_per_token_logits` | the same words joined with `-` |
| Kernel and attention selection reasons | `decode_streaming_fallback`, `path_override`, `path_override_adaptive_fallback`, `path_override_f16_fallthrough`, `path_override_fallback`, `path_override_quantized_fallthrough`, `streaming_fallback`, `subgroup_capable`, `tiled_large_capable`, `tiled_small_capable` | the same words joined with `-` |
| Receipt and trace versions | `doppler_provider_receipt_v1`, `doppler_serve_receipt_v2` | the same words joined with `-` |

Codes that carry a detail suffix change only their code part, for example
`missing-parity-check:<check id>` and `architecture-mismatch:<field>`.

Values that did **not** change are identifiers, not Doppler enum output. Each
category is listed with its values in `tools/policies/emitted-enum-values.json`:

- Execution-graph op ids, tensor roles, kernel and loader variant names and probe
  stage names (`q_proj`, `input_norm`, `logits_final`, `q4k_fused`). They mirror
  Hugging Face module names or the kernel registry and are bound into published
  manifests and `executionGraphHash` digests.
- Values written into manifests, functional descriptors and artifact identities
  whose bytes are hashed (for example `per_head`, `tokenizer_json`,
  `safetensors_index`).
- The peer-to-peer shard protocol message types (`doppler_p2p_shard_request`):
  peers on other versions send and expect these exact strings.
- GPU debug labels, object keys referenced as strings (metric and config field
  names such as `exact_match`), and third-party names Doppler passes through
  (Hugging Face model and layer types, GGUF dtypes, OpenAI error types, PEFT and
  diffusers file names).

## Supported compatibility reads

Doppler still **reads** the old spellings. Every reader below maps a legacy
value to its kebab-case form before validation, so the code behind it only sees
the new spelling. Unknown values are not rewritten; the owning validator still
rejects them.

| Input | Reader | Kept |
| --- | --- | --- |
| RDRR manifests (`manifest.json`), including published, digest-bound ones | `parseManifest` (`src/formats/rdrr/parsing.js`) | throughout 0.x; their bytes and hashes are not rewritten |
| Runtime config overrides: `loadOptions.runtimeConfig`, `setRuntimeConfig`, `resolveRuntimeConfig`, runtime profiles and every runtime-config merge | `resolveRuntimeConfig` and `mergeRuntimeValues` (`src/config/runtime.js`, `src/config/runtime-merge.js`) | throughout 0.x |
| Provider `policy.fallbackOn` | `normalizeFailureClass` (`src/client/failure-taxonomy.js`); the type accepts `LegacyFailureClass` | throughout 0.x |
| Pipeline `modelType` for structured heads | `normalizeModelType` (`src/inference/pipelines/registry.js`) | throughout 0.x |
| Diffusion scheduler `type` | `assertSupportedDiffusionSchedulerType` and the diffusion init path | throughout 0.x |
| Command requests: training stage | `src/tooling/command-request-normalization.js` | throughout 0.x |
| Training workload files (`distill.studentGraphMode`, `stagePlan[].id`) | `src/experimental/training/workloads.js` | throughout 0.x |
| Trainer handoff descriptors (`artifact.format`, including Gamma's `huggingface_safetensors`) | `src/experimental/bridge/trainer-artifact-bridge.js` | throughout 0.x |
| Trainer handoff descriptors and parity evidence: `selection.status`, `artifact.kind`, `parity.profile`, `requiredChecks`, check ids | `normalizeDescriptor`, `verifyTrainerArtifactParityEvidence` (`src/experimental/bridge/trainer-artifact-bridge.js`) | throughout 0.x |
| Training API and workloads: `backend` (`webgpu_native`), `taskType`, `evalKind`, `datasetFormat`, `datasetKind`, `runnerKind`, `rowOrder`, `targetTokenMode`, `baseline` and the other outcome fields | `assertTrainingBackend`, `trainSftLoRA`, `getTrainingCapabilities` (`src/training.js`), `normalizeTrainingWorkloadCapsule` and `loadTrainingWorkloadCapsule` | throughout 0.x |
| Recorded training artifacts and reports: `artifactType` | `validateVerifierGuidedArtifact`, the eval-report readers in `lora-pipeline.js` and `operator-command.js`, `qwen-adapter-training-state.js` | throughout 0.x |
| Agent-evaluation dataset and policy categories | `evaluateAgentHeldoutRows` (`src/experimental/training/operator-agent-eval.js`) | throughout 0.x |
| UL noise schedule `type` (`log_snr_linear`, `log_snr_cosine`) | `resolveUlScheduledLambda` | throughout 0.x |
| Translation competition policy (`state`, `blockers`) and recorded receipts | `evaluateTranslationArtifactCompetition`, `readTranslationArtifactCompetitionReceipt` | throughout 0.x; the policy file and recorded receipts keep their bytes and hashes |
| Capsule adapter format (`peft_safetensors`) in signed TargetPlans and adapter entries | `validateCapsuleAdapterExecution`, `resolveCapsuleAdapterSet`, `resolveLoRAFormatLayout` | throughout 0.x; signed documents are not rewritten |
| Diffusion-gemma `decoderCacheMode` in manifests and conversion configs | `parseManifest`, `parseDiffusionGemmaConfig` | throughout 0.x |
| Error codes thrown by a P2P transport written against 0.1.x (`p2p_*`, `p2p_unconfigured`) | `src/storage/distribution-transport.js`, `src/experimental/distribution/shard-delivery/receipt.js` | throughout 0.x |
| The recorded 2026-07-13 translation readiness receipt | `readTranslationArtifactCompetitionReceipt` verifies it with its original spelling and hash and returns kebab-case lanes | indefinitely; the receipt is historical evidence |

These reads follow the compatibility rule: legacy spellings are read throughout
0.x; aliases are removed no earlier than 1.0, and only with at least one minor
release of notice in the CHANGELOG. Every value converted by the emitted-value
audit is read the same way: wherever Doppler takes it back as input (runtime
config, manifests, command requests, training workloads, recorded receipts and
reports, trainer handoffs and parity evidence, adapter declarations, policy
files) the reader canonicalizes it before validation. No new snake_case value is
added to the alias table except a value already emitted by 0.1.7
(`tools/check-kebab-enum-values.js`).

Reads are not writes. Doppler never echoes a legacy spelling back: a receipt,
stats object or result produced from a legacy input carries the kebab-case
value. A consumer that compares Doppler output against values it stored itself
before 0.2.0 must map its own stored values.

Tests: `tests/integration/legacy-enum-reads.test.js` (runtime config, manifest,
provider, pipeline and scheduler reads), `tests/integration/legacy-outcome-reads.test.js`
(every converted value at config input, training, artifact, agent-evaluation and
diffusion-gemma reads), `tests/integration/emitted-enum-values.test.js` (no
snake_case value emitted, by source audit and by running public entry points on
CPU), `tests/integration/emitted-decode-stats-kebab-physical.test.js` (decode
stats on a real GPU), `tests/integration/failure-taxonomy.test.js` and
`tests/tooling/check-kebab-enum-values.test.js`.
