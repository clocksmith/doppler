# Doppler Dependency Graph

## Entry Points
- `src/index.ts`

## Graph (Mermaid)
```mermaid
graph TD
  subgraph app
    node_app_app_ts["app/app.ts"]
    node_app_chat_ui_ts["app/chat-ui.ts"]
    node_app_model_selector_ts["app/model-selector.ts"]
    node_app_progress_ui_ts["app/progress-ui.ts"]
    node_app_quickstart_ui_ts["app/quickstart-ui.ts"]
  end
  subgraph cli
    node_cli_commands_serve_ts["cli/commands/serve.ts"]
    node_cli_helpers_comparison_ts["cli/helpers/comparison.ts"]
    node_cli_helpers_html_report_ts["cli/helpers/html-report.ts"]
    node_cli_helpers_index_ts["cli/helpers/index.ts"]
    node_cli_helpers_inference_benchmark_ts["cli/helpers/inference-benchmark.ts"]
    node_cli_helpers_types_ts["cli/helpers/types.ts"]
    node_cli_helpers_utils_ts["cli/helpers/utils.ts"]
    node_cli_index_ts["cli/index.ts"]
  end
  subgraph client
    node_client_doppler_provider_ts["client/doppler-provider.ts"]
  end
  subgraph kernel-tests
    node_kernel_tests_browser_test_page_js["kernel-tests/browser/test-page.js"]
    node_kernel_tests_browser_test_page_ts["kernel-tests/browser/test-page.ts"]
    node_kernel_tests_scripts_ci_report_js["kernel-tests/scripts/ci-report.js"]
    node_kernel_tests_src_harness_benchmark_ts["kernel-tests/src/harness/benchmark.ts"]
    node_kernel_tests_src_harness_buffer_utils_ts["kernel-tests/src/harness/buffer-utils.ts"]
    node_kernel_tests_src_harness_index_ts["kernel-tests/src/harness/index.ts"]
    node_kernel_tests_src_harness_tolerance_ts["kernel-tests/src/harness/tolerance.ts"]
    node_kernel_tests_src_reference_attention_ts["kernel-tests/src/reference/attention.ts"]
    node_kernel_tests_src_reference_dequant_ts["kernel-tests/src/reference/dequant.ts"]
    node_kernel_tests_src_reference_gather_ts["kernel-tests/src/reference/gather.ts"]
    node_kernel_tests_src_reference_index_ts["kernel-tests/src/reference/index.ts"]
    node_kernel_tests_src_reference_matmul_ts["kernel-tests/src/reference/matmul.ts"]
    node_kernel_tests_src_reference_moe_gather_ts["kernel-tests/src/reference/moe-gather.ts"]
    node_kernel_tests_src_reference_residual_ts["kernel-tests/src/reference/residual.ts"]
    node_kernel_tests_src_reference_rmsnorm_ts["kernel-tests/src/reference/rmsnorm.ts"]
    node_kernel_tests_src_reference_rope_ts["kernel-tests/src/reference/rope.ts"]
    node_kernel_tests_src_reference_sample_ts["kernel-tests/src/reference/sample.ts"]
    node_kernel_tests_src_reference_scatter_add_ts["kernel-tests/src/reference/scatter-add.ts"]
    node_kernel_tests_src_reference_silu_ts["kernel-tests/src/reference/silu.ts"]
    node_kernel_tests_src_reference_softmax_ts["kernel-tests/src/reference/softmax.ts"]
    node_kernel_tests_src_reference_topk_ts["kernel-tests/src/reference/topk.ts"]
    node_kernel_tests_tests_benchmarks_config_ts["kernel-tests/tests/benchmarks/config.ts"]
    node_kernel_tests_tests_correctness_setup_ts["kernel-tests/tests/correctness/setup.ts"]
  end
  subgraph Root
    node_serve_ts["serve.ts"]
  end
  subgraph src
    node_src_adapters_adapter_manager_ts["src/adapters/adapter-manager.ts"]
    node_src_adapters_adapter_manifest_ts["src/adapters/adapter-manifest.ts"]
    node_src_adapters_adapter_registry_ts["src/adapters/adapter-registry.ts"]
    node_src_adapters_index_ts["src/adapters/index.ts"]
    node_src_adapters_lora_loader_ts["src/adapters/lora-loader.ts"]
    node_src_bridge_extension_client_ts["src/bridge/extension-client.ts"]
    node_src_bridge_extension_background_ts["src/bridge/extension/background.ts"]
    node_src_bridge_index_ts["src/bridge/index.ts"]
    node_src_bridge_native_native_host_ts["src/bridge/native/native-host.ts"]
    node_src_bridge_protocol_ts["src/bridge/protocol.ts"]
    node_src_browser_browser_converter_ts["src/browser/browser-converter.ts"]
    node_src_browser_file_picker_ts["src/browser/file-picker.ts"]
    node_src_browser_gguf_importer_ts["src/browser/gguf-importer.ts"]
    node_src_browser_gguf_parser_browser_ts["src/browser/gguf-parser-browser.ts"]
    node_src_browser_safetensors_parser_browser_ts["src/browser/safetensors-parser-browser.ts"]
    node_src_browser_shard_io_browser_ts["src/browser/shard-io-browser.ts"]
    node_src_config_index_ts["src/config/index.ts"]
    node_src_config_kernels_registry_js["src/config/kernels/registry.js"]
    node_src_config_kernels_selection_js["src/config/kernels/selection.js"]
    node_src_config_loader_ts["src/config/loader.ts"]
    node_src_config_platforms_loader_js["src/config/platforms/loader.js"]
    node_src_config_schema_bridge_schema_ts["src/config/schema/bridge.schema.ts"]
    node_src_config_schema_buffer_pool_schema_ts["src/config/schema/buffer-pool.schema.ts"]
    node_src_config_schema_conversion_schema_ts["src/config/schema/conversion.schema.ts"]
    node_src_config_schema_debug_schema_ts["src/config/schema/debug.schema.ts"]
    node_src_config_schema_distribution_schema_ts["src/config/schema/distribution.schema.ts"]
    node_src_config_schema_doppler_schema_ts["src/config/schema/doppler.schema.ts"]
    node_src_config_schema_gpu_cache_schema_ts["src/config/schema/gpu-cache.schema.ts"]
    node_src_config_schema_index_ts["src/config/schema/index.ts"]
    node_src_config_schema_inference_defaults_schema_ts["src/config/schema/inference-defaults.schema.ts"]
    node_src_config_schema_inference_schema_ts["src/config/schema/inference.schema.ts"]
    node_src_config_schema_kernel_registry_schema_ts["src/config/schema/kernel-registry.schema.ts"]
    node_src_config_schema_kvcache_schema_ts["src/config/schema/kvcache.schema.ts"]
    node_src_config_schema_loading_schema_ts["src/config/schema/loading.schema.ts"]
    node_src_config_schema_manifest_schema_ts["src/config/schema/manifest.schema.ts"]
    node_src_config_schema_memory_limits_schema_ts["src/config/schema/memory-limits.schema.ts"]
    node_src_config_schema_moe_schema_ts["src/config/schema/moe.schema.ts"]
    node_src_config_schema_platform_schema_ts["src/config/schema/platform.schema.ts"]
    node_src_config_schema_preset_schema_ts["src/config/schema/preset.schema.ts"]
    node_src_config_schema_storage_schema_ts["src/config/schema/storage.schema.ts"]
    node_src_config_schema_tuner_schema_ts["src/config/schema/tuner.schema.ts"]
    node_src_converter_core_ts["src/converter/core.ts"]
    node_src_converter_index_ts["src/converter/index.ts"]
    node_src_converter_io_node_ts["src/converter/io/node.ts"]
    node_src_converter_node_converter_ts["src/converter/node-converter.ts"]
    node_src_converter_quantizer_ts["src/converter/quantizer.ts"]
    node_src_converter_shard_packer_ts["src/converter/shard-packer.ts"]
    node_src_converter_test_model_ts["src/converter/test-model.ts"]
    node_src_converter_writer_ts["src/converter/writer.ts"]
    node_src_debug_diagnose_kernels_ts["src/debug/diagnose-kernels.ts"]
    node_src_debug_index_ts["src/debug/index.ts"]
    node_src_debug_tensor_ts["src/debug/tensor.ts"]
    node_src_formats_gguf_ts["src/formats/gguf.ts"]
    node_src_formats_gguf_index_ts["src/formats/gguf/index.ts"]
    node_src_formats_gguf_parser_ts["src/formats/gguf/parser.ts"]
    node_src_formats_gguf_types_ts["src/formats/gguf/types.ts"]
    node_src_formats_index_ts["src/formats/index.ts"]
    node_src_formats_rdrr_classification_ts["src/formats/rdrr/classification.ts"]
    node_src_formats_rdrr_groups_ts["src/formats/rdrr/groups.ts"]
    node_src_formats_rdrr_index_ts["src/formats/rdrr/index.ts"]
    node_src_formats_rdrr_manifest_ts["src/formats/rdrr/manifest.ts"]
    node_src_formats_rdrr_parsing_ts["src/formats/rdrr/parsing.ts"]
    node_src_formats_rdrr_types_ts["src/formats/rdrr/types.ts"]
    node_src_formats_rdrr_validation_ts["src/formats/rdrr/validation.ts"]
    node_src_formats_safetensors_ts["src/formats/safetensors.ts"]
    node_src_formats_safetensors_index_ts["src/formats/safetensors/index.ts"]
    node_src_formats_safetensors_parser_ts["src/formats/safetensors/parser.ts"]
    node_src_formats_safetensors_types_ts["src/formats/safetensors/types.ts"]
    node_src_formats_tokenizer_ts["src/formats/tokenizer.ts"]
    node_src_formats_tokenizer_index_ts["src/formats/tokenizer/index.ts"]
    node_src_formats_tokenizer_types_ts["src/formats/tokenizer/types.ts"]
    node_src_gpu_buffer_dtypes_ts["src/gpu/buffer-dtypes.ts"]
    node_src_gpu_buffer_pool_ts["src/gpu/buffer-pool.ts"]
    node_src_gpu_command_recorder_ts["src/gpu/command-recorder.ts"]
    node_src_gpu_device_ts["src/gpu/device.ts"]
    node_src_gpu_kernel_benchmark_ts["src/gpu/kernel-benchmark.ts"]
    node_src_gpu_kernel_hints_ts["src/gpu/kernel-hints.ts"]
    node_src_gpu_kernel_runtime_ts["src/gpu/kernel-runtime.ts"]
    node_src_gpu_kernel_selection_cache_ts["src/gpu/kernel-selection-cache.ts"]
    node_src_gpu_kernel_selector_ts["src/gpu/kernel-selector.ts"]
    node_src_gpu_kernel_tuner_ts["src/gpu/kernel-tuner.ts"]
    node_src_gpu_kernels_attention_ts["src/gpu/kernels/attention.ts"]
    node_src_gpu_kernels_cast_ts["src/gpu/kernels/cast.ts"]
    node_src_gpu_kernels_check_stop_ts["src/gpu/kernels/check-stop.ts"]
    node_src_gpu_kernels_constants_ts["src/gpu/kernels/constants.ts"]
    node_src_gpu_kernels_dequant_ts["src/gpu/kernels/dequant.ts"]
    node_src_gpu_kernels_dispatch_ts["src/gpu/kernels/dispatch.ts"]
    node_src_gpu_kernels_fused_ffn_ts["src/gpu/kernels/fused_ffn.ts"]
    node_src_gpu_kernels_fused_matmul_residual_ts["src/gpu/kernels/fused_matmul_residual.ts"]
    node_src_gpu_kernels_fused_matmul_rmsnorm_ts["src/gpu/kernels/fused_matmul_rmsnorm.ts"]
    node_src_gpu_kernels_gather_ts["src/gpu/kernels/gather.ts"]
    node_src_gpu_kernels_gelu_ts["src/gpu/kernels/gelu.ts"]
    node_src_gpu_kernels_index_ts["src/gpu/kernels/index.ts"]
    node_src_gpu_kernels_kernel_base_ts["src/gpu/kernels/kernel-base.ts"]
    node_src_gpu_kernels_matmul_ts["src/gpu/kernels/matmul.ts"]
    node_src_gpu_kernels_moe_ts["src/gpu/kernels/moe.ts"]
    node_src_gpu_kernels_residual_ts["src/gpu/kernels/residual.ts"]
    node_src_gpu_kernels_rmsnorm_ts["src/gpu/kernels/rmsnorm.ts"]
    node_src_gpu_kernels_rope_ts["src/gpu/kernels/rope.ts"]
    node_src_gpu_kernels_sample_ts["src/gpu/kernels/sample.ts"]
    node_src_gpu_kernels_scale_ts["src/gpu/kernels/scale.ts"]
    node_src_gpu_kernels_silu_ts["src/gpu/kernels/silu.ts"]
    node_src_gpu_kernels_softmax_ts["src/gpu/kernels/softmax.ts"]
    node_src_gpu_kernels_split_qkv_ts["src/gpu/kernels/split_qkv.ts"]
    node_src_gpu_kernels_types_ts["src/gpu/kernels/types.ts"]
    node_src_gpu_kernels_utils_ts["src/gpu/kernels/utils.ts"]
    node_src_gpu_multi_model_recorder_ts["src/gpu/multi-model-recorder.ts"]
    node_src_gpu_partitioned_buffer_pool_ts["src/gpu/partitioned-buffer-pool.ts"]
    node_src_gpu_perf_guards_ts["src/gpu/perf-guards.ts"]
    node_src_gpu_perf_profiler_ts["src/gpu/perf-profiler.ts"]
    node_src_gpu_profiler_ts["src/gpu/profiler.ts"]
    node_src_gpu_submit_tracker_ts["src/gpu/submit-tracker.ts"]
    node_src_gpu_uniform_cache_ts["src/gpu/uniform-cache.ts"]
    node_src_index_ts["src/index.ts"]
    node_src_inference_decode_buffers_ts["src/inference/decode-buffers.ts"]
    node_src_inference_expert_router_ts["src/inference/expert-router.ts"]
    node_src_inference_functiongemma_ts["src/inference/functiongemma.ts"]
    node_src_inference_kv_cache_ts["src/inference/kv-cache.ts"]
    node_src_inference_moe_router_ts["src/inference/moe-router.ts"]
    node_src_inference_multi_model_network_ts["src/inference/multi-model-network.ts"]
    node_src_inference_multi_pipeline_pool_ts["src/inference/multi-pipeline-pool.ts"]
    node_src_inference_network_evolution_ts["src/inference/network-evolution.ts"]
    node_src_inference_pipeline_ts["src/inference/pipeline.ts"]
    node_src_inference_pipeline_attention_ts["src/inference/pipeline/attention.ts"]
    node_src_inference_pipeline_buffer_types_ts["src/inference/pipeline/buffer-types.ts"]
    node_src_inference_pipeline_config_ts["src/inference/pipeline/config.ts"]
    node_src_inference_pipeline_debug_utils_ts["src/inference/pipeline/debug-utils.ts"]
    node_src_inference_pipeline_embed_ts["src/inference/pipeline/embed.ts"]
    node_src_inference_pipeline_init_ts["src/inference/pipeline/init.ts"]
    node_src_inference_pipeline_kernel_trace_ts["src/inference/pipeline/kernel-trace.ts"]
    node_src_inference_pipeline_layer_ts["src/inference/pipeline/layer.ts"]
    node_src_inference_pipeline_logits_ts["src/inference/pipeline/logits.ts"]
    node_src_inference_pipeline_lora_apply_ts["src/inference/pipeline/lora-apply.ts"]
    node_src_inference_pipeline_lora_types_ts["src/inference/pipeline/lora-types.ts"]
    node_src_inference_pipeline_lora_ts["src/inference/pipeline/lora.ts"]
    node_src_inference_pipeline_moe_impl_ts["src/inference/pipeline/moe-impl.ts"]
    node_src_inference_pipeline_sampling_ts["src/inference/pipeline/sampling.ts"]
    node_src_inference_pipeline_types_ts["src/inference/pipeline/types.ts"]
    node_src_inference_pipeline_weights_ts["src/inference/pipeline/weights.ts"]
    node_src_inference_speculative_ts["src/inference/speculative.ts"]
    node_src_inference_test_harness_ts["src/inference/test-harness.ts"]
    node_src_inference_tokenizer_ts["src/inference/tokenizer.ts"]
    node_src_loader_doppler_loader_ts["src/loader/doppler-loader.ts"]
    node_src_loader_dtype_utils_ts["src/loader/dtype-utils.ts"]
    node_src_loader_expert_cache_ts["src/loader/expert-cache.ts"]
    node_src_loader_loader_types_ts["src/loader/loader-types.ts"]
    node_src_loader_multi_model_loader_ts["src/loader/multi-model-loader.ts"]
    node_src_loader_shard_cache_ts["src/loader/shard-cache.ts"]
    node_src_loader_weights_ts["src/loader/weights.ts"]
    node_src_memory_address_table_ts["src/memory/address-table.ts"]
    node_src_memory_capability_ts["src/memory/capability.ts"]
    node_src_memory_heap_manager_ts["src/memory/heap-manager.ts"]
    node_src_memory_unified_detect_ts["src/memory/unified-detect.ts"]
    node_src_storage_download_types_ts["src/storage/download-types.ts"]
    node_src_storage_downloader_ts["src/storage/downloader.ts"]
    node_src_storage_preflight_ts["src/storage/preflight.ts"]
    node_src_storage_quickstart_downloader_ts["src/storage/quickstart-downloader.ts"]
    node_src_storage_quota_ts["src/storage/quota.ts"]
    node_src_storage_rdrr_format_ts["src/storage/rdrr-format.ts"]
    node_src_storage_shard_manager_ts["src/storage/shard-manager.ts"]
    node_src_types_gpu_ts["src/types/gpu.ts"]
    node_src_types_inference_ts["src/types/inference.ts"]
    node_src_types_model_ts["src/types/model.ts"]
  end
  subgraph tests
    node_tests_benchmark_index_js["tests/benchmark/index.js"]
    node_tests_benchmark_index_ts["tests/benchmark/index.ts"]
    node_tests_benchmark_pipeline_benchmark_js["tests/benchmark/pipeline-benchmark.js"]
    node_tests_benchmark_pipeline_benchmark_ts["tests/benchmark/pipeline-benchmark.ts"]
    node_tests_benchmark_prompts_js["tests/benchmark/prompts.js"]
    node_tests_benchmark_prompts_ts["tests/benchmark/prompts.ts"]
    node_tests_benchmark_results_storage_js["tests/benchmark/results-storage.js"]
    node_tests_benchmark_results_storage_ts["tests/benchmark/results-storage.ts"]
    node_tests_benchmark_system_benchmark_js["tests/benchmark/system-benchmark.js"]
    node_tests_benchmark_system_benchmark_ts["tests/benchmark/system-benchmark.ts"]
    node_tests_benchmark_types_js["tests/benchmark/types.js"]
    node_tests_benchmark_types_ts["tests/benchmark/types.ts"]
    node_tests_helpers_console_capture_ts["tests/helpers/console-capture.ts"]
    node_tests_helpers_demo_page_ts["tests/helpers/demo-page.ts"]
    node_tests_helpers_index_ts["tests/helpers/index.ts"]
  end
  subgraph tools
    node_tools_generate_fixture_ts["tools/generate-fixture.ts"]
    node_tools_purge_opfs_ts["tools/purge-opfs.ts"]
    node_tools_test_query_ts["tools/test-query.ts"]
    node_tools_update_manifest_ts["tools/update-manifest.ts"]
  end
  node_app_app_ts --> node_app_chat_ui_ts
  node_app_app_ts --> node_app_model_selector_ts
  node_app_app_ts --> node_app_progress_ui_ts
  node_app_app_ts --> node_app_quickstart_ui_ts
  node_app_app_ts --> node_src_browser_browser_converter_ts
  node_app_app_ts --> node_src_debug_index_ts
  node_app_app_ts --> node_src_gpu_buffer_pool_ts
  node_app_app_ts --> node_src_gpu_device_ts
  node_app_app_ts --> node_src_inference_pipeline_ts
  node_app_app_ts --> node_src_memory_capability_ts
  node_app_app_ts --> node_src_memory_heap_manager_ts
  node_app_app_ts --> node_src_storage_downloader_ts
  node_app_app_ts --> node_src_storage_quickstart_downloader_ts
  node_app_app_ts --> node_src_storage_rdrr_format_ts
  node_app_app_ts --> node_src_storage_shard_manager_ts
  node_app_quickstart_ui_ts --> node_src_storage_quota_ts
  node_cli_helpers_comparison_ts --> node_cli_helpers_types_ts
  node_cli_helpers_html_report_ts --> node_cli_helpers_comparison_ts
  node_cli_helpers_index_ts --> node_cli_helpers_comparison_ts
  node_cli_helpers_index_ts --> node_cli_helpers_html_report_ts
  node_cli_helpers_index_ts --> node_cli_helpers_inference_benchmark_ts
  node_cli_helpers_index_ts --> node_cli_helpers_types_ts
  node_cli_helpers_index_ts --> node_cli_helpers_utils_ts
  node_cli_helpers_inference_benchmark_ts --> node_cli_helpers_types_ts
  node_cli_helpers_inference_benchmark_ts --> node_cli_helpers_utils_ts
  node_cli_helpers_types_ts --> node_src_storage_rdrr_format_ts
  node_cli_helpers_utils_ts --> node_cli_helpers_types_ts
  node_cli_index_ts --> node_cli_helpers_comparison_ts
  node_cli_index_ts --> node_cli_helpers_html_report_ts
  node_cli_index_ts --> node_cli_helpers_inference_benchmark_ts
  node_cli_index_ts --> node_cli_helpers_types_ts
  node_cli_index_ts --> node_cli_helpers_utils_ts
  node_cli_index_ts --> node_src_storage_rdrr_format_ts
  node_client_doppler_provider_ts --> node_src_adapters_lora_loader_ts
  node_client_doppler_provider_ts --> node_src_bridge_index_ts
  node_client_doppler_provider_ts --> node_src_gpu_device_ts
  node_client_doppler_provider_ts --> node_src_gpu_kernel_runtime_ts
  node_client_doppler_provider_ts --> node_src_inference_pipeline_ts
  node_client_doppler_provider_ts --> node_src_loader_doppler_loader_ts
  node_client_doppler_provider_ts --> node_src_memory_capability_ts
  node_client_doppler_provider_ts --> node_src_memory_heap_manager_ts
  node_client_doppler_provider_ts --> node_src_storage_downloader_ts
  node_client_doppler_provider_ts --> node_src_storage_quota_ts
  node_client_doppler_provider_ts --> node_src_storage_rdrr_format_ts
  node_client_doppler_provider_ts --> node_src_storage_shard_manager_ts
  node_kernel_tests_browser_test_page_ts --> node_kernel_tests_src_harness_benchmark_ts
  node_kernel_tests_browser_test_page_ts --> node_kernel_tests_src_harness_buffer_utils_ts
  node_kernel_tests_browser_test_page_ts --> node_kernel_tests_src_harness_tolerance_ts
  node_kernel_tests_browser_test_page_ts --> node_kernel_tests_src_reference_index_ts
  node_kernel_tests_browser_test_page_ts --> node_src_gpu_buffer_dtypes_ts
  node_kernel_tests_browser_test_page_ts --> node_src_gpu_buffer_pool_ts
  node_kernel_tests_browser_test_page_ts --> node_src_gpu_device_ts
  node_kernel_tests_browser_test_page_ts --> node_src_gpu_kernel_hints_ts
  node_kernel_tests_browser_test_page_ts --> node_src_gpu_kernel_selector_ts
  node_kernel_tests_browser_test_page_ts --> node_src_gpu_kernels_sample_ts
  node_kernel_tests_src_harness_index_ts --> node_kernel_tests_src_harness_benchmark_ts
  node_kernel_tests_src_harness_index_ts --> node_kernel_tests_src_harness_buffer_utils_ts
  node_kernel_tests_src_harness_index_ts --> node_kernel_tests_src_harness_tolerance_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_attention_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_dequant_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_gather_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_matmul_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_moe_gather_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_residual_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_rmsnorm_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_rope_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_sample_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_scatter_add_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_silu_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_softmax_ts
  node_kernel_tests_src_reference_index_ts --> node_kernel_tests_src_reference_topk_ts
  node_src_adapters_adapter_manager_ts --> node_src_adapters_adapter_manifest_ts
  node_src_adapters_adapter_manager_ts --> node_src_adapters_lora_loader_ts
  node_src_adapters_adapter_manager_ts --> node_src_inference_pipeline_lora_types_ts
  node_src_adapters_adapter_manifest_ts --> node_src_inference_pipeline_lora_types_ts
  node_src_adapters_adapter_registry_ts --> node_src_adapters_adapter_manifest_ts
  node_src_adapters_adapter_registry_ts --> node_src_inference_pipeline_lora_types_ts
  node_src_adapters_index_ts --> node_src_adapters_adapter_manager_ts
  node_src_adapters_index_ts --> node_src_adapters_adapter_manifest_ts
  node_src_adapters_index_ts --> node_src_adapters_adapter_registry_ts
  node_src_adapters_index_ts --> node_src_adapters_lora_loader_ts
  node_src_adapters_lora_loader_ts --> node_src_adapters_adapter_manifest_ts
  node_src_adapters_lora_loader_ts --> node_src_inference_pipeline_lora_ts
  node_src_bridge_extension_client_ts --> node_src_bridge_protocol_ts
  node_src_bridge_index_ts --> node_src_bridge_extension_client_ts
  node_src_bridge_index_ts --> node_src_bridge_protocol_ts
  node_src_bridge_native_native_host_ts --> node_src_config_schema_bridge_schema_ts
  node_src_bridge_native_native_host_ts --> node_src_config_schema_distribution_schema_ts
  node_src_browser_browser_converter_ts --> node_src_browser_gguf_parser_browser_ts
  node_src_browser_browser_converter_ts --> node_src_browser_safetensors_parser_browser_ts
  node_src_browser_browser_converter_ts --> node_src_browser_shard_io_browser_ts
  node_src_browser_browser_converter_ts --> node_src_config_index_ts
  node_src_browser_browser_converter_ts --> node_src_converter_core_ts
  node_src_browser_browser_converter_ts --> node_src_converter_shard_packer_ts
  node_src_browser_browser_converter_ts --> node_src_storage_shard_manager_ts
  node_src_browser_gguf_importer_ts --> node_src_browser_file_picker_ts
  node_src_browser_gguf_importer_ts --> node_src_browser_gguf_parser_browser_ts
  node_src_browser_gguf_importer_ts --> node_src_storage_rdrr_format_ts
  node_src_browser_gguf_importer_ts --> node_src_storage_shard_manager_ts
  node_src_browser_gguf_parser_browser_ts --> node_src_formats_gguf_ts
  node_src_browser_safetensors_parser_browser_ts --> node_src_formats_safetensors_ts
  node_src_browser_safetensors_parser_browser_ts --> node_src_formats_tokenizer_ts
  node_src_browser_shard_io_browser_ts --> node_src_converter_shard_packer_ts
  node_src_browser_shard_io_browser_ts --> node_src_storage_rdrr_format_ts
  node_src_config_index_ts --> node_src_config_loader_ts
  node_src_config_index_ts --> node_src_config_schema_index_ts
  node_src_config_kernels_selection_js --> node_src_config_kernels_registry_js
  node_src_config_kernels_selection_js --> node_src_config_platforms_loader_js
  node_src_config_loader_ts --> node_src_config_schema_index_ts
  node_src_config_schema_conversion_schema_ts --> node_src_config_schema_manifest_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_bridge_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_buffer_pool_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_debug_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_distribution_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_gpu_cache_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_inference_defaults_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_kvcache_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_loading_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_memory_limits_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_moe_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_platform_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_preset_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_storage_schema_ts
  node_src_config_schema_doppler_schema_ts --> node_src_config_schema_tuner_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_bridge_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_buffer_pool_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_conversion_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_debug_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_distribution_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_doppler_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_gpu_cache_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_inference_defaults_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_inference_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_kernel_registry_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_kvcache_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_loading_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_manifest_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_memory_limits_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_moe_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_platform_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_preset_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_storage_schema_ts
  node_src_config_schema_index_ts --> node_src_config_schema_tuner_schema_ts
  node_src_config_schema_inference_defaults_schema_ts --> node_src_config_schema_inference_schema_ts
  node_src_config_schema_preset_schema_ts --> node_src_config_schema_inference_schema_ts
  node_src_config_schema_preset_schema_ts --> node_src_config_schema_loading_schema_ts
  node_src_config_schema_preset_schema_ts --> node_src_config_schema_manifest_schema_ts
  node_src_converter_core_ts --> node_src_config_schema_index_ts
  node_src_converter_core_ts --> node_src_debug_index_ts
  node_src_converter_core_ts --> node_src_storage_rdrr_format_ts
  node_src_converter_index_ts --> node_src_converter_core_ts
  node_src_converter_index_ts --> node_src_converter_io_node_ts
  node_src_converter_index_ts --> node_src_converter_quantizer_ts
  node_src_converter_index_ts --> node_src_converter_shard_packer_ts
  node_src_converter_index_ts --> node_src_converter_writer_ts
  node_src_converter_io_node_ts --> node_src_converter_shard_packer_ts
  node_src_converter_io_node_ts --> node_src_storage_rdrr_format_ts
  node_src_converter_node_converter_ts --> node_src_config_index_ts
  node_src_converter_node_converter_ts --> node_src_converter_core_ts
  node_src_converter_node_converter_ts --> node_src_converter_quantizer_ts
  node_src_converter_node_converter_ts --> node_src_converter_writer_ts
  node_src_converter_node_converter_ts --> node_src_debug_index_ts
  node_src_converter_node_converter_ts --> node_src_formats_gguf_index_ts
  node_src_converter_node_converter_ts --> node_src_formats_safetensors_index_ts
  node_src_converter_shard_packer_ts --> node_src_config_schema_index_ts
  node_src_converter_shard_packer_ts --> node_src_storage_rdrr_format_ts
  node_src_converter_test_model_ts --> node_src_converter_writer_ts
  node_src_converter_writer_ts --> node_src_config_schema_index_ts
  node_src_converter_writer_ts --> node_src_converter_test_model_ts
  node_src_converter_writer_ts --> node_src_debug_index_ts
  node_src_converter_writer_ts --> node_src_storage_rdrr_format_ts
  node_src_debug_index_ts --> node_src_config_schema_debug_schema_ts
  node_src_debug_index_ts --> node_src_debug_tensor_ts
  node_src_formats_gguf_index_ts --> node_src_formats_gguf_parser_ts
  node_src_formats_gguf_index_ts --> node_src_formats_gguf_types_ts
  node_src_formats_gguf_parser_ts --> node_src_formats_gguf_types_ts
  node_src_formats_index_ts --> node_src_formats_gguf_index_ts
  node_src_formats_index_ts --> node_src_formats_rdrr_index_ts
  node_src_formats_index_ts --> node_src_formats_safetensors_index_ts
  node_src_formats_index_ts --> node_src_formats_tokenizer_index_ts
  node_src_formats_rdrr_classification_ts --> node_src_formats_rdrr_types_ts
  node_src_formats_rdrr_groups_ts --> node_src_formats_rdrr_classification_ts
  node_src_formats_rdrr_groups_ts --> node_src_formats_rdrr_parsing_ts
  node_src_formats_rdrr_groups_ts --> node_src_formats_rdrr_types_ts
  node_src_formats_rdrr_index_ts --> node_src_formats_rdrr_classification_ts
  node_src_formats_rdrr_index_ts --> node_src_formats_rdrr_groups_ts
  node_src_formats_rdrr_index_ts --> node_src_formats_rdrr_manifest_ts
  node_src_formats_rdrr_index_ts --> node_src_formats_rdrr_parsing_ts
  node_src_formats_rdrr_index_ts --> node_src_formats_rdrr_types_ts
  node_src_formats_rdrr_index_ts --> node_src_formats_rdrr_validation_ts
  node_src_formats_rdrr_manifest_ts --> node_src_formats_rdrr_parsing_ts
  node_src_formats_rdrr_manifest_ts --> node_src_formats_rdrr_types_ts
  node_src_formats_rdrr_manifest_ts --> node_src_formats_rdrr_validation_ts
  node_src_formats_rdrr_parsing_ts --> node_src_formats_rdrr_types_ts
  node_src_formats_rdrr_parsing_ts --> node_src_formats_rdrr_validation_ts
  node_src_formats_rdrr_types_ts --> node_src_config_schema_index_ts
  node_src_formats_rdrr_validation_ts --> node_src_formats_rdrr_types_ts
  node_src_formats_safetensors_index_ts --> node_src_formats_safetensors_parser_ts
  node_src_formats_safetensors_index_ts --> node_src_formats_safetensors_types_ts
  node_src_formats_safetensors_parser_ts --> node_src_formats_safetensors_types_ts
  node_src_formats_safetensors_parser_ts --> node_src_formats_tokenizer_types_ts
  node_src_formats_tokenizer_index_ts --> node_src_formats_tokenizer_types_ts
  node_src_gpu_buffer_pool_ts --> node_src_config_schema_index_ts
  node_src_gpu_buffer_pool_ts --> node_src_debug_index_ts
  node_src_gpu_buffer_pool_ts --> node_src_gpu_device_ts
  node_src_gpu_buffer_pool_ts --> node_src_gpu_perf_guards_ts
  node_src_gpu_buffer_pool_ts --> node_src_types_gpu_ts
  node_src_gpu_command_recorder_ts --> node_src_debug_index_ts
  node_src_gpu_command_recorder_ts --> node_src_gpu_device_ts
  node_src_gpu_command_recorder_ts --> node_src_gpu_perf_guards_ts
  node_src_gpu_command_recorder_ts --> node_src_gpu_uniform_cache_ts
  node_src_gpu_device_ts --> node_src_config_kernels_registry_js
  node_src_gpu_device_ts --> node_src_config_platforms_loader_js
  node_src_gpu_device_ts --> node_src_config_schema_platform_schema_ts
  node_src_gpu_device_ts --> node_src_debug_index_ts
  node_src_gpu_device_ts --> node_src_gpu_submit_tracker_ts
  node_src_gpu_device_ts --> node_src_types_gpu_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_debug_index_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_gpu_device_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_gpu_kernels_attention_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_gpu_kernels_fused_matmul_rmsnorm_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_gpu_kernels_matmul_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_gpu_kernels_rmsnorm_ts
  node_src_gpu_kernel_benchmark_ts --> node_src_gpu_kernels_silu_ts
  node_src_gpu_kernel_hints_ts --> node_src_debug_index_ts
  node_src_gpu_kernel_hints_ts --> node_src_storage_rdrr_format_ts
  node_src_gpu_kernel_runtime_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernel_selection_cache_ts --> node_src_debug_index_ts
  node_src_gpu_kernel_selector_ts --> node_src_gpu_kernels_index_ts
  node_src_gpu_kernel_tuner_ts --> node_src_config_schema_index_ts
  node_src_gpu_kernel_tuner_ts --> node_src_debug_index_ts
  node_src_gpu_kernel_tuner_ts --> node_src_gpu_device_ts
  node_src_gpu_kernel_tuner_ts --> node_src_gpu_profiler_ts
  node_src_gpu_kernels_attention_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_kernels_kernel_base_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_attention_ts --> node_src_gpu_uniform_cache_ts
  node_src_gpu_kernels_cast_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_cast_ts --> node_src_gpu_perf_guards_ts
  node_src_gpu_kernels_check_stop_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_check_stop_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_check_stop_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_check_stop_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_check_stop_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_check_stop_ts --> node_src_gpu_perf_guards_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_dequant_ts --> node_src_gpu_uniform_cache_ts
  node_src_gpu_kernels_dispatch_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_fused_ffn_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_fused_ffn_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_fused_ffn_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_fused_ffn_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_fused_ffn_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_fused_ffn_ts --> node_src_gpu_kernels_kernel_base_ts
  node_src_gpu_kernels_fused_ffn_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_fused_ffn_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_fused_matmul_residual_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_fused_matmul_residual_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_fused_matmul_residual_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_fused_matmul_residual_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_fused_matmul_residual_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_fused_matmul_residual_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_fused_matmul_residual_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_fused_matmul_residual_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_fused_matmul_rmsnorm_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_fused_matmul_rmsnorm_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_fused_matmul_rmsnorm_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_fused_matmul_rmsnorm_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_fused_matmul_rmsnorm_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_fused_matmul_rmsnorm_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_fused_matmul_rmsnorm_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_fused_matmul_rmsnorm_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_gather_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_gather_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_gather_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_gather_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_gather_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_gather_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_gather_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_gather_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_gather_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_gelu_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_gelu_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_gelu_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_gelu_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_gelu_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_gelu_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_gelu_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_gelu_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernel_benchmark_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_attention_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_cast_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_dequant_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_fused_ffn_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_fused_matmul_residual_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_fused_matmul_rmsnorm_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_gather_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_gelu_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_matmul_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_moe_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_residual_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_rmsnorm_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_rope_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_sample_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_scale_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_silu_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_softmax_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_split_qkv_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_index_ts --> node_src_gpu_perf_profiler_ts
  node_src_gpu_kernels_kernel_base_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_kernel_base_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_kernel_base_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_matmul_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_kernel_hints_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_kernels_kernel_base_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_matmul_ts --> node_src_gpu_uniform_cache_ts
  node_src_gpu_kernels_moe_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_moe_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_moe_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_moe_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_moe_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_moe_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_moe_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_moe_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_residual_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_residual_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_residual_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_residual_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_residual_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_residual_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_residual_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_residual_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_rmsnorm_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_rmsnorm_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_rmsnorm_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_rmsnorm_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_rmsnorm_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_rmsnorm_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_rmsnorm_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_rmsnorm_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_rope_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_rope_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_rope_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_rope_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_rope_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_rope_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_rope_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_rope_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_sample_ts --> node_src_config_index_ts
  node_src_gpu_kernels_sample_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_sample_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_sample_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_sample_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_sample_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_sample_ts --> node_src_gpu_perf_guards_ts
  node_src_gpu_kernels_scale_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_scale_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_scale_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_scale_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_scale_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_scale_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_scale_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_scale_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_silu_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_silu_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_silu_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_silu_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_silu_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_silu_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_silu_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_silu_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_softmax_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_softmax_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_softmax_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_softmax_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_softmax_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_softmax_ts --> node_src_gpu_kernels_types_ts
  node_src_gpu_kernels_softmax_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_split_qkv_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_gpu_kernels_split_qkv_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_kernels_split_qkv_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_split_qkv_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_split_qkv_ts --> node_src_gpu_kernels_constants_ts
  node_src_gpu_kernels_split_qkv_ts --> node_src_gpu_kernels_dispatch_ts
  node_src_gpu_kernels_split_qkv_ts --> node_src_gpu_kernels_utils_ts
  node_src_gpu_kernels_utils_ts --> node_src_debug_index_ts
  node_src_gpu_kernels_utils_ts --> node_src_gpu_command_recorder_ts
  node_src_gpu_kernels_utils_ts --> node_src_gpu_device_ts
  node_src_gpu_kernels_utils_ts --> node_src_gpu_kernel_tuner_ts
  node_src_gpu_kernels_utils_ts --> node_src_gpu_uniform_cache_ts
  node_src_gpu_multi_model_recorder_ts --> node_src_inference_pipeline_ts
  node_src_gpu_partitioned_buffer_pool_ts --> node_src_gpu_buffer_pool_ts
  node_src_gpu_perf_guards_ts --> node_src_debug_index_ts
  node_src_gpu_perf_profiler_ts --> node_src_debug_index_ts
  node_src_gpu_perf_profiler_ts --> node_src_gpu_device_ts
  node_src_gpu_profiler_ts --> node_src_debug_index_ts
  node_src_gpu_profiler_ts --> node_src_gpu_device_ts
  node_src_gpu_profiler_ts --> node_src_gpu_perf_guards_ts
  node_src_gpu_profiler_ts --> node_src_types_gpu_ts
  node_src_gpu_submit_tracker_ts --> node_src_debug_index_ts
  node_src_gpu_submit_tracker_ts --> node_src_gpu_perf_guards_ts
  node_src_gpu_uniform_cache_ts --> node_src_config_schema_gpu_cache_schema_ts
  node_src_gpu_uniform_cache_ts --> node_src_gpu_device_ts
  node_src_index_ts --> node_src_adapters_index_ts
  node_src_index_ts --> node_src_inference_expert_router_ts
  node_src_index_ts --> node_src_inference_kv_cache_ts
  node_src_index_ts --> node_src_inference_moe_router_ts
  node_src_index_ts --> node_src_inference_multi_model_network_ts
  node_src_index_ts --> node_src_inference_multi_pipeline_pool_ts
  node_src_index_ts --> node_src_inference_network_evolution_ts
  node_src_index_ts --> node_src_inference_pipeline_config_ts
  node_src_index_ts --> node_src_inference_pipeline_lora_ts
  node_src_index_ts --> node_src_inference_pipeline_sampling_ts
  node_src_index_ts --> node_src_inference_pipeline_ts
  node_src_index_ts --> node_src_inference_speculative_ts
  node_src_index_ts --> node_src_inference_tokenizer_ts
  node_src_index_ts --> node_src_loader_doppler_loader_ts
  node_src_index_ts --> node_src_loader_multi_model_loader_ts
  node_src_index_ts --> node_src_storage_rdrr_format_ts
  node_src_inference_decode_buffers_ts --> node_src_gpu_device_ts
  node_src_inference_functiongemma_ts --> node_src_adapters_adapter_manager_ts
  node_src_inference_functiongemma_ts --> node_src_adapters_adapter_registry_ts
  node_src_inference_functiongemma_ts --> node_src_inference_multi_model_network_ts
  node_src_inference_functiongemma_ts --> node_src_inference_multi_pipeline_pool_ts
  node_src_inference_functiongemma_ts --> node_src_inference_network_evolution_ts
  node_src_inference_functiongemma_ts --> node_src_inference_pipeline_lora_ts
  node_src_inference_functiongemma_ts --> node_src_inference_pipeline_ts
  node_src_inference_functiongemma_ts --> node_src_loader_multi_model_loader_ts
  node_src_inference_kv_cache_ts --> node_src_config_schema_kvcache_schema_ts
  node_src_inference_kv_cache_ts --> node_src_debug_index_ts
  node_src_inference_kv_cache_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_inference_kv_cache_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_kv_cache_ts --> node_src_gpu_device_ts
  node_src_inference_kv_cache_ts --> node_src_gpu_perf_guards_ts
  node_src_inference_kv_cache_ts --> node_src_types_inference_ts
  node_src_inference_moe_router_ts --> node_src_config_index_ts
  node_src_inference_moe_router_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_moe_router_ts --> node_src_gpu_device_ts
  node_src_inference_moe_router_ts --> node_src_gpu_kernel_selector_ts
  node_src_inference_moe_router_ts --> node_src_types_inference_ts
  node_src_inference_multi_model_network_ts --> node_src_gpu_multi_model_recorder_ts
  node_src_inference_multi_model_network_ts --> node_src_inference_expert_router_ts
  node_src_inference_multi_model_network_ts --> node_src_inference_multi_pipeline_pool_ts
  node_src_inference_multi_model_network_ts --> node_src_inference_network_evolution_ts
  node_src_inference_multi_model_network_ts --> node_src_inference_pipeline_lora_ts
  node_src_inference_multi_model_network_ts --> node_src_inference_pipeline_ts
  node_src_inference_multi_model_network_ts --> node_src_loader_multi_model_loader_ts
  node_src_inference_multi_pipeline_pool_ts --> node_src_gpu_multi_model_recorder_ts
  node_src_inference_multi_pipeline_pool_ts --> node_src_gpu_partitioned_buffer_pool_ts
  node_src_inference_multi_pipeline_pool_ts --> node_src_inference_pipeline_lora_ts
  node_src_inference_multi_pipeline_pool_ts --> node_src_inference_pipeline_ts
  node_src_inference_multi_pipeline_pool_ts --> node_src_loader_multi_model_loader_ts
  node_src_inference_pipeline_attention_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_attention_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_inference_pipeline_attention_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_attention_ts --> node_src_gpu_device_ts
  node_src_inference_pipeline_attention_ts --> node_src_gpu_kernel_selector_ts
  node_src_inference_pipeline_attention_ts --> node_src_gpu_kernels_split_qkv_ts
  node_src_inference_pipeline_attention_ts --> node_src_inference_pipeline_config_ts
  node_src_inference_pipeline_attention_ts --> node_src_inference_pipeline_debug_utils_ts
  node_src_inference_pipeline_attention_ts --> node_src_inference_pipeline_kernel_trace_ts
  node_src_inference_pipeline_attention_ts --> node_src_inference_pipeline_lora_apply_ts
  node_src_inference_pipeline_attention_ts --> node_src_inference_pipeline_lora_ts
  node_src_inference_pipeline_attention_ts --> node_src_inference_pipeline_types_ts
  node_src_inference_pipeline_config_ts --> node_src_config_loader_ts
  node_src_inference_pipeline_config_ts --> node_src_config_schema_index_ts
  node_src_inference_pipeline_config_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_debug_utils_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_debug_utils_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_debug_utils_ts --> node_src_inference_kv_cache_ts
  node_src_inference_pipeline_embed_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_embed_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_embed_ts --> node_src_gpu_command_recorder_ts
  node_src_inference_pipeline_embed_ts --> node_src_gpu_device_ts
  node_src_inference_pipeline_embed_ts --> node_src_gpu_kernel_selector_ts
  node_src_inference_pipeline_init_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_init_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_inference_pipeline_init_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_init_ts --> node_src_gpu_device_ts
  node_src_inference_pipeline_init_ts --> node_src_inference_kv_cache_ts
  node_src_inference_pipeline_init_ts --> node_src_inference_moe_router_ts
  node_src_inference_pipeline_init_ts --> node_src_inference_pipeline_config_ts
  node_src_inference_pipeline_init_ts --> node_src_inference_pipeline_types_ts
  node_src_inference_pipeline_init_ts --> node_src_inference_speculative_ts
  node_src_inference_pipeline_init_ts --> node_src_inference_tokenizer_ts
  node_src_inference_pipeline_init_ts --> node_src_loader_doppler_loader_ts
  node_src_inference_pipeline_init_ts --> node_src_storage_rdrr_format_ts
  node_src_inference_pipeline_kernel_trace_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_kernel_trace_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_layer_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_layer_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_inference_pipeline_layer_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_layer_ts --> node_src_gpu_device_ts
  node_src_inference_pipeline_layer_ts --> node_src_gpu_kernel_selector_ts
  node_src_inference_pipeline_layer_ts --> node_src_gpu_perf_guards_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_decode_buffers_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_kv_cache_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_moe_router_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_attention_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_config_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_debug_utils_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_kernel_trace_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_lora_apply_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_lora_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_moe_impl_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_types_ts
  node_src_inference_pipeline_layer_ts --> node_src_inference_pipeline_weights_ts
  node_src_inference_pipeline_logits_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_logits_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_inference_pipeline_logits_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_logits_ts --> node_src_gpu_command_recorder_ts
  node_src_inference_pipeline_logits_ts --> node_src_gpu_device_ts
  node_src_inference_pipeline_logits_ts --> node_src_gpu_kernel_selector_ts
  node_src_inference_pipeline_logits_ts --> node_src_gpu_kernels_matmul_ts
  node_src_inference_pipeline_logits_ts --> node_src_gpu_kernels_rmsnorm_ts
  node_src_inference_pipeline_logits_ts --> node_src_gpu_perf_guards_ts
  node_src_inference_pipeline_logits_ts --> node_src_inference_pipeline_kernel_trace_ts
  node_src_inference_pipeline_lora_apply_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_lora_apply_ts --> node_src_gpu_command_recorder_ts
  node_src_inference_pipeline_lora_apply_ts --> node_src_gpu_kernel_selector_ts
  node_src_inference_pipeline_lora_apply_ts --> node_src_gpu_kernels_residual_ts
  node_src_inference_pipeline_lora_apply_ts --> node_src_gpu_kernels_scale_ts
  node_src_inference_pipeline_lora_apply_ts --> node_src_inference_pipeline_lora_ts
  node_src_inference_pipeline_lora_apply_ts --> node_src_inference_pipeline_types_ts
  node_src_inference_pipeline_lora_ts --> node_src_inference_pipeline_lora_types_ts
  node_src_inference_pipeline_lora_types_ts --> node_src_inference_pipeline_buffer_types_ts
  node_src_inference_pipeline_moe_impl_ts --> node_src_config_schema_index_ts
  node_src_inference_pipeline_moe_impl_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_moe_impl_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_moe_impl_ts --> node_src_gpu_device_ts
  node_src_inference_pipeline_moe_impl_ts --> node_src_gpu_kernel_selector_ts
  node_src_inference_pipeline_moe_impl_ts --> node_src_inference_moe_router_ts
  node_src_inference_pipeline_moe_impl_ts --> node_src_inference_pipeline_types_ts
  node_src_inference_pipeline_sampling_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_ts --> node_src_config_index_ts
  node_src_inference_pipeline_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_ts --> node_src_gpu_command_recorder_ts
  node_src_inference_pipeline_ts --> node_src_gpu_device_ts
  node_src_inference_pipeline_ts --> node_src_gpu_kernel_hints_ts
  node_src_inference_pipeline_ts --> node_src_gpu_kernel_selection_cache_ts
  node_src_inference_pipeline_ts --> node_src_gpu_kernels_check_stop_ts
  node_src_inference_pipeline_ts --> node_src_gpu_kernels_gather_ts
  node_src_inference_pipeline_ts --> node_src_gpu_kernels_sample_ts
  node_src_inference_pipeline_ts --> node_src_gpu_kernels_scale_ts
  node_src_inference_pipeline_ts --> node_src_gpu_perf_guards_ts
  node_src_inference_pipeline_ts --> node_src_gpu_submit_tracker_ts
  node_src_inference_pipeline_ts --> node_src_inference_decode_buffers_ts
  node_src_inference_pipeline_ts --> node_src_inference_kv_cache_ts
  node_src_inference_pipeline_ts --> node_src_inference_moe_router_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_config_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_embed_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_init_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_layer_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_logits_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_lora_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_moe_impl_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_sampling_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_types_ts
  node_src_inference_pipeline_ts --> node_src_inference_pipeline_weights_ts
  node_src_inference_pipeline_ts --> node_src_inference_speculative_ts
  node_src_inference_pipeline_ts --> node_src_inference_tokenizer_ts
  node_src_inference_pipeline_ts --> node_src_loader_doppler_loader_ts
  node_src_inference_pipeline_ts --> node_src_storage_rdrr_format_ts
  node_src_inference_pipeline_types_ts --> node_src_inference_pipeline_buffer_types_ts
  node_src_inference_pipeline_types_ts --> node_src_inference_pipeline_config_ts
  node_src_inference_pipeline_types_ts --> node_src_inference_pipeline_lora_types_ts
  node_src_inference_pipeline_weights_ts --> node_src_debug_index_ts
  node_src_inference_pipeline_weights_ts --> node_src_gpu_buffer_pool_ts
  node_src_inference_pipeline_weights_ts --> node_src_gpu_device_ts
  node_src_inference_pipeline_weights_ts --> node_src_inference_pipeline_types_ts
  node_src_inference_test_harness_ts --> node_src_debug_index_ts
  node_src_inference_test_harness_ts --> node_src_gpu_device_ts
  node_src_inference_test_harness_ts --> node_src_inference_pipeline_config_ts
  node_src_inference_test_harness_ts --> node_src_inference_pipeline_ts
  node_src_inference_test_harness_ts --> node_src_storage_rdrr_format_ts
  node_src_inference_tokenizer_ts --> node_src_config_index_ts
  node_src_inference_tokenizer_ts --> node_src_debug_index_ts
  node_src_inference_tokenizer_ts --> node_src_storage_shard_manager_ts
  node_src_inference_tokenizer_ts --> node_src_types_inference_ts
  node_src_loader_doppler_loader_ts --> node_src_config_schema_loading_schema_ts
  node_src_loader_doppler_loader_ts --> node_src_debug_index_ts
  node_src_loader_doppler_loader_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_loader_doppler_loader_ts --> node_src_gpu_buffer_pool_ts
  node_src_loader_doppler_loader_ts --> node_src_gpu_device_ts
  node_src_loader_doppler_loader_ts --> node_src_gpu_kernel_hints_ts
  node_src_loader_doppler_loader_ts --> node_src_gpu_kernel_selector_ts
  node_src_loader_doppler_loader_ts --> node_src_inference_pipeline_lora_ts
  node_src_loader_doppler_loader_ts --> node_src_loader_dtype_utils_ts
  node_src_loader_doppler_loader_ts --> node_src_loader_expert_cache_ts
  node_src_loader_doppler_loader_ts --> node_src_loader_loader_types_ts
  node_src_loader_doppler_loader_ts --> node_src_loader_shard_cache_ts
  node_src_loader_doppler_loader_ts --> node_src_loader_weights_ts
  node_src_loader_doppler_loader_ts --> node_src_memory_capability_ts
  node_src_loader_doppler_loader_ts --> node_src_memory_heap_manager_ts
  node_src_loader_doppler_loader_ts --> node_src_memory_unified_detect_ts
  node_src_loader_doppler_loader_ts --> node_src_storage_quota_ts
  node_src_loader_doppler_loader_ts --> node_src_storage_rdrr_format_ts
  node_src_loader_doppler_loader_ts --> node_src_storage_shard_manager_ts
  node_src_loader_dtype_utils_ts --> node_src_debug_index_ts
  node_src_loader_dtype_utils_ts --> node_src_gpu_buffer_dtypes_ts
  node_src_loader_dtype_utils_ts --> node_src_gpu_buffer_pool_ts
  node_src_loader_dtype_utils_ts --> node_src_gpu_device_ts
  node_src_loader_dtype_utils_ts --> node_src_gpu_kernels_cast_ts
  node_src_loader_dtype_utils_ts --> node_src_loader_loader_types_ts
  node_src_loader_expert_cache_ts --> node_src_config_schema_loading_schema_ts
  node_src_loader_expert_cache_ts --> node_src_debug_index_ts
  node_src_loader_expert_cache_ts --> node_src_gpu_buffer_pool_ts
  node_src_loader_expert_cache_ts --> node_src_loader_weights_ts
  node_src_loader_multi_model_loader_ts --> node_src_adapters_lora_loader_ts
  node_src_loader_multi_model_loader_ts --> node_src_inference_pipeline_config_ts
  node_src_loader_multi_model_loader_ts --> node_src_inference_pipeline_init_ts
  node_src_loader_multi_model_loader_ts --> node_src_inference_pipeline_lora_ts
  node_src_loader_multi_model_loader_ts --> node_src_inference_pipeline_ts
  node_src_loader_multi_model_loader_ts --> node_src_loader_doppler_loader_ts
  node_src_loader_multi_model_loader_ts --> node_src_storage_rdrr_format_ts
  node_src_loader_shard_cache_ts --> node_src_config_schema_loading_schema_ts
  node_src_loader_shard_cache_ts --> node_src_debug_index_ts
  node_src_loader_shard_cache_ts --> node_src_loader_loader_types_ts
  node_src_loader_shard_cache_ts --> node_src_storage_quota_ts
  node_src_loader_shard_cache_ts --> node_src_storage_rdrr_format_ts
  node_src_loader_shard_cache_ts --> node_src_storage_shard_manager_ts
  node_src_memory_capability_ts --> node_src_config_schema_memory_limits_schema_ts
  node_src_memory_capability_ts --> node_src_memory_unified_detect_ts
  node_src_memory_heap_manager_ts --> node_src_config_schema_memory_limits_schema_ts
  node_src_memory_heap_manager_ts --> node_src_memory_address_table_ts
  node_src_memory_heap_manager_ts --> node_src_memory_capability_ts
  node_src_storage_download_types_ts --> node_src_config_index_ts
  node_src_storage_download_types_ts --> node_src_storage_rdrr_format_ts
  node_src_storage_downloader_ts --> node_src_debug_index_ts
  node_src_storage_downloader_ts --> node_src_storage_download_types_ts
  node_src_storage_downloader_ts --> node_src_storage_quota_ts
  node_src_storage_downloader_ts --> node_src_storage_rdrr_format_ts
  node_src_storage_downloader_ts --> node_src_storage_shard_manager_ts
  node_src_storage_preflight_ts --> node_src_config_index_ts
  node_src_storage_preflight_ts --> node_src_memory_capability_ts
  node_src_storage_preflight_ts --> node_src_storage_quota_ts
  node_src_storage_quickstart_downloader_ts --> node_src_storage_download_types_ts
  node_src_storage_quickstart_downloader_ts --> node_src_storage_downloader_ts
  node_src_storage_quickstart_downloader_ts --> node_src_storage_preflight_ts
  node_src_storage_quickstart_downloader_ts --> node_src_storage_quota_ts
  node_src_storage_quickstart_downloader_ts --> node_src_storage_shard_manager_ts
  node_src_storage_quota_ts --> node_src_config_index_ts
  node_src_storage_quota_ts --> node_src_debug_index_ts
  node_src_storage_rdrr_format_ts --> node_src_formats_rdrr_index_ts
  node_src_storage_shard_manager_ts --> node_src_config_schema_loading_schema_ts
  node_src_storage_shard_manager_ts --> node_src_config_schema_storage_schema_ts
  node_src_storage_shard_manager_ts --> node_src_debug_index_ts
  node_src_storage_shard_manager_ts --> node_src_storage_quota_ts
  node_src_storage_shard_manager_ts --> node_src_storage_rdrr_format_ts
  node_src_types_inference_ts --> node_src_types_model_ts
  node_tests_benchmark_index_js --> node_tests_benchmark_pipeline_benchmark_ts
  node_tests_benchmark_index_js --> node_tests_benchmark_prompts_ts
  node_tests_benchmark_index_js --> node_tests_benchmark_results_storage_ts
  node_tests_benchmark_index_js --> node_tests_benchmark_system_benchmark_ts
  node_tests_benchmark_index_js --> node_tests_benchmark_types_ts
  node_tests_benchmark_index_ts --> node_src_debug_index_ts
  node_tests_benchmark_index_ts --> node_tests_benchmark_pipeline_benchmark_ts
  node_tests_benchmark_index_ts --> node_tests_benchmark_prompts_ts
  node_tests_benchmark_index_ts --> node_tests_benchmark_results_storage_ts
  node_tests_benchmark_index_ts --> node_tests_benchmark_system_benchmark_ts
  node_tests_benchmark_index_ts --> node_tests_benchmark_types_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_src_gpu_buffer_pool_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_src_gpu_device_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_src_gpu_perf_guards_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_src_gpu_profiler_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_src_gpu_submit_tracker_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_src_inference_pipeline_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_src_storage_downloader_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_src_storage_shard_manager_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_tests_benchmark_prompts_ts
  node_tests_benchmark_pipeline_benchmark_js --> node_tests_benchmark_types_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_gpu_buffer_pool_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_gpu_device_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_gpu_perf_guards_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_gpu_profiler_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_gpu_submit_tracker_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_inference_pipeline_init_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_inference_pipeline_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_storage_downloader_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_src_storage_shard_manager_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_tests_benchmark_prompts_ts
  node_tests_benchmark_pipeline_benchmark_ts --> node_tests_benchmark_types_ts
  node_tests_benchmark_prompts_ts --> node_tests_benchmark_types_ts
  node_tests_benchmark_results_storage_ts --> node_tests_benchmark_system_benchmark_ts
  node_tests_benchmark_results_storage_ts --> node_tests_benchmark_types_ts
  node_tests_benchmark_system_benchmark_ts --> node_tests_benchmark_types_ts
  node_tests_benchmark_types_ts --> node_src_storage_rdrr_format_ts
  node_tests_helpers_index_ts --> node_tests_helpers_console_capture_ts
  node_tests_helpers_index_ts --> node_tests_helpers_demo_page_ts
```
