# Config Schema Layout (Target)

This document defines the intended schema layout and ownership boundaries.

## Root Schemas

ConverterConfigSchema
- quantization
- sharding
- weightLayout
- manifest
- output
- presets

RuntimeConfigSchema
- shared
- loading
- inference

ManifestInferenceSchema (embedded in manifest.json)
- attention
- normalization
- ffn
- rope
- output
- layerPattern
- chatTemplate
- defaultKernelPath

## Runtime Subschemas

SharedRuntimeConfigSchema (cross-cutting for loading + inference)
- debug
- benchmark
- platform
- kernelRegistry
- kernelThresholds
- bufferPool
- gpuCache
- memory
- tuner
- hotSwap
- bridge

LoaderConfigSchema (runtime.loading)
- storage
- distribution
- shardCache
- memoryManagement
- opfsPath
- expertCache

InferenceConfigSchema (runtime.inference)
- batching
- sampling
- compute
- tokenizer
- largeWeights
- kvcache
- moe
- pipeline
- kernelPath
- chatTemplate
- prompt
- modelOverrides

## Rules

- Converter -> manifest is the only bridge into runtime.
- Loader must not mutate inference config.
- Shared runtime is the only cross-cutting config between loader and inference.
- Defaults live in schema files; runtime code should not hardcode fallbacks.
