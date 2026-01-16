# Error Codes

Doppler errors carry a stable code in the message prefix and on the error
object when created via `createDopplerError()`. Format:

```
[DOPPLER_*] Message text...
```

Use these codes in tests and user-facing error handling.

## Config Errors

- `DOPPLER_CONFIG_PRESET_UNKNOWN`
  - Unknown runtime preset requested by id.

## GPU Errors

- `DOPPLER_GPU_UNAVAILABLE`
  - WebGPU is not available in the current browser/worker context.
- `DOPPLER_GPU_DEVICE_FAILED`
  - Adapter/device creation failed.

## Loader Errors

- `DOPPLER_LOADER_MANIFEST_INVALID`
  - RDRR manifest failed validation.
- `DOPPLER_LOADER_SHARD_INDEX_INVALID`
  - Requested shard index does not exist in the manifest.
