export declare const ERROR_CODES: {
  readonly CONFIG_PRESET_UNKNOWN: 'DOPPLER_CONFIG_PRESET_UNKNOWN';
  readonly GPU_UNAVAILABLE: 'DOPPLER_GPU_UNAVAILABLE';
  readonly GPU_DEVICE_FAILED: 'DOPPLER_GPU_DEVICE_FAILED';
  readonly LOADER_MANIFEST_INVALID: 'DOPPLER_LOADER_MANIFEST_INVALID';
  readonly LOADER_SHARD_INDEX_INVALID: 'DOPPLER_LOADER_SHARD_INDEX_INVALID';
};

export declare function createDopplerError(code: string, message: string): Error & { code?: string };
