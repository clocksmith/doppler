import { describe, expect, it } from 'vitest';

import { resolvePreset } from '../../src/config/loader.js';
import { createManifest } from '../../src/formats/rdrr/manifest.js';
import { initDevice } from '../../src/gpu/device.js';
import { ERROR_CODES } from '../../src/errors/index.js';

describe('error codes', () => {
  it('tags unknown presets with a config error code', () => {
    try {
      resolvePreset('not-a-preset');
      throw new Error('Expected resolvePreset to throw');
    } catch (err) {
      const error =  (err);
      expect(error.message).toContain(ERROR_CODES.CONFIG_PRESET_UNKNOWN);
      expect(error.code).toBe(ERROR_CODES.CONFIG_PRESET_UNKNOWN);
    }
  });

  it('tags invalid manifests with a loader error code', () => {
    try {
      createManifest({ modelId: null });
      throw new Error('Expected createManifest to throw');
    } catch (err) {
      const error =  (err);
      expect(error.message).toContain(ERROR_CODES.LOADER_MANIFEST_INVALID);
      expect(error.code).toBe(ERROR_CODES.LOADER_MANIFEST_INVALID);
    }
  });

  it('tags missing WebGPU as a GPU error code', async () => {
    await expect(initDevice()).rejects.toMatchObject({
      message: expect.stringContaining(ERROR_CODES.GPU_UNAVAILABLE),
      code: ERROR_CODES.GPU_UNAVAILABLE,
    });
  });
});
