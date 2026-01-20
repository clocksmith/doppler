
import { log } from '../src/debug/index.js';
import {
  applyRuntimePreset,
  runBrowserSuite,
} from '../src/inference/browser-harness.js';
import {
  openModelStore,
  loadManifestFromStore,
  verifyIntegrity,
} from '../src/storage/shard-manager.js';
import { parseManifest } from '../src/storage/rdrr-format.js';
import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';

function resolveModelSource(model) {
  const sources = model?.sources || {};
  if (sources.browser?.id) {
    return { modelId: sources.browser.id, sourceType: 'browser' };
  }
  if (sources.server?.url) {
    return { modelUrl: sources.server.url, modelId: sources.server.id || model.key, sourceType: 'server' };
  }
  if (sources.remote?.url) {
    return { modelUrl: sources.remote.url, modelId: sources.remote.id || model.key, sourceType: 'remote' };
  }
  return { modelId: model?.key || null, sourceType: 'unknown' };
}

function applyRuntimeOverrides({ prompt, maxTokens }) {
  const current = getRuntimeConfig();
  const next = {
    ...current,
    inference: {
      ...current.inference,
      prompt: typeof prompt === 'string' && prompt.trim() ? prompt.trim() : current.inference?.prompt,
      batching: {
        ...current.inference?.batching,
        maxTokens: Number.isFinite(maxTokens)
          ? Math.max(1, Math.floor(maxTokens))
          : current.inference?.batching?.maxTokens,
      },
    },
  };
  setRuntimeConfig(next);
  return next;
}

export class DiagnosticsController {
  #isRunning = false;

  #isVerifying = false;

  #callbacks;

  constructor(callbacks = {}) {
    this.#callbacks = callbacks;
  }

  get isRunning() {
    return this.#isRunning;
  }

  get isVerifying() {
    return this.#isVerifying;
  }

  async verifyModel(model) {
    if (this.#isVerifying) {
      return null;
    }

    const { modelId } = resolveModelSource(model);
    if (!modelId) {
      throw new Error('Model id is required for verification');
    }

    this.#isVerifying = true;
    this.#callbacks.onVerifyStart?.(modelId);

    try {
      await openModelStore(modelId);
      const manifestText = await loadManifestFromStore();
      if (!manifestText) {
        throw new Error('Manifest not found in storage');
      }
      parseManifest(manifestText);
      const result = await verifyIntegrity();
      this.#callbacks.onVerifyComplete?.(result);
      return result;
    } catch (error) {
      this.#callbacks.onVerifyError?.(error);
      throw error;
    } finally {
      this.#isVerifying = false;
      this.#callbacks.onVerifyFinish?.();
    }
  }

  async runSuite(model, options = {}) {
    if (this.#isRunning) {
      return null;
    }

    const suite = options.suite || 'inference';
    const runtimePreset = options.runtimePreset || null;
    const resolved = resolveModelSource(model);

    this.#isRunning = true;
    this.#callbacks.onSuiteStart?.(suite, resolved);

    try {
      if (runtimePreset) {
        await applyRuntimePreset(runtimePreset);
      }
      applyRuntimeOverrides({ prompt: options.prompt, maxTokens: options.maxTokens });

      const result = await runBrowserSuite({
        suite,
        modelId: resolved.modelId,
        modelUrl: resolved.modelUrl,
        runtimePreset,
        prompt: options.prompt,
        maxTokens: options.maxTokens,
      });

      this.#callbacks.onSuiteComplete?.(result);
      return result;
    } catch (error) {
      log.error('Diagnostics', 'Suite run failed:', error);
      this.#callbacks.onSuiteError?.(error);
      throw error;
    } finally {
      this.#isRunning = false;
      this.#callbacks.onSuiteFinish?.();
    }
  }
}
