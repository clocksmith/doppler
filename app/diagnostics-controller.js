
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

function resolveSuiteCommand(suite) {
  const normalized = String(suite || '').trim().toLowerCase();
  if (normalized === 'bench') return 'bench';
  if (normalized === 'debug') return 'debug';
  return 'test';
}

function assertToolingIntent(command, runtime) {
  if (command === 'run' || command === 'convert' || command === 'tool') return;
  const intent = runtime?.shared?.tooling?.intent ?? null;
  if (!intent) {
    throw new Error('runtime.shared.tooling.intent is required for diagnostics runs.');
  }

  const allowed = {
    debug: new Set(['investigate']),
    test: new Set(['verify']),
    bench: new Set(['calibrate', 'investigate']),
  }[command];

  if (allowed && !allowed.has(intent)) {
    throw new Error(
      `diagnostics suite "${command}" requires runtime.shared.tooling.intent to be ` +
      `${[...allowed].join(' or ')}.`
    );
  }
}

export class DiagnosticsController {
  #isRunning = false;

  #isVerifying = false;

  #callbacks;

  #lastReport = null;

  #lastReportInfo = null;

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
    const runtimeConfig = options.runtimeConfig || null;
    const resolved = resolveModelSource(model);

    this.#isRunning = true;
    this.#callbacks.onSuiteStart?.(suite, resolved);

    try {
      if (runtimeConfig) {
        setRuntimeConfig(runtimeConfig);
      } else if (runtimePreset) {
        await applyRuntimePreset(runtimePreset);
      }

      const runtime = getRuntimeConfig();
      const command = resolveSuiteCommand(suite);
      assertToolingIntent(command, runtime);

      const result = await runBrowserSuite({
        suite,
        modelId: resolved.modelId,
        modelUrl: resolved.modelUrl,
        runtimePreset,
        runtime,
      });

      this.#lastReport = result.report ?? null;
      this.#lastReportInfo = result.reportInfo ?? null;
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

  exportLastReport() {
    if (!this.#lastReport) {
      throw new Error('No diagnostics report available to export.');
    }
    const timestamp = String(this.#lastReport.timestamp || new Date().toISOString()).replace(/[:]/g, '-');
    const modelId = String(this.#lastReport.modelId || 'report').replace(/[^a-zA-Z0-9_-]/g, '_');
    const filename = `${modelId}-${timestamp}.json`;
    const payload = JSON.stringify(this.#lastReport, null, 2);
    const blob = new Blob([payload], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}
