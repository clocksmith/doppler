import { log } from '../src/debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { TOOLING_INTENTS } from '../src/config/schema/tooling.schema.js';
import { applyRuntimePreset, runBrowserSuite } from '../src/inference/browser-harness.js';

const BENCH_INTENTS = new Set(['investigate', 'calibrate']);
const ALLOWED_INTENTS = new Set(TOOLING_INTENTS);

function normalizeSuite(suite) {
  return String(suite || '').trim().toLowerCase();
}

function resolveModelRef(model, options = {}) {
  const modelId =
    options.modelId ||
    model?.modelId ||
    model?.id ||
    model?.key ||
    model?.sources?.browser?.id ||
    null;
  const modelUrl =
    options.modelUrl ||
    model?.sources?.browser?.url ||
    null;
  return { modelId, modelUrl };
}

function resolveRuntimeConfig(options = {}) {
  if (options.runtimeConfig && typeof options.runtimeConfig === 'object') {
    return options.runtimeConfig;
  }
  return getRuntimeConfig();
}

export class DiagnosticsController {
  constructor(options = {}) {
    this.log = options.log || log;
    this.lastReport = null;
    this.lastReportInfo = null;
  }

  requireIntent(runtimeConfig, suite) {
    const intent = runtimeConfig?.shared?.tooling?.intent ?? null;
    if (!intent || !ALLOWED_INTENTS.has(intent)) {
      throw new Error('runtime.shared.tooling.intent is required for diagnostics');
    }
    if ((suite === 'bench' || suite === 'diffusion') && !BENCH_INTENTS.has(intent)) {
      const target = suite === 'diffusion' ? 'diffusion' : 'bench';
      throw new Error(`runtime.shared.tooling.intent must be investigate or calibrate for ${target}`);
    }
    return intent;
  }

  async applyRuntimePreset(presetId) {
    return applyRuntimePreset(presetId);
  }

  async verifySuite(model, options = {}) {
    const suite = normalizeSuite(options.suite || 'inference');
    const runtimeConfig = resolveRuntimeConfig(options);
    this.requireIntent(runtimeConfig, suite);
    return { ok: true, suite };
  }

  async runSuite(model, options = {}) {
    const suite = normalizeSuite(options.suite || 'inference');
    const runtimeConfig = resolveRuntimeConfig(options);
    this.requireIntent(runtimeConfig, suite);

    if (options.runtimePreset) {
      await applyRuntimePreset(options.runtimePreset);
    }
    if (options.runtimeConfig) {
      setRuntimeConfig(options.runtimeConfig);
    }

    const runtime = { runtimeConfig: getRuntimeConfig() };
    const { modelId, modelUrl } = resolveModelRef(model, options);
    if (suite !== 'kernels' && !modelId && !modelUrl) {
      throw new Error('modelId or modelUrl is required for this suite');
    }

    const result = await runBrowserSuite({
      suite,
      modelId,
      modelUrl,
      runtimePreset: options.runtimePreset ?? null,
      captureOutput: options.captureOutput === true,
      runtime,
      report: options.report,
      keepPipeline: options.keepPipeline,
    });

    this.lastReport = result.report ?? null;
    this.lastReportInfo = result.reportInfo ?? null;
    return result;
  }
}
