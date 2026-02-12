import {
  log,
  getRuntimeConfig,
  TOOLING_INTENTS,
  applyRuntimePreset,
  runBrowserCommand,
} from '@doppler/core';

const ALLOWED_INTENTS = new Set(TOOLING_INTENTS);
const SUPPORTED_VERIFY_SUITES = new Set(['kernels', 'inference', 'diffusion', 'energy']);

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

function mapSuiteToCommand(suite) {
  if (suite === 'bench') return { command: 'bench', suite: null };
  if (suite === 'debug') return { command: 'debug', suite: null };
  if (SUPPORTED_VERIFY_SUITES.has(suite)) {
    return { command: 'test-model', suite };
  }
  throw new Error(`Unsupported diagnostics suite "${suite}"`);
}

export class DiagnosticsController {
  constructor(options = {}) {
    this.log = options.log || log;
    this.lastReport = null;
    this.lastReportInfo = null;
  }

  requireIntent(runtimeConfig) {
    const intent = runtimeConfig?.shared?.tooling?.intent ?? null;
    if (intent !== null && !ALLOWED_INTENTS.has(intent)) {
      throw new Error('runtime.shared.tooling.intent is invalid for diagnostics');
    }
    return intent;
  }

  async applyRuntimePreset(presetId) {
    return applyRuntimePreset(presetId);
  }

  async verifySuite(model, options = {}) {
    const suite = normalizeSuite(options.suite || 'inference');
    const runtimeConfig = resolveRuntimeConfig(options);
    this.requireIntent(runtimeConfig);
    mapSuiteToCommand(suite);
    return { ok: true, suite };
  }

  async runSuite(model, options = {}) {
    const suite = normalizeSuite(options.suite || 'inference');
    const runtimeConfig = resolveRuntimeConfig(options);
    this.requireIntent(runtimeConfig);

    const { modelId, modelUrl } = resolveModelRef(model, options);
    const mapped = mapSuiteToCommand(suite);

    if (suite !== 'kernels' && !modelId && !modelUrl) {
      throw new Error('modelId or modelUrl is required for this suite');
    }

    const response = await runBrowserCommand({
      command: mapped.command,
      suite: mapped.suite ?? undefined,
      modelId,
      modelUrl,
      runtimePreset: options.runtimePreset ?? null,
      runtimeConfigUrl: options.runtimeConfigUrl ?? null,
      runtimeConfig: options.runtimeConfig ?? null,
      captureOutput: options.captureOutput === true,
      keepPipeline: options.keepPipeline === true,
      report: options.report ?? null,
    });

    const result = response.result;
    this.lastReport = result.report ?? null;
    this.lastReportInfo = result.reportInfo ?? null;
    return result;
  }
}
