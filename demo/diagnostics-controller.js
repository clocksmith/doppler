import {
  log,
  TOOLING_INTENTS,
  TOOLING_VERIFY_WORKLOADS,
  applyRuntimeProfile,
  runBrowserCommand,
} from 'doppler-gpu/tooling';

const ALLOWED_INTENTS = new Set(TOOLING_INTENTS);
const SUPPORTED_VERIFY_WORKLOADS = new Set(TOOLING_VERIFY_WORKLOADS);

function normalizeWorkload(workload) {
  return String(workload || '').trim().toLowerCase();
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
  return null;
}

function mapWorkloadToCommand(workload) {
  if (workload === 'bench') return { command: 'bench', workload: 'inference' };
  if (workload === 'debug') return { command: 'debug', workload: 'inference' };
  if (SUPPORTED_VERIFY_WORKLOADS.has(workload)) {
    return { command: 'verify', workload };
  }
  throw new Error(`Unsupported diagnostics workload "${workload}"`);
}

export class DiagnosticsController {
  constructor(options = {}) {
    this.log = options.log || log;
    this.runCommand = options.runCommand || runBrowserCommand;
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

  async applyRuntimeProfile(profileId) {
    return applyRuntimeProfile(profileId);
  }

  async verifySuite(model, options = {}) {
    return this.runSuite(model, options);
  }

  async runSuite(model, options = {}) {
    const workload = normalizeWorkload(options.workload || options.suite || 'inference');
    const { modelId, modelUrl } = resolveModelRef(model, options);
    const mapped = mapWorkloadToCommand(workload);
    const runtimeConfig = resolveRuntimeConfig(options);
    if (options.configChain != null) {
      throw new Error(
        'Diagnostics controller does not accept configChain. Use runtimeProfile, runtimeConfigUrl, or runtimeConfig.',
      );
    }
    if (runtimeConfig) {
      this.requireIntent(runtimeConfig);
    }

    const response = await this.runCommand({
      command: mapped.command,
      workload: mapped.workload ?? undefined,
      modelId,
      modelUrl,
      runtimeProfile: options.runtimeProfile ?? null,
      runtimeConfigUrl: options.runtimeConfigUrl ?? null,
      runtimeConfig: runtimeConfig ?? null,
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
