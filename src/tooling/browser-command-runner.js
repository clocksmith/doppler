import {
  applyRuntimeProfile,
  applyRuntimeConfigFromUrl,
} from '../inference/browser-harness-runtime-helpers.js';
import {
  getRuntimeConfig,
  setRuntimeConfig,
  resetRuntimeConfig,
} from '../config/runtime.js';
import {
  normalizeToolingCommandRequest,
  ensureCommandSupportedOnSurface,
} from './command-api.js';
import {
  createToolingSuccessEnvelope,
  normalizeToToolingCommandError,
} from './command-envelope.js';
import { assertCommandRequestIsObject, normalizeCommandOptions } from './command-validation.js';
import {
  applyRuntimeInputs,
  buildSuiteOptions,
  runWithRuntimeIsolation,
} from './command-runner-shared.js';
import {
  getActiveKernelPath,
  getActiveKernelPathPolicy,
  getActiveKernelPathSource,
  setActiveKernelPath,
} from '../config/kernel-path-loader.js';
import { validateProgramBundle } from '../config/schema/program-bundle.schema.js';

let browserHarnessModulePromise = null;

async function loadBrowserHarnessModule() {
  browserHarnessModulePromise ??= import('../inference/browser-harness.js');
  return browserHarnessModulePromise;
}

export async function runBrowserCommand(commandRequest, options = {}) {
  assertCommandRequestIsObject(commandRequest, 'browser');
  const validatedOptions = normalizeCommandOptions(options, 'browser');
  let request = null;
  try {
    ({ request } = ensureCommandSupportedOnSurface(commandRequest, 'browser'));

    if (request.command === 'convert') {
      if (typeof validatedOptions.convertHandler !== 'function') {
        throw new Error(
          'browser command convert requires options.convertHandler(request) to be provided.'
        );
      }
      const result = await validatedOptions.convertHandler(request);
      return createToolingSuccessEnvelope({
        surface: 'browser',
        request,
        result,
      });
    }

    if (
      request.command === 'verify'
      && request.workload === 'inference'
      && request.workloadType === 'program-bundle'
    ) {
      if (request.programBundlePath) {
        throw new Error('browser command: program-bundle parity requires inline programBundle; programBundlePath is Node-only.');
      }
      const providers = request.parityProviders ?? ['browser-webgpu'];
      const unsupported = providers.filter((provider) => provider !== 'browser-webgpu');
      if (unsupported.length > 0) {
        throw new Error(
          `browser command: program-bundle parity provider(s) ${unsupported.join(', ')} are Node-only.`
        );
      }
      const bundle = validateProgramBundle(request.programBundle);
      const result = {
        schema: 'doppler.program-bundle-parity/v1',
        ok: true,
        mode: 'contract',
        bundleId: bundle.bundleId,
        modelId: bundle.modelId,
        executionGraphHash: bundle.sources.executionGraph.hash,
        providers: [
          {
            provider: 'browser-webgpu',
            status: 'reference',
            ok: true,
          },
        ],
      };
      return createToolingSuccessEnvelope({
        surface: 'browser',
        request,
        result,
      });
    }

    const runtimeBridge = {
      applyRuntimeProfile,
      applyRuntimeConfigFromUrl,
      getRuntimeConfig,
      setRuntimeConfig,
      resetRuntimeConfig,
      getActiveKernelPath,
      getActiveKernelPathPolicy,
      getActiveKernelPathSource,
      setActiveKernelPath,
    };

    const result = await runWithRuntimeIsolation(runtimeBridge, async () => {
      const { runBrowserSuite } = await loadBrowserHarnessModule();
      await applyRuntimeInputs(request, runtimeBridge, validatedOptions.runtimeLoadOptions || {});
      return runBrowserSuite(buildSuiteOptions(request, 'browser'));
    });

    return createToolingSuccessEnvelope({
      surface: 'browser',
      request,
      result,
    });
  } catch (error) {
    throw normalizeToToolingCommandError(error, {
      surface: 'browser',
      request,
    });
  }
}

export function normalizeBrowserCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
