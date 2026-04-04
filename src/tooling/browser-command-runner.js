import {
  runBrowserSuite,
  applyRuntimeProfile,
  applyRuntimeConfigFromUrl,
} from '../inference/browser-harness.js';
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
