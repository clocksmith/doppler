import {
  runBrowserSuite,
  applyRuntimePreset,
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
  const { request } = ensureCommandSupportedOnSurface(commandRequest, 'browser');

  if (request.command === 'convert') {
    if (typeof options.convertHandler !== 'function') {
      throw new Error(
        'browser command convert requires options.convertHandler(request) to be provided.'
      );
    }
    const result = await options.convertHandler(request);
    return {
      ok: true,
      surface: 'browser',
      request,
      result,
    };
  }

  const runtimeBridge = {
    applyRuntimePreset,
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
    await applyRuntimeInputs(request, runtimeBridge, options.runtimeLoadOptions || {});
    return runBrowserSuite(buildSuiteOptions(request, 'browser'));
  });

  return {
    ok: true,
    surface: 'browser',
    request,
    result,
  };
}

export function normalizeBrowserCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
