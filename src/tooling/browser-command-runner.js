import {
  runBrowserSuite,
  applyRuntimePreset,
  applyRuntimeConfigFromUrl,
} from '../inference/browser-harness.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import {
  normalizeToolingCommandRequest,
  ensureCommandSupportedOnSurface,
} from './command-api.js';
import { applyRuntimeInputs, buildSuiteOptions } from './command-runner-shared.js';

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

  await applyRuntimeInputs(request, {
    applyRuntimePreset,
    applyRuntimeConfigFromUrl,
    getRuntimeConfig,
    setRuntimeConfig,
  }, options.runtimeLoadOptions || {});
  const result = await runBrowserSuite(buildSuiteOptions(request));

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
