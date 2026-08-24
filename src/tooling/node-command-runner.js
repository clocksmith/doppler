import {
  normalizeToolingCommandRequest,
  ensureCommandSupportedOnSurface,
} from './command-api.js';
import {
  createToolingSuccessEnvelope,
  normalizeToToolingCommandError,
} from './command-envelope.js';
import { assertCommandRequestIsObject, normalizeCommandOptions } from './command-validation.js';
import { runNodeCommandExecution } from './node-command-execution.js';
import { checkProgramBundleParity } from './program-bundle-parity.js';

export { hasNodeWebGPUSupport } from './node-command-execution.js';

export async function runNodeCommand(commandRequest, options = {}) {
  assertCommandRequestIsObject(commandRequest, 'node');
  const validatedOptions = normalizeCommandOptions(options, 'node');
  let request = null;
  try {
    ({ request } = ensureCommandSupportedOnSurface(commandRequest, 'node'));
  } catch (error) {
    throw normalizeToToolingCommandError(error, {
      surface: 'node',
      request,
    });
  }

  if (
    request.command !== 'verify'
    || request.workload !== 'inference'
    || request.workloadType !== 'program-bundle'
  ) {
    return runNodeCommandExecution(commandRequest, validatedOptions);
  }

  try {
    const result = await checkProgramBundleParity({
      bundle: request.programBundle ?? undefined,
      bundlePath: request.programBundlePath ?? undefined,
      providers: request.parityProviders,
      mode: request.programBundleParityMode,
    });
    return createToolingSuccessEnvelope({
      surface: 'node',
      request,
      result,
    });
  } catch (error) {
    throw normalizeToToolingCommandError(error, {
      surface: 'node',
      request,
    });
  }
}

export function normalizeNodeCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
