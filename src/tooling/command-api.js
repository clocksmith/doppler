import { isPlainObject } from '../utils/plain-object.js';
import {
  TOOLING_COMMAND_SET,
  TOOLING_SURFACE_SET,
  TOOLING_SUITE_SET,
  VERIFY_SUITES,
  TRAINING_COMMAND_SCHEMA_VERSION,
} from './command-api-constants.js';
import {
  asOptionalString,
  assertCommand,
} from './command-api-helpers.js';
import {
  normalizeConvert,
  normalizeTrainingOperatorCommand,
  normalizeSuiteCommand,
} from './command-api-family-normalizers.js';

export const TOOLING_COMMANDS = Object.freeze([...TOOLING_COMMAND_SET]);
export const TOOLING_SURFACES = Object.freeze([...TOOLING_SURFACE_SET]);
export const TOOLING_SUITES = Object.freeze([...TOOLING_SUITE_SET]);
export const TOOLING_VERIFY_SUITES = Object.freeze([...VERIFY_SUITES]);
export const TOOLING_TRAINING_COMMAND_SCHEMA_VERSION = TRAINING_COMMAND_SCHEMA_VERSION;

function resolveHarnessMode(request) {
  if (request.command === 'debug') {
    return 'debug';
  }
  if (request.command === 'bench') {
    return 'bench';
  }
  return request.suite;
}

export function normalizeToolingCommandRequest(input) {
  if (!isPlainObject(input)) {
    throw new Error('tooling command: request must be an object.');
  }
  const command = assertCommand(input.command);
  if (command === 'convert') {
    return normalizeConvert(input);
  }
  if (command === 'lora' || command === 'distill') {
    return normalizeTrainingOperatorCommand(input, command);
  }
  return normalizeSuiteCommand(input, command);
}

export function buildRuntimeContractPatch(commandRequest) {
  const request = normalizeToolingCommandRequest(commandRequest);
  if (!request.suite || !request.intent) {
    return null;
  }
  const harnessMode = resolveHarnessMode(request);

  return {
    shared: {
      harness: {
        mode: harnessMode,
        modelId: request.modelId ?? null,
      },
      tooling: {
        intent: request.intent,
      },
    },
  };
}

export function ensureCommandSupportedOnSurface(commandRequest, surface) {
  const request = normalizeToolingCommandRequest(commandRequest);
  const normalizedSurface = asOptionalString(surface, 'surface');
  if (!normalizedSurface || !TOOLING_SURFACE_SET.includes(normalizedSurface)) {
    throw new Error(`tooling command: unsupported surface "${surface}".`);
  }

  if (
    normalizedSurface === 'browser'
    && (request.command === 'lora' || request.command === 'distill')
  ) {
    throw new Error(`tooling command: ${request.command} is currently Node-only and must fail closed on browser.`);
  }

  return {
    request,
    surface: normalizedSurface,
  };
}
