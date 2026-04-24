import { isPlainObject } from '../utils/plain-object.js';
import {
  TOOLING_COMMAND_SET,
  TOOLING_SURFACE_SET,
  TOOLING_WORKLOAD_SET,
  VERIFY_WORKLOADS,
  TRAINING_COMMAND_SCHEMA_VERSION,
} from './command-api-constants.js';
import {
  asOptionalString,
  assertCommand,
} from './command-api-helpers.js';
import {
  normalizeConvert,
  normalizeRefreshIntegrity,
  normalizeTrainingOperatorCommand,
  normalizeSuiteCommand,
} from './command-api-family-normalizers.js';

export const TOOLING_COMMANDS = Object.freeze([...TOOLING_COMMAND_SET]);
export const TOOLING_SURFACES = Object.freeze([...TOOLING_SURFACE_SET]);
export const TOOLING_WORKLOADS = Object.freeze([...TOOLING_WORKLOAD_SET]);
export const TOOLING_VERIFY_WORKLOADS = Object.freeze([...VERIFY_WORKLOADS]);
export const TOOLING_TRAINING_COMMAND_SCHEMA_VERSION = TRAINING_COMMAND_SCHEMA_VERSION;

export function normalizeToolingCommandRequest(input) {
  if (!isPlainObject(input)) {
    throw new Error('tooling command: request must be an object.');
  }
  const command = assertCommand(input.command);
  if (command === 'convert') {
    return normalizeConvert(input);
  }
  if (command === 'refresh-integrity') {
    return normalizeRefreshIntegrity(input);
  }
  if (command === 'lora' || command === 'distill') {
    return normalizeTrainingOperatorCommand(input, command);
  }
  return normalizeSuiteCommand(input, command);
}

export function ensureCommandSupportedOnSurface(commandRequest, surface) {
  const request = normalizeToolingCommandRequest(commandRequest);
  const normalizedSurface = asOptionalString(surface, 'surface');
  if (!normalizedSurface || !TOOLING_SURFACE_SET.includes(normalizedSurface)) {
    throw new Error(`tooling command: unsupported surface "${surface}".`);
  }

  if (
    normalizedSurface === 'browser'
    && (
      request.command === 'lora'
      || request.command === 'distill'
      || request.command === 'diagnose'
      || request.command === 'refresh-integrity'
    )
  ) {
    throw new Error(`tooling command: ${request.command} is currently Node-only and must fail closed on browser.`);
  }

  return {
    request,
    surface: normalizedSurface,
  };
}
