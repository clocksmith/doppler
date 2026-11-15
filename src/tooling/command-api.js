import { isPlainObject } from '../formats/plain-object.js';
import {
  TOOLING_COMMAND_SET,
  TOOLING_SURFACE_SET,
  NODE_ONLY_TOOLING_COMMANDS,
  TOOLING_WORKLOAD_SET,
  VERIFY_WORKLOADS,
  TRAINING_COMMAND_SCHEMA_VERSION,
} from './command-api-constants.js';
import {
  asOptionalString,
  assertCommand,
} from './command-request-normalization.js';
import {
  normalizeConvert,
  normalizeRefreshIntegrity,
  normalizeRelease,
  normalizeTrainingOperatorCommand,
  normalizeSuiteCommand,
} from './command-api-family-normalizers.js';

export const TOOLING_COMMANDS = Object.freeze([...TOOLING_COMMAND_SET]);
export const TOOLING_SURFACES = Object.freeze([...TOOLING_SURFACE_SET]);
export const TOOLING_WORKLOADS = Object.freeze([...TOOLING_WORKLOAD_SET]);
export const TOOLING_VERIFY_WORKLOADS = Object.freeze([...VERIFY_WORKLOADS]);
export const TOOLING_TRAINING_COMMAND_SCHEMA_VERSION = TRAINING_COMMAND_SCHEMA_VERSION;

const COMMAND_NORMALIZERS = Object.freeze({
  convert: normalizeConvert,
  'refresh-integrity': normalizeRefreshIntegrity,
  release: normalizeRelease,
  lora: normalizeTrainingOperatorCommand,
  distill: normalizeTrainingOperatorCommand,
  debug: normalizeSuiteCommand,
  bench: normalizeSuiteCommand,
  verify: normalizeSuiteCommand,
  diagnose: normalizeSuiteCommand,
});

export function normalizeToolingCommandRequest(input) {
  if (!isPlainObject(input)) {
    throw new Error('tooling command: request must be an object.');
  }
  const command = assertCommand(input.command);
  return COMMAND_NORMALIZERS[command](input, command);
}

export function ensureCommandSupportedOnSurface(commandRequest, surface) {
  const request = normalizeToolingCommandRequest(commandRequest);
  const normalizedSurface = asOptionalString(surface, 'surface');
  if (!normalizedSurface || !TOOLING_SURFACE_SET.includes(normalizedSurface)) {
    throw new Error(`tooling command: unsupported surface "${surface}".`);
  }

  if (
    normalizedSurface === 'browser'
    && NODE_ONLY_TOOLING_COMMANDS.includes(request.command)
  ) {
    throw new Error(`tooling command: ${request.command} is currently Node-only and must fail closed on browser.`);
  }

  return {
    request,
    surface: normalizedSurface,
  };
}
