import { isPlainObject } from '../utils/plain-object.js';

const TOOLING_COMMAND_SET = ['convert', 'debug', 'bench', 'test-model'];
const TOOLING_SURFACE_SET = ['browser', 'node'];
const TOOLING_SUITE_SET = ['kernels', 'inference', 'bench', 'debug', 'diffusion', 'energy'];
const VERIFY_SUITES = ['kernels', 'inference', 'diffusion', 'energy'];

export const TOOLING_COMMANDS = Object.freeze([...TOOLING_COMMAND_SET]);
export const TOOLING_SURFACES = Object.freeze([...TOOLING_SURFACE_SET]);
export const TOOLING_SUITES = Object.freeze([...TOOLING_SUITE_SET]);

const COMMAND_RUNTIME_CONTRACT = Object.freeze({
  debug: Object.freeze({ suite: 'debug', intent: 'investigate' }),
  bench: Object.freeze({ suite: 'bench', intent: 'calibrate' }),
  'test-model': Object.freeze({ suite: null, intent: 'verify' }),
});

function asOptionalString(value, label) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value !== 'string') {
    throw new Error(`tooling command: ${label} must be a string when provided.`);
  }
  const trimmed = value.trim();
  return trimmed || null;
}

function asOptionalBoolean(value, label) {
  if (value === undefined) return null;
  if (typeof value !== 'boolean') {
    throw new Error(`tooling command: ${label} must be a boolean when provided.`);
  }
  return value;
}

function asOptionalObject(value, label) {
  if (value === undefined || value === null) return null;
  if (!isPlainObject(value)) {
    throw new Error(`tooling command: ${label} must be an object when provided.`);
  }
  return value;
}

function assertCommand(value) {
  const command = asOptionalString(value, 'command');
  if (!command) {
    throw new Error('tooling command: command is required.');
  }
  if (!TOOLING_COMMAND_SET.includes(command)) {
    throw new Error(`tooling command: unsupported command "${command}".`);
  }
  return command;
}

function assertSuite(value, command) {
  const suite = asOptionalString(value, 'suite');
  if (!suite) {
    throw new Error(`tooling command: suite is required for "${command}".`);
  }
  if (!TOOLING_SUITE_SET.includes(suite)) {
    throw new Error(`tooling command: unsupported suite "${suite}".`);
  }
  return suite;
}

function assertModelId(value, command, suite) {
  const modelId = asOptionalString(value, 'modelId');
  if (!modelId) {
    throw new Error(
      `tooling command: modelId is required for command "${command}" (suite "${suite}").`
    );
  }
  return modelId;
}

function normalizeConvertPayload(value) {
  const payload = asOptionalObject(value, 'convertPayload');
  if (!payload) {
    throw new Error(
      'tooling command: convert requires convertPayload.converterConfig.'
    );
  }
  const converterConfig = asOptionalObject(
    payload.converterConfig,
    'convertPayload.converterConfig'
  );
  if (!converterConfig) {
    throw new Error(
      'tooling command: convert requires convertPayload.converterConfig.'
    );
  }
  return {
    ...payload,
    converterConfig,
  };
}

function normalizeConvert(raw) {
  const inputDir = asOptionalString(raw.inputDir, 'inputDir');
  const outputDir = asOptionalString(raw.outputDir, 'outputDir');
  const modelId = asOptionalString(raw.modelId, 'modelId');
  const payload = normalizeConvertPayload(raw.convertPayload);

  if (!inputDir) {
    throw new Error(
      'tooling command: convert requires inputDir.'
    );
  }
  if (modelId) {
    throw new Error(
      'tooling command: convert does not accept modelId. Set convertPayload.converterConfig.output.modelId.'
    );
  }

  return {
    command: 'convert',
    suite: null,
    intent: null,
    modelId: null,
    modelUrl: asOptionalString(raw.modelUrl, 'modelUrl'),
    runtimePreset: asOptionalString(raw.runtimePreset, 'runtimePreset'),
    runtimeConfigUrl: asOptionalString(raw.runtimeConfigUrl, 'runtimeConfigUrl'),
    runtimeConfig: asOptionalObject(raw.runtimeConfig, 'runtimeConfig'),
    inputDir,
    outputDir,
    convertPayload: payload,
    captureOutput: false,
    keepPipeline: false,
    report: asOptionalObject(raw.report, 'report'),
    timestamp: raw.timestamp ?? null,
    searchParams: raw.searchParams ?? null,
  };
}

function normalizeSuiteCommand(raw, command) {
  const runtimeContract = COMMAND_RUNTIME_CONTRACT[command];
  let suite = runtimeContract.suite;
  if (!suite) {
    suite = assertSuite(raw.suite, command);
    if (!VERIFY_SUITES.includes(suite)) {
      throw new Error(
        `tooling command: "${command}" suite must be one of ${VERIFY_SUITES.join(', ')}.`
      );
    }
  }

  const requiresModel = suite !== 'kernels';
  const modelId = requiresModel
    ? assertModelId(raw.modelId, command, suite)
    : asOptionalString(raw.modelId, 'modelId');

  return {
    command,
    suite,
    intent: runtimeContract.intent,
    modelId,
    modelUrl: asOptionalString(raw.modelUrl, 'modelUrl'),
    runtimePreset: asOptionalString(raw.runtimePreset, 'runtimePreset'),
    runtimeConfigUrl: asOptionalString(raw.runtimeConfigUrl, 'runtimeConfigUrl'),
    runtimeConfig: asOptionalObject(raw.runtimeConfig, 'runtimeConfig'),
    inputDir: null,
    outputDir: null,
    convertPayload: null,
    captureOutput: asOptionalBoolean(raw.captureOutput, 'captureOutput') ?? false,
    keepPipeline: asOptionalBoolean(raw.keepPipeline, 'keepPipeline') ?? false,
    report: asOptionalObject(raw.report, 'report'),
    timestamp: raw.timestamp ?? null,
    searchParams: raw.searchParams ?? null,
  };
}

export function normalizeToolingCommandRequest(input) {
  if (!isPlainObject(input)) {
    throw new Error('tooling command: request must be an object.');
  }
  const command = assertCommand(input.command);
  if (command === 'convert') {
    return normalizeConvert(input);
  }
  return normalizeSuiteCommand(input, command);
}

export function buildRuntimeContractPatch(commandRequest) {
  const request = normalizeToolingCommandRequest(commandRequest);
  if (!request.suite || !request.intent) {
    return null;
  }

  return {
    shared: {
      harness: {
        mode: request.suite,
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

  // All commands are contractually available on both surfaces.
  // Surface-specific capability checks happen in the runners.
  return {
    request,
    surface: normalizedSurface,
  };
}
