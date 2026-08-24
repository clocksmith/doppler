import { TOOLING_COMMANDS } from '../tooling/command-api.js';
import { asStringOrNull } from './cli-model-resolution.js';

const COMMON_CLI_FLAGS = Object.freeze([
  'config',
  'surface',
  'pretty',
  'json',
  'help',
  'h',
  'runtime-config',
  'runtime-profile',
]);

export function parseCliArguments(argv) {
  const parsed = { command: null, action: null, flags: {} };
  if (!argv.length) return parsed;
  parsed.command = asStringOrNull(argv[0]);

  for (let index = 1; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '-h') {
      parsed.flags.h = true;
      continue;
    }
    if (token.startsWith('-') && !token.startsWith('--')) {
      throw new Error(`Unsupported short flag "${token}". Use long-form flags (for example --help).`);
    }
    if (!token.startsWith('--')) {
      if ((parsed.command === 'onboard' || parsed.command === 'boundary') && parsed.action === null) {
        parsed.action = token;
        continue;
      }
      throw new Error('Positional arguments are not supported. Use --config for command payloads.');
    }

    const key = token.slice(2);
    if (['json', 'pretty', 'help', 'h', 'skip-convert', 'skip-capture'].includes(key)) {
      parsed.flags[key] = true;
      continue;
    }
    const value = argv[index + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }
    parsed.flags[key] = value;
    index += 1;
  }
  return parsed;
}

function levenshteinDistance(a, b) {
  const source = String(a ?? '');
  const target = String(b ?? '');
  if (source === target) return 0;
  if (source.length === 0) return target.length;
  if (target.length === 0) return source.length;

  const previous = new Array(target.length + 1);
  const current = new Array(target.length + 1);
  for (let index = 0; index <= target.length; index += 1) previous[index] = index;
  for (let sourceIndex = 1; sourceIndex <= source.length; sourceIndex += 1) {
    current[0] = sourceIndex;
    for (let targetIndex = 1; targetIndex <= target.length; targetIndex += 1) {
      const cost = source[sourceIndex - 1] === target[targetIndex - 1] ? 0 : 1;
      current[targetIndex] = Math.min(
        previous[targetIndex] + 1,
        current[targetIndex - 1] + 1,
        previous[targetIndex - 1] + cost
      );
    }
    for (let targetIndex = 0; targetIndex <= target.length; targetIndex += 1) {
      previous[targetIndex] = current[targetIndex];
    }
  }
  return previous[target.length];
}

function findClosestFlag(flag, allowedFlags) {
  const normalizedFlag = String(flag ?? '').trim();
  if (!normalizedFlag) return null;
  let candidate = null;
  let distance = Infinity;
  for (const allowedFlag of allowedFlags) {
    const nextDistance = levenshteinDistance(normalizedFlag, allowedFlag);
    if (nextDistance < distance) {
      candidate = allowedFlag;
      distance = nextDistance;
    }
  }
  return distance <= 3 ? candidate : null;
}

function assertAllowedFlags(parsed, allowedFlags, commandLabel) {
  for (const key of Object.keys(parsed.flags || {})) {
    if (allowedFlags.has(key)) continue;
    throw new Error(`Unknown flag --${key} for "${commandLabel}".`);
  }
}

export function validateCommandFlags(parsed) {
  const command = parsed?.command;
  if (!command || !TOOLING_COMMANDS.includes(command)) return;
  const allowedFlags = new Set(command === 'release'
    ? [
      'config', 'manifest', 'action', 'out', 'repo-root', 'forge-config', 'target',
      'device-identity', 'fleet-receipts', 'pack-trusted-signers', 'fleet-trusted-signers',
      'signing-private-key', 'signing-public-key', 'signing-authority',
      'surface', 'pretty', 'json', 'help', 'h',
    ]
    : COMMON_CLI_FLAGS);
  for (const key of Object.keys(parsed.flags || {})) {
    if (allowedFlags.has(key)) continue;
    const suggestion = findClosestFlag(key, allowedFlags);
    if (suggestion) {
      throw new Error(`Unknown flag --${key} for "${command}". Did you mean --${suggestion}?`);
    }
    throw new Error(`Unknown flag --${key} for "${command}".`);
  }
}

export function validateProgramBundleFlags(parsed) {
  assertAllowedFlags(parsed, new Set([
    'config', 'manifest', 'model-dir', 'reference-report', 'conversion-config',
    'runtime-config', 'out', 'bundle-id', 'created-at', 'pretty', 'json', 'help', 'h',
  ]), 'program-bundle');
}

export function validateIntakeFlags(parsed) {
  assertAllowedFlags(parsed, new Set([
    'convert-config', 'manifest', 'model-dir', 'out', 'skip-convert',
    'pretty', 'json', 'help', 'h',
  ]), 'intake');
}

export function validateOnboardFlags(parsed) {
  if (parsed.action !== 'inspect') {
    throw new Error('onboard: expected the action "inspect".');
  }
  assertAllowedFlags(parsed, new Set([
    'source', 'out', 'family-intake', 'pretty', 'json', 'help', 'h',
  ]), 'onboard inspect');
  if (!asStringOrNull(parsed.flags.source)) {
    throw new Error('onboard inspect: --source <checkpoint-dir> is required.');
  }
  if (!asStringOrNull(parsed.flags.out)) {
    throw new Error('onboard inspect: --out <dir> is required.');
  }
}

export function validateBoundaryFlags(parsed) {
  const allowedByAction = {
    capture: new Set(['report', 'out', 'tolerance-policy', 'pretty', 'json', 'help', 'h']),
    compare: new Set([
      'source-pack', 'runtime-capture', 'source-control', 'token-evidence',
      'artifact-precision', 'out', 'pretty', 'json', 'help', 'h',
    ]),
    'token-evidence': new Set(['reference-transcript', 'out', 'pretty', 'json', 'help', 'h']),
    'source-pack': new Set(['provider-capture', 'out', 'pretty', 'json', 'help', 'h']),
  };
  const allowedFlags = Object.hasOwn(allowedByAction, parsed.action)
    ? allowedByAction[parsed.action]
    : null;
  if (!allowedFlags) {
    throw new Error('boundary: expected "capture", "source-pack", "token-evidence", or "compare".');
  }
  assertAllowedFlags(parsed, allowedFlags, `boundary ${parsed.action}`);
  if (!asStringOrNull(parsed.flags.out)) {
    throw new Error(`boundary ${parsed.action}: --out <path> is required.`);
  }
  if (parsed.action === 'capture' && !asStringOrNull(parsed.flags.report)) {
    throw new Error('boundary capture: --report <diagnose-report.json> is required.');
  }
  if (parsed.action === 'source-pack' && !asStringOrNull(parsed.flags['provider-capture'])) {
    throw new Error('boundary source-pack: --provider-capture <provider-capture.json> is required.');
  }
  if (parsed.action === 'token-evidence' && !asStringOrNull(parsed.flags['reference-transcript'])) {
    throw new Error('boundary token-evidence: --reference-transcript <reference-transcript.json> is required.');
  }
  if (parsed.action === 'compare') {
    for (const key of ['source-pack', 'runtime-capture', 'token-evidence']) {
      if (!asStringOrNull(parsed.flags[key])) {
        throw new Error(`boundary compare: --${key} <path> is required.`);
      }
    }
  }
}

export function validateBundleFlags(parsed) {
  assertAllowedFlags(parsed, new Set([
    'convert-config', 'manifest', 'model-dir', 'model-url', 'conversion-config',
    'prompt', 'max-tokens', 'surface', 'runtime-config', 'out', 'bundle-id',
    'created-at', 'skip-convert', 'skip-capture', 'reference-report',
    'reference-transcript', 'pretty', 'json', 'help', 'h',
  ]), 'bundle');
}

export function validateProfilesFlags(parsed) {
  assertAllowedFlags(parsed, new Set(['pretty', 'json', 'help', 'h']), 'profiles');
}
