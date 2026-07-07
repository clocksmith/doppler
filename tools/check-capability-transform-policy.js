#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { TRANSFORMS } from '../src/config/transforms/execution-graph-transforms.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const POLICY_PATH = 'src/rules/inference/capability-transforms.rules.json';
const POLICY_ABSOLUTE_PATH = path.join(REPO_ROOT, POLICY_PATH);

const VALID_KINDS = new Set([
  'hardware-compatibility',
  'runtime-session-compatibility',
  'explicit-lane',
  'platform-workaround',
  'lane-mismatch-guard',
  'capability-optimization',
  'default-noop',
]);

const VALID_DTYPE_EFFECTS = new Set([
  'none',
  'widen-to-f32',
  'full-f32',
  'narrow-to-f16',
  'selective-f16',
  'fail-closed',
]);

const MODEL_IDENTITY_KEYS = new Set(['modelId', 'manifestModelType']);
const FORBIDDEN_MODEL_IDENTITY_OPERATORS = new Set(['contains', 'startsWith', 'endsWith', 'neq']);
const ALLOWED_MODEL_IDENTITY_OPERATORS = new Set(['eq', 'in']);
const PLATFORM_KEYS = new Set(['platformVendor', 'platformId', 'platformArchitecture']);
const F32_TRANSFORMS = new Set(['widenToF32Activations', 'widenToF32CorrectnessFallback']);
const F16_TRANSFORMS = new Set([
  'narrowToF16Activations',
  'useQwenF16PrimaryMatmuls',
  'useQwen36F16Activations',
  'useGemma4Int4PleSelectiveF16Decode',
  'useGemma4TextF16Activations',
  'useGemma412BTextF16Activations',
  'useGemma431BTextF16Activations',
  'useGemma4Int4PleAf16Activations',
]);

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function formatRule(rule, index) {
  const kind = typeof rule?.kind === 'string' ? rule.kind : 'unknown-kind';
  const transforms = Array.isArray(rule?.transforms) ? rule.transforms.join('+') : 'unknown-transform';
  return `rule[${index}] ${kind}/${transforms}`;
}

function hasOwn(value, key) {
  return isPlainObject(value) && Object.prototype.hasOwnProperty.call(value, key);
}

function isEmptyObject(value) {
  return isPlainObject(value) && Object.keys(value).length === 0;
}

function hasAnyKey(value, keys) {
  if (!isPlainObject(value)) return false;
  return Object.keys(value).some((key) => keys.has(key));
}

function matchHasModelIdentity(match) {
  return hasAnyKey(match, MODEL_IDENTITY_KEYS);
}

function matchHasPlatformIdentity(match) {
  return hasAnyKey(match, PLATFORM_KEYS);
}

function containsTransform(rule, names) {
  return Array.isArray(rule.transforms) && rule.transforms.some((name) => names.has(name));
}

function normalizeEvidencePath(value) {
  if (typeof value !== 'string') return '';
  const normalized = value.trim();
  if (!normalized || path.isAbsolute(normalized) || normalized.includes('\\')) return '';
  if (normalized.split('/').includes('..')) return '';
  return normalized;
}

async function pathExists(repoRelativePath) {
  try {
    await fs.access(path.join(REPO_ROOT, repoRelativePath));
    return true;
  } catch {
    return false;
  }
}

function validateModelIdentityMatcher(matchValue, label, errors) {
  if (typeof matchValue === 'string') {
    if (!matchValue.trim()) {
      errors.push(`${label} must not be an empty string`);
    }
    return;
  }
  if (!isPlainObject(matchValue)) {
    errors.push(`${label} must be an exact string or an object with eq/in`);
    return;
  }
  const operators = Object.keys(matchValue);
  for (const operator of operators) {
    if (FORBIDDEN_MODEL_IDENTITY_OPERATORS.has(operator)) {
      errors.push(`${label} uses ${operator}; model identity rules must use exact values or explicit lists`);
      continue;
    }
    if (!ALLOWED_MODEL_IDENTITY_OPERATORS.has(operator)) {
      errors.push(`${label} uses unsupported operator ${operator}; allowed operators are eq and in`);
    }
  }
  if (hasOwn(matchValue, 'eq') && typeof matchValue.eq !== 'string') {
    errors.push(`${label}.eq must be a string`);
  }
  if (hasOwn(matchValue, 'in')) {
    if (!Array.isArray(matchValue.in) || matchValue.in.length === 0) {
      errors.push(`${label}.in must be a non-empty array`);
    } else {
      for (const entry of matchValue.in) {
        if (typeof entry !== 'string' || !entry.trim()) {
          errors.push(`${label}.in entries must be non-empty strings`);
        }
      }
    }
  }
}

function scanModelIdentityMatchers(value, keyPath, errors) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => scanModelIdentityMatchers(entry, keyPath.concat(String(index)), errors));
    return;
  }
  if (!isPlainObject(value)) return;
  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    if (MODEL_IDENTITY_KEYS.has(key)) {
      validateModelIdentityMatcher(child, nextPath.join('.'), errors);
    }
    scanModelIdentityMatchers(child, nextPath, errors);
  }
}

function validateDtypeEffect(rule, label, errors) {
  if (containsTransform(rule, F32_TRANSFORMS) && !['widen-to-f32', 'full-f32'].includes(rule.dtypeEffect)) {
    errors.push(`${label}: f32-widening transforms require dtypeEffect widen-to-f32 or full-f32`);
  }
  if (containsTransform(rule, F16_TRANSFORMS) && !['narrow-to-f16', 'selective-f16'].includes(rule.dtypeEffect)) {
    errors.push(`${label}: f16 transforms require dtypeEffect narrow-to-f16 or selective-f16`);
  }
  if (rule.transforms?.includes('failClosedLaneMismatch') && rule.dtypeEffect !== 'fail-closed') {
    errors.push(`${label}: failClosedLaneMismatch requires dtypeEffect fail-closed`);
  }
  if (rule.dtypeEffect === 'none') {
    if (containsTransform(rule, F32_TRANSFORMS) || containsTransform(rule, F16_TRANSFORMS) || rule.transforms?.includes('failClosedLaneMismatch')) {
      errors.push(`${label}: dtypeEffect none cannot use dtype-changing or fail-closed transforms`);
    }
  }
}

function validateKind(rule, index, errors) {
  const label = formatRule(rule, index);
  const match = rule.match;
  switch (rule.kind) {
    case 'hardware-compatibility':
      if (matchHasModelIdentity(match) || matchHasPlatformIdentity(match)) {
        errors.push(`${label}: hardware-compatibility must not match model or platform identity`);
      }
      if (match.hasF16 !== false && match.hasSubgroups !== false) {
        errors.push(`${label}: hardware-compatibility must match a missing hardware feature`);
      }
      if (match.hasF16 === false && !containsTransform(rule, F32_TRANSFORMS)) {
        errors.push(`${label}: hasF16=false hardware rules must include an f32-widening transform`);
      }
      if (match.hasSubgroups === false && !rule.transforms.includes('removeSubgroups')) {
        errors.push(`${label}: hasSubgroups=false hardware rules must include removeSubgroups`);
      }
      break;
    case 'runtime-session-compatibility':
      if (matchHasModelIdentity(match) || matchHasPlatformIdentity(match)) {
        errors.push(`${label}: runtime-session-compatibility must not match model or platform identity`);
      }
      if (match.kvDtype !== 'f32') {
        errors.push(`${label}: runtime-session-compatibility must be driven by resolved kvDtype=f32`);
      }
      if (!containsTransform(rule, F32_TRANSFORMS)) {
        errors.push(`${label}: runtime-session-compatibility must include an f32-widening transform`);
      }
      break;
    case 'explicit-lane':
      if (
        rule.dtypeEffect === 'selective-f16'
          ? (match.activationDtype !== 'f16' && match.requestedActivationDtype !== 'f16')
          : match.activationDtype !== 'f16'
      ) {
        errors.push(`${label}: explicit-lane must match activationDtype=f16 or selective requestedActivationDtype=f16`);
      }
      if (match.hasF16 !== true) {
        errors.push(`${label}: explicit-lane must require hasF16=true`);
      }
      if (!matchHasModelIdentity(match) && match.requiresF16ActivationNarrowing !== true) {
        errors.push(`${label}: generic explicit-lane rules must require requiresF16ActivationNarrowing=true`);
      }
      if (!containsTransform(rule, F16_TRANSFORMS)) {
        errors.push(`${label}: explicit-lane must include a named f16 transform`);
      }
      break;
    case 'platform-workaround':
      if (!matchHasPlatformIdentity(match)) {
        errors.push(`${label}: platform-workaround must match platform identity`);
      }
      if (!matchHasModelIdentity(match)) {
        errors.push(`${label}: platform-workaround must match exact model identity or an explicit model list`);
      }
      if (containsTransform(rule, F32_TRANSFORMS) || containsTransform(rule, F16_TRANSFORMS)) {
        errors.push(`${label}: platform-workaround must not silently change dtype`);
      }
      break;
    case 'lane-mismatch-guard':
      if (!matchHasModelIdentity(match)) {
        errors.push(`${label}: lane-mismatch-guard must match exact model identity`);
      }
      if (rule.transforms.length !== 1 || rule.transforms[0] !== 'failClosedLaneMismatch') {
        errors.push(`${label}: lane-mismatch-guard must use only failClosedLaneMismatch`);
      }
      break;
    case 'capability-optimization':
      if (!matchHasModelIdentity(match)) {
        errors.push(`${label}: capability-optimization must match exact model identity or an explicit model list`);
      }
      if (match.hasF16 !== true) {
        errors.push(`${label}: capability-optimization must declare the positive capability it depends on`);
      }
      if (rule.dtypeEffect !== 'none') {
        errors.push(`${label}: capability-optimization must not change dtypeEffect`);
      }
      break;
    case 'default-noop':
      if (!isEmptyObject(match)) {
        errors.push(`${label}: default-noop must use an empty match`);
      }
      if (rule.transforms.length !== 0 || rule.dtypeEffect !== 'none') {
        errors.push(`${label}: default-noop must not transform dtype or graph`);
      }
      break;
    default:
      errors.push(`${label}: unknown kind ${JSON.stringify(rule.kind)}`);
  }
}

async function validateEvidence(rule, index, errors) {
  const label = formatRule(rule, index);
  if (!Array.isArray(rule.evidence)) {
    errors.push(`${label}: evidence must be an array`);
    return;
  }
  if (rule.kind === 'default-noop') {
    if (rule.evidence.length !== 0) {
      errors.push(`${label}: default-noop evidence must be empty`);
    }
    return;
  }
  if (rule.evidence.length === 0) {
    errors.push(`${label}: non-default rules require evidence`);
    return;
  }
  if (!rule.evidence.some((entry) => typeof entry === 'string' && entry.startsWith('tests/'))) {
    errors.push(`${label}: evidence must include at least one tests/ path`);
  }
  for (const rawPath of rule.evidence) {
    const normalized = normalizeEvidencePath(rawPath);
    if (!normalized) {
      errors.push(`${label}: evidence path must be repo-relative: ${JSON.stringify(rawPath)}`);
      continue;
    }
    if (!await pathExists(normalized)) {
      errors.push(`${label}: evidence path does not exist: ${normalized}`);
    }
  }
}

function validateRuleShape(rule, index, errors) {
  const label = formatRule(rule, index);
  if (!isPlainObject(rule)) {
    errors.push(`rule[${index}] must be an object`);
    return;
  }
  if (!VALID_KINDS.has(rule.kind)) {
    errors.push(`${label}: kind must be one of ${Array.from(VALID_KINDS).join(', ')}`);
  }
  if (!VALID_DTYPE_EFFECTS.has(rule.dtypeEffect)) {
    errors.push(`${label}: dtypeEffect must be one of ${Array.from(VALID_DTYPE_EFFECTS).join(', ')}`);
  }
  if (!isPlainObject(rule.match)) {
    errors.push(`${label}: match must be an object`);
  }
  if (!Array.isArray(rule.transforms)) {
    errors.push(`${label}: transforms must be an array`);
  } else {
    const seenTransforms = new Set();
    for (const transformName of rule.transforms) {
      if (typeof transformName !== 'string' || !transformName.trim()) {
        errors.push(`${label}: transforms must contain non-empty strings`);
        continue;
      }
      if (seenTransforms.has(transformName)) {
        errors.push(`${label}: duplicate transform ${transformName}`);
      }
      seenTransforms.add(transformName);
      if (!TRANSFORMS[transformName]) {
        errors.push(`${label}: unknown transform ${transformName}`);
      }
    }
  }
  if (typeof rule.reason !== 'string' || !rule.reason.trim()) {
    errors.push(`${label}: reason is required`);
  }
}

async function readPolicy() {
  return JSON.parse(await fs.readFile(POLICY_ABSOLUTE_PATH, 'utf8'));
}

export async function buildCapabilityTransformPolicyReport() {
  const errors = [];
  const policy = await readPolicy();
  if (!isPlainObject(policy)) {
    return {
      ok: false,
      policyPath: POLICY_PATH,
      rules: 0,
      kinds: {},
      errors: ['policy must be an object'],
    };
  }
  if (policy.$schema !== '../../config/schema/capability-transform-policy.schema.json') {
    errors.push('policy $schema must be ../../config/schema/capability-transform-policy.schema.json');
  }
  if (policy.schemaVersion !== 1) {
    errors.push('policy schemaVersion must be 1');
  }
  if (policy.source !== 'doppler') {
    errors.push('policy source must be "doppler"');
  }
  const rules = Array.isArray(policy.capabilityTransforms) ? policy.capabilityTransforms : [];
  if (rules.length === 0) {
    errors.push('policy capabilityTransforms must be a non-empty array');
  }

  let defaultNoopCount = 0;
  const kinds = {};
  for (let index = 0; index < rules.length; index++) {
    const rule = rules[index];
    const kind = typeof rule?.kind === 'string' ? rule.kind : 'unknown-kind';
    kinds[kind] = (kinds[kind] ?? 0) + 1;
    if (kind === 'default-noop') {
      defaultNoopCount++;
      if (index !== rules.length - 1) {
        errors.push(`${formatRule(rule, index)}: default-noop must be the last rule`);
      }
    }
    validateRuleShape(rule, index, errors);
    if (isPlainObject(rule?.match)) {
      scanModelIdentityMatchers(rule.match, [`rule[${index}]`, 'match'], errors);
    }
    if (isPlainObject(rule) && isPlainObject(rule.match) && Array.isArray(rule.transforms)) {
      validateDtypeEffect(rule, formatRule(rule, index), errors);
      validateKind(rule, index, errors);
    }
    await validateEvidence(rule, index, errors);
  }
  if (defaultNoopCount !== 1) {
    errors.push(`policy must contain exactly one default-noop rule, found ${defaultNoopCount}`);
  }

  return {
    ok: errors.length === 0,
    policyPath: POLICY_PATH,
    rules: rules.length,
    kinds,
    errors,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) {
    throw new Error(`Unknown argument: ${unsupported[0]}`);
  }
  const report = await buildCapabilityTransformPolicyReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(`capability-transform-policy: ok (${report.rules} rules)`);
  } else {
    for (const error of report.errors) {
      console.error(`capability-transform-policy: ${error}`);
    }
  }
  if (!report.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
