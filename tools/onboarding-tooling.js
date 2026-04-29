#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { resolveConvertedModelId } from '../src/converter/conversion-plan.js';
import { buildQuantizationInfo } from '../src/converter/quantization-info.js';
import { checkProgramBundleFile } from '../src/tooling/program-bundle.js';
import { generateWgslVariants } from './wgsl-variant-generator.js';
import {
  isObject,
  normalizeTrimmedText,
  resolvePolicyText,
  resolveText,
  resolveTextArray,
} from './utils/policy-utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const WARN = 'warning';
const ERROR = 'error';
const REQUIRED_COMPARE_TIMING_METRICS = Object.freeze([
  'decodeTokensPerSec',
  'promptTokensPerSecToFirstToken',
  'firstTokenMs',
  'firstResponseMs',
  'decodeMs',
  'decodeMsPerTokenP50',
  'decodeMsPerTokenP95',
  'decodeMsPerTokenP99',
  'totalRunMs',
  'modelLoadMs',
]);
const RUNTIME_NON_PROFILE_SCHEMAS = new Set([
  'drift-policies-v1',
]);
const ONBOARDING_POLICY_PATH = fileURLToPath(new URL('./policies/onboarding-tooling-policy.json', import.meta.url));
const DEFAULT_ONBOARDING_POLICY = Object.freeze({
  modes: ['check', 'scaffold'],
  scaffoldKinds: ['conversion', 'kernel', 'behavior'],
  defaults: {
    root: {
      path: 'REPO_ROOT',
    },
    conversion: {
      family: 'custom',
      baseDir: 'models/local',
      quantization: {
        weights: 'f16',
        embeddings: 'weights',
        lmHead: 'embeddings',
      },
    },
    kernelPath: {
      activationDtype: 'f16',
      kvDtype: 'f16',
      statusDefault: 'experimental',
      statusReason: 'scaffolded',
      excludeStatuses: ['legacy'],
      descriptionTemplate: 'Scaffolded kernel path for {id}.',
    },
    behavior: {
      scope: 'profiles',
    },
    errors: {
      unsupportedMode: 'Unsupported mode',
      unsupportedModeHint: 'Expected one of: {supportedModes}',
      unsupportedKind: 'Unsupported scaffold kind',
      unsupportedKindHint: 'Expected one of: {supportedKinds}',
      missingId: '--id is required for scaffold',
      missingKind: '--kind is required for scaffold',
    },
  },
  paths: {
    conversion: 'src/config/conversion',
    kernelPath: 'src/config/transforms',
    runtimeProfile: 'src/config/runtime',
  },
});
let activeOnboardingPolicy = DEFAULT_ONBOARDING_POLICY;

function policyDefaultsFromActivePolicy(policy = getActivePolicy()) {
  return policy.defaults || {};
}

function resolveOnboardingPolicy(payload) {
  const input = isObject(payload) ? payload : {};
  const defaults = isObject(input.defaults) ? input.defaults : {};
  const paths = isObject(input.paths) ? input.paths : {};
  const conversionDefaults = isObject(defaults.conversion) ? defaults.conversion : {};
  const kernelDefaults = isObject(defaults.kernelPath) ? defaults.kernelPath : {};
  const behaviorDefaults = isObject(defaults.behavior) ? defaults.behavior : {};
  const rootDefaults = isObject(defaults.root) ? defaults.root : {};
  const errorDefaults = isObject(defaults.errors) ? defaults.errors : {};
  const quantizationDefaults = isObject(conversionDefaults.quantization) ? conversionDefaults.quantization : {};

  return {
    modes: resolveTextArray(input.modes, DEFAULT_ONBOARDING_POLICY.modes),
    scaffoldKinds: resolveTextArray(input.scaffoldKinds, DEFAULT_ONBOARDING_POLICY.scaffoldKinds),
    defaults: {
      root: {
        path: resolvePolicyText(rootDefaults.path, DEFAULT_ONBOARDING_POLICY.defaults.root.path),
      },
      conversion: {
        family: resolvePolicyText(conversionDefaults.family, DEFAULT_ONBOARDING_POLICY.defaults.conversion.family),
        baseDir: resolvePolicyText(conversionDefaults.baseDir, DEFAULT_ONBOARDING_POLICY.defaults.conversion.baseDir),
        quantization: {
          weights: resolvePolicyText(quantizationDefaults.weights, DEFAULT_ONBOARDING_POLICY.defaults.conversion.quantization.weights),
          embeddings: resolvePolicyText(quantizationDefaults.embeddings, DEFAULT_ONBOARDING_POLICY.defaults.conversion.quantization.embeddings),
          lmHead: resolvePolicyText(quantizationDefaults.lmHead, DEFAULT_ONBOARDING_POLICY.defaults.conversion.quantization.lmHead),
        },
      },
      kernelPath: {
        activationDtype: resolvePolicyText(kernelDefaults.activationDtype, DEFAULT_ONBOARDING_POLICY.defaults.kernelPath.activationDtype),
        kvDtype: resolvePolicyText(kernelDefaults.kvDtype, DEFAULT_ONBOARDING_POLICY.defaults.kernelPath.kvDtype),
        statusDefault: resolvePolicyText(kernelDefaults.statusDefault, DEFAULT_ONBOARDING_POLICY.defaults.kernelPath.statusDefault),
        statusReason: resolvePolicyText(kernelDefaults.statusReason, DEFAULT_ONBOARDING_POLICY.defaults.kernelPath.statusReason),
        excludeStatuses: resolveTextArray(kernelDefaults.excludeStatuses, DEFAULT_ONBOARDING_POLICY.defaults.kernelPath.excludeStatuses),
        descriptionTemplate: resolvePolicyText(kernelDefaults.descriptionTemplate, DEFAULT_ONBOARDING_POLICY.defaults.kernelPath.descriptionTemplate),
      },
      behavior: {
        scope: resolvePolicyText(behaviorDefaults.scope, DEFAULT_ONBOARDING_POLICY.defaults.behavior.scope),
      },
      errors: {
        unsupportedMode: resolvePolicyText(errorDefaults.unsupportedMode, DEFAULT_ONBOARDING_POLICY.defaults.errors.unsupportedMode),
        unsupportedModeHint: resolvePolicyText(errorDefaults.unsupportedModeHint, DEFAULT_ONBOARDING_POLICY.defaults.errors.unsupportedModeHint),
        unsupportedKind: resolvePolicyText(errorDefaults.unsupportedKind, DEFAULT_ONBOARDING_POLICY.defaults.errors.unsupportedKind),
        unsupportedKindHint: resolvePolicyText(errorDefaults.unsupportedKindHint, DEFAULT_ONBOARDING_POLICY.defaults.errors.unsupportedKindHint),
        missingId: resolvePolicyText(errorDefaults.missingId, DEFAULT_ONBOARDING_POLICY.defaults.errors.missingId),
        missingKind: resolvePolicyText(errorDefaults.missingKind, DEFAULT_ONBOARDING_POLICY.defaults.errors.missingKind),
      },
    },
    paths: {
      conversion: resolvePolicyText(paths.conversion, DEFAULT_ONBOARDING_POLICY.paths.conversion),
      kernelPath: resolvePolicyText(paths.kernelPath, DEFAULT_ONBOARDING_POLICY.paths.kernelPath),
      runtimeProfile: resolvePolicyText(paths.runtimeProfile, DEFAULT_ONBOARDING_POLICY.paths.runtimeProfile),
    },
  };
}

function setActivePolicy(payload) {
  activeOnboardingPolicy = resolveOnboardingPolicy(payload);
}

function getActivePolicy() {
  return activeOnboardingPolicy;
}

function usage() {
  return [
    'Usage:',
    '  node tools/onboarding-tooling.js check [--root <dir>] [--strict] [--json]',
    '  node tools/onboarding-tooling.js scaffold --kind <conversion|kernel|behavior> --id <id> [options]',
    '',
    'Check options:',
    '  --root <dir>               Repo root (default: current repo).',
    '  --strict                    Treat warnings as failures.',
    '  --json                      Emit JSON report.',
    '',
    'Scaffold options:',
    '  --kind <kind>               conversion | kernel | behavior',
    '  --id <id>                   Artifact id / file stem.',
    '  --force                     Overwrite destination file.',
    '  --output <path>             Custom output path.',
    '  --family <name>             Conversion family folder (conversion only).',
    '  --base-dir <path>           Conversion output base dir (conversion only).',
    '  --status <status>           Kernel registry status hint.',
    '  --status-reason <text>      Optional status reason.',
    '  --scope <dir>               Runtime profile scope directory.',
    '',
    'Examples:',
    '  node tools/onboarding-tooling.js check --strict',
    '  node tools/onboarding-tooling.js scaffold --kind conversion --id gemma3-my-new-model --family gemma3',
    '  node tools/onboarding-tooling.js scaffold --kind kernel --id gemma3-f16-fused-f16a-online-audit --status experimental',
    '  node tools/onboarding-tooling.js scaffold --kind behavior --id super-flash-memory-15-0',
    '',
    'Notes:',
    '  The script is a tooling helper, not a runtime path.',
  ].join('\n');
}

function toJsonText(value) {
  return JSON.stringify(value, null, 2);
}

function assertString(value, label, options = {}) {
  if (typeof value !== 'string' || value.trim() === '') {
    if (options.required) {
      throw new Error(`${label} must be a non-empty string`);
    }
    return false;
  }
  return true;
}

function normalizeId(rawValue, options = {}) {
  const value = normalizeTrimmedText(rawValue).toLowerCase();
  if (!value) return options.defaultValue;
  return value.replace(/[^a-z0-9._-]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

function toIssue(severity, code, location, message, hint) {
  const hintText = typeof hint === 'string' && hint.length > 0 ? hint : null;
  return {
    severity,
    code,
    location,
    message,
    hint: hintText,
  };
}

function issueList(severity, issues) {
  return issues.filter((issue) => issue.severity === severity);
}

function issueCounts(issues) {
  return {
    errors: issueList(ERROR, issues).length,
    warnings: issueList(WARN, issues).length,
  };
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJson(filePath, expectedType = 'object', opts = {}) {
  const raw = await fs.readFile(filePath, 'utf-8');
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`Invalid JSON in ${filePath}: ${error.message}`);
  }
  if (expectedType === 'object' && !isObject(parsed)) {
    throw new Error(`Expected object in ${filePath}`);
  }
  if (expectedType === 'array' && !Array.isArray(parsed)) {
    throw new Error(`Expected array in ${filePath}`);
  }
  if (opts.requireString && !assertString(parsed, opts.requireString)) {
    throw new Error(`Invalid payload in ${filePath}: expected string`);
  }
  return parsed;
}

async function collectJsonFiles(rootDir) {
  const files = [];
  const entries = (await fs.readdir(rootDir, { withFileTypes: true }))
    .sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      const children = await collectJsonFiles(fullPath);
      files.push(...children);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(fullPath);
    }
  }
  return files;
}

function safeTrim(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function resolveConversionConfigModelId(config, policy = getActivePolicy()) {
  const modelBaseId = safeTrim(config?.output?.modelBaseId);
  if (!modelBaseId) {
    return {
      modelId: null,
      error: 'output.modelBaseId is required',
    };
  }
  const conversionDefaults = isObject(policy.defaults?.conversion) ? policy.defaults.conversion : {};
  const conversionQuant = isObject(conversionDefaults.quantization) ? conversionDefaults.quantization : {};
  const weightsDefault = resolvePolicyText(conversionQuant.weights, 'f16');
  const embeddingsDefault = resolvePolicyText(conversionQuant.embeddings, 'weights');
  const lmHeadDefault = resolvePolicyText(conversionQuant.lmHead, 'embeddings');

  const weights = resolvePolicyText(config?.quantization?.weights, weightsDefault);
  const embeddingsByPolicy = embeddingsDefault === 'weights'
    ? weights
    : resolvePolicyText(config?.quantization?.embeddings, embeddingsDefault);
  const embeddings = resolvePolicyText(config?.quantization?.embeddings, embeddingsByPolicy);
  const lmHeadByPolicy = lmHeadDefault === 'embeddings'
    ? embeddings
    : resolvePolicyText(config?.quantization?.lmHead, lmHeadDefault);
  const lmHead = resolvePolicyText(config?.quantization?.lmHead, lmHeadByPolicy);

  try {
    buildQuantizationInfo(
      { converterConfig: config },
      weights,
      embeddings,
      lmHead
    );
  } catch (error) {
    return {
      modelId: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }

  const modelId = resolveConvertedModelId({
    explicitModelId: modelBaseId,
    converterConfig: config,
  });
  if (!modelId) {
    return {
      modelId: null,
      error: 'failed to resolve modelId from output.modelBaseId + quantization config',
    };
  }
  return { modelId, error: null };
}

function toSafeJsIdentifier(value) {
  const tokens = String(value)
    .trim()
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
  if (!tokens.length) return 'modelConfig';
  const head = tokens[0].replace(/^[^a-z]+/, '') || 'model';
  const tail = tokens.slice(1).map((token) => token[0].toUpperCase() + token.slice(1)).join('');
  const identifier = `${head}${tail}`;
  return /^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(identifier)
    ? identifier
    : `model_${identifier.replace(/[^a-z0-9_]/gi, '_')}`;
}

function toRuntimeProfileId(runtimeProfileRoot, filePath) {
  const relativePath = path.relative(runtimeProfileRoot, filePath);
  return relativePath.replace(/\\/g, '/').replace(/\.json$/i, '');
}

const KNOWN_BOOLEAN_FLAGS = new Set(['strict', 'json', 'force']);
const KNOWN_VALUE_FLAGS = new Set([
  'root', 'kind', 'id', 'output', 'family',
  'base-dir', 'status', 'status-reason', 'scope',
]);

function parseCommandLine(argv) {
  if (!argv.length) {
    return { mode: 'help' };
  }
  const mode = argv[0];
  const flags = {};

  for (let i = 1; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) {
      throw new Error(`Unexpected positional argument: ${JSON.stringify(token)}`);
    }
    const key = token.slice(2);
    if (key === 'help' || key === 'h') {
      return { mode: 'help' };
    }
    if (KNOWN_BOOLEAN_FLAGS.has(key)) {
      flags[key] = true;
      continue;
    }
    if (KNOWN_VALUE_FLAGS.has(key)) {
      const value = argv[i + 1];
      if (value == null || value.startsWith('--')) {
        throw new Error(`Missing value for --${key}`);
      }
      flags[key] = value;
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag: --${key}`);
  }
  return { mode, flags, positional: [] };
}

async function detectLoaderConfigIds() {
  // Loader-owned model-family config registries have been removed.
  return {
    ids: new Set(),
    detectionOrder: [],
    error: null,
  };
}

function collectStringSetFromArray(values) {
  const output = new Set();
  if (!Array.isArray(values)) return output;
  for (const value of values) {
    if (assertString(value)) {
      output.add(value.trim());
    }
  }
  return output;
}

function validateSinglePathValue(fieldPath, paths, key, issues, code = 'HARNESS_PATHS_NON_CANONICAL') {
  const candidates = paths?.[key];
  if (!Array.isArray(candidates)) {
    issues.push(toIssue(ERROR, code, fieldPath, `"${key}" must be an array of canonical paths`));
    return;
  }
  if (candidates.length < 1) {
    issues.push(toIssue(
      ERROR,
      code,
      fieldPath,
      `"${key}" must map to at least one canonical path`
    ));
    return;
  }
  for (const [index, candidate] of candidates.entries()) {
    if (!assertString(candidate)) {
      issues.push(toIssue(ERROR, code, fieldPath, `"${key}" canonical path #${index + 1} must be a non-empty string`));
    }
  }
}

function validateStringList(fieldPath, values, issues, code) {
  if (values == null) return;
  if (!Array.isArray(values)) {
    issues.push(toIssue(
      ERROR,
      code || 'JSON_ARRAY_EXPECTED',
      fieldPath,
      'must be an array of non-empty strings'
    ));
    return;
  }
  for (const item of values) {
    if (!assertString(item)) {
      issues.push(toIssue(
        ERROR,
        code || 'JSON_STRING_REQUIRED',
        fieldPath,
        'all items must be non-empty strings'
      ));
      return;
    }
  }
}

function toSnakeCase(value) {
  return String(value ?? '')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/-/g, '_')
    .toLowerCase();
}

function collectDirectRuleStringValues(ruleEntries) {
  const values = new Set();
  if (!Array.isArray(ruleEntries)) {
    return values;
  }
  for (const entry of ruleEntries) {
    if (!isObject(entry)) continue;
    const rawValue = entry.value;
    if (typeof rawValue !== 'string') continue;
    const normalized = rawValue.trim();
    if (normalized) {
      values.add(normalized);
    }
  }
  return values;
}

function inferOperationFromRuleName(baseOperation, ruleName, operations, variantValues) {
  if (!isObject(operations)) return null;
  const operationIds = new Set(Object.keys(operations));
  if (operationIds.size === 0) return null;

  const candidates = [];
  const appendCandidate = (value) => {
    if (!assertString(value)) return;
    const trimmed = value.trim();
    if (!trimmed || candidates.includes(trimmed)) return;
    candidates.push(trimmed);
  };

  const normalizedRuleName = String(ruleName ?? '');
  if (normalizedRuleName === 'variant') {
    appendCandidate(baseOperation);
  } else if (/variant$/i.test(normalizedRuleName) && !/suffix$/i.test(normalizedRuleName)) {
    const stem = normalizedRuleName.slice(0, -'Variant'.length);
    const suffix = toSnakeCase(stem);
    appendCandidate(`${baseOperation}_${suffix}`);
    appendCandidate(suffix);
    appendCandidate(baseOperation);
  } else {
    return null;
  }

  const scoreOperation = (operationId) => {
    const variants = isObject(operations[operationId]?.variants) ? operations[operationId].variants : {};
    const variantIds = new Set(Object.keys(variants));
    let missing = 0;
    for (const value of variantValues.values()) {
      if (!variantIds.has(value)) {
        missing += 1;
      }
    }
    return { candidate: operationId, missing };
  };

  const scored = candidates
    .filter((candidate) => operationIds.has(candidate))
    .map((candidate) => scoreOperation(candidate));

  if (!scored.length) {
    const fallback = Array.from(operationIds.values()).map((operationId) => scoreOperation(operationId));
    fallback.sort((left, right) => left.missing - right.missing || left.candidate.localeCompare(right.candidate));
    return fallback[0]?.missing === 0 ? fallback[0].candidate : null;
  }

  scored.sort((left, right) => {
    if (left.missing !== right.missing) {
      return left.missing - right.missing;
    }
    return candidates.indexOf(left.candidate) - candidates.indexOf(right.candidate);
  });

  const localBest = scored[0];
  if (localBest.missing === 0) {
    return localBest.candidate;
  }

  const globalBest = Array.from(operationIds.values())
    .map((operationId) => scoreOperation(operationId))
    .sort((left, right) => left.missing - right.missing || left.candidate.localeCompare(right.candidate))[0];
  if (!globalBest) {
    return localBest.candidate;
  }
  if (globalBest.missing < localBest.missing) {
    return globalBest.candidate;
  }
  return localBest.candidate;
}

async function validateKernelRuleVariantParity(root, issues, context) {
  const rulesDir = path.join(root, 'src/rules/kernels');
  const kernelRegistryPath = path.join(root, 'src/config/kernels/registry.json');

  let kernelRegistry;
  try {
    kernelRegistry = await readJson(kernelRegistryPath, 'object');
  } catch (error) {
    issues.push(toIssue(ERROR, 'KERNEL_RULE_REGISTRY_READ', kernelRegistryPath, error.message));
    return;
  }

  const operations = isObject(kernelRegistry?.operations) ? kernelRegistry.operations : null;
  if (!operations) {
    issues.push(toIssue(
      ERROR,
      'KERNEL_RULE_REGISTRY_FORMAT',
      kernelRegistryPath,
      'kernel registry operations must be an object'
    ));
    return;
  }

  let entries = [];
  try {
    entries = await fs.readdir(rulesDir, { withFileTypes: true });
  } catch (error) {
    issues.push(toIssue(ERROR, 'KERNEL_RULE_READ', rulesDir, error.message));
    return;
  }

  let checkedRuleSets = 0;
  let checkedVariants = 0;
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isFile()) continue;
    if (!entry.name.endsWith('.rules.json')) continue;

    const stem = entry.name.slice(0, -'.rules.json'.length);
    const baseOperation = stem.replace(/-/g, '_');
    const rulesPath = path.join(rulesDir, entry.name);
    let rulesPayload;
    try {
      rulesPayload = await readJson(rulesPath, 'object');
    } catch (error) {
      issues.push(toIssue(ERROR, 'KERNEL_RULE_JSON', rulesPath, error.message));
      continue;
    }

    for (const [ruleName, ruleEntries] of Object.entries(rulesPayload)) {
      if (!/variant$/i.test(ruleName) || /suffix$/i.test(ruleName)) {
        continue;
      }
      const variantValues = collectDirectRuleStringValues(ruleEntries);
      if (variantValues.size === 0) {
        continue;
      }

      checkedRuleSets += 1;
      const operationId = inferOperationFromRuleName(baseOperation, ruleName, operations, variantValues);
      if (!operationId) {
        issues.push(toIssue(
          WARN,
          'KERNEL_RULE_VARIANT_OPERATION_UNKNOWN',
          `${rulesPath}::${ruleName}`,
          `could not map kernel rule set to a registry operation (baseOperation="${baseOperation}")`
        ));
        continue;
      }

      const variants = isObject(operations?.[operationId]?.variants)
        ? operations[operationId].variants
        : null;
      if (!variants) {
        issues.push(toIssue(
          ERROR,
          'KERNEL_RULE_OPERATION_VARIANTS_MISSING',
          `${kernelRegistryPath}::${operationId}`,
          `operation "${operationId}" does not define variants`
        ));
        continue;
      }

      for (const value of variantValues.values()) {
        checkedVariants += 1;
        if (!(value in variants)) {
          issues.push(toIssue(
            ERROR,
            'KERNEL_RULE_VARIANT_MISSING',
            `${rulesPath}::${ruleName}`,
            `rule variant "${value}" is not declared in kernel registry operation "${operationId}"`
          ));
        }
      }
    }
  }

  context.kernelRuleVariantParity = {
    ruleSets: checkedRuleSets,
    variants: checkedVariants,
  };
}

async function validateGeneratedWgsl(root, issues, context) {
  let report;
  try {
    report = await generateWgslVariants({
      rootDir: root,
      checkOnly: true,
    });
  } catch (error) {
    issues.push(toIssue(
      ERROR,
      'WGSL_GENERATED_CHECK_FAILED',
      path.join(root, 'src/gpu/kernels/codegen/wgsl-variants.js'),
      error.message
    ));
    return;
  }

  for (const errorMessage of report.errors) {
    issues.push(toIssue(
      ERROR,
      'WGSL_GENERATED_INVALID',
      path.join(root, 'src/gpu/kernels/codegen/wgsl-variants.js'),
      errorMessage
    ));
  }

  for (const target of report.changedTargets) {
    issues.push(toIssue(
      ERROR,
      'WGSL_GENERATED_DRIFT',
      path.join(root, target),
      'generated WGSL file is out of date',
      'Run `npm run kernels:codegen:sync`.'
    ));
  }

  context.generatedWgsl = {
    variants: report.variantCount,
    drift: report.changedCount,
  };
}

async function validateKernelPathRegistry(root, issues, context, policy = getActivePolicy()) {
  // Kernel path registry was replaced by execution graph transforms (a5d5e601).
  // This validation is now a no-op; execution graphs are validated at conversion
  // config level and at runtime via compileExecutionV1.
  return { ids: new Set(), statusById: new Map() };
  /* eslint-disable-next-line no-unreachable -- retained for git history */
  const registryPath = path.join(root, 'src/config/kernel-paths/registry.json');
  let registryPayload;
  try {
    registryPayload = await readJson(registryPath, 'object');
  } catch (error) {
    issues.push(toIssue(ERROR, 'REGISTRY_READ', registryPath, error.message));
    return { ids: new Set(), statusById: new Map() };
  }

  const entries = Array.isArray(registryPayload?.entries) ? registryPayload.entries : [];
  if (!entries.length) {
    issues.push(toIssue(ERROR, 'REGISTRY_EMPTY', registryPath, 'kernel path registry has no entries'));
  }

  const ids = new Set();
  const statusById = new Map();
  const entryById = new Map();
  const seen = new Set();

  for (const [index, entry] of entries.entries()) {
    if (!isObject(entry)) {
      issues.push(toIssue(ERROR, 'REGISTRY_ENTRY_FORMAT', registryPath, `entries[${index}] must be an object`));
      continue;
    }
    const id = normalizeTrimmedText(entry.id);
    if (!id) {
      issues.push(toIssue(ERROR, 'REGISTRY_ID_MISSING', registryPath, `entries[${index}] has no id`));
      continue;
    }
    if (seen.has(id)) {
      issues.push(toIssue(ERROR, 'REGISTRY_DUPLICATE_ID', registryPath, `duplicate registry entry "${id}"`));
      continue;
    }
    seen.add(id);
    ids.add(id);
    const defaultStatus = resolveText(policy.defaults?.kernelPath?.statusDefault, 'canonical');
    const status = normalizeTrimmedText(entry.status);
    statusById.set(id, status || defaultStatus);
    entryById.set(id, entry);

    if (!assertString(entry.file) && !assertString(entry.aliasOf)) {
      issues.push(toIssue(
        ERROR,
        'REGISTRY_ENTRY_TARGET',
        `${registryPath}::${id}`,
        `registry entry "${id}" must define file or aliasOf`
      ));
      continue;
    }

    if (entry.file && entry.aliasOf) {
      issues.push(toIssue(
        WARN,
        'REGISTRY_ENTRY_BOTH',
        `${registryPath}::${id}`,
        `registry entry "${id}" defines both file and aliasOf`
      ));
    }
  }

  for (const [id, entry] of entryById.entries()) {
    if (assertString(entry.aliasOf)) {
      if (!ids.has(String(entry.aliasOf).trim())) {
        issues.push(toIssue(
          ERROR,
          'REGISTRY_ALIAS_MISSING',
          `${registryPath}::${id}`,
          `aliasOf target "${entry.aliasOf}" does not exist`
        ));
      }
      continue;
    }

    const kernelFile = path.join(root, 'src/config/kernel-paths', String(entry.file).trim());
    if (!(await fileExists(kernelFile))) {
      issues.push(toIssue(
        ERROR,
        'KERNEL_PATH_FILE_MISSING',
        `${registryPath}::${id}`,
        `kernel file "${entry.file}" not found`
      ));
      continue;
    }
    let kernelPathPayload;
    try {
      kernelPathPayload = await readJson(kernelFile, 'object');
    } catch (error) {
      issues.push(toIssue(ERROR, 'KERNEL_PATH_INVALID', kernelFile, error.message));
      continue;
    }

    const payloadId = normalizeTrimmedText(kernelPathPayload.id);
    if (payloadId !== id) {
      issues.push(toIssue(
        WARN,
        'KERNEL_PATH_ID_MISMATCH',
        kernelFile,
        `id="${payloadId}" does not match registry id "${id}"`
      ));
    }

    const activationDtype = resolveText(kernelPathPayload.activationDtype, policy.defaults?.kernelPath?.activationDtype);
    if (!activationDtype) {
      issues.push(toIssue(ERROR, 'KERNEL_PATH_MISSING_DTYPE', kernelFile, 'activationDtype is required'));
    }
    const decodeSteps = kernelPathPayload.decode?.steps;
    if (!Array.isArray(decodeSteps) || decodeSteps.length === 0) {
      issues.push(toIssue(WARN, 'KERNEL_PATH_STEPS', `${kernelFile}.decode`, 'decode.steps missing or empty'));
    }
    const prefillSteps = kernelPathPayload.prefill?.steps;
    if (!Array.isArray(prefillSteps) || prefillSteps.length === 0) {
      issues.push(toIssue(WARN, 'KERNEL_PATH_STEPS', `${kernelFile}.prefill`, 'prefill.steps missing or empty'));
    }

    const validateSteps = async (sectionName, steps) => {
      if (!Array.isArray(steps)) return;
      for (let index = 0; index < steps.length; index += 1) {
        const step = steps[index];
        if (!isObject(step)) {
          issues.push(toIssue(
            ERROR,
            'KERNEL_STEP_FORMAT',
            `${kernelFile}.${sectionName}[${index}]`,
            'step must be an object'
          ));
          continue;
        }
        if (!assertString(step.op)) {
          issues.push(toIssue(
            ERROR,
            'KERNEL_STEP_MISSING_OP',
            `${kernelFile}.${sectionName}[${index}]`,
            'step.op is required'
          ));
        }
        if (!assertString(step.kernel)) {
          issues.push(toIssue(
            ERROR,
            'KERNEL_STEP_MISSING_KERNEL',
            `${kernelFile}.${sectionName}[${index}]`,
            'step.kernel is required'
          ));
        } else {
          const kernelSource = path.join(root, 'src/gpu/kernels', String(step.kernel));
          if (!(await fileExists(kernelSource))) {
            issues.push(toIssue(
              WARN,
              'KERNEL_FILE_MISSING',
              `${kernelFile}.${sectionName}[${index}]`,
              `kernel file "${step.kernel}" not found under src/gpu/kernels`
            ));
          }
        }
      }
    };
    await validateSteps('decode', decodeSteps);
    await validateSteps('prefill', prefillSteps);
    await validateSteps('preLayer', kernelPathPayload.preLayer);
    await validateSteps('postLayer', kernelPathPayload.postLayer);
    await validateSteps('sampling', kernelPathPayload.sampling);
  }

  return { ids, statusById };
}

async function validateModelFamilies(root, issues, context) {
  context.modelFamilyIds = new Set();
  context.modelFamiliesWithoutLoaderConfig = new Set();
  context.modelFamiliesMissingDetectionOrder = new Set();
}

async function validateConversionConfigs(root, issues, context, policy = getActivePolicy()) {
  const conversionRoot = path.join(root, 'src/config/conversion');
  const files = await collectJsonFiles(conversionRoot);
  const modelIds = new Set();

  for (const filePath of files) {
    let config;
    try {
      config = await readJson(filePath, 'object');
    } catch (error) {
      issues.push(toIssue(ERROR, 'CONVERSION_JSON', filePath, error.message));
      continue;
    }

    if (isLeanExecutionFixtureMap(config)) {
      continue;
    }

    const resolved = resolveConversionConfigModelId(config, policy);
    if (!resolved.modelId) {
      issues.push(toIssue(ERROR, 'CONVERSION_MODEL_ID', filePath, resolved.error));
    } else if (modelIds.has(resolved.modelId)) {
      issues.push(toIssue(
        ERROR,
        'CONVERSION_DUPLICATE_MODEL_ID',
        filePath,
        `duplicate resolved modelId "${resolved.modelId}"`
      ));
    } else {
      modelIds.add(resolved.modelId);
    }

    if (!assertString(config.output?.baseDir, `${filePath}.output.baseDir`)) {
      issues.push(toIssue(WARN, 'CONVERSION_BASEDIR', filePath, 'output.baseDir should be a non-empty string'));
    }

    if (config.inference?.defaultKernelPath !== undefined) {
      issues.push(toIssue(
        ERROR,
        'CONVERSION_DEFAULT_KERNEL_REMOVED',
        filePath,
        'inference.defaultKernelPath has been removed. Use inference.execution as the sole dispatch contract.'
      ));
    }
  }

  context.conversionModelIds = modelIds;
}

function isLeanExecutionFixtureMap(config) {
  return config?.schemaVersion === 1
    && config?.source === 'doppler'
    && Array.isArray(config?.mappings)
    && Array.isArray(config?.exclusions)
    && config?.family === undefined
    && config?.output === undefined;
}

async function validateRuntimeProfiles(root, issues, context) {
  const runtimeProfileRoot = path.join(root, 'src/config/runtime');
  const files = await collectJsonFiles(runtimeProfileRoot);
  const profilesById = new Map();
  const ids = new Set();

  for (const filePath of files) {
    let profileData;
    try {
      profileData = await readJson(filePath, 'object');
    } catch (error) {
      issues.push(toIssue(ERROR, 'RUNTIME_PROFILE_JSON', filePath, error.message));
      continue;
    }

    const profileId = toRuntimeProfileId(runtimeProfileRoot, filePath);
    if (RUNTIME_NON_PROFILE_SCHEMAS.has(String(profileData.$schema ?? ''))) {
      continue;
    }

    if (profilesById.has(profileId)) {
      issues.push(toIssue(ERROR, 'RUNTIME_PROFILE_DUP', filePath, `duplicate runtime profile id "${profileId}"`));
      continue;
    }

    if (!assertString(profileData.name, `${filePath}.name`)) {
      issues.push(toIssue(WARN, 'RUNTIME_PROFILE_NAME_MISSING', filePath, `runtime profile "${profileId}" missing name`));
    }

    if (profileData.extends != null && !assertString(profileData.extends, `${filePath}.extends`)) {
      issues.push(toIssue(
        ERROR,
        'RUNTIME_PROFILE_EXTENDS_FORMAT',
        filePath,
        `extends must be a non-empty string when provided`
      ));
    }

    if (!isObject(profileData.runtime)) {
      issues.push(toIssue(
        WARN,
        'RUNTIME_PROFILE_RUNTIME_MISSING',
        filePath,
        `runtime profile "${profileId}" does not define runtime`
      ));
    } else {
      validateRuntimeProfileKernelPath(filePath, profileData.runtime, issues);
    }

    profilesById.set(profileId, { filePath, profileData });
    ids.add(profileId);
  }

  const visiting = new Set();
  const visited = new Set();
  const stack = [];
  const validateChain = (profileId) => {
    if (visited.has(profileId)) {
      return;
    }
    if (visiting.has(profileId)) {
      const loopStart = stack.indexOf(profileId);
      const cycle = stack.slice(loopStart).concat(profileId);
      issues.push(toIssue(
        ERROR,
        'RUNTIME_PROFILE_EXTENDS_CYCLE',
        `${runtimeProfileRoot}/${profileId}.json`,
        `extends cycle detected: ${cycle.join(' -> ')}`
      ));
      return;
    }

    const entry = profilesById.get(profileId);
    if (!entry) {
      return;
    }
    const parent = safeTrim(entry.profileData.extends);

    visiting.add(profileId);
    stack.push(profileId);

    if (parent.length > 0) {
      if (!profilesById.has(parent)) {
        issues.push(toIssue(
          ERROR,
          'RUNTIME_PROFILE_EXTENDS_MISSING',
          entry.filePath,
          `extends "${parent}" is not a known runtime profile`
        ));
      } else {
        validateChain(parent);
      }
    }

    stack.pop();
    visiting.delete(profileId);
    visited.add(profileId);
  };

  for (const profileId of profilesById.keys()) {
    validateChain(profileId);
  }

  context.runtimeProfileIds = ids;
}

function validateRuntimeProfileKernelPath(filePath, runtime, issues) {
  const inference = isObject(runtime.inference) ? runtime.inference : null;
  if (!inference || !Object.prototype.hasOwnProperty.call(inference, 'kernelPath')) {
    return;
  }
  const kernelPath = inference.kernelPath;
  if (kernelPath === null) {
    return;
  }
  if (typeof kernelPath === 'string') {
    issues.push(toIssue(
      ERROR,
      'RUNTIME_PROFILE_STRING_KERNEL_PATH',
      filePath,
      'runtime.inference.kernelPath no longer accepts string registry IDs; use an inline execution-v1-derived kernelPath object or omit the field'
    ));
    return;
  }
  if (!isObject(kernelPath)) {
    issues.push(toIssue(
      ERROR,
      'RUNTIME_PROFILE_KERNEL_PATH_FORMAT',
      filePath,
      'runtime.inference.kernelPath must be an inline object or null'
    ));
  }
}

async function validateCompareConfigs(root, issues, context) {
  const compareConfigPath = path.join(root, 'benchmarks/vendors/compare-engines.config.json');
  const compareMetricsPath = path.join(root, 'benchmarks/vendors/compare-metrics.json');
  const benchmarkPolicyPath = path.join(root, 'benchmarks/vendors/benchmark-policy.json');
  const dopplerHarnessPath = path.join(root, 'benchmarks/vendors/harnesses/doppler.json');
  const tjsHarnessPath = path.join(root, 'benchmarks/vendors/harnesses/transformersjs.json');

  let compareConfig;
  try {
    compareConfig = await readJson(compareConfigPath, 'object');
  } catch (error) {
    issues.push(toIssue(ERROR, 'COMPARE_CONFIG_READ', compareConfigPath, error.message));
    return;
  }
  let compareMetrics;
  try {
    compareMetrics = await readJson(compareMetricsPath, 'object');
  } catch (error) {
    issues.push(toIssue(ERROR, 'COMPARE_METRICS_READ', compareMetricsPath, error.message));
    return;
  }

  let dopplerHarness;
  let tjsHarness;
  try {
    dopplerHarness = await readJson(dopplerHarnessPath, 'object');
  } catch (error) {
    issues.push(toIssue(ERROR, 'HARNESS_DOPPLER_READ', dopplerHarnessPath, error.message));
  }
  try {
    tjsHarness = await readJson(tjsHarnessPath, 'object');
  } catch (error) {
    issues.push(toIssue(ERROR, 'HARNESS_TJS_READ', tjsHarnessPath, error.message));
  }
  let benchmarkPolicy = null;
  try {
    benchmarkPolicy = await readJson(benchmarkPolicyPath, 'object');
  } catch {
    benchmarkPolicy = null;
  }
  const knownBadByModel = benchmarkPolicy?.kernelPathPolicy?.knownBadByModel;
  const benchmarkPolicyKnownBadModels = new Set(
    isObject(knownBadByModel) ? Object.keys(knownBadByModel) : []
  );

  const compareProfiles = Array.isArray(compareConfig.modelProfiles)
    ? compareConfig.modelProfiles
    : [];
  if (!compareProfiles.length) {
    issues.push(toIssue(WARN, 'COMPARE_NO_PROFILES', compareConfigPath, 'no modelProfiles configured'));
  }
  const profileIds = new Set();
  const noConfigProfiles = new Set();
  for (const [index, profile] of compareProfiles.entries()) {
    if (!isObject(profile)) {
      issues.push(toIssue(ERROR, 'COMPARE_PROFILE_FORMAT', compareConfigPath, `modelProfiles[${index}] must be an object`));
      continue;
    }
    const modelId = String(profile.dopplerModelId ?? '').trim();
    if (!modelId) {
      issues.push(toIssue(ERROR, 'COMPARE_PROFILE_MISSING_ID', compareConfigPath, `modelProfiles[${index}] missing dopplerModelId`));
      continue;
    }
    if (profileIds.has(modelId)) {
      issues.push(toIssue(ERROR, 'COMPARE_PROFILE_DUP_ID', compareConfigPath, `duplicate model profile "${modelId}"`));
      continue;
    }
    profileIds.add(modelId);
    const missingConfigOwner = (
      !(context?.conversionModelIds?.has(modelId)
        || context?.modelFamilyIds?.has(modelId)
        || context?.loaderConfigIds?.has(modelId))
    );
    if (missingConfigOwner && !benchmarkPolicyKnownBadModels.has(modelId)) {
      noConfigProfiles.add(modelId);
      issues.push(toIssue(
        WARN,
        'COMPARE_PROFILE_NO_MATCH',
        `${compareConfigPath}::${modelId}`,
        `compare profile "${modelId}" has no matching conversion config or loader-owned config`
      ));
    }
    if (Object.prototype.hasOwnProperty.call(profile, 'defaultKernelPath')) {
      issues.push(toIssue(
        ERROR,
        'COMPARE_PROFILE_DEFAULT_KERNEL_REMOVED',
        `${compareConfigPath}::${modelId}`,
        'compare profiles must not declare defaultKernelPath. Use manifest execution or an explicit CLI override.'
      ));
    }
  }

  if (!Array.isArray(compareMetrics.metrics)) {
    issues.push(toIssue(ERROR, 'COMPARE_METRICS_FORMAT', compareMetricsPath, 'metrics must be an array'));
    return;
  }

  const compareMetricIds = new Set();
  for (const [index, metric] of compareMetrics.metrics.entries()) {
    if (!isObject(metric)) {
      issues.push(toIssue(ERROR, 'COMPARE_METRIC_FORMAT', `${compareMetricsPath}[${index}]`, 'metric entry must be object'));
      continue;
    }
    const metricId = String(metric.id ?? '').trim();
    if (!metricId) {
      issues.push(toIssue(ERROR, 'COMPARE_METRIC_ID_MISSING', `${compareMetricsPath}[${index}]`, 'metric.id required'));
      continue;
    }
    if (compareMetricIds.has(metricId)) {
      issues.push(toIssue(ERROR, 'COMPARE_METRIC_DUP', `${compareMetricsPath}[${index}]`, `duplicate metric id "${metricId}"`));
      continue;
    }
    compareMetricIds.add(metricId);
  }
  const compareMetricById = Object.fromEntries(
    compareMetrics.metrics
      .filter((metric) => metric && typeof metric === 'object' && !Array.isArray(metric))
      .map((metric) => [String(metric.id || '').trim(), metric])
  );
  for (const metricId of REQUIRED_COMPARE_TIMING_METRICS) {
    const metric = compareMetricById[metricId];
    if (!metric) {
      issues.push(toIssue(ERROR, 'COMPARE_METRIC_REQUIRED_MISSING', compareMetricsPath, `Missing required canonical metric "${metricId}"`));
      continue;
    }
    if (metric.required !== true) {
      issues.push(toIssue(
        ERROR,
        'COMPARE_METRIC_REQUIRED_FLAG',
        `${compareMetricsPath}::${metricId}`,
        `Canonical metric "${metricId}" must be marked required`
      ));
    }
  }

  const dopplerPaths = dopplerHarness?.normalization?.metricPaths;
  const tjsPaths = tjsHarness?.normalization?.metricPaths;
  const dopplerRequired = collectStringSetFromArray(dopplerHarness?.normalization?.requiredMetrics);
  const tjsRequired = collectStringSetFromArray(tjsHarness?.normalization?.requiredMetrics);

  if (!isObject(dopplerPaths)) {
    issues.push(toIssue(ERROR, 'HARNESS_PATHS_MISSING', dopplerHarnessPath, 'normalization.metricPaths missing'));
  }
  if (!isObject(tjsPaths)) {
    issues.push(toIssue(ERROR, 'HARNESS_PATHS_MISSING', tjsHarnessPath, 'normalization.metricPaths missing'));
  }

  for (const metricId of compareMetricIds.values()) {
    validateSinglePathValue(
      `${compareMetricsPath}::${metricId}`,
      dopplerPaths,
      metricId,
      issues,
      'COMPARE_METRIC_NO_DOPPLER_PATH'
    );
    validateSinglePathValue(
      `${compareMetricsPath}::${metricId}`,
      tjsPaths,
      metricId,
      issues,
      'COMPARE_METRIC_NO_TJS_PATH'
    );
  }

  for (const metricId of dopplerRequired.values()) {
    if (!compareMetricIds.has(metricId)) {
      issues.push(toIssue(
        WARN,
        'HARNESS_REQUIRED_METRIC_MISSING',
        dopplerHarnessPath,
        `required metric "${metricId}" not present in compare metrics`
      ));
    }
  }
  for (const metricId of tjsRequired.values()) {
    if (!compareMetricIds.has(metricId)) {
      issues.push(toIssue(
        WARN,
        'HARNESS_REQUIRED_METRIC_MISSING',
        tjsHarnessPath,
        `required metric "${metricId}" not present in compare metrics`
      ));
    }
  }

  for (const [metaKey] of Object.entries(dopplerHarness?.normalization?.metadataPaths || {})) {
    validateSinglePathValue(
      `${dopplerHarnessPath}::metadata.${metaKey}`,
      dopplerHarness.normalization.metadataPaths,
      metaKey,
      issues,
      'HARNESS_METADATA_PATHS_NON_CANONICAL'
    );
  }
  for (const [metaKey] of Object.entries(tjsHarness?.normalization?.metadataPaths || {})) {
    validateSinglePathValue(
      `${tjsHarnessPath}::metadata.${metaKey}`,
      tjsHarness.normalization.metadataPaths,
      metaKey,
      issues,
      'HARNESS_METADATA_PATHS_NON_CANONICAL'
    );
  }

  context.compareProfileIds = profileIds;
  context.compareProfilesWithoutConversion = noConfigProfiles;
}

function kernelRegistrySnippet(kindId, fileName, status, statusReason, policy = getActivePolicy()) {
  const defaults = policy.defaults?.kernelPath || {};
  const statusDefault = resolveText(defaults.statusDefault, 'experimental');
  const statusValue = resolveText(status, statusDefault);
  const statusReasonValue = resolveText(statusReason, resolveText(defaults.statusReason, 'scaffolded'));
  return toJsonText({
    id: kindId,
    file: fileName,
    status: statusValue,
    statusReason: statusReasonValue,
    notes: `Add ${kindId} kernel path and update as needed.`,
  });
}

function renderKernelPathTemplate(id, options = {}, policy = getActivePolicy()) {
  const kernelId = normalizeId(id);
  const kernelDefaults = policy.defaults?.kernelPath || {};
  const statusValue = resolveText(options.status, resolveText(kernelDefaults.statusDefault, 'experimental'));
  const excludedStatuses = resolveTextArray(kernelDefaults.excludeStatuses, []);
  const shouldHideStatus = excludedStatuses.includes(statusValue);
  return {
    id: kernelId,
    name: `${kernelId}`,
    description: `${kernelDefaults.descriptionTemplate ? kernelDefaults.descriptionTemplate.replace('{id}', kernelId) : `Scaffolded kernel path for ${kernelId}.`}`,
    activationDtype: resolveText(kernelDefaults.activationDtype, 'f16'),
    kvDtype: resolveText(kernelDefaults.kvDtype, 'f16'),
    decode: {
      steps: [
        { op: 'input_norm', kernel: 'rmsnorm.wgsl', entry: 'main' },
        { op: 'q_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.self_attn.q_proj' },
        { op: 'k_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.self_attn.k_proj' },
        { op: 'v_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.self_attn.v_proj' },
        { op: 'rope_q', kernel: 'rope.wgsl', entry: 'main' },
        { op: 'rope_k', kernel: 'rope.wgsl', entry: 'main' },
        { op: 'attention', kernel: 'attention_decode_chunked_f16kv.wgsl', entry: 'main' },
        { op: 'o_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.self_attn.o_proj' },
        { op: 'attn_residual', kernel: 'residual.wgsl', entry: 'main' },
        { op: 'post_attn_norm', kernel: 'rmsnorm.wgsl', entry: 'main' },
        { op: 'gate_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.mlp.gate_proj' },
        { op: 'up_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.mlp.up_proj' },
        { op: 'activation', kernel: 'gelu.wgsl', entry: 'main' },
        { op: 'down_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.mlp.down_proj' },
        { op: 'ffn_residual', kernel: 'residual.wgsl', entry: 'main' },
      ],
    },
    prefill: {
      steps: [
        { op: 'input_norm', kernel: 'rmsnorm.wgsl', entry: 'main' },
        { op: 'q_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.self_attn.q_proj' },
        { op: 'k_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.self_attn.k_proj' },
        { op: 'v_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.self_attn.v_proj' },
        { op: 'rope_q', kernel: 'rope.wgsl', entry: 'main' },
        { op: 'rope_k', kernel: 'rope.wgsl', entry: 'main' },
        { op: 'attention', kernel: 'attention_streaming_f16kv.wgsl', entry: 'main' },
        { op: 'o_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.self_attn.o_proj' },
        { op: 'attn_residual', kernel: 'residual.wgsl', entry: 'main' },
        { op: 'post_attn_norm', kernel: 'rmsnorm.wgsl', entry: 'main' },
        { op: 'gate_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.mlp.gate_proj' },
        { op: 'up_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.mlp.up_proj' },
        { op: 'activation', kernel: 'gelu.wgsl', entry: 'main' },
        { op: 'down_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'layer.{L}.mlp.down_proj' },
        { op: 'ffn_residual', kernel: 'residual.wgsl', entry: 'main' },
      ],
    },
    preLayer: [
      { op: 'embed', kernel: 'gather.wgsl', entry: 'main', weights: 'embed_tokens' },
    ],
    postLayer: [
      { op: 'final_norm', kernel: 'rmsnorm.wgsl', entry: 'main' },
      { op: 'lm_head', kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_multicol', weights: 'lm_head' },
    ],
    sampling: [
      { op: 'sample', kernel: 'sample.wgsl', entry: 'sample_single_pass' },
    ],
    ...(shouldHideStatus
      ? {}
      : {
        status: statusValue,
      }),
  };
}

function renderModelFamilyTemplate(id, baseFamilyId = null, policy = getActivePolicy()) {
  const modelDefaults = policy.defaults?.model || {};
  return {
    id,
    name: id,
    extends: resolveText(baseFamilyId, resolveText(modelDefaults.baseFamily, 'transformer')),
    architecture: {},
    inference: {
      attention: {},
      normalization: {},
      ffn: {},
      output: {},
      layerPattern: {
        type: 'every_n',
        period: 1,
      },
      kernelPaths: {},
    },
    tokenizer: {
      bosToken: '<bos>',
      eosTokens: ['<eos>'],
      addBosToken: true,
    },
    detection: {
      architecturePatterns: [id],
      modelTypePatterns: [id],
    },
  };
}

function renderConversionTemplate(id, options = {}, policy = getActivePolicy()) {
  const defaults = policyDefaultsFromActivePolicy(policy);
  const conversionDefaults = isObject(defaults.conversion) ? defaults.conversion : {};
  const quantizationDefaults = isObject(conversionDefaults.quantization) ? conversionDefaults.quantization : {};
  const defaultWeights = resolveText(quantizationDefaults.weights, 'f16');
  const defaultEmbeddingsSource = resolveText(quantizationDefaults.embeddings, 'weights');
  const defaultLmHeadSource = resolveText(quantizationDefaults.lmHead, 'embeddings');
  const weights = resolveText(options.weights, defaultWeights);
  const embeddingsSource = resolveText(options.embeddings, defaultEmbeddingsSource);
  const lmHeadSource = resolveText(options.lmHead, defaultLmHeadSource);
  const embeddings = embeddingsSource === 'weights'
    ? weights
    : resolveText(embeddingsSource, weights);
  const lmHead = lmHeadSource === 'embeddings'
    ? embeddings
    : resolveText(lmHeadSource, embeddings);
  return {
    output: {
      baseDir: resolveText(options.baseDir, resolveText(conversionDefaults.baseDir, 'models/local')),
      modelBaseId: id,
    },
    quantization: {
      weights,
      embeddings,
      lmHead,
      computePrecision: 'f16',
    },
  };
}

function renderBehaviorTemplate(id, scope = null, policy = getActivePolicy()) {
  const safeId = normalizeId(id);
  const behaviorDefaults = policyDefaultsFromActivePolicy(policy).behavior || {};
  const scopeDefault = resolveText(behaviorDefaults.scope, 'profiles');
  const safeScope = normalizeId(scope, { defaultValue: scopeDefault });
  return {
    name: safeId,
    description: `Runtime profile for ${safeId}`,
    extends: `${safeScope}/default`,
    runtime: {
      inference: {
        batching: {
          batchSize: 2,
          stopCheckMode: 'batch',
          readbackInterval: 2,
          ringTokens: 2,
          ringStop: 2,
          ringStaging: 2,
        },
      },
      kvcache: {
        pageSize: 32,
      },
      shared: {
        tooling: {
          profile: safeId,
        },
      },
    },
  };
}

async function ensureDirectory(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function writeJsonFile(filePath, payload) {
  await ensureDirectory(filePath);
  await fs.writeFile(filePath, `${toJsonText(payload)}\n`, 'utf-8');
}

async function validateProgramBundleExamples(root, issues, context) {
  const bundleDir = path.join(root, 'examples/program-bundles');
  if (!(await fileExists(bundleDir))) {
    context.programBundles = { count: 0 };
    return;
  }
  const files = (await collectJsonFiles(bundleDir))
    .filter((filePath) => filePath.endsWith('.program-bundle.json'))
    .sort((left, right) => left.localeCompare(right));
  for (const filePath of files) {
    try {
      await checkProgramBundleFile(filePath);
    } catch (error) {
      issues.push(toIssue(ERROR, 'PROGRAM_BUNDLE_INVALID', filePath, error.message));
    }
  }
  context.programBundles = { count: files.length };
}

async function runCheck(context, policy = getActivePolicy()) {
  const issues = Array.isArray(context.issues) ? context.issues : [];
  await validateKernelRuleVariantParity(context.rootDir, issues, context);
  await validateGeneratedWgsl(context.rootDir, issues, context);
  await validateModelFamilies(context.rootDir, issues, context);
  await validateConversionConfigs(context.rootDir, issues, context, policy);
  await validateRuntimeProfiles(context.rootDir, issues, context);
  await validateCompareConfigs(context.rootDir, issues, context);
  await validateProgramBundleExamples(context.rootDir, issues, context);
  const {
    modelFamilyIds = new Set(),
    loaderConfigIds = new Set(),
    loaderConfigDetectionOrder = [],
    kernelPathIds = new Set(),
    kernelStatusById = new Map(),
    runtimeProfileIds = new Set(),
    conversionModelIds = new Set(),
    compareProfileIds = new Set(),
    compareProfilesWithoutConversion = new Set(),
    modelFamiliesWithoutLoaderConfig = new Set(),
    modelFamiliesMissingDetectionOrder = new Set(),
    kernelRuleVariantParity = { ruleSets: 0, variants: 0 },
    generatedWgsl = { variants: 0, drift: 0 },
    programBundles = { count: 0 },
  } = context;
  const resolvedContext = {
    ...context,
    modelFamilyIds,
    conversionModelIds: context.conversionModelIds,
    runtimeProfileIds: context.runtimeProfileIds,
    compareProfileIds: context.compareProfileIds,
    compareProfilesWithoutConversion: context.compareProfilesWithoutConversion,
    modelFamiliesWithoutLoaderConfig: context.modelFamiliesWithoutLoaderConfig,
    modelFamiliesMissingDetectionOrder: context.modelFamiliesMissingDetectionOrder,
  };

  return {
    status: issueCounts(issues).errors === 0 ? 'pass' : 'fail',
    summary: issueCounts(issues),
    issues,
    metadata: {
      root: context.rootDir,
      checks: {
        modelFamilies: modelFamilyIds.size,
        loaderConfigs: loaderConfigIds.size,
        kernelPaths: kernelPathIds.size,
        kernelRuleVariantRuleSets: kernelRuleVariantParity.ruleSets,
        kernelRuleVariantValues: kernelRuleVariantParity.variants,
        generatedWgslVariants: generatedWgsl.variants,
        generatedWgslDrifted: generatedWgsl.drift,
        loaderDetectionOrder: Array.isArray(loaderConfigDetectionOrder) ? loaderConfigDetectionOrder.length : 0,
        runtimeProfiles: resolvedContext.runtimeProfileIds ? resolvedContext.runtimeProfileIds.size : 0,
        conversionProfiles: resolvedContext.conversionModelIds ? resolvedContext.conversionModelIds.size : 0,
        compareProfiles: resolvedContext.compareProfileIds ? resolvedContext.compareProfileIds.size : 0,
        programBundles: programBundles.count,
      },
      stats: {
        ...issueCounts(issues),
      },
      coverage: {
        compareProfilesWithoutConversion: Array.from(resolvedContext.compareProfilesWithoutConversion || []),
        modelFamiliesWithoutLoaderConfig: Array.from(resolvedContext.modelFamiliesWithoutLoaderConfig || []),
        modelFamiliesMissingDetectionOrder: Array.from(resolvedContext.modelFamiliesMissingDetectionOrder || []),
      },
    },
    compatibility: {
      kernelStatusCoverage: Array.from(kernelStatusById.entries()).map(([id, status]) => ({ id, status })),
    },
  };
}

async function runScaffold(context, policy = getActivePolicy()) {
  const {
    kind,
    id,
    force,
    outputOverride,
    rootDir,
  } = context;

  const safeId = normalizeId(id);
  if (!safeId) {
    throw new Error('--id must be a non-empty string');
  }

  if (kind === 'conversion') {
    const family = normalizeId(context.family, { defaultValue: policy.defaults?.conversion?.family || 'custom' });
    const target = outputOverride
      ? path.resolve(rootDir, outputOverride)
      : path.join(
        rootDir,
        resolveText(policy.paths?.conversion, 'src/config/conversion'),
        family,
        `${safeId}.json`
      );
    if ((await fileExists(target)) && !force) {
      throw new Error(`Refusing to overwrite ${target} (use --force)`);
    }
    const payload = renderConversionTemplate(safeId, context, policy);
    await writeJsonFile(target, payload);
    console.log(`[onboarding] wrote conversion config: ${path.relative(rootDir, target)}`);
    return 0;
  }

  if (kind === 'kernel') {
    throw new Error(
      'Kernel path scaffolding is no longer supported. ' +
      'Kernel paths are now derived from execution graphs in conversion configs (src/config/conversion/). ' +
      'Use `--kind conversion` instead.'
    );
  }

  if (kind === 'behavior') {
    const scope = normalizeId(context.scope, { defaultValue: resolveText(policy.defaults?.behavior?.scope, 'profiles') });
    const target = outputOverride
      ? path.resolve(rootDir, outputOverride)
      : path.join(
        rootDir,
        resolveText(policy.paths?.runtimeProfile, 'src/config/runtime'),
        scope,
        `${safeId}.json`
      );
    if ((await fileExists(target)) && !force) {
      throw new Error(`Refusing to overwrite ${target} (use --force)`);
    }
    const payload = renderBehaviorTemplate(safeId, scope, policy);
    await writeJsonFile(target, payload);
    console.log(`[onboarding] wrote runtime profile: ${path.relative(rootDir, target)}`);
    return 0;
  }

  const supportedKinds = resolveTextArray(policy.scaffoldKinds, ['conversion', 'kernel', 'behavior']);
  const kindError = resolveText(policy.defaults?.errors?.unsupportedKind, 'Unsupported scaffold kind');
  throw new Error(`${kindError} "${kind}"`);
}

function printReport(report, jsonMode = false) {
  if (jsonMode) {
    console.log(toJsonText(report));
    return;
  }
  console.log(`status: ${report.status}`);
  for (const severity of [ERROR, WARN]) {
    const selected = issueList(severity, report.issues);
    if (!selected.length) continue;
    console.log(`\n${severity.toUpperCase()} (${selected.length})`);
    for (const item of selected) {
      const hint = item.hint ? `  hint: ${item.hint}` : '';
      console.log(`- [${item.code}] ${item.location}: ${item.message}${hint}`);
    }
  }
  const counts = issueCounts(report.issues);
  console.log(`\nerrors=${counts.errors} warnings=${counts.warnings}`);
}

async function main() {
  const args = parseCommandLine(process.argv.slice(2));
  if (args.mode === 'help' || args.mode == null) {
    console.log(usage());
    process.exit(1);
  }

  const policy = resolveOnboardingPolicy(await readJson(ONBOARDING_POLICY_PATH, 'object'));
  setActivePolicy(policy);
  const rootPolicyPath = resolveText(policy.defaults?.root?.path, 'REPO_ROOT');
  const configuredRoot = normalizeTrimmedText(args.flags.root);
  const rootDir = configuredRoot
    ? path.resolve(process.cwd(), configuredRoot)
    : (rootPolicyPath === 'REPO_ROOT'
      ? REPO_ROOT
      : path.resolve(process.cwd(), rootPolicyPath));
  const mode = String(args.mode);
  const supportedModes = resolveTextArray(policy.modes, ['check', 'scaffold']);
  if (!supportedModes.includes(mode)) {
    const unsupportedModeError = resolveText(policy.defaults?.errors?.unsupportedMode, 'Unsupported mode');
    const unsupportedHint = resolveText(policy.defaults?.errors?.unsupportedModeHint, 'Expected one of: {supportedModes}')
      .replace('{supportedModes}', supportedModes.join(', '));
    throw new Error(`${unsupportedModeError} \"${mode}\". ${unsupportedHint}`);
  }
  if (mode === 'check') {
    const {
      ids: loaderConfigIds,
      detectionOrder: loaderConfigDetectionOrder,
      error: loaderError,
    } = await detectLoaderConfigIds();
    const kernelRegistry = await validateKernelPathRegistry(rootDir, [], {});
    const context = {
      rootDir,
      loaderConfigIds,
      loaderConfigDetectionOrder,
      kernelPathIds: kernelRegistry.ids,
      kernelStatusById: kernelRegistry.statusById,
      issues: [],
    };

    if (loaderError) {
      context.issues.push(toIssue(ERROR, 'LOADER_IMPORT', 'src/config/', loaderError.message));
    }
    await validateKernelPathRegistry(rootDir, context.issues, context, policy);
    const report = await runCheck(context, policy);

    report.registries = {
      loaderConfigCount: loaderConfigIds.size,
      modelFamilyCount: report.metadata.checks.modelFamilies,
      runtimeProfileCount: report.metadata.checks.runtimeProfiles,
      conversionProfileCount: report.metadata.checks.conversionProfiles,
      kernelPathCount: report.metadata.checks.kernelPaths,
      compareProfileCount: report.metadata.checks.compareProfiles,
    };
    report.strict = Boolean(args.flags.strict);
    printReport(report, Boolean(args.flags.json));
    const strictMode = Boolean(args.flags.strict);
    const failing = issueCounts(report.issues).errors > 0 || (strictMode && issueCounts(report.issues).warnings > 0);
    report.strict = strictMode;
    process.exit(failing ? 1 : 0);
  }

  if (mode === 'scaffold') {
    const kind = normalizeTrimmedText(args.flags.kind);
    const supportedKinds = resolveTextArray(policy.scaffoldKinds, ['conversion', 'kernel', 'behavior']);
    if (!kind) {
      const missingKind = resolveText(policy.defaults?.errors?.missingKind, '--kind is required for scaffold');
      throw new Error(missingKind);
    }
    if (!supportedKinds.includes(kind)) {
      const unsupportedKindError = resolveText(policy.defaults?.errors?.unsupportedKind, 'Unsupported scaffold kind');
      const unsupportedKindHint = resolveText(policy.defaults?.errors?.unsupportedKindHint, 'Expected one of: {supportedKinds}')
        .replace('{supportedKinds}', supportedKinds.join(', '));
      throw new Error(`${unsupportedKindError} "${kind}". ${unsupportedKindHint}`);
    }
    const id = normalizeTrimmedText(args.flags.id);
    if (!id) {
      const missingId = resolveText(policy.defaults?.errors?.missingId, '--id is required for scaffold');
      throw new Error(missingId);
    }
    const scaffoldContext = {
      kind,
      id,
      rootDir,
      force: Boolean(args.flags.force),
      outputOverride: args.flags.output,
      family: args.flags.family,
      baseDir: args.flags['base-dir'],
      status: args.flags.status,
      statusReason: args.flags['status-reason'],
      scope: args.flags.scope,
    };
    await runScaffold(scaffoldContext, policy);
    return;
  }

  const unsupportedModeError = resolveText(policy.defaults?.errors?.unsupportedMode, 'Unsupported mode');
  const unsupportedModeHint = resolveText(policy.defaults?.errors?.unsupportedModeHint, 'Expected one of: {supportedModes}')
    .replace('{supportedModes}', supportedModes.join(', '));
  throw new Error(`${unsupportedModeError} "${mode}". ${unsupportedModeHint}`);
}

main().catch((error) => {
  console.error(`[onboarding] ${error.message}`);
  process.exit(1);
});
