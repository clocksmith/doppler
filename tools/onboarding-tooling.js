#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { resolveConvertedModelId } from '../src/converter/conversion-plan.js';
import { buildQuantizationInfo } from '../src/converter/quantization-info.js';
import { generateWgslVariants } from './wgsl-variant-generator.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const DEFAULT_ROOT = REPO_ROOT;
const WARN = 'warning';
const ERROR = 'error';
const REQUIRED_COMPARE_TIMING_METRICS = Object.freeze([
  'decodeTokensPerSec',
  'prefillTokensPerSec',
  'firstTokenMs',
  'firstResponseMs',
  'prefillMs',
  'decodeMs',
  'decodeMsPerTokenP50',
  'decodeMsPerTokenP95',
  'decodeMsPerTokenP99',
  'totalRunMs',
  'modelLoadMs',
]);

function usage() {
  return [
    'Usage:',
    '  node tools/onboarding-tooling.js check [--root <dir>] [--strict] [--json]',
    '  node tools/onboarding-tooling.js scaffold --kind <model|conversion|kernel|behavior> --id <id> [options]',
    '',
    'Check options:',
    '  --root <dir>               Repo root (default: current repo).',
    '  --strict                    Treat warnings as failures.',
    '  --json                      Emit JSON report.',
    '',
    'Scaffold options:',
    '  --kind <kind>               model | conversion | kernel | behavior',
    '  --id <id>                   Artifact id / file stem.',
    '  --force                     Overwrite destination file.',
    '  --output <path>             Custom output path.',
    '  --preset <id>               Preset binding (conversion/model only).',
    '  --family <name>             Conversion family folder (conversion only).',
    '  --base-dir <path>           Conversion output base dir (conversion only).',
    '  --default-kernel-path <id>   Conversion default kernel path.',
    '  --status <status>           Kernel registry status hint.',
    '  --status-reason <text>      Optional status reason.',
    '  --scope <dir>               Behavior preset scope directory.',
    '',
    'Examples:',
    '  node tools/onboarding-tooling.js check --strict',
    '  node tools/onboarding-tooling.js scaffold --kind model --id gemma3-my-new-model',
    '  node tools/onboarding-tooling.js scaffold --kind conversion --id gemma3-my-new-model --family gemma3 --preset gemma3',
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

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
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

function normalizeId(rawValue, fallback = null) {
  const value = String(rawValue ?? '').trim().toLowerCase();
  if (!value) return fallback;
  return value.replace(/[^a-z0-9._-]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

function toIssue(severity, code, location, message, hint) {
  return {
    severity,
    code,
    location,
    message,
    hint: hint || null,
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

function resolveConversionConfigModelId(config) {
  const modelBaseId = safeTrim(config?.output?.modelBaseId);
  if (!modelBaseId) {
    return {
      modelId: null,
      error: 'output.modelBaseId is required',
    };
  }
  const weights = safeTrim(config?.quantization?.weights) || 'f16';
  const embeddings = safeTrim(config?.quantization?.embeddings) || weights;
  const lmHead = safeTrim(config?.quantization?.lmHead) || embeddings;

  let quantizationInfo;
  try {
    quantizationInfo = buildQuantizationInfo(
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
    converterConfig: config,
    detectedModelId: modelBaseId,
    quantizationInfo,
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
  if (!tokens.length) return 'modelPreset';
  const head = tokens[0].replace(/^[^a-z]+/, '') || 'model';
  const tail = tokens.slice(1).map((token) => token[0].toUpperCase() + token.slice(1)).join('');
  const identifier = `${head}${tail}`;
  return /^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(identifier)
    ? identifier
    : `model_${identifier.replace(/[^a-z0-9_]/gi, '_')}`;
}

function toRuntimePresetId(runtimePresetRoot, filePath) {
  const relativePath = path.relative(runtimePresetRoot, filePath);
  return relativePath.replace(/\\/g, '/').replace(/\.json$/i, '');
}

function parseCommandLine(argv) {
  if (!argv.length) {
    return { mode: 'help' };
  }
  const mode = argv[0];
  const flags = {};
  const positional = [];

  for (let i = 1; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) {
      positional.push(token);
      continue;
    }
    const key = token.slice(2);
    if (key === 'help' || key === 'h') {
      return { mode: 'help' };
    }
    if (key === 'strict' || key === 'json' || key === 'force') {
      flags[key] = true;
      continue;
    }
    const value = argv[i + 1];
    if (value == null || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }
    flags[key] = value;
    i += 1;
  }
  return { mode, flags, positional };
}

async function detectLoaderPresetIds() {
  try {
    const loader = await import('../src/config/loader.js');
    const presetIds = Array.isArray(loader.listPresets()) ? loader.listPresets() : [];
    return {
      ids: new Set(presetIds.filter((value) => assertString(value, 'loader preset id'))),
      detectionOrder: Array.isArray(loader.PRESET_DETECTION_ORDER) ? loader.PRESET_DETECTION_ORDER : [],
      error: null,
    };
  } catch (error) {
    return {
      ids: new Set(),
      detectionOrder: [],
      error,
    };
  }
}

function collectKernelPathRefsFromPreset(preset) {
  const refs = new Set();
  const root = preset?.inference?.kernelPaths;
  if (!isObject(root)) {
    return refs;
  }

  const visit = (value) => {
    if (value == null) return;
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (trimmed) refs.add(trimmed);
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    if (!isObject(value)) return;
    for (const next of Object.values(value)) {
      visit(next);
    }
  };

  visit(root);
  return refs;
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
  if (candidates.length !== 1) {
    issues.push(toIssue(
      ERROR,
      code,
      fieldPath,
      `"${key}" must map to exactly one canonical path`
    ));
    return;
  }
  if (!assertString(candidates[0])) {
    issues.push(toIssue(ERROR, code, fieldPath, `"${key}" canonical path must be a non-empty string`));
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
      path.join(root, 'tools/configs/wgsl-variants.js'),
      error.message
    ));
    return;
  }

  for (const errorMessage of report.errors) {
    issues.push(toIssue(
      ERROR,
      'WGSL_GENERATED_INVALID',
      path.join(root, 'tools/configs/wgsl-variants.js'),
      errorMessage
    ));
  }

  for (const target of report.changedTargets) {
    issues.push(toIssue(
      ERROR,
      'WGSL_GENERATED_DRIFT',
      path.join(root, target),
      'generated WGSL file is out of date',
      'Run `npm run kernels:generate`.'
    ));
  }

  context.generatedWgsl = {
    variants: report.variantCount,
    drift: report.changedCount,
  };
}

async function validateKernelPathRegistry(root, issues, context) {
  const registryPath = path.join(root, 'src/config/presets/kernel-paths/registry.json');
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
    const id = String(entry.id ?? '').trim();
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
    statusById.set(id, String(entry.status ?? 'canonical').trim() || 'canonical');
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

    const kernelFile = path.join(root, 'src/config/presets/kernel-paths', String(entry.file).trim());
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

    if (String(kernelPathPayload.id ?? '').trim() !== id) {
      issues.push(toIssue(
        WARN,
        'KERNEL_PATH_ID_MISMATCH',
        kernelFile,
        `id="${kernelPathPayload.id ?? ''}" does not match registry id "${id}"`
      ));
    }

    const activationDtype = String(kernelPathPayload.activationDtype ?? '').trim();
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

async function validateModelPresets(root, issues, context) {
  const modelPresetDir = path.join(root, 'src/config/presets/models');
  const files = await collectJsonFiles(modelPresetDir);
  const presetsById = new Map();
  const ids = new Set();
  const notInLoaderIds = new Set();
  const detectionOrderMissingIds = new Set();
  const loaderPresetDetectionOrder = Array.isArray(context.loaderPresetDetectionOrder)
    ? new Set(context.loaderPresetDetectionOrder)
    : new Set();

  for (const filePath of files) {
    const fileBase = path.basename(filePath, '.json');
    let preset;
    try {
      preset = await readJson(filePath, 'object');
    } catch (error) {
      issues.push(toIssue(ERROR, 'MODEL_PRESET_JSON', filePath, error.message));
      continue;
    }
    const presetId = String(preset.id ?? fileBase).trim() || fileBase;
    if (!assertString(preset.id, `${filePath}.id`)) {
      issues.push(toIssue(
        WARN,
        'MODEL_PRESET_ID_MISSING',
        filePath,
        `preset file id defaults to filename "${fileBase}"`
      ));
    }
    if (presetsById.has(presetId)) {
      issues.push(toIssue(ERROR, 'MODEL_PRESET_DUP', filePath, `duplicate model preset id "${presetId}"`));
      continue;
    }
    if (!assertString(preset.name, `${filePath}.name`)) {
      issues.push(toIssue(WARN, 'MODEL_PRESET_NAME_MISSING', filePath, `model preset "${presetId}" missing name`));
    }

    if (preset.extends != null && !assertString(preset.extends, `${filePath}.extends`)) {
      issues.push(toIssue(
        ERROR,
        'MODEL_PRESET_EXTENDS_FORMAT',
        filePath,
        `preset "${presetId}" extends must be a non-empty string when provided`
      ));
    }

    if (preset.detection != null && !isObject(preset.detection)) {
      issues.push(toIssue(
        ERROR,
        'MODEL_PRESET_DETECTION_FORMAT',
        `${filePath}.detection`,
        'detection must be an object when set'
      ));
    } else if (isObject(preset.detection)) {
      validateStringList(
        `${filePath}.detection.architecturePatterns`,
        preset.detection.architecturePatterns,
        issues,
        'MODEL_PRESET_DETECTION_ARCHITECTURE_PATTERNS'
      );
      validateStringList(
        `${filePath}.detection.modelTypePatterns`,
        preset.detection.modelTypePatterns,
        issues,
        'MODEL_PRESET_DETECTION_MODEL_TYPE_PATTERNS'
      );
      if (preset.detection.configPatterns != null && !isObject(preset.detection.configPatterns)) {
        issues.push(toIssue(
          ERROR,
          'MODEL_PRESET_DETECTION_CONFIG_PATTERNS',
          `${filePath}.detection.configPatterns`,
          'detection.configPatterns must be an object when set'
        ));
      }
    }

    presetsById.set(presetId, { filePath, preset });
    ids.add(presetId);

    const kernelRefs = collectKernelPathRefsFromPreset(preset);
    for (const kernelId of kernelRefs) {
      if (!context.kernelPathIds.has(kernelId)) {
        issues.push(toIssue(
          ERROR,
          'MODEL_PRESET_KERNEL_MISSING',
          `${filePath}.inference.kernelPaths`,
          `kernel path "${kernelId}" is not registered`
        ));
      }
    }
  }

  const visiting = new Set();
  const visited = new Set();
  const stack = [];
  const validateChain = (presetId) => {
    if (visited.has(presetId)) {
      return;
    }
    if (visiting.has(presetId)) {
      const loopStart = stack.indexOf(presetId);
      const cycle = stack.slice(loopStart).concat(presetId);
      issues.push(toIssue(
        ERROR,
        'MODEL_PRESET_EXTENDS_CYCLE',
        `${modelPresetDir}/${presetId}.json`,
        `extends cycle detected: ${cycle.join(' -> ')}`
      ));
      return;
    }

    const entry = presetsById.get(presetId);
    if (!entry) {
      return;
    }

    const parent = safeTrim(entry.preset.extends);
    visiting.add(presetId);
    stack.push(presetId);

    if (parent.length > 0) {
      if (presetsById.has(parent)) {
        validateChain(parent);
      } else if (!context.loaderPresetIds.has(parent)) {
        issues.push(toIssue(
          ERROR,
          'MODEL_PRESET_EXTENDS_MISSING',
          entry.filePath,
          `extends "${parent}" is not a known local or loader preset`
        ));
      }
    } else if (!context.loaderPresetIds.has(presetId)) {
      issues.push(toIssue(
        ERROR,
        'MODEL_PRESET_NOT_IN_LOADER',
        entry.filePath,
        `model preset "${presetId}" is not exposed by src/config/loader.js`
      ));
      notInLoaderIds.add(presetId);
    } else if (
      loaderPresetDetectionOrder.size > 0
      && !loaderPresetDetectionOrder.has(presetId)
    ) {
      issues.push(toIssue(
        WARN,
        'MODEL_PRESET_DETECTION_ORDER_MISSING',
        entry.filePath,
        `model preset "${presetId}" is in loader registry but not detection order`
      ));
      detectionOrderMissingIds.add(presetId);
    }

    stack.pop();
    visiting.delete(presetId);
    visited.add(presetId);
  };

  for (const presetId of presetsById.keys()) {
    validateChain(presetId);
  }

  context.modelPresetIds = ids;
  context.modelPresetNotInLoaderIds = notInLoaderIds;
  context.modelPresetDetectionOrderMissingIds = detectionOrderMissingIds;
}

async function validateConversionConfigs(root, issues, context) {
  const conversionRoot = path.join(root, 'tools/configs/conversion');
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

    const presetId = String(config.presets?.model ?? '').trim();
    const resolved = resolveConversionConfigModelId(config);
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

    if (!assertString(config.presets?.model, `${filePath}.presets.model`)) {
      issues.push(toIssue(ERROR, 'CONVERSION_PRESET_MISSING', filePath, 'presets.model is required'));
    } else if (!context.modelPresetIds.has(presetId) && !context.loaderPresetIds.has(presetId)) {
      issues.push(toIssue(
        ERROR,
        'CONVERSION_PRESET_UNKNOWN',
        filePath,
        `presets.model "${presetId}" is not a known model preset`
      ));
    }

    const kernelId = config.inference?.defaultKernelPath;
    if (kernelId != null && !context.kernelPathIds.has(String(kernelId).trim())) {
      issues.push(toIssue(
        ERROR,
        'CONVERSION_KERNEL_MISSING',
        filePath,
        `inference.defaultKernelPath "${kernelId}" is not a registered kernel path`
      ));
    }
  }

  context.conversionModelIds = modelIds;
}

async function validateRuntimePresets(root, issues, context) {
  const runtimePresetRoot = path.join(root, 'src/config/presets/runtime');
  const files = await collectJsonFiles(runtimePresetRoot);
  const presetsById = new Map();
  const ids = new Set();

  for (const filePath of files) {
    let preset;
    try {
      preset = await readJson(filePath, 'object');
    } catch (error) {
      issues.push(toIssue(ERROR, 'RUNTIME_PRESET_JSON', filePath, error.message));
      continue;
    }

    const presetId = toRuntimePresetId(runtimePresetRoot, filePath);
    if (presetsById.has(presetId)) {
      issues.push(toIssue(ERROR, 'RUNTIME_PRESET_DUP', filePath, `duplicate runtime preset id "${presetId}"`));
      continue;
    }

    if (!assertString(preset.name, `${filePath}.name`)) {
      issues.push(toIssue(WARN, 'RUNTIME_PRESET_NAME_MISSING', filePath, `runtime preset "${presetId}" missing name`));
    }

    if (preset.extends != null && !assertString(preset.extends, `${filePath}.extends`)) {
      issues.push(toIssue(
        ERROR,
        'RUNTIME_PRESET_EXTENDS_FORMAT',
        filePath,
        `extends must be a non-empty string when provided`
      ));
    }

    if (!isObject(preset.runtime)) {
      issues.push(toIssue(
        WARN,
        'RUNTIME_PRESET_RUNTIME_MISSING',
        filePath,
        `runtime preset "${presetId}" does not define runtime`
      ));
    }

    presetsById.set(presetId, { filePath, preset });
    ids.add(presetId);
  }

  const visiting = new Set();
  const visited = new Set();
  const stack = [];
  const validateChain = (presetId) => {
    if (visited.has(presetId)) {
      return;
    }
    if (visiting.has(presetId)) {
      const loopStart = stack.indexOf(presetId);
      const cycle = stack.slice(loopStart).concat(presetId);
      issues.push(toIssue(
        ERROR,
        'RUNTIME_PRESET_EXTENDS_CYCLE',
        `${runtimePresetRoot}/${presetId}.json`,
        `extends cycle detected: ${cycle.join(' -> ')}`
      ));
      return;
    }

    const entry = presetsById.get(presetId);
    if (!entry) {
      return;
    }
    const parent = safeTrim(entry.preset.extends);

    visiting.add(presetId);
    stack.push(presetId);

    if (parent.length > 0) {
      if (!presetsById.has(parent)) {
        issues.push(toIssue(
          ERROR,
          'RUNTIME_PRESET_EXTENDS_MISSING',
          entry.filePath,
          `extends "${parent}" is not a known runtime preset`
        ));
      } else {
        validateChain(parent);
      }
    }

    stack.pop();
    visiting.delete(presetId);
    visited.add(presetId);
  };

  for (const presetId of presetsById.keys()) {
    validateChain(presetId);
  }

  context.runtimePresetIds = ids;
}

async function validateCompareConfigs(root, issues, context) {
  const compareConfigPath = path.join(root, 'benchmarks/vendors/compare-engines.config.json');
  const compareMetricsPath = path.join(root, 'benchmarks/vendors/compare-metrics.json');
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
    if (
      !(context?.conversionModelIds?.has(modelId)
        || context?.modelPresetIds?.has(modelId)
        || context?.loaderPresetIds?.has(modelId))
    ) {
      noConfigProfiles.add(modelId);
      issues.push(toIssue(
        WARN,
        'COMPARE_PROFILE_NO_MATCH',
        `${compareConfigPath}::${modelId}`,
        `compare profile "${modelId}" has no matching conversion config or model preset`
      ));
    }
    if (profile.defaultKernelPath != null
      && !context.kernelPathIds.has(String(profile.defaultKernelPath).trim())) {
      issues.push(toIssue(
        ERROR,
        'COMPARE_PROFILE_KERNEL_MISSING',
        `${compareConfigPath}::${modelId}`,
        `defaultKernelPath "${profile.defaultKernelPath}" is not registered`
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

function kernelRegistrySnippet(kindId, fileName, status = 'experimental', statusReason = null) {
  return toJsonText({
    id: kindId,
    file: fileName,
    status,
    statusReason: statusReason || 'scaffolded',
    notes: `Add ${kindId} kernel path and update as needed.`,
  });
}

function renderKernelPathTemplate(id, options = {}) {
  const kernelId = normalizeId(id);
  const status = (String(options.status ?? 'experimental')).trim() || 'experimental';
  return {
    id: kernelId,
    name: `${kernelId}`,
    description: `Scaffolded kernel path for ${kernelId}.`,
    activationDtype: 'f16',
    kvDtype: 'f16',
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
    ...((String(status).trim() === 'legacy')
      ? {}
      : {
        status,
      }),
  };
}

function renderModelPresetTemplate(id, runtimePresetId = null) {
  return {
    id,
    name: id,
    extends: runtimePresetId ?? 'transformer',
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

function renderLoaderRegistrationTemplate(presetId) {
  const constName = `${toSafeJsIdentifier(presetId)}Preset`;
  return [
    '// src/config/loader.js',
    `const ${constName} = await loadJson('./presets/models/${presetId}.json', import.meta.url, 'Failed to load preset');`,
    `// Add "${presetId}" to PRESET_REGISTRY.`,
    `// Example: ... PRESET_REGISTRY = { ... , ['${presetId}']: ${constName} };`,
    `// Add "${presetId}" to PRESET_DETECTION_ORDER when model detection should be exact.`,
  ];
}

function renderConversionTemplate(id, presetId, options = {}) {
  return {
    output: {
      baseDir: String(options.baseDir ?? 'models/local'),
      modelBaseId: id,
    },
    presets: {
      model: presetId || 'transformer',
    },
    quantization: {
      weights: 'f16',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f16',
    },
    inference: options.defaultKernelPath ? {
      defaultKernelPath: String(options.defaultKernelPath),
    } : undefined,
  };
}

function renderBehaviorTemplate(id, scope = 'modes') {
  const safeId = normalizeId(id);
  return {
    name: safeId,
    description: `Behavior preset for ${safeId}`,
    extends: 'modes/default',
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

async function runCheck(context) {
  const issues = Array.isArray(context.issues) ? context.issues : [];
  await validateKernelRuleVariantParity(context.rootDir, issues, context);
  await validateGeneratedWgsl(context.rootDir, issues, context);
  await validateModelPresets(context.rootDir, issues, context);
  await validateConversionConfigs(context.rootDir, issues, context);
  await validateRuntimePresets(context.rootDir, issues, context);
  await validateCompareConfigs(context.rootDir, issues, context);
  const {
    modelPresetIds = new Set(),
    loaderPresetIds = new Set(),
    loaderPresetDetectionOrder = [],
    kernelPathIds = new Set(),
    kernelStatusById = new Map(),
    runtimePresetIds = new Set(),
    conversionModelIds = new Set(),
    compareProfileIds = new Set(),
    compareProfilesWithoutConversion = new Set(),
    modelPresetNotInLoaderIds = new Set(),
    modelPresetDetectionOrderMissingIds = new Set(),
    kernelRuleVariantParity = { ruleSets: 0, variants: 0 },
    generatedWgsl = { variants: 0, drift: 0 },
  } = context;
  const resolvedContext = {
    ...context,
    modelPresetIds,
    conversionModelIds: context.conversionModelIds,
    runtimePresetIds: context.runtimePresetIds,
    compareProfileIds: context.compareProfileIds,
    compareProfilesWithoutConversion: context.compareProfilesWithoutConversion,
    modelPresetNotInLoaderIds: context.modelPresetNotInLoaderIds,
    modelPresetDetectionOrderMissingIds: context.modelPresetDetectionOrderMissingIds,
  };

  return {
    status: issueCounts(issues).errors === 0 ? 'pass' : 'fail',
    summary: issueCounts(issues),
    issues,
    metadata: {
      root: context.rootDir,
      checks: {
        modelPresets: modelPresetIds.size,
        loaderPresets: loaderPresetIds.size,
        kernelPaths: kernelPathIds.size,
        kernelRuleVariantRuleSets: kernelRuleVariantParity.ruleSets,
        kernelRuleVariantValues: kernelRuleVariantParity.variants,
        generatedWgslVariants: generatedWgsl.variants,
        generatedWgslDrifted: generatedWgsl.drift,
        loaderDetectionOrder: Array.isArray(loaderPresetDetectionOrder) ? loaderPresetDetectionOrder.length : 0,
        runtimePresets: resolvedContext.runtimePresetIds ? resolvedContext.runtimePresetIds.size : 0,
        conversionProfiles: resolvedContext.conversionModelIds ? resolvedContext.conversionModelIds.size : 0,
        compareProfiles: resolvedContext.compareProfileIds ? resolvedContext.compareProfileIds.size : 0,
      },
      stats: {
        ...issueCounts(issues),
      },
      coverage: {
        compareProfilesWithoutConversion: Array.from(resolvedContext.compareProfilesWithoutConversion || []),
        modelPresetsWithoutLoader: Array.from(resolvedContext.modelPresetNotInLoaderIds || []),
        modelPresetsDetectionOrderMissing: Array.from(resolvedContext.modelPresetDetectionOrderMissingIds || []),
      },
    },
    compatibility: {
      kernelStatusCoverage: Array.from(kernelStatusById.entries()).map(([id, status]) => ({ id, status })),
    },
  };
}

async function runScaffold(context) {
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

  if (kind === 'model') {
    const target = outputOverride
      ? path.resolve(rootDir, outputOverride)
      : path.join(rootDir, 'src/config/presets/models', `${safeId}.json`);
    if ((await fileExists(target)) && !force) {
      throw new Error(`Refusing to overwrite ${target} (use --force)`);
    }
    const payload = renderModelPresetTemplate(safeId, context.presetId);
    await writeJsonFile(target, payload);
    console.log(`[onboarding] wrote model preset: ${path.relative(rootDir, target)}`);
    console.log('[onboarding] add the following model preset registration steps:');
    for (const line of renderLoaderRegistrationTemplate(safeId)) {
      console.log(line);
    }
    return 0;
  }

  if (kind === 'conversion') {
    const family = normalizeId(context.family ?? 'custom');
    const target = outputOverride
      ? path.resolve(rootDir, outputOverride)
      : path.join(rootDir, 'tools/configs/conversion', family, `${safeId}.json`);
    if ((await fileExists(target)) && !force) {
      throw new Error(`Refusing to overwrite ${target} (use --force)`);
    }
    const payload = renderConversionTemplate(safeId, context.presetId || 'transformer', context);
    await writeJsonFile(target, payload);
    console.log(`[onboarding] wrote conversion config: ${path.relative(rootDir, target)}`);
    return 0;
  }

  if (kind === 'kernel') {
    const target = outputOverride
      ? path.resolve(rootDir, outputOverride)
      : path.join(rootDir, 'src/config/presets/kernel-paths', `${safeId}.json`);
    if ((await fileExists(target)) && !force) {
      throw new Error(`Refusing to overwrite ${target} (use --force)`);
    }
    const payload = renderKernelPathTemplate(safeId, context);
    await writeJsonFile(target, payload);
    console.log(`[onboarding] wrote kernel-path template: ${path.relative(rootDir, target)}`);
    console.log('[onboarding] add this entry to registry.json:');
    console.log(kernelRegistrySnippet(safeId, `${safeId}.json`, context.status || 'experimental', context.statusReason));
    return 0;
  }

  if (kind === 'behavior') {
    const scope = normalizeId(context.scope ?? 'modes');
    const target = outputOverride
      ? path.resolve(rootDir, outputOverride)
      : path.join(rootDir, 'src/config/presets/runtime', scope, `${safeId}.json`);
    if ((await fileExists(target)) && !force) {
      throw new Error(`Refusing to overwrite ${target} (use --force)`);
    }
    const payload = renderBehaviorTemplate(safeId, scope);
    await writeJsonFile(target, payload);
    console.log(`[onboarding] wrote runtime preset: ${path.relative(rootDir, target)}`);
    return 0;
  }

  throw new Error(`Unsupported scaffold kind "${kind}"`);
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

  const rootDir = path.resolve(process.cwd(), args.flags.root || DEFAULT_ROOT);
  const mode = String(args.mode);
  if (mode === 'check') {
    const {
      ids: loaderPresetIds,
      detectionOrder: loaderPresetDetectionOrder,
      error: loaderError,
    } = await detectLoaderPresetIds();
    const kernelRegistry = await validateKernelPathRegistry(rootDir, [], {});
    const context = {
      rootDir,
      loaderPresetIds,
      loaderPresetDetectionOrder,
      kernelPathIds: kernelRegistry.ids,
      kernelStatusById: kernelRegistry.statusById,
      issues: [],
    };

    if (loaderError) {
      context.issues.push(toIssue(ERROR, 'LOADER_IMPORT', 'src/config/loader.js', loaderError.message));
    }
    await validateKernelPathRegistry(rootDir, context.issues, context);
    const report = await runCheck(context);

    report.registries = {
      loaderPresetCount: loaderPresetIds.size,
      modelPresetCount: report.metadata.checks.modelPresets,
      runtimePresetCount: report.metadata.checks.runtimePresets,
      conversionProfileCount: report.metadata.checks.conversionProfiles,
      kernelPathCount: report.metadata.checks.kernelPaths,
      compareProfileCount: report.metadata.checks.compareProfiles,
    };
    report.strict = Boolean(args.flags.strict);
    printReport(report, Boolean(args.flags.json));
    const failing = issueCounts(report.issues).errors > 0 || (Boolean(args.flags.strict) && issueCounts(report.issues).warnings > 0);
    process.exit(failing ? 1 : 0);
  }

  if (mode !== 'scaffold') {
    console.error(`Unsupported mode "${mode}".`);
    process.exit(1);
  }

  const kind = String(args.flags.kind ?? '').trim();
  if (!kind) {
    throw new Error('--kind is required for scaffold');
  }
  const id = String(args.flags.id ?? '').trim();
  if (!id) {
    throw new Error('--id is required for scaffold');
  }
  const scaffoldContext = {
    kind,
    id,
    rootDir,
    force: Boolean(args.flags.force),
    outputOverride: args.flags.output,
    presetId: args.flags.preset,
    family: args.flags.family,
    baseDir: args.flags['base-dir'],
    defaultKernelPath: args.flags['default-kernel-path'],
    status: args.flags.status,
    statusReason: args.flags['status-reason'],
    scope: args.flags.scope,
  };
  await runScaffold(scaffoldContext);
}

main().catch((error) => {
  console.error(`[onboarding] ${error.message}`);
  process.exit(1);
});
