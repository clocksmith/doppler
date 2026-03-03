#!/usr/bin/env node

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import crypto from 'node:crypto';
import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mergeRuntimeValues } from '../src/config/runtime-merge.js';

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOPPLER_ROOT = path.resolve(__dirname, '..');
const WORKLOADS_PATH = path.join(DOPPLER_ROOT, 'benchmarks', 'vendors', 'workloads.json');
const WORKLOADS_SCHEMA_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'schema',
  'workloads.schema.json',
);
const COMPARE_ENGINES_CONFIG_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'compare-engines.config.json',
);
const COMPARE_METRIC_CONTRACT_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'compare-metrics.json',
);
const BENCHMARK_POLICY_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'benchmark-policy.json',
);
const COMPARE_ENGINES_CONFIG_SCHEMA_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'schema',
  'compare-engines-config.schema.json',
);
const COMPARE_METRIC_CONTRACT_SCHEMA_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'schema',
  'metric-contract.schema.json',
);
const COMPARE_HARNESS_SCHEMA_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'schema',
  'harness.schema.json',
);
const DOPPLER_HARNESS_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'harnesses',
  'doppler.json',
);
const TJS_HARNESS_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'vendors',
  'harnesses',
  'transformersjs.json',
);

const DEFAULT_PROMPT = 'word0 word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15 word16 word17 word18 word19 word20 word21 word22 word23 word24 word25 word26 word27 word28 word29 word30 word31';
const DEFAULT_MAX_TOKENS = 64;
const DEFAULT_WARMUP = 1;
const DEFAULT_RUNS = 3;
const DEFAULT_SEED = 0;
const DEFAULT_SHARED_SAMPLING = Object.freeze({
  temperature: 0,
  topK: 1,
  topP: 1,
});
const DEFAULT_BENCHMARK_POLICY = Object.freeze({
  timeoutsMs: Object.freeze({
    compare: 600_000,
    doppler: 600_000,
    transformersjs: 600_000,
  }),
  requiredTimingFields: Object.freeze([
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
  ]),
  decodeProfiles: Object.freeze({
    default: 'parity',
    presets: Object.freeze({
      parity: Object.freeze({
        batchSize: 1,
        readbackInterval: 1,
        label: 'TJS-like per-token cadence',
      }),
      throughput: Object.freeze({
        batchSize: 4,
        readbackInterval: 4,
        label: 'Doppler throughput-tuned cadence',
      }),
    }),
  }),
  browser: Object.freeze({
    stableArgs: Object.freeze([
      '--disable-breakpad',
      '--disable-gpu-sandbox',
      '--no-sandbox',
    ]),
  }),
  kernelPathPolicy: Object.freeze({
    knownBadByModel: Object.freeze({
      'gemma-3-270m-it-wf16-ef16-hf16': Object.freeze(['gemma3-f16-fused-f16a-online']),
    }),
  }),
});

function normalizeStringArray(value, label) {
  if (!Array.isArray(value) || value.length < 1) {
    throw new Error(`${label} must be a non-empty array`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== 'string' || entry.trim() === '') {
      throw new Error(`${label}[${index}] must be a non-empty string`);
    }
    return entry.trim();
  });
}

function normalizePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function normalizeDecodeProfiles(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('decodeProfiles must be an object');
  }
  const defaultId = typeof raw.default === 'string' ? raw.default.trim() : '';
  if (!defaultId) {
    throw new Error('decodeProfiles.default must be a non-empty string');
  }
  const presetsRaw = raw.presets;
  if (!presetsRaw || typeof presetsRaw !== 'object' || Array.isArray(presetsRaw)) {
    throw new Error('decodeProfiles.presets must be an object');
  }
  const presets = {};
  for (const [profileId, profileValue] of Object.entries(presetsRaw)) {
    if (typeof profileId !== 'string' || profileId.trim() === '') {
      throw new Error('decodeProfiles.presets contains an invalid profile id');
    }
    if (!profileValue || typeof profileValue !== 'object' || Array.isArray(profileValue)) {
      throw new Error(`decodeProfiles.presets.${profileId} must be an object`);
    }
    presets[profileId] = Object.freeze({
      batchSize: normalizePositiveInteger(profileValue.batchSize, `decodeProfiles.presets.${profileId}.batchSize`),
      readbackInterval: normalizePositiveInteger(
        profileValue.readbackInterval,
        `decodeProfiles.presets.${profileId}.readbackInterval`
      ),
      label: typeof profileValue.label === 'string' && profileValue.label.trim()
        ? profileValue.label.trim()
        : `${profileId} decode profile`,
    });
  }
  if (!Object.prototype.hasOwnProperty.call(presets, defaultId)) {
    throw new Error(`decodeProfiles.default "${defaultId}" is not defined in decodeProfiles.presets`);
  }
  return {
    default: defaultId,
    presets: Object.freeze(presets),
  };
}

function normalizeKnownBadKernelPaths(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return Object.freeze({});
  }
  const normalized = {};
  for (const [modelId, paths] of Object.entries(raw)) {
    if (typeof modelId !== 'string' || modelId.trim() === '') {
      throw new Error('kernelPathPolicy.knownBadByModel contains an invalid model id');
    }
    normalized[modelId] = Object.freeze(normalizeStringArray(paths, `kernelPathPolicy.knownBadByModel.${modelId}`));
  }
  return Object.freeze(normalized);
}

function loadBenchmarkPolicy() {
  let raw;
  try {
    raw = fsSync.readFileSync(BENCHMARK_POLICY_PATH, 'utf-8');
  } catch (error) {
    throw new Error(`Failed to read benchmark policy at ${BENCHMARK_POLICY_PATH}: ${error.message}`);
  }

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`Invalid benchmark policy JSON at ${BENCHMARK_POLICY_PATH}: ${error.message}`);
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`benchmark policy at ${BENCHMARK_POLICY_PATH} must be an object`);
  }
  if (!Number.isInteger(parsed.schemaVersion) || parsed.schemaVersion !== 1) {
    throw new Error(`benchmark policy at ${BENCHMARK_POLICY_PATH} schemaVersion must be 1`);
  }

  const policyTimeouts = parsed.timeoutsMs && typeof parsed.timeoutsMs === 'object'
    ? parsed.timeoutsMs
    : DEFAULT_BENCHMARK_POLICY.timeoutsMs;
  const timeoutsMs = Object.freeze({
    compare: normalizePositiveInteger(policyTimeouts.compare, 'timeoutsMs.compare'),
    doppler: normalizePositiveInteger(policyTimeouts.doppler, 'timeoutsMs.doppler'),
    transformersjs: normalizePositiveInteger(policyTimeouts.transformersjs, 'timeoutsMs.transformersjs'),
  });
  const requiredTimingFields = Object.freeze(
    normalizeStringArray(
      parsed.requiredTimingFields ?? DEFAULT_BENCHMARK_POLICY.requiredTimingFields,
      'requiredTimingFields'
    )
  );
  const decodeProfiles = normalizeDecodeProfiles(
    parsed.decodeProfiles ?? DEFAULT_BENCHMARK_POLICY.decodeProfiles
  );
  const stableArgs = Object.freeze(
    normalizeStringArray(
      parsed?.browser?.stableArgs ?? DEFAULT_BENCHMARK_POLICY.browser.stableArgs,
      'browser.stableArgs'
    )
  );
  const knownBadByModel = normalizeKnownBadKernelPaths(
    parsed?.kernelPathPolicy?.knownBadByModel
      ?? DEFAULT_BENCHMARK_POLICY.kernelPathPolicy.knownBadByModel
  );

  return Object.freeze({
    source: BENCHMARK_POLICY_PATH,
    sourceSha256: hashText(raw),
    schemaVersion: parsed.schemaVersion,
    updated: parsed.updated || null,
    timeoutsMs,
    requiredTimingFields,
    decodeProfiles,
    browser: Object.freeze({ stableArgs }),
    kernelPathPolicy: Object.freeze({ knownBadByModel }),
  });
}

const BENCHMARK_POLICY = loadBenchmarkPolicy();
const DEFAULT_DOPPLER_KERNEL_PATH = null;
const KNOWN_BAD_DOPPLER_KERNEL_PATHS_BY_MODEL = BENCHMARK_POLICY.kernelPathPolicy.knownBadByModel;
const DEFAULT_DECODE_PROFILE = BENCHMARK_POLICY.decodeProfiles.default;
const DEFAULT_TJS_PROFILE_OPS = false;
const DEFAULT_COMPARE_TIMEOUT_MS = BENCHMARK_POLICY.timeoutsMs.compare;
const DEFAULT_DOPPLER_TIMEOUT_MS = BENCHMARK_POLICY.timeoutsMs.doppler;
const DEFAULT_TJS_TIMEOUT_MS = BENCHMARK_POLICY.timeoutsMs.transformersjs;
const DEFAULT_TJS_SERVER_PORT = 0;
const DEFAULT_DOPPLER_BROWSER_PORT = 0;
const STABLE_BROWSER_ARGS = BENCHMARK_POLICY.browser.stableArgs;
const DEFAULT_COMPARE_CONFIG_SCHEMA_VERSION = 1;
const DEFAULT_COMPARE_METRIC_CONTRACT_SCHEMA_VERSION = 1;
const DEFAULT_COMPARE_PROFILE_SOURCE = 'fallback';
const DEFAULT_COMPARE_CONFIG_PATH = COMPARE_ENGINES_CONFIG_PATH;
const DEFAULT_COMPARE_METRIC_CONTRACT_PATH = COMPARE_METRIC_CONTRACT_PATH;
const schemaCache = new Map();
const REQUIRED_COMPARE_TIMING_FIELDS = BENCHMARK_POLICY.requiredTimingFields;
const DECODE_PROFILE_PRESETS = BENCHMARK_POLICY.decodeProfiles.presets;
const VALID_DECODE_PROFILES = Object.freeze([
  ...Object.keys(DECODE_PROFILE_PRESETS),
  'custom',
]);
const DEFAULT_DOPPLER_BATCH_SIZE = DECODE_PROFILE_PRESETS[DEFAULT_DECODE_PROFILE].batchSize;
const DEFAULT_DOPPLER_READBACK_INTERVAL = DECODE_PROFILE_PRESETS[DEFAULT_DECODE_PROFILE].readbackInterval;

function usage() {
  return [
    'Usage: node tools/compare-engines.js [options]',
    '',
    'Common options:',
    '  --help, -h                    Show this help text',
    '  --model-id <id>               Doppler model ID (default: first profile in compare-engines.config.json)',
    '  --model-url <url>             Doppler model URL path (default: /models/local/<model-id>)',
    '  --tjs-model <id>              Transformers.js model ID (default: profile mapping in compare-engines.config.json)',
    '  --tjs-version <3|4>           Transformers.js version (default: 4)',
    '  --tjs-dtype <fp16|q4|q4f16>   Transformers.js dtype (default: fp16)',
    '  --workload <id>               Shared workload id from benchmarks/vendors/workloads.json',
    '  --prompt <text>               Prompt used for both engines (overrides workload prompt)',
    '  --prefill-tokens <n>          Synthetic prompt token target when --prompt is omitted',
    '  --mode <compute|cold|warm|all> Run mode',
    '  --max-tokens <n>              Max new tokens',
    '  --temperature <n>             Sampling temperature',
    '  --top-k <n>                   Sampling top-k',
    '  --top-p <n>                   Sampling top-p',
    '  --warmup <n>                  Warmup runs per engine',
    '  --runs <n>                    Timed runs per engine',
    '  --decode-profile <profile>     parity|throughput|custom (default: parity)',
    '  --compare-config <path>        Compare model profile config path',
    '  --compare-metric-contract <path> Compare metric contract path',
    '  --seed <n>                    Deterministic seed',
    '  --doppler-kernel-path <id>     Doppler kernel path override',
    '  --doppler-batch-size <n>       Doppler decode batch size',
    '  --doppler-readback-interval <n> Doppler decode readback interval',
    '  --doppler-no-opfs-cache        Disable Doppler OPFS cache for browser runs',
    '  --use-chat-template <on|off>   Enable/disable chat template handling',
    '  --tjs-local-model-path <path>   Path override for local TJS model files',
    '  --load-mode <opfs|http|memory>  Asset-load mode for both engines (default depends on cache-mode)',
  '  --doppler-browser-user-data <path> Doppler Chromium profile dir',
  '  --doppler-browser-port <n>      Doppler browser relay static port (0 = random)',
  '  --browser-base-url <url>         Base URL for both benchmark runners (skips local server startup)',
  '  --browser-executable <path>      Browser executable for both benchmark runners',
  '  --runtime-config-json <json>      JSON overlay merged into Doppler runtime config',
  '  --timestamp <iso|ms>               Report timestamp override (ISO-8601 or epoch milliseconds)',
    '  --tjs-profile-ops <on|off>       TJS ORT op profiling',
  '  --timeout-ms <ms>                 Shared benchmark timeout',
  '  --doppler-timeout-ms <ms>         Doppler-only benchmark timeout',
  '  --tjs-timeout-ms <ms>             TJS-only benchmark timeout',
    '  --tjs-server-port <n>             TJS server port (0 = random)',
    '  --tjs-browser-console             Stream browser console on TJS failures/retries',
    '  --save                           Save results to benchmarks/vendors/results/',
    '  --save-dir <dir>                 Directory for saved results (default: ./benchmarks/vendors/results)',
    '  --skip-matrix-update             Skip automatic vendor release-matrix refresh when --save is enabled',
    '  --json                           JSON-only output',
    '',
    'See these usage lines for the full authoritative option list.',
  ].join('\n');
}

function assertComparableString(value, fieldLabel) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${fieldLabel} must be a non-empty string`);
  }
}

function assertComparableObject(value, fieldLabel) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${fieldLabel} must be an object`);
  }
}

function isJsonObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function valueType(value) {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function ensureSchemaMatch(value, schema, trace) {
  const label = trace == null ? '$' : String(trace);
  if (!schema || !isJsonObject(schema)) {
    throw new Error(`Invalid schema at ${label}: expected object`);
  }

  if (schema.const !== undefined && value !== schema.const) {
    throw new Error(`${label}: value must equal ${JSON.stringify(schema.const)}`);
  }
  if (schema.enum && !schema.enum.includes(value)) {
    throw new Error(`${label}: value must be one of ${schema.enum.map((item) => JSON.stringify(item)).join(', ')}`);
  }

  if (schema.type) {
    const expectedTypes = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (expectedTypes.includes('integer')) {
      if (typeof value !== 'number' || !Number.isInteger(value)) {
        throw new Error(`${label}: expected integer, got ${valueType(value)}`);
      }
    }
    if (!expectedTypes.includes(valueType(value))) {
      if (!(expectedTypes.includes('integer') && typeof value === 'number')) {
        throw new Error(`${label}: expected ${expectedTypes.join(' | ')}, got ${valueType(value)}`);
      }
    }
  }

  if (typeof schema.minLength === 'number' && value !== null && value.length < schema.minLength) {
    throw new Error(`${label}: minimum length is ${schema.minLength}`);
  }
  if (typeof schema.minItems === 'number' && Array.isArray(value) && value.length < schema.minItems) {
    throw new Error(`${label}: minimum items is ${schema.minItems}`);
  }
  if (typeof schema.minProperties === 'number' && isJsonObject(value) && Object.keys(value).length < schema.minProperties) {
    throw new Error(`${label}: minimum properties is ${schema.minProperties}`);
  }

  if (isJsonObject(value) && schema.required) {
    for (const requiredKey of schema.required) {
      if (!Object.prototype.hasOwnProperty.call(value, requiredKey)) {
        throw new Error(`${label}: required property "${requiredKey}" is missing`);
      }
    }
  }

  if (schema.pattern && typeof value === 'string') {
    const expression = new RegExp(schema.pattern);
    if (!expression.test(value)) {
      throw new Error(`${label}: value does not match pattern ${schema.pattern}`);
    }
  }

  if (Array.isArray(value) && schema.items) {
    for (let i = 0; i < value.length; i += 1) {
      ensureSchemaMatch(value[i], schema.items, `${label}[${i}]`);
    }
  }

  if (isJsonObject(value)) {
    if (schema.properties) {
      for (const [key, nestedSchema] of Object.entries(schema.properties)) {
        if (Object.prototype.hasOwnProperty.call(value, key)) {
          ensureSchemaMatch(value[key], nestedSchema, `${label}.${key}`);
        }
      }
    }
    if (schema.additionalProperties === false) {
      const propertyKeys = new Set(schema.properties ? Object.keys(schema.properties) : []);
      for (const key of Object.keys(value)) {
        if (!propertyKeys.has(key)) {
          throw new Error(`${label}: unexpected property "${key}"`);
        }
      }
    }
  }
}

async function loadSchema(schemaPath) {
  if (!schemaCache.has(schemaPath)) {
    const raw = await fs.readFile(schemaPath, 'utf-8');
    schemaCache.set(schemaPath, JSON.parse(raw));
  }
  return schemaCache.get(schemaPath);
}

async function assertMatchesSchema(payload, schemaPath, label) {
  const schema = await loadSchema(schemaPath);
  ensureSchemaMatch(payload, schema, label || 'document');
}

function resolveCompareMetricContractPath(rawPath) {
  const resolvedPath = rawPath == null || String(rawPath).trim() === '' ? DEFAULT_COMPARE_METRIC_CONTRACT_PATH : String(rawPath).trim();
  if (path.isAbsolute(resolvedPath)) return resolvedPath;
  return path.join(process.cwd(), resolvedPath);
}

async function loadCompareMetricContract(rawPath) {
  const resolvedPath = resolveCompareMetricContractPath(rawPath);
  const raw = await fs.readFile(resolvedPath, 'utf-8');
  const sourceSha256 = hashText(raw);
  const payload = JSON.parse(raw);
  await assertMatchesSchema(payload, COMPARE_METRIC_CONTRACT_SCHEMA_PATH, `compare metric contract ${resolvedPath}`);

  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`compare metric contract at ${resolvedPath} must be a JSON object`);
  }
  if (!Number.isInteger(payload.schemaVersion) || payload.schemaVersion !== DEFAULT_COMPARE_METRIC_CONTRACT_SCHEMA_VERSION) {
    throw new Error(
      `compare metric contract at ${resolvedPath} schemaVersion must be ${DEFAULT_COMPARE_METRIC_CONTRACT_SCHEMA_VERSION}, got ${payload.schemaVersion}`
    );
  }

  const rows = Array.isArray(payload.metrics) ? payload.metrics : [];
  if (rows.length === 0) {
    throw new Error(`compare metric contract at ${resolvedPath} must include at least one metric`);
  }

  const seen = new Set();
  for (const metric of rows) {
    if (!metric || typeof metric !== 'object' || Array.isArray(metric)) {
      throw new Error(`compare-metrics.json metric entry at ${resolvedPath} must be an object`);
    }
    assertComparableString(metric.id, `metric id in ${resolvedPath}`);
    if (seen.has(metric.id)) {
      throw new Error(`compare metric contract contains duplicate id: ${metric.id}`);
    }
    seen.add(metric.id);
    assertComparableString(metric.label, `metric label for ${metric.id}`);
    assertComparableString(metric.unit, `metric unit for ${metric.id}`);
    if (typeof metric.higherBetter !== 'boolean') {
      throw new Error(`metric.higherBetter for ${metric.id} must be boolean`);
    }
    if (metric.derived != null) {
      if (!metric.derived || typeof metric.derived !== 'object' || Array.isArray(metric.derived)) {
        throw new Error(`metric.derived for ${metric.id} must be an object`);
      }
      const { doppler, transformersjs } = metric.derived;
      if (doppler != null
        && (!Array.isArray(doppler.numeratorPaths) || !Array.isArray(doppler.denominatorPaths))) {
        throw new Error(`metric.derived.doppler for ${metric.id} must include numeratorPaths and denominatorPaths`);
      }
      if (transformersjs != null
        && (!Array.isArray(transformersjs.numeratorPaths) || !Array.isArray(transformersjs.denominatorPaths))) {
        throw new Error(`metric.derived.transformersjs for ${metric.id} must include numeratorPaths and denominatorPaths`);
      }
    }
  }

  return {
    source: resolvedPath,
    schemaVersion: payload.schemaVersion,
    updated: payload.updated || null,
    sourceSha256,
    metrics: rows,
  };
}

async function loadCompareHarnessMetricPaths(harnessPath, expectedId) {
  const raw = await fs.readFile(harnessPath, 'utf-8');
  const sourceSha256 = hashText(raw);
  const payload = JSON.parse(raw);
  await assertMatchesSchema(payload, COMPARE_HARNESS_SCHEMA_PATH, `harness ${harnessPath}`);
  assertComparableObject(payload, `harness at ${harnessPath}`);
  if (payload.id !== expectedId) {
    throw new Error(`harness id mismatch at ${harnessPath}: expected ${expectedId}, got ${payload.id}`);
  }
  const paths = payload?.normalization?.metricPaths;
  const metadataPaths = payload?.normalization?.metadataPaths;
  if (!paths || typeof paths !== 'object' || Array.isArray(paths)) {
    throw new Error(`harness at ${harnessPath} must include normalization.metricPaths object`);
  }
  if (metadataPaths != null && (typeof metadataPaths !== 'object' || Array.isArray(metadataPaths))) {
    throw new Error(`harness at ${harnessPath} has invalid normalization.metadataPaths`);
  }
  for (const [metricId, candidates] of Object.entries(paths)) {
    if (typeof metricId !== 'string' || metricId.trim() === '') {
      throw new Error(`harness at ${harnessPath} has an invalid metric id in normalization.metricPaths`);
    }
    if (!Array.isArray(candidates) || candidates.length < 1) {
      throw new Error(`harness at ${harnessPath} has an invalid metric path list for "${metricId}"; expected at least one path`);
    }
    for (const candidate of candidates) {
      if (typeof candidate !== 'string' || candidate.trim() === '') {
        throw new Error(`harness at ${harnessPath} has an invalid path for "${metricId}" in normalization.metricPaths`);
      }
    }
  }
  if (metadataPaths) {
    for (const [metadataId, candidates] of Object.entries(metadataPaths)) {
      if (typeof metadataId !== 'string' || metadataId.trim() === '') {
        throw new Error(`harness at ${harnessPath} has an invalid metadata id in normalization.metadataPaths`);
      }
      if (!Array.isArray(candidates) || candidates.length < 1) {
        throw new Error(`harness at ${harnessPath} has invalid metadata path list for "${metadataId}"; expected at least one path`);
      }
      for (const candidate of candidates) {
        if (typeof candidate !== 'string' || candidate.trim() === '') {
          throw new Error(`harness at ${harnessPath} has an invalid path for "${metadataId}" in normalization.metadataPaths`);
        }
      }
    }
  }

  const requiredMetrics = payload?.normalization?.requiredMetrics;
  if (requiredMetrics == null) {
    throw new Error(`harness at ${harnessPath} must include normalization.requiredMetrics`);
  }
  if (!Array.isArray(requiredMetrics) || requiredMetrics.length < 1) {
    throw new Error(`normalization.requiredMetrics at ${harnessPath} must be a non-empty array`);
  }

  for (const metricId of requiredMetrics) {
    if (typeof metricId !== 'string' || metricId.trim() === '') {
      throw new Error(`harness at ${harnessPath} has invalid required metric id in normalization.requiredMetrics`);
    }
    if (!Array.isArray(paths[metricId]) || paths[metricId].length < 1) {
      throw new Error(`harness at ${harnessPath} is missing metricPaths for required metric "${metricId}"`);
    }
  }

  return {
    source: harnessPath,
    sourceSha256,
    paths,
    requiredMetrics,
  };
}

function buildCompareMetricContract(contractRows, harnessMetricPathsByEngine) {
  const { doppler: dopplerPaths, transformersjs: tjsPaths } = harnessMetricPathsByEngine;
  return contractRows.map((entry) => {
    const dopplerPathCandidates = Array.isArray(dopplerPaths?.[entry.id]) ? dopplerPaths[entry.id] : [];
    const tjsPathCandidates = Array.isArray(tjsPaths?.[entry.id]) ? tjsPaths[entry.id] : [];

    if (!entry.derived?.doppler && dopplerPathCandidates.length === 0) {
      throw new Error(`compare metric contract metric "${entry.id}" missing harness path mapping for doppler`);
    }
    if (!entry.derived?.transformersjs && tjsPathCandidates.length === 0) {
      throw new Error(`compare metric contract metric "${entry.id}" missing harness path mapping for transformersjs`);
    }

    return {
      id: entry.id,
      label: entry.label,
      unit: entry.unit,
      higherBetter: entry.higherBetter,
      required: Boolean(entry.required),
      paths: {
        doppler: dopplerPathCandidates,
        transformersjs: tjsPathCandidates,
      },
      derived: entry.derived || null,
    };
  });
}

function parsePositiveInt(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInt(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return parsed;
}

function parseOnOff(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'on' || normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
  if (normalized === 'off' || normalized === 'false' || normalized === '0' || normalized === 'no') return false;
  throw new Error(`${label} must be one of: on, off, true, false, 1, 0`);
}

function parseNonNegativeNumber(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative number`);
  }
  return parsed;
}

function parseLoadMode(value, label, fallback) {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'opfs' || normalized === 'http' || normalized === 'memory') {
    return normalized;
  }
  throw new Error(`${label} must be one of: opfs, http, memory`);
}

function parseTopP(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
    throw new Error(`${label} must be in the range (0, 1]`);
  }
  return parsed;
}

function parseChoice(value, allowed, label, fallback) {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (allowed.includes(normalized)) return normalized;
  throw new Error(`${label} must be one of: ${allowed.join(', ')}`);
}

function assertCompareMetricContractCompleteness(metricRows) {
  const rowsById = new Map();
  for (const row of Array.isArray(metricRows) ? metricRows : []) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) continue;
    const metricId = String(row.id || '').trim();
    if (metricId) {
      rowsById.set(metricId, row);
    }
  }

  for (const field of REQUIRED_COMPARE_TIMING_FIELDS) {
    const row = rowsById.get(field);
    if (!row) {
      throw new Error(`compare metric contract is missing required timing metric "${field}"`);
    }
    if (row.required !== true) {
      throw new Error(`compare metric contract must mark "${field}" as required for apples-to-apples mode`);
    }
  }
}

function resolveCompareEnginesConfigPath(rawPath) {
  const resolvedPath = rawPath == null || String(rawPath).trim() === ''
    ? DEFAULT_COMPARE_CONFIG_PATH
    : String(rawPath).trim();
  if (path.isAbsolute(resolvedPath)) return resolvedPath;
  return path.join(process.cwd(), resolvedPath);
}

function buildSyntheticPrompt(prefillTokens) {
  const words = [];
  for (let i = 0; i < prefillTokens; i++) {
    words.push(`word${i}`);
  }
  return words.join(' ');
}

async function loadWorkloadCatalog() {
  const raw = await fs.readFile(WORKLOADS_PATH, 'utf-8');
  const payload = JSON.parse(raw);
  await assertMatchesSchema(payload, WORKLOADS_SCHEMA_PATH, `workload catalog ${WORKLOADS_PATH}`);
  const rows = Array.isArray(payload?.workloads) ? payload.workloads : [];
  const configuredDefault = payload?.defaults?.compareEngines;
  return {
    rows,
    defaultWorkloadId: typeof configuredDefault === 'string' && configuredDefault.trim() !== ''
      ? configuredDefault
      : null,
  };
}

function resolveWorkloadById(workloadCatalog, workloadId) {
  const rows = Array.isArray(workloadCatalog?.rows) ? workloadCatalog.rows : [];
  const selected = rows.find((row) => row.id === workloadId) || null;
  if (!selected) {
    throw new Error(`Unknown workload "${workloadId}". Available: ${rows.map((row) => row.id).join(', ')}`);
  }
  return selected;
}

function resolveDefaultDopplerModelId(compareConfig) {
  const firstProfile = Array.isArray(compareConfig?.profiles) ? compareConfig.profiles[0] : null;
  const modelId = typeof firstProfile?.dopplerModelId === 'string'
    ? firstProfile.dopplerModelId.trim()
    : '';
  if (!modelId) {
    throw new Error(
      'compare-engines.config.json must include at least one modelProfiles entry with dopplerModelId'
    );
  }
  return modelId;
}

async function loadCompareEnginesConfig(rawPath) {
  const resolvedPath = resolveCompareEnginesConfigPath(rawPath);
  const raw = await fs.readFile(resolvedPath, 'utf-8');
  const sourceSha256 = hashText(raw);
  const payload = JSON.parse(raw);
  await assertMatchesSchema(payload, COMPARE_ENGINES_CONFIG_SCHEMA_PATH, `compare-engine config ${resolvedPath}`);

  const rows = Array.isArray(payload?.modelProfiles) ? payload.modelProfiles : [];

  if (payload == null || typeof payload !== 'object') {
    throw new Error(`compare profile config at ${resolvedPath} must be a JSON object`);
  }
  if (!Number.isInteger(payload.schemaVersion) || payload.schemaVersion !== DEFAULT_COMPARE_CONFIG_SCHEMA_VERSION) {
    throw new Error(
      `compare profile config at ${resolvedPath} schemaVersion must be ${DEFAULT_COMPARE_CONFIG_SCHEMA_VERSION}, got ${payload.schemaVersion}`
    );
  }

  const seen = new Set();
  for (const row of rows) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) {
      throw new Error('compare-engines.config.json model profile entry must be an object');
    }
    if (typeof row.dopplerModelId !== 'string' || row.dopplerModelId.trim() === '') {
      throw new Error('compare-engines.config.json model profile entry is missing dopplerModelId');
    }
    const key = row.dopplerModelId.trim().toLowerCase();
    if (seen.has(key)) {
      throw new Error(`compare-engines.config.json duplicate dopplerModelId: ${row.dopplerModelId}`);
    }
    seen.add(key);
    if (row.defaultTjsModelId != null && (typeof row.defaultTjsModelId !== 'string' || row.defaultTjsModelId.trim() === '')) {
      throw new Error(`compare-engines.config.json defaultTjsModelId for ${row.dopplerModelId} must be a non-empty string or null`);
    }
    if (row.defaultKernelPath != null && (typeof row.defaultKernelPath !== 'string' || row.defaultKernelPath.trim() === '')) {
      throw new Error(`compare-engines.config.json defaultKernelPath for ${row.dopplerModelId} must be a non-empty string or null`);
    }
    if (row.modelBaseDir != null && (typeof row.modelBaseDir !== 'string' || row.modelBaseDir.trim() === '')) {
      throw new Error(`compare-engines.config.json modelBaseDir for ${row.dopplerModelId} must be a non-empty string or null`);
    }
  }

  const modelProfileById = new Map(rows.map((row) => [String(row.dopplerModelId).trim().toLowerCase(), row]));
  return {
    source: resolvedPath,
    schemaVersion: payload.schemaVersion,
    updated: payload.updated || null,
    sourceSha256,
    modelProfileById,
    profiles: rows,
  };
}

function resolveCompareProfile(compareConfig, modelId) {
  const normalizedModelId = String(modelId ?? '').trim().toLowerCase();
  const profile = compareConfig?.modelProfileById?.get(normalizedModelId) || null;
  return {
    source: profile == null ? DEFAULT_COMPARE_PROFILE_SOURCE : 'config',
    modelBaseDir: profile?.modelBaseDir || 'local',
    defaultTjsModelId: profile?.defaultTjsModelId || null,
    defaultKernelPath: profile?.defaultKernelPath || null,
  };
}

function getByPath(objectValue, dottedPath) {
  const segments = String(dottedPath || '').split('.').filter(Boolean);
  let current = objectValue;
  for (const segment of segments) {
    if (current == null) return undefined;
    if (/^\d+$/.test(segment)) {
      const index = Number(segment);
      if (!Array.isArray(current)) return undefined;
      current = current[index];
      continue;
    }
    if (typeof current !== 'object') return undefined;
    current = current[segment];
  }
  return current;
}

function firstFiniteNumber(objectValue, paths) {
  for (const dottedPath of paths || []) {
    const candidate = getByPath(objectValue, dottedPath);
    if (Number.isFinite(candidate)) {
      return candidate;
    }
  }
  return null;
}

function clipTail(text, maxChars = 3000) {
  if (text == null) return '';
  const str = String(text);
  if (str.length <= maxChars) return str;
  return str.slice(str.length - maxChars);
}

function parseJsonBlock(stdout, label) {
  const normalized = String(stdout == null ? '' : stdout);
  if (!normalized.trim()) {
    throw new Error(`No output to parse for ${label}`);
  }

  const asObject = (candidate) => {
    try {
      const parsed = JSON.parse(candidate);
      return parsed === null || typeof parsed !== 'object' ? null : parsed;
    } catch {
      return null;
    }
  };

  const direct = asObject(normalized.trim());
  if (direct !== null) return direct;

  throw new Error(`Could not parse strict JSON payload from ${label}. Full output tail:\n${clipTail(normalized, 2000)}`);
}

function hashText(input) {
  return crypto.createHash('sha256').update(input).digest('hex');
}

const DTYPE_PATTERN = /^(f\d+a?|q\d+[a-z]*|bf16|fp16|fp32|int[48])$/i;

function parseDopplerDtype(modelId) {
  if (!modelId) return null;
  const parts = modelId.split('-');
  const dtypes = [];
  for (let i = parts.length - 1; i >= 0; i--) {
    if (DTYPE_PATTERN.test(parts[i])) {
      dtypes.unshift(parts[i].toUpperCase());
    } else {
      break;
    }
  }
  return dtypes.length > 0 ? dtypes.join('/') : null;
}

function parseRuntimeConfigJson(raw) {
  if (raw == null) return null;
  const parsed = (() => {
    try {
      return JSON.parse(raw);
    } catch (error) {
      throw new Error(`--runtime-config-json must be valid JSON: ${error.message}`);
    }
  })();
  if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('--runtime-config-json must be a JSON object');
  }
  return parsed;
}

function toFailurePayload(library, error) {
  return {
    failed: true,
    env: { library },
    error: {
      message: String(error?.message || error),
      code: error?.code ?? null,
      signal: error?.signal ?? null,
      killed: error?.killed === true,
      stderrTail: clipTail(error?.stderr, 3000),
    },
  };
}

function hashJsonPayload(payload) {
  const text = JSON.stringify(payload, null, 0);
  return crypto.createHash('sha256').update(text).digest('hex');
}

function parseDecodeProfile(value) {
  if (value == null || value === '') return DEFAULT_DECODE_PROFILE;
  const profile = String(value);
  if (!VALID_DECODE_PROFILES.includes(profile)) {
    throw new Error(
      `--decode-profile must be one of: ${VALID_DECODE_PROFILES.join(', ')}`
    );
  }
  return profile;
}

function parseTimestampValue(rawValue, label) {
  if (rawValue == null || rawValue === '') return null;
  if (typeof rawValue !== 'string') {
    throw new Error(`${label} must be a string`);
  }
  const trimmed = rawValue.trim();
  if (trimmed === '') return null;
  const asMs = /^[-+]?\d+$/.test(trimmed) ? Number(trimmed) : NaN;
  const parsed = Number.isFinite(asMs) ? new Date(asMs) : new Date(trimmed);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error(`${label} must be ISO-8601 or epoch milliseconds`);
  }
  return parsed.toISOString();
}

function resolveDopplerKernelPath(profile, kernelPathOverride) {
  if (kernelPathOverride != null && kernelPathOverride !== '') {
    return {
      kernelPath: String(kernelPathOverride),
      source: 'cli',
    };
  }
  const configuredKernelPath = profile?.defaultKernelPath ?? null;
  if (configuredKernelPath) {
    return {
      kernelPath: configuredKernelPath,
      source: 'compare-config',
    };
  }
  return {
    kernelPath: DEFAULT_DOPPLER_KERNEL_PATH,
    source: 'manifest-default',
  };
}

function assertKernelPathAllowedForModel(modelId, kernelPath) {
  if (kernelPath == null || kernelPath === '') return;
  const normalizedModelId = String(modelId ?? '').trim().toLowerCase();
  const blockedKernelPaths = KNOWN_BAD_DOPPLER_KERNEL_PATHS_BY_MODEL[normalizedModelId] || [];
  if (!blockedKernelPaths.includes(String(kernelPath))) return;
  throw new Error(
    `Kernel path "${kernelPath}" is blocked for model "${modelId}" in compare runs due to deterministic correctness failures. `
    + 'Use manifest-default (omit --doppler-kernel-path) or choose a validated kernel path.'
  );
}

function parseArgs(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token === '-h') {
      flags.h = true;
      continue;
    }
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (
      key === 'json'
      || key === 'save'
      || key === 'doppler-no-opfs-cache'
      || key === 'tjs-browser-console'
      || key === 'use-chat-template'
      || key === 'skip-matrix-update'
      || key === 'help'
      || key === 'h'
    ) {
      flags[key] = true;
      continue;
    }
    const value = argv[i + 1];
    if (value === undefined) {
      throw new Error(`Missing value for --${key}`);
    }
    flags[key] = value;
    i++;
  }
  return flags;
}

function appendBrowserArgs(baseArgs = [], extraArgs = []) {
  const values = [...baseArgs];
  for (const browserArg of extraArgs) {
    if (browserArg == null) continue;
    const value = String(browserArg).trim();
    if (value) values.push(value);
  }
  return [...new Set(values)];
}

function buildSharedBenchmarkContract({
  prompt,
  maxTokens,
  warmupRuns,
  timedRuns,
  seed,
  sampling = DEFAULT_SHARED_SAMPLING,
  useChatTemplate = false,
  loadMode = null,
}) {
  return {
    prompt: String(prompt ?? DEFAULT_PROMPT),
    maxTokens: parsePositiveInt(maxTokens, DEFAULT_MAX_TOKENS, '--max-tokens'),
    warmupRuns: parseNonNegativeInt(warmupRuns, DEFAULT_WARMUP, '--warmup'),
    timedRuns: parsePositiveInt(timedRuns, DEFAULT_RUNS, '--runs'),
    seed: parseNonNegativeInt(seed, DEFAULT_SEED, '--seed'),
    sampling: {
      temperature: Number(sampling.temperature),
      topK: Number(sampling.topK),
      topP: Number(sampling.topP),
    },
    useChatTemplate: useChatTemplate === true,
    loadMode,
  };
}

function resolveRunLoadMode(cacheMode, sharedLoadMode) {
  if (sharedLoadMode != null) return sharedLoadMode;
  return cacheMode === 'cold' ? 'http' : 'opfs';
}

function buildDopplerRuntimeConfig(sharedContract, engineOverlay) {
  const sampling = {
    temperature: sharedContract.sampling.temperature,
    topK: sharedContract.sampling.topK,
    topP: sharedContract.sampling.topP,
  };
  const runtimeConfig = {
    shared: {
      benchmark: {
        run: {
          customPrompt: sharedContract.prompt,
          maxNewTokens: sharedContract.maxTokens,
          warmupRuns: sharedContract.warmupRuns,
          timedRuns: sharedContract.timedRuns,
          useChatTemplate: sharedContract.useChatTemplate,
          loadMode: sharedContract.loadMode,
          sampling,
        },
      },
    },
    inference: {
      prompt: sharedContract.prompt,
      chatTemplate: {
        enabled: sharedContract.useChatTemplate,
      },
      batching: {
        maxTokens: sharedContract.maxTokens,
        batchSize: engineOverlay.batchSize,
        readbackInterval: engineOverlay.readbackInterval,
        stopCheckMode: 'per-token',
      },
      sampling,
    },
  };
  if (engineOverlay.kernelPath) {
    runtimeConfig.inference.kernelPath = engineOverlay.kernelPath;
  }

  if (engineOverlay.runtimeConfigJson) {
    if (typeof engineOverlay.runtimeConfigJson !== 'object' || engineOverlay.runtimeConfigJson == null) {
      throw new Error('--runtime-config-json override is expected to be a parsed JSON object');
    }
    return mergeRuntimeValues(runtimeConfig, engineOverlay.runtimeConfigJson);
  }

  return runtimeConfig;
}

async function runDoppler(modelId, modelUrl, sharedContract, cacheMode, options = {}) {
  const resolvedKernelPath = options.kernelPath ?? DEFAULT_DOPPLER_KERNEL_PATH;
  const resolvedLoadMode = options.loadMode ?? null;
  const resolvedTimeoutMs = parsePositiveInt(options.timeoutMs, DEFAULT_DOPPLER_TIMEOUT_MS, '--doppler-timeout-ms');
  const resolvedBatchSize = parsePositiveInt(options.batchSize, DEFAULT_DOPPLER_BATCH_SIZE, '--doppler-batch-size');
  const resolvedReadbackInterval = parsePositiveInt(
    options.readbackInterval,
    DEFAULT_DOPPLER_READBACK_INTERVAL,
    '--doppler-readback-interval'
  );
  const resolvedBrowserPort = parseNonNegativeInt(
    options.browserPort,
    DEFAULT_DOPPLER_BROWSER_PORT,
    '--doppler-browser-port'
  );
  const resolvedBrowserBaseUrl = options.browserBaseUrl || null;
  const resolvedBrowserExecutable = options.browserExecutable || null;
  const runtimeConfig = buildDopplerRuntimeConfig(sharedContract, {
    kernelPath: resolvedKernelPath,
    batchSize: resolvedBatchSize,
    readbackInterval: resolvedReadbackInterval,
    runtimeConfigJson: options.runtimeConfigJson,
  });
  const stableBrowserArgs = appendBrowserArgs([], STABLE_BROWSER_ARGS);

  const buildCliConfig = (forceNoOpfs = false) => ({
    request: {
      modelId,
      modelUrl,
      cacheMode,
      ...(resolvedLoadMode != null ? { loadMode: String(resolvedLoadMode) } : {}),
    },
    run: {
      surface: 'browser',
      browser: {
        ...(resolvedBrowserBaseUrl
          ? { baseUrl: String(resolvedBrowserBaseUrl) }
          : (options.browserPort != null ? { port: resolvedBrowserPort } : {})),
        ...(resolvedBrowserExecutable ? { executablePath: String(resolvedBrowserExecutable) } : {}),
        ...(options.browserUserData ? { userDataDir: String(options.browserUserData) } : {}),
        ...(options.noOpfsCache || forceNoOpfs ? { opfsCache: false } : {}),
        browserArgs: stableBrowserArgs,
      },
    },
  });

  console.error(`[compare] running Doppler (${cacheMode})...`);
  const runOnce = async ({ forceNoOpfs = false } = {}) => {
    const cliConfig = buildCliConfig(forceNoOpfs);
    const args = [
      path.join(DOPPLER_ROOT, 'tools', 'doppler-cli.js'),
      'bench',
      '--config',
      JSON.stringify(cliConfig),
      '--runtime-config',
      JSON.stringify(runtimeConfig),
      '--json',
    ];
    const { stdout } = await execFileAsync('node', args, {
      cwd: DOPPLER_ROOT,
      timeout: resolvedTimeoutMs,
      maxBuffer: 10 * 1024 * 1024,
    });
    return parseJsonBlock(stdout, `Doppler (${cacheMode})`);
  };

  try {
    return await runOnce();
  } catch (error) {
    const message = String(error?.message || '');
    const shouldRetryNoOpfs = !options.noOpfsCache && message.includes('Invalid manifest');
    if (shouldRetryNoOpfs) {
      console.error('[compare] Doppler failed with cached manifest mismatch; retrying with --no-opfs-cache...');
      try {
        return await runOnce({ forceNoOpfs: true });
      } catch (retryError) {
        console.error(`[compare] Doppler (${cacheMode}) retry failed: ${retryError.message}`);
        return toFailurePayload('doppler', retryError);
      }
    }
    console.error(`[compare] Doppler (${cacheMode}) failed: ${error.message}`);
    return toFailurePayload('doppler', error);
  }
}

async function runTjs(modelId, sharedContract, cacheMode, tjsVersion, localModelPath, options = {}) {
  const resolvedProfileOps = options.profileOps ?? DEFAULT_TJS_PROFILE_OPS;
  const resolvedLoadMode = options.loadMode ?? null;
  const resolvedTimeoutMs = parsePositiveInt(options.timeoutMs, DEFAULT_TJS_TIMEOUT_MS, '--tjs-timeout-ms');
  const resolvedServerPort = parseNonNegativeInt(options.serverPort, DEFAULT_TJS_SERVER_PORT, '--tjs-server-port');
  const resolvedTjsDtype = parseChoice(options.tjsDtype, ['fp16', 'q4', 'q4f16'], '--tjs-dtype', 'fp16');
  const resolvedBrowserBaseUrl = options.browserBaseUrl || null;
  const resolvedBrowserExecutable = options.browserExecutable || null;
  const args = [
    path.join(DOPPLER_ROOT, 'benchmarks', 'runners', 'transformersjs-bench.js'),
    '--model', modelId,
    '--prompt', String(sharedContract.prompt),
    '--max-tokens', String(sharedContract.maxTokens),
    '--warmup', String(sharedContract.warmupRuns),
    '--runs', String(sharedContract.timedRuns),
    '--cache-mode', cacheMode,
    '--tjs-version', tjsVersion,
    '--dtype', resolvedTjsDtype,
    '--profile-ops', resolvedProfileOps ? 'on' : 'off',
    '--timeout', String(resolvedTimeoutMs),
    '--server-port', String(resolvedServerPort),
    '--seed', String(sharedContract.seed),
    '--temperature', String(sharedContract.sampling.temperature),
    '--top-k', String(sharedContract.sampling.topK),
    '--top-p', String(sharedContract.sampling.topP),
  ];
  const stableBrowserArgs = appendBrowserArgs([], STABLE_BROWSER_ARGS);
  for (const browserArg of stableBrowserArgs) {
    args.push('--browser-arg', browserArg);
  }
  if (resolvedLoadMode != null) {
    args.push('--load-mode', String(resolvedLoadMode));
  }
  if (sharedContract.useChatTemplate === true) {
    args.push('--use-chat-template');
  }
  if (localModelPath) args.push('--local-model-path', localModelPath);
  if (resolvedBrowserBaseUrl) args.push('--browser-base-url', String(resolvedBrowserBaseUrl));
  if (resolvedBrowserExecutable) args.push('--browser-executable', String(resolvedBrowserExecutable));
  if (options.browserConsole === true) args.push('--browser-console');

  console.error(`[compare] running TJS v${tjsVersion} (${cacheMode})...`);
  const runOnce = async (overrideArgs = []) => {
    const { stdout } = await execFileAsync('node', [...args, ...overrideArgs], {
      cwd: DOPPLER_ROOT,
      timeout: resolvedTimeoutMs,
      maxBuffer: 10 * 1024 * 1024,
    });
    return parseJsonBlock(stdout, `TJS ${modelId} (${cacheMode})`);
  };
  try {
    return await runOnce();
  } catch (error) {
    const message = String(error?.message || '');
    const pageClosed = /Target page, context or browser has been closed/i.test(message);
    if (pageClosed) {
      const retryArgs = resolvedProfileOps ? ['--profile-ops', 'off'] : [];
      console.error(
        resolvedProfileOps
          ? '[compare] TJS closed page/context during profiled run; retrying with --profile-ops off...'
          : '[compare] TJS closed page/context; retrying once...'
      );
      try {
        return await runOnce(retryArgs);
      } catch (retryError) {
        console.error(`[compare] TJS (${cacheMode}) retry failed: ${retryError.message}`);
        return toFailurePayload('transformers.js', retryError);
      }
    }
    console.error(`[compare] TJS (${cacheMode}) failed: ${error.message}`);
    return toFailurePayload('transformers.js', error);
  }
}

function resolveMetric(result, definition, engine) {
  if (definition == null || typeof definition !== 'object') return null;
  const metricEngine = engine;
  if (definition.derived != null) {
    const spec = definition.derived[metricEngine];
    if (spec != null) {
      const numerator = firstFiniteNumber(result, spec.numeratorPaths || []);
      const denominator = firstFiniteNumber(result, spec.denominatorPaths || []);
      if (Number.isFinite(numerator) && Number.isFinite(denominator) && denominator > 0) {
        return numerator / denominator;
      }
    }
  }
  const candidates = definition.paths?.[metricEngine];
  if (!Array.isArray(candidates)) return null;
  return firstFiniteNumber(result, candidates);
}

function assertRequiredHarnessMetrics(result, harnessMetricConfig, phaseLabel, engineLabel) {
  if (!harnessMetricConfig || result?.failed) return;
  assertCanonicalRunContract(result, phaseLabel, engineLabel);
  const requiredMetrics = harnessMetricConfig.requiredMetrics || [];
  const requiredPaths = harnessMetricConfig.paths || {};

  for (const metricId of requiredMetrics) {
    const candidates = requiredPaths[metricId];
    const value = firstFiniteNumber(result, candidates);
    if (!Number.isFinite(value)) {
      throw new Error(`${engineLabel} result missing required metric "${metricId}" during ${phaseLabel}`);
    }
  }
}

function resolveCanonicalTimingContainer(result) {
  const candidates = [
    result,
    result?.result,
    result?.result?.result,
  ];
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
      continue;
    }
    if (candidate.timing && typeof candidate.timing === 'object' && !Array.isArray(candidate.timing)) {
      return candidate;
    }
  }
  return null;
}

function assertCanonicalRunContract(result, phaseLabel, engineLabel) {
  if (result?.failed) return;

  const timingContainer = resolveCanonicalTimingContainer(result);
  const timing = timingContainer?.timing;
  if (!timingContainer || !timing || typeof timing !== 'object' || Array.isArray(timing)) {
    throw new Error(`${engineLabel} result missing timing object during ${phaseLabel}`);
  }

  for (const field of REQUIRED_COMPARE_TIMING_FIELDS) {
    const value = timing[field];
    if (!Number.isFinite(value)) {
      throw new Error(`${engineLabel} result missing timing.${field} during ${phaseLabel}`);
    }
  }

  if (timing.cacheMode !== 'cold' && timing.cacheMode !== 'warm') {
    throw new Error(`${engineLabel} result has invalid timing.cacheMode "${timing.cacheMode}" during ${phaseLabel}`);
  }

  if (timing.loadMode !== 'opfs' && timing.loadMode !== 'http' && timing.loadMode !== 'memory') {
    throw new Error(`${engineLabel} result has invalid timing.loadMode "${timing.loadMode}" during ${phaseLabel}`);
  }

  const containerCacheMode = timingContainer.cacheMode;
  if (containerCacheMode != null && containerCacheMode !== timing.cacheMode) {
    throw new Error(`${engineLabel} result cacheMode mismatch: top-level "${containerCacheMode}" vs timing.cacheMode "${timing.cacheMode}"`);
  }
  const containerLoadMode = timingContainer.loadMode;
  if (containerLoadMode != null && containerLoadMode !== timing.loadMode) {
    throw new Error(`${engineLabel} result loadMode mismatch: top-level "${containerLoadMode}" vs timing.loadMode "${timing.loadMode}"`);
  }
}

function assertHarnessRequiredMetricCoverage(harnessMetricConfig, requiredMetricIds, engineLabel) {
  const declaredRequired = Array.isArray(harnessMetricConfig.requiredMetrics)
    ? harnessMetricConfig.requiredMetrics
    : [];
  const requiredSet = new Set(declaredRequired);
  for (const metricId of requiredMetricIds) {
    if (!requiredSet.has(metricId)) {
      throw new Error(`harness for ${engineLabel} is missing required metric "${metricId}" in normalization.requiredMetrics`);
    }
    if (!Array.isArray(harnessMetricConfig.paths?.[metricId]) || harnessMetricConfig.paths[metricId].length < 1) {
      throw new Error(`harness for ${engineLabel} is missing metric path mapping for required metric "${metricId}"`);
    }
  }
}

function getDopplerGeneratedText(result) {
  if (typeof result?.result?.output === 'string' && result.result.output.trim()) {
    return result.result.output;
  }
  if (typeof result?.output === 'string' && result.output.trim()) {
    return result.output;
  }
  if (typeof result?.result?.result?.output === 'string' && result.result.result.output.trim()) {
    return result.result.result.output;
  }
  return result?.result?.metrics?.generatedText ?? result?.metrics?.generatedText ?? null;
}

function getTjsGeneratedText(result) {
  return result?.generatedText ?? result?.output ?? null;
}

function normalizeWhitespace(value) {
  return String(value || '').trim().replace(/\s+/g, ' ');
}

function firstMismatchIndex(a, b) {
  const left = String(a || '');
  const right = String(b || '');
  const limit = Math.min(left.length, right.length);
  for (let i = 0; i < limit; i += 1) {
    if (left[i] !== right[i]) return i;
  }
  if (left.length !== right.length) return limit;
  return -1;
}

function computeTokenPrefixMatch(a, b) {
  const left = normalizeWhitespace(a);
  const right = normalizeWhitespace(b);
  if (!left && !right) {
    return {
      leftTokenCount: 0,
      rightTokenCount: 0,
      matchingPrefixTokens: 0,
      firstMismatchTokenIndex: -1,
    };
  }
  const leftTokens = left ? left.split(' ') : [];
  const rightTokens = right ? right.split(' ') : [];
  const limit = Math.min(leftTokens.length, rightTokens.length);
  let prefix = 0;
  while (prefix < limit && leftTokens[prefix] === rightTokens[prefix]) {
    prefix += 1;
  }
  const mismatchIndex = prefix < limit
    ? prefix
    : (leftTokens.length === rightTokens.length ? -1 : limit);
  return {
    leftTokenCount: leftTokens.length,
    rightTokenCount: rightTokens.length,
    matchingPrefixTokens: prefix,
    firstMismatchTokenIndex: mismatchIndex,
  };
}

function buildCorrectnessReport(dopplerResult, tjsResult, prompt) {
  const dopplerFailed = dopplerResult?.failed === true;
  const tjsFailed = tjsResult?.failed === true;
  const dopplerText = getDopplerGeneratedText(dopplerResult);
  const tjsText = getTjsGeneratedText(tjsResult);
  if (dopplerFailed || tjsFailed) {
    return {
      prompt: typeof prompt === 'string' ? prompt.slice(0, 120) : null,
      status: 'unavailable',
      reason: 'engine-run-failed',
      engines: {
        doppler: dopplerFailed ? (dopplerResult?.error?.message || 'failed') : null,
        transformersjs: tjsFailed ? (tjsResult?.error?.message || 'failed') : null,
      },
      doppler: dopplerText,
      transformersjs: tjsText,
      exactMatch: null,
      normalizedMatch: null,
    };
  }
  if (dopplerText == null || tjsText == null) {
    return {
      prompt: typeof prompt === 'string' ? prompt.slice(0, 120) : null,
      status: 'unavailable',
      reason: 'missing-generated-text',
      doppler: dopplerText,
      transformersjs: tjsText,
      exactMatch: null,
      normalizedMatch: null,
    };
  }
  const exactMatch = typeof dopplerText === 'string' && typeof tjsText === 'string'
    ? dopplerText === tjsText
    : null;
  const normalizedMatch = typeof dopplerText === 'string' && typeof tjsText === 'string'
    ? normalizeWhitespace(dopplerText) === normalizeWhitespace(tjsText)
    : null;
  const charMismatchIndex = firstMismatchIndex(dopplerText, tjsText);
  const tokenMatch = computeTokenPrefixMatch(dopplerText, tjsText);
  const status = exactMatch === true || normalizedMatch === true ? 'match' : 'mismatch';
  return {
    prompt: typeof prompt === 'string' ? prompt.slice(0, 120) : null,
    status,
    doppler: dopplerText,
    transformersjs: tjsText,
    exactMatch,
    normalizedMatch,
    charMismatchIndex,
    tokenMatch,
  };
}

function assertDeterministicParityContract(sharedContract, tjsResult, phaseLabel) {
  const temperature = Number(sharedContract?.sampling?.temperature);
  if (!Number.isFinite(temperature) || temperature > 0) {
    return;
  }
  if (tjsResult?.failed) {
    return;
  }
  const strictGreedy = tjsResult?.generationConfig?.strictDeterministicGreedy;
  if (strictGreedy !== true) {
    throw new Error(
      `TJS result missing strict deterministic greedy contract during ${phaseLabel}; `
      + 'refusing correctness comparison.'
    );
  }
  const doSample = tjsResult?.sampling?.doSample;
  if (doSample !== false) {
    throw new Error(
      `TJS sampling.doSample must be false during deterministic parity (${phaseLabel}); `
      + `got ${String(doSample)}.`
    );
  }
}

function formatVal(v, unit) {
  if (v == null || !Number.isFinite(v)) return '-';
  if (unit === 'ms') {
    if (v >= 1000) return `${(v / 1000).toFixed(1)}s`;
    return `${v.toFixed(0)}ms`;
  }
  if (unit === 'tok/s') return `${v.toFixed(1)}`;
  return String(v);
}

function formatDelta(dopplerVal, tjsVal, higherBetter) {
  if (dopplerVal == null || tjsVal == null || !Number.isFinite(dopplerVal) || !Number.isFinite(tjsVal)) return '-';
  if (tjsVal === 0 && dopplerVal === 0) return 'same';
  const ref = Math.min(Math.abs(dopplerVal), Math.abs(tjsVal));
  if (ref === 0) return '-';
  // Express as "X wins by Y%"
  const pct = Math.abs(dopplerVal - tjsVal) / ref * 100;
  const dopplerWins = higherBetter ? dopplerVal > tjsVal : dopplerVal < tjsVal;
  const winner = dopplerWins ? 'Doppler' : 'TJS';
  return `${pct.toFixed(0)}% ${winner}`;
}

function printRow(label, dopplerVal, tjsVal, unit, higherBetter) {
  const dStr = formatVal(dopplerVal, unit);
  const tStr = formatVal(tjsVal, unit);
  const delta = formatDelta(dopplerVal, tjsVal, higherBetter);
  console.log(`  ${label.padEnd(20)} ${dStr.padStart(14)} ${tStr.padStart(14)} ${delta.padStart(18)}`);
}

function printSection(title, dopplerResult, tjsResult, rows) {
  console.log(`\n=== ${title} ===`);
  console.log(`  ${''.padEnd(20)} ${'Doppler'.padStart(14)} ${'TJS'.padStart(14)} ${'delta'.padStart(18)}`);
  for (const row of rows) {
    const dVal = resolveMetric(dopplerResult, row, 'doppler');
    const tVal = resolveMetric(tjsResult, row, 'transformersjs');
    printRow(row.label, dVal, tVal, row.unit, row.higherBetter);
  }
}

function compactTimestamp(timestamp = null) {
  const d = timestamp == null ? new Date() : new Date(timestamp);
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}T${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  if (flags.help === true || flags.h === true) {
    console.log(usage());
    return;
  }
  const runTimestamp = parseTimestampValue(flags.timestamp, '--timestamp');
  const workloadCatalog = await loadWorkloadCatalog();
  const defaultWorkloadId = workloadCatalog.defaultWorkloadId;
  if (!defaultWorkloadId) {
    throw new Error('workloads.json defaults.compareEngines is required for compare-engines defaults');
  }
  resolveWorkloadById(workloadCatalog, defaultWorkloadId);
  const useDefaultWorkload = flags.workload == null
    && flags.prompt == null
    && flags['prefill-tokens'] == null
    && flags['max-tokens'] == null;
  const workloadId = flags.workload ?? (useDefaultWorkload ? defaultWorkloadId : null);
  const workload = workloadId ? resolveWorkloadById(workloadCatalog, workloadId) : null;
  const prefillTokenTarget = parsePositiveInt(
    flags['prefill-tokens'],
    workload?.prefillTokens ?? null,
    '--prefill-tokens'
  );
  const workloadSampling = workload?.sampling || DEFAULT_SHARED_SAMPLING;
  const sampling = {
    temperature: parseNonNegativeNumber(flags.temperature, workloadSampling.temperature, '--temperature'),
    topK: parsePositiveInt(flags['top-k'], workloadSampling.topK, '--top-k'),
    topP: parseTopP(flags['top-p'], workloadSampling.topP, '--top-p'),
  };
  const promptInput = flags.prompt
    || (Number.isFinite(prefillTokenTarget) ? buildSyntheticPrompt(prefillTokenTarget) : DEFAULT_PROMPT);
  const maxTokensInput = flags['max-tokens'] ?? workload?.decodeTokens ?? DEFAULT_MAX_TOKENS;

  const compareProfileConfig = await loadCompareEnginesConfig(flags['compare-config']);
  const defaultDopplerModelId = resolveDefaultDopplerModelId(compareProfileConfig);
  const dopplerModelId = flags['model-id'] || defaultDopplerModelId;
  const compareProfile = resolveCompareProfile(compareProfileConfig, dopplerModelId);
  const dopplerModelBaseDir = compareProfile.modelBaseDir || 'local';
  const compareMetricContractConfig = await loadCompareMetricContract(flags['compare-metric-contract']);
  assertCompareMetricContractCompleteness(compareMetricContractConfig.metrics);
  const [dopplerHarnessMetricPaths, tjsHarnessMetricPaths] = await Promise.all([
    loadCompareHarnessMetricPaths(DOPPLER_HARNESS_PATH, 'doppler'),
    loadCompareHarnessMetricPaths(TJS_HARNESS_PATH, 'transformersjs'),
  ]);
  const compareMetricContract = buildCompareMetricContract(
    compareMetricContractConfig.metrics,
    {
      doppler: dopplerHarnessMetricPaths.paths,
      transformersjs: tjsHarnessMetricPaths.paths,
      }
  );
  const requiredMetricIds = compareMetricContract
    .filter((entry) => entry.required === true)
    .map((entry) => entry.id);
  assertHarnessRequiredMetricCoverage(dopplerHarnessMetricPaths, requiredMetricIds, 'doppler');
  assertHarnessRequiredMetricCoverage(tjsHarnessMetricPaths, requiredMetricIds, 'transformersjs');

  const dopplerModelUrl = flags['model-url'] || `/models/${dopplerModelBaseDir}/${dopplerModelId}`;
  const resolvedDefaultTjsModel = compareProfile.defaultTjsModelId || null;
  const tjsModelOverridden = flags['tjs-model'] != null;
  const tjsModelId = flags['tjs-model'] || resolvedDefaultTjsModel;
  if (!tjsModelId) {
    throw new Error(
      `No default Transformers.js model mapping for "${dopplerModelId}". `
      + 'Set defaultTjsModelId in compare-engines.config.json or pass --tjs-model.'
    );
  }
  const tjsLocalModelPath = flags['tjs-local-model-path'] || null;
  const tjsVersion = flags['tjs-version'] || '4';
  const tjsDtype = parseChoice(flags['tjs-dtype'], ['fp16', 'q4', 'q4f16'], '--tjs-dtype', 'fp16');
  const mode = flags.mode || 'all';
  const sharedContract = buildSharedBenchmarkContract({
    prompt: promptInput,
    maxTokens: maxTokensInput,
    warmupRuns: flags.warmup,
    timedRuns: flags.runs,
    seed: flags.seed,
    loadMode: parseLoadMode(flags['load-mode'], '--load-mode', null),
    sampling,
    useChatTemplate: flags['use-chat-template'] === true
      ? true
      : parseOnOff(flags['use-chat-template'], false, '--use-chat-template'),
  });
  const prompt = sharedContract.prompt;
  const maxTokens = sharedContract.maxTokens;
  const warmupRuns = sharedContract.warmupRuns;
  const runs = sharedContract.timedRuns;
  const seed = sharedContract.seed;
  const decodeProfile = parseDecodeProfile(flags['decode-profile']);
  const tjsProfileOps = parseOnOff(flags['tjs-profile-ops'], DEFAULT_TJS_PROFILE_OPS, '--tjs-profile-ops');
  const compareSharedTimeoutMs = flags['timeout-ms'] != null
    ? parsePositiveInt(flags['timeout-ms'], DEFAULT_COMPARE_TIMEOUT_MS, '--timeout-ms')
    : null;
  const compareDopplerTimeoutMsRequested = flags['doppler-timeout-ms'] != null
    ? parsePositiveInt(flags['doppler-timeout-ms'], DEFAULT_DOPPLER_TIMEOUT_MS, '--doppler-timeout-ms')
    : null;
  const compareTjsTimeoutMsRequested = flags['tjs-timeout-ms'] != null
    ? parsePositiveInt(flags['tjs-timeout-ms'], DEFAULT_TJS_TIMEOUT_MS, '--tjs-timeout-ms')
    : null;
  if (
    compareSharedTimeoutMs != null
    && compareDopplerTimeoutMsRequested != null
    && compareSharedTimeoutMs !== compareDopplerTimeoutMsRequested
  ) {
    throw new Error('Do not use --timeout-ms and --doppler-timeout-ms with different values.');
  }
  if (
    compareSharedTimeoutMs != null
    && compareTjsTimeoutMsRequested != null
    && compareSharedTimeoutMs !== compareTjsTimeoutMsRequested
  ) {
    throw new Error('Do not use --timeout-ms and --tjs-timeout-ms with different values.');
  }
  const compareDopplerTimeoutMs = compareDopplerTimeoutMsRequested
    ?? compareSharedTimeoutMs
    ?? DEFAULT_DOPPLER_TIMEOUT_MS;
  const compareTjsTimeoutMs = compareTjsTimeoutMsRequested
    ?? compareSharedTimeoutMs
    ?? DEFAULT_TJS_TIMEOUT_MS;
  const tjsServerPort = parseNonNegativeInt(flags['tjs-server-port'], DEFAULT_TJS_SERVER_PORT, '--tjs-server-port');
  const tjsBrowserConsole = flags['tjs-browser-console'] === true;
  const browserBaseUrl = flags['browser-base-url'] || null;
  const browserExecutable = flags['browser-executable'] || null;
  const hasCustomDopplerBatchSize = flags['doppler-batch-size'] != null;
  const hasCustomDopplerReadbackInterval = flags['doppler-readback-interval'] != null;
  const hasCustomDopplerDecodeTuning = hasCustomDopplerBatchSize || hasCustomDopplerReadbackInterval;
  if (hasCustomDopplerDecodeTuning && decodeProfile !== 'custom') {
    throw new Error(
      'Use --decode-profile custom when setting --doppler-batch-size or --doppler-readback-interval.'
    );
  }
  const decodeProfilePreset = DECODE_PROFILE_PRESETS[decodeProfile] || DECODE_PROFILE_PRESETS[DEFAULT_DECODE_PROFILE];
    const dopplerKernelPathOverride = flags['doppler-kernel-path'] ?? DEFAULT_DOPPLER_KERNEL_PATH;
    const dopplerKernelResolution = resolveDopplerKernelPath(compareProfile, dopplerKernelPathOverride);
    const dopplerKernelPath = dopplerKernelResolution.kernelPath;
    assertKernelPathAllowedForModel(dopplerModelId, dopplerKernelPath);
    const dopplerBatchSize = parsePositiveInt(
      flags['doppler-batch-size'],
      decodeProfilePreset.batchSize,
    '--doppler-batch-size'
  );
  const dopplerReadbackInterval = parsePositiveInt(
    flags['doppler-readback-interval'],
    decodeProfilePreset.readbackInterval,
    '--doppler-readback-interval'
  );
  const dopplerTokensPerReadback = dopplerBatchSize * dopplerReadbackInterval;
  const dopplerNoOpfsCache = flags['doppler-no-opfs-cache'] === true;
  const dopplerBrowserUserData = flags['doppler-browser-user-data'] || null;
    const dopplerBrowserPort = flags['doppler-browser-port'] != null
      ? parseNonNegativeInt(
        flags['doppler-browser-port'],
        DEFAULT_DOPPLER_BROWSER_PORT,
        '--doppler-browser-port'
      )
      : null;
  const runtimeConfigOverride = parseRuntimeConfigJson(flags['runtime-config-json']);
  const runtimeConfigOverrideHash = runtimeConfigOverride == null
    ? null
    : hashJsonPayload(runtimeConfigOverride);
  const jsonOutput = flags.json === true;
  const shouldSave = flags.save === true;
  const skipMatrixUpdate = flags['skip-matrix-update'] === true;
  const saveDir = flags['save-dir'] || path.join(DOPPLER_ROOT, 'benchmarks', 'vendors', 'results');

  const validModes = ['compute', 'cold', 'warm', 'all'];
  if (!validModes.includes(mode)) {
    console.error(`Invalid --mode "${mode}". Must be one of: ${validModes.join(', ')}`);
    process.exit(1);
  }

  console.error(`[compare] Doppler model: ${dopplerModelId}`);
  console.error(`[compare] TJS model:     ${tjsModelId} (v${tjsVersion})`);
  if (tjsModelOverridden && tjsModelId !== resolvedDefaultTjsModel) {
    console.error(
      `[compare] warning: --tjs-model overrides profile default (${resolvedDefaultTjsModel}); `
      + 'this may invalidate correctness comparability.'
    );
  }
  console.error(
    `[compare] model-profile source: ${compareProfile.source}`
    + ` (dopplerBaseDir=${dopplerModelBaseDir}, defaultKernelPath=${dopplerKernelResolution.kernelPath ?? 'manifest-default'})`
  );
  console.error(
    `[compare] Doppler kernel path: ${dopplerKernelPath ?? 'manifest-default'} `
    + `(${dopplerKernelResolution.source})`
  );
  console.error(
    `[compare] mode: ${mode}, maxTokens: ${maxTokens}, warmupRuns: ${warmupRuns}, runs: ${runs}, `
    + `decodeProfile: ${decodeProfile}, workload=${workloadId ?? 'none'}, `
    + `sampling=(temp=${sharedContract.sampling.temperature}, topK=${sharedContract.sampling.topK}, topP=${sharedContract.sampling.topP}), `
    + `useChatTemplate: ${sharedContract.useChatTemplate === true ? 'on' : 'off'}, `
    + `dopplerBatchSize: ${dopplerBatchSize}, `
    + `dopplerReadbackInterval: ${dopplerReadbackInterval}, `
    + `dopplerTokensPerReadback: ${dopplerTokensPerReadback}, `
    + `dopplerTimeoutMs: ${compareDopplerTimeoutMs}, `
    + `tjsTimeoutMs: ${compareTjsTimeoutMs}, `
    + `tjsDtype: ${tjsDtype}`
  );

  const report = {
    timestamp: runTimestamp ?? new Date().toISOString(),
    benchmarkPolicy: {
      source: BENCHMARK_POLICY.source,
      schemaVersion: BENCHMARK_POLICY.schemaVersion,
      sourceSha256: BENCHMARK_POLICY.sourceSha256,
      updated: BENCHMARK_POLICY.updated,
    },
    compareConfig: {
      source: compareProfileConfig.source || 'benchmarks/vendors/compare-engines.config.json',
      schemaVersion: compareProfileConfig.schemaVersion,
      sourceSha256: compareProfileConfig.sourceSha256,
      profileSource: compareProfile.source,
      updated: compareProfileConfig.updated,
    },
    metricContract: {
      source: compareMetricContractConfig.source || 'benchmarks/vendors/compare-metrics.json',
      schemaVersion: compareMetricContractConfig.schemaVersion ?? DEFAULT_COMPARE_METRIC_CONTRACT_SCHEMA_VERSION,
      sourceSha256: compareMetricContractConfig.sourceSha256,
      updated: compareMetricContractConfig.updated,
      metrics: compareMetricContract.map((entry) => ({
        id: entry.id,
        label: entry.label,
        unit: entry.unit,
        higherBetter: entry.higherBetter,
        required: entry.required,
      })),
    },
    harnesses: {
      doppler: {
        source: dopplerHarnessMetricPaths.source,
        sourceSha256: dopplerHarnessMetricPaths.sourceSha256,
        requiredMetrics: requiredMetricIds,
      },
      transformersjs: {
        source: tjsHarnessMetricPaths.source,
        sourceSha256: tjsHarnessMetricPaths.sourceSha256,
        requiredMetrics: requiredMetricIds,
      },
    },
    runtimeConfig: {
      overrideProvided: runtimeConfigOverride !== null,
      runtimeConfigSha256: runtimeConfigOverrideHash,
      compareProfileSchemaVersion: compareProfileConfig.schemaVersion ?? DEFAULT_COMPARE_CONFIG_SCHEMA_VERSION,
    },
    dopplerModelId,
    dopplerDtype: parseDopplerDtype(dopplerModelId),
    dopplerKernelPath: dopplerKernelPath ?? 'manifest-default',
    dopplerKernelPathSource: dopplerKernelResolution.source,
    decodeProfile,
    dopplerBatchSize,
    dopplerReadbackInterval,
    dopplerTokensPerReadback,
    tjsModelId,
    transformersjsDtype: tjsDtype,
    mode,
    prompt,
    maxTokens,
    warmupRuns,
    runs,
    seed,
    workload: {
      id: workloadId,
      prefillTokenTarget: prefillTokenTarget ?? null,
      decodeTokenTarget: maxTokens,
    },
    environment: {
      host: {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
      },
      browser: {
        executable: browserExecutable || null,
        baseUrl: browserBaseUrl || null,
      },
    },
    timeoutMs: compareSharedTimeoutMs ?? compareDopplerTimeoutMs,
    dopplerTimeoutMs: compareDopplerTimeoutMs,
    tjsTimeoutMs: compareTjsTimeoutMs,
    methodology: {
      prefillTokensPerSec: 'prompt_tokens / prefill_ms',
      deterministicDecoding: {
        seed,
        temperature: sharedContract.sampling.temperature,
        topK: sharedContract.sampling.topK,
        topP: sharedContract.sampling.topP,
      },
      promptParity: {
        dopplerChatTemplateEnabled: sharedContract.useChatTemplate,
        transformersChatTemplateEquivalent: sharedContract.useChatTemplate
          ? 'model-chat-template'
          : 'raw-prompt',
      },
      cacheSemantics: {
        warm: 'Prime engine cache before timed run, then run from local persistent cache only (Doppler OPFS; TJS persistent browser cache).',
        cold: 'Wipe engine-specific persistent cache state before run (Doppler OPFS/profile wipe; TJS profile wipe).',
      },
      dopplerDecodeCadence: {
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        tokensPerReadback: dopplerTokensPerReadback,
      },
      transformersjsDecodeCadence: {
        streamerCallbackGranularityTokens: 1,
        readbackControl: 'runtime-internal',
      },
    },
    sections: {},
  };

  // Compute measures warm-cache behavior and reports both parity/throughput Doppler cadence.
  const needCompute = mode === 'compute' || mode === 'all';
  const needWarm = mode === 'warm' || mode === 'all';
  const needCold = mode === 'cold' || mode === 'all';

  let dopplerComputeParity = null;
  let dopplerComputeThroughput = null;
  let tjsCompute = null;
  let dopplerWarm = null;
  let tjsWarm = null;
  let dopplerCold = null;
  let tjsCold = null;

  if (needCompute) {
    tjsCompute = await runTjs(
      tjsModelId,
      sharedContract,
      'warm',
      tjsVersion,
      tjsLocalModelPath,
      {
        profileOps: tjsProfileOps,
        timeoutMs: compareTjsTimeoutMs,
        serverPort: tjsServerPort,
        tjsDtype,
        browserExecutable,
        browserConsole: tjsBrowserConsole,
        browserBaseUrl,
        loadMode: resolveRunLoadMode('warm', sharedContract.loadMode),
      }
    );
    dopplerComputeParity = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: DECODE_PROFILE_PRESETS.parity.batchSize,
        readbackInterval: DECODE_PROFILE_PRESETS.parity.readbackInterval,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
        browserExecutable,
        browserBaseUrl,
        timeoutMs: compareDopplerTimeoutMs,
        runtimeConfigJson: runtimeConfigOverride,
        loadMode: resolveRunLoadMode('warm', sharedContract.loadMode),
      }
    );

    assertRequiredHarnessMetrics(
      dopplerComputeParity,
      { ...dopplerHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'compute/parity',
      'Doppler'
    );
    assertRequiredHarnessMetrics(
      tjsCompute,
      { ...tjsHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'compute',
      'TJS'
    );
    assertDeterministicParityContract(sharedContract, tjsCompute, 'compute');

    dopplerComputeThroughput = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: DECODE_PROFILE_PRESETS.throughput.batchSize,
        readbackInterval: DECODE_PROFILE_PRESETS.throughput.readbackInterval,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
        browserExecutable,
        browserBaseUrl,
        timeoutMs: compareDopplerTimeoutMs,
        runtimeConfigJson: runtimeConfigOverride,
        loadMode: resolveRunLoadMode('warm', sharedContract.loadMode),
      }
    );

    assertRequiredHarnessMetrics(
      dopplerComputeThroughput,
      { ...dopplerHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'compute/throughput',
      'Doppler'
    );

    report.sections.compute = {
      parity: { doppler: dopplerComputeParity, transformersjs: tjsCompute },
      throughput: { doppler: dopplerComputeThroughput, transformersjs: tjsCompute },
    };
    report.correctness = buildCorrectnessReport(dopplerComputeParity, tjsCompute, sharedContract.prompt);
  }

  if (needWarm) {
    dopplerWarm = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
        browserExecutable,
        browserBaseUrl,
        timeoutMs: compareDopplerTimeoutMs,
        runtimeConfigJson: runtimeConfigOverride,
        loadMode: resolveRunLoadMode('warm', sharedContract.loadMode),
      }
    );
    assertRequiredHarnessMetrics(
      dopplerWarm,
      { ...dopplerHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'warm',
      'Doppler'
    );

    tjsWarm = await runTjs(
      tjsModelId,
      sharedContract,
      'warm',
      tjsVersion,
      tjsLocalModelPath,
      {
        profileOps: tjsProfileOps,
        timeoutMs: compareTjsTimeoutMs,
        serverPort: tjsServerPort,
        tjsDtype,
        browserExecutable,
        browserConsole: tjsBrowserConsole,
        browserBaseUrl,
        loadMode: resolveRunLoadMode('warm', sharedContract.loadMode),
      }
    );
    assertDeterministicParityContract(sharedContract, tjsWarm, 'warm');
    assertRequiredHarnessMetrics(
      tjsWarm,
      { ...tjsHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'warm',
      'TJS'
    );

    report.sections.warm = { doppler: dopplerWarm, transformersjs: tjsWarm };
  }

  if (needCold) {
    dopplerCold = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      sharedContract,
      'cold',
      {
        kernelPath: dopplerKernelPath,
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
        browserExecutable,
        browserBaseUrl,
        timeoutMs: compareDopplerTimeoutMs,
        runtimeConfigJson: runtimeConfigOverride,
        loadMode: resolveRunLoadMode('cold', sharedContract.loadMode),
      }
    );
    assertRequiredHarnessMetrics(
      dopplerCold,
      { ...dopplerHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'cold',
      'Doppler'
    );

    tjsCold = await runTjs(
      tjsModelId,
      sharedContract,
      'cold',
      tjsVersion,
      tjsLocalModelPath,
      {
        profileOps: tjsProfileOps,
        timeoutMs: compareTjsTimeoutMs,
        serverPort: tjsServerPort,
        tjsDtype,
        browserExecutable,
        browserConsole: tjsBrowserConsole,
        browserBaseUrl,
        loadMode: resolveRunLoadMode('cold', sharedContract.loadMode),
      }
    );
    assertDeterministicParityContract(sharedContract, tjsCold, 'cold');
    assertRequiredHarnessMetrics(
      tjsCold,
      { ...tjsHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'cold',
      'TJS'
    );

    report.sections.cold = { doppler: dopplerCold, transformersjs: tjsCold };
  }

  if (jsonOutput) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    const decodeProfileLabel = decodeProfilePreset?.label || 'custom decode cadence';
    console.log(
      `[method] prefill=prompt_tokens / prefill_ms, decodeProfile=${decodeProfile} ` +
      `(${decodeProfileLabel}), Doppler tokens/readback=${dopplerTokensPerReadback}`
    );
    const computeRows = compareMetricContract;

    if (mode === 'compute' || mode === 'all') {
      printSection('COMPUTE (PARITY)', dopplerComputeParity, tjsCompute, computeRows);
      printSection('COMPUTE (THROUGHPUT)', dopplerComputeThroughput, tjsCompute, computeRows);
    }
    if (mode === 'warm' || mode === 'all') {
      printSection('WARM START', dopplerWarm, tjsWarm, computeRows);
    }
    if (mode === 'cold' || mode === 'all') {
      printSection('COLD START', dopplerCold, tjsCold, computeRows);
    }
    if (report.correctness) {
      const c = report.correctness;
      const truncate = (s, n) => typeof s === 'string' ? (s.length > n ? s.slice(0, n) + '…' : s) : '(unavailable)';
      console.log('\n=== CORRECTNESS (last compute run, parity cadence) ===');
      console.log(`  Doppler:   ${truncate(c.doppler, 200)}`);
      console.log(`  TJS:       ${truncate(c.transformersjs, 200)}`);
      if (c.status === 'unavailable') {
        const reason = c.reason || 'unavailable';
        console.log(`  Match:     unavailable (${reason})`);
        if (c.engines?.doppler) {
          console.log(`  Doppler error: ${truncate(c.engines.doppler, 200)}`);
        }
        if (c.engines?.transformersjs) {
          console.log(`  TJS error:     ${truncate(c.engines.transformersjs, 200)}`);
        }
      } else {
        const label = c.exactMatch
          ? 'exact match'
          : (c.normalizedMatch ? 'normalized match (whitespace diff only)' : 'MISMATCH');
        console.log(`  Match:     ${label}`);
        if (c.charMismatchIndex != null && c.charMismatchIndex >= 0) {
          console.log(`  First char mismatch index:  ${c.charMismatchIndex}`);
        }
        if (c.tokenMatch && Number.isFinite(c.tokenMatch.matchingPrefixTokens)) {
          console.log(
            `  Matching token prefix:      ${c.tokenMatch.matchingPrefixTokens}/` +
            `${Math.min(c.tokenMatch.leftTokenCount, c.tokenMatch.rightTokenCount)}`
          );
          if (c.tokenMatch.firstMismatchTokenIndex >= 0) {
            console.log(`  First token mismatch index: ${c.tokenMatch.firstMismatchTokenIndex}`);
          }
        }
      }
    }
    console.log('');
  }

  if (shouldSave) {
    await fs.mkdir(saveDir, { recursive: true });
    const ts = compactTimestamp(runTimestamp);
    const filename = `compare_${ts}.json`;
    const filePath = path.join(saveDir, filename);
    const latestPath = path.join(saveDir, 'compare_latest.json');
    await fs.writeFile(filePath, JSON.stringify(report, null, 2), 'utf-8');
    await fs.writeFile(latestPath, JSON.stringify(report, null, 2), 'utf-8');
    console.error(`[compare] saved to ${filePath}`);
    if (!skipMatrixUpdate) {
      const matrixArgs = [
        path.join(DOPPLER_ROOT, 'tools', 'vendor-bench.js'),
        'matrix',
      ];
      if (runTimestamp) {
        matrixArgs.push('--timestamp', runTimestamp);
      }
      await execFileAsync('node', matrixArgs, {
        cwd: DOPPLER_ROOT,
        timeout: 120_000,
        maxBuffer: 10 * 1024 * 1024,
      });
      console.error('[compare] refreshed release matrix artifacts (committed fixtures only by default)');
    }
  }
}

main().catch((error) => {
  console.error(`[compare] ${error.message}`);
  process.exit(1);
});
