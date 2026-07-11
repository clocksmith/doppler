#!/usr/bin/env node

import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import crypto from 'node:crypto';
import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildExecutionContractArtifact } from '../src/config/execution-contract-check.js';
import { buildManifestRequiredInferenceFieldsArtifact } from '../src/config/required-inference-fields-contract-check.js';
import { mergeKernelPathPolicy } from '../src/config/merge-helpers.js';
import { mergeRuntimeValues } from '../src/config/runtime-merge.js';
import {
  DEFAULT_EXTERNAL_MODELS_ROOT,
  buildEntryRemoteBaseUrl,
} from '../src/tooling/hf-registry-utils.js';
import { resolveSyntheticPromptForModel } from '../benchmarks/vendors/workload-prompt.js';
import {
  DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS,
  createResourceTelemetrySampler,
  parseResourceTelemetryMode,
} from './resource-telemetry.js';

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOPPLER_ROOT = path.resolve(__dirname, '..');
const MODEL_INDEX_PATH = path.join(DEFAULT_EXTERNAL_MODELS_ROOT, 'MODEL_INDEX.json');
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
const MODEL_CATALOG_PATH = path.join(
  DOPPLER_ROOT,
  'models',
  'catalog.json',
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
const QUICKSTART_REGISTRY_PATH = path.join(
  DOPPLER_ROOT,
  'src',
  'client',
  'doppler-registry.json',
);
const TRANSFORMERSJS_RUNNER_HTML_PATH = path.join(
  DOPPLER_ROOT,
  'benchmarks',
  'runners',
  'transformersjs-runner.html',
);
const TJS_PACKAGE_JSON_PATH = path.join(
  DOPPLER_ROOT,
  'node_modules',
  '@huggingface',
  'transformers',
  'package.json',
);
const TJS_ORT_WEB_PACKAGE_JSON_PATH = path.join(
  DOPPLER_ROOT,
  'node_modules',
  '@huggingface',
  'transformers',
  'node_modules',
  'onnxruntime-web',
  'package.json',
);
const TJS_ORT_COMMON_PACKAGE_JSON_PATH = path.join(
  DOPPLER_ROOT,
  'node_modules',
  '@huggingface',
  'transformers',
  'node_modules',
  'onnxruntime-common',
  'package.json',
);
const TJS_V4_ORT_COMMON_IMPORT_PATH = '/node_modules/@huggingface/transformers/node_modules/onnxruntime-common/dist/esm/index.js';
const TJS_V4_ORT_WEB_IMPORT_PATH = '/node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs';

const DEFAULT_PROMPT = 'word0 word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15 word16 word17 word18 word19 word20 word21 word22 word23 word24 word25 word26 word27 word28 word29 word30 word31';
const DEFAULT_MAX_TOKENS = 64;
const DEFAULT_WARMUP = 1;
const DEFAULT_RUNS = 3;
const DEFAULT_SEED = 0;
const DEFAULT_DOPPLER_RUN_MAX_BUFFER_BYTES = 10 * 1024 * 1024;
const DEFAULT_DOPPLER_DIAGNOSTIC_MAX_BUFFER_BYTES = 64 * 1024 * 1024;
const DOPPLER_BENCH_CAPTURE_DIR_PREFIX = 'doppler-compare-bench-';
const CHILD_PROCESS_TAIL_CHARS = 8000;
const DOPPLER_COMPARE_COMMANDS = Object.freeze(['bench', 'debug']);
const DEFAULT_SHARED_SAMPLING = Object.freeze({
  temperature: 0,
  topK: 1,
  topP: 1,
  repetitionPenalty: 1,
  greedyThreshold: 0.01,
});
const DOPPLER_MODEL_SOURCES = Object.freeze(['local', 'quickstart-registry']);
const COMPARE_LANES = Object.freeze(['performance_comparable', 'capability_only']);
const COMPARE_PROFILE_RUNTIME_DECODE_PROFILES = Object.freeze(['parity', 'throughput', 'custom']);
const DOPPLER_LOAD_MODES = Object.freeze(['opfs', 'http', 'memory']);
const DEFAULT_COMPARE_LANE = 'performance_comparable';
const DEFAULT_DOPPLER_MODEL_SOURCE = 'local';
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
  requiredCompareMetricIds: Object.freeze([
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
  ]),
  decodeProfiles: Object.freeze({
    default: 'parity',
    profiles: Object.freeze({
      parity: Object.freeze({
        batchSize: 1,
        readbackInterval: 1,
        disableMultiTokenDecode: true,
        speculationMode: 'none',
        stopCheckMode: 'per-token',
        label: 'TJS-like per-token cadence',
      }),
      throughput: Object.freeze({
        batchSize: 8,
        readbackInterval: 8,
        disableMultiTokenDecode: false,
        speculationMode: 'none',
        stopCheckMode: 'batch',
        label: 'Doppler 8x8 throughput-tuned cadence',
      }),
    }),
  }),
  promotionGates: Object.freeze({
    throughputCadence: Object.freeze({
      requireBatchAccounting: true,
      minDecodeTokensPerSecRatioVsParity: 1,
      minBatchResolutionEfficiency: 0.95,
      maxBatchOverrunTokens: 0,
      description: 'A throughput decode cadence is promotable only when it is not slower than parity cadence and its reported batch work resolves almost all executed batch decode tokens before returning to the generation loop.',
    }),
  }),
  browser: Object.freeze({
    compareChannelByPlatform: Object.freeze({
      darwin: 'chromium',
      linux: 'chromium',
      win32: 'chromium',
    }),
    stableArgs: Object.freeze([
      '--disable-breakpad',
      '--disable-gpu-sandbox',
      '--no-sandbox',
    ]),
  }),
  kernelPathPolicy: Object.freeze({
    compareRuntime: Object.freeze({
      mode: 'capability-aware',
      sourceScope: Object.freeze(['model', 'manifest', 'config']),
      allowSources: Object.freeze(['model', 'manifest', 'config']),
      onIncompatible: 'remap',
    }),
    knownBadByModel: Object.freeze({
      'gemma-3-270m-it-f16-af32': Object.freeze(['gemma3-f16-fused-f16a-online']),
    }),
  }),
  doppler: Object.freeze({
    loadModeRuntimeOverlays: Object.freeze({
      http: Object.freeze({
        description: 'Cold HTTP compare runs use direct range reads, skip eager full-shard hash verification, coalesce ranges into shard-sized blocks, and opt RDRR manifests into bounded whole-shard prefetch; manifest preflight and pinned hosted revisions remain recorded in the compare artifact.',
        runtimeConfig: Object.freeze({
          loading: Object.freeze({
            distribution: Object.freeze({
              sourceOrder: Object.freeze(['http']),
            }),
            shardCache: Object.freeze({
              verifyHashes: false,
              rangeCacheBlockBytes: 64 * 1024 * 1024,
              rangeCacheMaxBytes: 512 * 1024 * 1024,
              rangeCacheMinBytes: 4096,
              maxConcurrentLoads: 8,
            }),
            prefetch: Object.freeze({
              allowRangeLoaderPrefetch: true,
              layersAhead: 1,
              maxShards: 8,
            }),
          }),
        }),
      }),
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

function normalizePositiveNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive number`);
  }
  return parsed;
}

function normalizeNonNegativeNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative number`);
  }
  return parsed;
}

function normalizeUnitIntervalNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`${label} must be between 0 and 1`);
  }
  return parsed;
}

function normalizeRequiredBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be a boolean`);
  }
  return value;
}

function normalizeStopCheckMode(value, label) {
  const normalized = typeof value === 'string' ? value.trim() : '';
  if (normalized === 'batch' || normalized === 'per-token') {
    return normalized;
  }
  throw new Error(`${label} must be "batch" or "per-token"`);
}

function normalizeCompareLane(value, label, fallback = DEFAULT_COMPARE_LANE) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (!normalized) {
    return fallback;
  }
  if (COMPARE_LANES.includes(normalized)) {
    return normalized;
  }
  throw new Error(`${label} must be one of: ${COMPARE_LANES.join(', ')}`);
}

function normalizeDopplerModelSource(value, label, fallback = DEFAULT_DOPPLER_MODEL_SOURCE) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (!normalized) {
    return fallback;
  }
  if (DOPPLER_MODEL_SOURCES.includes(normalized)) {
    return normalized;
  }
  throw new Error(`${label} must be one of: ${DOPPLER_MODEL_SOURCES.join(', ')}`);
}

function normalizeDecodeProfiles(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('decodeProfiles must be an object');
  }
  const defaultId = typeof raw.default === 'string' ? raw.default.trim() : '';
  if (!defaultId) {
    throw new Error('decodeProfiles.default must be a non-empty string');
  }
  const profilesRaw = raw.profiles;
  if (!profilesRaw || typeof profilesRaw !== 'object' || Array.isArray(profilesRaw)) {
    throw new Error('decodeProfiles.profiles must be an object');
  }
  const profiles = {};
  for (const [profileId, profileValue] of Object.entries(profilesRaw)) {
    if (typeof profileId !== 'string' || profileId.trim() === '') {
      throw new Error('decodeProfiles.profiles contains an invalid profile id');
    }
    if (!profileValue || typeof profileValue !== 'object' || Array.isArray(profileValue)) {
      throw new Error(`decodeProfiles.profiles.${profileId} must be an object`);
    }
    profiles[profileId] = Object.freeze({
      batchSize: normalizePositiveInteger(profileValue.batchSize, `decodeProfiles.profiles.${profileId}.batchSize`),
      readbackInterval: normalizePositiveInteger(
        profileValue.readbackInterval,
        `decodeProfiles.profiles.${profileId}.readbackInterval`
      ),
      disableMultiTokenDecode: profileValue.disableMultiTokenDecode === true,
      speculationMode: (() => {
        const normalized = typeof profileValue.speculationMode === 'string'
          ? profileValue.speculationMode.trim().toLowerCase()
          : '';
        if (!normalized) return null;
        if (normalized === 'none' || normalized === 'self') return normalized;
        throw new Error(
          `decodeProfiles.profiles.${profileId}.speculationMode must be "none", "self", or omitted`
        );
      })(),
      stopCheckMode: normalizeStopCheckMode(
        profileValue.stopCheckMode ?? 'per-token',
        `decodeProfiles.profiles.${profileId}.stopCheckMode`
      ),
      label: typeof profileValue.label === 'string' && profileValue.label.trim()
        ? profileValue.label.trim()
        : `${profileId} decode profile`,
    });
  }
  if (!Object.prototype.hasOwnProperty.call(profiles, defaultId)) {
    throw new Error(`decodeProfiles.default "${defaultId}" is not defined in decodeProfiles.profiles`);
  }
  return {
    default: defaultId,
    profiles: Object.freeze(profiles),
  };
}

function normalizeThroughputCadencePromotionGate(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('promotionGates.throughputCadence must be an object');
  }
  const description = typeof raw.description === 'string' ? raw.description.trim() : '';
  if (!description) {
    throw new Error('promotionGates.throughputCadence.description must be a non-empty string');
  }
  return Object.freeze({
    requireBatchAccounting: normalizeRequiredBoolean(
      raw.requireBatchAccounting,
      'promotionGates.throughputCadence.requireBatchAccounting'
    ),
    minDecodeTokensPerSecRatioVsParity: normalizePositiveNumber(
      raw.minDecodeTokensPerSecRatioVsParity,
      'promotionGates.throughputCadence.minDecodeTokensPerSecRatioVsParity'
    ),
    minBatchResolutionEfficiency: normalizeUnitIntervalNumber(
      raw.minBatchResolutionEfficiency,
      'promotionGates.throughputCadence.minBatchResolutionEfficiency'
    ),
    maxBatchOverrunTokens: normalizeNonNegativeNumber(
      raw.maxBatchOverrunTokens,
      'promotionGates.throughputCadence.maxBatchOverrunTokens'
    ),
    description,
  });
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

function normalizeChannelByPlatform(raw, label) {
  if (raw == null) {
    return Object.freeze({});
  }
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`${label} must be an object`);
  }
  const normalized = {};
  for (const [platform, channel] of Object.entries(raw)) {
    if (typeof platform !== 'string' || platform.trim() === '') {
      throw new Error(`${label} contains an invalid platform key`);
    }
    if (channel == null) {
      normalized[platform] = null;
      continue;
    }
    if (typeof channel !== 'string' || channel.trim() === '') {
      throw new Error(`${label}.${platform} must be a non-empty string or null`);
    }
    normalized[platform] = channel.trim();
  }
  return Object.freeze(normalized);
}

function cloneJsonObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return JSON.parse(JSON.stringify(value));
}

function normalizeDopplerLoadModeRuntimeOverlays(raw, label) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`${label} must be an object`);
  }

  const normalized = {};
  for (const [loadMode, overlay] of Object.entries(raw)) {
    const normalizedLoadMode = String(loadMode ?? '').trim().toLowerCase();
    if (!DOPPLER_LOAD_MODES.includes(normalizedLoadMode)) {
      throw new Error(`${label}.${loadMode} must use one of these load modes: ${DOPPLER_LOAD_MODES.join(', ')}`);
    }
    if (!overlay || typeof overlay !== 'object' || Array.isArray(overlay)) {
      throw new Error(`${label}.${loadMode} must be an object`);
    }
    const description = typeof overlay.description === 'string' ? overlay.description.trim() : '';
    if (!description) {
      throw new Error(`${label}.${loadMode}.description must be a non-empty string`);
    }
    const runtimeConfig = cloneJsonObject(
      overlay.runtimeConfig,
      `${label}.${loadMode}.runtimeConfig`
    );
    normalized[normalizedLoadMode] = Object.freeze({
      description,
      runtimeConfig,
    });
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
  const requiredCompareMetricIds = Object.freeze(
    normalizeStringArray(
      parsed.requiredCompareMetricIds ?? DEFAULT_BENCHMARK_POLICY.requiredCompareMetricIds,
      'requiredCompareMetricIds'
    )
  );
  const decodeProfiles = normalizeDecodeProfiles(
    parsed.decodeProfiles ?? DEFAULT_BENCHMARK_POLICY.decodeProfiles
  );
  const throughputCadencePromotionGate = normalizeThroughputCadencePromotionGate(
    parsed?.promotionGates?.throughputCadence
      ?? DEFAULT_BENCHMARK_POLICY.promotionGates.throughputCadence
  );
  const stableArgs = Object.freeze(
    normalizeStringArray(
      parsed?.browser?.stableArgs ?? DEFAULT_BENCHMARK_POLICY.browser.stableArgs,
      'browser.stableArgs'
    )
  );
  const compareChannelByPlatform = normalizeChannelByPlatform(
    parsed?.browser?.compareChannelByPlatform
      ?? DEFAULT_BENCHMARK_POLICY.browser.compareChannelByPlatform,
    'browser.compareChannelByPlatform'
  );
  const knownBadByModel = normalizeKnownBadKernelPaths(
    parsed?.kernelPathPolicy?.knownBadByModel
      ?? DEFAULT_BENCHMARK_POLICY.kernelPathPolicy.knownBadByModel
  );
  const compareRuntime = Object.freeze(
    mergeKernelPathPolicy(
      undefined,
      parsed?.kernelPathPolicy?.compareRuntime
        ?? DEFAULT_BENCHMARK_POLICY.kernelPathPolicy.compareRuntime
    )
  );
  const loadModeRuntimeOverlays = normalizeDopplerLoadModeRuntimeOverlays(
    parsed?.doppler?.loadModeRuntimeOverlays
      ?? DEFAULT_BENCHMARK_POLICY.doppler.loadModeRuntimeOverlays,
    'doppler.loadModeRuntimeOverlays'
  );

  return Object.freeze({
    source: BENCHMARK_POLICY_PATH,
    sourceSha256: hashText(raw),
    schemaVersion: parsed.schemaVersion,
    updated: parsed.updated || null,
    timeoutsMs,
    requiredTimingFields,
    requiredCompareMetricIds,
    decodeProfiles,
    promotionGates: Object.freeze({
      throughputCadence: throughputCadencePromotionGate,
    }),
    browser: Object.freeze({ stableArgs, compareChannelByPlatform }),
    kernelPathPolicy: Object.freeze({ compareRuntime, knownBadByModel }),
    doppler: Object.freeze({ loadModeRuntimeOverlays }),
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
const REQUIRED_CANONICAL_TIMING_FIELDS = BENCHMARK_POLICY.requiredTimingFields;
const REQUIRED_COMPARE_METRIC_IDS = BENCHMARK_POLICY.requiredCompareMetricIds;
const DECODE_PROFILE_CONFIGS = BENCHMARK_POLICY.decodeProfiles.profiles;
const THROUGHPUT_CADENCE_PROMOTION_GATE = BENCHMARK_POLICY.promotionGates.throughputCadence;
const DOPPLER_SURFACES = Object.freeze(['auto', 'node', 'browser', 'bun']);
const DOPPLER_FORMATS = Object.freeze(['rdrr', 'safetensors']);
const TJS_FORMATS = Object.freeze(['onnx', 'safetensors']);
const TJS_DTYPES = Object.freeze(['fp16', 'q4', 'q4f16']);
const OUTPUT_PARITY_MATCH_MODES = Object.freeze(['exact-or-normalized', 'decode-valid']);
const DEFAULT_DOPPLER_FORMAT = 'rdrr';
const DEFAULT_TJS_FORMAT = 'onnx';
const DEFAULT_TJS_DTYPE = 'fp16';

function loadModelIndex() {
  if (!fsSync.existsSync(MODEL_INDEX_PATH)) return null;
  try {
    return JSON.parse(fsSync.readFileSync(MODEL_INDEX_PATH, 'utf-8'));
  } catch {
    return null;
  }
}

function resolveFormatSourcePath(modelIndex, format, sourceId) {
  if (!modelIndex || !sourceId) return null;
  const section = modelIndex[format];
  if (!section || typeof section !== 'object') return null;
  const entry = section[sourceId];
  if (!entry) return null;
  const relative = typeof entry === 'string' ? entry : entry.path;
  if (typeof relative !== 'string') return null;
  const volumeRoot = modelIndex.volumeRoot || DEFAULT_EXTERNAL_MODELS_ROOT;
  return path.join(volumeRoot, relative);
}

function assertDopplerLoadModeSupportedForFormat(format, loadMode) {
  if (format === 'rdrr' && loadMode === 'memory') {
    throw new Error(
      'Doppler RDRR compare lanes do not support --load-mode memory. '
      + 'Use --load-mode opfs or --load-mode http for RDRR artifacts, '
      + 'or use --doppler-format safetensors for direct-source memory loading.'
    );
  }
}

const VALID_DECODE_PROFILES = Object.freeze([
  ...Object.keys(DECODE_PROFILE_CONFIGS),
  'custom',
]);
const DEFAULT_DOPPLER_BATCH_SIZE = DECODE_PROFILE_CONFIGS[DEFAULT_DECODE_PROFILE].batchSize;
const DEFAULT_DOPPLER_READBACK_INTERVAL = DECODE_PROFILE_CONFIGS[DEFAULT_DECODE_PROFILE].readbackInterval;
const DEFAULT_DOPPLER_STOP_CHECK_MODE = DECODE_PROFILE_CONFIGS[DEFAULT_DECODE_PROFILE].stopCheckMode;

function resolveComputeDecodeCadence(decodeProfile, profileName, customCadence, runtimeBaseConfig) {
  const readbackMode = resolveRuntimeReadbackMode(runtimeBaseConfig);
  if (decodeProfile === 'custom') {
    return {
      batchSize: customCadence.batchSize,
      readbackInterval: customCadence.readbackInterval,
      stopCheckMode: customCadence.stopCheckMode,
      readbackMode,
      disableMultiTokenDecode: customCadence.disableMultiTokenDecode === true,
      speculationMode: customCadence.speculationMode ?? null,
      runtimeBaseConfig,
    };
  }
  const profileConfig = DECODE_PROFILE_CONFIGS[profileName];
  if (!profileConfig) {
    throw new Error(`Unknown compute decode profile: ${profileName}`);
  }
  return {
    batchSize: profileConfig.batchSize,
    readbackInterval: profileConfig.readbackInterval,
    stopCheckMode: profileConfig.stopCheckMode,
    readbackMode,
    disableMultiTokenDecode: profileConfig.disableMultiTokenDecode === true,
    speculationMode: profileConfig.speculationMode ?? null,
    runtimeBaseConfig,
  };
}

function resolveRuntimeReadbackMode(runtimeConfig) {
  const mode = runtimeConfig?.inference?.session?.decodeLoop?.readbackMode
    ?? runtimeConfig?.inference?.batching?.readbackMode
    ?? null;
  return typeof mode === 'string' && mode.length > 0 ? mode : null;
}

function resolveDopplerBenchmarkKvCachePlan({
  prefillTokenTarget,
  maxTokens,
  promptContract = null,
}) {
  const promptIsShared = promptContract?.enginesReceiveRenderedPrompt === true;
  const prefill = Number.isFinite(prefillTokenTarget) ? Math.trunc(prefillTokenTarget) : null;
  const decode = Number.isFinite(maxTokens) ? Math.trunc(maxTokens) : null;
  const basePlan = {
    schemaVersion: 1,
    enabled: false,
    source: 'uncapped',
    maxSeqLen: null,
    promptTokenTarget: prefill,
    decodeTokenTarget: decode,
    tokenizerDeltaMargin: MAX_TOLERATED_TOKENIZER_DELTA,
    runtimePaths: [
      'inference.kvcache.maxSeqLen',
      'inference.session.kvcache.maxSeqLen',
    ],
    reason: null,
  };

  if (prefill == null || prefill <= 0) {
    return {
      ...basePlan,
      reason: 'prefill-token-target-unavailable',
    };
  }
  if (decode == null || decode <= 0) {
    return {
      ...basePlan,
      reason: 'decode-token-target-unavailable',
    };
  }
  if (promptIsShared !== true) {
    return {
      ...basePlan,
      reason: 'prompt-rendering-not-shared',
    };
  }

  return {
    ...basePlan,
    enabled: true,
    source: 'declared-workload-token-budget',
    maxSeqLen: prefill + decode + MAX_TOLERATED_TOKENIZER_DELTA,
    reason: 'tokenizer-accurate synthetic prompt budget plus decode target and tokenizer tolerance',
  };
}

const PROTECTED_RUNTIME_CONFIG_PREFIXES = Object.freeze([
  {
    prefix: 'shared.benchmark',
    reason: 'shared benchmark fairness inputs are owned by compare-engines flags/workloads',
  },
  {
    prefix: 'shared.harness',
    reason: 'harness command contract is owned by compare-engines',
  },
  {
    prefix: 'shared.tooling',
    reason: 'tooling intent/command contract is owned by compare-engines',
  },
  {
    prefix: 'inference.prompt',
    reason: 'prompt parity is a shared fairness axis; use --prompt or --workload',
  },
  {
    prefix: 'inference.chatTemplate',
    reason: 'chat-template parity is a shared fairness axis; use --use-chat-template',
  },
  {
    prefix: 'inference.sampling',
    reason: 'sampling parity is a shared fairness axis; use --temperature/--top-k/--top-p/--seed',
  },
  {
    prefix: 'inference.batching.maxTokens',
    reason: 'decode token budget is a shared fairness axis; use --max-tokens or --workload',
  },
  {
    prefix: 'inference.kvcache.maxSeqLen',
    reason: 'Doppler KV cache capacity is compare-managed from the declared workload token budget',
  },
  {
    prefix: 'inference.session.kvcache.maxSeqLen',
    reason: 'Doppler KV cache capacity is compare-managed from the declared workload token budget',
  },
  {
    prefix: 'inference.generation.disableMultiTokenDecode',
    reason: 'decode token grouping is compare-managed for apples-to-apples decode semantics',
  },
  {
    prefix: 'inference.batching.stopCheckMode',
    reason: 'decode stop-check cadence is compare-managed for consistent semantics',
  },
  {
    prefix: 'inference.batching.batchSize',
    reason: 'decode cadence is compare-managed; use --decode-profile or --doppler-batch-size',
  },
  {
    prefix: 'inference.batching.readbackInterval',
    reason: 'decode cadence is compare-managed; use --decode-profile or --doppler-readback-interval',
  },
  {
    prefix: 'inference.batching.readbackMode',
    reason: 'decode readback mode is compare-managed by the selected runtime profile',
  },
  {
    prefix: 'inference.session.decodeLoop.stopCheckMode',
    reason: 'decode stop-check cadence is compare-managed for consistent semantics',
  },
  {
    prefix: 'inference.session.decodeLoop.batchSize',
    reason: 'decode cadence is compare-managed; use --decode-profile or --doppler-batch-size',
  },
  {
    prefix: 'inference.session.decodeLoop.readbackInterval',
    reason: 'decode cadence is compare-managed; use --decode-profile or --doppler-readback-interval',
  },
  {
    prefix: 'inference.session.decodeLoop.readbackMode',
    reason: 'decode readback mode is compare-managed by the selected runtime profile',
  },
  {
    prefix: 'inference.session.decodeLoop.disableCommandBatching',
    reason: 'decode batching contract is compare-managed for fair cadence measurement',
  },
  {
    prefix: 'inference.session.speculation',
    reason: 'same-model speculation is compare-managed for apples-to-apples decode semantics',
  },
  {
    prefix: 'inference.kernelPath',
    reason: 'kernel-path selection is manifest execution-v1 owned',
  },
]);

function usage() {
  return [
    'Usage: node tools/compare-engines.js [options]',
    '',
    'Common options:',
    '  --help, -h                    Show this help text',
    '  --model-id <id>               Doppler model ID (default: first profile in compare-engines.config.json)',
    '  --model-url <url>             Doppler model base URL override (default: compare profile source resolution)',
    '  --tjs-model <id>              Transformers.js model ID (default: profile mapping in compare-engines.config.json)',
    '  --tjs-version <3|4>           Transformers.js version (default: 4)',
    '  --tjs-dtype <fp16|q4|q4f16>   Transformers.js dtype (default: catalog benchmark dtype or fp16)',
    '  --workload <id>               Shared workload id from benchmarks/vendors/workloads.json',
    '  --prompt <text>               Prompt used for both engines (overrides workload prompt)',
    '  --prefill-tokens <n>          Model-input prompt token target when --prompt is omitted',
    '  --mode <compute|cold|warm|all> Run mode',
    '  --max-tokens <n>              Max new tokens',
    '  --temperature <n>             Sampling temperature',
    '  --top-k <n>                   Sampling top-k',
    '  --top-p <n>                   Sampling top-p',
    '  --warmup <n>                  Warmup runs per engine',
    '  --runs <n>                    Timed runs per engine',
    '  --decode-profile <profile>     parity|throughput|custom (default: parity)',
    '  --doppler-surface <surface>     auto|node|browser|bun (default: from compare profile, fallback auto)',
    '  --compare-config <path>        Compare model profile config path',
    '  --compare-metric-contract <path> Compare metric contract path',
    '  --seed <n>                    Deterministic seed',
    '  --doppler-kernel-path <id>     Removed; use manifest execution-v1 graphs',
    '  --doppler-batch-size <n>       Doppler decode batch size',
    '  --doppler-readback-interval <n> Doppler decode readback interval',
    '  --doppler-stop-check-mode <batch|per-token> Doppler decode stop-check cadence',
    '  --doppler-no-opfs-cache        Disable Doppler OPFS cache for browser runs',
    '  --use-chat-template <on|off>   Enable/disable chat template handling',
    '  --tjs-local-model-path <path>   Path override for local TJS model files',
    '  --load-mode <opfs|http|memory>  Asset-load mode override (memory requires --doppler-format safetensors)',
  '  --doppler-browser-user-data <path> Doppler Chromium profile dir',
  '  --doppler-browser-port <n>      Doppler browser relay static port (0 = random)',
  '  --doppler-browser-channel <id>   Doppler browser channel override (default: benchmark policy)',
  '  --browser-base-url <url>         Base URL for both benchmark runners (skips local server startup)',
  '  --browser-executable <path>      Browser executable for both benchmark runners',
  '  --runtime-config-json <json>      Engine-only JSON overlay merged into Doppler runtime config (fairness axes protected)',
  '  --doppler-bottleneck-profile <on|off> Run a Doppler debug-profiler pass as a diagnostic artifact',
  `  --resource-telemetry <on|off> Capture process-tree RAM/CPU, system RAM, and optional ROCm GPU telemetry for child benchmark runs (default: off; interval default: ${DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS}ms)`,
  '  --resource-telemetry-samples Include raw telemetry samples in saved JSON (default: summary only)',
  '  --timestamp <iso|ms>               Report timestamp override (ISO-8601 or epoch milliseconds)',
    '  --tjs-profile-ops <on|off>       TJS ORT op profiling',
  '  --timeout-ms <ms>                 Shared benchmark timeout',
  '  --doppler-timeout-ms <ms>         Doppler-only benchmark timeout',
  '  --tjs-timeout-ms <ms>             TJS-only benchmark timeout',
    '  --allow-non-comparable-lane      Allow capability-only or stale-artifact diagnostic lanes',
    '  --tjs-server-port <n>             TJS server port (0 = random)',
    '  --tjs-browser-console             Stream browser console on TJS failures',
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
  if (DOPPLER_LOAD_MODES.includes(normalized)) {
    return normalized;
  }
  throw new Error(`${label} must be one of: ${DOPPLER_LOAD_MODES.join(', ')}`);
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

  for (const metricId of REQUIRED_COMPARE_METRIC_IDS) {
    const row = rowsById.get(metricId);
    if (!row) {
      throw new Error(`compare metric contract is missing required compare metric "${metricId}"`);
    }
    if (row.required !== true) {
      throw new Error(`compare metric contract must mark "${metricId}" as required for apples-to-apples mode`);
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

function toRepoRelativeSourcePath(sourcePath) {
  if (typeof sourcePath !== 'string' || sourcePath.trim() === '') return sourcePath;
  if (/^[a-z]+:\/\//i.test(sourcePath.trim())) {
    return sourcePath.trim();
  }
  const absolutePath = path.isAbsolute(sourcePath)
    ? sourcePath
    : path.resolve(process.cwd(), sourcePath);
  const relativePath = path.relative(DOPPLER_ROOT, absolutePath);
  if (relativePath && !relativePath.startsWith('..') && !path.isAbsolute(relativePath)) {
    return relativePath.split(path.sep).join('/');
  }
  return sourcePath;
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

function normalizeCompareLoadModeDefaults(value, label = 'compare-engines.config.json defaults') {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const warm = parseLoadMode(value.warmLoadMode, `${label}.warmLoadMode`, null);
  const cold = parseLoadMode(value.coldLoadMode, `${label}.coldLoadMode`, null);
  if (warm == null) {
    throw new Error(`${label}.warmLoadMode is required`);
  }
  if (cold == null) {
    throw new Error(`${label}.coldLoadMode is required`);
  }
  return {
    warm,
    cold,
  };
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
  const defaults = normalizeCompareLoadModeDefaults(
    payload.defaults,
    `compare profile config at ${resolvedPath} defaults`
  );

  const normalizeDopplerRuntimeProfileByDecodeProfile = (value, label) => {
    if (value == null) {
      return Object.freeze({});
    }
    if (typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(`${label} must be an object or null`);
    }
    const normalized = {};
    for (const [decodeProfile, runtimeProfileId] of Object.entries(value)) {
      const normalizedDecodeProfile = String(decodeProfile ?? '').trim().toLowerCase();
      if (!COMPARE_PROFILE_RUNTIME_DECODE_PROFILES.includes(normalizedDecodeProfile)) {
        throw new Error(
          `${label} key "${decodeProfile}" must be one of: ${COMPARE_PROFILE_RUNTIME_DECODE_PROFILES.join(', ')}`
        );
      }
      if (
        runtimeProfileId != null
        && (typeof runtimeProfileId !== 'string' || runtimeProfileId.trim() === '')
      ) {
        throw new Error(`${label}.${decodeProfile} must be a non-empty string or null`);
      }
      if (runtimeProfileId != null) {
        normalized[normalizedDecodeProfile] = runtimeProfileId.trim();
      }
    }
    return Object.freeze(normalized);
  };

  const normalizeDopplerRuntimeProfileByDecodeProfileByPlatform = (value, label) => {
    if (value == null) {
      return Object.freeze({});
    }
    if (typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(`${label} must be an object or null`);
    }
    const normalized = {};
    for (const [platform, profileMap] of Object.entries(value)) {
      const normalizedPlatform = String(platform ?? '').trim();
      if (!normalizedPlatform) {
        throw new Error(`${label} contains an invalid platform key`);
      }
      normalized[normalizedPlatform] = normalizeDopplerRuntimeProfileByDecodeProfile(
        profileMap,
        `${label}.${normalizedPlatform}`
      );
    }
    return Object.freeze(normalized);
  };

  const normalizeOutputParityPolicy = (value, label) => {
    if (value == null) {
      return null;
    }
    if (typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(`${label} must be an object or null`);
    }
    if (value.schemaVersion != null && value.schemaVersion !== 1) {
      throw new Error(`${label}.schemaVersion must be 1 when provided`);
    }
    if (typeof value.requireMatch !== 'boolean') {
      throw new Error(`${label}.requireMatch must be a boolean`);
    }
    const matchMode = String(value.matchMode ?? '').trim();
    if (!OUTPUT_PARITY_MATCH_MODES.includes(matchMode)) {
      throw new Error(
        `${label}.matchMode must be one of: ${OUTPUT_PARITY_MATCH_MODES.join(', ')}`
      );
    }
    if (typeof value.reason !== 'string' || value.reason.trim() === '') {
      throw new Error(`${label}.reason must be a non-empty string`);
    }
    return Object.freeze({
      schemaVersion: 1,
      requireMatch: value.requireMatch,
      matchMode,
      reason: value.reason.trim(),
    });
  };

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
    if (row.modelBaseDir != null && (typeof row.modelBaseDir !== 'string' || row.modelBaseDir.trim() === '')) {
      throw new Error(`compare-engines.config.json modelBaseDir for ${row.dopplerModelId} must be a non-empty string or null`);
    }
    row.defaultDopplerSource = normalizeDopplerModelSource(
      row.defaultDopplerSource,
      `compare-engines.config.json defaultDopplerSource for ${row.dopplerModelId}`
    );
    row.compareLane = normalizeCompareLane(
      row.compareLane,
      `compare-engines.config.json compareLane for ${row.dopplerModelId}`
    );
    if (
      row.compareLaneReason != null
      && (typeof row.compareLaneReason !== 'string' || row.compareLaneReason.trim() === '')
    ) {
      throw new Error(
        `compare-engines.config.json compareLaneReason for ${row.dopplerModelId} must be a non-empty string or null`
      );
    }
    if (row.defaultDopplerSurface != null) {
      const normalizedSurface = String(row.defaultDopplerSurface).trim().toLowerCase();
      if (!DOPPLER_SURFACES.includes(normalizedSurface)) {
        throw new Error(
          `compare-engines.config.json defaultDopplerSurface for ${row.dopplerModelId} must be one of: ${DOPPLER_SURFACES.join(', ')}`
        );
      }
      row.defaultDopplerSurface = normalizedSurface;
    }
    if (row.defaultUseChatTemplate != null && typeof row.defaultUseChatTemplate !== 'boolean') {
      throw new Error(
        `compare-engines.config.json defaultUseChatTemplate for ${row.dopplerModelId} must be a boolean or null`
      );
    }
    row.defaultLoadMode = parseLoadMode(
      row.defaultLoadMode,
      `compare-engines.config.json defaultLoadMode for ${row.dopplerModelId}`,
      null
    );
    if (
      row.defaultLoadModeReason != null
      && (typeof row.defaultLoadModeReason !== 'string' || row.defaultLoadModeReason.trim() === '')
    ) {
      throw new Error(
        `compare-engines.config.json defaultLoadModeReason for ${row.dopplerModelId} must be a non-empty string or null`
      );
    }
    if (row.defaultLoadMode == null && row.defaultLoadModeReason != null) {
      throw new Error(
        `compare-engines.config.json defaultLoadModeReason for ${row.dopplerModelId} requires defaultLoadMode`
      );
    }
    if (row.defaultLoadMode != null && row.defaultLoadModeReason == null) {
      throw new Error(
        `compare-engines.config.json defaultLoadMode for ${row.dopplerModelId} requires defaultLoadModeReason`
      );
    }
    if (row.defaultLoadMode != null && row.defaultLoadModeReason != null) {
      row.defaultLoadModeReason = row.defaultLoadModeReason.trim();
    }
    row.dopplerRuntimeProfileByDecodeProfile = normalizeDopplerRuntimeProfileByDecodeProfile(
      row.dopplerRuntimeProfileByDecodeProfile ?? null,
      `compare-engines.config.json dopplerRuntimeProfileByDecodeProfile for ${row.dopplerModelId}`
    );
    row.dopplerRuntimeProfileByDecodeProfileByPlatform = normalizeDopplerRuntimeProfileByDecodeProfileByPlatform(
      row.dopplerRuntimeProfileByDecodeProfileByPlatform ?? null,
      `compare-engines.config.json dopplerRuntimeProfileByDecodeProfileByPlatform for ${row.dopplerModelId}`
    );
    row.outputParityPolicy = normalizeOutputParityPolicy(
      row.outputParityPolicy ?? null,
      `compare-engines.config.json outputParityPolicy for ${row.dopplerModelId}`
    );
    if (row.defaultDopplerSource === 'quickstart-registry' && row.modelBaseDir != null) {
      throw new Error(
        `compare-engines.config.json modelBaseDir for ${row.dopplerModelId} must be null when defaultDopplerSource is "quickstart-registry"`
      );
    }
  }

  const modelProfileById = new Map(rows.map((row) => [String(row.dopplerModelId).trim().toLowerCase(), row]));
  return {
    source: resolvedPath,
    schemaVersion: payload.schemaVersion,
    updated: payload.updated || null,
    sourceSha256,
    defaults,
    modelProfileById,
    profiles: rows,
  };
}

function resolveCompareProfile(compareConfig, modelId) {
  const normalizedModelId = String(modelId ?? '').trim().toLowerCase();
  const profile = compareConfig?.modelProfileById?.get(normalizedModelId) || null;
  return {
    source: profile == null ? DEFAULT_COMPARE_PROFILE_SOURCE : 'config',
    defaultDopplerSource: profile?.defaultDopplerSource || DEFAULT_DOPPLER_MODEL_SOURCE,
    modelBaseDir: profile?.modelBaseDir ?? null,
    defaultTjsModelId: profile?.defaultTjsModelId || null,
    defaultDopplerSurface: profile?.defaultDopplerSurface || 'auto',
    defaultUseChatTemplate: profile?.defaultUseChatTemplate ?? null,
    defaultDopplerFormat: profile?.defaultDopplerFormat || DEFAULT_DOPPLER_FORMAT,
    defaultTjsFormat: profile?.defaultTjsFormat || DEFAULT_TJS_FORMAT,
    defaultLoadMode: profile?.defaultLoadMode || null,
    defaultLoadModeReason: profile?.defaultLoadModeReason || null,
    safetensorsSourceId: profile?.safetensorsSourceId || null,
    dopplerRuntimeProfileByDecodeProfile: profile?.dopplerRuntimeProfileByDecodeProfile || Object.freeze({}),
    dopplerRuntimeProfileByDecodeProfileByPlatform: profile?.dopplerRuntimeProfileByDecodeProfileByPlatform || Object.freeze({}),
    compareLane: profile?.compareLane || DEFAULT_COMPARE_LANE,
    compareLaneReason: profile?.compareLaneReason || null,
    outputParityPolicy: profile?.outputParityPolicy || null,
  };
}

function resolveRuntimeProfilePath(profileId) {
  const normalized = String(profileId ?? '').trim().replace(/\.json$/u, '');
  if (!normalized) {
    throw new Error('runtime profile id is required');
  }
  return path.join(DOPPLER_ROOT, 'src', 'config', 'runtime', `${normalized}.json`);
}

async function loadResolvedRuntimeProfileBundle(profileId, stack = []) {
  const resolvedPath = path.resolve(resolveRuntimeProfilePath(profileId));
  if (stack.includes(resolvedPath)) {
    throw new Error(`Runtime profile extends cycle: ${[...stack, resolvedPath].join(' -> ')}`);
  }
  const raw = await fs.readFile(resolvedPath, 'utf-8');
  const payload = JSON.parse(raw);
  const runtime = payload?.runtime;
  if (!runtime || typeof runtime !== 'object' || Array.isArray(runtime)) {
    throw new Error(`Runtime profile "${profileId}" is missing runtime.`);
  }
  const extendsRefs = Array.isArray(payload.extends)
    ? payload.extends
    : (typeof payload.extends === 'string' ? [payload.extends] : []);
  let mergedRuntime = null;
  for (const ref of extendsRefs) {
    const baseBundle = await loadResolvedRuntimeProfileBundle(ref, [...stack, resolvedPath]);
    mergedRuntime = mergedRuntime == null
      ? baseBundle.resolvedRuntime
      : mergeRuntimeValues(mergedRuntime, baseBundle.resolvedRuntime);
  }
  const resolvedRuntime = mergedRuntime == null
    ? runtime
    : mergeRuntimeValues(mergedRuntime, runtime);
  return {
    profileId,
    source: resolvedPath,
    sourceSha256: hashText(raw),
    resolvedRuntime,
    resolvedRuntimeSha256: hashJsonPayload(resolvedRuntime),
  };
}

function resolveCompareRuntimeProfileMap(compareProfile, platform = process.platform) {
  const baseProfiles = compareProfile?.dopplerRuntimeProfileByDecodeProfile || Object.freeze({});
  const platformProfiles = compareProfile?.dopplerRuntimeProfileByDecodeProfileByPlatform?.[platform]
    || Object.freeze({});
  return Object.freeze({
    ...baseProfiles,
    ...platformProfiles,
  });
}

async function loadCompareProfileRuntimeBundles(compareProfile, platform = process.platform) {
  const entries = Object.entries(resolveCompareRuntimeProfileMap(compareProfile, platform));
  if (entries.length === 0) {
    return new Map();
  }
  const bundles = await Promise.all(entries.map(async ([decodeProfile, profileId]) => [
    decodeProfile,
    await loadResolvedRuntimeProfileBundle(profileId),
  ]));
  return new Map(bundles);
}

async function loadModelCatalogBundle(catalogPath = MODEL_CATALOG_PATH) {
  const raw = await fs.readFile(catalogPath, 'utf-8');
  const payload = JSON.parse(raw);
  const rows = Array.isArray(payload?.models) ? payload.models : [];
  const byModelId = new Map();
  const transformersjsBenchmarkByRepoId = new Map();
  for (const row of rows) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) {
      continue;
    }
    const modelId = typeof row.modelId === 'string' ? row.modelId.trim() : '';
    if (!modelId) {
      continue;
    }
    byModelId.set(modelId.toLowerCase(), row);
    const vendorBenchmark = row.vendorBenchmark?.transformersjs;
    const repoId = typeof vendorBenchmark?.repoId === 'string' ? vendorBenchmark.repoId.trim() : '';
    const dtype = typeof vendorBenchmark?.dtype === 'string' ? vendorBenchmark.dtype.trim().toLowerCase() : '';
    if (!repoId || !dtype) {
      continue;
    }
    if (!TJS_DTYPES.includes(dtype)) {
      throw new Error(
        `models/catalog.json vendorBenchmark.transformersjs.dtype for ${modelId} `
        + `must be one of: ${TJS_DTYPES.join(', ')}`
      );
    }
    const existing = transformersjsBenchmarkByRepoId.get(repoId.toLowerCase()) || null;
    if (existing && existing.dtype !== dtype) {
      throw new Error(
        `models/catalog.json has conflicting vendorBenchmark.transformersjs dtype values for `
        + `"${repoId}": ${existing.dtype} vs ${dtype}`
      );
    }
    if (!existing) {
      transformersjsBenchmarkByRepoId.set(repoId.toLowerCase(), {
        repoId,
        dtype,
        sourceModelId: modelId,
      });
    }
  }
  return {
    source: catalogPath,
    sourceSha256: hashText(raw),
    updatedAt: typeof payload?.updatedAt === 'string' ? payload.updatedAt : null,
    byModelId,
    transformersjsBenchmarkByRepoId,
  };
}

function resolveCatalogTransformersjsBenchmarkTarget(catalogBundle, dopplerModelId, repoId = null) {
  const normalizedModelId = typeof dopplerModelId === 'string' ? dopplerModelId.trim().toLowerCase() : '';
  const normalizedRepoId = typeof repoId === 'string' ? repoId.trim().toLowerCase() : '';
  if (normalizedModelId) {
    const entry = catalogBundle?.byModelId?.get(normalizedModelId) || null;
    const vendorBenchmark = entry?.vendorBenchmark?.transformersjs || null;
    const benchmarkRepoId = typeof vendorBenchmark?.repoId === 'string' ? vendorBenchmark.repoId.trim() : '';
    const benchmarkDtype = typeof vendorBenchmark?.dtype === 'string' ? vendorBenchmark.dtype.trim().toLowerCase() : '';
    if (benchmarkRepoId && benchmarkDtype && TJS_DTYPES.includes(benchmarkDtype)) {
      return {
        repoId: benchmarkRepoId,
        dtype: benchmarkDtype,
        source: 'catalog-model',
        sourceModelId: entry.modelId,
      };
    }
  }
  if (normalizedRepoId) {
    const benchmark = catalogBundle?.transformersjsBenchmarkByRepoId?.get(normalizedRepoId) || null;
    if (benchmark) {
      return {
        repoId: benchmark.repoId,
        dtype: benchmark.dtype,
        source: 'catalog-repo',
        sourceModelId: benchmark.sourceModelId,
      };
    }
  }
  return null;
}

async function loadQuickstartRegistry() {
  const raw = await fs.readFile(QUICKSTART_REGISTRY_PATH, 'utf-8');
  const payload = JSON.parse(raw);
  const rows = Array.isArray(payload?.models) ? payload.models : [];
  const modelById = new Map();
  for (const row of rows) {
    const modelId = typeof row?.modelId === 'string' ? row.modelId.trim() : '';
    if (!modelId) {
      throw new Error('doppler-registry.json quickstart entries must define modelId');
    }
    if (
      typeof row?.sourceCheckpointId !== 'string'
      || row.sourceCheckpointId.trim().length === 0
      || typeof row?.weightPackId !== 'string'
      || row.weightPackId.trim().length === 0
      || typeof row?.manifestVariantId !== 'string'
      || row.manifestVariantId.trim().length === 0
    ) {
      throw new Error(`doppler-registry.json quickstart entry "${modelId}" must define sourceCheckpointId, weightPackId, and manifestVariantId`);
    }
    if (row?.artifactCompleteness !== 'complete' || row?.runtimePromotionState !== 'manifest-owned' || row?.weightsRefAllowed !== false) {
      throw new Error(`doppler-registry.json quickstart entry "${modelId}" must be complete and manifest-owned`);
    }
    if (modelById.has(modelId.toLowerCase())) {
      throw new Error(`doppler-registry.json has duplicate quickstart modelId "${modelId}"`);
    }
    modelById.set(modelId.toLowerCase(), row);
  }
  return {
    source: QUICKSTART_REGISTRY_PATH,
    sourceSha256: hashText(raw),
    modelById,
  };
}

async function resolveExplicitModelManifestSource(modelUrl) {
  const normalizedUrl = String(modelUrl ?? '').trim().replace(/\/+$/, '');
  if (!normalizedUrl) {
    throw new Error('Explicit Doppler model URL must be a non-empty string.');
  }
  const explicitManifestUrl = normalizedUrl.endsWith('/manifest.json')
    ? normalizedUrl
    : `${normalizedUrl}/manifest.json`;
  if (/^https?:\/\//i.test(normalizedUrl)) {
    return {
      manifestSourceType: 'remote',
      manifestSource: explicitManifestUrl,
    };
  }
  if (normalizedUrl.startsWith('/models/')) {
    return {
      manifestSourceType: 'local',
      manifestSource: normalizedUrl.endsWith('/manifest.json')
        ? path.join(DOPPLER_ROOT, normalizedUrl.replace(/^\/+/, ''))
        : path.join(DOPPLER_ROOT, normalizedUrl.replace(/^\/+/, ''), 'manifest.json'),
    };
  }
  if (normalizedUrl.startsWith('file://')) {
    const localPath = fileURLToPath(normalizedUrl.endsWith('/manifest.json') ? normalizedUrl : `${normalizedUrl}/manifest.json`);
    return {
      manifestSourceType: 'local',
      manifestSource: localPath,
    };
  }
  const resolvedPath = path.isAbsolute(normalizedUrl)
    ? normalizedUrl
    : path.resolve(process.cwd(), normalizedUrl);
  const manifestPath = resolvedPath.endsWith(`${path.sep}manifest.json`)
    ? resolvedPath
    : path.join(resolvedPath, 'manifest.json');
  return {
    manifestSourceType: 'local',
    manifestSource: manifestPath,
  };
}

async function resolveDopplerModelSource(compareProfile, dopplerModelId, explicitModelUrl) {
  if (explicitModelUrl != null && String(explicitModelUrl).trim() !== '') {
    const normalizedModelUrl = String(explicitModelUrl).trim().replace(/\/+$/, '');
    const explicitManifest = await resolveExplicitModelManifestSource(normalizedModelUrl);
    return {
      source: 'explicit',
      modelUrl: normalizedModelUrl,
      locator: normalizedModelUrl,
      registrySource: null,
      modelBaseDir: null,
      ...explicitManifest,
    };
  }

  if (compareProfile.defaultDopplerSource === 'quickstart-registry') {
    const quickstartRegistry = await loadQuickstartRegistry();
    const registryEntry = quickstartRegistry.modelById.get(String(dopplerModelId).trim().toLowerCase()) || null;
    if (!registryEntry) {
      throw new Error(
        `compare profile for "${dopplerModelId}" requires quickstart-registry resolution, `
        + `but no matching quickstart entry exists in ${toRepoRelativeSourcePath(QUICKSTART_REGISTRY_PATH)}.`
      );
    }
    const modelUrl = buildEntryRemoteBaseUrl(registryEntry);
    if (!modelUrl) {
      throw new Error(`Quickstart registry entry for "${dopplerModelId}" is missing complete hosted HF coordinates.`);
    }
    return {
      source: 'quickstart-registry',
      modelUrl,
      locator: registryEntry.modelId,
      registrySource: {
        source: toRepoRelativeSourcePath(quickstartRegistry.source),
        sourceSha256: quickstartRegistry.sourceSha256,
        sourceCheckpointId: registryEntry.sourceCheckpointId,
        weightPackId: registryEntry.weightPackId,
        manifestVariantId: registryEntry.manifestVariantId,
      },
      modelBaseDir: null,
      manifestSourceType: 'remote',
      manifestSource: `${modelUrl}/manifest.json`,
    };
  }

  const modelBaseDir = compareProfile.modelBaseDir || 'local';
  const localModelDir = path.join(DOPPLER_ROOT, 'models', modelBaseDir, dopplerModelId);
  return {
    source: 'local',
    modelUrl: pathToFileURL(localModelDir).href,
    locator: dopplerModelId,
    registrySource: null,
    modelBaseDir,
    manifestSourceType: 'local',
    manifestSource: path.join(localModelDir, 'manifest.json'),
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

function finiteStatNumber(value) {
  if (Number.isFinite(value)) {
    return value;
  }
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  if (Number.isFinite(value.samples) && value.samples <= 0) {
    return null;
  }
  for (const key of ['mean', 'median', 'p50', 'value']) {
    if (Number.isFinite(value[key])) {
      return value[key];
    }
  }
  return null;
}

function firstFiniteStatNumber(objectValue, paths) {
  for (const dottedPath of paths || []) {
    const candidate = finiteStatNumber(getByPath(objectValue, dottedPath));
    if (Number.isFinite(candidate)) {
      return candidate;
    }
  }
  return null;
}

function firstArrayValue(objectValue, paths) {
  for (const dottedPath of paths || []) {
    const candidate = getByPath(objectValue, dottedPath);
    if (Array.isArray(candidate)) {
      return candidate;
    }
  }
  return null;
}

function roundBottleneckNumber(value) {
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.round(value * 1000) / 1000;
}

function normalizeCommandRecordTopOps(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized = [];
  for (const entry of value) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      continue;
    }
    const label = typeof entry.label === 'string' && entry.label.length > 0
      ? entry.label
      : null;
    const count = finiteStatNumber(entry.count);
    if (!label || !Number.isFinite(count) || count <= 0) {
      continue;
    }
    normalized.push({
      label,
      count: roundBottleneckNumber(count),
      shareOfOps: roundBottleneckNumber(finiteStatNumber(entry.shareOfOps)),
    });
  }
  return normalized;
}

function shareOfDecode(value, decodeMs) {
  if (!Number.isFinite(value) || !Number.isFinite(decodeMs) || decodeMs <= 0) {
    return null;
  }
  return roundBottleneckNumber(value / decodeMs);
}

function clipTail(text, maxChars = 3000) {
  if (text == null) return '';
  const str = redactSecrets(String(text));
  if (str.length <= maxChars) return str;
  return str.slice(str.length - maxChars);
}

function redactSecrets(text) {
  if (text == null) return '';
  return String(text)
    .replace(/hfToken=hf_[A-Za-z0-9._-]+/g, 'hfToken=<redacted>')
    .replace(/\bhf_[A-Za-z0-9._-]+\b/g, '<HF_TOKEN_REDACTED>')
    .replace(/(authorization:\s*bearer\s+)[^\s"'`]+/gi, '$1<redacted>');
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

  // Scan forward from each `\n{` candidate and try to parse.
  // The first match that yields a valid top-level object wins.
  // This avoids `lastIndexOf` landing inside a nested `{`.
  let searchFrom = 0;
  while (searchFrom < normalized.length) {
    const objectStart = normalized.indexOf('\n{', searchFrom);
    if (objectStart < 0) break;
    const parsed = asObject(normalized.slice(objectStart + 1).trim());
    if (parsed !== null) return parsed;
    searchFrom = objectStart + 2;
  }

  // Also try a leading `{` at position 0 (no preceding newline).
  if (normalized.trimStart().startsWith('{')) {
    const parsed = asObject(normalized.trimStart());
    if (parsed !== null) return parsed;
  }

  throw new Error(`Could not parse strict JSON payload from ${label}. Full output tail:\n${clipTail(normalized, 2000)}`);
}

function applyDopplerBenchCaptureRunConfig(runConfig, captureDir) {
  const normalizedDir = typeof captureDir === 'string' ? captureDir.trim() : '';
  if (!normalizedDir) {
    return runConfig;
  }
  return {
    ...runConfig,
    bench: {
      ...(runConfig && typeof runConfig.bench === 'object' && !Array.isArray(runConfig.bench)
        ? runConfig.bench
        : {}),
      save: true,
      saveDir: normalizedDir,
    },
  };
}

async function createDopplerBenchCaptureDir() {
  return fs.mkdtemp(path.join(os.tmpdir(), DOPPLER_BENCH_CAPTURE_DIR_PREFIX));
}

async function readDopplerBenchCaptureResult(captureDir, label) {
  const latestPath = path.join(captureDir, 'latest.json');
  let raw;
  try {
    raw = await fs.readFile(latestPath, 'utf8');
  } catch (error) {
    throw new Error(`Could not read saved Doppler bench result for ${label} at ${latestPath}: ${error.message}`);
  }
  const parsed = parseJsonBlock(raw, `saved Doppler bench result for ${label}`);
  if (isComparePlainObject(parsed?.result)) {
    return parsed;
  }
  return {
    ok: true,
    request: isComparePlainObject(parsed?.request) ? parsed.request : null,
    result: parsed,
  };
}

async function removeDopplerBenchCaptureDir(captureDir) {
  if (!captureDir) return;
  await fs.rm(captureDir, { recursive: true, force: true });
}

function appendBoundedTail(previous, chunk, maxChars = CHILD_PROCESS_TAIL_CHARS) {
  const text = `${previous || ''}${chunk == null ? '' : String(chunk)}`;
  if (text.length <= maxChars) return text;
  return text.slice(text.length - maxChars);
}

async function execFileCaptureStdout(command, args, options) {
  return new Promise((resolve, reject) => {
    const telemetry = createResourceTelemetrySampler(options.resourceTelemetry);
    const child = spawn(command, args, {
      cwd: options.cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    telemetry.start(child.pid);
    const stdoutChunks = [];
    const stderrChunks = [];
    let stdoutBytes = 0;
    let stderrTail = '';
    let settled = false;
    let timedOut = false;
    const timeoutId = setTimeout(() => {
      timedOut = true;
      child.kill('SIGTERM');
    }, options.timeout);

    child.stdout.on('data', (chunk) => {
      stdoutBytes += chunk.length;
      if (Number.isFinite(options.maxBuffer) && stdoutBytes > options.maxBuffer) {
        child.kill('SIGTERM');
        return;
      }
      stdoutChunks.push(chunk);
    });

    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => {
      stderrChunks.push(Buffer.from(chunk));
      stderrTail = appendBoundedTail(stderrTail, chunk);
    });

    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutId);
      telemetry.stop();
      reject(error);
    });

    child.on('close', (code, signal) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutId);
      const resourceTelemetry = telemetry.stop();
      if (timedOut) {
        reject(new Error(`Child process timed out after ${options.timeout}ms. Stderr tail:\n${clipTail(stderrTail, 2000)}`));
        return;
      }
      if (Number.isFinite(options.maxBuffer) && stdoutBytes > options.maxBuffer) {
        reject(new Error(`Child process stdout exceeded ${options.maxBuffer} bytes. Stderr tail:\n${clipTail(stderrTail, 2000)}`));
        return;
      }
      if (code !== 0) {
        reject(new Error(`Child process exited with code ${code ?? 'null'} signal ${signal ?? 'null'}. Stderr tail:\n${clipTail(stderrTail, 2000)}`));
        return;
      }
      resolve({
        stdout: Buffer.concat(stdoutChunks).toString('utf8'),
        stderr: Buffer.concat(stderrChunks).toString('utf8'),
        resourceTelemetry,
      });
    });
  });
}

async function execFileForSavedDopplerBench(command, args, options) {
  return new Promise((resolve, reject) => {
    const telemetry = createResourceTelemetrySampler(options.resourceTelemetry);
    const child = spawn(command, args, {
      cwd: options.cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    telemetry.start(child.pid);
    let stdoutTail = '';
    let stderrTail = '';
    let settled = false;
    let timedOut = false;
    const timeoutId = setTimeout(() => {
      timedOut = true;
      child.kill('SIGTERM');
    }, options.timeout);

    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      stdoutTail = appendBoundedTail(stdoutTail, chunk);
    });

    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => {
      stderrTail = appendBoundedTail(stderrTail, chunk);
    });

    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutId);
      telemetry.stop();
      reject(error);
    });

    child.on('close', (code, signal) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutId);
      const resourceTelemetry = telemetry.stop();
      if (timedOut) {
        reject(new Error(
          `Doppler bench child timed out after ${options.timeout}ms. ` +
          `Stderr tail:\n${clipTail(stderrTail, 2000)}\nStdout tail:\n${clipTail(stdoutTail, 2000)}`
        ));
        return;
      }
      if (code !== 0) {
        reject(new Error(
          `Doppler bench child exited with code ${code ?? 'null'} signal ${signal ?? 'null'}. ` +
          `Stderr tail:\n${clipTail(stderrTail, 2000)}\nStdout tail:\n${clipTail(stdoutTail, 2000)}`
        ));
        return;
      }
      resolve({ stdout: stdoutTail, stderr: stderrTail, resourceTelemetry });
    });
  });
}

function hashText(input) {
  return crypto.createHash('sha256').update(input).digest('hex');
}

function renderComparePrompt({ modelId, prompt, useChatTemplate }) {
  const promptRaw = String(prompt ?? DEFAULT_PROMPT);
  if (useChatTemplate !== true) {
    return {
      schemaVersion: 1,
      promptRaw,
      promptRendered: promptRaw,
      promptRenderer: 'raw',
      chatTemplateSource: 'disabled',
      promptRenderedSha256: hashText(promptRaw),
      requestedUseChatTemplate: false,
      enginesReceiveRenderedPrompt: true,
    };
  }

  if (String(modelId || '').startsWith('gemma-4-')) {
    const promptRendered = `<bos><|turn>user\n${promptRaw}<turn|>\n<|turn>model\n`;
    return {
      schemaVersion: 1,
      promptRaw,
      promptRendered,
      promptRenderer: 'gemma4-compare-template',
      chatTemplateSource: 'compare-engines',
      promptRenderedSha256: hashText(promptRendered),
      requestedUseChatTemplate: true,
      enginesReceiveRenderedPrompt: true,
    };
  }

  return {
    schemaVersion: 1,
    promptRaw,
    promptRendered: promptRaw,
    promptRenderer: 'engine-chat-template',
    chatTemplateSource: 'engine',
    promptRenderedSha256: hashText(promptRaw),
    requestedUseChatTemplate: true,
    enginesReceiveRenderedPrompt: false,
  };
}

function resolveCompareOwnedPromptRenderer({ modelId, useChatTemplate }) {
  const probe = renderComparePrompt({ modelId, prompt: '', useChatTemplate });
  if (probe.requestedUseChatTemplate !== true || probe.enginesReceiveRenderedPrompt !== true) {
    return null;
  }
  return (prompt) => renderComparePrompt({ modelId, prompt, useChatTemplate }).promptRendered;
}

function buildComparabilityWarnings({ dopplerSurface }) {
  const warnings = [];
  if (dopplerSurface !== 'browser') {
    warnings.push(
      `Doppler surface is ${dopplerSurface}; Transformers.js runs in browser. `
      + 'Do not infer browser-vs-browser regression from this artifact.'
    );
  }
  return warnings;
}

function formatDopplerRuntimeCacheLabel(dopplerSurface, loadMode) {
  if (dopplerSurface === 'browser' || dopplerSurface === 'auto') {
    return `Doppler browser load mode ${loadMode}`;
  }
  if (dopplerSurface === 'bun') {
    return 'Doppler bun runtime cache';
  }
  return 'Doppler node runtime cache';
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

function matchProtectedRuntimeConfigPrefix(dottedPath) {
  for (const entry of PROTECTED_RUNTIME_CONFIG_PREFIXES) {
    if (dottedPath === entry.prefix || dottedPath.startsWith(`${entry.prefix}.`)) {
      return entry;
    }
  }
  return null;
}

function collectProtectedRuntimeConfigMutations(value, currentPath = '', out = []) {
  if (currentPath) {
    const match = matchProtectedRuntimeConfigPrefix(currentPath);
    if (match) {
      out.push({
        path: currentPath,
        reason: match.reason,
      });
      return out;
    }
  }

  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return out;
  }

  for (const [key, child] of Object.entries(value)) {
    if (typeof key !== 'string' || key.trim() === '') continue;
    const nextPath = currentPath ? `${currentPath}.${key}` : key;
    collectProtectedRuntimeConfigMutations(child, nextPath, out);
  }
  return out;
}

function assertAllowedRuntimeConfigOverride(override, label = '--runtime-config-json') {
  if (override == null) return;
  const violations = collectProtectedRuntimeConfigMutations(override);
  if (violations.length === 0) return;

  const seen = new Set();
  const lines = [];
  for (const violation of violations) {
    const key = `${violation.path}::${violation.reason}`;
    if (seen.has(key)) continue;
    seen.add(key);
    lines.push(`- ${violation.path}: ${violation.reason}`);
  }

  throw new Error(
    `${label} must not override compare-managed fairness or cadence fields:\n`
    + lines.join('\n')
  );
}

function assertAllowedCompareRuntimeConfigOverride(runtimeConfigOverride) {
  assertAllowedRuntimeConfigOverride(runtimeConfigOverride);
}

function toFailurePayload(library, error) {
  return {
    failed: true,
    env: { library },
    error: {
      message: redactSecrets(error?.message || error),
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

function normalizeJsonPayload(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeJsonPayload(entry));
  }
  if (value && typeof value === 'object') {
    const normalized = {};
    for (const key of Object.keys(value).sort()) {
      normalized[key] = normalizeJsonPayload(value[key]);
    }
    return normalized;
  }
  return value;
}

function hashStableJsonPayload(payload) {
  const text = JSON.stringify(normalizeJsonPayload(payload), null, 0);
  return crypto.createHash('sha256').update(text).digest('hex');
}

function resolveDopplerLoadModeRuntimeOverlay(loadMode) {
  const normalized = String(loadMode ?? '').trim().toLowerCase();
  if (!normalized) return null;
  return BENCHMARK_POLICY.doppler.loadModeRuntimeOverlays[normalized] ?? null;
}

function summarizeDopplerLoadModeRuntimeOverlay(loadMode) {
  const overlay = resolveDopplerLoadModeRuntimeOverlay(loadMode);
  if (!overlay) return null;
  return {
    loadMode,
    description: overlay.description,
    runtimeConfig: overlay.runtimeConfig,
    runtimeConfigSha256: hashJsonPayload(overlay.runtimeConfig),
  };
}

async function readJsonFileWithHash(filePath, label) {
  const raw = await fs.readFile(filePath, 'utf-8');
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
  return {
    payload,
    source: filePath,
    sourceSha256: hashText(raw),
  };
}

async function fetchJsonWithHash(url, label) {
  let response;
  try {
    response = await fetch(url, { method: 'GET', redirect: 'follow' });
  } catch (error) {
    throw new Error(`${label} fetch failed: ${error.message}`);
  }
  if (!response.ok) {
    throw new Error(`${label} HTTP ${response.status} at ${url}`);
  }
  const raw = await response.text();
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} returned invalid JSON: ${error.message}`);
  }
  return {
    payload,
    source: url,
    sourceSha256: hashText(raw),
  };
}

async function loadManifestForCompareSource(dopplerModelSource) {
  if (dopplerModelSource.manifestSourceType === 'remote') {
    return fetchJsonWithHash(dopplerModelSource.manifestSource, 'compare Doppler manifest');
  }
  return readJsonFileWithHash(dopplerModelSource.manifestSource, 'compare Doppler manifest');
}

function buildDopplerManifestFreshnessArtifact({
  dopplerModelId,
  manifestSourceType,
  activeManifest,
  activeManifestSource,
  activeManifestSha256,
  localManifest,
  localManifestSource,
  localManifestSha256,
  enforced = false,
} = {}) {
  const active = {
    manifestSource: toRepoRelativeSourcePath(activeManifestSource ?? null),
    manifestSha256: activeManifestSha256 ?? null,
    canonicalManifestSha256: activeManifest == null ? null : hashStableJsonPayload(activeManifest),
  };
  const localCurrent = {
    manifestSource: toRepoRelativeSourcePath(localManifestSource ?? null),
    manifestSha256: localManifestSha256 ?? null,
    canonicalManifestSha256: localManifest == null ? null : hashStableJsonPayload(localManifest),
    present: localManifest != null,
  };
  if (manifestSourceType !== 'remote') {
    return {
      schemaVersion: 1,
      checked: false,
      ok: true,
      enforced,
      reason: 'active-manifest-is-local',
      active,
      localCurrent,
      errors: [],
    };
  }
  if (localManifest == null) {
    return {
      schemaVersion: 1,
      checked: false,
      ok: true,
      enforced,
      reason: 'local-current-manifest-missing',
      active,
      localCurrent,
      errors: [],
    };
  }
  const ok = active.canonicalManifestSha256 === localCurrent.canonicalManifestSha256;
  return {
    schemaVersion: 1,
    checked: true,
    ok,
    enforced,
    reason: ok ? 'remote-matches-local-current-manifest' : 'remote-differs-from-local-current-manifest',
    active,
    localCurrent,
    errors: ok ? [] : [
      `Remote Doppler manifest for "${dopplerModelId}" does not match the current local manifest. `
      + `Use --model-url ${pathToFileURL(path.dirname(localManifestSource)).href} for a current-local claim run, `
      + 'refresh the hosted artifact, or pass --allow-non-comparable-lane for diagnostic-only stale-artifact evidence.',
    ],
  };
}

async function resolveDopplerManifestFreshnessArtifact(dopplerModelSource, dopplerModelId, manifestBundle, options = {}) {
  if (dopplerModelSource.manifestSourceType !== 'remote') {
    return buildDopplerManifestFreshnessArtifact({
      dopplerModelId,
      manifestSourceType: dopplerModelSource.manifestSourceType,
      activeManifest: manifestBundle.payload,
      activeManifestSource: manifestBundle.source,
      activeManifestSha256: manifestBundle.sourceSha256,
      enforced: options.enforceLocalManifestFreshness === true,
    });
  }
  const localManifestSource = path.join(DOPPLER_ROOT, 'models', 'local', dopplerModelId, 'manifest.json');
  let localManifestBundle = null;
  try {
    localManifestBundle = await readJsonFileWithHash(localManifestSource, 'local current Doppler manifest freshness candidate');
  } catch (error) {
    if (error?.code !== 'ENOENT') {
      throw error;
    }
  }
  return buildDopplerManifestFreshnessArtifact({
    dopplerModelId,
    manifestSourceType: dopplerModelSource.manifestSourceType,
    activeManifest: manifestBundle.payload,
    activeManifestSource: manifestBundle.source,
    activeManifestSha256: manifestBundle.sourceSha256,
    localManifest: localManifestBundle?.payload ?? null,
    localManifestSource,
    localManifestSha256: localManifestBundle?.sourceSha256 ?? null,
    enforced: options.enforceLocalManifestFreshness === true,
  });
}

async function preflightDopplerManifestContract(dopplerModelSource, dopplerModelId, options = {}) {
  const manifestBundle = await loadManifestForCompareSource(dopplerModelSource);
  const manifest = manifestBundle.payload;
  const requiredInferenceFieldsArtifact = buildManifestRequiredInferenceFieldsArtifact(
    manifest?.inference ?? null,
    `${dopplerModelId}.manifest.inference`
  );
  const executionContractArtifact = buildExecutionContractArtifact(manifest);
  const executionContractOk = executionContractArtifact == null || executionContractArtifact.ok === true;
  const artifactFreshness = await resolveDopplerManifestFreshnessArtifact(
    dopplerModelSource,
    dopplerModelId,
    manifestBundle,
    options
  );
  const artifactFreshnessOk = artifactFreshness.ok === true || artifactFreshness.enforced !== true;
  const artifactIdentity = manifest?.artifactIdentity && typeof manifest.artifactIdentity === 'object'
    ? {
      sourceCheckpointId: manifest.artifactIdentity.sourceCheckpointId ?? null,
      weightPackId: manifest.artifactIdentity.weightPackId ?? null,
      manifestVariantId: manifest.artifactIdentity.manifestVariantId ?? null,
      shardSetHash: manifest.artifactIdentity.shardSetHash ?? null,
      artifactCompleteness: manifest.artifactIdentity.artifactCompleteness ?? null,
    }
    : null;
  const errors = [
    ...(Array.isArray(requiredInferenceFieldsArtifact?.errors) ? requiredInferenceFieldsArtifact.errors : []),
    ...(Array.isArray(executionContractArtifact?.errors) ? executionContractArtifact.errors : []),
    ...(artifactFreshness.enforced === true && Array.isArray(artifactFreshness.errors) ? artifactFreshness.errors : []),
    ...(!artifactIdentity ? [`${dopplerModelId}.manifest.artifactIdentity is required for compare evidence`] : []),
  ];
  return {
    manifestSource: toRepoRelativeSourcePath(manifestBundle.source),
    manifestSha256: manifestBundle.sourceSha256,
    artifactIdentity,
    weightsRef: manifest?.weightsRef ?? null,
    requiredInferenceFieldsArtifact,
    executionContractArtifact,
    artifactFreshness,
    ok: requiredInferenceFieldsArtifact.ok === true && executionContractOk && artifactFreshnessOk && artifactIdentity !== null,
    errors,
  };
}

function formatPreflightErrors(errors = []) {
  return errors.map((entry) => `- ${entry}`).join('\n');
}

async function resolveTransformersjsRuntimeStack(tjsVersion) {
  const normalizedVersion = String(tjsVersion ?? '').trim();
  const runnerHtmlRaw = await fs.readFile(TRANSFORMERSJS_RUNNER_HTML_PATH, 'utf-8');
  const runnerHtmlSha256 = hashText(runnerHtmlRaw);
  const stack = {
    requestedMajor: normalizedVersion || null,
    runner: {
      source: toRepoRelativeSourcePath(TRANSFORMERSJS_RUNNER_HTML_PATH),
      sourceSha256: runnerHtmlSha256,
    },
  };

  if (normalizedVersion !== '4') {
    return stack;
  }

  const [tjsPackage, ortWebPackage, ortCommonPackage] = await Promise.all([
    readJsonFileWithHash(TJS_PACKAGE_JSON_PATH, '@huggingface/transformers package.json'),
    readJsonFileWithHash(TJS_ORT_WEB_PACKAGE_JSON_PATH, 'nested onnxruntime-web package.json'),
    readJsonFileWithHash(TJS_ORT_COMMON_PACKAGE_JSON_PATH, 'nested onnxruntime-common package.json'),
  ]);

  if (!runnerHtmlRaw.includes(TJS_V4_ORT_COMMON_IMPORT_PATH) || !runnerHtmlRaw.includes(TJS_V4_ORT_WEB_IMPORT_PATH)) {
    throw new Error(
      'transformersjs-runner.html is not pinned to the installed TJS v4 nested ORT modules. '
      + `Expected import paths ${TJS_V4_ORT_COMMON_IMPORT_PATH} and ${TJS_V4_ORT_WEB_IMPORT_PATH}.`
    );
  }

  return {
    ...stack,
    library: {
      packageName: '@huggingface/transformers',
      version: tjsPackage.payload?.version ?? null,
      source: toRepoRelativeSourcePath(tjsPackage.source),
      sourceSha256: tjsPackage.sourceSha256,
    },
    onnxruntime: {
      web: {
        version: ortWebPackage.payload?.version ?? null,
        dependencyOnCommon: ortWebPackage.payload?.dependencies?.['onnxruntime-common'] ?? null,
        source: toRepoRelativeSourcePath(ortWebPackage.source),
        sourceSha256: ortWebPackage.sourceSha256,
        importPath: TJS_V4_ORT_WEB_IMPORT_PATH,
      },
      common: {
        version: ortCommonPackage.payload?.version ?? null,
        source: toRepoRelativeSourcePath(ortCommonPackage.source),
        sourceSha256: ortCommonPackage.sourceSha256,
        importPath: TJS_V4_ORT_COMMON_IMPORT_PATH,
      },
    },
  };
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
    throw new Error(
      '--doppler-kernel-path is removed because string kernel-path IDs are no longer supported. ' +
      'Use the model manifest execution-v1 graph.'
    );
  }
  return {
    kernelPath: DEFAULT_DOPPLER_KERNEL_PATH,
    source: 'manifest-execution-v1',
  };
}

function assertKernelPathAllowedForModel(modelId, kernelPath) {
  if (kernelPath == null || kernelPath === '') return;
  const normalizedModelId = String(modelId ?? '').trim().toLowerCase();
  const blockedKernelPaths = KNOWN_BAD_DOPPLER_KERNEL_PATHS_BY_MODEL[normalizedModelId] || [];
  if (!blockedKernelPaths.includes(String(kernelPath))) return;
  throw new Error(
    `Kernel path "${kernelPath}" is blocked for model "${modelId}" in compare runs due to deterministic correctness failures. `
    + 'Use the manifest execution-v1 graph.'
  );
}

function resolveDopplerCliExecutor(surface) {
  if (surface === 'bun') return 'bun';
  return 'node';
}

function resolveDopplerCommandSurface(surface, isSafetensorsFormat) {
  if (isSafetensorsFormat || surface === 'bun') return 'node';
  return surface;
}

function resolveDopplerExecutionIdentity(surface, format = DEFAULT_DOPPLER_FORMAT) {
  const requestedSurface = parseChoice(
    surface,
    DOPPLER_SURFACES,
    '--doppler-surface',
    'auto'
  );
  const resolvedFormat = parseChoice(
    format,
    DOPPLER_FORMATS,
    '--doppler-format',
    DEFAULT_DOPPLER_FORMAT
  );
  const commandSurface = resolveDopplerCommandSurface(
    requestedSurface,
    resolvedFormat === 'safetensors'
  );
  const cliExecutor = resolveDopplerCliExecutor(requestedSurface);
  const commandSurfaceReason = (() => {
    if (resolvedFormat === 'safetensors' && commandSurface === 'node') {
      return 'safetensors-format-node-command-surface';
    }
    if (requestedSurface === 'bun' && commandSurface === 'node') {
      return 'bun-executor-node-command-surface';
    }
    return 'matches-requested-surface';
  })();
  return {
    requestedSurface,
    commandSurface,
    cliExecutor,
    format: resolvedFormat,
    commandSurfaceReason,
  };
}

function isDopplerBrowserCommandSurface(surface) {
  return surface === 'auto' || surface === 'browser';
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
      || key === 'skip-matrix-update'
      || key === 'allow-non-comparable-lane'
      || key === 'resource-telemetry'
      || key === 'resource-telemetry-samples'
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

function resolveBrowserChannelOverride(requestedChannel) {
  if (requestedChannel == null || requestedChannel === '') {
    return BENCHMARK_POLICY.browser.compareChannelByPlatform[process.platform] ?? null;
  }
  const normalized = String(requestedChannel).trim();
  return normalized === '' ? null : normalized;
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
  promptContract = null,
}) {
  const normalizedPromptContract = promptContract && typeof promptContract === 'object'
    ? promptContract
    : null;
  return {
    prompt: String(prompt ?? DEFAULT_PROMPT),
    promptRaw: normalizedPromptContract?.promptRaw ?? String(prompt ?? DEFAULT_PROMPT),
    promptContract: normalizedPromptContract,
    maxTokens: parsePositiveInt(maxTokens, DEFAULT_MAX_TOKENS, '--max-tokens'),
    warmupRuns: parseNonNegativeInt(warmupRuns, DEFAULT_WARMUP, '--warmup'),
    timedRuns: parsePositiveInt(timedRuns, DEFAULT_RUNS, '--runs'),
    seed: parseNonNegativeInt(seed, DEFAULT_SEED, '--seed'),
    sampling: {
      temperature: parseNonNegativeNumber(sampling.temperature, DEFAULT_SHARED_SAMPLING.temperature, 'sampling.temperature'),
      topK: parsePositiveInt(sampling.topK, DEFAULT_SHARED_SAMPLING.topK, 'sampling.topK'),
      topP: parseTopP(sampling.topP, DEFAULT_SHARED_SAMPLING.topP, 'sampling.topP'),
      repetitionPenalty: (() => {
        const value = parseNonNegativeNumber(
          sampling.repetitionPenalty,
          DEFAULT_SHARED_SAMPLING.repetitionPenalty,
          'sampling.repetitionPenalty'
        );
        if (!(value > 0)) {
          throw new Error('sampling.repetitionPenalty must be greater than 0');
        }
        return value;
      })(),
      greedyThreshold: parseNonNegativeNumber(
        sampling.greedyThreshold,
        DEFAULT_SHARED_SAMPLING.greedyThreshold,
        'sampling.greedyThreshold'
      ),
    },
    useChatTemplate: useChatTemplate === true,
    loadMode,
  };
}

function resolveRunLoadMode(cacheMode, sharedLoadMode, compareLoadModeDefaults) {
  if (sharedLoadMode != null) return sharedLoadMode;
  if (cacheMode !== 'warm' && cacheMode !== 'cold') {
    throw new Error(`cacheMode must be "warm" or "cold"; got ${String(cacheMode)}`);
  }
  const defaults = compareLoadModeDefaults ?? null;
  const resolved = defaults?.[cacheMode] ?? null;
  if (resolved != null) {
    return resolved;
  }
  throw new Error(
    `No explicit ${cacheMode} load mode is configured. `
    + 'Set compare-engines.config.json defaults or pass --load-mode.'
  );
}

function resolveCompareLoadModes(sharedLoadMode, compareLoadModeDefaults) {
  return {
    warm: resolveRunLoadMode('warm', sharedLoadMode, compareLoadModeDefaults),
    cold: resolveRunLoadMode('cold', sharedLoadMode, compareLoadModeDefaults),
  };
}

function buildDopplerRuntimeConfig(sharedContract, engineOverlay) {
  const resolvedLoadMode = engineOverlay.loadMode ?? sharedContract.loadMode ?? null;
  const kvCachePlan = engineOverlay.kvCachePlan?.enabled === true
    && Number.isFinite(engineOverlay.kvCachePlan.maxSeqLen)
    && engineOverlay.kvCachePlan.maxSeqLen > 0
    ? engineOverlay.kvCachePlan
    : null;
  const sampling = {
    temperature: sharedContract.sampling.temperature,
    topK: sharedContract.sampling.topK,
    topP: sharedContract.sampling.topP,
    repetitionPenalty: sharedContract.sampling.repetitionPenalty,
    greedyThreshold: sharedContract.sampling.greedyThreshold,
  };
  const compareManagedRuntimeConfig = {
    shared: {
      benchmark: {
        run: {
          customPrompt: sharedContract.prompt,
          maxNewTokens: sharedContract.maxTokens,
          warmupRuns: sharedContract.warmupRuns,
          timedRuns: sharedContract.timedRuns,
          useChatTemplate: sharedContract.useChatTemplate,
          loadMode: resolvedLoadMode,
          sampling,
        },
      },
    },
    inference: {
      kernelPathPolicy: engineOverlay.kernelPathPolicy,
      prompt: sharedContract.prompt,
      chatTemplate: {
        enabled: sharedContract.useChatTemplate,
      },
      generation: {
        maxTokens: sharedContract.maxTokens,
        disableMultiTokenDecode: engineOverlay.disableMultiTokenDecode === true,
      },
      ...(kvCachePlan
        ? {
            kvcache: {
              maxSeqLen: Math.trunc(kvCachePlan.maxSeqLen),
            },
          }
        : {}),
      batching: {
        maxTokens: sharedContract.maxTokens,
        batchSize: engineOverlay.batchSize,
        readbackInterval: engineOverlay.readbackInterval,
        stopCheckMode: engineOverlay.stopCheckMode,
        ...(engineOverlay.readbackMode ? { readbackMode: engineOverlay.readbackMode } : {}),
      },
      session: {
        ...(kvCachePlan
          ? {
              kvcache: {
                maxSeqLen: Math.trunc(kvCachePlan.maxSeqLen),
              },
            }
          : {}),
        decodeLoop: {
          batchSize: engineOverlay.batchSize,
          stopCheckMode: engineOverlay.stopCheckMode,
          readbackInterval: engineOverlay.readbackInterval,
          ...(engineOverlay.readbackMode ? { readbackMode: engineOverlay.readbackMode } : {}),
          disableCommandBatching: false,
        },
        ...(engineOverlay.speculationMode
          ? {
              speculation: {
                mode: engineOverlay.speculationMode,
                tokens: 1,
                verify: 'greedy',
                threshold: null,
                rollbackOnReject: true,
              },
            }
          : {}),
      },
      sampling,
    },
  };
  if (engineOverlay.kernelPath) {
    compareManagedRuntimeConfig.inference.kernelPath = engineOverlay.kernelPath;
  }

  let runtimeConfig = {};
  if (engineOverlay.runtimeBaseConfigJson != null) {
    if (
      typeof engineOverlay.runtimeBaseConfigJson !== 'object'
      || engineOverlay.runtimeBaseConfigJson == null
      || Array.isArray(engineOverlay.runtimeBaseConfigJson)
    ) {
      throw new Error('compare profile runtime base override is expected to be a parsed JSON object');
    }
    runtimeConfig = engineOverlay.runtimeBaseConfigJson;
  }

  if (engineOverlay.runtimeLoadModeConfigJson != null) {
    if (
      typeof engineOverlay.runtimeLoadModeConfigJson !== 'object'
      || engineOverlay.runtimeLoadModeConfigJson == null
      || Array.isArray(engineOverlay.runtimeLoadModeConfigJson)
    ) {
      throw new Error('benchmark-policy load-mode runtime overlay is expected to be a parsed JSON object');
    }
    assertAllowedRuntimeConfigOverride(
      engineOverlay.runtimeLoadModeConfigJson,
      'benchmark-policy doppler.loadModeRuntimeOverlays runtimeConfig'
    );
    runtimeConfig = mergeRuntimeValues(runtimeConfig, engineOverlay.runtimeLoadModeConfigJson);
  }

  runtimeConfig = mergeRuntimeValues(runtimeConfig, compareManagedRuntimeConfig);

  if (engineOverlay.runtimeConfigJson) {
    if (typeof engineOverlay.runtimeConfigJson !== 'object' || engineOverlay.runtimeConfigJson == null) {
      throw new Error('--runtime-config-json override is expected to be a parsed JSON object');
    }
    assertAllowedRuntimeConfigOverride(engineOverlay.runtimeConfigJson);
    runtimeConfig = mergeRuntimeValues(runtimeConfig, engineOverlay.runtimeConfigJson);
  }

  if (engineOverlay.runtimeDiagnosticConfigJson != null) {
    if (
      typeof engineOverlay.runtimeDiagnosticConfigJson !== 'object'
      || engineOverlay.runtimeDiagnosticConfigJson == null
      || Array.isArray(engineOverlay.runtimeDiagnosticConfigJson)
    ) {
      throw new Error('compare-owned diagnostic runtime overlay is expected to be a parsed JSON object');
    }
    runtimeConfig = mergeRuntimeValues(runtimeConfig, engineOverlay.runtimeDiagnosticConfigJson);
  }
  return runtimeConfig;
}

async function runDoppler(modelId, modelUrl, sharedContract, cacheMode, options = {}) {
  const resolvedCommand = parseChoice(
    options.command,
    DOPPLER_COMPARE_COMMANDS,
    'runDoppler command',
    'bench'
  );
  const resolvedWorkload = options.workload ?? (resolvedCommand === 'debug' ? 'inference' : null);
  const resolvedSurface = parseChoice(
    options.surface,
    DOPPLER_SURFACES,
    '--doppler-surface',
    'auto'
  );
  const resolvedFormat = parseChoice(
    options.format,
    DOPPLER_FORMATS,
    '--doppler-format',
    DEFAULT_DOPPLER_FORMAT
  );
  const resolvedKernelPath = options.kernelPath ?? DEFAULT_DOPPLER_KERNEL_PATH;
  const isSafetensorsFormat = resolvedFormat === 'safetensors';
  const resolvedLoadMode = isSafetensorsFormat ? 'memory' : (options.loadMode ?? null);
  const resolvedTimeoutMs = parsePositiveInt(options.timeoutMs, DEFAULT_DOPPLER_TIMEOUT_MS, '--doppler-timeout-ms');
  const resolvedBatchSize = parsePositiveInt(options.batchSize, DEFAULT_DOPPLER_BATCH_SIZE, '--doppler-batch-size');
  const resolvedReadbackInterval = parsePositiveInt(
    options.readbackInterval,
    DEFAULT_DOPPLER_READBACK_INTERVAL,
    '--doppler-readback-interval'
  );
  const resolvedStopCheckMode = normalizeStopCheckMode(
    options.stopCheckMode ?? DEFAULT_DOPPLER_STOP_CHECK_MODE,
    '--doppler-stop-check-mode'
  );
  const resolvedDisableMultiTokenDecode = options.disableMultiTokenDecode === true;
  const resolvedSpeculationMode = typeof options.speculationMode === 'string'
    ? options.speculationMode.trim().toLowerCase()
    : '';
  if (resolvedSpeculationMode && resolvedSpeculationMode !== 'none' && resolvedSpeculationMode !== 'self') {
    throw new Error('runDoppler speculationMode must be "none", "self", or omitted.');
  }
  const resolvedBrowserPort = parseNonNegativeInt(
    options.browserPort,
    DEFAULT_DOPPLER_BROWSER_PORT,
    '--doppler-browser-port'
  );
  const resolvedMaxBufferBytes = parsePositiveInt(
    options.maxBufferBytes,
    DEFAULT_DOPPLER_RUN_MAX_BUFFER_BYTES,
    'runDoppler maxBufferBytes'
  );
  const resolvedBrowserChannel = resolveBrowserChannelOverride(options.browserChannel);
  const resolvedBrowserBaseUrl = options.browserBaseUrl || null;
  const resolvedBrowserExecutable = options.browserExecutable || null;
  const dopplerExecution = resolveDopplerExecutionIdentity(resolvedSurface, resolvedFormat);
  const loadModeRuntimeOverlay = resolveDopplerLoadModeRuntimeOverlay(resolvedLoadMode);
  const runtimeConfig = buildDopplerRuntimeConfig(sharedContract, {
    kernelPath: resolvedKernelPath,
    kernelPathPolicy: BENCHMARK_POLICY.kernelPathPolicy.compareRuntime,
    batchSize: resolvedBatchSize,
    readbackInterval: resolvedReadbackInterval,
    disableMultiTokenDecode: resolvedDisableMultiTokenDecode,
    speculationMode: resolvedSpeculationMode || null,
    stopCheckMode: resolvedStopCheckMode,
    loadMode: resolvedLoadMode,
    kvCachePlan: options.kvCachePlan ?? null,
    runtimeBaseConfigJson: options.runtimeBaseConfigJson ?? null,
    runtimeLoadModeConfigJson: loadModeRuntimeOverlay?.runtimeConfig ?? null,
    runtimeConfigJson: options.runtimeConfigJson,
    runtimeDiagnosticConfigJson: options.runtimeDiagnosticConfigJson,
  });
  const stableBrowserArgs = appendBrowserArgs([], STABLE_BROWSER_ARGS);
  const cliExecutor = dopplerExecution.cliExecutor;
  const resourceTelemetryOptions = options.resourceTelemetry?.enabled === true
    ? {
        ...options.resourceTelemetry,
        label: options.resourceTelemetry.label ?? `doppler-${cacheMode}`,
      }
    : { enabled: false };

  const buildCliConfig = (benchCaptureDir = null) => {
    const resolvedModelUrl = isSafetensorsFormat && options.safetensorsSourcePath
      ? options.safetensorsSourcePath
      : modelUrl;
    const commandSurface = dopplerExecution.commandSurface;
    let run = {
      surface: commandSurface,
    };
    if (isDopplerBrowserCommandSurface(commandSurface)) {
      run.browser = {
        ...(resolvedBrowserBaseUrl
          ? { baseUrl: String(resolvedBrowserBaseUrl) }
          : (options.browserPort != null ? { port: resolvedBrowserPort } : {})),
        ...(resolvedBrowserChannel ? { channel: resolvedBrowserChannel } : {}),
        ...(resolvedBrowserExecutable ? { executablePath: String(resolvedBrowserExecutable) } : {}),
        ...(options.browserUserData ? { userDataDir: String(options.browserUserData) } : {}),
        ...(options.noOpfsCache ? { opfsCache: false } : {}),
        browserArgs: stableBrowserArgs,
      };
    }
    if (resolvedCommand === 'bench' && benchCaptureDir) {
      run = applyDopplerBenchCaptureRunConfig(run, benchCaptureDir);
    }
    return {
      request: {
        ...(resolvedWorkload ? { workload: resolvedWorkload } : {}),
        modelId,
        modelUrl: resolvedModelUrl,
        cacheMode,
        ...(resolvedLoadMode != null ? { loadMode: String(resolvedLoadMode) } : {}),
      },
      run,
    };
  };

  const diagnosticLabel = options.diagnosticLabel ? `, ${options.diagnosticLabel}` : '';
  console.error(`[compare] running Doppler ${resolvedCommand} (${cacheMode}${diagnosticLabel})...`);
  const runOnce = async () => {
    const benchCaptureDir = resolvedCommand === 'bench'
      ? await createDopplerBenchCaptureDir()
      : null;
    try {
      const cliConfig = buildCliConfig(benchCaptureDir);
      const args = [
        path.join(DOPPLER_ROOT, 'src', 'cli', 'doppler-cli.js'),
        resolvedCommand,
        '--config',
        JSON.stringify(cliConfig),
        '--runtime-config',
        JSON.stringify(runtimeConfig),
        '--json',
      ];
      let parsed;
      if (benchCaptureDir) {
        const childResult = await execFileForSavedDopplerBench(cliExecutor, args, {
          cwd: DOPPLER_ROOT,
          timeout: resolvedTimeoutMs,
          resourceTelemetry: resourceTelemetryOptions,
        });
        parsed = await readDopplerBenchCaptureResult(benchCaptureDir, `Doppler ${resolvedCommand} (${cacheMode})`);
        if (childResult.resourceTelemetry != null) {
          parsed.resourceTelemetry = childResult.resourceTelemetry;
        }
      } else {
        const { stdout, resourceTelemetry } = await execFileCaptureStdout(cliExecutor, args, {
          cwd: DOPPLER_ROOT,
          timeout: resolvedTimeoutMs,
          maxBuffer: resolvedMaxBufferBytes,
          resourceTelemetry: resourceTelemetryOptions,
        });
        parsed = parseJsonBlock(stdout, `Doppler ${resolvedCommand} (${cacheMode})`);
        if (resourceTelemetry != null) {
          parsed.resourceTelemetry = resourceTelemetry;
        }
      }
      return {
        ...parsed,
        dopplerExecution,
      };
    } catch (error) {
      throw error;
    } finally {
      await removeDopplerBenchCaptureDir(benchCaptureDir);
    }
  };

  try {
    return await runOnce();
  } catch (error) {
    console.error(`[compare] Doppler ${resolvedCommand} (${cacheMode}) failed: ${redactSecrets(error.message)}`);
    return {
      ...toFailurePayload('doppler', error),
      dopplerExecution,
    };
  }
}

async function runTjs(modelId, sharedContract, cacheMode, tjsVersion, localModelPath, options = {}) {
  const resolvedProfileOps = options.profileOps ?? DEFAULT_TJS_PROFILE_OPS;
  const resolvedLoadMode = options.loadMode ?? null;
  const resolvedTimeoutMs = parsePositiveInt(options.timeoutMs, DEFAULT_TJS_TIMEOUT_MS, '--tjs-timeout-ms');
  const resolvedServerPort = parseNonNegativeInt(options.serverPort, DEFAULT_TJS_SERVER_PORT, '--tjs-server-port');
  const resolvedTjsDtype = parseChoice(options.tjsDtype, TJS_DTYPES, '--tjs-dtype', DEFAULT_TJS_DTYPE);
  const resolvedTjsFormat = parseChoice(options.tjsFormat, TJS_FORMATS, '--tjs-format', DEFAULT_TJS_FORMAT);
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
    '--format', resolvedTjsFormat,
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
  const resourceTelemetryOptions = options.resourceTelemetry?.enabled === true
    ? {
        ...options.resourceTelemetry,
        label: options.resourceTelemetry.label ?? `transformersjs-${cacheMode}`,
      }
    : { enabled: false };

  console.error(`[compare] running TJS v${tjsVersion} (${cacheMode})...`);
  const runOnce = async () => {
    const { stdout, resourceTelemetry } = await execFileCaptureStdout('node', args, {
      cwd: DOPPLER_ROOT,
      timeout: resolvedTimeoutMs,
      maxBuffer: 10 * 1024 * 1024,
      resourceTelemetry: resourceTelemetryOptions,
    });
    const parsed = parseJsonBlock(stdout, `TJS ${modelId} (${cacheMode})`);
    if (resourceTelemetry != null) {
      parsed.resourceTelemetry = resourceTelemetry;
    }
    return parsed;
  };
  try {
    return await runOnce();
  } catch (error) {
    console.error(`[compare] TJS (${cacheMode}) failed: ${redactSecrets(error.message)}`);
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

function resolveDopplerRuntimeConfigFromResult(result) {
  const candidates = [
    result?.request?.runtimeConfig,
    result?.result?.request?.runtimeConfig,
    result?.result?.result?.request?.runtimeConfig,
  ];
  for (const candidate of candidates) {
    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
      return candidate;
    }
  }
  return null;
}

function isComparePlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function resolveDopplerMetricsDecodeCadenceFromResult(result) {
  const candidates = [
    result?.metrics?.decodeCadence,
    result?.result?.metrics?.decodeCadence,
    result?.result?.result?.metrics?.decodeCadence,
  ];
  for (const candidate of candidates) {
    if (isComparePlainObject(candidate)) {
      return candidate;
    }
  }
  return null;
}

function resolveDopplerDecodeCadenceFromRuntimeConfig(runtimeConfig) {
  const batching = runtimeConfig?.inference?.batching;
  const decodeLoop = runtimeConfig?.inference?.session?.decodeLoop;
  const generation = runtimeConfig?.inference?.generation;
  const speculation = runtimeConfig?.inference?.session?.speculation;
  if (
    !batching || typeof batching !== 'object' || Array.isArray(batching)
    || !decodeLoop || typeof decodeLoop !== 'object' || Array.isArray(decodeLoop)
  ) {
    return null;
  }
  return {
    batchSize: decodeLoop.batchSize,
    readbackInterval: decodeLoop.readbackInterval,
    maxBatchDecodeTokens: decodeLoop.maxBatchDecodeTokens ?? null,
    stopCheckMode: decodeLoop.stopCheckMode,
    readbackMode: decodeLoop.readbackMode ?? batching.readbackMode ?? null,
    disableCommandBatching: decodeLoop.disableCommandBatching,
    disableMultiTokenDecode: generation?.disableMultiTokenDecode === true,
    speculationMode: speculation?.mode ?? null,
    tokensPerReadback: Number.isFinite(decodeLoop.batchSize) && Number.isFinite(decodeLoop.readbackInterval)
      ? decodeLoop.batchSize * decodeLoop.readbackInterval
      : null,
    runtimeMirror: {
      batching: {
        batchSize: batching.batchSize,
        readbackInterval: batching.readbackInterval,
        stopCheckMode: batching.stopCheckMode,
        readbackMode: batching.readbackMode ?? null,
      },
      decodeLoop: {
        batchSize: decodeLoop.batchSize,
        readbackInterval: decodeLoop.readbackInterval,
        maxBatchDecodeTokens: decodeLoop.maxBatchDecodeTokens ?? null,
        stopCheckMode: decodeLoop.stopCheckMode,
        readbackMode: decodeLoop.readbackMode ?? null,
      },
    },
  };
}

function mergeDopplerDecodeCadence(runtimeCadence, metricsCadence) {
  if (!metricsCadence) {
    return runtimeCadence;
  }
  if (!runtimeCadence) {
    return metricsCadence;
  }
  return {
    batchSize: metricsCadence.batchSize ?? runtimeCadence.batchSize,
    readbackInterval: metricsCadence.readbackInterval ?? runtimeCadence.readbackInterval,
    maxBatchDecodeTokens: metricsCadence.maxBatchDecodeTokens ?? runtimeCadence.maxBatchDecodeTokens,
    stopCheckMode: metricsCadence.stopCheckMode ?? runtimeCadence.stopCheckMode,
    readbackMode: metricsCadence.readbackMode ?? runtimeCadence.readbackMode,
    disableCommandBatching: metricsCadence.disableCommandBatching ?? runtimeCadence.disableCommandBatching,
    disableMultiTokenDecode: metricsCadence.disableMultiTokenDecode ?? runtimeCadence.disableMultiTokenDecode,
    speculationMode: metricsCadence.speculationMode ?? runtimeCadence.speculationMode,
    tokensPerReadback: metricsCadence.tokensPerReadback ?? runtimeCadence.tokensPerReadback,
    runtimeMirror: metricsCadence.runtimeMirror ?? runtimeCadence.runtimeMirror,
    executionPlan: metricsCadence.executionPlan ?? null,
  };
}

function resolveDopplerDecodeCadenceFromResult(result) {
  if (result?.failed) {
    return null;
  }
  const runtimeConfig = resolveDopplerRuntimeConfigFromResult(result);
  const runtimeCadence = resolveDopplerDecodeCadenceFromRuntimeConfig(runtimeConfig);
  const metricsCadence = resolveDopplerMetricsDecodeCadenceFromResult(result);
  return mergeDopplerDecodeCadence(runtimeCadence, metricsCadence);
}

function resolveDopplerMemoryAccountingFromResult(result) {
  if (result?.failed) {
    return null;
  }
  const usedBytes = firstFiniteStatNumber(result, [
    'result.memoryStats.used',
    'memoryStats.used',
    'result.result.memoryStats.used',
  ]);
  const peakBytesAllocated = firstFiniteStatNumber(result, [
    'result.memoryStats.pool.peakBytesAllocated',
    'memoryStats.pool.peakBytesAllocated',
    'result.result.memoryStats.pool.peakBytesAllocated',
  ]);
  const peakBytesRequested = firstFiniteStatNumber(result, [
    'result.memoryStats.pool.peakBytesRequested',
    'memoryStats.pool.peakBytesRequested',
    'result.result.memoryStats.pool.peakBytesRequested',
  ]);
  const currentBytesAllocated = firstFiniteStatNumber(result, [
    'result.memoryStats.pool.currentBytesAllocated',
    'memoryStats.pool.currentBytesAllocated',
    'result.result.memoryStats.pool.currentBytesAllocated',
  ]);
  const currentBytesRequested = firstFiniteStatNumber(result, [
    'result.memoryStats.pool.currentBytesRequested',
    'memoryStats.pool.currentBytesRequested',
    'result.result.memoryStats.pool.currentBytesRequested',
  ]);
  const activeBuffers = firstFiniteStatNumber(result, [
    'result.memoryStats.pool.activeBuffers',
    'memoryStats.pool.activeBuffers',
    'result.result.memoryStats.pool.activeBuffers',
  ]);
  const pooledBuffers = firstFiniteStatNumber(result, [
    'result.memoryStats.pool.pooledBuffers',
    'memoryStats.pool.pooledBuffers',
    'result.result.memoryStats.pool.pooledBuffers',
  ]);
  const kvAllocatedBytes = firstFiniteStatNumber(result, [
    'result.memoryStats.kvCache.allocated',
    'memoryStats.kvCache.allocated',
    'result.result.memoryStats.kvCache.allocated',
  ]);
  const kvUsedBytes = firstFiniteStatNumber(result, [
    'result.memoryStats.kvCache.used',
    'memoryStats.kvCache.used',
    'result.result.memoryStats.kvCache.used',
  ]);
  const kvTheoreticalBytes = firstFiniteStatNumber(result, [
    'result.memoryStats.kvCache.theoretical',
    'memoryStats.kvCache.theoretical',
    'result.result.memoryStats.kvCache.theoretical',
  ]);
  const kvEfficiency = firstFiniteStatNumber(result, [
    'result.memoryStats.kvCache.efficiency',
    'memoryStats.kvCache.efficiency',
    'result.result.memoryStats.kvCache.efficiency',
  ]);
  const kvSeqLen = firstFiniteStatNumber(result, [
    'result.memoryStats.kvCache.seqLen',
    'memoryStats.kvCache.seqLen',
    'result.result.memoryStats.kvCache.seqLen',
  ]);
  const kvMaxSeqLen = firstFiniteStatNumber(result, [
    'result.memoryStats.kvCache.maxSeqLen',
    'memoryStats.kvCache.maxSeqLen',
    'result.result.memoryStats.kvCache.maxSeqLen',
  ]);
  const kvLayout = getByPath(result, 'result.memoryStats.kvCache.layout')
    ?? getByPath(result, 'memoryStats.kvCache.layout')
    ?? getByPath(result, 'result.result.memoryStats.kvCache.layout')
    ?? null;
  const kvDtype = getByPath(result, 'result.memoryStats.kvCache.kvDtype')
    ?? getByPath(result, 'memoryStats.kvCache.kvDtype')
    ?? getByPath(result, 'result.result.memoryStats.kvCache.kvDtype')
    ?? null;
  if (
    !Number.isFinite(usedBytes)
    && !Number.isFinite(peakBytesAllocated)
    && !Number.isFinite(kvAllocatedBytes)
  ) {
    return null;
  }
  return {
    schemaVersion: 1,
    source: 'doppler-memoryStats',
    gpuMemoryComparable: true,
    usedBytes: Number.isFinite(usedBytes) ? usedBytes : null,
    pool: {
      peakBytesAllocated: Number.isFinite(peakBytesAllocated) ? peakBytesAllocated : null,
      peakBytesRequested: Number.isFinite(peakBytesRequested) ? peakBytesRequested : null,
      currentBytesAllocated: Number.isFinite(currentBytesAllocated) ? currentBytesAllocated : null,
      currentBytesRequested: Number.isFinite(currentBytesRequested) ? currentBytesRequested : null,
      activeBuffers: Number.isFinite(activeBuffers) ? activeBuffers : null,
      pooledBuffers: Number.isFinite(pooledBuffers) ? pooledBuffers : null,
    },
    kvCache: {
      allocatedBytes: Number.isFinite(kvAllocatedBytes) ? kvAllocatedBytes : null,
      usedBytes: Number.isFinite(kvUsedBytes) ? kvUsedBytes : null,
      theoreticalBytes: Number.isFinite(kvTheoreticalBytes) ? kvTheoreticalBytes : null,
      efficiency: Number.isFinite(kvEfficiency) ? roundBottleneckNumber(kvEfficiency) : null,
      seqLen: Number.isFinite(kvSeqLen) ? kvSeqLen : null,
      maxSeqLen: Number.isFinite(kvMaxSeqLen) ? kvMaxSeqLen : null,
      layout: typeof kvLayout === 'string' ? kvLayout : null,
      kvDtype: typeof kvDtype === 'string' ? kvDtype : null,
    },
  };
}

function resolveTransformersjsMemoryAccountingFromResult(result) {
  if (result?.failed) {
    return null;
  }
  const beforeUsed = firstFiniteStatNumber(result, [
    'memoryInfo.before.usedJSHeapSize',
    'metrics.memoryInfo.before.usedJSHeapSize',
    'result.memoryInfo.before.usedJSHeapSize',
  ]);
  const afterUsed = firstFiniteStatNumber(result, [
    'memoryInfo.after.usedJSHeapSize',
    'metrics.memoryInfo.after.usedJSHeapSize',
    'result.memoryInfo.after.usedJSHeapSize',
  ]);
  const beforeTotal = firstFiniteStatNumber(result, [
    'memoryInfo.before.totalJSHeapSize',
    'metrics.memoryInfo.before.totalJSHeapSize',
    'result.memoryInfo.before.totalJSHeapSize',
  ]);
  const afterTotal = firstFiniteStatNumber(result, [
    'memoryInfo.after.totalJSHeapSize',
    'metrics.memoryInfo.after.totalJSHeapSize',
    'result.memoryInfo.after.totalJSHeapSize',
  ]);
  const heapLimit = firstFiniteStatNumber(result, [
    'memoryInfo.after.jsHeapSizeLimit',
    'memoryInfo.before.jsHeapSizeLimit',
    'metrics.memoryInfo.after.jsHeapSizeLimit',
    'metrics.memoryInfo.before.jsHeapSizeLimit',
    'result.memoryInfo.after.jsHeapSizeLimit',
    'result.memoryInfo.before.jsHeapSizeLimit',
  ]);
  if (
    !Number.isFinite(beforeUsed)
    && !Number.isFinite(afterUsed)
    && !Number.isFinite(afterTotal)
  ) {
    return null;
  }
  return {
    schemaVersion: 1,
    source: 'browser-performance-memory',
    gpuMemoryComparable: false,
    usedJSHeapBeforeBytes: Number.isFinite(beforeUsed) ? beforeUsed : null,
    usedJSHeapAfterBytes: Number.isFinite(afterUsed) ? afterUsed : null,
    usedJSHeapDeltaBytes: Number.isFinite(beforeUsed) && Number.isFinite(afterUsed)
      ? afterUsed - beforeUsed
      : null,
    totalJSHeapBeforeBytes: Number.isFinite(beforeTotal) ? beforeTotal : null,
    totalJSHeapAfterBytes: Number.isFinite(afterTotal) ? afterTotal : null,
    jsHeapSizeLimitBytes: Number.isFinite(heapLimit) ? heapLimit : null,
  };
}

function buildMemoryAccounting(doppler, transformersjs) {
  const dopplerMemory = resolveDopplerMemoryAccountingFromResult(doppler);
  const transformersjsMemory = resolveTransformersjsMemoryAccountingFromResult(transformersjs);
  if (!dopplerMemory && !transformersjsMemory) {
    return null;
  }
  return {
    schemaVersion: 1,
    gpuMemoryComparable: dopplerMemory?.gpuMemoryComparable === true
      && transformersjsMemory?.gpuMemoryComparable === true,
    doppler: dopplerMemory,
    transformersjs: transformersjsMemory,
    comparability: {
      gpuMemory: dopplerMemory?.gpuMemoryComparable === true
        && transformersjsMemory?.gpuMemoryComparable === true,
      reason: dopplerMemory?.gpuMemoryComparable === true && transformersjsMemory?.gpuMemoryComparable !== true
        ? 'Doppler reports GPU buffer-pool/KV memory; Transformers.js exposes browser JS heap here, not WebGPU allocation residency.'
        : null,
    },
  };
}

function resolveResourceTelemetryFromResult(result) {
  if (result?.failed) {
    return null;
  }
  const candidates = [
    result?.resourceTelemetry,
    result?.result?.resourceTelemetry,
    result?.result?.result?.resourceTelemetry,
  ];
  for (const candidate of candidates) {
    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
      return candidate;
    }
  }
  return null;
}

function resolvePeakRssBytes(resourceTelemetry) {
  return finiteStatNumber(resourceTelemetry?.process?.rssBytes?.max)
    ?? finiteStatNumber(resourceTelemetry?.process?.rssBytes);
}

function resolvePeakGpuMemoryBytes(resourceTelemetry) {
  return finiteStatNumber(resourceTelemetry?.gpu?.memoryUsedBytes?.max)
    ?? finiteStatNumber(resourceTelemetry?.gpu?.memoryUsedBytes);
}

function resolveMeanCpuPercent(resourceTelemetry) {
  return finiteStatNumber(resourceTelemetry?.process?.cpuPercent?.mean)
    ?? finiteStatNumber(resourceTelemetry?.process?.cpuPercent);
}

function resolveMeanGpuUtilizationPercent(resourceTelemetry) {
  return finiteStatNumber(resourceTelemetry?.gpu?.utilizationPercent?.mean)
    ?? finiteStatNumber(resourceTelemetry?.gpu?.utilizationPercent);
}

function resolveMeanPowerWatts(resourceTelemetry) {
  return finiteStatNumber(resourceTelemetry?.gpu?.powerWatts?.mean)
    ?? finiteStatNumber(resourceTelemetry?.gpu?.powerWatts);
}

function resolveDecodeTokensPerSecFromResult(result) {
  return firstFiniteStatNumber(result, [
    'result.timing.decodeTokensPerSec',
    'timing.decodeTokensPerSec',
    'result.metrics.decodeTokensPerSec',
    'metrics.decodeTokensPerSec',
    'result.result.timing.decodeTokensPerSec',
    'result.result.metrics.decodeTokensPerSec',
  ]);
}

function tokensPerSecPerGiB(tokensPerSec, bytes) {
  if (!Number.isFinite(tokensPerSec) || !Number.isFinite(bytes) || bytes <= 0) {
    return null;
  }
  return roundBottleneckNumber(tokensPerSec / (bytes / (1024 ** 3)));
}

function buildResourceEfficiencyMetrics(doppler, transformersjs) {
  const dopplerTelemetry = resolveResourceTelemetryFromResult(doppler);
  const tjsTelemetry = resolveResourceTelemetryFromResult(transformersjs);
  if (!dopplerTelemetry && !tjsTelemetry) {
    return null;
  }
  const dopplerDecode = resolveDecodeTokensPerSecFromResult(doppler);
  const tjsDecode = resolveDecodeTokensPerSecFromResult(transformersjs);
  const dopplerPeakRss = resolvePeakRssBytes(dopplerTelemetry);
  const tjsPeakRss = resolvePeakRssBytes(tjsTelemetry);
  const dopplerPeakGpuMemory = resolvePeakGpuMemoryBytes(dopplerTelemetry);
  const tjsPeakGpuMemory = resolvePeakGpuMemoryBytes(tjsTelemetry);
  return {
    schemaVersion: 1,
    units: {
      decodeTokensPerSecPerPeakRssGiB: 'tok/s/GiB RSS',
      decodeTokensPerSecPerPeakGpuGiB: 'tok/s/GiB GPU memory',
    },
    doppler: dopplerTelemetry
      ? {
          peakRssBytes: Number.isFinite(dopplerPeakRss) ? dopplerPeakRss : null,
          peakGpuMemoryBytes: Number.isFinite(dopplerPeakGpuMemory) ? dopplerPeakGpuMemory : null,
          meanCpuPercent: resolveMeanCpuPercent(dopplerTelemetry),
          meanGpuUtilizationPercent: resolveMeanGpuUtilizationPercent(dopplerTelemetry),
          meanPowerWatts: resolveMeanPowerWatts(dopplerTelemetry),
          decodeTokensPerSecPerPeakRssGiB: tokensPerSecPerGiB(dopplerDecode, dopplerPeakRss),
          decodeTokensPerSecPerPeakGpuGiB: tokensPerSecPerGiB(dopplerDecode, dopplerPeakGpuMemory),
        }
      : null,
    transformersjs: tjsTelemetry
      ? {
          peakRssBytes: Number.isFinite(tjsPeakRss) ? tjsPeakRss : null,
          peakGpuMemoryBytes: Number.isFinite(tjsPeakGpuMemory) ? tjsPeakGpuMemory : null,
          meanCpuPercent: resolveMeanCpuPercent(tjsTelemetry),
          meanGpuUtilizationPercent: resolveMeanGpuUtilizationPercent(tjsTelemetry),
          meanPowerWatts: resolveMeanPowerWatts(tjsTelemetry),
          decodeTokensPerSecPerPeakRssGiB: tokensPerSecPerGiB(tjsDecode, tjsPeakRss),
          decodeTokensPerSecPerPeakGpuGiB: tokensPerSecPerGiB(tjsDecode, tjsPeakGpuMemory),
        }
      : null,
    comparability: {
      processTree: dopplerTelemetry != null && tjsTelemetry != null,
      gpuTelemetry: dopplerTelemetry?.gpu != null && tjsTelemetry?.gpu != null,
      reason: dopplerTelemetry != null && tjsTelemetry != null
        ? 'Host process-tree RSS/CPU and system/GPU telemetry are sampled by the same parent benchmark tool; GPU memory is device-global when sourced from rocm-smi.'
        : 'Resource telemetry was not captured for both engines.',
    },
  };
}

function buildResourceTelemetryAccounting(doppler, transformersjs) {
  const dopplerTelemetry = resolveResourceTelemetryFromResult(doppler);
  const transformersjsTelemetry = resolveResourceTelemetryFromResult(transformersjs);
  if (!dopplerTelemetry && !transformersjsTelemetry) {
    return null;
  }
  return {
    schemaVersion: 1,
    doppler: dopplerTelemetry,
    transformersjs: transformersjsTelemetry,
    comparability: {
      processTree: dopplerTelemetry != null && transformersjsTelemetry != null,
      gpuTelemetry: dopplerTelemetry?.gpu != null && transformersjsTelemetry?.gpu != null,
      reason: 'Process-tree RSS/CPU and system RAM are comparable host-side samples. ROCm GPU utilization/memory is device-global and should be treated as run-context evidence, not strict per-process GPU residency.',
    },
  };
}

function resolveDopplerBottleneckClass(componentId) {
  if (
    componentId === 'submit_readback_wait'
    || componentId === 'submit_readback_slack'
    || componentId === 'submit_readback_unattributed'
    || componentId === 'readback_map_wait'
    || componentId === 'readback_cleanup'
    || componentId === 'readback_copy'
  ) {
    return 'submit-readback-wait';
  }
  if (componentId === 'command_record') {
    return 'command-recording';
  }
  if (componentId === 'gpu_compute') {
    return 'gpu-compute';
  }
  if (componentId === 'orchestration') {
    return 'orchestration';
  }
  if (componentId === 'unattributed') {
    return 'unattributed';
  }
  return null;
}

function resolveDopplerBottleneckFromResult(result) {
  if (result?.failed) {
    return null;
  }

  const decodeWallMs = firstFiniteStatNumber(result, [
    'result.timing.decodeMs',
    'timing.decodeMs',
    'result.metrics.decodeMs',
    'metrics.decodeMs',
    'result.result.timing.decodeMs',
    'result.result.metrics.decodeMs',
  ]);
  if (!Number.isFinite(decodeWallMs) || decodeWallMs <= 0) {
    return null;
  }

  const commandRecordMs = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordMs',
    'metrics.gpu.decodeRecordMs',
    'result.result.metrics.gpu.decodeRecordMs',
  ]);
  const commandRecordOps = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordOps',
    'metrics.gpu.decodeRecordOps',
    'result.result.metrics.gpu.decodeRecordOps',
  ]);
  const commandRecordPasses = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordPasses',
    'metrics.gpu.decodeRecordPasses',
    'result.result.metrics.gpu.decodeRecordPasses',
    'result.phase.gpu.decodeRecordPasses',
    'phase.gpu.decodeRecordPasses',
  ]);
  const commandRecordMsPerOp = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordMsPerOp',
    'metrics.gpu.decodeRecordMsPerOp',
    'result.result.metrics.gpu.decodeRecordMsPerOp',
  ]);
  const commandRecordMsPerPass = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordMsPerPass',
    'metrics.gpu.decodeRecordMsPerPass',
    'result.result.metrics.gpu.decodeRecordMsPerPass',
    'result.phase.gpu.decodeRecordMsPerPass',
    'phase.gpu.decodeRecordMsPerPass',
  ]);
  const commandRecordPassesPerOp = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordPassesPerOp',
    'metrics.gpu.decodeRecordPassesPerOp',
    'result.result.metrics.gpu.decodeRecordPassesPerOp',
    'result.phase.gpu.decodeRecordPassesPerOp',
    'phase.gpu.decodeRecordPassesPerOp',
  ]);
  const commandRecordMsPerExecutedBatchToken = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordMsPerExecutedBatchToken',
    'metrics.gpu.decodeRecordMsPerExecutedBatchToken',
    'result.result.metrics.gpu.decodeRecordMsPerExecutedBatchToken',
  ]);
  const commandRecordOpsPerExecutedBatchToken = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordOpsPerExecutedBatchToken',
    'metrics.gpu.decodeRecordOpsPerExecutedBatchToken',
    'result.result.metrics.gpu.decodeRecordOpsPerExecutedBatchToken',
  ]);
  const commandRecordPassesPerExecutedBatchToken = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordPassesPerExecutedBatchToken',
    'metrics.gpu.decodeRecordPassesPerExecutedBatchToken',
    'result.result.metrics.gpu.decodeRecordPassesPerExecutedBatchToken',
    'result.phase.gpu.decodeRecordPassesPerExecutedBatchToken',
    'phase.gpu.decodeRecordPassesPerExecutedBatchToken',
  ]);
  const commandRecordUniqueOpLabels = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeRecordUniqueOpLabels',
    'metrics.gpu.decodeRecordUniqueOpLabels',
    'result.result.metrics.gpu.decodeRecordUniqueOpLabels',
    'result.phase.gpu.decodeRecordUniqueOpLabels',
    'phase.gpu.decodeRecordUniqueOpLabels',
  ]);
  const commandRecordTopOps = normalizeCommandRecordTopOps(firstArrayValue(result, [
    'result.metrics.gpu.decodeRecordTopOps',
    'metrics.gpu.decodeRecordTopOps',
    'result.result.metrics.gpu.decodeRecordTopOps',
    'result.phase.gpu.decodeRecordTopOps',
    'phase.gpu.decodeRecordTopOps',
  ]));
  const submitWaitMs = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeSubmitWaitMs',
    'metrics.gpu.decodeSubmitWaitMs',
    'result.result.metrics.gpu.decodeSubmitWaitMs',
  ]);
  const readbackWaitMs = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeReadbackWaitMs',
    'metrics.gpu.decodeReadbackWaitMs',
    'result.result.metrics.gpu.decodeReadbackWaitMs',
  ]);
  const readbackMapWaitMs = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeReadbackMapWaitMs',
    'metrics.gpu.decodeReadbackMapWaitMs',
    'result.result.metrics.gpu.decodeReadbackMapWaitMs',
  ]);
  const readbackCleanupMs = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeReadbackCleanupMs',
    'metrics.gpu.decodeReadbackCleanupMs',
    'result.result.metrics.gpu.decodeReadbackCleanupMs',
  ]);
  const readbackCopyMs = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeReadbackCopyMs',
    'metrics.gpu.decodeReadbackCopyMs',
    'result.result.metrics.gpu.decodeReadbackCopyMs',
  ]);
  const orchestrationMs = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeOrchestrationMs',
    'metrics.gpu.decodeOrchestrationMs',
    'result.result.metrics.gpu.decodeOrchestrationMs',
  ]);
  const gpuTimestampMs = firstFiniteStatNumber(result, [
    'result.metrics.gpu.decodeMs',
    'metrics.gpu.decodeMs',
    'result.result.metrics.gpu.decodeMs',
  ]);

  const waitCandidates = [submitWaitMs, readbackWaitMs].filter(Number.isFinite);
  const effectiveSubmitReadbackWaitMs = waitCandidates.length > 0
    ? Math.max(...waitCandidates)
    : null;
  const submitReadbackSlackMs = Number.isFinite(effectiveSubmitReadbackWaitMs) && Number.isFinite(gpuTimestampMs)
    ? Math.max(0, effectiveSubmitReadbackWaitMs - gpuTimestampMs)
    : null;
  const readbackSubcomponentMs = [
    readbackMapWaitMs,
    readbackCleanupMs,
    readbackCopyMs,
  ]
    .filter(Number.isFinite)
    .reduce((sum, value) => sum + value, 0);
  const hasReadbackSubcomponents = readbackSubcomponentMs > 0;
  const readbackUnattributedMs = hasReadbackSubcomponents && Number.isFinite(effectiveSubmitReadbackWaitMs)
    ? Math.max(0, effectiveSubmitReadbackWaitMs - readbackSubcomponentMs)
    : null;
  const accountedMs = [
    commandRecordMs,
    effectiveSubmitReadbackWaitMs,
    orchestrationMs,
  ]
    .filter(Number.isFinite)
    .reduce((sum, value) => sum + value, 0);
  const residualMs = Math.max(0, decodeWallMs - accountedMs);
  const components = [
    {
      id: 'command_record',
      label: 'command recording',
      ms: commandRecordMs,
    },
    {
      id: Number.isFinite(gpuTimestampMs) ? 'submit_readback_slack' : 'submit_readback_wait',
      label: Number.isFinite(gpuTimestampMs) ? 'submit/readback slack' : 'submit/readback wait',
      ms: hasReadbackSubcomponents
        ? null
        : (Number.isFinite(gpuTimestampMs) ? submitReadbackSlackMs : effectiveSubmitReadbackWaitMs),
    },
    {
      id: 'readback_map_wait',
      label: 'readback map wait',
      ms: readbackMapWaitMs,
    },
    {
      id: 'readback_cleanup',
      label: 'readback cleanup',
      ms: readbackCleanupMs,
    },
    {
      id: 'readback_copy',
      label: 'readback CPU copy',
      ms: readbackCopyMs,
    },
    {
      id: 'submit_readback_unattributed',
      label: 'submit/readback unattributed',
      ms: readbackUnattributedMs,
    },
    {
      id: 'gpu_compute',
      label: 'GPU timestamp work',
      ms: gpuTimestampMs,
    },
    {
      id: 'orchestration',
      label: 'decode orchestration',
      ms: orchestrationMs,
    },
    {
      id: 'unattributed',
      label: 'unattributed decode wall',
      ms: residualMs,
    },
  ]
    .filter((component) => Number.isFinite(component.ms) && component.ms > 0);
  const dominant = components.length > 0
    ? components.reduce((best, component) => component.ms > best.ms ? component : best, components[0])
    : null;
  const dominantRecord = dominant
    ? {
        id: dominant.id,
        label: dominant.label,
        ms: roundBottleneckNumber(dominant.ms),
        shareOfDecode: shareOfDecode(dominant.ms, decodeWallMs),
      }
    : null;

  return {
    schemaVersion: 1,
    dominant: dominantRecord,
    bottleneckClass: dominantRecord ? resolveDopplerBottleneckClass(dominantRecord.id) : null,
    decodeWallMs: roundBottleneckNumber(decodeWallMs),
    componentsMs: {
      commandRecordMs: roundBottleneckNumber(commandRecordMs),
      submitWaitMs: roundBottleneckNumber(submitWaitMs),
      readbackWaitMs: roundBottleneckNumber(readbackWaitMs),
      effectiveSubmitReadbackWaitMs: roundBottleneckNumber(effectiveSubmitReadbackWaitMs),
      readbackMapWaitMs: roundBottleneckNumber(readbackMapWaitMs),
      readbackCleanupMs: roundBottleneckNumber(readbackCleanupMs),
      readbackCopyMs: roundBottleneckNumber(readbackCopyMs),
      readbackUnattributedMs: roundBottleneckNumber(readbackUnattributedMs),
      gpuTimestampMs: roundBottleneckNumber(gpuTimestampMs),
      submitReadbackSlackMs: roundBottleneckNumber(submitReadbackSlackMs),
      orchestrationMs: roundBottleneckNumber(orchestrationMs),
      residualMs: roundBottleneckNumber(residualMs),
    },
    recording: {
      opCount: roundBottleneckNumber(commandRecordOps),
      passCount: roundBottleneckNumber(commandRecordPasses),
      uniqueOpLabels: roundBottleneckNumber(commandRecordUniqueOpLabels),
      msPerOp: roundBottleneckNumber(commandRecordMsPerOp),
      msPerPass: roundBottleneckNumber(commandRecordMsPerPass),
      passesPerOp: roundBottleneckNumber(commandRecordPassesPerOp),
      msPerExecutedBatchToken: roundBottleneckNumber(commandRecordMsPerExecutedBatchToken),
      opsPerExecutedBatchToken: roundBottleneckNumber(commandRecordOpsPerExecutedBatchToken),
      passesPerExecutedBatchToken: roundBottleneckNumber(commandRecordPassesPerExecutedBatchToken),
      topOps: commandRecordTopOps,
      semantics: {
        opCount: 'Logical compute dispatches recorded into batch-decode command buffers.',
        passCount: 'Actual WebGPU compute passes opened while recording batch-decode command buffers.',
        uniqueOpLabels: 'Number of distinct exact compute-pass labels recorded in batch-decode command buffers.',
        msPerOp: 'decodeRecordMs divided by logical recorded compute dispatches.',
        msPerPass: 'decodeRecordMs divided by actual WebGPU compute passes.',
        passesPerOp: 'Actual WebGPU compute passes divided by logical recorded compute dispatches.',
        msPerExecutedBatchToken: 'decodeRecordMs divided by batch-path tokens submitted for execution.',
        opsPerExecutedBatchToken: 'Logical recorded compute dispatches divided by batch-path tokens submitted for execution.',
        passesPerExecutedBatchToken: 'Actual WebGPU compute passes divided by batch-path tokens submitted for execution.',
        topOps: 'Highest-count exact compute-pass labels observed during command recording.',
      },
    },
    shares: {
      commandRecord: shareOfDecode(commandRecordMs, decodeWallMs),
      submitWait: shareOfDecode(submitWaitMs, decodeWallMs),
      readbackWait: shareOfDecode(readbackWaitMs, decodeWallMs),
      effectiveSubmitReadbackWait: shareOfDecode(effectiveSubmitReadbackWaitMs, decodeWallMs),
      readbackMapWait: shareOfDecode(readbackMapWaitMs, decodeWallMs),
      readbackCleanup: shareOfDecode(readbackCleanupMs, decodeWallMs),
      readbackCopy: shareOfDecode(readbackCopyMs, decodeWallMs),
      readbackUnattributed: shareOfDecode(readbackUnattributedMs, decodeWallMs),
      gpuTimestamp: shareOfDecode(gpuTimestampMs, decodeWallMs),
      submitReadbackSlack: shareOfDecode(submitReadbackSlackMs, decodeWallMs),
      orchestration: shareOfDecode(orchestrationMs, decodeWallMs),
      residual: shareOfDecode(residualMs, decodeWallMs),
    },
    semantics: {
      effectiveSubmitReadbackWaitMs: 'max(decodeSubmitWaitMs, decodeReadbackWaitMs); submit and readback waits overlap.',
      readbackMapWaitMs: 'Wall time awaiting staging-buffer mapAsync; usually includes GPU completion behind the readback.',
      gpuTimestampMs: 'Timestamp-query GPU work when available; null means it was not captured.',
    },
  };
}

function divideOrNull(numerator, denominator) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) {
    return null;
  }
  return roundBottleneckNumber(numerator / denominator);
}

function resolveDopplerBatchAccountingFromResult(result) {
  if (result?.failed) {
    return null;
  }
  const requestedBatchTokens = firstFiniteStatNumber(result, [
    'result.metrics.batching.requestedBatchTokens',
    'metrics.batching.requestedBatchTokens',
    'result.result.metrics.batching.requestedBatchTokens',
  ]);
  const effectiveBatchTokens = firstFiniteStatNumber(result, [
    'result.metrics.batching.effectiveBatchTokens',
    'metrics.batching.effectiveBatchTokens',
    'result.result.metrics.batching.effectiveBatchTokens',
  ]);
  const executedBatchTokens = firstFiniteStatNumber(result, [
    'result.metrics.batching.executedBatchTokens',
    'metrics.batching.executedBatchTokens',
    'result.result.metrics.batching.executedBatchTokens',
  ]);
  const resolvedBatchTokens = firstFiniteStatNumber(result, [
    'result.metrics.batching.resolvedBatchTokens',
    'metrics.batching.resolvedBatchTokens',
    'result.result.metrics.batching.resolvedBatchTokens',
  ]);
  const outputDecodeTokens = firstFiniteStatNumber(result, [
    'result.metrics.avgDecodeTokens',
    'metrics.avgDecodeTokens',
    'result.result.metrics.avgDecodeTokens',
    'result.metrics.tokens.decode',
    'metrics.tokens.decode',
    'result.result.metrics.tokens.decode',
  ]);
  const gpuSubmissions = firstFiniteStatNumber(result, [
    'result.metrics.batching.gpuSubmissions',
    'metrics.batching.gpuSubmissions',
    'result.result.metrics.batching.gpuSubmissions',
  ]);
  if (
    !Number.isFinite(requestedBatchTokens)
    && !Number.isFinite(effectiveBatchTokens)
    && !Number.isFinite(executedBatchTokens)
    && !Number.isFinite(resolvedBatchTokens)
    && !Number.isFinite(outputDecodeTokens)
  ) {
    return null;
  }
  return {
    schemaVersion: 1,
    requestedBatchTokens: roundBottleneckNumber(requestedBatchTokens),
    effectiveBatchTokens: roundBottleneckNumber(effectiveBatchTokens),
    executedBatchTokens: roundBottleneckNumber(executedBatchTokens),
    resolvedBatchTokens: roundBottleneckNumber(resolvedBatchTokens),
    outputDecodeTokens: roundBottleneckNumber(outputDecodeTokens),
    gpuSubmissions: roundBottleneckNumber(gpuSubmissions),
    batchResolutionEfficiency: divideOrNull(resolvedBatchTokens, executedBatchTokens),
    outputEfficiency: divideOrNull(outputDecodeTokens, executedBatchTokens),
    batchOverrunTokens: Number.isFinite(executedBatchTokens) && Number.isFinite(resolvedBatchTokens)
      ? roundBottleneckNumber(Math.max(0, executedBatchTokens - resolvedBatchTokens))
      : null,
    outputOverrunTokens: Number.isFinite(executedBatchTokens) && Number.isFinite(outputDecodeTokens)
      ? roundBottleneckNumber(Math.max(0, executedBatchTokens - outputDecodeTokens))
      : null,
    semantics: {
      executedBatchTokens: 'Total GPU batch-path decode tokens recorded/submitted for execution.',
      resolvedBatchTokens: 'Batch-path tokens retained by stop resolution before returning to the generation loop.',
      outputDecodeTokens: 'Decode tokens credited to output throughput metrics.',
    },
  };
}

function extractDopplerDecodeTokensPerSecFromResult(result) {
  return firstFiniteNumber(result, [
    'result.timing.decodeTokensPerSec',
    'timing.decodeTokensPerSec',
    'result.metrics.decodeTokensPerSec',
    'metrics.decodeTokensPerSec',
    'result.result.timing.decodeTokensPerSec',
    'result.result.metrics.decodeTokensPerSec',
  ]);
}

function buildDopplerDiagnosticCoverage(sharedContract, batchAccounting) {
  const maxTokens = Number.isFinite(sharedContract?.maxTokens)
    ? Math.max(1, Math.floor(sharedContract.maxTokens))
    : null;
  const expectedBatchDecodeTokens = Number.isFinite(maxTokens)
    ? Math.max(0, maxTokens - 1)
    : null;
  const resolvedBatchTokens = Number.isFinite(batchAccounting?.resolvedBatchTokens)
    ? batchAccounting.resolvedBatchTokens
    : null;
  const executedBatchTokens = Number.isFinite(batchAccounting?.executedBatchTokens)
    ? batchAccounting.executedBatchTokens
    : null;
  const profileCoverage = Number.isFinite(resolvedBatchTokens)
    && Number.isFinite(expectedBatchDecodeTokens)
    && expectedBatchDecodeTokens > 0
      ? roundBottleneckNumber(resolvedBatchTokens / expectedBatchDecodeTokens)
      : null;
  const completeDecodeTarget = Number.isFinite(resolvedBatchTokens)
    && Number.isFinite(expectedBatchDecodeTokens)
    && resolvedBatchTokens >= expectedBatchDecodeTokens;

  return {
    schemaVersion: 1,
    maxTokens,
    expectedBatchDecodeTokens,
    executedBatchTokens,
    resolvedBatchTokens,
    profileCoverage,
    completeDecodeTarget,
    warning: completeDecodeTarget === false
      ? 'diagnostic-run-ended-before-source-decode-target'
      : null,
    semantics: {
      expectedBatchDecodeTokens: 'maxTokens - 1 because the first generated token is produced by the prefill logits path.',
      completeDecodeTarget: 'false means the diagnostic profiler captured a shorter generated path than the source compare section.',
    },
  };
}

function resolveDopplerThroughputCadenceGate({
  paritySection,
  throughputSection,
  policy = THROUGHPUT_CADENCE_PROMOTION_GATE,
}) {
  const thresholds = {
    requireBatchAccounting: policy.requireBatchAccounting === true,
    minDecodeTokensPerSecRatioVsParity: policy.minDecodeTokensPerSecRatioVsParity,
    minBatchResolutionEfficiency: policy.minBatchResolutionEfficiency,
    maxBatchOverrunTokens: policy.maxBatchOverrunTokens,
  };
  const parityDecodeTokensPerSec = extractDopplerDecodeTokensPerSecFromResult(paritySection?.doppler);
  const throughputDecodeTokensPerSec = extractDopplerDecodeTokensPerSecFromResult(throughputSection?.doppler);
  const decodeTokensPerSecRatioVsParity = Number.isFinite(parityDecodeTokensPerSec)
    && parityDecodeTokensPerSec > 0
    && Number.isFinite(throughputDecodeTokensPerSec)
      ? roundBottleneckNumber(throughputDecodeTokensPerSec / parityDecodeTokensPerSec)
      : null;
  const accounting = throughputSection?.dopplerBatchAccounting ?? null;
  const batchResolutionEfficiency = Number.isFinite(accounting?.batchResolutionEfficiency)
    ? accounting.batchResolutionEfficiency
    : null;
  const batchOverrunTokens = Number.isFinite(accounting?.batchOverrunTokens)
    ? accounting.batchOverrunTokens
    : null;
  const invalidReasons = [];

  if (paritySection && paritySection.pairedComparable !== true) {
    invalidReasons.push('parity-section-not-comparable');
  }
  if (throughputSection && throughputSection.pairedComparable !== true) {
    invalidReasons.push('throughput-section-not-comparable');
  }
  if (!Number.isFinite(parityDecodeTokensPerSec)) {
    invalidReasons.push('missing-parity-decode-tokens-per-sec');
  }
  if (!Number.isFinite(throughputDecodeTokensPerSec)) {
    invalidReasons.push('missing-throughput-decode-tokens-per-sec');
  }
  if (
    Number.isFinite(decodeTokensPerSecRatioVsParity)
    && decodeTokensPerSecRatioVsParity < thresholds.minDecodeTokensPerSecRatioVsParity
  ) {
    invalidReasons.push('throughput-decode-not-faster-than-parity');
  }
  if (!accounting && thresholds.requireBatchAccounting) {
    invalidReasons.push('missing-throughput-batch-accounting');
  }
  if (accounting && !Number.isFinite(batchResolutionEfficiency)) {
    invalidReasons.push('missing-throughput-batch-resolution-efficiency');
  }
  if (
    Number.isFinite(batchResolutionEfficiency)
    && batchResolutionEfficiency < thresholds.minBatchResolutionEfficiency
  ) {
    invalidReasons.push('throughput-batch-resolution-efficiency-below-threshold');
  }
  if (accounting && !Number.isFinite(batchOverrunTokens)) {
    invalidReasons.push('missing-throughput-batch-overrun-tokens');
  }
  if (Number.isFinite(batchOverrunTokens) && batchOverrunTokens > thresholds.maxBatchOverrunTokens) {
    invalidReasons.push('throughput-batch-overrun-exceeds-threshold');
  }

  return {
    schemaVersion: 1,
    ok: invalidReasons.length === 0,
    invalidReasons,
    thresholds,
    observed: {
      parityDecodeTokensPerSec: roundBottleneckNumber(parityDecodeTokensPerSec),
      throughputDecodeTokensPerSec: roundBottleneckNumber(throughputDecodeTokensPerSec),
      decodeTokensPerSecRatioVsParity,
      batchResolutionEfficiency,
      batchOverrunTokens,
      executedBatchTokens: accounting?.executedBatchTokens ?? null,
      resolvedBatchTokens: accounting?.resolvedBatchTokens ?? null,
      outputDecodeTokens: accounting?.outputDecodeTokens ?? null,
      parityInvalidReason: paritySection?.invalidReason ?? null,
      throughputInvalidReason: throughputSection?.invalidReason ?? null,
    },
    semantics: {
      ok: 'true means the throughput cadence is promotable as a claim lane; false means it remains tuning evidence.',
      minDecodeTokensPerSecRatioVsParity: 'throughput decode tok/s divided by parity decode tok/s.',
      batchResolutionEfficiency: 'resolved batch-path decode tokens divided by executed batch-path decode tokens.',
    },
  };
}

function buildDiagnosticSharedBenchmarkContract(sharedContract) {
  return {
    prompt: sharedContract.prompt,
    promptRaw: sharedContract.promptRaw,
    promptContract: sharedContract.promptContract,
    maxTokens: sharedContract.maxTokens,
    warmupRuns: 0,
    timedRuns: 1,
    seed: sharedContract.seed,
    sampling: sharedContract.sampling,
    useChatTemplate: sharedContract.useChatTemplate,
    loadMode: sharedContract.loadMode,
  };
}

function buildDopplerBottleneckDiagnosticRuntimeOverlay() {
  return {
    shared: {
      debug: {
        logOutput: {
          stdout: false,
          file: null,
        },
        logLevel: {
          defaultLogLevel: 'silent',
        },
        trace: {
          enabled: false,
        },
        profiler: {
          enabled: true,
          logEveryDecodeSteps: 1,
        },
      },
    },
  };
}

function resolveDopplerMetricsPayload(result) {
  const candidates = [
    result?.result?.metrics,
    result?.metrics,
    result?.result?.result?.metrics,
  ];
  for (const candidate of candidates) {
    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
      return candidate;
    }
  }
  return null;
}

function resolveDopplerDecodeProfileStepsFromResult(result) {
  const metrics = resolveDopplerMetricsPayload(result);
  const candidates = [
    metrics?.decodeProfileSteps,
    result?.result?.phase?.decodeProfileSteps,
    result?.phase?.decodeProfileSteps,
  ];
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      return candidate;
    }
  }
  return [];
}

function summarizeDopplerDecodeProfileSteps(steps, limit = 12) {
  const totals = new Map();
  let profileStepCount = 0;
  let profileOperationCount = 0;

  for (const step of Array.isArray(steps) ? steps : []) {
    const timings = step?.timings;
    if (!timings || typeof timings !== 'object' || Array.isArray(timings)) {
      continue;
    }
    profileStepCount += 1;
    for (const [operation, value] of Object.entries(timings)) {
      if (!Number.isFinite(value)) continue;
      profileOperationCount += 1;
      const existing = totals.get(operation) ?? {
        operation,
        operationKind: String(operation).split(':')[0],
        totalMs: 0,
        count: 0,
      };
      existing.totalMs += value;
      existing.count += 1;
      totals.set(operation, existing);
    }
  }

  const topOperations = [...totals.values()]
    .map((entry) => ({
      operation: entry.operation,
      operationKind: entry.operationKind,
      totalMs: roundBottleneckNumber(entry.totalMs),
      count: entry.count,
    }))
    .sort((a, b) => b.totalMs - a.totalMs)
    .slice(0, limit);
  const totalTimestampMs = roundBottleneckNumber(
    [...totals.values()].reduce((sum, entry) => sum + entry.totalMs, 0)
  );

  return {
    schemaVersion: 1,
    captured: profileStepCount > 0,
    profileStepCount,
    profileOperationCount,
    averageOperationsPerStep: profileStepCount > 0
      ? roundBottleneckNumber(profileOperationCount / profileStepCount)
      : null,
    totalTimestampMs,
    topOperations,
  };
}

function buildDopplerBottleneckDiagnostic({
  doppler,
  sourceSection = 'compute/throughput',
  sharedContract = null,
}) {
  const failed = doppler?.failed === true;
  const profile = summarizeDopplerDecodeProfileSteps(
    resolveDopplerDecodeProfileStepsFromResult(doppler)
  );
  const batchAccounting = resolveDopplerBatchAccountingFromResult(doppler);
  const coverage = buildDopplerDiagnosticCoverage(sharedContract, batchAccounting);
  const invalidReasons = [];
  if (failed) {
    invalidReasons.push(doppler?.error?.message || 'doppler-diagnostic-failed');
  }
  if (coverage.completeDecodeTarget === false) {
    invalidReasons.push(coverage.warning || 'diagnostic-run-ended-before-source-decode-target');
  }
  if (profile.captured !== true) {
    invalidReasons.push('missing-doppler-decode-profile-steps');
  }
  return {
    schemaVersion: 1,
    kind: 'doppler-bottleneck-profile',
    claimUse: 'diagnostic-only',
    sourceSection,
    failed,
    complete: failed === false && invalidReasons.length === 0,
    invalidReason: invalidReasons[0] ?? null,
    invalidReasons,
    command: {
      dopplerCommand: 'debug',
      workload: 'inference',
      profileEnabled: true,
      warmupRuns: sharedContract?.warmupRuns ?? null,
      timedRuns: sharedContract?.timedRuns ?? null,
      promptRenderedSha256: sharedContract?.promptContract?.promptRenderedSha256 ?? null,
      maxTokens: sharedContract?.maxTokens ?? null,
      sampling: sharedContract?.sampling ?? null,
      loadMode: sharedContract?.loadMode ?? null,
    },
    decodeTokensPerSec: roundBottleneckNumber(extractDopplerDecodeTokensPerSecFromResult(doppler)),
    firstTokenMs: roundBottleneckNumber(firstFiniteStatNumber(doppler, [
      'result.metrics.firstTokenMs',
      'metrics.firstTokenMs',
      'result.result.metrics.firstTokenMs',
    ])),
    modelLoadMs: roundBottleneckNumber(firstFiniteStatNumber(doppler, [
      'result.metrics.modelLoadMs',
      'metrics.modelLoadMs',
      'result.result.metrics.modelLoadMs',
    ])),
    dopplerDecodeCadence: resolveDopplerDecodeCadenceFromResult(doppler),
    dopplerBottleneck: resolveDopplerBottleneckFromResult(doppler),
    dopplerBatchAccounting: batchAccounting,
    coverage,
    profile,
    semantics: {
      claimUse: 'This diagnostic is not used for compare rows, correctness gates, or promoted throughput claims.',
      complete: 'true means the diagnostic covered the requested decode target and captured timestamp profile steps.',
      profile: 'Timestamp-query operation totals captured by a separate Doppler debug run with the same prompt, sampling, model artifact, load mode, and decode cadence as the source section.',
    },
  };
}

function assertCadenceField(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function assertDopplerDecodeCadence(result, expectedCadence, phaseLabel) {
  if (result?.failed) return;
  const runtimeConfig = resolveDopplerRuntimeConfigFromResult(result);
  if (!runtimeConfig) {
    throw new Error(`Doppler result missing runtimeConfig during ${phaseLabel}; cannot prove declared decode cadence`);
  }
  const batching = runtimeConfig.inference?.batching;
  const decodeLoop = runtimeConfig.inference?.session?.decodeLoop;
  const generation = runtimeConfig.inference?.generation;
  if (!batching || typeof batching !== 'object' || Array.isArray(batching)) {
    throw new Error(`Doppler result missing runtimeConfig.inference.batching during ${phaseLabel}`);
  }
  if (!decodeLoop || typeof decodeLoop !== 'object' || Array.isArray(decodeLoop)) {
    throw new Error(`Doppler result missing runtimeConfig.inference.session.decodeLoop during ${phaseLabel}`);
  }
  if (!generation || typeof generation !== 'object' || Array.isArray(generation)) {
    throw new Error(`Doppler result missing runtimeConfig.inference.generation during ${phaseLabel}`);
  }
  assertCadenceField(
    batching.batchSize,
    expectedCadence.batchSize,
    `Doppler declared cadence mismatch during ${phaseLabel}: inference.batching.batchSize`
  );
  assertCadenceField(
    batching.readbackInterval,
    expectedCadence.readbackInterval,
    `Doppler declared cadence mismatch during ${phaseLabel}: inference.batching.readbackInterval`
  );
  assertCadenceField(
    batching.stopCheckMode,
    expectedCadence.stopCheckMode,
    `Doppler declared cadence mismatch during ${phaseLabel}: inference.batching.stopCheckMode`
  );
  if (expectedCadence.readbackMode != null) {
    assertCadenceField(
      batching.readbackMode,
      expectedCadence.readbackMode,
      `Doppler declared cadence mismatch during ${phaseLabel}: inference.batching.readbackMode`
    );
  }
  assertCadenceField(
    decodeLoop.batchSize,
    expectedCadence.batchSize,
    `Doppler declared cadence mismatch during ${phaseLabel}: inference.session.decodeLoop.batchSize`
  );
  assertCadenceField(
    decodeLoop.readbackInterval,
    expectedCadence.readbackInterval,
    `Doppler declared cadence mismatch during ${phaseLabel}: inference.session.decodeLoop.readbackInterval`
  );
  assertCadenceField(
    decodeLoop.stopCheckMode,
    expectedCadence.stopCheckMode,
    `Doppler declared cadence mismatch during ${phaseLabel}: inference.session.decodeLoop.stopCheckMode`
  );
  if (expectedCadence.readbackMode != null) {
    assertCadenceField(
      decodeLoop.readbackMode,
      expectedCadence.readbackMode,
      `Doppler declared cadence mismatch during ${phaseLabel}: inference.session.decodeLoop.readbackMode`
    );
  }
  if (expectedCadence.maxBatchDecodeTokens !== undefined) {
    assertCadenceField(
      decodeLoop.maxBatchDecodeTokens ?? null,
      expectedCadence.maxBatchDecodeTokens,
      `Doppler declared cadence mismatch during ${phaseLabel}: inference.session.decodeLoop.maxBatchDecodeTokens`
    );
  }
  assertCadenceField(
    decodeLoop.disableCommandBatching,
    false,
    `Doppler declared cadence mismatch during ${phaseLabel}: inference.session.decodeLoop.disableCommandBatching`
  );
  assertCadenceField(
    generation.disableMultiTokenDecode,
    expectedCadence.disableMultiTokenDecode,
    `Doppler declared cadence mismatch during ${phaseLabel}: inference.generation.disableMultiTokenDecode`
  );
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

  for (const field of REQUIRED_CANONICAL_TIMING_FIELDS) {
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
  if (typeof result?.result?.result?.metrics?.generatedText === 'string') {
    return result.result.result.metrics.generatedText;
  }
  return result?.result?.metrics?.generatedText ?? result?.metrics?.generatedText ?? null;
}

function getTjsGeneratedText(result) {
  return result?.generatedText ?? result?.output ?? null;
}

function normalizeTokenIdArray(value) {
  if (!Array.isArray(value)) return null;
  const normalized = value
    .map((entry) => Number(entry))
    .filter((entry) => Number.isInteger(entry));
  return normalized.length === value.length ? normalized : null;
}

function getDopplerGeneratedTokenIds(result) {
  return normalizeTokenIdArray(firstArrayValue(result, [
    'result.metrics.referenceTranscript.tokens.ids',
    'metrics.referenceTranscript.tokens.ids',
    'result.result.metrics.referenceTranscript.tokens.ids',
    'result.result.result.metrics.referenceTranscript.tokens.ids',
    'result.referenceTranscript.tokens.ids',
    'referenceTranscript.tokens.ids',
    'result.metrics.generatedTokenIds',
    'metrics.generatedTokenIds',
    'generatedTokenIds',
  ]));
}

function getTjsGeneratedTokenIds(result) {
  return normalizeTokenIdArray(firstArrayValue(result, [
    'generatedTokenIds',
    'result.generatedTokenIds',
    'metrics.generatedTokenIds',
    'result.metrics.generatedTokenIds',
    'runs.0.generatedTokenIds',
    'result.runs.0.generatedTokenIds',
  ]));
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

function computeTokenIdPrefixMatch(leftIds, rightIds) {
  if (!Array.isArray(leftIds) || !Array.isArray(rightIds)) {
    return null;
  }
  const limit = Math.min(leftIds.length, rightIds.length);
  let prefix = 0;
  while (prefix < limit && leftIds[prefix] === rightIds[prefix]) {
    prefix += 1;
  }
  const mismatchIndex = prefix < limit
    ? prefix
    : (leftIds.length === rightIds.length ? -1 : limit);
  return {
    leftTokenCount: leftIds.length,
    rightTokenCount: rightIds.length,
    matchingPrefixTokens: prefix,
    firstMismatchTokenIndex: mismatchIndex,
    firstMismatch: mismatchIndex >= 0
      ? {
          doppler: leftIds[mismatchIndex] ?? null,
          transformersjs: rightIds[mismatchIndex] ?? null,
        }
      : null,
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
  const tokenIdMatch = computeTokenIdPrefixMatch(
    getDopplerGeneratedTokenIds(dopplerResult),
    getTjsGeneratedTokenIds(tjsResult)
  );
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
    tokenIdMatch,
  };
}

function buildOutputParitySummary(dopplerResult, tjsResult) {
  const dopplerFailed = dopplerResult?.failed === true;
  const tjsFailed = tjsResult?.failed === true;
  const dopplerText = getDopplerGeneratedText(dopplerResult);
  const tjsText = getTjsGeneratedText(tjsResult);
  if (dopplerFailed || tjsFailed) {
    return {
      schemaVersion: 1,
      status: 'unavailable',
      reason: 'engine-run-failed',
      exactMatch: null,
      normalizedMatch: null,
      charMismatchIndex: null,
      tokenMatch: null,
      tokenIdMatch: null,
      lengths: {
        dopplerChars: typeof dopplerText === 'string' ? dopplerText.length : null,
        transformersjsChars: typeof tjsText === 'string' ? tjsText.length : null,
      },
    };
  }
  if (dopplerText == null || tjsText == null) {
    return {
      schemaVersion: 1,
      status: 'unavailable',
      reason: 'missing-generated-text',
      exactMatch: null,
      normalizedMatch: null,
      charMismatchIndex: null,
      tokenMatch: null,
      tokenIdMatch: null,
      lengths: {
        dopplerChars: typeof dopplerText === 'string' ? dopplerText.length : null,
        transformersjsChars: typeof tjsText === 'string' ? tjsText.length : null,
      },
    };
  }
  const exactMatch = dopplerText === tjsText;
  const normalizedMatch = normalizeWhitespace(dopplerText) === normalizeWhitespace(tjsText);
  return {
    schemaVersion: 1,
    status: exactMatch === true || normalizedMatch === true ? 'match' : 'mismatch',
    reason: null,
    exactMatch,
    normalizedMatch,
    charMismatchIndex: firstMismatchIndex(dopplerText, tjsText),
    tokenMatch: computeTokenPrefixMatch(dopplerText, tjsText),
    tokenIdMatch: computeTokenIdPrefixMatch(
      getDopplerGeneratedTokenIds(dopplerResult),
      getTjsGeneratedTokenIds(tjsResult)
    ),
    lengths: {
      dopplerChars: dopplerText.length,
      transformersjsChars: tjsText.length,
    },
  };
}

function resolveOutputParityPolicy(sharedContract, compareProfile = null) {
  const temperature = Number(sharedContract?.sampling?.temperature);
  const deterministicGreedy = Number.isFinite(temperature) && temperature <= 0;
  const profilePolicy = compareProfile?.outputParityPolicy ?? null;
  if (profilePolicy && typeof profilePolicy === 'object' && !Array.isArray(profilePolicy)) {
    return {
      schemaVersion: 1,
      requireMatch: profilePolicy.requireMatch === true,
      matchMode: profilePolicy.matchMode ?? 'exact-or-normalized',
      reason: profilePolicy.reason ?? null,
      source: 'compare-profile',
      deterministicGreedy,
    };
  }
  return {
    schemaVersion: 1,
    requireMatch: deterministicGreedy,
    matchMode: 'exact-or-normalized',
    reason: deterministicGreedy
      ? 'deterministic-greedy-sampling'
      : 'stochastic-or-non-greedy-sampling',
    source: 'sampling-default',
    deterministicGreedy,
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
  if (unit === 'bytes') {
    if (v >= 1024 ** 3) return `${(v / (1024 ** 3)).toFixed(2)}GiB`;
    if (v >= 1024 ** 2) return `${(v / (1024 ** 2)).toFixed(1)}MiB`;
    if (v >= 1024) return `${(v / 1024).toFixed(1)}KiB`;
    return `${v}B`;
  }
  if (unit === 'tok/s') return `${v.toFixed(1)}`;
  return String(v);
}

function extractDopplerPrefillTokens(result) {
  return firstFiniteNumber(result, [
    'result.result.metrics.avgPrefillTokens',
    'result.metrics.avgPrefillTokens',
    'metrics.avgPrefillTokens',
    'result.result.tokens.prefill.median',
    'result.tokens.prefill.median',
    'tokens.prefill.median',
    'result.result.runs.0.prefillTokens',
    'result.runs.0.prefillTokens',
    'runs.0.prefillTokens',
  ]);
}

function extractTransformersjsPrefillTokens(result) {
  return firstFiniteNumber(result, [
    'runs.0.prefillTokens',
    'result.runs.0.prefillTokens',
    'metrics.prefillTokens',
    'result.metrics.prefillTokens',
    'timing.prefillTokens',
    'result.timing.prefillTokens',
  ]);
}

// Maximum absolute tokenizer divergence between Doppler's tokenizer and the
// Transformers.js tokenizer on a single compare-rendered prompt that we are
// willing to tolerate before marking a section non-comparable. Two independent
// tokenizers on the exact same byte sequence can legitimately disagree by a
// small amount (leading-space handling, BOS conventions, etc.). Larger deltas
// point at a real prompt-rendering drift and must stay strict.
const MAX_TOLERATED_TOKENIZER_DELTA = 1;

function resolvePromptTokenComparison({ prefillTokenTarget, doppler, transformersjs }) {
  const dopplerPrefillTokens = extractDopplerPrefillTokens(doppler);
  const transformersjsPrefillTokens = extractTransformersjsPrefillTokens(transformersjs);
  const target = Number.isFinite(prefillTokenTarget) ? Number(prefillTokenTarget) : null;

  const bothKnown =
    Number.isFinite(dopplerPrefillTokens) && Number.isFinite(transformersjsPrefillTokens);
  const tokenizerDelta = bothKnown
    ? Math.abs(dopplerPrefillTokens - transformersjsPrefillTokens)
    : null;
  const dopplerTargetDelta = target != null && Number.isFinite(dopplerPrefillTokens)
    ? Math.abs(dopplerPrefillTokens - target)
    : null;
  const transformersjsTargetDelta = target != null && Number.isFinite(transformersjsPrefillTokens)
    ? Math.abs(transformersjsPrefillTokens - target)
    : null;

  let strictOk = true;
  let invalidReason = null;
  if (dopplerPrefillTokens == null || transformersjsPrefillTokens == null) {
    strictOk = false;
    invalidReason = 'missing-prompt-token-count';
  } else if (tokenizerDelta !== 0) {
    strictOk = false;
    invalidReason = 'prompt-token-count-mismatch';
  } else if (target != null && dopplerPrefillTokens !== target) {
    strictOk = false;
    invalidReason = 'prompt-token-target-mismatch';
  }

  return {
    target,
    doppler: dopplerPrefillTokens,
    transformersjs: transformersjsPrefillTokens,
    tokenizerDelta,
    dopplerTargetDelta,
    transformersjsTargetDelta,
    ok: strictOk,
    invalidReason,
    // These fields are populated by buildCompareSection once the prompt
    // contract and decode validity are known. Strict-ok sections come in with
    // pairedComparable=true and no toleration record; sections that fail
    // strictly may still be relaxed to pairedComparable=true if every gate
    // passes and the delta is within MAX_TOLERATED_TOKENIZER_DELTA.
    pairedComparable: strictOk,
    toleratedTokenizerDelta: null,
  };
}

function extractDopplerDecodeTokens(result) {
  return firstFiniteNumber(result, [
    'result.result.metrics.avgDecodeTokens',
    'result.metrics.avgDecodeTokens',
    'metrics.avgDecodeTokens',
    'result.result.metrics.decodeTokens',
    'result.metrics.decodeTokens',
    'metrics.decodeTokens',
    'result.result.runs.0.phase.decodeTokens',
    'result.runs.0.phase.decodeTokens',
    'runs.0.phase.decodeTokens',
    'result.result.runs.0.decodeTokens',
    'result.runs.0.decodeTokens',
    'runs.0.decodeTokens',
  ]);
}

function extractTransformersjsDecodeTokens(result) {
  return firstFiniteNumber(result, [
    'runs.0.decodeTokens',
    'result.runs.0.decodeTokens',
    'metrics.avgDecodeTokens',
    'result.metrics.avgDecodeTokens',
    'metrics.decodeTokens',
    'result.metrics.decodeTokens',
  ]);
}

function resolveDecodeValidity({ doppler, transformersjs, maxTokens }) {
  const requested = Number(maxTokens);
  if (!Number.isFinite(requested) || requested <= 1) {
    return {
      ok: true,
      invalidReason: null,
      doppler: null,
      transformersjs: null,
    };
  }

  const dopplerDecodeTokens = extractDopplerDecodeTokens(doppler);
  const transformersjsDecodeTokens = extractTransformersjsDecodeTokens(transformersjs);
  const dopplerText = getDopplerGeneratedText(doppler);
  const transformersjsText = getTjsGeneratedText(transformersjs);
  let invalidReason = null;

  if (Number.isFinite(dopplerDecodeTokens) && dopplerDecodeTokens <= 0) {
    invalidReason = 'doppler-invalid-zero-decode-tokens';
  } else if (Number.isFinite(transformersjsDecodeTokens) && transformersjsDecodeTokens <= 0) {
    invalidReason = 'transformersjs-invalid-zero-decode-tokens';
  } else if (typeof dopplerText === 'string' && dopplerText.trim() === '') {
    invalidReason = 'doppler-invalid-empty-generated-text';
  } else if (typeof transformersjsText === 'string' && transformersjsText.trim() === '') {
    invalidReason = 'transformersjs-invalid-empty-generated-text';
  }

  return {
    ok: invalidReason == null,
    invalidReason,
    code: invalidReason == null ? null : 'INVALID_BENCHMARK_ZERO_DECODE_TOKENS',
    doppler: {
      decodeTokens: dopplerDecodeTokens,
      generatedTextEmpty: typeof dopplerText === 'string' ? dopplerText.trim() === '' : null,
    },
    transformersjs: {
      decodeTokens: transformersjsDecodeTokens,
      generatedTextEmpty: typeof transformersjsText === 'string' ? transformersjsText.trim() === '' : null,
    },
  };
}

function resolvePromptContractValidity(promptContract) {
  if (!promptContract) {
    return { ok: true, invalidReason: null };
  }
  if (promptContract.enginesReceiveRenderedPrompt !== true) {
    return {
      ok: false,
      invalidReason: 'prompt-rendering-not-shared',
    };
  }
  if (typeof promptContract.promptRendered !== 'string' || promptContract.promptRendered.length === 0) {
    return {
      ok: false,
      invalidReason: 'prompt-rendered-empty',
    };
  }
  return { ok: true, invalidReason: null };
}

function buildCompareSection({
  cacheMode,
  loadMode,
  doppler,
  transformersjs,
  prefillTokenTarget = null,
  maxTokens = null,
  promptContract = null,
  outputParityPolicy = null,
}) {
  const dopplerFailed = doppler?.failed === true;
  const tjsFailed = transformersjs?.failed === true;
  const baseComparable = !dopplerFailed && !tjsFailed && doppler != null && transformersjs != null;
  const promptTokens = resolvePromptTokenComparison({
    prefillTokenTarget,
    doppler,
    transformersjs,
  });
  const promptContractEvidence = promptContract
    ? {
        ...promptContract,
        promptTokenCount: promptTokens.ok === true ? promptTokens.doppler : null,
        promptTokenCountSource: promptTokens.ok === true ? 'paired-engine-prefill-tokens' : null,
      }
    : null;
  const promptContractValidity = resolvePromptContractValidity(promptContractEvidence);
  const decodeValidity = baseComparable
    ? resolveDecodeValidity({ doppler, transformersjs, maxTokens })
    : {
        ok: true,
        invalidReason: null,
        doppler: null,
        transformersjs: null,
      };
  const outputParity = baseComparable
    ? buildOutputParitySummary(doppler, transformersjs)
    : {
        schemaVersion: 1,
        status: 'unavailable',
        reason: 'engine-run-missing-or-failed',
        exactMatch: null,
        normalizedMatch: null,
        charMismatchIndex: null,
        tokenMatch: null,
        tokenIdMatch: null,
        lengths: {
          dopplerChars: null,
          transformersjsChars: null,
        },
      };
  const resolvedOutputParityPolicy = outputParityPolicy && typeof outputParityPolicy === 'object'
    ? {
        schemaVersion: outputParityPolicy.schemaVersion ?? 1,
        requireMatch: outputParityPolicy.requireMatch === true,
        matchMode: outputParityPolicy.matchMode ?? 'exact-or-normalized',
        reason: outputParityPolicy.reason ?? null,
        source: outputParityPolicy.source ?? null,
        deterministicGreedy: outputParityPolicy.deterministicGreedy ?? null,
      }
    : {
        schemaVersion: 1,
        requireMatch: false,
        matchMode: 'exact-or-normalized',
        reason: 'not-required-by-section',
        source: 'section-default',
        deterministicGreedy: null,
      };
  const outputParityOk = resolvedOutputParityPolicy.requireMatch !== true
    || outputParity.status === 'match';

  // Structured tokenizer-delta tolerance gate.
  //
  // When Doppler's tokenizer and Transformers.js's tokenizer produce counts
  // that differ by <= MAX_TOLERATED_TOKENIZER_DELTA on the same compare-
  // rendered prompt, the difference is noise in the tokenizer vocabularies,
  // not evidence of prompt contamination. We only relax the strict
  // pairedComparable check when every other axis of the compare contract is
  // already clean: compare-engines owns the rendered prompt (so both engines
  // received the exact same byte sequence), promptContractValidity passes
  // (promptRendered is non-empty and enginesReceiveRenderedPrompt=true), and
  // decodeValidity passes (both sides emitted non-zero decode tokens and
  // non-empty generated text). The relaxation is recorded on the prompt
  // tokens object so downstream readers can see exactly what was tolerated
  // and why.
  if (
    baseComparable
    && promptTokens.ok !== true
    && promptTokens.invalidReason === 'prompt-token-count-mismatch'
    && Number.isFinite(promptTokens.tokenizerDelta)
    && promptTokens.tokenizerDelta <= MAX_TOLERATED_TOKENIZER_DELTA
    && promptContractValidity.ok === true
    && promptContract?.enginesReceiveRenderedPrompt === true
    && decodeValidity.ok === true
  ) {
    const targetWithinTolerance =
      promptTokens.target == null
      || (
        Number.isFinite(promptTokens.dopplerTargetDelta)
        && Number.isFinite(promptTokens.transformersjsTargetDelta)
        && promptTokens.dopplerTargetDelta <= MAX_TOLERATED_TOKENIZER_DELTA
        && promptTokens.transformersjsTargetDelta <= MAX_TOLERATED_TOKENIZER_DELTA
      );
    if (targetWithinTolerance) {
      promptTokens.pairedComparable = true;
      promptTokens.toleratedTokenizerDelta = {
        delta: promptTokens.tokenizerDelta,
        maxAllowed: MAX_TOLERATED_TOKENIZER_DELTA,
        doppler: promptTokens.doppler,
        transformersjs: promptTokens.transformersjs,
        rationale: 'Different tokenizers on an identical compare-rendered prompt can disagree by a small amount. Relaxed because promptContract.enginesReceiveRenderedPrompt=true (compare owns rendering, both engines received the exact same promptRendered), promptContract is structurally valid, and decodeValidity is ok (both engines produced non-zero decode tokens and non-empty generated text). Strict invalidReason is preserved on promptTokens for audit.',
      };
    }
  }

  const pairedComparable = baseComparable
    && promptTokens.pairedComparable === true
    && promptContractValidity.ok === true
    && decodeValidity.ok === true
    && outputParityOk === true;

  // invalidReason cascade — report the most evidence-breaking failure first.
  // Engine failures dominate because they mean the lane never ran. Decode
  // validity comes next because a zero-decode or empty-text result tells you
  // the lane is broken before the tokenizer or rendering contract could even
  // matter. Prompt contract comes next because a non-shared rendered prompt
  // invalidates any per-engine token-count comparison. The tokenizer delta
  // comes last because it's the narrowest signal: two known tokenizers
  // disagreeing by a bounded amount on a shared rendered prompt.
  let invalidReason = null;
  if (!pairedComparable) {
    if (!baseComparable && dopplerFailed && tjsFailed) {
      invalidReason = 'doppler-and-transformersjs-failed';
    } else if (!baseComparable && dopplerFailed) {
      invalidReason = 'doppler-failed';
    } else if (!baseComparable && tjsFailed) {
      invalidReason = 'transformersjs-failed';
    } else if (!baseComparable) {
      invalidReason = 'missing-engine-result';
    } else if (decodeValidity.ok !== true) {
      invalidReason = decodeValidity.invalidReason || 'decode-contract-invalid';
    } else if (promptContractValidity.ok !== true) {
      invalidReason = promptContractValidity.invalidReason || 'prompt-contract-invalid';
    } else if (promptTokens.pairedComparable !== true) {
      invalidReason = promptTokens.invalidReason || 'prompt-token-contract-invalid';
    } else if (outputParityOk !== true) {
      invalidReason = outputParity.status === 'unavailable'
        ? (outputParity.reason || 'output-parity-unavailable')
        : 'output-parity-mismatch';
    } else {
      invalidReason = 'compare-contract-invalid';
    }
  }

  return {
    cacheMode,
    loadMode,
    pairedComparable,
    invalidReason,
    promptTokens,
    promptContract: promptContractEvidence,
    decodeValidity,
    outputParity,
    outputParityPolicy: resolvedOutputParityPolicy,
    dopplerDecodeCadence: resolveDopplerDecodeCadenceFromResult(doppler),
    dopplerBottleneck: resolveDopplerBottleneckFromResult(doppler),
    dopplerBatchAccounting: resolveDopplerBatchAccountingFromResult(doppler),
    memoryAccounting: buildMemoryAccounting(doppler, transformersjs),
    resourceTelemetry: buildResourceTelemetryAccounting(doppler, transformersjs),
    efficiencyMetrics: buildResourceEfficiencyMetrics(doppler, transformersjs),
    doppler: doppler ?? null,
    transformersjs: transformersjs ?? null,
  };
}

function classifyFormatFairness(dopplerFormat, tjsFormat) {
  const normalizedDopplerFormat = String(dopplerFormat || '').trim().toLowerCase() || null;
  const normalizedTjsFormat = String(tjsFormat || '').trim().toLowerCase() || null;
  const sameFormat = normalizedDopplerFormat != null
    && normalizedTjsFormat != null
    && normalizedDopplerFormat === normalizedTjsFormat;
  const neutral = sameFormat && normalizedDopplerFormat === 'safetensors';
  const optimized = normalizedDopplerFormat === 'rdrr' && normalizedTjsFormat === 'onnx';
  return {
    schemaVersion: 1,
    doppler: normalizedDopplerFormat,
    transformersjs: normalizedTjsFormat,
    sameFormat,
    class: neutral
      ? 'neutral-safetensors-vs-safetensors'
      : (optimized ? 'optimized-rdrr-vs-onnx' : (sameFormat ? 'same-format' : 'mixed-format')),
    disclosureRequired: sameFormat !== true,
    blocksClaim: false,
    reason: sameFormat === true
      ? 'same declared model format'
      : 'formats differ and must be disclosed with any claim',
  };
}

function evaluateFairnessSection({
  section,
  sectionId,
  compareLane,
  dopplerExecution,
  tjsModelOverridden,
  throughputCadenceGate = null,
}) {
  const invalidReasons = [];
  const gates = {
    performanceComparableLane: compareLane?.declared === 'performance_comparable'
      && compareLane?.allowNonComparableLane !== true,
    sameRuntimeSurface: dopplerExecution?.commandSurface === 'browser',
    tjsModelPinned: tjsModelOverridden !== true,
    pairedComparable: section?.pairedComparable === true,
    throughputCadencePromotable: sectionId === 'compute/throughput'
      ? throughputCadenceGate?.ok === true
      : null,
  };

  if (gates.performanceComparableLane !== true) {
    invalidReasons.push(
      compareLane?.allowNonComparableLane === true
        ? 'non-comparable-lane-override'
        : 'non-performance-comparable-lane'
    );
  }
  if (gates.sameRuntimeSurface !== true) {
    invalidReasons.push(
      `cross-surface-diagnostic:doppler-${dopplerExecution?.commandSurface ?? 'unknown'}-vs-transformersjs-browser`
    );
  }
  if (gates.tjsModelPinned !== true) {
    invalidReasons.push('transformersjs-model-overridden');
  }
  if (!section || typeof section !== 'object' || Array.isArray(section)) {
    invalidReasons.push('missing-section');
  } else if (section.pairedComparable !== true) {
    invalidReasons.push(section.invalidReason || 'section-not-paired-comparable');
  }
  if (sectionId === 'compute/throughput' && throughputCadenceGate?.ok !== true) {
    const gateReasons = Array.isArray(throughputCadenceGate?.invalidReasons)
      ? throughputCadenceGate.invalidReasons
      : [];
    if (gateReasons.length === 0) {
      invalidReasons.push('throughput-cadence-gate-failed');
    } else {
      for (const reason of gateReasons) {
        invalidReasons.push(`throughput-cadence:${reason}`);
      }
    }
  }

  return {
    schemaVersion: 1,
    sectionId,
    claimGrade: invalidReasons.length === 0,
    invalidReasons,
    gates,
    invalidReason: invalidReasons[0] ?? null,
  };
}

function buildCompareFairnessAudit({
  sections,
  compareLane,
  dopplerExecution,
  dopplerFormat,
  tjsFormat,
  tjsModelOverridden,
  dopplerModelSource,
}) {
  const compute = sections?.compute && typeof sections.compute === 'object' && !Array.isArray(sections.compute)
    ? sections.compute
    : null;
  const sectionAudits = {};
  if (compute?.parity) {
    sectionAudits['compute/parity'] = evaluateFairnessSection({
      section: compute.parity,
      sectionId: 'compute/parity',
      compareLane,
      dopplerExecution,
      tjsModelOverridden,
    });
  }
  if (compute?.throughput) {
    sectionAudits['compute/throughput'] = evaluateFairnessSection({
      section: compute.throughput,
      sectionId: 'compute/throughput',
      compareLane,
      dopplerExecution,
      tjsModelOverridden,
      throughputCadenceGate: compute.throughputCadenceGate,
    });
  }
  if (sections?.warm) {
    sectionAudits.warm = evaluateFairnessSection({
      section: sections.warm,
      sectionId: 'warm',
      compareLane,
      dopplerExecution,
      tjsModelOverridden,
    });
  }
  if (sections?.cold) {
    sectionAudits.cold = evaluateFairnessSection({
      section: sections.cold,
      sectionId: 'cold',
      compareLane,
      dopplerExecution,
      tjsModelOverridden,
    });
  }

  const primarySection = sectionAudits['compute/parity']
    ?? sectionAudits.warm
    ?? sectionAudits.cold
    ?? null;
  const primaryInvalidReasons = primarySection?.invalidReasons ?? ['missing-primary-section'];
  const claimGrade = primarySection?.claimGrade === true;
  const releaseClaimable = claimGrade === true && dopplerModelSource?.source === 'quickstart-registry';
  const localComparable = claimGrade === true && releaseClaimable !== true;
  const correctnessOk = primarySection?.gates?.pairedComparable === true;

  return {
    schemaVersion: 1,
    claimGrade,
    releaseClaimable,
    localComparable,
    correctnessOk,
    primarySection: primarySection?.sectionId ?? null,
    invalidReason: primaryInvalidReasons[0] ?? null,
    invalidReasons: primaryInvalidReasons,
    formatFairness: classifyFormatFairness(dopplerFormat, tjsFormat),
    surfaceFairness: {
      schemaVersion: 1,
      dopplerRequestedSurface: dopplerExecution?.requestedSurface ?? null,
      dopplerCommandSurface: dopplerExecution?.commandSurface ?? null,
      transformersjsSurface: 'browser',
      sameRuntimeSurface: dopplerExecution?.commandSurface === 'browser',
      blocksClaim: dopplerExecution?.commandSurface !== 'browser',
    },
    sections: sectionAudits,
    semantics: {
      claimGrade: 'true means the primary compare section passed its configured output-parity policy, prompt/decode validity, same runtime surface, pinned comparator model, and performance-comparable lane gates.',
      releaseClaimable: 'true means claimGrade evidence measured the hosted quickstart-registry Doppler artifact.',
      localComparable: 'true means claimGrade evidence exists but is local-artifact evidence, not a hosted release claim.',
      formatFairness: 'different optimized formats are allowed only as disclosed product-format comparisons, not format-identical engine claims.',
    },
  };
}

function normalizeFairnessCompareLane(compareLane) {
  if (isJsonObject(compareLane)) {
    return {
      ...compareLane,
      declared: typeof compareLane.declared === 'string' ? compareLane.declared : null,
      reason: typeof compareLane.reason === 'string' ? compareLane.reason : null,
      allowNonComparableLane: compareLane.allowNonComparableLane === true,
    };
  }
  return {
    declared: typeof compareLane === 'string' ? compareLane : null,
    reason: null,
    allowNonComparableLane: false,
  };
}

function resolveReportFormatFairnessInput(report) {
  const methodologyFormats = isJsonObject(report?.methodology?.formats)
    ? report.methodology.formats
    : {};
  return {
    dopplerFormat: typeof report?.dopplerExecution?.format === 'string'
      ? report.dopplerExecution.format
      : (typeof methodologyFormats.doppler === 'string' ? methodologyFormats.doppler : null),
    tjsFormat: typeof methodologyFormats.transformersjs === 'string'
      ? methodologyFormats.transformersjs
      : null,
  };
}

function resolveReportDopplerExecution(report) {
  const current = isJsonObject(report?.dopplerExecution) ? report.dopplerExecution : {};
  const fallbackSurface = typeof report?.dopplerSurface === 'string' ? report.dopplerSurface : null;
  const { dopplerFormat } = resolveReportFormatFairnessInput(report);
  let resolved = {};
  if (fallbackSurface) {
    try {
      resolved = resolveDopplerExecutionIdentity(
        fallbackSurface,
        dopplerFormat || DEFAULT_DOPPLER_FORMAT
      );
    } catch {
      resolved = {};
    }
  }
  return {
    ...current,
    requestedSurface: typeof current.requestedSurface === 'string'
      ? current.requestedSurface
      : (resolved.requestedSurface ?? fallbackSurface),
    commandSurface: typeof current.commandSurface === 'string'
      ? current.commandSurface
      : (resolved.commandSurface ?? fallbackSurface),
    cliExecutor: typeof current.cliExecutor === 'string'
      ? current.cliExecutor
      : (resolved.cliExecutor ?? null),
    format: typeof current.format === 'string'
      ? current.format
      : (resolved.format ?? dopplerFormat ?? null),
    commandSurfaceReason: typeof current.commandSurfaceReason === 'string'
      ? current.commandSurfaceReason
      : (resolved.commandSurfaceReason ?? null),
  };
}

function attachCompareFairnessAudit(report, options = {}) {
  if (!isJsonObject(report)) {
    throw new Error('compare fairness report must be an object');
  }
  const compareLane = normalizeFairnessCompareLane(report.compareLane);
  const dopplerExecution = options.dopplerExecution || resolveReportDopplerExecution(report);
  const { dopplerFormat, tjsFormat } = options.dopplerFormat || options.tjsFormat
    ? {
        dopplerFormat: options.dopplerFormat ?? null,
        tjsFormat: options.tjsFormat ?? null,
      }
    : resolveReportFormatFairnessInput(report);
  const tjsModelOverridden = options.tjsModelOverridden ?? (
    report.transformersjsModelSource === 'flag'
  );
  const dopplerModelSource = options.dopplerModelSource || (
    isJsonObject(report.dopplerModelSource) ? report.dopplerModelSource : null
  );

  report.dopplerExecution = dopplerExecution;
  report.fairness = buildCompareFairnessAudit({
    sections: report.sections,
    compareLane,
    dopplerExecution,
    dopplerFormat,
    tjsFormat,
    tjsModelOverridden,
    dopplerModelSource,
  });
  report.compareLane = {
    ...compareLane,
    lane: compareLane.declared,
    claimGrade: report.fairness.claimGrade,
    claimable: report.fairness.releaseClaimable,
    releaseClaimable: report.fairness.releaseClaimable,
    localComparable: report.fairness.localComparable,
    correctnessOk: report.fairness.correctnessOk,
    fairnessInvalidReason: report.fairness.invalidReason,
    fairnessInvalidReasons: report.fairness.invalidReasons,
  };
  return report;
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

function formatBottleneckShare(value) {
  if (!Number.isFinite(value)) {
    return '-';
  }
  return `${(value * 100).toFixed(0)}%`;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) {
    return '-';
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return String(roundBottleneckNumber(value));
}

function printSection(title, section, rows) {
  const dopplerResult = section?.doppler ?? null;
  const tjsResult = section?.transformersjs ?? section?.tjs ?? null;
  console.log(`\n=== ${title} ===`);
  if (section?.cacheMode || section?.loadMode) {
    console.log(
      `  mode: ${section?.cacheMode ?? '-'}`
      + `, load: ${section?.loadMode ?? '-'}`
    );
  }
  if (section?.pairedComparable === false) {
    console.log(`  note: paired comparison invalid (${section.invalidReason || 'unknown'})`);
  }
  if (dopplerResult?.failed === true) {
    console.log(`  Doppler failed: ${dopplerResult?.error?.message || 'failed'}`);
  }
  if (tjsResult?.failed === true) {
    console.log(`  TJS failed:     ${tjsResult?.error?.message || 'failed'}`);
  }
  if (section?.dopplerBottleneck?.dominant) {
    const dominant = section.dopplerBottleneck.dominant;
    console.log(
      `  Doppler bottleneck: ${dominant.label} `
      + `${formatVal(dominant.ms, 'ms')} (${formatBottleneckShare(dominant.shareOfDecode)} of decode wall)`
    );
  }
  if (section?.dopplerBottleneck?.recording?.opCount != null) {
    const recording = section.dopplerBottleneck.recording;
    console.log(
      `  Doppler command recording: ops=${formatNumber(recording.opCount)} `
      + `passes=${formatNumber(recording.passCount)} `
      + `ms/op=${formatNumber(recording.msPerOp)} `
      + `ms/pass=${formatNumber(recording.msPerPass)} `
      + `ms/executedToken=${formatNumber(recording.msPerExecutedBatchToken)}`
    );
    if (Array.isArray(recording.topOps) && recording.topOps.length > 0) {
      const topOps = recording.topOps.slice(0, 3)
        .map((entry) => `${entry.label}=${formatNumber(entry.count)}`)
        .join(', ');
      console.log(`  Doppler recorded top ops: ${topOps}`);
    }
  }
  if (section?.dopplerBatchAccounting?.executedBatchTokens != null) {
    const accounting = section.dopplerBatchAccounting;
    console.log(
      `  Doppler batch work: executed=${formatNumber(accounting.executedBatchTokens)} `
      + `resolved=${formatNumber(accounting.resolvedBatchTokens)} `
      + `output=${formatNumber(accounting.outputDecodeTokens)}`
    );
  }
  if (section?.efficiencyMetrics) {
    const dopplerEfficiency = section.efficiencyMetrics.doppler || {};
    const tjsEfficiency = section.efficiencyMetrics.transformersjs || {};
    console.log(
      `  Resource efficiency: RSS tok/s/GiB Doppler=${formatNumber(dopplerEfficiency.decodeTokensPerSecPerPeakRssGiB)} `
      + `TJS=${formatNumber(tjsEfficiency.decodeTokensPerSecPerPeakRssGiB)}, `
      + `peak RSS Doppler=${formatVal(dopplerEfficiency.peakRssBytes, 'bytes')} `
      + `TJS=${formatVal(tjsEfficiency.peakRssBytes, 'bytes')}`
    );
  }
  console.log(`  ${''.padEnd(20)} ${'Doppler'.padStart(14)} ${'TJS'.padStart(14)} ${'delta'.padStart(18)}`);
  for (const row of rows) {
    const dVal = resolveMetric(dopplerResult, row, 'doppler');
    const tVal = resolveMetric(tjsResult, row, 'transformersjs');
    printRow(row.label, dVal, tVal, row.unit, row.higherBetter);
  }
}

function printThroughputCadenceGate(gate) {
  if (!gate || typeof gate !== 'object') return;
  const status = gate.ok === true ? 'promotable' : 'tuning-only';
  const observed = gate.observed || {};
  const reasons = Array.isArray(gate.invalidReasons) && gate.invalidReasons.length > 0
    ? ` (${gate.invalidReasons.join(', ')})`
    : '';
  console.log(
    `\n=== THROUGHPUT CADENCE GATE ===\n`
    + `  status: ${status}${reasons}\n`
    + `  decode ratio vs parity: ${formatNumber(observed.decodeTokensPerSecRatioVsParity)}\n`
    + `  batch resolution efficiency: ${formatBottleneckShare(observed.batchResolutionEfficiency)}\n`
    + `  batch overrun tokens: ${formatNumber(observed.batchOverrunTokens)}`
  );
}

function printDopplerBottleneckDiagnostic(diagnostic) {
  if (!diagnostic || typeof diagnostic !== 'object') return;
  const bottleneck = diagnostic.dopplerBottleneck?.dominant;
  const profile = diagnostic.profile || {};
  const coverage = diagnostic.coverage || {};
  const topOperation = Array.isArray(profile.topOperations) ? profile.topOperations[0] : null;
  const status = diagnostic.failed === true
    ? 'failed'
    : (coverage.completeDecodeTarget === false ? 'partial' : 'captured');
  const lines = [
    '',
    '=== DOPPLER BOTTLENECK DIAGNOSTIC ===',
    `  status: ${status}`,
    `  source section: ${diagnostic.sourceSection ?? '-'}`,
    `  claim use: ${diagnostic.claimUse ?? 'diagnostic-only'}`,
    `  decode tok/s: ${formatNumber(diagnostic.decodeTokensPerSec)}`,
  ];
  if (diagnostic.invalidReason) {
    lines.push(`  invalid reason: ${diagnostic.invalidReason}`);
  }
  if (coverage.completeDecodeTarget === false) {
    lines.push(
      `  decode target coverage: ${formatBottleneckShare(coverage.profileCoverage)} `
      + `(${formatNumber(coverage.resolvedBatchTokens)}/${formatNumber(coverage.expectedBatchDecodeTokens)} batch tokens)`
    );
  }
  if (bottleneck) {
    lines.push(
      `  dominant wall component: ${bottleneck.label} `
      + `${formatVal(bottleneck.ms, 'ms')} (${formatBottleneckShare(bottleneck.shareOfDecode)} of decode wall)`
    );
  }
  if (profile.captured === true) {
    lines.push(
      `  timestamped GPU work: ${formatVal(profile.totalTimestampMs, 'ms')} `
      + `across ${formatNumber(profile.profileOperationCount)} profiled ops`
    );
  }
  if (topOperation) {
    lines.push(
      `  top timestamp op: ${topOperation.operation} ${formatVal(topOperation.totalMs, 'ms')}`
    );
  }
  const topRecordedOps = diagnostic.dopplerBottleneck?.recording?.topOps;
  if (Array.isArray(topRecordedOps) && topRecordedOps.length > 0) {
    lines.push(
      `  top recorded ops: ${topRecordedOps.slice(0, 3)
        .map((entry) => `${entry.label}=${formatNumber(entry.count)}`)
        .join(', ')}`
    );
  }
  console.log(lines.join('\n'));
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
    repetitionPenalty: Number.isFinite(Number(workloadSampling.repetitionPenalty))
      ? Number(workloadSampling.repetitionPenalty)
      : DEFAULT_SHARED_SAMPLING.repetitionPenalty,
    greedyThreshold: Number.isFinite(Number(workloadSampling.greedyThreshold))
      ? Number(workloadSampling.greedyThreshold)
      : DEFAULT_SHARED_SAMPLING.greedyThreshold,
  };
  let promptInput = flags.prompt || null;
  const maxTokensInput = flags['max-tokens'] ?? workload?.decodeTokens ?? DEFAULT_MAX_TOKENS;

  const compareProfileConfig = await loadCompareEnginesConfig(flags['compare-config']);
  const defaultDopplerModelId = resolveDefaultDopplerModelId(compareProfileConfig);
  const dopplerModelId = flags['model-id'] || defaultDopplerModelId;
  const compareProfile = resolveCompareProfile(compareProfileConfig, dopplerModelId);
  const compareProfileRuntimeBundles = await loadCompareProfileRuntimeBundles(compareProfile);
  const modelCatalog = await loadModelCatalogBundle();
  const catalogCompareBenchmark = resolveCatalogTransformersjsBenchmarkTarget(modelCatalog, dopplerModelId, null);
  if (
    compareProfile.defaultTjsModelId
    && catalogCompareBenchmark?.repoId
    && compareProfile.defaultTjsModelId !== catalogCompareBenchmark.repoId
  ) {
    throw new Error(
      `compare-engines.config.json defaultTjsModelId for "${dopplerModelId}" `
      + `(${compareProfile.defaultTjsModelId}) does not match models/catalog.json vendorBenchmark `
      + `(${catalogCompareBenchmark.repoId})`
    );
  }
  const allowNonComparableLane = flags['allow-non-comparable-lane'] === true;
  if (compareProfile.compareLane !== 'performance_comparable' && !allowNonComparableLane) {
    throw new Error(
      `Model "${dopplerModelId}" is classified as ${compareProfile.compareLane} in compare-engines.config.json. `
      + `${compareProfile.compareLaneReason || 'This lane is not apples-to-apples performance evidence.'} `
      + 'Use --allow-non-comparable-lane to run it as a diagnostic/support-only lane.'
    );
  }
  const compareProfileDefaultDopplerSurface = compareProfile.defaultDopplerSurface || 'auto';
  const dopplerSurface = parseChoice(
    flags['doppler-surface'],
    DOPPLER_SURFACES,
    '--doppler-surface',
    compareProfileDefaultDopplerSurface
  );
  const dopplerNoOpfsCache = flags['doppler-no-opfs-cache'] === true;
  if (dopplerNoOpfsCache && !isDopplerBrowserCommandSurface(dopplerSurface)) {
    throw new Error('--doppler-no-opfs-cache is browser-only; use --doppler-surface browser|auto or remove this flag.');
  }
  const mode = flags.mode || 'all';
  const validModes = ['compute', 'cold', 'warm', 'all'];
  if (!validModes.includes(mode)) {
    console.error(`Invalid --mode "${mode}". Must be one of: ${validModes.join(', ')}`);
    process.exit(1);
  }
  const needCompute = mode === 'compute' || mode === 'all';
  const needWarm = mode === 'warm' || mode === 'all';
  const needCold = mode === 'cold' || mode === 'all';
  const dopplerBottleneckProfile = parseOnOff(
    flags['doppler-bottleneck-profile'],
    false,
    '--doppler-bottleneck-profile'
  );
  if (dopplerBottleneckProfile && !needCompute) {
    throw new Error('--doppler-bottleneck-profile requires --mode compute or --mode all.');
  }
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

  const dopplerFormat = parseChoice(
    flags['doppler-format'],
    DOPPLER_FORMATS,
    '--doppler-format',
    compareProfile.defaultDopplerFormat
  );
  const tjsFormat = parseChoice(
    flags['tjs-format'],
    TJS_FORMATS,
    '--tjs-format',
    compareProfile.defaultTjsFormat
  );
  const flagLoadMode = parseLoadMode(flags['load-mode'], '--load-mode', null);
  const sharedLoadMode = flagLoadMode ?? compareProfile.defaultLoadMode ?? null;
  const sharedLoadModeSource = flagLoadMode != null
    ? 'flag'
    : (compareProfile.defaultLoadMode != null ? 'compare-profile' : null);
  const resolvedLoadModes = resolveCompareLoadModes(sharedLoadMode, compareProfileConfig.defaults);
  assertDopplerLoadModeSupportedForFormat(dopplerFormat, resolvedLoadModes.warm);
  assertDopplerLoadModeSupportedForFormat(dopplerFormat, resolvedLoadModes.cold);
  const safetensorsSourceId = flags['safetensors-source'] || compareProfile.safetensorsSourceId || null;
  const needsSafetensors = dopplerFormat === 'safetensors' || tjsFormat === 'safetensors';
  if (needsSafetensors && !safetensorsSourceId) {
    throw new Error(
      'SafeTensors format requires --safetensors-source <hf-model-id> or safetensorsSourceId in the model profile.'
    );
  }
  const modelIndex = needsSafetensors ? loadModelIndex() : null;
  const safetensorsSourcePath = needsSafetensors
    ? (flags['safetensors-path'] || resolveFormatSourcePath(modelIndex, 'safetensors', safetensorsSourceId))
    : null;
  if (dopplerFormat === 'safetensors' && !safetensorsSourcePath) {
    throw new Error(
      `Cannot resolve SafeTensors source path for "${safetensorsSourceId}". `
      + `Check MODEL_INDEX.json on the external volume (${MODEL_INDEX_PATH}) or pass --safetensors-path.`
    );
  }
  const runtimeConfigOverride = parseRuntimeConfigJson(flags['runtime-config-json']);
  assertAllowedCompareRuntimeConfigOverride(runtimeConfigOverride);
  const runtimeConfigOverrideHash = runtimeConfigOverride == null
    ? null
    : hashJsonPayload(runtimeConfigOverride);
  const dopplerModelSource = dopplerFormat === 'safetensors'
    ? {
        source: 'safetensors',
        modelUrl: safetensorsSourcePath,
        locator: safetensorsSourceId,
        registrySource: null,
        modelBaseDir: null,
        manifestSourceType: null,
        manifestSource: null,
      }
    : await resolveDopplerModelSource(compareProfile, dopplerModelId, flags['model-url']);
  const enforceLocalManifestFreshness = dopplerModelSource.manifestSourceType === 'remote'
    && compareProfile.compareLane === 'performance_comparable'
    && allowNonComparableLane !== true;
  const dopplerManifestPreflight = dopplerFormat === 'safetensors'
    ? null
    : await preflightDopplerManifestContract(
      dopplerModelSource,
      dopplerModelId,
      { enforceLocalManifestFreshness }
    );
  if (dopplerManifestPreflight && dopplerManifestPreflight.ok !== true) {
    throw new Error(
      `Doppler compare preflight failed for "${dopplerModelId}" from ${dopplerManifestPreflight.manifestSource}:\n`
      + formatPreflightErrors(dopplerManifestPreflight.errors)
    );
  }
  const resolvedDefaultTjsModel = compareProfile.defaultTjsModelId || catalogCompareBenchmark?.repoId || null;
  const tjsModelOverridden = flags['tjs-model'] != null;
  const tjsModelId = flags['tjs-model']
    || (tjsFormat === 'safetensors' ? safetensorsSourceId : resolvedDefaultTjsModel);
  if (!tjsModelId) {
    throw new Error(
      `No default Transformers.js model mapping for "${dopplerModelId}". `
      + 'Set defaultTjsModelId in compare-engines.config.json, add vendorBenchmark.transformersjs.repoId '
      + 'to models/catalog.json, or pass --tjs-model.'
    );
  }
  const tjsLocalModelPath = flags['tjs-local-model-path'] || null;
  const tjsVersion = flags['tjs-version'] || '4';
  const tjsRuntimeStack = await resolveTransformersjsRuntimeStack(tjsVersion);
  const catalogResolvedTjsBenchmark = resolveCatalogTransformersjsBenchmarkTarget(
    modelCatalog,
    dopplerModelId,
    tjsFormat === 'onnx' ? tjsModelId : null
  );
  const tjsModelSource = tjsModelOverridden === true
    ? 'flag'
    : (compareProfile.defaultTjsModelId ? 'compare-profile' : (catalogCompareBenchmark?.source || 'fallback'));
  const tjsDtype = parseChoice(
    flags['tjs-dtype'],
    TJS_DTYPES,
    '--tjs-dtype',
    catalogResolvedTjsBenchmark?.dtype || DEFAULT_TJS_DTYPE
  );
  const tjsDtypeSource = flags['tjs-dtype'] != null
    ? 'flag'
    : (catalogResolvedTjsBenchmark?.source || 'fallback');
  const useChatTemplate = parseOnOff(
    flags['use-chat-template'],
    compareProfile.defaultUseChatTemplate ?? false,
    '--use-chat-template'
  );
  let promptContract = null;
  if (promptInput == null) {
    if (Number.isFinite(prefillTokenTarget)) {
      promptContract = await resolveSyntheticPromptForModel({
        prefillTokens: prefillTokenTarget,
        modelId: tjsModelId,
        localModelPath: tjsLocalModelPath,
        useChatTemplate,
        renderPrompt: resolveCompareOwnedPromptRenderer({
          modelId: dopplerModelId,
          useChatTemplate,
        }),
      });
      promptInput = promptContract.prompt;
    } else {
      promptInput = DEFAULT_PROMPT;
    }
  } else if (Number.isFinite(prefillTokenTarget)) {
    promptContract = {
      prompt: promptInput,
      prefillTokens: prefillTokenTarget,
      promptSource: 'explicit-prompt',
      tokenizerLocator: null,
      tokenizerResolutionSource: null,
    };
  }
  const effectivePrefillTokenTarget = promptContract?.promptSource === 'tokenizer-accurate-synthetic'
    ? promptContract.prefillTokens
    : null;
  const promptRenderContract = renderComparePrompt({
    modelId: dopplerModelId,
    prompt: promptInput,
    useChatTemplate,
  });
  const sharedContract = buildSharedBenchmarkContract({
    prompt: promptRenderContract.promptRendered,
    maxTokens: maxTokensInput,
    warmupRuns: flags.warmup,
    timedRuns: flags.runs,
    seed: flags.seed,
    loadMode: sharedLoadMode,
    sampling,
    useChatTemplate: promptRenderContract.enginesReceiveRenderedPrompt ? false : useChatTemplate,
    promptContract: promptRenderContract,
  });
  const prompt = sharedContract.promptRaw;
  const maxTokens = sharedContract.maxTokens;
  const warmupRuns = sharedContract.warmupRuns;
  const runs = sharedContract.timedRuns;
  const seed = sharedContract.seed;
  const dopplerKvCachePlan = resolveDopplerBenchmarkKvCachePlan({
    prefillTokenTarget: effectivePrefillTokenTarget,
    maxTokens,
    promptContract: sharedContract.promptContract,
  });
  const outputParityPolicy = resolveOutputParityPolicy(sharedContract, compareProfile);
  const decodeProfile = parseDecodeProfile(flags['decode-profile']);
  const resolveCompareProfileRuntimeBundle = (profileId) => compareProfileRuntimeBundles.get(profileId) || null;
  const compareProfileRuntimeBundleMetadata = Object.fromEntries(
    [...compareProfileRuntimeBundles.entries()].map(([profileId, bundle]) => [
      profileId,
      {
        runtimeProfileId: bundle.profileId,
        source: toRepoRelativeSourcePath(bundle.source),
        sourceSha256: bundle.sourceSha256,
        resolvedRuntimeSha256: bundle.resolvedRuntimeSha256,
      },
    ])
  );
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
  const resourceTelemetryOptions = {
    enabled: parseResourceTelemetryMode(flags['resource-telemetry'], false, '--resource-telemetry'),
    intervalMs: parsePositiveInt(
      flags['resource-telemetry-interval-ms'],
      DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS,
      '--resource-telemetry-interval-ms'
    ),
    includeSamples: flags['resource-telemetry-samples'] === true,
  };
  const buildRunResourceTelemetryOptions = (label) => resourceTelemetryOptions.enabled === true
    ? { ...resourceTelemetryOptions, label }
    : resourceTelemetryOptions;
  const tjsServerPort = parseNonNegativeInt(flags['tjs-server-port'], DEFAULT_TJS_SERVER_PORT, '--tjs-server-port');
  const tjsBrowserConsole = flags['tjs-browser-console'] === true;
  const browserBaseUrl = flags['browser-base-url'] || null;
  const browserExecutable = flags['browser-executable'] || null;
  const hasCustomDopplerBatchSize = flags['doppler-batch-size'] != null;
  const hasCustomDopplerReadbackInterval = flags['doppler-readback-interval'] != null;
  const hasCustomDopplerStopCheckMode = flags['doppler-stop-check-mode'] != null;
  const hasCustomDopplerDecodeTuning = (
    hasCustomDopplerBatchSize
    || hasCustomDopplerReadbackInterval
    || hasCustomDopplerStopCheckMode
  );
  if (hasCustomDopplerDecodeTuning && decodeProfile !== 'custom') {
    throw new Error(
      'Use --decode-profile custom when setting Doppler decode cadence flags.'
    );
  }
  const decodeProfileConfig = decodeProfile === 'custom'
    ? DECODE_PROFILE_CONFIGS.throughput
    : (DECODE_PROFILE_CONFIGS[decodeProfile] || DECODE_PROFILE_CONFIGS[DEFAULT_DECODE_PROFILE]);
    const dopplerKernelPathOverride = flags['doppler-kernel-path'] ?? DEFAULT_DOPPLER_KERNEL_PATH;
    const dopplerKernelResolution = resolveDopplerKernelPath(compareProfile, dopplerKernelPathOverride);
    const dopplerKernelPath = dopplerKernelResolution.kernelPath;
    assertKernelPathAllowedForModel(dopplerModelId, dopplerKernelPath);
    const dopplerBatchSize = parsePositiveInt(
      flags['doppler-batch-size'],
      decodeProfileConfig.batchSize,
    '--doppler-batch-size'
  );
  const dopplerReadbackInterval = parsePositiveInt(
    flags['doppler-readback-interval'],
    decodeProfileConfig.readbackInterval,
    '--doppler-readback-interval'
  );
  const dopplerStopCheckMode = normalizeStopCheckMode(
    flags['doppler-stop-check-mode'] ?? decodeProfileConfig.stopCheckMode,
    '--doppler-stop-check-mode'
  );
  const dopplerTokensPerReadback = dopplerBatchSize * dopplerReadbackInterval;
  const selectedDopplerDecodeCadence = {
    batchSize: dopplerBatchSize,
    readbackInterval: dopplerReadbackInterval,
    stopCheckMode: dopplerStopCheckMode,
    disableMultiTokenDecode: decodeProfileConfig.disableMultiTokenDecode === true,
    speculationMode: decodeProfileConfig.speculationMode ?? null,
  };
  const dopplerBrowserChannel = resolveBrowserChannelOverride(flags['doppler-browser-channel']);
  const dopplerBrowserUserData = flags['doppler-browser-user-data'] || null;
    const dopplerBrowserPort = flags['doppler-browser-port'] != null
      ? parseNonNegativeInt(
        flags['doppler-browser-port'],
        DEFAULT_DOPPLER_BROWSER_PORT,
        '--doppler-browser-port'
      )
      : null;
  const jsonOutput = flags.json === true;
  const shouldSave = flags.save === true;
  const skipMatrixUpdate = flags['skip-matrix-update'] === true;
  const saveDir = flags['save-dir'] || path.join(DOPPLER_ROOT, 'benchmarks', 'vendors', 'results');

  console.error(`[compare] Doppler model: ${dopplerModelId} (format=${dopplerFormat})`);
  console.error(`[compare] TJS model:     ${tjsModelId} (v${tjsVersion}, format=${tjsFormat})`);
  if (tjsModelOverridden && tjsModelId !== resolvedDefaultTjsModel) {
    console.error(
      `[compare] warning: --tjs-model overrides profile default (${resolvedDefaultTjsModel}); `
      + 'this may invalidate correctness comparability.'
    );
  }
  console.error(
    `[compare] model-profile source: ${compareProfile.source}`
    + ` (dopplerSource=${dopplerModelSource.source}, locator=${dopplerModelSource.locator ?? 'n/a'}, `
    + `compareLane=${compareProfile.compareLane})`
  );
  if (compareProfile.compareLaneReason) {
    console.error(`[compare] compare-lane note: ${compareProfile.compareLaneReason}`);
  }
  console.error(
    `[compare] Doppler kernel path: ${dopplerKernelPath ?? 'manifest-default'} `
    + `(${dopplerKernelResolution.source})`
  );
  if (dopplerManifestPreflight) {
    console.error(
      `[compare] Doppler manifest: ${dopplerManifestPreflight.manifestSource} `
      + `(sha256=${dopplerManifestPreflight.manifestSha256.slice(0, 12)}…, `
      + `requiredInference=${dopplerManifestPreflight.requiredInferenceFieldsArtifact.ok ? 'ok' : 'fail'}, `
      + `executionContract=${dopplerManifestPreflight.executionContractArtifact?.ok !== false ? 'ok' : 'fail'})`
    );
  }
  console.error(
    `[compare] mode: ${mode}, maxTokens: ${maxTokens}, warmupRuns: ${warmupRuns}, runs: ${runs}, `
    + `decodeProfile: ${decodeProfile}, workload=${workloadId ?? 'none'}, `
    + `dopplerSurface: ${dopplerSurface}, `
    + `dopplerBrowserChannel: ${dopplerBrowserChannel ?? 'default'}, `
    + `loadModes: warm=${resolvedLoadModes.warm}, cold=${resolvedLoadModes.cold}, `
    + `loadModeOverride: ${sharedContract.loadMode ?? 'none'}`
    + `${sharedLoadModeSource ? ` (${sharedLoadModeSource})` : ''}, `
    + `sampling=(temp=${sharedContract.sampling.temperature}, topK=${sharedContract.sampling.topK}, topP=${sharedContract.sampling.topP}), `
    + `useChatTemplate: requested=${sharedContract.promptContract?.requestedUseChatTemplate === true ? 'on' : 'off'} `
    + `engine=${sharedContract.useChatTemplate === true ? 'on' : 'off'}, `
    + `dopplerBatchSize: ${dopplerBatchSize}, `
    + `dopplerReadbackInterval: ${dopplerReadbackInterval}, `
    + `dopplerStopCheckMode: ${dopplerStopCheckMode}, `
    + `dopplerTokensPerReadback: ${dopplerTokensPerReadback}, `
    + `dopplerTimeoutMs: ${compareDopplerTimeoutMs}, `
    + `tjsTimeoutMs: ${compareTjsTimeoutMs}, `
    + `tjsDtype: ${tjsDtype} (${tjsDtypeSource}), `
    + `tjsModelSource: ${tjsModelSource}`
  );
  if (effectivePrefillTokenTarget != null) {
    console.error(
      `[compare] prompt target: ${effectivePrefillTokenTarget} model-input tokens `
      + `(source=${promptContract?.promptSource ?? 'unknown'}, `
      + `tokenizer=${promptContract?.tokenizerLocator ?? 'n/a'}, `
      + `tokenizerSource=${promptContract?.tokenizerResolutionSource ?? 'n/a'})`
    );
  }
  if (dopplerKvCachePlan.enabled === true) {
    console.error(
      `[compare] Doppler KV cache maxSeqLen: ${dopplerKvCachePlan.maxSeqLen} `
      + `(source=${dopplerKvCachePlan.source}, prompt=${dopplerKvCachePlan.promptTokenTarget}, `
      + `decode=${dopplerKvCachePlan.decodeTokenTarget}, margin=${dopplerKvCachePlan.tokenizerDeltaMargin})`
    );
  }

  const selectedMethodologyRuntimeBaseConfig = resolveCompareProfileRuntimeBundle(decodeProfile)?.resolvedRuntime ?? null;
  const dopplerExecution = resolveDopplerExecutionIdentity(dopplerSurface, dopplerFormat);
  const report = {
    timestamp: runTimestamp ?? new Date().toISOString(),
    benchmarkPolicy: {
      source: BENCHMARK_POLICY.source,
      schemaVersion: BENCHMARK_POLICY.schemaVersion,
      sourceSha256: BENCHMARK_POLICY.sourceSha256,
      updated: BENCHMARK_POLICY.updated,
      promotionGates: {
        throughputCadence: THROUGHPUT_CADENCE_PROMOTION_GATE,
      },
    },
    compareConfig: {
      source: toRepoRelativeSourcePath(compareProfileConfig.source || 'benchmarks/vendors/compare-engines.config.json'),
      schemaVersion: compareProfileConfig.schemaVersion,
      sourceSha256: compareProfileConfig.sourceSha256,
      profileSource: compareProfile.source,
      updated: compareProfileConfig.updated,
      defaults: {
        warmLoadMode: compareProfileConfig.defaults.warm,
        coldLoadMode: compareProfileConfig.defaults.cold,
      },
    },
    metricContract: {
      source: toRepoRelativeSourcePath(compareMetricContractConfig.source || 'benchmarks/vendors/compare-metrics.json'),
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
        source: toRepoRelativeSourcePath(dopplerHarnessMetricPaths.source),
        sourceSha256: dopplerHarnessMetricPaths.sourceSha256,
        requiredMetrics: requiredMetricIds,
      },
      transformersjs: {
        source: toRepoRelativeSourcePath(tjsHarnessMetricPaths.source),
        sourceSha256: tjsHarnessMetricPaths.sourceSha256,
        requiredMetrics: requiredMetricIds,
      },
    },
    runtimeConfig: {
      compareProfileRuntimeProfiles: compareProfileRuntimeBundleMetadata,
      overrideProvided: runtimeConfigOverride !== null,
      runtimeConfigSha256: runtimeConfigOverrideHash,
      loadModeRuntimeOverlays: {
        warm: summarizeDopplerLoadModeRuntimeOverlay(resolvedLoadModes.warm),
        cold: summarizeDopplerLoadModeRuntimeOverlay(resolvedLoadModes.cold),
      },
      dopplerKvCachePlan,
      compareProfileSchemaVersion: compareProfileConfig.schemaVersion ?? DEFAULT_COMPARE_CONFIG_SCHEMA_VERSION,
    },
    compareLane: {
      declared: compareProfile.compareLane,
      reason: compareProfile.compareLaneReason,
      allowNonComparableLane,
    },
    dopplerModelId,
    dopplerArtifactIdentity: dopplerManifestPreflight?.artifactIdentity ?? dopplerModelSource.registrySource ?? null,
    dopplerDtype: parseDopplerDtype(dopplerModelId),
    dopplerModelSource: {
      source: dopplerModelSource.source,
      locator: dopplerModelSource.locator ?? null,
      modelUrl: dopplerModelSource.modelUrl ?? null,
      modelBaseDir: dopplerModelSource.modelBaseDir ?? null,
      manifestSource: dopplerManifestPreflight?.manifestSource ?? null,
      manifestSha256: dopplerManifestPreflight?.manifestSha256 ?? null,
      registrySource: dopplerModelSource.registrySource ?? null,
    },
    dopplerManifestPreflight,
    dopplerSurface,
    dopplerExecution,
    dopplerKernelPath: dopplerKernelPath ?? 'manifest-default',
    dopplerKernelPathSource: dopplerKernelResolution.source,
    decodeProfile,
    dopplerBatchSize,
    dopplerReadbackInterval,
    dopplerStopCheckMode,
    dopplerTokensPerReadback,
    tjsModelId,
    transformersjsDtype: tjsDtype,
    transformersjsModelSource: tjsModelSource,
    transformersjsDtypeSource: tjsDtypeSource,
    transformersjsBenchmarkCatalog: {
      source: toRepoRelativeSourcePath(modelCatalog.source || 'models/catalog.json'),
      sourceSha256: modelCatalog.sourceSha256,
      updatedAt: modelCatalog.updatedAt,
      sourceModelId: catalogResolvedTjsBenchmark?.sourceModelId ?? null,
    },
    mode,
    prompt,
    promptContract: sharedContract.promptContract,
    maxTokens,
    warmupRuns,
    runs,
    seed,
    workload: {
      id: workloadId,
      prefillTokenTarget: effectivePrefillTokenTarget,
      prefillTokenSource: promptContract?.promptSource ?? (flags.prompt ? 'explicit-prompt' : 'default-prompt'),
      prefillTokenSemantics: 'model_input_tokens',
      promptTokenizerLocator: promptContract?.tokenizerLocator ?? null,
      promptTokenizerResolutionSource: promptContract?.tokenizerResolutionSource ?? null,
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
        channel: dopplerBrowserChannel,
      },
    },
    comparatorStack: {
      transformersjs: tjsRuntimeStack,
    },
    timeoutMs: compareSharedTimeoutMs ?? compareDopplerTimeoutMs,
    dopplerTimeoutMs: compareDopplerTimeoutMs,
    tjsTimeoutMs: compareTjsTimeoutMs,
    loadModeResolution: {
      sharedOverride: sharedContract.loadMode,
      sharedOverrideSource: sharedLoadModeSource,
      profileDefault: compareProfile.defaultLoadMode,
      profileDefaultReason: compareProfile.defaultLoadModeReason,
      warm: resolvedLoadModes.warm,
      cold: resolvedLoadModes.cold,
    },
    methodology: {
      promptTokensPerSecToFirstToken: 'prompt_tokens / firstTokenMs',
      prefillPhaseMs: {
        doppler: 'available inside Doppler payload timing.prefillMs as internal_prefill_phase',
        transformersjs: 'excluded from compare contract because runner prefillMs is a ttft_proxy',
      },
      deterministicDecoding: {
        seed,
        temperature: sharedContract.sampling.temperature,
        topK: sharedContract.sampling.topK,
        topP: sharedContract.sampling.topP,
        repetitionPenalty: sharedContract.sampling.repetitionPenalty,
        greedyThreshold: sharedContract.sampling.greedyThreshold,
      },
      outputParity: outputParityPolicy,
      promptParity: {
        promptRenderer: sharedContract.promptContract?.promptRenderer ?? 'unknown',
        chatTemplateSource: sharedContract.promptContract?.chatTemplateSource ?? 'unknown',
        enginesReceiveRenderedPrompt: sharedContract.promptContract?.enginesReceiveRenderedPrompt === true,
        dopplerChatTemplateEnabled: sharedContract.useChatTemplate,
        transformersChatTemplateEquivalent: sharedContract.useChatTemplate
          ? 'model-chat-template'
          : 'pre-rendered-or-raw-prompt',
      },
      loadMode: sharedContract.loadMode,
      cacheSemantics: {
        warm: `Prime engine cache before timed run, then run from local persistent cache only (loadMode=${resolvedLoadModes.warm}; ${formatDopplerRuntimeCacheLabel(dopplerSurface, resolvedLoadModes.warm)}; TJS persistent browser cache).`,
        cold: `Wipe engine-specific persistent cache state before run (loadMode=${resolvedLoadModes.cold}; ${formatDopplerRuntimeCacheLabel(dopplerSurface, resolvedLoadModes.cold)}/profile wipe; TJS profile wipe).`,
      },
      formats: {
        doppler: dopplerFormat,
        transformersjs: tjsFormat,
        safetensorsSourceId,
      },
      dopplerDecodeCadence: {
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        stopCheckMode: dopplerStopCheckMode,
        readbackMode: resolveRuntimeReadbackMode(selectedMethodologyRuntimeBaseConfig),
        disableMultiTokenDecode: DECODE_PROFILE_CONFIGS[decodeProfile]?.disableMultiTokenDecode === true,
        speculationMode: DECODE_PROFILE_CONFIGS[decodeProfile]?.speculationMode ?? 'inherit',
        tokensPerReadback: dopplerTokensPerReadback,
      },
      transformersjsDecodeCadence: {
        streamerCallbackGranularityTokens: 1,
        readbackControl: 'runtime-internal',
      },
      resourceTelemetry: {
        enabled: resourceTelemetryOptions.enabled,
        intervalMs: resourceTelemetryOptions.intervalMs,
        includeSamples: resourceTelemetryOptions.includeSamples,
        sources: {
          processTree: 'linux procfs process tree',
          systemRam: 'linux /proc/meminfo',
          gpu: 'optional rocm-smi device-global telemetry when available',
        },
      },
      comparabilityWarnings: buildComparabilityWarnings({ dopplerSurface }),
    },
    sections: {},
  };

  // Compute measures warm-cache behavior and reports both parity/throughput Doppler cadence.
  let dopplerComputeParity = null;
  let dopplerComputeThroughput = null;
  let dopplerComputeThroughputDiagnostic = null;
  let tjsCompute = null;
  let dopplerWarm = null;
  let tjsWarm = null;
  let dopplerCold = null;
  let tjsCold = null;
  const parityRuntimeBaseConfig = resolveCompareProfileRuntimeBundle('parity')?.resolvedRuntime ?? null;
  const throughputRuntimeBaseConfig = resolveCompareProfileRuntimeBundle('throughput')?.resolvedRuntime ?? null;
  const selectedRuntimeBaseConfig = resolveCompareProfileRuntimeBundle(decodeProfile)?.resolvedRuntime ?? null;
  const customComputeCadence = {
    batchSize: dopplerBatchSize,
    readbackInterval: dopplerReadbackInterval,
    stopCheckMode: dopplerStopCheckMode,
    disableMultiTokenDecode: decodeProfileConfig.disableMultiTokenDecode === true,
    speculationMode: decodeProfileConfig.speculationMode ?? null,
  };
  const computeParityCadence = resolveComputeDecodeCadence(
    decodeProfile,
    'parity',
    customComputeCadence,
    decodeProfile === 'custom' ? selectedRuntimeBaseConfig : parityRuntimeBaseConfig
  );
  const computeThroughputCadence = resolveComputeDecodeCadence(
    decodeProfile,
    'throughput',
    customComputeCadence,
    decodeProfile === 'custom' ? selectedRuntimeBaseConfig : throughputRuntimeBaseConfig
  );

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
        tjsFormat,
        browserExecutable,
        browserConsole: tjsBrowserConsole,
        browserBaseUrl,
        loadMode: resolvedLoadModes.warm,
        resourceTelemetry: buildRunResourceTelemetryOptions('transformersjs-compute'),
      }
    );
    dopplerComputeParity = await runDoppler(
      dopplerModelId,
      dopplerModelSource.modelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: computeParityCadence.batchSize,
        readbackInterval: computeParityCadence.readbackInterval,
        stopCheckMode: computeParityCadence.stopCheckMode,
        disableMultiTokenDecode: computeParityCadence.disableMultiTokenDecode,
        speculationMode: computeParityCadence.speculationMode,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
        browserChannel: dopplerBrowserChannel,
        browserExecutable,
        browserBaseUrl,
        timeoutMs: compareDopplerTimeoutMs,
        runtimeBaseConfigJson: computeParityCadence.runtimeBaseConfig,
        runtimeConfigJson: runtimeConfigOverride,
        kvCachePlan: dopplerKvCachePlan,
        surface: dopplerSurface,
        format: dopplerFormat,
        safetensorsSourcePath,
        loadMode: resolvedLoadModes.warm,
        resourceTelemetry: buildRunResourceTelemetryOptions('doppler-compute-parity'),
      }
    );
    assertDopplerDecodeCadence(dopplerComputeParity, computeParityCadence, 'compute/parity');

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
      dopplerModelSource.modelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: computeThroughputCadence.batchSize,
        readbackInterval: computeThroughputCadence.readbackInterval,
        stopCheckMode: computeThroughputCadence.stopCheckMode,
        disableMultiTokenDecode: computeThroughputCadence.disableMultiTokenDecode,
        speculationMode: computeThroughputCadence.speculationMode,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
        browserChannel: dopplerBrowserChannel,
        browserExecutable,
        browserBaseUrl,
        timeoutMs: compareDopplerTimeoutMs,
        runtimeBaseConfigJson: computeThroughputCadence.runtimeBaseConfig,
        runtimeConfigJson: runtimeConfigOverride,
        kvCachePlan: dopplerKvCachePlan,
        surface: dopplerSurface,
        format: dopplerFormat,
        safetensorsSourcePath,
        loadMode: resolvedLoadModes.warm,
        resourceTelemetry: buildRunResourceTelemetryOptions('doppler-compute-throughput'),
      }
    );
    assertDopplerDecodeCadence(dopplerComputeThroughput, computeThroughputCadence, 'compute/throughput');

    assertRequiredHarnessMetrics(
      dopplerComputeThroughput,
      { ...dopplerHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'compute/throughput',
      'Doppler'
    );

    const computeParitySection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: resolvedLoadModes.warm,
      doppler: dopplerComputeParity,
      transformersjs: tjsCompute,
      prefillTokenTarget: effectivePrefillTokenTarget,
      maxTokens,
      promptContract: sharedContract.promptContract,
      outputParityPolicy,
    });
    const computeThroughputSection = buildCompareSection({
      cacheMode: 'warm',
      loadMode: resolvedLoadModes.warm,
      doppler: dopplerComputeThroughput,
      transformersjs: tjsCompute,
      prefillTokenTarget: effectivePrefillTokenTarget,
      maxTokens,
      promptContract: sharedContract.promptContract,
      outputParityPolicy,
    });
    report.sections.compute = {
      parity: computeParitySection,
      throughput: computeThroughputSection,
      throughputCadenceGate: resolveDopplerThroughputCadenceGate({
        paritySection: computeParitySection,
        throughputSection: computeThroughputSection,
      }),
    };
    if (dopplerBottleneckProfile) {
      const diagnosticSharedContract = buildDiagnosticSharedBenchmarkContract(sharedContract);
      dopplerComputeThroughputDiagnostic = await runDoppler(
        dopplerModelId,
        dopplerModelSource.modelUrl,
        diagnosticSharedContract,
        'warm',
        {
          command: 'debug',
          workload: 'inference',
          diagnosticLabel: 'bottleneck-profile',
          kernelPath: dopplerKernelPath,
          batchSize: computeThroughputCadence.batchSize,
          readbackInterval: computeThroughputCadence.readbackInterval,
          stopCheckMode: computeThroughputCadence.stopCheckMode,
          disableMultiTokenDecode: computeThroughputCadence.disableMultiTokenDecode,
          speculationMode: computeThroughputCadence.speculationMode,
          noOpfsCache: dopplerNoOpfsCache,
          browserUserData: dopplerBrowserUserData,
          browserPort: dopplerBrowserPort,
          browserChannel: dopplerBrowserChannel,
          browserExecutable,
          browserBaseUrl,
          timeoutMs: compareDopplerTimeoutMs,
          maxBufferBytes: DEFAULT_DOPPLER_DIAGNOSTIC_MAX_BUFFER_BYTES,
          runtimeBaseConfigJson: computeThroughputCadence.runtimeBaseConfig,
          runtimeConfigJson: runtimeConfigOverride,
          kvCachePlan: dopplerKvCachePlan,
          runtimeDiagnosticConfigJson: buildDopplerBottleneckDiagnosticRuntimeOverlay(),
          surface: dopplerSurface,
          format: dopplerFormat,
          safetensorsSourcePath,
          loadMode: resolvedLoadModes.warm,
          resourceTelemetry: buildRunResourceTelemetryOptions('doppler-diagnostic-bottleneck'),
        }
      );
      assertDopplerDecodeCadence(
        dopplerComputeThroughputDiagnostic,
        computeThroughputCadence,
        'diagnostics/doppler-bottleneck'
      );
      report.diagnostics = {
        ...(report.diagnostics ?? {}),
        dopplerBottleneck: buildDopplerBottleneckDiagnostic({
          doppler: dopplerComputeThroughputDiagnostic,
          sourceSection: 'compute/throughput',
          sharedContract: diagnosticSharedContract,
        }),
      };
    }
    report.correctness = buildCorrectnessReport(dopplerComputeParity, tjsCompute, sharedContract.promptRaw);
  }

  if (needWarm) {
    dopplerWarm = await runDoppler(
      dopplerModelId,
      dopplerModelSource.modelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        stopCheckMode: dopplerStopCheckMode,
        disableMultiTokenDecode: decodeProfileConfig.disableMultiTokenDecode === true,
        speculationMode: decodeProfileConfig.speculationMode ?? null,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
        browserChannel: dopplerBrowserChannel,
        browserExecutable,
        browserBaseUrl,
        timeoutMs: compareDopplerTimeoutMs,
        runtimeBaseConfigJson: selectedRuntimeBaseConfig,
        runtimeConfigJson: runtimeConfigOverride,
        kvCachePlan: dopplerKvCachePlan,
        surface: dopplerSurface,
        format: dopplerFormat,
        safetensorsSourcePath,
        loadMode: resolvedLoadModes.warm,
        resourceTelemetry: buildRunResourceTelemetryOptions('doppler-warm'),
      }
    );
    assertDopplerDecodeCadence(dopplerWarm, selectedDopplerDecodeCadence, 'warm');
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
        tjsFormat,
        browserExecutable,
        browserConsole: tjsBrowserConsole,
        browserBaseUrl,
        loadMode: resolvedLoadModes.warm,
        resourceTelemetry: buildRunResourceTelemetryOptions('transformersjs-warm'),
      }
    );
    assertDeterministicParityContract(sharedContract, tjsWarm, 'warm');
    assertRequiredHarnessMetrics(
      tjsWarm,
      { ...tjsHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'warm',
      'TJS'
    );

    report.sections.warm = buildCompareSection({
      cacheMode: 'warm',
      loadMode: resolvedLoadModes.warm,
      doppler: dopplerWarm,
      transformersjs: tjsWarm,
      prefillTokenTarget: effectivePrefillTokenTarget,
      maxTokens,
      promptContract: sharedContract.promptContract,
      outputParityPolicy,
    });
  }

  if (needCold) {
    dopplerCold = await runDoppler(
      dopplerModelId,
      dopplerModelSource.modelUrl,
      sharedContract,
      'cold',
      {
        kernelPath: dopplerKernelPath,
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        stopCheckMode: dopplerStopCheckMode,
        disableMultiTokenDecode: decodeProfileConfig.disableMultiTokenDecode === true,
        speculationMode: decodeProfileConfig.speculationMode ?? null,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
        browserChannel: dopplerBrowserChannel,
        browserExecutable,
        browserBaseUrl,
        timeoutMs: compareDopplerTimeoutMs,
        runtimeBaseConfigJson: selectedRuntimeBaseConfig,
        runtimeConfigJson: runtimeConfigOverride,
        kvCachePlan: dopplerKvCachePlan,
        surface: dopplerSurface,
        format: dopplerFormat,
        safetensorsSourcePath,
        loadMode: resolvedLoadModes.cold,
        resourceTelemetry: buildRunResourceTelemetryOptions('doppler-cold'),
      }
    );
    assertDopplerDecodeCadence(dopplerCold, selectedDopplerDecodeCadence, 'cold');
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
        tjsFormat,
        browserExecutable,
        browserConsole: tjsBrowserConsole,
        browserBaseUrl,
        loadMode: resolvedLoadModes.cold,
        resourceTelemetry: buildRunResourceTelemetryOptions('transformersjs-cold'),
      }
    );
    assertDeterministicParityContract(sharedContract, tjsCold, 'cold');
    assertRequiredHarnessMetrics(
      tjsCold,
      { ...tjsHarnessMetricPaths, requiredMetrics: requiredMetricIds },
      'cold',
      'TJS'
    );

    report.sections.cold = buildCompareSection({
      cacheMode: 'cold',
      loadMode: resolvedLoadModes.cold,
      doppler: dopplerCold,
      transformersjs: tjsCold,
      prefillTokenTarget: effectivePrefillTokenTarget,
      maxTokens,
      promptContract: sharedContract.promptContract,
      outputParityPolicy,
    });
  }

  attachCompareFairnessAudit(report, {
    dopplerExecution,
    dopplerFormat,
    tjsFormat,
    tjsModelOverridden,
    dopplerModelSource,
  });

  if (jsonOutput) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    const decodeProfileLabel = decodeProfile === 'custom'
      ? 'custom decode cadence'
      : (decodeProfileConfig?.label || 'custom decode cadence');
    console.log(
      `[method] prompt tok/s uses prompt_tokens / firstTokenMs, decodeProfile=${decodeProfile} ` +
      `(${decodeProfileLabel}), Doppler tokens/readback=${dopplerTokensPerReadback}, ` +
      `stopCheckMode=${dopplerStopCheckMode}`
    );
    const computeRows = compareMetricContract;

    if (mode === 'compute' || mode === 'all') {
      printSection('COMPUTE (PARITY)', report.sections.compute?.parity, computeRows);
      printSection('COMPUTE (THROUGHPUT)', report.sections.compute?.throughput, computeRows);
      printThroughputCadenceGate(report.sections.compute?.throughputCadenceGate);
      printDopplerBottleneckDiagnostic(report.diagnostics?.dopplerBottleneck);
    }
    if (mode === 'warm' || mode === 'all') {
      printSection('WARM START', report.sections.warm, computeRows);
    }
    if (mode === 'cold' || mode === 'all') {
      printSection('COLD START', report.sections.cold, computeRows);
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

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(`[compare] ${redactSecrets(error.message)}`);
    process.exit(1);
  });
}

export {
  applyDopplerBenchCaptureRunConfig,
  attachCompareFairnessAudit,
  buildCompareSection,
  buildCompareFairnessAudit,
  buildDiagnosticSharedBenchmarkContract,
  buildDopplerBottleneckDiagnostic,
  buildDopplerManifestFreshnessArtifact,
  buildDopplerBottleneckDiagnosticRuntimeOverlay,
  buildDopplerRuntimeConfig,
  buildSharedBenchmarkContract,
  assertDopplerDecodeCadence,
  loadModelCatalogBundle,
  loadResolvedRuntimeProfileBundle,
  normalizeCompareLoadModeDefaults,
  parseArgs,
  parseJsonBlock,
  parseOnOff,
  redactSecrets,
  readDopplerBenchCaptureResult,
  resolveComputeDecodeCadence,
  resolveCatalogTransformersjsBenchmarkTarget,
  renderComparePrompt,
  resolveCompareOwnedPromptRenderer,
  resolveCompareProfile,
  resolveCompareRuntimeProfileMap,
  resolveCompareLoadModes,
  resolveDopplerBenchmarkKvCachePlan,
  resolveDopplerExecutionIdentity,
  resolveDopplerModelSource,
  resolveDopplerThroughputCadenceGate,
  summarizeDopplerDecodeProfileSteps,
  resolveOutputParityPolicy,
  resolveRunLoadMode,
  usage,
};
