import { sha256Hex } from '../utils/sha256.js';
import { UL_TRAINING_SCHEMA_VERSION } from '../config/schema/ul-training.schema.js';

const UL_MANIFEST_SCHEMA_VERSION = 1;
const TRAINING_METRICS_SCHEMA_VERSION = 1;

function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

function normalizeTimestamp(value) {
  const date = value instanceof Date
    ? value
    : (typeof value === 'string' && value.trim() ? new Date(value) : new Date());
  return date.toISOString().replace(/[:]/g, '-');
}

function stableSortObject(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableSortObject(entry));
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  const sorted = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = stableSortObject(value[key]);
  }
  return sorted;
}

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function hashStableJson(value) {
  return sha256Hex(stableJson(value));
}

function toFiniteNumber(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function buildUlContractPayload(ulConfig) {
  if (!ulConfig) return {};
  return {
    lambda0: toFiniteNumber(ulConfig.lambda0, 5),
    noiseSchedule: ulConfig.noiseSchedule || null,
    priorAlignment: ulConfig.priorAlignment || null,
    decoderSigmoidWeight: ulConfig.decoderSigmoidWeight || null,
    lossWeights: ulConfig.lossWeights || null,
  };
}

function buildResolvedUlConfigSnapshot(ulConfig) {
  if (!ulConfig) return null;
  return {
    enabled: ulConfig.enabled === true,
    stage: ulConfig.stage || null,
    lambda0: toFiniteNumber(ulConfig.lambda0, 5),
    seed: toFiniteNumber(ulConfig.seed, 1337),
    noiseSchedule: ulConfig.noiseSchedule || null,
    priorAlignment: ulConfig.priorAlignment || null,
    decoderSigmoidWeight: ulConfig.decoderSigmoidWeight || null,
    freeze: ulConfig.freeze || null,
  };
}

function buildUlRuntimeDump(ulConfig, runOptions = {}) {
  if (!ulConfig) return null;
  return {
    stage: ulConfig.stage || null,
    lambda0: toFiniteNumber(ulConfig.lambda0, 5),
    seed: toFiniteNumber(ulConfig.seed, 1337),
    noiseSchedule: ulConfig.noiseSchedule || null,
    priorAlignment: ulConfig.priorAlignment || null,
    decoderSigmoidWeight: ulConfig.decoderSigmoidWeight || null,
    lossWeights: ulConfig.lossWeights || null,
    freeze: ulConfig.freeze || null,
    artifactDir: runOptions.ulArtifactDir || ulConfig.artifactDir || 'bench/out/ul',
    stage1Artifact: runOptions.stage1Artifact || ulConfig.stage1Artifact || null,
    stage1ArtifactHash: runOptions.stage1ArtifactHash || ulConfig.stage1ArtifactHash || null,
  };
}

function buildDeterministicManifestView(manifestBase) {
  return {
    schemaVersion: manifestBase.schemaVersion,
    stage: manifestBase.stage,
    configHash: manifestBase.configHash,
    modelHash: manifestBase.modelHash,
    datasetHash: manifestBase.datasetHash,
    ulContractHash: manifestBase.ulContractHash,
    ulResolvedConfig: manifestBase.ulResolvedConfig || null,
    runtimeDump: manifestBase.runtimeDump || null,
    buildProvenance: manifestBase.buildProvenance || null,
    freeze: manifestBase.freeze || null,
    metrics: {
      count: manifestBase.metrics?.count ?? 0,
    },
    latentDataset: manifestBase.latentDataset
      ? {
        hash: manifestBase.latentDataset.hash || null,
        count: manifestBase.latentDataset.count || 0,
        summary: manifestBase.latentDataset.summary || null,
      }
      : null,
    lineage: manifestBase.lineage || null,
    stage1Dependency: manifestBase.stage1Dependency
      ? {
        hash: manifestBase.stage1Dependency.hash,
        manifestHash: manifestBase.stage1Dependency.manifestHash || null,
      }
      : null,
  };
}

function toLatentDatasetRecord(entry) {
  if (!entry || typeof entry !== 'object') return null;
  if (entry.objective !== 'ul_stage1_joint') return null;
  const required = [
    entry.step,
    entry.lambda,
    entry.latent_clean_std,
    entry.latent_noisy_std,
    entry.latent_noise_std,
  ];
  if (!required.every(isFiniteNumber)) {
    return null;
  }
  const shape = Array.isArray(entry.latent_shape)
    ? entry.latent_shape.map((value) => Math.max(1, Math.floor(Number(value) || 1)))
    : null;
  const cleanValues = Array.isArray(entry.latent_clean_values) ? entry.latent_clean_values : null;
  const noiseValues = Array.isArray(entry.latent_noise_values) ? entry.latent_noise_values : null;
  const noisyValues = Array.isArray(entry.latent_noisy_values) ? entry.latent_noisy_values : null;
  const elementCount = shape
    ? shape.reduce((acc, value) => acc * value, 1)
    : 0;
  const hasVectorLatents = !!(
    shape
    && elementCount > 0
    && cleanValues?.length === elementCount
    && noiseValues?.length === elementCount
    && noisyValues?.length === elementCount
    && cleanValues.every(isFiniteNumber)
    && noiseValues.every(isFiniteNumber)
    && noisyValues.every(isFiniteNumber)
  );
  return {
    step: entry.step,
    lambda: entry.lambda,
    schedule_step_index: isFiniteNumber(entry.schedule_step_index) ? entry.schedule_step_index : 0,
    latent_clean_mean: isFiniteNumber(entry.latent_clean_mean) ? entry.latent_clean_mean : 0,
    latent_clean_std: entry.latent_clean_std,
    latent_noisy_mean: isFiniteNumber(entry.latent_noisy_mean) ? entry.latent_noisy_mean : 0,
    latent_noisy_std: entry.latent_noisy_std,
    latent_noise_mean: isFiniteNumber(entry.latent_noise_mean) ? entry.latent_noise_mean : 0,
    latent_noise_std: entry.latent_noise_std,
    latent_shape: shape,
    latent_clean_values: hasVectorLatents ? cleanValues : null,
    latent_noise_values: hasVectorLatents ? noiseValues : null,
    latent_noisy_values: hasVectorLatents ? noisyValues : null,
  };
}

async function readNdjson(filePath) {
  const { readFile } = await nodeFs();
  const raw = await readFile(filePath, 'utf8');
  const lines = raw.split('\n').map((line) => line.trim()).filter(Boolean);
  return {
    raw,
    entries: lines.map((line) => JSON.parse(line)),
  };
}

function summarizeLatentDataset(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return {
      lambdaMean: 0,
      noisyStdMean: 0,
      cleanStdMean: 0,
      noiseStdMean: 0,
      scheduleMaxStep: 0,
    };
  }
  let lambdaSum = 0;
  let noisyStdSum = 0;
  let cleanStdSum = 0;
  let noiseStdSum = 0;
  let scheduleMaxStep = 0;
  for (const entry of entries) {
    lambdaSum += toFiniteNumber(entry.lambda, 0);
    noisyStdSum += toFiniteNumber(entry.latent_noisy_std, 0);
    cleanStdSum += toFiniteNumber(entry.latent_clean_std, 0);
    noiseStdSum += toFiniteNumber(entry.latent_noise_std, 0);
    scheduleMaxStep = Math.max(scheduleMaxStep, Math.floor(toFiniteNumber(entry.schedule_step_index, 0)));
  }
  const count = entries.length;
  const vectorCount = entries.filter((entry) => Array.isArray(entry?.latent_noisy_values)).length;
  return {
    lambdaMean: lambdaSum / count,
    noisyStdMean: noisyStdSum / count,
    cleanStdMean: cleanStdSum / count,
    noiseStdMean: noiseStdSum / count,
    scheduleMaxStep,
    vectorCount,
  };
}

async function resolveBuildProvenance() {
  if (!isNodeRuntime()) {
    return {
      runtime: 'browser',
      nodeVersion: null,
      dopplerVersion: null,
      commitHash: null,
      kernelRegistryDigest: null,
      schemaVersions: {
        ulManifest: UL_MANIFEST_SCHEMA_VERSION,
        ulTraining: UL_TRAINING_SCHEMA_VERSION,
        trainingMetrics: TRAINING_METRICS_SCHEMA_VERSION,
      },
    };
  }
  const { readFile } = await nodeFs();
  const { join } = await nodePath();
  let dopplerVersion = null;
  let kernelRegistryDigest = null;
  try {
    const packagePath = join(process.cwd(), 'package.json');
    const packageRaw = await readFile(packagePath, 'utf8');
    const parsed = JSON.parse(packageRaw);
    dopplerVersion = typeof parsed.version === 'string' ? parsed.version : null;
  } catch {
    dopplerVersion = null;
  }
  try {
    const kernelPath = join(process.cwd(), 'src/config/kernels/registry.json');
    const kernelRaw = await readFile(kernelPath, 'utf8');
    kernelRegistryDigest = sha256Hex(kernelRaw);
  } catch {
    kernelRegistryDigest = null;
  }
  return {
    runtime: 'node',
    nodeVersion: process.version || null,
    dopplerVersion,
    commitHash: process.env.DOPPLER_GIT_COMMIT || process.env.GITHUB_SHA || null,
    kernelRegistryDigest,
    schemaVersions: {
      ulManifest: UL_MANIFEST_SCHEMA_VERSION,
      ulTraining: UL_TRAINING_SCHEMA_VERSION,
      trainingMetrics: TRAINING_METRICS_SCHEMA_VERSION,
    },
  };
}

async function nodePath() {
  return import('node:path');
}

async function nodeFs() {
  return import('node:fs/promises');
}

async function resolveNodePath(p) {
  const { resolve } = await nodePath();
  return resolve(String(p));
}

async function readJson(filePath) {
  const { readFile } = await nodeFs();
  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function computeFileHash(filePath) {
  const { readFile } = await nodeFs();
  const raw = await readFile(filePath, 'utf8');
  return sha256Hex(raw);
}

export function resolveUlTrainingContract(ulConfig) {
  if (!ulConfig?.enabled) {
    return {
      enabled: false,
      stage: null,
      artifactDir: null,
      stage1Artifact: null,
      stage1ArtifactHash: null,
    };
  }
  return {
    enabled: true,
    stage: ulConfig.stage,
    artifactDir: ulConfig.artifactDir || 'bench/out/ul',
    stage1Artifact: ulConfig.stage1Artifact || null,
    stage1ArtifactHash: ulConfig.stage1ArtifactHash || null,
  };
}

async function validateStage2Dependency(config, contractHash) {
  const ulConfig = config.training?.ul;
  if (!ulConfig?.stage1Artifact) {
    throw new Error('UL stage2 requires training.ul.stage1Artifact.');
  }
  const manifestPath = await resolveNodePath(ulConfig.stage1Artifact);
  const manifestHash = await computeFileHash(manifestPath);
  const manifest = await readJson(manifestPath);
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('UL stage2 requires a valid Stage1 manifest JSON.');
  }
  const providedHash = ulConfig.stage1ArtifactHash;
  if (providedHash) {
    const accepted = [
      manifestHash,
      manifest.manifestHash || null,
      manifest.manifestContentHash || null,
      manifest.manifestFileHash || null,
    ];
    if (!accepted.includes(providedHash)) {
      throw new Error(
        `UL stage2 artifact hash mismatch: expected ${providedHash}, got ${manifestHash}.`
      );
    }
  }
  if (manifest.stage !== 'stage1_joint') {
    throw new Error(`UL stage2 requires stage1_joint artifact, got "${manifest.stage}".`);
  }
  if (manifest.ulContractHash !== contractHash) {
    throw new Error(
      `UL stage2 contract mismatch: expected ${contractHash}, got ${manifest.ulContractHash || 'unknown'}.`
    );
  }
  if (!manifest.latentDataset || typeof manifest.latentDataset !== 'object') {
    throw new Error('UL stage2 requires stage1 latentDataset metadata.');
  }
  const latentDatasetPath = await resolveNodePath(manifest.latentDataset.path || '');
  const latentDatasetHash = await computeFileHash(latentDatasetPath);
  if (manifest.latentDataset.hash && manifest.latentDataset.hash !== latentDatasetHash) {
    throw new Error(
      `UL stage2 latentDataset hash mismatch: expected ${manifest.latentDataset.hash}, got ${latentDatasetHash}.`
    );
  }
  const latentDataset = await readNdjson(latentDatasetPath);
  if (!Array.isArray(latentDataset.entries) || latentDataset.entries.length === 0) {
    throw new Error('UL stage2 requires non-empty stage1 latentDataset.');
  }
  const vectorEntries = latentDataset.entries.filter((entry) => {
    const shape = Array.isArray(entry?.latent_shape) ? entry.latent_shape : null;
    if (!shape) return false;
    const elementCount = shape.reduce((acc, value) => acc * Math.max(1, Math.floor(Number(value) || 1)), 1);
    return (
      elementCount > 0
      && Array.isArray(entry?.latent_noisy_values)
      && entry.latent_noisy_values.length === elementCount
      && entry.latent_noisy_values.every(isFiniteNumber)
    );
  });
  if (vectorEntries.length === 0) {
    throw new Error('UL stage2 requires stage1 latentDataset entries with latent vectors.');
  }
  return {
    path: manifestPath,
    hash: manifestHash,
    manifest,
    latentDataset: {
      path: latentDatasetPath,
      hash: latentDatasetHash,
      entries: latentDataset.entries,
      summary: summarizeLatentDataset(latentDataset.entries),
    },
  };
}

export async function resolveStage1ArtifactContext(config) {
  const ulConfig = config?.training?.ul;
  if (!ulConfig?.enabled || ulConfig.stage !== 'stage2_base') {
    return null;
  }
  if (!isNodeRuntime()) {
    throw new Error('UL stage2 artifact context currently requires Node runtime.');
  }
  const ulContractHash = hashStableJson(buildUlContractPayload(ulConfig));
  const dependency = await validateStage2Dependency(config, ulContractHash);
  return {
    manifestPath: dependency.path,
    manifestHash: dependency.hash,
    ulContractHash: dependency.manifest?.ulContractHash || null,
    latentDataset: {
      path: dependency.latentDataset.path,
      hash: dependency.latentDataset.hash,
      count: dependency.latentDataset.entries.length,
      summary: dependency.latentDataset.summary,
      entries: dependency.latentDataset.entries,
    },
  };
}

export async function createUlArtifactSession(options) {
  const {
    config,
    stage,
    runOptions = {},
  } = options || {};
  const ulConfig = config?.training?.ul;
  if (!ulConfig?.enabled) {
    return null;
  }
  if (!isNodeRuntime()) {
    throw new Error('UL artifacts currently require Node runtime.');
  }
  const { mkdir, appendFile, writeFile } = await nodeFs();
  const { join, relative } = await nodePath();

  const resolvedStage = stage || ulConfig.stage || 'stage1_joint';
  const timestamp = normalizeTimestamp(runOptions.timestamp);
  const artifactRoot = await resolveNodePath(
    runOptions.ulArtifactDir || ulConfig.artifactDir || 'bench/out/ul'
  );
  const runDir = join(artifactRoot, `${resolvedStage}_${timestamp}`);
  await mkdir(runDir, { recursive: true });
  const metricsPath = join(runDir, 'metrics.ndjson');
  const latentDatasetPath = join(runDir, 'latents.ndjson');
  const manifestPath = join(
    runDir,
    resolvedStage === 'stage2_base' ? 'ul_stage2_manifest.json' : 'ul_stage1_manifest.json'
  );
  const ulContractHash = hashStableJson(buildUlContractPayload(ulConfig));
  const stageDependency = resolvedStage === 'stage2_base'
    ? await validateStage2Dependency(config, ulContractHash)
    : null;
  let latentDatasetCount = 0;

  return {
    async appendStep(entry) {
      await appendFile(metricsPath, `${JSON.stringify(entry)}\n`, 'utf8');
      if (resolvedStage === 'stage1_joint') {
        const latentRecord = toLatentDatasetRecord(entry);
        if (latentRecord) {
          latentDatasetCount += 1;
          await appendFile(latentDatasetPath, `${JSON.stringify(latentRecord)}\n`, 'utf8');
        }
      }
    },
    async finalize(stepMetrics) {
      const configHash = hashStableJson(config);
      const modelHash = sha256Hex(
        stableJson({
          modelId: runOptions.modelId || config?.model?.modelId || null,
          modelUrl: runOptions.modelUrl || null,
        })
      );
      const datasetHash = sha256Hex(
        stableJson({
          batchSize: runOptions.batchSize ?? null,
          epochs: runOptions.epochs ?? null,
          maxSteps: runOptions.maxSteps ?? null,
        })
      );
      const buildProvenance = await resolveBuildProvenance();
      let latentDataset = null;
      if (resolvedStage === 'stage1_joint' && latentDatasetCount > 0) {
        const parsed = await readNdjson(latentDatasetPath);
        latentDataset = {
          path: relative(process.cwd(), latentDatasetPath),
          hash: sha256Hex(parsed.raw),
          count: parsed.entries.length,
          summary: summarizeLatentDataset(parsed.entries),
        };
      }
      const manifestBase = {
        schemaVersion: UL_MANIFEST_SCHEMA_VERSION,
        stage: resolvedStage,
        createdAt: new Date().toISOString(),
        runId: `${resolvedStage}_${timestamp}`,
        configHash,
        modelHash,
        datasetHash,
        ulContractHash,
        ulResolvedConfig: buildResolvedUlConfigSnapshot(ulConfig),
        runtimeDump: buildUlRuntimeDump(ulConfig, runOptions),
        buildProvenance,
        freeze: ulConfig.freeze,
        metrics: {
          count: Array.isArray(stepMetrics) ? stepMetrics.length : 0,
          stepMetricsPath: relative(process.cwd(), metricsPath),
        },
        latentDataset: latentDataset || stageDependency?.manifest?.latentDataset || null,
        lineage: {
          parentManifestHash: stageDependency?.manifest?.manifestHash || null,
          parentContractHash: stageDependency?.manifest?.ulContractHash || null,
        },
        stage1Dependency: stageDependency
          ? {
            path: relative(process.cwd(), stageDependency.path),
            hash: stageDependency.hash,
            manifestHash: stageDependency.manifest?.manifestHash || null,
          }
          : null,
      };
      const manifestContentHash = hashStableJson(
        buildDeterministicManifestView(manifestBase)
      );
      const manifestHash = manifestContentHash;
      const manifest = { ...manifestBase, manifestHash, manifestContentHash };
      await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
      const manifestFileHash = await computeFileHash(manifestPath);
      return {
        runDir: relative(process.cwd(), runDir),
        metricsPath: relative(process.cwd(), metricsPath),
        manifestPath: relative(process.cwd(), manifestPath),
        manifestHash,
        manifestContentHash,
        manifestFileHash,
        stage1Dependency: manifest.stage1Dependency,
      };
    },
  };
}
