#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { listTrackedFilesInDirectory } from './git-file-inventory.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');
const SUPPORT_INVENTORY_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'model-support-inventory.json');
const RELEASE_MATRIX_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'release-matrix.json');
const EMBEDDING_COMPARE_CONFIG_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'embedding-compare.config.json');
const RERANK_COMPARE_CONFIG_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'rerank-compare.config.json');
const EMBEDDING_RESULTS_DIR = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'results');
const RERANK_RESULTS_DIR = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'results');
const DEFAULT_JSON_OUTPUT_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'model-competition-scoreboard.json');
const DEFAULT_MARKDOWN_OUTPUT_PATH = path.join(REPO_ROOT, 'docs', 'model-competition-scoreboard.md');
const PLATFORM_IDS = Object.freeze(['browser', 'node', 'bun']);
const SOURCE_FILES = Object.freeze([
  'models/catalog.json',
  'benchmarks/vendors/model-support-inventory.json',
  'benchmarks/vendors/release-matrix.json',
  'benchmarks/vendors/embedding-compare.config.json',
  'benchmarks/vendors/rerank-compare.config.json',
  'benchmarks/vendors/results/embedding_compare_*.json',
  'benchmarks/vendors/results/rerank_compare_*.json',
]);

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function hasText(value) {
  return normalizeText(value).length > 0;
}

function hasFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function repoRelative(filePath) {
  return path.relative(REPO_ROOT, filePath).split(path.sep).join('/');
}

function isRepoRelativePath(value) {
  const candidate = normalizeText(value);
  return Boolean(
    candidate
    && !path.isAbsolute(candidate)
    && !candidate.includes('\\')
    && !candidate.split('/').includes('..')
  );
}

async function pathExists(repoPath) {
  if (!isRepoRelativePath(repoPath)) return false;
  try {
    await fs.access(path.join(REPO_ROOT, repoPath));
    return true;
  } catch {
    return false;
  }
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function parseArgs(argv) {
  const args = {
    check: false,
    jsonOutputPath: DEFAULT_JSON_OUTPUT_PATH,
    markdownOutputPath: DEFAULT_MARKDOWN_OUTPUT_PATH,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const entry = argv[index];
    const nextValue = () => {
      const candidate = argv[index + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${entry}`);
      }
      index += 1;
      return path.resolve(REPO_ROOT, String(candidate).trim());
    };
    if (entry === '--check') {
      args.check = true;
      continue;
    }
    if (entry === '--json-output') {
      args.jsonOutputPath = nextValue();
      continue;
    }
    if (entry === '--markdown-output') {
      args.markdownOutputPath = nextValue();
      continue;
    }
    throw new Error(`Unknown argument: ${entry}`);
  }
  return args;
}

function latestTimestamp(...values) {
  const normalized = values
    .flat()
    .map((value) => normalizeText(value))
    .filter(Boolean)
    .map((value) => {
      if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return `${value}T00:00:00.000Z`;
      const parsed = Date.parse(value);
      return Number.isNaN(parsed) ? null : new Date(parsed).toISOString();
    })
    .filter(Boolean)
    .sort();
  return normalized.at(-1) || null;
}

function stableJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

async function writeOrCheck(filePath, content, check) {
  if (check) {
    let current = null;
    try {
      current = await fs.readFile(filePath, 'utf8');
    } catch {
      throw new Error(`${repoRelative(filePath)} is missing; run npm run support:competition:sync`);
    }
    if (current !== content) {
      throw new Error(`${repoRelative(filePath)} is stale; run npm run support:competition:sync`);
    }
    return;
  }
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content);
}

function mapByModelId(entries, getModelId) {
  const out = new Map();
  for (const entry of Array.isArray(entries) ? entries : []) {
    const modelId = normalizeText(getModelId(entry));
    if (modelId && !out.has(modelId)) out.set(modelId, entry);
  }
  return out;
}

function buildCatalogModelIndex(catalog) {
  return mapByModelId(catalog?.models, (entry) => entry?.modelId);
}

function catalogVariant(model) {
  const surfaces = Array.isArray(model?.lifecycle?.tested?.surface)
    ? model.lifecycle.tested.surface.map(normalizeText).filter(Boolean)
    : [];
  const tested = normalizeText(model?.lifecycle?.status?.tested);
  return {
    modelId: normalizeText(model?.modelId),
    label: normalizeText(model?.label || model?.modelId),
    family: normalizeText(model?.family) || 'unknown',
    modes: Array.isArray(model?.modes) ? model.modes.map(normalizeText).filter(Boolean) : [],
    sizeBytes: hasFiniteNumber(model?.sizeBytes) ? model.sizeBytes : null,
    sizeLabel: describeBytes(model?.sizeBytes),
    sourceCheckpointId: normalizeText(model?.sourceCheckpointId) || null,
    architecture: {
      artifactFormat: normalizeText(model?.artifact?.format) || null,
      label: normalizeText(model?.artifact?.format) ? normalizeText(model.artifact.format).toUpperCase() : 'architecture metadata unavailable',
    },
    lifecycle: {
      tested: tested || null,
      result: normalizeText(model?.lifecycle?.tested?.result) || null,
      lastVerifiedAt: normalizeText(model?.lifecycle?.tested?.lastVerifiedAt) || null,
      surfaces,
    },
    hf: {
      published: model?.lifecycle?.availability?.hf === true,
      repoId: normalizeText(model?.hf?.repoId) || null,
      revision: normalizeText(model?.hf?.revision) || null,
      path: normalizeText(model?.hf?.path) || null,
    },
    compare: {
      profile: null,
      benchmarkComparable: false,
      benchmarkEvidenceOk: false,
    },
    evidence: {
      runtimeReport: null,
      compareResult: null,
      summarySvg: null,
      localClaimLaneId: null,
    },
    actions: {
      verifyCommand: normalizeText(model?.modelId) ? `node tools/run-registry-verify.js ${model.modelId} --surface auto` : null,
      hfDryRunCommand: normalizeText(model?.modelId) ? `node tools/publish-hf-registry-model.js --model-id ${model.modelId} --dry-run` : null,
      compareCommands: [],
      primaryNextCommand: null,
    },
    missing: [],
    nextGate: tested === 'verified' ? 'compare-profile' : 'runtime-verify',
  };
}

function flattenInventoryVariants(inventory, catalog) {
  const catalogByModelId = buildCatalogModelIndex(catalog);
  const variants = [];
  for (const source of Array.isArray(inventory?.sourceModels) ? inventory.sourceModels : []) {
    for (const variant of Array.isArray(source?.variants) ? source.variants : []) {
      variants.push({
        ...variant,
        sourceStatus: normalizeText(source?.status) || null,
        sourceNextAction: normalizeText(source?.nextAction) || null,
      });
    }
  }
  const seen = new Set(variants.map((variant) => variant.modelId));
  for (const model of Array.isArray(catalog?.models) ? catalog.models : []) {
    const modelId = normalizeText(model?.modelId);
    if (!modelId || seen.has(modelId)) continue;
    variants.push(catalogVariant(model));
  }
  variants.sort(compareVariants);
  return variants.map((variant) => ({
    ...variant,
    catalog: catalogByModelId.get(variant.modelId) || null,
  }));
}

function compareVariants(left, right) {
  const leftFamily = normalizeText(left.family);
  const rightFamily = normalizeText(right.family);
  if (leftFamily !== rightFamily) return leftFamily.localeCompare(rightFamily);
  const leftSize = hasFiniteNumber(left.sizeBytes) ? left.sizeBytes : Number.MAX_SAFE_INTEGER;
  const rightSize = hasFiniteNumber(right.sizeBytes) ? right.sizeBytes : Number.MAX_SAFE_INTEGER;
  if (leftSize !== rightSize) return leftSize - rightSize;
  return normalizeText(left.modelId).localeCompare(normalizeText(right.modelId));
}

function buildEmbeddingProfileIndex(config) {
  return mapByModelId(config?.modelProfiles, (entry) => entry?.dopplerModelId);
}

function buildRerankProfileIndex(config) {
  return mapByModelId(config?.modelProfiles, (entry) => entry?.dopplerModelId);
}

function timestampScore(value) {
  const parsed = Date.parse(normalizeText(value));
  return Number.isNaN(parsed) ? 0 : parsed;
}

async function collectLatestEmbeddingResults() {
  const byModelId = new Map();
  const files = listTrackedFilesInDirectory(REPO_ROOT, EMBEDDING_RESULTS_DIR);
  for (const fullPath of files) {
    const fileName = path.basename(fullPath);
    if (!fileName.startsWith('embedding_compare_') || !fileName.endsWith('.json')) continue;
    const payload = await readJson(fullPath);
    const modelId = normalizeText(payload?.model?.dopplerModelId);
    if (!modelId) continue;
    const record = {
      path: repoRelative(fullPath),
      timestamp: normalizeText(payload?.timestamp) || null,
      isLatestAlias: fileName === 'embedding_compare_latest.json',
      payload,
    };
    const current = byModelId.get(modelId);
    const newer = !current || timestampScore(record.timestamp) > timestampScore(current.timestamp);
    const sameTimestampPreferred = current
      && timestampScore(record.timestamp) === timestampScore(current.timestamp)
      && current.isLatestAlias
      && !record.isLatestAlias;
    if (newer || sameTimestampPreferred) byModelId.set(modelId, record);
  }
  return byModelId;
}

async function collectLatestRerankResults() {
  const byModelId = new Map();
  const files = listTrackedFilesInDirectory(REPO_ROOT, RERANK_RESULTS_DIR);
  for (const fullPath of files) {
    const fileName = path.basename(fullPath);
    if (!fileName.startsWith('rerank_compare_') || !fileName.endsWith('.json')) continue;
    const payload = await readJson(fullPath);
    const modelId = normalizeText(payload?.model?.dopplerModelId);
    if (!modelId) continue;
    const record = {
      path: repoRelative(fullPath),
      timestamp: normalizeText(payload?.timestamp) || null,
      isLatestAlias: fileName === 'rerank_compare_latest.json',
      payload,
    };
    const current = byModelId.get(modelId);
    const newer = !current || timestampScore(record.timestamp) > timestampScore(current.timestamp);
    const sameTimestampPreferred = current
      && timestampScore(record.timestamp) === timestampScore(current.timestamp)
      && current.isLatestAlias
      && !record.isLatestAlias;
    if (newer || sameTimestampPreferred) byModelId.set(modelId, record);
  }
  return byModelId;
}

function describeBytes(value) {
  if (!hasFiniteNumber(value)) return 'unknown';
  const gib = value / 1073741824;
  return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
}

function roundNumber(value, digits = 2) {
  if (!hasFiniteNumber(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function ratio(numerator, denominator) {
  if (!hasFiniteNumber(numerator) || !hasFiniteNumber(denominator) || denominator === 0) {
    return null;
  }
  return roundNumber(numerator / denominator, 4);
}

function variantLifecycleSurfaces(variant) {
  return Array.isArray(variant?.lifecycle?.surfaces)
    ? variant.lifecycle.surfaces.map(normalizeText).filter(Boolean)
    : [];
}

function platformStatus(variant, benchmarkSurfaces = []) {
  const verified = new Set(variantLifecycleSurfaces(variant));
  const benchmarked = new Set(benchmarkSurfaces.map(normalizeText).filter(Boolean));
  const out = {};
  for (const platform of PLATFORM_IDS) {
    if (benchmarked.has(platform)) {
      out[platform] = 'benchmarked';
      continue;
    }
    out[platform] = verified.has(platform) ? 'verified' : 'missing';
  }
  return out;
}

function buildArtifact(variant) {
  return {
    format: normalizeText(variant?.architecture?.artifactFormat) || null,
    architecture: normalizeText(variant?.architecture?.label) || null,
    sizeBytes: hasFiniteNumber(variant?.sizeBytes) ? variant.sizeBytes : null,
    sizeLabel: normalizeText(variant?.sizeLabel) || describeBytes(variant?.sizeBytes),
    hf: {
      published: variant?.hf?.published === true,
      repoId: normalizeText(variant?.hf?.repoId) || null,
      revision: normalizeText(variant?.hf?.revision) || null,
      path: normalizeText(variant?.hf?.path) || null,
    },
  };
}

function resolveTjsModelId(variant, coverage, embeddingResult = null) {
  return normalizeText(embeddingResult?.payload?.model?.transformersjsModelId)
    || normalizeText(variant?.compare?.profile?.tjsModelId)
    || normalizeText(coverage?.defaultTjsModelId)
    || normalizeText(variant?.catalog?.vendorBenchmark?.transformersjs?.repoId)
    || null;
}

function resolveTjsDtype(variant, embeddingResult = null) {
  return normalizeText(embeddingResult?.payload?.model?.transformersjsDtype)
    || normalizeText(variant?.compare?.profile?.tjsDtype)
    || normalizeText(variant?.catalog?.vendorBenchmark?.transformersjs?.dtype)
    || null;
}

function resolveTjsFormat(variant, embeddingResult = null) {
  return normalizeText(embeddingResult?.payload?.model?.transformersjsFormat)
    || normalizeText(variant?.compare?.profile?.tjsFormat)
    || null;
}

function buildCompetitor(variant, coverage, embeddingResult = null) {
  return {
    engine: 'transformersjs',
    modelId: resolveTjsModelId(variant, coverage, embeddingResult),
    format: resolveTjsFormat(variant, embeddingResult),
    dtype: resolveTjsDtype(variant, embeddingResult),
  };
}

function baseRow(variant, coverage, fields) {
  return {
    rowId: fields.rowId,
    modelId: variant.modelId,
    label: normalizeText(variant.label) || variant.modelId,
    family: normalizeText(variant.family) || 'unknown',
    modes: Array.isArray(variant.modes) ? variant.modes : [],
    sourceCheckpointId: normalizeText(variant.sourceCheckpointId) || null,
    workloadMode: fields.workloadMode,
    surface: fields.surface || null,
    workloadId: fields.workloadId || null,
    decodeProfile: fields.decodeProfile || null,
    artifact: buildArtifact(variant),
    platforms: fields.platforms || platformStatus(variant),
    competitor: fields.competitor || buildCompetitor(variant, coverage),
    correctness: fields.correctness || null,
    metrics: fields.metrics || null,
    claimStatus: fields.claimStatus,
    claimReason: fields.claimReason || null,
    evidence: fields.evidence || {},
    missing: fields.missing || [],
    nextGate: fields.nextGate || null,
    nextCommand: fields.nextCommand || null,
  };
}

function generationClaimStatus(lane, surface) {
  if (lane?.claimReady === true) return 'claim-ready';
  if (!surface) return normalizeText(lane?.status) || 'compare-missing';
  if (!hasText(surface?.compareResult)) return 'compare-result-missing';
  if (!hasText(surface?.summarySvg)) return 'summary-svg-missing';
  return normalizeText(lane?.status) || 'candidate';
}

async function generationEvidence(surface, lane, variant) {
  const compareResult = normalizeText(surface?.compareResult);
  const summarySvg = normalizeText(surface?.summarySvg);
  return {
    runtimeReport: normalizeText(variant?.evidence?.runtimeReport) || null,
    compareResult: compareResult || null,
    compareResultExists: compareResult ? await pathExists(compareResult) : false,
    summarySvg: summarySvg || null,
    summarySvgExists: summarySvg ? await pathExists(summarySvg) : false,
    localClaimLaneId: normalizeText(lane?.laneId) || normalizeText(variant?.evidence?.localClaimLaneId) || null,
  };
}

function generationMissing(surface, lane) {
  if (!surface) {
    return [
      ...(Array.isArray(lane?.missingBackendIds) ? lane.missingBackendIds : []),
      ...(Array.isArray(lane?.missingWorkloadIds) ? lane.missingWorkloadIds : []),
      ...(Array.isArray(lane?.missingDecodeProfileIds) ? lane.missingDecodeProfileIds : []),
      ...(Array.isArray(lane?.missingSurfaceWorkloads) ? lane.missingSurfaceWorkloads : []),
    ].map(normalizeText).filter(Boolean);
  }
  const missing = [];
  if (!hasText(surface?.compareResult)) missing.push('compare-result');
  if (!hasText(surface?.summarySvg)) missing.push('summary-svg');
  return missing;
}

function generationMetrics(surface) {
  if (!surface) return null;
  const dopplerDecode = surface.dopplerDecodeTokensPerSec;
  const tjsDecode = surface.transformersjsDecodeTokensPerSec;
  const dopplerPrompt = surface.dopplerPromptTokensPerSecToFirstToken;
  const tjsPrompt = surface.transformersjsPromptTokensPerSecToFirstToken;
  return {
    doppler: {
      decodeTokensPerSec: roundNumber(dopplerDecode),
      promptTokensPerSecToFirstToken: roundNumber(dopplerPrompt),
    },
    transformersjs: {
      decodeTokensPerSec: roundNumber(tjsDecode),
      promptTokensPerSecToFirstToken: roundNumber(tjsPrompt),
    },
    ratios: {
      decodeDopplerOverTransformersjs: ratio(dopplerDecode, tjsDecode),
      promptDopplerOverTransformersjs: ratio(dopplerPrompt, tjsPrompt),
    },
    leader: {
      decode: normalizeText(surface.decodeLeader) || null,
      prompt: normalizeText(surface.promptLeader) || null,
    },
    bottleneck: normalizeText(surface.bottleneck) || null,
    bottleneckClass: normalizeText(surface.bottleneckClass) || null,
  };
}

async function buildGenerationRows(variantsByModelId, coverageByModelId, lanesByModelId) {
  const rows = [];
  for (const lane of lanesByModelId.values()) {
    const modelId = normalizeText(lane?.dopplerModelId);
    const variant = variantsByModelId.get(modelId);
    if (!variant) continue;
    const coverage = coverageByModelId.get(modelId) || null;
    const surfaces = Array.isArray(lane?.surfaces) ? lane.surfaces : [];
    if (surfaces.length === 0) {
      rows.push(baseRow(variant, coverage, {
        rowId: `${modelId}:generation:pending`,
        workloadMode: 'generation',
        claimStatus: generationClaimStatus(lane, null),
        claimReason: normalizeText(lane?.statusReason) || null,
        correctness: normalizeText(variant?.lifecycle?.result) || null,
        evidence: {
          runtimeReport: normalizeText(variant?.evidence?.runtimeReport) || null,
          compareResult: normalizeText(variant?.evidence?.compareResult) || null,
          summarySvg: normalizeText(variant?.evidence?.summarySvg) || null,
          localClaimLaneId: normalizeText(lane?.laneId) || null,
        },
        missing: generationMissing(null, lane),
        nextGate: normalizeText(variant?.nextGate) || 'compare-result',
        nextCommand: normalizeText(variant?.actions?.primaryNextCommand) || null,
      }));
      continue;
    }
    for (const surface of surfaces) {
      const missing = generationMissing(surface, lane);
      rows.push(baseRow(variant, coverage, {
        rowId: [
          modelId,
          'generation',
          normalizeText(surface?.surface) || 'unknown-surface',
          normalizeText(surface?.workloadId) || 'unknown-workload',
          normalizeText(surface?.decodeProfile) || 'unknown-decode-profile',
        ].join(':'),
        workloadMode: 'generation',
        surface: normalizeText(surface?.surface) || null,
        workloadId: normalizeText(surface?.workloadId) || null,
        decodeProfile: normalizeText(surface?.decodeProfile) || null,
        platforms: platformStatus(variant, [surface?.surface]),
        correctness: normalizeText(surface?.correctness) || null,
        metrics: generationMetrics(surface),
        claimStatus: generationClaimStatus(lane, surface),
        claimReason: normalizeText(lane?.statusReason) || null,
        evidence: await generationEvidence(surface, lane, variant),
        missing,
        nextGate: missing[0] || normalizeText(variant?.nextGate) || null,
        nextCommand: missing.length > 0
          ? normalizeText(variant?.actions?.compareCommands?.[0]?.command || variant?.actions?.primaryNextCommand) || null
          : null,
      }));
    }
  }
  return rows;
}

function embeddingMetrics(result) {
  const summary = result?.payload?.summary;
  const doppler = summary?.doppler?.speed || {};
  const tjs = summary?.transformersjs?.speed || {};
  const dopplerMedian = doppler.medianEmbeddingMs;
  const tjsMedian = tjs.medianEmbeddingMs;
  const dopplerThroughput = doppler.avgEmbeddingsPerSec;
  const tjsThroughput = tjs.avgEmbeddingsPerSec;
  const dopplerLoad = doppler.modelLoadMs;
  const tjsLoad = tjs.modelLoadMs;
  return {
    doppler: {
      medianEmbeddingMs: roundNumber(dopplerMedian),
      avgEmbeddingsPerSec: roundNumber(dopplerThroughput),
      p95EmbeddingMs: roundNumber(doppler.p95EmbeddingMs),
      modelLoadMs: roundNumber(dopplerLoad),
      embeddingDim: hasFiniteNumber(doppler.embeddingDim) ? doppler.embeddingDim : null,
    },
    transformersjs: {
      medianEmbeddingMs: roundNumber(tjsMedian),
      avgEmbeddingsPerSec: roundNumber(tjsThroughput),
      p95EmbeddingMs: roundNumber(tjs.p95EmbeddingMs),
      modelLoadMs: roundNumber(tjsLoad),
      embeddingDim: hasFiniteNumber(tjs.embeddingDim) ? tjs.embeddingDim : null,
    },
    ratios: {
      medianEmbeddingMsDopplerOverTransformersjs: ratio(dopplerMedian, tjsMedian),
      avgEmbeddingsPerSecDopplerOverTransformersjs: ratio(dopplerThroughput, tjsThroughput),
      modelLoadMsDopplerOverTransformersjs: ratio(dopplerLoad, tjsLoad),
    },
    leader: {
      embeddingLatency: hasFiniteNumber(dopplerMedian) && hasFiniteNumber(tjsMedian)
        ? (dopplerMedian <= tjsMedian ? 'doppler' : 'transformersjs')
        : null,
      embeddingThroughput: hasFiniteNumber(dopplerThroughput) && hasFiniteNumber(tjsThroughput)
        ? (dopplerThroughput >= tjsThroughput ? 'doppler' : 'transformersjs')
        : null,
      modelLoad: hasFiniteNumber(dopplerLoad) && hasFiniteNumber(tjsLoad)
        ? (dopplerLoad <= tjsLoad ? 'doppler' : 'transformersjs')
        : null,
    },
  };
}

function rerankMetrics(result) {
  const summary = result?.payload?.summary;
  const doppler = summary?.doppler?.speed || {};
  const tjs = summary?.transformersjs?.speed || {};
  const dopplerMedian = doppler.medianRerankMs;
  const tjsMedian = tjs.medianRerankMs;
  const dopplerThroughput = doppler.avgReranksPerSec;
  const tjsThroughput = tjs.avgReranksPerSec;
  const dopplerLoad = doppler.modelLoadMs;
  const tjsLoad = tjs.modelLoadMs;
  return {
    doppler: {
      medianRerankMs: roundNumber(dopplerMedian),
      avgReranksPerSec: roundNumber(dopplerThroughput),
      p95RerankMs: roundNumber(doppler.p95RerankMs),
      modelLoadMs: roundNumber(dopplerLoad),
      topDocumentIndex: hasFiniteNumber(doppler.topDocumentIndex) ? doppler.topDocumentIndex : null,
    },
    transformersjs: {
      medianRerankMs: roundNumber(tjsMedian),
      avgReranksPerSec: roundNumber(tjsThroughput),
      p95RerankMs: roundNumber(tjs.p95RerankMs),
      modelLoadMs: roundNumber(tjsLoad),
      topDocumentIndex: hasFiniteNumber(tjs.topDocumentIndex) ? tjs.topDocumentIndex : null,
    },
    ratios: {
      medianRerankMsDopplerOverTransformersjs: ratio(dopplerMedian, tjsMedian),
      avgReranksPerSecDopplerOverTransformersjs: ratio(dopplerThroughput, tjsThroughput),
      modelLoadMsDopplerOverTransformersjs: ratio(dopplerLoad, tjsLoad),
    },
    leader: {
      rerankLatency: hasFiniteNumber(dopplerMedian) && hasFiniteNumber(tjsMedian)
        ? (dopplerMedian <= tjsMedian ? 'doppler' : 'transformersjs')
        : null,
      rerankThroughput: hasFiniteNumber(dopplerThroughput) && hasFiniteNumber(tjsThroughput)
        ? (dopplerThroughput >= tjsThroughput ? 'doppler' : 'transformersjs')
        : null,
      modelLoad: hasFiniteNumber(dopplerLoad) && hasFiniteNumber(tjsLoad)
        ? (dopplerLoad <= tjsLoad ? 'doppler' : 'transformersjs')
        : null,
    },
  };
}

function embeddingClaimStatus(result, profile) {
  const compareLane = result?.payload?.compareLane || {};
  if (compareLane.claimable === true || profile?.releaseClaimable === true) return 'claimable';
  if (compareLane.localComparable === true) return 'local-comparable';
  if (compareLane.correctnessOk === false) return 'correctness-failed';
  return normalizeText(compareLane.lane || profile?.compareLane) || 'compare-missing';
}

function embeddingNextGate(claimStatus, result, variant) {
  if (claimStatus !== 'local-comparable') return null;
  if (variant?.hf?.published !== true) return 'hf-publish';
  if (normalizeText(result?.payload?.model?.dopplerSource) === 'local') {
    return 'published-artifact-compare';
  }
  return 'release-claim-promotion';
}

function embeddingNextCommand(modelId, nextGate, variant) {
  if (!nextGate) return null;
  if (nextGate === 'hf-publish') {
    return normalizeText(variant?.actions?.hfDryRunCommand) || null;
  }
  if (nextGate === 'published-artifact-compare') {
    return `node tools/compare-embeddings.js --model-id ${modelId} --doppler-source quickstart-registry --save --json`;
  }
  return normalizeText(variant?.actions?.primaryNextCommand) || null;
}

function embeddingClaimReason(result, profile, nextGate) {
  if (nextGate === 'hf-publish') {
    return 'Comparable embedding evidence exists, but the catalog row is not marked as published on Hugging Face.';
  }
  if (nextGate === 'published-artifact-compare') {
    return 'Local Doppler artifact compared against the browser WebGPU Transformers.js ONNX embedding baseline; rerun the compare against the hosted quickstart registry artifact.';
  }
  if (nextGate === 'release-claim-promotion') {
    return 'Published Doppler artifact compared against the browser WebGPU Transformers.js ONNX embedding baseline; promote the compare receipt before claiming release-level competitor performance.';
  }
  return normalizeText(result?.payload?.compareLane?.reason || profile?.compareLaneReason) || null;
}

function rerankClaimStatus(result, profile) {
  const compareLane = result?.payload?.compareLane || {};
  if (compareLane.claimable === true || compareLane.releaseClaimable === true) return 'claimable';
  if (compareLane.localComparable === true) return 'local-comparable';
  if (compareLane.correctnessOk === false) return 'correctness-failed';
  return normalizeText(compareLane.lane || profile?.compareLane) || 'compare-missing';
}

function rerankNextGate(claimStatus, result, variant) {
  if (claimStatus !== 'local-comparable') return null;
  if (variant?.hf?.published !== true) return 'hf-publish';
  if (normalizeText(result?.payload?.model?.dopplerSource) === 'local') {
    return 'published-artifact-compare';
  }
  return 'release-claim-promotion';
}

function rerankNextCommand(modelId, nextGate, variant) {
  if (!nextGate) return null;
  if (nextGate === 'hf-publish') {
    return normalizeText(variant?.actions?.hfDryRunCommand) || null;
  }
  if (nextGate === 'published-artifact-compare') {
    return `node tools/compare-rerankers.js --model-id ${modelId} --doppler-source quickstart-registry --save --json`;
  }
  return normalizeText(variant?.actions?.primaryNextCommand) || null;
}

function rerankClaimReason(result, profile, nextGate) {
  if (nextGate === 'hf-publish') {
    return 'Comparable rerank evidence exists, but the catalog row is not marked as published on Hugging Face.';
  }
  if (nextGate === 'published-artifact-compare') {
    return 'Local Doppler artifact compared against the browser WebGPU Transformers.js ONNX rerank baseline; rerun the compare against the hosted quickstart registry artifact.';
  }
  if (nextGate === 'release-claim-promotion') {
    return 'Published Doppler artifact compared against the browser WebGPU Transformers.js ONNX rerank baseline; promote the compare receipt before claiming release-level competitor performance.';
  }
  return normalizeText(result?.payload?.compareLane?.reason || profile?.compareLaneReason) || null;
}

async function buildRerankRows(variantsByModelId, coverageByModelId, rerankProfilesByModelId, latestRerankResults) {
  const rows = [];
  for (const [modelId, result] of latestRerankResults) {
    const variant = variantsByModelId.get(modelId);
    if (!variant) continue;
    const coverage = coverageByModelId.get(modelId) || null;
    const profile = rerankProfilesByModelId.get(modelId) || null;
    const correctnessOk = result?.payload?.summary?.correctnessOk === true;
    const claimStatus = rerankClaimStatus(result, profile);
    const nextGate = rerankNextGate(claimStatus, result, variant);
    const evidencePath = normalizeText(result?.path);
    rows.push(baseRow(variant, coverage, {
      rowId: `${modelId}:rerank:${normalizeText(result?.timestamp) || 'latest'}`,
      workloadMode: 'rerank',
      surface: normalizeText(result?.payload?.model?.dopplerSurface) || normalizeText(profile?.defaultDopplerSurface) || 'browser',
      workloadId: 'rerank',
      competitor: buildCompetitor(variant, coverage, result),
      correctness: correctnessOk ? 'semantic-pass' : 'failed',
      metrics: rerankMetrics(result),
      claimStatus,
      claimReason: rerankClaimReason(result, profile, nextGate),
      evidence: {
        rerankCompareResult: evidencePath || null,
        rerankCompareResultExists: evidencePath ? await pathExists(evidencePath) : false,
        timestamp: normalizeText(result?.timestamp) || null,
      },
      missing: nextGate ? [nextGate] : [],
      nextGate,
      nextCommand: rerankNextCommand(modelId, nextGate, variant),
    }));
  }
  return rows;
}

async function buildEmbeddingRows(variantsByModelId, coverageByModelId, embeddingProfilesByModelId, latestEmbeddingResults) {
  const rows = [];
  for (const [modelId, result] of latestEmbeddingResults) {
    const variant = variantsByModelId.get(modelId);
    if (!variant) continue;
    const coverage = coverageByModelId.get(modelId) || null;
    const profile = embeddingProfilesByModelId.get(modelId) || null;
    const correctnessOk = result?.payload?.summary?.correctnessOk === true;
    const claimStatus = embeddingClaimStatus(result, profile);
    const nextGate = embeddingNextGate(claimStatus, result, variant);
    const evidencePath = normalizeText(result?.path);
    rows.push(baseRow(variant, coverage, {
      rowId: `${modelId}:embedding:${normalizeText(result?.timestamp) || 'latest'}`,
      workloadMode: 'embedding',
      surface: normalizeText(result?.payload?.model?.dopplerSurface) || normalizeText(profile?.defaultDopplerSurface) || 'auto',
      workloadId: 'embedding',
      competitor: buildCompetitor(variant, coverage, result),
      correctness: correctnessOk ? 'semantic-pass' : 'failed',
      metrics: embeddingMetrics(result),
      claimStatus,
      claimReason: embeddingClaimReason(result, profile, nextGate),
      evidence: {
        embeddingCompareResult: evidencePath || null,
        embeddingCompareResultExists: evidencePath ? await pathExists(evidencePath) : false,
        timestamp: normalizeText(result?.timestamp) || null,
      },
      missing: nextGate ? [nextGate] : [],
      nextGate,
      nextCommand: embeddingNextCommand(modelId, nextGate, variant),
    }));
  }
  return rows;
}

function supportClaimStatus(variant) {
  const tested = normalizeText(variant?.lifecycle?.tested);
  if (tested === 'failed') return 'failed';
  if (tested !== 'verified') return 'verification-needed';
  if (variant?.compare?.benchmarkEvidenceOk === true) return 'benchmark-selected';
  if (variant?.compare?.benchmarkComparable === true) return 'compare-missing';
  if (variant?.compare?.profile) return 'capability-only';
  return 'verified-no-compare';
}

function buildSupportRows(variants, coverageByModelId, comparedModelIds) {
  const rows = [];
  for (const variant of variants) {
    if (comparedModelIds.has(variant.modelId)) continue;
    const coverage = coverageByModelId.get(variant.modelId) || null;
    const claimStatus = supportClaimStatus(variant);
    rows.push(baseRow(variant, coverage, {
      rowId: `${variant.modelId}:support`,
      workloadMode: Array.isArray(variant.modes) && variant.modes.length === 1 ? variant.modes[0] : 'support',
      claimStatus,
      claimReason: normalizeText(variant?.sourceNextAction) || null,
      correctness: normalizeText(variant?.lifecycle?.result || variant?.lifecycle?.tested) || null,
      evidence: {
        runtimeReport: normalizeText(variant?.evidence?.runtimeReport) || null,
        compareResult: normalizeText(variant?.evidence?.compareResult) || null,
        summarySvg: normalizeText(variant?.evidence?.summarySvg) || null,
        localClaimLaneId: normalizeText(variant?.evidence?.localClaimLaneId) || null,
      },
      missing: Array.isArray(variant.missing) ? variant.missing.map(normalizeText).filter(Boolean) : [],
      nextGate: normalizeText(variant?.nextGate) || null,
      nextCommand: normalizeText(variant?.actions?.primaryNextCommand) || null,
    }));
  }
  return rows;
}

function summarizeRows(catalog, variants, rows) {
  const generationBenchmarks = rows.filter((row) => row.workloadMode === 'generation' && row.metrics);
  const generationGaps = rows.filter((row) => row.workloadMode === 'generation' && !row.metrics);
  const embeddingRows = rows.filter((row) => row.workloadMode === 'embedding' && row.metrics);
  const rerankRows = rows.filter((row) => row.workloadMode === 'rerank' && row.metrics);
  const supportRows = rows.filter((row) => !row.metrics && row.workloadMode !== 'generation');
  const runtimeVerified = variants.filter((variant) => normalizeText(variant?.lifecycle?.tested) === 'verified');
  const platformCounts = {};
  for (const platform of PLATFORM_IDS) {
    platformCounts[platform] = variants.filter((variant) => variantLifecycleSurfaces(variant).includes(platform)).length;
  }
  return {
    catalogModelCount: Array.isArray(catalog?.models) ? catalog.models.length : variants.length,
    scoreboardRowCount: rows.length,
    runtimeVerifiedModelCount: runtimeVerified.length,
    hfPublishedModelCount: variants.filter((variant) => variant?.hf?.published === true).length,
    failedModelCount: variants.filter((variant) => normalizeText(variant?.lifecycle?.tested) === 'failed').length,
    verificationNeededModelCount: variants.filter((variant) => normalizeText(variant?.lifecycle?.tested) !== 'verified' && normalizeText(variant?.lifecycle?.tested) !== 'failed').length,
    platformVerifiedModelCount: platformCounts,
    generationCompareRowCount: generationBenchmarks.length,
    generationCompareGapRowCount: generationGaps.length,
    embeddingCompareRowCount: embeddingRows.length,
    rerankCompareRowCount: rerankRows.length,
    supportGapRowCount: supportRows.filter((row) => row.claimStatus !== 'benchmark-selected').length,
    dopplerDecodeLeaderRowCount: generationBenchmarks.filter((row) => row.metrics?.leader?.decode === 'doppler').length,
    transformersjsDecodeLeaderRowCount: generationBenchmarks.filter((row) => row.metrics?.leader?.decode === 'transformersjs').length,
    dopplerEmbeddingLatencyLeaderRowCount: embeddingRows.filter((row) => row.metrics?.leader?.embeddingLatency === 'doppler').length,
    transformersjsEmbeddingLatencyLeaderRowCount: embeddingRows.filter((row) => row.metrics?.leader?.embeddingLatency === 'transformersjs').length,
    dopplerRerankLatencyLeaderRowCount: rerankRows.filter((row) => row.metrics?.leader?.rerankLatency === 'doppler').length,
    transformersjsRerankLatencyLeaderRowCount: rerankRows.filter((row) => row.metrics?.leader?.rerankLatency === 'transformersjs').length,
    claimReadyRowCount: rows.filter((row) => row.claimStatus === 'claim-ready' || row.claimStatus === 'claimable').length,
    localComparableRowCount: rows.filter((row) => row.claimStatus === 'local-comparable').length,
    evidenceIncompleteRowCount: rows.filter((row) => Array.isArray(row.missing) && row.missing.length > 0).length,
  };
}

function sortRows(rows) {
  return rows.sort((left, right) => {
    const leftFamily = normalizeText(left.family);
    const rightFamily = normalizeText(right.family);
    if (leftFamily !== rightFamily) return leftFamily.localeCompare(rightFamily);
    if (left.modelId !== right.modelId) return left.modelId.localeCompare(right.modelId);
    const leftMode = normalizeText(left.workloadMode);
    const rightMode = normalizeText(right.workloadMode);
    if (leftMode !== rightMode) return leftMode.localeCompare(rightMode);
    return normalizeText(left.rowId).localeCompare(normalizeText(right.rowId));
  });
}

async function buildScoreboard() {
  const [
    catalog,
    supportInventory,
    releaseMatrix,
    embeddingCompareConfig,
    rerankCompareConfig,
    latestEmbeddingResults,
    latestRerankResults,
  ] = await Promise.all([
    readJson(CATALOG_PATH),
    readJson(SUPPORT_INVENTORY_PATH),
    readJson(RELEASE_MATRIX_PATH),
    readJson(EMBEDDING_COMPARE_CONFIG_PATH),
    readJson(RERANK_COMPARE_CONFIG_PATH),
    collectLatestEmbeddingResults(),
    collectLatestRerankResults(),
  ]);
  const variants = flattenInventoryVariants(supportInventory, catalog);
  const variantsByModelId = mapByModelId(variants, (entry) => entry?.modelId);
  const coverageByModelId = mapByModelId(releaseMatrix?.modelCoverage, (entry) => entry?.dopplerModelId);
  const lanesByModelId = mapByModelId(releaseMatrix?.localClaimLanes, (entry) => entry?.dopplerModelId);
  const embeddingProfilesByModelId = buildEmbeddingProfileIndex(embeddingCompareConfig);
  const rerankProfilesByModelId = buildRerankProfileIndex(rerankCompareConfig);
  const generationRows = await buildGenerationRows(variantsByModelId, coverageByModelId, lanesByModelId);
  const embeddingRows = await buildEmbeddingRows(
    variantsByModelId,
    coverageByModelId,
    embeddingProfilesByModelId,
    latestEmbeddingResults
  );
  const rerankRows = await buildRerankRows(
    variantsByModelId,
    coverageByModelId,
    rerankProfilesByModelId,
    latestRerankResults
  );
  const comparedModelIds = new Set([
    ...generationRows.map((row) => row.modelId),
    ...embeddingRows.map((row) => row.modelId),
    ...rerankRows.map((row) => row.modelId),
  ]);
  const supportRows = buildSupportRows(variants, coverageByModelId, comparedModelIds);
  const rows = sortRows([...generationRows, ...embeddingRows, ...rerankRows, ...supportRows]);
  return {
    schemaVersion: 1,
    updatedAt: latestTimestamp(
      catalog?.updatedAt,
      supportInventory?.updated,
      releaseMatrix?.generatedAt,
      [...latestEmbeddingResults.values()].map((entry) => entry.timestamp),
      [...latestRerankResults.values()].map((entry) => entry.timestamp)
    ),
    sources: SOURCE_FILES,
    summary: summarizeRows(catalog, variants, rows),
    rows,
  };
}

function renderTableRow(values) {
  return values
    .map((value) => String(value ?? '').replace(/\|/g, '\\|'))
    .join(' | ')
    .replace(/^/, '| ')
    .replace(/$/, ' |');
}

function metricValue(value, suffix = '') {
  if (!hasFiniteNumber(value)) return 'n/a';
  return `${value}${suffix}`;
}

function platformSummary(platforms) {
  return PLATFORM_IDS
    .map((platform) => `${platform}:${normalizeText(platforms?.[platform]) || 'missing'}`)
    .join('<br>');
}

function hfSummary(artifact) {
  if (artifact?.hf?.published !== true) return 'missing';
  return `${artifact.hf.repoId || 'HF'}@${artifact.hf.revision || 'unknown'}<br>${artifact.hf.path || 'path-missing'}`;
}

function competitorSummary(competitor) {
  const parts = [competitor?.engine || 'transformersjs'];
  if (competitor?.modelId) parts.push(competitor.modelId);
  if (competitor?.format || competitor?.dtype) {
    parts.push([competitor.format, competitor.dtype].filter(Boolean).join('/'));
  }
  return parts.join('<br>');
}

function evidenceSummary(evidence) {
  const entries = [
    evidence?.runtimeReport,
    evidence?.compareResult,
    evidence?.summarySvg,
    evidence?.embeddingCompareResult,
    evidence?.rerankCompareResult,
  ].filter(Boolean);
  return entries.length > 0 ? entries.join('<br>') : 'none';
}

function commandSummary(command) {
  if (!command) return 'none';
  return `\`${String(command).replace(/`/g, '\\`')}\``;
}

function renderGenerationRows(lines, rows) {
  lines.push('## Generation Competition Rows');
  lines.push('');
  lines.push('| Model | Surface | Workload | Correctness | Doppler decode | TJS decode | Decode leader | Doppler prompt | TJS prompt | Prompt leader | Bottleneck | Claim | Evidence |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const row of rows) {
    lines.push(renderTableRow([
      row.modelId,
      row.surface || 'pending',
      [row.workloadId, row.decodeProfile].filter(Boolean).join('<br>') || 'pending',
      row.correctness || 'missing',
      metricValue(row.metrics?.doppler?.decodeTokensPerSec, ' tok/s'),
      metricValue(row.metrics?.transformersjs?.decodeTokensPerSec, ' tok/s'),
      row.metrics?.leader?.decode || 'n/a',
      metricValue(row.metrics?.doppler?.promptTokensPerSecToFirstToken, ' tok/s'),
      metricValue(row.metrics?.transformersjs?.promptTokensPerSecToFirstToken, ' tok/s'),
      row.metrics?.leader?.prompt || 'n/a',
      row.metrics?.bottleneck || 'n/a',
      row.claimStatus,
      evidenceSummary(row.evidence),
    ]));
  }
  lines.push('');
}

function renderEmbeddingRows(lines, rows) {
  lines.push('## Embedding Competition Rows');
  lines.push('');
  lines.push('| Model | Correctness | Doppler median | TJS median | Latency leader | Doppler throughput | TJS throughput | Throughput leader | Load leader | Claim | Evidence |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const row of rows) {
    lines.push(renderTableRow([
      row.modelId,
      row.correctness || 'missing',
      metricValue(row.metrics?.doppler?.medianEmbeddingMs, ' ms'),
      metricValue(row.metrics?.transformersjs?.medianEmbeddingMs, ' ms'),
      row.metrics?.leader?.embeddingLatency || 'n/a',
      metricValue(row.metrics?.doppler?.avgEmbeddingsPerSec, ' emb/s'),
      metricValue(row.metrics?.transformersjs?.avgEmbeddingsPerSec, ' emb/s'),
      row.metrics?.leader?.embeddingThroughput || 'n/a',
      row.metrics?.leader?.modelLoad || 'n/a',
      row.claimStatus,
      evidenceSummary(row.evidence),
    ]));
  }
  lines.push('');
}

function renderRerankRows(lines, rows) {
  lines.push('## Rerank Competition Rows');
  lines.push('');
  lines.push('| Model | Correctness | Doppler median | TJS median | Latency leader | Doppler throughput | TJS throughput | Throughput leader | Load leader | Claim | Evidence |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const row of rows) {
    lines.push(renderTableRow([
      row.modelId,
      row.correctness || 'missing',
      metricValue(row.metrics?.doppler?.medianRerankMs, ' ms'),
      metricValue(row.metrics?.transformersjs?.medianRerankMs, ' ms'),
      row.metrics?.leader?.rerankLatency || 'n/a',
      metricValue(row.metrics?.doppler?.avgReranksPerSec, ' rerank/s'),
      metricValue(row.metrics?.transformersjs?.avgReranksPerSec, ' rerank/s'),
      row.metrics?.leader?.rerankThroughput || 'n/a',
      row.metrics?.leader?.modelLoad || 'n/a',
      row.claimStatus,
      evidenceSummary(row.evidence),
    ]));
  }
  lines.push('');
}

function renderGapRows(lines, rows) {
  lines.push('## Support And Competition Gaps');
  lines.push('');
  lines.push('| Model | Mode | HF | Platforms | Competitor | Claim/gate | Missing | Evidence |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const row of rows) {
    lines.push(renderTableRow([
      row.modelId,
      row.workloadMode,
      hfSummary(row.artifact),
      platformSummary(row.platforms),
      competitorSummary(row.competitor),
      [row.claimStatus, row.nextGate].filter(Boolean).join('<br>'),
      row.missing.length > 0 ? row.missing.join('<br>') : 'none',
      evidenceSummary(row.evidence),
    ]));
  }
  lines.push('');
}

function renderNextCommands(lines, rows) {
  const commandRows = [];
  const seen = new Set();
  for (const row of rows) {
    if (!hasText(row.nextCommand)) continue;
    const key = [row.modelId, row.nextGate || row.claimStatus, row.nextCommand].join('\0');
    if (seen.has(key)) continue;
    seen.add(key);
    commandRows.push(row);
  }
  lines.push('## Next Commands');
  lines.push('');
  lines.push('These commands are gates, not evidence. A row becomes evidence only after its saved artifact is committed and referenced by the catalog, release matrix, or support inventory.');
  lines.push('');
  lines.push('| Model | Gate | Command |');
  lines.push('| --- | --- | --- |');
  for (const row of commandRows) {
    lines.push(renderTableRow([
      row.modelId,
      row.nextGate || row.claimStatus,
      commandSummary(row.nextCommand),
    ]));
  }
  lines.push('');
}

function renderMarkdown(scoreboard) {
  const generationRows = scoreboard.rows.filter((row) => row.workloadMode === 'generation' && row.metrics);
  const embeddingRows = scoreboard.rows.filter((row) => row.workloadMode === 'embedding' && row.metrics);
  const rerankRows = scoreboard.rows.filter((row) => row.workloadMode === 'rerank' && row.metrics);
  const gapRows = scoreboard.rows.filter((row) => !row.metrics || row.missing.length > 0);
  const lines = [];
  lines.push('# Model Competition Scoreboard');
  lines.push('');
  lines.push('Generated from the catalog, support inventory, release matrix, embedding/rerank compare configs, and saved compare receipts.');
  lines.push('This file is an evidence ledger: it records what is verified, what is on Hugging Face according to catalog metadata, where Doppler has comparable performance receipts, and which gates remain.');
  lines.push('');
  lines.push(`Updated at: ${scoreboard.updatedAt || 'unknown'}`);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Catalog models: ${scoreboard.summary.catalogModelCount}`);
  lines.push(`- Runtime-verified models: ${scoreboard.summary.runtimeVerifiedModelCount}`);
  lines.push(`- HF-published models: ${scoreboard.summary.hfPublishedModelCount}`);
  lines.push(`- Failed models: ${scoreboard.summary.failedModelCount}`);
  lines.push(`- Verification-needed models: ${scoreboard.summary.verificationNeededModelCount}`);
  lines.push(`- Generation compare rows: ${scoreboard.summary.generationCompareRowCount}`);
  lines.push(`- Generation compare gap rows: ${scoreboard.summary.generationCompareGapRowCount}`);
  lines.push(`- Embedding compare rows: ${scoreboard.summary.embeddingCompareRowCount}`);
  lines.push(`- Rerank compare rows: ${scoreboard.summary.rerankCompareRowCount}`);
  lines.push(`- Doppler decode-leading generation rows: ${scoreboard.summary.dopplerDecodeLeaderRowCount}`);
  lines.push(`- Transformers.js decode-leading generation rows: ${scoreboard.summary.transformersjsDecodeLeaderRowCount}`);
  lines.push(`- Doppler embedding latency-leading rows: ${scoreboard.summary.dopplerEmbeddingLatencyLeaderRowCount}`);
  lines.push(`- Transformers.js embedding latency-leading rows: ${scoreboard.summary.transformersjsEmbeddingLatencyLeaderRowCount}`);
  lines.push(`- Doppler rerank latency-leading rows: ${scoreboard.summary.dopplerRerankLatencyLeaderRowCount}`);
  lines.push(`- Transformers.js rerank latency-leading rows: ${scoreboard.summary.transformersjsRerankLatencyLeaderRowCount}`);
  lines.push(`- Evidence-incomplete rows: ${scoreboard.summary.evidenceIncompleteRowCount}`);
  lines.push('');
  lines.push('## Claim Status Rules');
  lines.push('');
  lines.push('- `claim-ready` and `claimable` mean the row has enough committed evidence for that row-level claim.');
  lines.push('- `candidate`, `local-comparable`, and `*-missing` rows are useful engineering evidence, but not release claims.');
  lines.push('- Missing HF, runtime, compare JSON, or SVG evidence stays visible as a gate instead of being inferred from local state.');
  lines.push('');
  renderGenerationRows(lines, generationRows);
  renderEmbeddingRows(lines, embeddingRows);
  renderRerankRows(lines, rerankRows);
  renderGapRows(lines, gapRows);
  renderNextCommands(lines, scoreboard.rows);
  lines.push('## Source Files');
  lines.push('');
  for (const source of scoreboard.sources) {
    lines.push(`- ${source}`);
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const scoreboard = await buildScoreboard();
  await writeOrCheck(args.jsonOutputPath, stableJson(scoreboard), args.check);
  await writeOrCheck(args.markdownOutputPath, renderMarkdown(scoreboard), args.check);
  if (!args.check) {
    console.log(`Wrote ${repoRelative(args.jsonOutputPath)}`);
    console.log(`Wrote ${repoRelative(args.markdownOutputPath)}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
