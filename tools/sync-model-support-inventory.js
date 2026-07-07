#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const CONVERSION_CONFIG_DIR = path.join(REPO_ROOT, 'src', 'config', 'conversion');
const CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');
const SUPPORT_ROLLOUT_POLICY_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'support-rollout-policy.json');
const COMPARE_CONFIG_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'compare-engines.config.json');
const EMBEDDING_COMPARE_CONFIG_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'embedding-compare.config.json');
const RERANK_COMPARE_CONFIG_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'rerank-compare.config.json');
const CLAIM_MATRIX_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'local-inference-claim-matrix.json');
const RELEASE_MATRIX_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'release-matrix.json');
const RELEASE_CLAIM_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'release-claim-policy.json');
const WORKLOADS_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'workloads.json');
const BENCHMARK_POLICY_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'benchmark-policy.json');
const COMPARE_RESULTS_DIR = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'results');
const DEFAULT_JSON_OUTPUT_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'model-support-inventory.json');
const DEFAULT_MARKDOWN_OUTPUT_PATH = path.join(REPO_ROOT, 'docs', 'model-support-inventory.md');

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function latestSourceDate(...values) {
  const dates = values
    .map((value) => normalizeText(value))
    .filter((value) => /^\d{4}-\d{2}-\d{2}$/.test(value));
  if (dates.length === 0) {
    return '';
  }
  return dates.sort().at(-1);
}

function normalizeKey(value) {
  return normalizeText(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasText(value) {
  return normalizeText(value).length > 0;
}

function hasFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function timestampScore(value) {
  const parsed = Date.parse(normalizeText(value));
  return Number.isNaN(parsed) ? 0 : parsed;
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

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
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

function validateSupportRolloutPolicy(policy) {
  if (!isObject(policy)) {
    throw new Error('support-rollout-policy.json must be an object');
  }
  if (policy.schemaVersion !== 1) {
    throw new Error('support-rollout-policy.json schemaVersion must be 1');
  }
  const rollout = policy;
  if (rollout.ordering !== 'artifact-size-ascending') {
    throw new Error('supportRollout.ordering must be "artifact-size-ascending"');
  }
  if (!Array.isArray(rollout.sizeTiers) || rollout.sizeTiers.length === 0) {
    throw new Error('supportRollout.sizeTiers must be a non-empty array');
  }
  if (!Array.isArray(rollout.gateOrder) || rollout.gateOrder.length === 0) {
    throw new Error('supportRollout.gateOrder must be a non-empty array');
  }
  const tierIds = new Set();
  let previousMax = null;
  let unknownTierSeen = false;
  for (const tier of rollout.sizeTiers) {
    if (!isObject(tier)) {
      throw new Error('supportRollout.sizeTiers entries must be objects');
    }
    const id = normalizeText(tier.id);
    if (!id) {
      throw new Error('supportRollout.sizeTiers entries require id');
    }
    if (tierIds.has(id)) {
      throw new Error(`duplicate support rollout tier id: ${id}`);
    }
    tierIds.add(id);
    if (!hasText(tier.label)) {
      throw new Error(`supportRollout.sizeTiers.${id}.label is required`);
    }
    const min = tier.minSizeBytesInclusive;
    const max = tier.maxSizeBytesExclusive;
    if (id === 'unknown') {
      if (min !== null || max !== null) {
        throw new Error('supportRollout.sizeTiers.unknown bounds must be null');
      }
      unknownTierSeen = true;
      continue;
    }
    if (!Number.isInteger(min) || min < 0) {
      throw new Error(`supportRollout.sizeTiers.${id}.minSizeBytesInclusive must be a non-negative integer`);
    }
    if (max !== null && (!Number.isInteger(max) || max <= min)) {
      throw new Error(`supportRollout.sizeTiers.${id}.maxSizeBytesExclusive must be null or greater than min`);
    }
    if (previousMax !== null && min !== previousMax) {
      throw new Error(`supportRollout.sizeTiers.${id} must start at previous tier max ${previousMax}`);
    }
    previousMax = max;
  }
  if (!unknownTierSeen) {
    throw new Error('supportRollout.sizeTiers must include an unknown tier');
  }
  const gateIds = new Set();
  for (const gate of rollout.gateOrder) {
    if (!isObject(gate)) {
      throw new Error('supportRollout.gateOrder entries must be objects');
    }
    const id = normalizeText(gate.id);
    if (!id) {
      throw new Error('supportRollout.gateOrder entries require id');
    }
    if (gateIds.has(id)) {
      throw new Error(`duplicate support rollout gate id: ${id}`);
    }
    if (!hasText(gate.label)) {
      throw new Error(`supportRollout.gateOrder.${id}.label is required`);
    }
    gateIds.add(id);
  }
  const requiredGates = [
    'conversion-config',
    'manifest-weights',
    'runtime-verify',
    'hf-publish',
    'compare-profile',
    'benchmark-receipts',
    'preferred-architecture',
  ];
  for (const gateId of requiredGates) {
    if (!gateIds.has(gateId)) {
      throw new Error(`supportRollout.gateOrder must include ${gateId}`);
    }
  }
  const benchmarkCommands = rollout.benchmarkCommands;
  if (!isObject(benchmarkCommands)) {
    throw new Error('supportRollout.benchmarkCommands must be an object');
  }
  if (!['compute', 'warm', 'cold', 'all'].includes(benchmarkCommands.mode)) {
    throw new Error('supportRollout.benchmarkCommands.mode must be compute, warm, cold, or all');
  }
  if (!Number.isInteger(benchmarkCommands.warmupRuns) || benchmarkCommands.warmupRuns < 0) {
    throw new Error('supportRollout.benchmarkCommands.warmupRuns must be a non-negative integer');
  }
  if (!Number.isInteger(benchmarkCommands.timedRuns) || benchmarkCommands.timedRuns <= 0) {
    throw new Error('supportRollout.benchmarkCommands.timedRuns must be a positive integer');
  }
  if (!Array.isArray(benchmarkCommands.workloads) || benchmarkCommands.workloads.length === 0) {
    throw new Error('supportRollout.benchmarkCommands.workloads must be a non-empty array');
  }
  if (!Array.isArray(benchmarkCommands.decodeProfiles) || benchmarkCommands.decodeProfiles.length === 0) {
    throw new Error('supportRollout.benchmarkCommands.decodeProfiles must be a non-empty array');
  }
  if (typeof benchmarkCommands.save !== 'boolean') {
    throw new Error('supportRollout.benchmarkCommands.save must be boolean');
  }
  if (typeof benchmarkCommands.json !== 'boolean') {
    throw new Error('supportRollout.benchmarkCommands.json must be boolean');
  }
  const preferred = rollout.preferredArchitecture;
  if (!isObject(preferred)) {
    throw new Error('supportRollout.preferredArchitecture must be an object');
  }
  if (preferred.selection !== 'benchmark-evidence-only') {
    throw new Error('supportRollout.preferredArchitecture.selection must be "benchmark-evidence-only"');
  }
  const evidence = Array.isArray(preferred.requiredEvidence) ? preferred.requiredEvidence : [];
  for (const id of ['runtimeReport', 'compareResult', 'summarySvg']) {
    if (!evidence.includes(id)) {
      throw new Error(`supportRollout.preferredArchitecture.requiredEvidence must include ${id}`);
    }
  }
  return rollout;
}

function validateBenchmarkCommandReferences(supportRollout, workloads, benchmarkPolicy) {
  const workloadIds = new Set(
    (Array.isArray(workloads?.workloads) ? workloads.workloads : [])
      .map((entry) => normalizeText(entry?.id))
      .filter(Boolean)
  );
  const decodeProfileIds = new Set(Object.keys(benchmarkPolicy?.decodeProfiles?.profiles || {}));
  for (const workloadId of supportRollout.benchmarkCommands.workloads) {
    if (!workloadIds.has(workloadId)) {
      throw new Error(`supportRollout.benchmarkCommands.workloads references unknown workload ${workloadId}`);
    }
  }
  for (const decodeProfileId of supportRollout.benchmarkCommands.decodeProfiles) {
    if (!decodeProfileIds.has(decodeProfileId)) {
      throw new Error(`supportRollout.benchmarkCommands.decodeProfiles references unknown decode profile ${decodeProfileId}`);
    }
  }
}

async function collectJsonFiles(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await collectJsonFiles(fullPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(fullPath);
    }
  }
  return files.sort();
}

function conversionKeys(record) {
  const payload = record.payload;
  const identity = payload?.manifest?.artifactIdentity;
  return [
    record.modelBaseId,
    path.basename(record.path, '.json'),
    identity?.sourceCheckpointId,
    identity?.weightPackId,
    identity?.manifestVariantId,
  ].map(normalizeKey).filter(Boolean);
}

async function collectConversionConfigs() {
  const files = await collectJsonFiles(CONVERSION_CONFIG_DIR);
  const records = [];
  for (const filePath of files) {
    const payload = await readJson(filePath);
    const modelBaseId = normalizeText(payload?.output?.modelBaseId || payload?.modelId || path.basename(filePath, '.json'));
    records.push({
      path: repoRelative(filePath),
      modelBaseId,
      family: path.relative(CONVERSION_CONFIG_DIR, path.dirname(filePath)).split(path.sep)[0] || 'root',
      payload,
    });
  }
  return records;
}

async function collectLatestCompareResults(prefix) {
  const byModelId = new Map();
  let entries = [];
  try {
    entries = await fs.readdir(COMPARE_RESULTS_DIR, { withFileTypes: true });
  } catch {
    return byModelId;
  }
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    if (!entry.name.startsWith(prefix) || !entry.name.endsWith('.json')) continue;
    const fullPath = path.join(COMPARE_RESULTS_DIR, entry.name);
    const payload = await readJson(fullPath);
    if (payload?.summary?.correctnessOk !== true) continue;
    const modelId = normalizeText(payload?.model?.dopplerModelId);
    if (!modelId) continue;
    const record = {
      path: repoRelative(fullPath),
      timestamp: normalizeText(payload?.timestamp) || null,
      isLatestAlias: entry.name === `${prefix}latest.json`,
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

function buildConversionIndex(conversionRecords) {
  const index = new Map();
  for (const record of conversionRecords) {
    for (const key of conversionKeys(record)) {
      if (!index.has(key)) index.set(key, []);
      index.get(key).push(record);
    }
  }
  return index;
}

function catalogLookupKeys(model) {
  return [
    model?.modelId,
    model?.sourceCheckpointId,
    model?.weightPackId,
    model?.manifestVariantId,
    ...(Array.isArray(model?.aliases) ? model.aliases : []),
  ].map(normalizeKey).filter(Boolean);
}

function findConversionForCatalogModel(model, conversionIndex) {
  for (const key of catalogLookupKeys(model)) {
    const matches = conversionIndex.get(key);
    if (matches && matches.length > 0) {
      return matches[0];
    }
  }
  return null;
}

function mapByModelId(entries, getModelId) {
  const out = new Map();
  for (const entry of Array.isArray(entries) ? entries : []) {
    const modelId = normalizeText(getModelId(entry));
    if (modelId && !out.has(modelId)) {
      out.set(modelId, entry);
    }
  }
  return out;
}

function buildCompareProfileIndex(catalog, compareConfig, embeddingCompareConfig, rerankCompareConfig) {
  const index = new Map();
  const catalogByModelId = mapByModelId(catalog?.models, (entry) => entry?.modelId);
  for (const entry of Array.isArray(compareConfig?.modelProfiles) ? compareConfig.modelProfiles : []) {
    const modelId = normalizeText(entry?.dopplerModelId);
    if (!modelId) continue;
    index.set(modelId, {
      ...entry,
      profileKind: 'generation',
    });
  }
  for (const entry of Array.isArray(embeddingCompareConfig?.modelProfiles) ? embeddingCompareConfig.modelProfiles : []) {
    const modelId = normalizeText(entry?.dopplerModelId);
    if (!modelId) continue;
    const catalogEntry = catalogByModelId.get(modelId);
    const modes = Array.isArray(catalogEntry?.modes) ? catalogEntry.modes.map(normalizeText) : [];
    if (!modes.includes('embedding')) continue;
    index.set(modelId, {
      ...entry,
      defaultTjsModelId: normalizeText(entry?.defaultTjsModelId)
        || normalizeText(catalogEntry?.vendorBenchmark?.transformersjs?.repoId)
        || null,
      defaultTjsDtype: normalizeText(entry?.defaultTjsDtype)
        || normalizeText(catalogEntry?.vendorBenchmark?.transformersjs?.dtype)
        || null,
      profileKind: 'embedding',
    });
  }
  for (const entry of Array.isArray(rerankCompareConfig?.modelProfiles) ? rerankCompareConfig.modelProfiles : []) {
    const modelId = normalizeText(entry?.dopplerModelId);
    if (!modelId) continue;
    const catalogEntry = catalogByModelId.get(modelId);
    const modes = Array.isArray(catalogEntry?.modes) ? catalogEntry.modes.map(normalizeText) : [];
    if (!modes.includes('rerank')) continue;
    index.set(modelId, {
      ...entry,
      defaultTjsModelId: normalizeText(entry?.defaultTjsModelId)
        || normalizeText(catalogEntry?.vendorBenchmark?.transformersjs?.repoId)
        || null,
      defaultTjsDtype: normalizeText(entry?.defaultTjsDtype)
        || normalizeText(catalogEntry?.vendorBenchmark?.transformersjs?.dtype)
        || null,
      profileKind: 'rerank',
    });
  }
  return index;
}

function resolveTier(sizeBytes, sizeTiers) {
  if (!hasFiniteNumber(sizeBytes)) {
    return 'unknown';
  }
  for (const tier of sizeTiers) {
    if (tier.id === 'unknown') continue;
    const min = tier.minSizeBytesInclusive;
    const max = tier.maxSizeBytesExclusive;
    if (sizeBytes >= min && (max === null || sizeBytes < max)) {
      return tier.id;
    }
  }
  return 'unknown';
}

function describeBytes(value) {
  if (!hasFiniteNumber(value)) return 'unknown';
  const gib = value / 1073741824;
  return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
}

function describeArchitecture(model, conversionRecord) {
  const quantization = conversionRecord?.payload?.quantization;
  const artifactFormat = normalizeText(model?.artifact?.format) || null;
  const weights = normalizeText(quantization?.weights) || null;
  const embeddings = normalizeText(quantization?.embeddings) || null;
  const lmHead = normalizeText(quantization?.lmHead) || null;
  const computePrecision = normalizeText(quantization?.computePrecision) || null;
  const q4kLayout = normalizeText(quantization?.q4kLayout) || null;
  const parts = [];
  if (artifactFormat) parts.push(artifactFormat.toUpperCase());
  if (weights) parts.push(`${weights} weights`);
  if (embeddings) parts.push(`${embeddings} embeddings`);
  if (lmHead) parts.push(`${lmHead} LM head`);
  if (computePrecision) parts.push(`${computePrecision} compute`);
  if (q4kLayout) parts.push(`${q4kLayout} Q4K layout`);
  return {
    artifactFormat,
    weights,
    embeddings,
    lmHead,
    computePrecision,
    q4kLayout,
    label: parts.length > 0 ? parts.join(', ') : 'architecture metadata unavailable',
  };
}

function buildCompareCommands(modelId, benchmarkCommands) {
  const commands = [];
  for (const workloadId of benchmarkCommands.workloads) {
    for (const decodeProfile of benchmarkCommands.decodeProfiles) {
      const args = [
        'node',
        'tools/compare-engines.js',
        '--model-id',
        modelId,
        '--workload',
        workloadId,
        '--mode',
        benchmarkCommands.mode,
        '--decode-profile',
        decodeProfile,
        '--warmup',
        String(benchmarkCommands.warmupRuns),
        '--runs',
        String(benchmarkCommands.timedRuns),
      ];
      if (benchmarkCommands.save === true) args.push('--save');
      if (benchmarkCommands.json === true) args.push('--json');
      commands.push({
        workloadId,
        decodeProfile,
        command: args.join(' '),
      });
    }
  }
  return commands;
}

function buildEmbeddingCompareCommands(modelId, compareProfile, embeddingDefaults) {
  const args = [
    'node',
    'tools/compare-embeddings.js',
    '--model-id',
    modelId,
    '--warmup',
    String(compareProfile.warmupRuns ?? embeddingDefaults.warmupRuns ?? 1),
    '--runs',
    String(compareProfile.timedRuns ?? embeddingDefaults.timedRuns ?? 3),
  ];
  const dopplerSource = normalizeText(compareProfile.defaultDopplerSource);
  if (dopplerSource) {
    args.push('--doppler-source', dopplerSource);
  }
  const dopplerSurface = normalizeText(compareProfile.defaultDopplerSurface || embeddingDefaults.dopplerSurface);
  if (dopplerSurface) {
    args.push('--doppler-surface', dopplerSurface);
  }
  const cacheMode = normalizeText(compareProfile.cacheMode || embeddingDefaults.cacheMode);
  if (cacheMode) {
    args.push('--cache-mode', cacheMode);
  }
  const loadMode = normalizeText(compareProfile.loadMode || embeddingDefaults.loadMode);
  if (loadMode) {
    args.push('--load-mode', loadMode);
  }
  args.push('--save', '--json');
  return [{
    workloadId: 'embedding',
    decodeProfile: null,
    command: args.join(' '),
  }];
}

function buildRerankCompareCommands(modelId, compareProfile, rerankDefaults) {
  const args = [
    'node',
    'tools/compare-rerankers.js',
    '--model-id',
    modelId,
    '--warmup',
    String(compareProfile.warmupRuns ?? rerankDefaults.warmupRuns ?? 1),
    '--runs',
    String(compareProfile.timedRuns ?? rerankDefaults.timedRuns ?? 3),
  ];
  const dopplerSource = normalizeText(compareProfile.defaultDopplerSource);
  if (dopplerSource) {
    args.push('--doppler-source', dopplerSource);
  }
  const dopplerSurface = normalizeText(compareProfile.defaultDopplerSurface || rerankDefaults.dopplerSurface);
  if (dopplerSurface) {
    args.push('--doppler-surface', dopplerSurface);
  }
  const cacheMode = normalizeText(compareProfile.cacheMode || rerankDefaults.cacheMode);
  if (cacheMode) {
    args.push('--cache-mode', cacheMode);
  }
  const loadMode = normalizeText(compareProfile.loadMode || rerankDefaults.loadMode);
  if (loadMode) {
    args.push('--load-mode', loadMode);
  }
  args.push('--save', '--json');
  return [{
    workloadId: 'rerank',
    decodeProfile: null,
    command: args.join(' '),
  }];
}

function buildCompareCommandsForProfile(modelId, compareProfile, supportRollout, embeddingDefaults, rerankDefaults) {
  if (compareProfile?.profileKind === 'embedding') {
    return buildEmbeddingCompareCommands(modelId, compareProfile, embeddingDefaults);
  }
  if (compareProfile?.profileKind === 'rerank') {
    return buildRerankCompareCommands(modelId, compareProfile, rerankDefaults);
  }
  return buildCompareCommands(modelId, supportRollout.benchmarkCommands);
}

function buildActions(modelId, compareProfile, benchmarkComparable, supportRollout, options = {}) {
  const bootstrapFlag = options.hfPublished === false ? ' --bootstrap' : '';
  return {
    verifyCommand: `node tools/run-registry-verify.js ${modelId} --surface auto`,
    hfDryRunCommand: `node tools/publish-hf-registry-model.js --model-id ${modelId} --dry-run${bootstrapFlag}`,
    compareCommands: compareProfile && benchmarkComparable
      ? buildCompareCommandsForProfile(
        modelId,
        compareProfile,
        supportRollout,
        options.embeddingDefaults || {},
        options.rerankDefaults || {}
      )
      : [],
    primaryNextCommand: null,
  };
}

function resolvePrimaryNextCommand(nextGate, actions, compareProfile = null) {
  if (nextGate === 'runtime-verify') return actions.verifyCommand;
  if (nextGate === 'hf-publish') return actions.hfDryRunCommand;
  if (
    nextGate === 'summary-svg'
    && ['embedding', 'rerank'].includes(normalizeText(compareProfile?.profileKind))
  ) {
    return null;
  }
  if (
    nextGate === 'claim-lane'
    || nextGate === 'compare-result'
    || nextGate === 'summary-svg'
  ) {
    return actions.compareCommands[0]?.command || null;
  }
  return null;
}

function resolveReleaseClaimPath(modelId, releaseClaimByModelId) {
  const claim = releaseClaimByModelId.get(modelId);
  return normalizeText(claim?.evidence?.reportPath || claim?.performanceEvidence?.reportPath) || null;
}

async function buildVariant(model, context) {
  const conversion = findConversionForCatalogModel(model, context.conversionIndex);
  const modelId = normalizeText(model.modelId);
  const compareProfile = context.compareProfileByModelId.get(modelId) || null;
  const releaseCoverage = context.releaseCoverageByModelId.get(modelId) || null;
  const claimLane = context.claimLaneByModelId.get(modelId) || null;
  const runtimeReport = normalizeText(claimLane?.evidence?.localExecutionReport) || resolveReleaseClaimPath(modelId, context.releaseClaimByModelId);
  const profileKind = normalizeText(compareProfile?.profileKind) || null;
  const specialtyCompareResult = profileKind === 'embedding'
    ? context.latestEmbeddingCompareResultByModelId.get(modelId)
    : (profileKind === 'rerank' ? context.latestRerankCompareResultByModelId.get(modelId) : null);
  const compareResult = normalizeText(claimLane?.evidence?.compareResult) || normalizeText(specialtyCompareResult?.path);
  const summarySvg = normalizeText(claimLane?.evidence?.summarySvg);
  const hfPublished = model?.lifecycle?.availability?.hf === true;
  const tested = normalizeText(model?.lifecycle?.status?.tested);
  const artifactCompleteness = normalizeText(model?.artifactCompleteness);
  const manifestWeightsOk = artifactCompleteness === 'complete' || artifactCompleteness === 'weights-ref';
  const compareLane = normalizeText(compareProfile?.compareLane || releaseCoverage?.compareLane);
  const benchmarkComparable = compareLane === 'performance_comparable';
  const benchmarkEvidenceOk = benchmarkComparable && hasText(compareResult) && hasText(summarySvg);
  const missing = [];

  if (!conversion) missing.push('conversion-config');
  if (!manifestWeightsOk || !hasText(model?.weightPackId) || !hasText(model?.manifestVariantId)) {
    missing.push('manifest-weights');
  }
  if (tested !== 'verified' || !hasText(runtimeReport)) {
    missing.push('runtime-verify');
  }
  if (!hfPublished || !hasText(model?.hf?.repoId) || !hasText(model?.hf?.revision) || !hasText(model?.hf?.path)) {
    missing.push('hf-publish');
  }
  if (!compareProfile) {
    missing.push('compare-profile');
  }
  if (benchmarkComparable) {
    if (!claimLane && profileKind !== 'embedding' && profileKind !== 'rerank') missing.push('claim-lane');
    if (!hasText(compareResult)) missing.push('compare-result');
    if (!hasText(summarySvg)) missing.push('summary-svg');
  } else if (compareProfile) {
    missing.push('benchmark-lane-capability-only');
  }

  const evidence = {
    runtimeReport: runtimeReport || null,
    runtimeReportExists: runtimeReport ? await pathExists(runtimeReport) : false,
    compareResult: compareResult || null,
    compareResultExists: compareResult ? await pathExists(compareResult) : false,
    summarySvg: summarySvg || null,
    summarySvgExists: summarySvg ? await pathExists(summarySvg) : false,
    localClaimLaneId: normalizeText(claimLane?.id) || null,
  };
  if (evidence.runtimeReport && !evidence.runtimeReportExists) missing.push('runtime-report-missing-on-disk');
  if (evidence.compareResult && !evidence.compareResultExists) missing.push('compare-result-missing-on-disk');
  if (evidence.summarySvg && !evidence.summarySvgExists) missing.push('summary-svg-missing-on-disk');

  const nextGate = missing[0] || (benchmarkEvidenceOk ? 'preferred-architecture' : 'benchmark-receipts');
  const actions = buildActions(modelId, compareProfile, benchmarkComparable, context.supportRollout, {
    hfPublished,
    embeddingDefaults: context.embeddingCompareDefaults,
    rerankDefaults: context.rerankCompareDefaults,
  });
  actions.primaryNextCommand = resolvePrimaryNextCommand(nextGate, actions, compareProfile);
  return {
    modelId,
    label: normalizeText(model.label) || modelId,
    family: normalizeText(model.family) || 'unknown',
    modes: Array.isArray(model.modes) ? model.modes.map(normalizeText).filter(Boolean) : [],
    sizeBytes: hasFiniteNumber(model.sizeBytes) ? model.sizeBytes : null,
    sizeLabel: describeBytes(model.sizeBytes),
    tier: resolveTier(model.sizeBytes, context.supportRollout.sizeTiers),
    sourceCheckpointId: normalizeText(model.sourceCheckpointId) || null,
    weightPackId: normalizeText(model.weightPackId) || null,
    manifestVariantId: normalizeText(model.manifestVariantId) || null,
    artifactCompleteness: artifactCompleteness || null,
    runtimePromotionState: normalizeText(model.runtimePromotionState) || null,
    conversionConfig: conversion?.path || null,
    architecture: describeArchitecture(model, conversion),
    lifecycle: {
      runtime: normalizeText(model?.lifecycle?.status?.runtime) || null,
      conversion: normalizeText(model?.lifecycle?.status?.conversion) || null,
      tested: tested || null,
      result: normalizeText(model?.lifecycle?.tested?.result) || null,
      lastVerifiedAt: normalizeText(model?.lifecycle?.tested?.lastVerifiedAt) || null,
      surfaces: Array.isArray(model?.lifecycle?.tested?.surface)
        ? model.lifecycle.tested.surface.map(normalizeText).filter(Boolean)
        : [],
    },
    hf: {
      published: hfPublished,
      repoId: normalizeText(model?.hf?.repoId) || null,
      revision: normalizeText(model?.hf?.revision) || null,
      path: normalizeText(model?.hf?.path) || null,
    },
    quickstart: model.quickstart === true,
    compare: {
      profile: compareProfile ? {
        kind: profileKind,
        dopplerSource: normalizeText(compareProfile.defaultDopplerSource) || null,
        dopplerFormat: normalizeText(compareProfile.defaultDopplerFormat) || null,
        tjsModelId: normalizeText(compareProfile.defaultTjsModelId) || null,
        tjsDtype: normalizeText(compareProfile.defaultTjsDtype) || null,
        tjsFormat: normalizeText(compareProfile.defaultTjsFormat) || null,
        lane: compareLane || null,
        laneReason: normalizeText(compareProfile.compareLaneReason) || null,
      } : null,
      benchmarkComparable,
      benchmarkEvidenceOk,
    },
    evidence,
    actions,
    missing,
    nextGate,
  };
}

function choosePreferredArchitecture(variants, supportRollout) {
  const benchmarked = variants
    .filter((variant) => variant.compare.benchmarkEvidenceOk)
    .sort(compareVariants);
  if (benchmarked.length > 0) {
    const selected = benchmarked[0];
    return {
      status: 'benchmark-selected',
      modelId: selected.modelId,
      architecture: selected.architecture.label,
      evidence: {
        runtimeReport: selected.evidence.runtimeReport,
        compareResult: selected.evidence.compareResult,
        summarySvg: selected.evidence.summarySvg,
      },
    };
  }
  const candidates = variants
    .filter((variant) => variant.lifecycle.tested === 'verified')
    .sort(compareVariants);
  if (candidates.length > 0) {
    const candidate = candidates[0];
    return {
      status: supportRollout.preferredArchitecture.unselectedStatus,
      modelId: candidate.modelId,
      architecture: candidate.architecture.label,
      evidence: {
        runtimeReport: candidate.evidence.runtimeReport,
        compareResult: null,
        summarySvg: null,
      },
    };
  }
  if (variants.some((variant) => variant.lifecycle.tested === 'failed')) {
    return {
      status: 'verification-failed',
      modelId: null,
      architecture: null,
      evidence: {
        runtimeReport: null,
        compareResult: null,
        summarySvg: null,
      },
    };
  }
  return {
    status: supportRollout.preferredArchitecture.unselectedStatus,
    modelId: null,
    architecture: null,
    evidence: {
      runtimeReport: null,
      compareResult: null,
      summarySvg: null,
    },
  };
}

function compareVariants(left, right) {
  const leftSize = hasFiniteNumber(left.sizeBytes) ? left.sizeBytes : Number.MAX_SAFE_INTEGER;
  const rightSize = hasFiniteNumber(right.sizeBytes) ? right.sizeBytes : Number.MAX_SAFE_INTEGER;
  if (leftSize !== rightSize) return leftSize - rightSize;
  return left.modelId.localeCompare(right.modelId);
}

function compareSourceModels(left, right) {
  const leftSize = hasFiniteNumber(left.minSizeBytes) ? left.minSizeBytes : Number.MAX_SAFE_INTEGER;
  const rightSize = hasFiniteNumber(right.minSizeBytes) ? right.minSizeBytes : Number.MAX_SAFE_INTEGER;
  if (leftSize !== rightSize) return leftSize - rightSize;
  return left.sourceCheckpointId.localeCompare(right.sourceCheckpointId);
}

function summarizeSourceStatus(variants, preferred) {
  if (preferred.status === 'benchmark-selected') return 'benchmark-selected';
  if (variants.some((variant) => variant.compare.benchmarkComparable && variant.evidence.compareResult)) {
    return 'benchmark-receipts-incomplete';
  }
  if (variants.some((variant) => variant.lifecycle.tested === 'verified')) {
    return 'benchmark-comparison-needed';
  }
  if (variants.some((variant) => variant.lifecycle.tested === 'failed')) {
    return 'verification-failed';
  }
  if (variants.some((variant) => variant.conversionConfig)) {
    return 'runtime-verification-needed';
  }
  return 'conversion-needed';
}

function sourceNextAction(sourceModel) {
  if (sourceModel.preferredArchitecture.status === 'benchmark-selected') {
    return `${sourceModel.preferredArchitecture.modelId} is selected from committed runtime, compare JSON, and SVG receipts.`;
  }
  const variants = sourceModel.variants;
  const smallest = [...variants].sort(compareVariants)[0];
  if (!smallest) return 'Add a catalog or conversion entry.';
  if (smallest.missing.includes('conversion-config')) {
    return `Add or repair the conversion config for ${smallest.modelId}.`;
  }
  if (smallest.missing.includes('manifest-weights')) {
    return `Refresh manifest and weight identity for ${smallest.modelId}.`;
  }
  if (smallest.missing.includes('runtime-verify')) {
    if (smallest.lifecycle.tested === 'failed' || smallest.lifecycle.result === 'fail') {
      return `Fix failed runtime verification for ${smallest.modelId}; keep it unpromoted until a passing receipt exists.`;
    }
    return `Run deterministic runtime verification for ${smallest.modelId}.`;
  }
  if (smallest.missing.includes('hf-publish')) {
    return `Publish or refresh the Hugging Face manifest/weights for ${smallest.modelId}.`;
  }
  if (smallest.missing.includes('compare-profile')) {
    return `Add a compare profile for ${smallest.modelId}.`;
  }
  if (smallest.missing.includes('benchmark-lane-capability-only')) {
    return `Decide whether ${smallest.modelId} needs a claimable generation compare lane or stays capability-only.`;
  }
  if (smallest.missing.includes('claim-lane')) {
    return `Add a local inference claim lane for ${smallest.modelId}.`;
  }
  if (smallest.missing.includes('summary-svg') && !smallest.missing.includes('compare-result')) {
    if (['embedding', 'rerank'].includes(normalizeText(smallest.compare?.profile?.kind))) {
      return `Add summary SVG evidence for ${smallest.modelId}.`;
    }
  }
  if (smallest.missing.includes('compare-result') || smallest.missing.includes('summary-svg')) {
    return `Run compare/bench receipts for ${smallest.modelId}.`;
  }
  return `Review promotion gates for ${smallest.modelId}.`;
}

function sourceNextCommand(sourceModel) {
  const smallest = [...sourceModel.variants].sort(compareVariants)[0];
  return smallest?.actions?.primaryNextCommand || null;
}

function sourceNextGate(sourceModel) {
  const smallest = [...sourceModel.variants].sort(compareVariants)[0];
  return smallest?.nextGate || null;
}

async function buildInventory() {
  const [
    catalog,
    supportRolloutPolicy,
    compareConfig,
    embeddingCompareConfig,
    rerankCompareConfig,
    claimMatrix,
    releaseMatrix,
    releaseClaimPolicy,
    workloads,
    benchmarkPolicy,
    conversionRecords,
    latestEmbeddingCompareResultByModelId,
    latestRerankCompareResultByModelId,
  ] = await Promise.all([
    readJson(CATALOG_PATH),
    readJson(SUPPORT_ROLLOUT_POLICY_PATH),
    readJson(COMPARE_CONFIG_PATH),
    readJson(EMBEDDING_COMPARE_CONFIG_PATH),
    readJson(RERANK_COMPARE_CONFIG_PATH),
    readJson(CLAIM_MATRIX_PATH),
    readJson(RELEASE_MATRIX_PATH),
    readJson(RELEASE_CLAIM_POLICY_PATH),
    readJson(WORKLOADS_PATH),
    readJson(BENCHMARK_POLICY_PATH),
    collectConversionConfigs(),
    collectLatestCompareResults('embedding_compare_'),
    collectLatestCompareResults('rerank_compare_'),
  ]);
  const supportRollout = validateSupportRolloutPolicy(supportRolloutPolicy);
  validateBenchmarkCommandReferences(supportRollout, workloads, benchmarkPolicy);
  const conversionIndex = buildConversionIndex(conversionRecords);
  const compareProfileByModelId = buildCompareProfileIndex(catalog, compareConfig, embeddingCompareConfig, rerankCompareConfig);
  const claimLaneByModelId = mapByModelId(claimMatrix.lanes, (entry) => entry?.model?.dopplerModelId);
  const releaseCoverageByModelId = mapByModelId(releaseMatrix.modelCoverage, (entry) => entry?.dopplerModelId);
  const releaseClaimByModelId = mapByModelId(releaseClaimPolicy.claims, (entry) => entry?.modelId);
  const context = {
    supportRollout,
    conversionIndex,
    compareProfileByModelId,
    embeddingCompareDefaults: embeddingCompareConfig.defaults || {},
    rerankCompareDefaults: rerankCompareConfig.defaults || {},
    claimLaneByModelId,
    releaseCoverageByModelId,
    releaseClaimByModelId,
    latestEmbeddingCompareResultByModelId,
    latestRerankCompareResultByModelId,
  };
  const variants = [];
  const assignedConversionPaths = new Set();
  for (const model of catalog.models || []) {
    const variant = await buildVariant(model, context);
    variants.push(variant);
    if (variant.conversionConfig) {
      assignedConversionPaths.add(variant.conversionConfig);
    }
  }
  const sourceGroups = new Map();
  for (const variant of variants) {
    const sourceId = variant.sourceCheckpointId || variant.modelId;
    if (!sourceGroups.has(sourceId)) sourceGroups.set(sourceId, []);
    sourceGroups.get(sourceId).push(variant);
  }
  const sourceModels = [];
  for (const [sourceCheckpointId, sourceVariants] of sourceGroups) {
    const sortedVariants = sourceVariants.sort(compareVariants);
    const minSizeBytes = sortedVariants.find((variant) => hasFiniteNumber(variant.sizeBytes))?.sizeBytes ?? null;
    const preferredArchitecture = choosePreferredArchitecture(sortedVariants, supportRollout);
    const sourceModel = {
      sourceCheckpointId,
      family: sortedVariants[0]?.family || 'unknown',
      tier: resolveTier(minSizeBytes, supportRollout.sizeTiers),
      minSizeBytes,
      minSizeLabel: describeBytes(minSizeBytes),
      status: summarizeSourceStatus(sortedVariants, preferredArchitecture),
      preferredArchitecture,
      variants: sortedVariants,
    };
    sourceModel.nextAction = sourceNextAction(sourceModel);
    sourceModel.nextGate = sourceNextGate(sourceModel);
    sourceModel.nextCommand = sourceNextCommand(sourceModel);
    sourceModels.push(sourceModel);
  }
  const conversionOnly = conversionRecords
    .filter((record) => !assignedConversionPaths.has(record.path))
    .map((record) => ({
      modelBaseId: record.modelBaseId,
      family: record.family,
      conversionConfig: record.path,
      sourceCheckpointId: normalizeText(record.payload?.manifest?.artifactIdentity?.sourceCheckpointId) || null,
      architecture: describeArchitecture({ artifact: { format: 'rdrr' } }, record),
      status: 'conversion-ready',
      nextAction: `Catalog, verify, and publish ${record.modelBaseId} before claiming runtime support.`,
      nextCommand: null,
    }))
    .sort((left, right) => left.modelBaseId.localeCompare(right.modelBaseId));
  sourceModels.sort(compareSourceModels);
  const summary = {
    catalogModelCount: variants.length,
    sourceCheckpointCount: sourceModels.length,
    conversionOnlyConfigCount: conversionOnly.length,
    hfPublishedModelCount: variants.filter((variant) => variant.hf.published).length,
    verifiedModelCount: variants.filter((variant) => variant.lifecycle.tested === 'verified').length,
    benchmarkSelectedSourceCount: sourceModels.filter((entry) => entry.preferredArchitecture.status === 'benchmark-selected').length,
    benchmarkPendingSourceCount: sourceModels.filter((entry) => entry.preferredArchitecture.status !== 'benchmark-selected').length,
    tierCounts: Object.fromEntries(supportRollout.sizeTiers.map((tier) => [
      tier.id,
      sourceModels.filter((entry) => entry.tier === tier.id).length,
    ])),
  };
  return {
    schemaVersion: 1,
    updated: latestSourceDate(catalog.updatedAt, supportRollout.updated),
    sources: [
      'models/catalog.json',
      'src/config/conversion/**',
      'benchmarks/vendors/support-rollout-policy.json',
      'benchmarks/vendors/benchmark-policy.json',
      'benchmarks/vendors/workloads.json',
      'benchmarks/vendors/compare-engines.config.json',
      'benchmarks/vendors/embedding-compare.config.json',
      'benchmarks/vendors/rerank-compare.config.json',
      'benchmarks/vendors/results/embedding_compare_*.json',
      'benchmarks/vendors/results/rerank_compare_*.json',
      'benchmarks/vendors/local-inference-claim-matrix.json',
      'benchmarks/vendors/release-matrix.json',
      'tools/policies/release-claim-policy.json',
    ],
    policy: {
      source: 'benchmarks/vendors/support-rollout-policy.json',
      supportRollout,
    },
    summary,
    sourceModels,
    conversionOnly,
  };
}

function renderGateList(gates) {
  return gates.map((gate, index) => `${index + 1}. ${gate.id}: ${gate.label}`).join('\n');
}

function renderPreferred(preferred) {
  if (preferred.status === 'benchmark-selected') {
    return `${preferred.modelId} (${preferred.architecture})`;
  }
  if (preferred.modelId) {
    return `${preferred.modelId} candidate; benchmark comparison pending`;
  }
  if (preferred.status === 'verification-failed') {
    return 'failed verification';
  }
  return 'pending verification';
}

function renderEvidence(preferred) {
  const entries = [
    preferred.evidence?.runtimeReport,
    preferred.evidence?.compareResult,
    preferred.evidence?.summarySvg,
  ].filter(Boolean);
  return entries.length > 0 ? entries.join('<br>') : 'none';
}

function renderTableRow(values) {
  return values
    .map((value) => String(value ?? '').replace(/\|/g, '\\|'))
    .join(' | ')
    .replace(/^/, '| ')
    .replace(/$/, ' |');
}

function renderCommand(command) {
  if (!command) return 'none';
  return `\`${String(command).replace(/`/g, '\\`')}\``;
}

function renderMarkdown(inventory) {
  const lines = [];
  lines.push('# Model Support Inventory');
  lines.push('');
  lines.push('Generated from model catalog, conversion configs, support rollout policy, compare profiles, saved compare receipts, release matrix, claim lanes, and release-claim receipts.');
  lines.push('');
  lines.push(`Updated at: ${inventory.updated}`);
  lines.push('');
  lines.push('Policy: smallest artifact size first. Size tiers use catalog artifact bytes, not parameter class. A preferred architecture is selected only when runtime verification, compare JSON, and summary SVG evidence exist for that lane.');
  lines.push('');
  lines.push('## Gate Order');
  lines.push('');
  lines.push(renderGateList(inventory.policy.supportRollout.gateOrder));
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Catalog models: ${inventory.summary.catalogModelCount}`);
  lines.push(`- Source checkpoints: ${inventory.summary.sourceCheckpointCount}`);
  lines.push(`- Conversion-only configs: ${inventory.summary.conversionOnlyConfigCount}`);
  lines.push(`- HF-published catalog models: ${inventory.summary.hfPublishedModelCount}`);
  lines.push(`- Runtime-verified catalog models: ${inventory.summary.verifiedModelCount}`);
  lines.push(`- Benchmark-selected source architectures: ${inventory.summary.benchmarkSelectedSourceCount}`);
  lines.push(`- Sources pending benchmark-selected architecture: ${inventory.summary.benchmarkPendingSourceCount}`);
  lines.push('');
  lines.push('## Rollout Queue');
  lines.push('');
  lines.push('| Tier | Source checkpoint | Smallest artifact | Status | Preferred architecture | Evidence | Next action |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- |');
  for (const source of inventory.sourceModels) {
    lines.push(renderTableRow([
      source.tier,
      source.sourceCheckpointId,
      source.minSizeLabel,
      source.status,
      renderPreferred(source.preferredArchitecture),
      renderEvidence(source.preferredArchitecture),
      source.nextAction,
    ]));
  }
  lines.push('');
  lines.push('## Next Commands');
  lines.push('');
  lines.push('These are policy-generated command recipes, not evidence. A command becomes support evidence only after its saved artifact is committed and referenced by the claim lane.');
  lines.push('');
  lines.push('| Tier | Source checkpoint | Next gate | Command |');
  lines.push('| --- | --- | --- | --- |');
  for (const source of inventory.sourceModels) {
    if (!source.nextCommand) continue;
    lines.push(renderTableRow([
      source.tier,
      source.sourceCheckpointId,
      source.nextGate,
      renderCommand(source.nextCommand),
    ]));
  }
  for (const tier of inventory.policy.supportRollout.sizeTiers.map((entry) => entry.id)) {
    const tierSources = inventory.sourceModels.filter((source) => source.tier === tier);
    if (tierSources.length === 0) continue;
    lines.push('');
    lines.push(`## ${tier[0].toUpperCase()}${tier.slice(1)} Models`);
    lines.push('');
    lines.push('| Model ID | Family | Size | Architecture | Runtime verify | HF | Compare lane | Benchmark evidence | Next gate |');
    lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- |');
    for (const source of tierSources) {
      for (const variant of source.variants) {
        lines.push(renderTableRow([
          variant.modelId,
          variant.family,
          variant.sizeLabel,
          variant.architecture.label,
          variant.lifecycle.tested === 'verified'
            ? `${variant.lifecycle.lastVerifiedAt || 'verified'} (${variant.lifecycle.surfaces.join('+') || 'surface-unlisted'})`
            : (variant.lifecycle.tested === 'failed'
              ? `${variant.lifecycle.lastVerifiedAt || 'failed'} failed`
              : 'missing'),
          variant.hf.published ? `${variant.hf.repoId}@${variant.hf.revision}` : 'missing',
          variant.compare.profile ? `${variant.compare.profile.lane || 'unknown'} / ${variant.compare.profile.tjsModelId || 'no TJS mapping'}` : 'missing',
          variant.compare.benchmarkEvidenceOk ? `${variant.evidence.compareResult}<br>${variant.evidence.summarySvg}` : 'missing',
          variant.nextGate,
        ]));
      }
    }
  }
  lines.push('');
  lines.push('## Conversion-Only Configs');
  lines.push('');
  lines.push('These entries have checked-in conversion configs but are not catalog-supported runtime artifacts yet.');
  lines.push('');
  lines.push('| Config | Model base ID | Family | Architecture | Next action |');
  lines.push('| --- | --- | --- | --- | --- |');
  for (const entry of inventory.conversionOnly) {
    lines.push(renderTableRow([
      entry.conversionConfig,
      entry.modelBaseId,
      entry.family,
      entry.architecture.label,
      entry.nextAction,
    ]));
  }
  lines.push('');
  lines.push('## Source Files');
  lines.push('');
  for (const source of inventory.sources) {
    lines.push(`- ${source}`);
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
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
      throw new Error(`${repoRelative(filePath)} is missing; run npm run support:inventory:sync`);
    }
    if (current !== content) {
      throw new Error(`${repoRelative(filePath)} is stale; run npm run support:inventory:sync`);
    }
    return;
  }
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inventory = await buildInventory();
  const json = stableJson(inventory);
  const markdown = renderMarkdown(inventory);
  await writeOrCheck(args.jsonOutputPath, json, args.check);
  await writeOrCheck(args.markdownOutputPath, markdown, args.check);
  console.log(
    `model-support-inventory: ${args.check ? 'ok' : 'synced'} `
    + `(${repoRelative(args.jsonOutputPath)}, ${repoRelative(args.markdownOutputPath)})`
  );
}

await main();
