#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  buildModelTypeClusters,
  resolveModelTypeCluster,
  validateCatalogClassifications,
} from './lib/model-type-taxonomy.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const DEFAULT_OUTPUT_PATH = path.join(REPO_ROOT, 'docs/model-support-matrix.md');
const README_PATH = path.join(REPO_ROOT, 'README.md');
const README_MODEL_TYPES_START = '<!-- model-type-clusters:start -->';
const README_MODEL_TYPES_END = '<!-- model-type-clusters:end -->';
const CONVERSION_CONFIG_DIR = path.join(REPO_ROOT, 'src/config/conversion');
const CATALOG_PATH = path.join(REPO_ROOT, 'models/catalog.json');
const GEMMA4_TARGETS_PATH = path.join(REPO_ROOT, 'models/gemma4-targets.json');
const QUICKSTART_REGISTRY_PATH = path.join(REPO_ROOT, 'src', 'client', 'doppler-registry.json');
const RUNTIME_BLOCKED_MODEL_TYPES = new Set(['mamba', 'rwkv']);
const GEMMA4_TARGET_STATUS = new Set(['partially_verified', 'gap']);
const GEMMA4_SURFACE_STATUS = new Set(['verified', 'unverified', 'unsupported']);
const GEMMA4_CLAIM_STATUS = new Set(['verified', 'verified-local', 'experimental']);
const GEMMA4_MTP_STATUS = new Set(['not_implemented']);
const GEMMA4_SERVE_STATUS = new Set(['verified', 'unverified', 'unsupported']);
const GEMMA4_EVIDENCE_STATUS = new Set(['pass', 'performance_evidence', 'diagnostic']);
const GEMMA4_EVIDENCE_SURFACE = new Set(['browser', 'electron', 'node']);
const GEMMA4_SERVE_EVIDENCE_STATUS = new Set(['pass', 'diagnostic']);
const GEMMA4_PREFLIGHT_EVIDENCE_STATUS = new Set(['pass', 'diagnostic']);
const GEMMA4_SOURCE_PACKAGE_STATUS = new Set(['blocked', 'unverified', 'verified']);
const GEMMA4_BLOCKER_STATE = new Set(['missing', 'unsupported', 'unverified', 'diagnostic', 'not_implemented', 'incomplete']);
const GEMMA4_BLOCKER_SURFACE = new Set(['browser', 'electron', 'node', 'serve', 'mtp', 'model', 'benchmark']);
const GEMMA4_BLOCKER_CODE_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const GEMMA4_REQUIRED_SOURCE_URLS = Object.freeze([
  'https://ai.google.dev/gemma/docs/core',
  'https://ai.google.dev/gemma/docs/releases',
  'https://developers.google.com/edge/litert-lm/models/gemma-4',
]);
const GEMMA4_REQUIRED_TARGET_IDS = Object.freeze([
  'gemma-4-e2b',
  'gemma-4-e4b',
  'gemma-4-12b-unified',
  'gemma-4-31b',
  'gemma-4-26b-a4b',
]);
const FAMILY_ORDER = Object.freeze([
  'embeddinggemma',
  'gemma3',
  'translategemma',
  'gemma4',
  'qwen3',
  'lfm2',
]);

export function parseArgs(argv) {
  const args = {
    check: false,
    outputPath: DEFAULT_OUTPUT_PATH,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const entry = argv[i];
    const nextValue = () => {
      const candidate = argv[i + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${entry}`);
      }
      i += 1;
      return String(candidate).trim();
    };
    if (entry === '--check') {
      args.check = true;
      continue;
    }
    if (entry === '--output') {
      const candidate = nextValue();
      if (!candidate) {
        throw new Error('Missing value for --output');
      }
      args.outputPath = path.resolve(REPO_ROOT, candidate);
      continue;
    }
    throw new Error(`Unknown argument: ${entry}`);
  }
  return args;
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

function normalizeModelId(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : 'unknown-model';
}

function normalizeFamily(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : 'unknown';
}

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasText(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function hasFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isRepoRelativeJsonPath(value) {
  const candidate = typeof value === 'string' ? value.trim() : '';
  return !!(
    candidate
    && !path.isAbsolute(candidate)
    && !candidate.includes('\\')
    && !candidate.split('/').includes('..')
    && candidate.endsWith('.json')
  );
}

function sha256Hex(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

function hasGemma4Blocker(blockers, query) {
  return Array.isArray(blockers) && blockers.some((blocker) => {
    if (query.code && blocker?.code !== query.code) return false;
    if (query.surface && blocker?.surface !== query.surface) return false;
    if (query.state && blocker?.state !== query.state) return false;
    return true;
  });
}

function hasGemma4VerifiedSurfaceEvidence(runtimeReceipts, surface) {
  return Array.isArray(runtimeReceipts)
    && runtimeReceipts.some((receipt) => normalizeText(receipt?.surface) === surface && normalizeText(receipt?.status) === 'pass');
}

function hasGemma4VerifiedServeEvidence(serveReceipts) {
  return Array.isArray(serveReceipts)
    && serveReceipts.some((receipt) => normalizeText(receipt?.status) === 'pass');
}

function resolveReceiptModelId(payload) {
  return normalizeModelId(payload?.modelId || payload?.dopplerModelId);
}

function resolveRuntimeOutputEvidence(payload) {
  if (hasText(payload?.output)) return true;
  if (hasText(payload?.metrics?.generatedText)) return true;
  if (hasFiniteNumber(payload?.referenceTranscript?.output?.tokensGenerated)) return payload.referenceTranscript.output.tokensGenerated > 0;
  return false;
}

function resolveRuntimeContractEvidence(payload) {
  if (payload?.metrics?.executionContractArtifact?.ok === true) return true;
  if (payload?.executionContractArtifact?.ok === true) return true;
  if (payload?.schema === 'doppler.program-bundle/v1') {
    return hasText(payload?.sources?.manifest?.hash)
      && hasText(payload?.sources?.executionGraph?.hash)
      && hasText(payload?.execution?.graphHash)
      && Array.isArray(payload?.execution?.steps)
      && payload.execution.steps.length > 0;
  }
  return false;
}

function resolveServeReceiptModelId(payload) {
  return normalizeModelId(payload?.modelId || payload?.resolvedModel || payload?.dopplerModelId);
}

function resolveSha256Evidence(value) {
  return isObject(value)
    && normalizeText(value?.algorithm) === 'sha256'
    && /^[0-9a-f]{64}$/.test(String(value?.value || ''))
    && hasFiniteNumber(value?.bytes);
}

function resolveServeRequestEvidence(payload) {
  const messages = payload?.request?.messages;
  return isObject(messages)
    && hasFiniteNumber(messages?.count)
    && messages.count > 0
    && resolveSha256Evidence(messages?.digest)
    && resolveSha256Evidence(payload?.request?.generationDigest);
}

function resolveServeOutputEvidence(payload) {
  const output = payload?.output;
  return isObject(output)
    && normalizeText(output?.role) === 'assistant'
    && resolveSha256Evidence(output?.digest)
    && hasFiniteNumber(output?.textLength)
    && typeof output?.empty === 'boolean';
}

function resolveServeTranscriptEvidence(payload) {
  return resolveSha256Evidence(payload?.transcript?.digest);
}

function resolveServeFailureEvidence(payload) {
  const failure = payload?.failure;
  return isObject(failure)
    && hasText(failure?.code)
    && hasText(failure?.stage)
    && hasText(failure?.message)
    && hasText(failure?.modelId);
}

function resolveServeRuntimeModelSourceEvidence(payload) {
  const source = payload?.runtimeModelSource;
  if (!isObject(source)) {
    return false;
  }
  const kind = normalizeText(source.kind);
  if (kind === 'quickstart-registry') {
    return hasText(source.modelId);
  }
  if (kind === 'url') {
    return hasText(source.url);
  }
  if (kind === 'inline-manifest') {
    return hasText(source.modelId) && (source.baseUrl === null || hasText(source.baseUrl));
  }
  return false;
}

function resolveBenchmarkFields(payload) {
  const compareParity = payload?.sections?.compute?.parity;
  if (isObject(compareParity?.doppler?.result)) {
    const result = compareParity.doppler.result;
    return {
      modelId: normalizeModelId(payload?.dopplerModelId || result?.modelId),
      timing: result.timing,
      memoryStats: result.memoryStats,
      cacheMode: compareParity.cacheMode || result.cacheMode || result.timing?.cacheMode,
      loadMode: compareParity.loadMode || result.loadMode || result.timing?.loadMode,
      outputText: result.metrics?.generatedText || payload?.correctness?.doppler,
      correctnessStatus: payload?.correctness?.status,
      decodeValidityOk: compareParity?.decodeValidity?.ok,
    };
  }
  return {
    modelId: normalizeModelId(payload?.modelId),
    timing: payload?.timing,
    memoryStats: payload?.memoryStats,
    cacheMode: payload?.cacheMode || payload?.timing?.cacheMode,
    loadMode: payload?.loadMode || payload?.timing?.loadMode,
    outputText: payload?.metrics?.generatedText || payload?.output,
    correctnessStatus: null,
    decodeValidityOk: null,
  };
}

function compareFamilies(left, right) {
  const leftIndex = FAMILY_ORDER.indexOf(left);
  const rightIndex = FAMILY_ORDER.indexOf(right);
  if (leftIndex !== rightIndex) {
    if (leftIndex === -1) return 1;
    if (rightIndex === -1) return -1;
    return leftIndex - rightIndex;
  }
  return left.localeCompare(right);
}

function summarizeModes(model) {
  const modes = Array.isArray(model?.modes)
    ? model.modes.filter((entry) => typeof entry === 'string' && entry.trim())
    : [];
  return modes.length > 0 ? modes.join(', ') : 'run';
}

export function validateCatalogMatrixInputs(payload) {
  const errors = [];
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return ['catalog payload must be a JSON object'];
  }
  if (!Array.isArray(payload.models)) {
    return ['catalog payload must include a models array'];
  }
  const updatedAt = normalizeText(payload.updatedAt);
  if (!updatedAt) {
    errors.push('catalog updatedAt must be a non-empty string');
  }

  const seenModelIds = new Set();
  for (const model of payload.models) {
    const modelId = normalizeText(model?.modelId);
    if (!modelId) {
      errors.push('catalog entries must include modelId');
      continue;
    }
    if (seenModelIds.has(modelId)) {
      errors.push(`duplicate catalog modelId: ${modelId}`);
    }
    seenModelIds.add(modelId);

    const family = normalizeText(model?.family);
    if (!family) {
      errors.push(`${modelId}: family is required`);
    }

    const lifecycle = model?.lifecycle && typeof model.lifecycle === 'object' ? model.lifecycle : {};
    const availability = lifecycle?.availability && typeof lifecycle.availability === 'object'
      ? lifecycle.availability
      : {};
    const status = lifecycle?.status && typeof lifecycle.status === 'object' ? lifecycle.status : {};
    const artifact = model?.artifact && typeof model.artifact === 'object' ? model.artifact : {};
    const demo = normalizeText(status.demo);
    const baseUrl = normalizeText(model?.baseUrl);
    const hf = model?.hf && typeof model.hf === 'object' ? model.hf : {};
    const artifactFormat = normalizeText(artifact.format);

    if (!artifactFormat) {
      errors.push(`${modelId}: artifact.format is required`);
    } else if (artifactFormat !== 'rdrr' && artifactFormat !== 'direct-source') {
      errors.push(`${modelId}: artifact.format must be "rdrr" or "direct-source"`);
    }
    if (artifactFormat === 'direct-source' && normalizeText(artifact.sourceRuntimeSchema) !== 'direct-source/v1') {
      errors.push(`${modelId}: direct-source artifacts require artifact.sourceRuntimeSchema="direct-source/v1"`);
    }

    if (availability.hf === true) {
      if (!normalizeText(hf.repoId)) {
        errors.push(`${modelId}: lifecycle.availability.hf=true requires hf.repoId`);
      }
      if (!normalizeText(hf.revision)) {
        errors.push(`${modelId}: lifecycle.availability.hf=true requires hf.revision`);
      }
      if (!normalizeText(hf.path)) {
        errors.push(`${modelId}: lifecycle.availability.hf=true requires hf.path`);
      }
    }

    if (availability.curated === true && !(baseUrl.startsWith('./local/') || baseUrl.startsWith('local/'))) {
      errors.push(`${modelId}: lifecycle.availability.curated=true requires a repo-local baseUrl`);
    }
    if (demo === 'curated' && !(baseUrl.startsWith('./local/') || baseUrl.startsWith('local/'))) {
      errors.push(`${modelId}: lifecycle.status.demo=curated requires a repo-local baseUrl`);
    }
    if (demo === 'local' && !(baseUrl.startsWith('./local/') || baseUrl.startsWith('local/'))) {
      errors.push(`${modelId}: lifecycle.status.demo=local requires a local baseUrl`);
    }
  }

  errors.push(...validateCatalogClassifications(payload));

  return errors;
}

export function validateGemma4TargetMatrixInputs(payload, catalogModels, quickstartModels = []) {
  const errors = [];
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return ['Gemma 4 target matrix payload must be a JSON object'];
  }
  if (payload.schemaVersion !== 1) {
    errors.push('Gemma 4 target matrix schemaVersion must be 1');
  }
  const sourceUrls = Array.isArray(payload.sourceUrls) ? payload.sourceUrls : [];
  for (const sourceUrl of GEMMA4_REQUIRED_SOURCE_URLS) {
    if (!sourceUrls.includes(sourceUrl)) {
      errors.push(`Gemma 4 target matrix sourceUrls must include ${sourceUrl}`);
    }
  }
  if (!Array.isArray(payload.targets)) {
    errors.push('Gemma 4 target matrix must include a targets array');
    return errors;
  }
  const catalogById = new Map(
    (Array.isArray(catalogModels) ? catalogModels : [])
      .map((model) => [normalizeModelId(model?.modelId), model])
  );
  const quickstartById = new Map(
    (Array.isArray(quickstartModels) ? quickstartModels : [])
      .map((model) => [normalizeModelId(model?.modelId), model])
  );
  const seenTargetIds = new Set();

  for (const target of payload.targets) {
    const targetId = normalizeModelId(target?.targetId);
    if (!targetId || targetId === 'unknown-model') {
      errors.push('Gemma 4 target entries must include targetId');
      continue;
    }
    if (seenTargetIds.has(targetId)) {
      errors.push(`duplicate Gemma 4 targetId: ${targetId}`);
    }
    seenTargetIds.add(targetId);
    if (!GEMMA4_REQUIRED_TARGET_IDS.includes(targetId)) {
      errors.push(`${targetId}: unexpected Gemma 4 targetId`);
    }
    if (!normalizeText(target?.officialName)) {
      errors.push(`${targetId}: officialName is required`);
    }
    if (!GEMMA4_TARGET_STATUS.has(target?.dopplerStatus)) {
      errors.push(`${targetId}: invalid dopplerStatus`);
    }
    if (!GEMMA4_MTP_STATUS.has(target?.mtpStatus)) {
      errors.push(`${targetId}: invalid mtpStatus`);
    }
    if (target?.officialMtp !== true) {
      errors.push(`${targetId}: officialMtp must be true`);
    }
    const evidence = target?.evidence && typeof target.evidence === 'object' && !Array.isArray(target.evidence)
      ? target.evidence
      : null;
    if (!evidence) {
      errors.push(`${targetId}: evidence is required`);
    }
    const runtimeReceipts = Array.isArray(evidence?.runtimeReceipts) ? evidence.runtimeReceipts : null;
    const benchmarkReceipts = Array.isArray(evidence?.benchmarkReceipts) ? evidence.benchmarkReceipts : null;
    const serveReceipts = Array.isArray(evidence?.serveReceipts) ? evidence.serveReceipts : null;
    const preflightReceipts = Array.isArray(evidence?.preflightReceipts) ? evidence.preflightReceipts : null;
    if (!runtimeReceipts) {
      errors.push(`${targetId}: evidence.runtimeReceipts must be an array`);
    }
    if (!benchmarkReceipts) {
      errors.push(`${targetId}: evidence.benchmarkReceipts must be an array`);
    }
    if (!serveReceipts) {
      errors.push(`${targetId}: evidence.serveReceipts must be an array`);
    }
    if (!preflightReceipts) {
      errors.push(`${targetId}: evidence.preflightReceipts must be an array`);
    }
    if (!GEMMA4_SERVE_STATUS.has(target?.serveStatus)) {
      errors.push(`${targetId}: invalid serveStatus`);
    }
    for (const surface of ['browser', 'electron', 'node']) {
      if (!GEMMA4_SURFACE_STATUS.has(target?.surfaceStatus?.[surface])) {
        errors.push(`${targetId}: invalid ${surface} surfaceStatus`);
      }
    }
    const lanes = Array.isArray(target?.currentLanes) ? target.currentLanes : null;
    if (!lanes) {
      errors.push(`${targetId}: currentLanes must be an array`);
      continue;
    }
    const sourcePackages = target?.sourcePackages;
    if (sourcePackages !== undefined) {
      if (!Array.isArray(sourcePackages)) {
        errors.push(`${targetId}: sourcePackages must be an array when present`);
      } else {
        const seenSourcePackageIds = new Set();
        for (const sourcePackage of sourcePackages) {
          const sourcePackageId = typeof sourcePackage?.id === 'string' ? sourcePackage.id.trim() : '';
          const sourcePackageStatus = normalizeText(sourcePackage?.status);
          const blockerCode = typeof sourcePackage?.blockerCode === 'string' ? sourcePackage.blockerCode.trim() : '';
          if (!sourcePackageId) {
            errors.push(`${targetId}: source package id is required`);
          } else if (seenSourcePackageIds.has(sourcePackageId)) {
            errors.push(`${targetId}: duplicate source package ${sourcePackageId}`);
          }
          seenSourcePackageIds.add(sourcePackageId);
          if (!GEMMA4_SOURCE_PACKAGE_STATUS.has(sourcePackageStatus)) {
            errors.push(`${targetId}: source package ${sourcePackageId || 'unknown'} has invalid status`);
          }
          if (sourcePackageStatus === 'blocked' && !GEMMA4_BLOCKER_CODE_PATTERN.test(blockerCode)) {
            errors.push(`${targetId}: blocked source package ${sourcePackageId || 'unknown'} requires blockerCode`);
          }
          if (typeof sourcePackage?.reason !== 'string' || !sourcePackage.reason.trim()) {
            errors.push(`${targetId}: source package ${sourcePackageId || 'unknown'} reason is required`);
          }
        }
      }
    }
    const laneIds = new Set(lanes.map((lane) => normalizeModelId(lane?.modelId)));
    const servedLanes = Array.isArray(target?.servedLanes) ? target.servedLanes : null;
    if (!servedLanes) {
      errors.push(`${targetId}: servedLanes must be an array`);
    } else if ((target.serveStatus === 'verified' || target.serveStatus === 'unverified') && servedLanes.length === 0) {
      errors.push(`${targetId}: ${target.serveStatus} serveStatus must list servedLanes`);
    } else if (target.serveStatus === 'unsupported' && servedLanes.length !== 0) {
      errors.push(`${targetId}: unsupported serveStatus must not list servedLanes`);
    }
    const missing = Array.isArray(target?.missing) ? target.missing : null;
    if (!missing) {
      errors.push(`${targetId}: missing must be an array`);
    } else if (target.dopplerStatus === 'gap' && missing.length === 0) {
      errors.push(`${targetId}: gap targets must explain missing work`);
    }
    const blockers = Array.isArray(target?.blockers) ? target.blockers : null;
    if (!blockers) {
      errors.push(`${targetId}: blockers must be an array`);
    } else {
      const seenBlockerCodes = new Set();
      for (const blocker of blockers) {
        const code = typeof blocker?.code === 'string' ? blocker.code.trim() : '';
        const surface = normalizeText(blocker?.surface);
        const state = normalizeText(blocker?.state);
        const reason = typeof blocker?.reason === 'string' ? blocker.reason.trim() : '';
        if (!GEMMA4_BLOCKER_CODE_PATTERN.test(code)) {
          errors.push(`${targetId}: blocker code must be kebab-case`);
        } else if (seenBlockerCodes.has(code)) {
          errors.push(`${targetId}: duplicate blocker ${code}`);
        }
        seenBlockerCodes.add(code);
        if (!GEMMA4_BLOCKER_SURFACE.has(surface)) {
          errors.push(`${targetId}: blocker ${code || 'unknown'} has invalid surface`);
        }
        if (!GEMMA4_BLOCKER_STATE.has(state)) {
          errors.push(`${targetId}: blocker ${code || 'unknown'} has invalid state`);
        }
        if (!reason) {
          errors.push(`${targetId}: blocker ${code || 'unknown'} reason is required`);
        }
      }
      if (target.dopplerStatus === 'gap' && blockers.length === 0) {
        errors.push(`${targetId}: gap targets must list blockers`);
      }
      for (const surface of ['browser', 'electron', 'node']) {
        const surfaceStatus = target?.surfaceStatus?.[surface];
        if ((surfaceStatus === 'unverified' || surfaceStatus === 'unsupported')
          && !hasGemma4Blocker(blockers, { surface, state: surfaceStatus })) {
          errors.push(`${targetId}: ${surface} ${surfaceStatus} status must have a matching blocker`);
        }
      }
      if (target.serveStatus === 'unverified' && !hasGemma4Blocker(blockers, { surface: 'serve', state: 'unverified' })) {
        errors.push(`${targetId}: unverified serveStatus must have a matching blocker`);
      }
      if (target.serveStatus === 'unsupported' && !hasGemma4Blocker(blockers, { surface: 'serve', state: 'unsupported' })) {
        errors.push(`${targetId}: unsupported serveStatus must have a matching blocker`);
      }
      if (target.mtpStatus === 'not_implemented' && !hasGemma4Blocker(blockers, { surface: 'mtp', state: 'not_implemented' })) {
        errors.push(`${targetId}: not_implemented MTP status must have a matching blocker`);
      }
      if (target.dopplerStatus === 'gap' && !blockers.some((blocker) => blocker?.surface === 'model')) {
        errors.push(`${targetId}: gap targets must include at least one model blocker`);
      }
    }
    if (target.dopplerStatus === 'gap' && lanes.length !== 0) {
      errors.push(`${targetId}: gap targets must not list current lanes`);
    }
    if (target.dopplerStatus !== 'gap' && lanes.length === 0) {
      errors.push(`${targetId}: non-gap targets must list current lanes`);
    }
    if (target.dopplerStatus !== 'gap' && runtimeReceipts && runtimeReceipts.length === 0) {
      errors.push(`${targetId}: non-gap targets must list runtime receipt evidence`);
    }
    if (target.dopplerStatus === 'gap') {
      if (runtimeReceipts && runtimeReceipts.length !== 0) {
        errors.push(`${targetId}: gap targets must not list runtime receipt evidence`);
      }
      if (benchmarkReceipts && benchmarkReceipts.length !== 0) {
        errors.push(`${targetId}: gap targets must not list benchmark receipt evidence`);
      }
      if (serveReceipts && serveReceipts.length !== 0) {
        errors.push(`${targetId}: gap targets must not list serve receipt evidence`);
      }
      if (preflightReceipts && preflightReceipts.length !== 0) {
        errors.push(`${targetId}: gap targets must not list preflight receipt evidence`);
      }
    }
    if (target.mtpStatus === 'not_implemented' && missing && !missing.includes('mtp lane')) {
      errors.push(`${targetId}: not_implemented MTP status must list missing "mtp lane"`);
    }
    if (target.serveStatus === 'verified' && !hasGemma4VerifiedServeEvidence(serveReceipts)) {
      errors.push(`${targetId}: verified serveStatus must list serve pass receipt evidence`);
    }
    if (target.serveStatus === 'unverified' && missing && !missing.includes('doppler-serve runtime pass receipt')) {
      errors.push(`${targetId}: unverified serveStatus must list missing "doppler-serve runtime pass receipt"`);
    }
    if (target.serveStatus === 'unsupported' && missing && lanes.length > 0 && !missing.includes('doppler-serve quickstart lane')) {
      errors.push(`${targetId}: unsupported serveStatus with current lanes must list missing "doppler-serve quickstart lane"`);
    }
    const seenLaneIds = new Set();
    for (const lane of lanes) {
      const laneModelId = normalizeModelId(lane?.modelId);
      if (seenLaneIds.has(laneModelId)) {
        errors.push(`${targetId}: duplicate lane ${laneModelId}`);
      }
      seenLaneIds.add(laneModelId);
      if (!normalizeText(lane?.role)) {
        errors.push(`${targetId}: lane ${laneModelId} role is required`);
      }
      const catalogModel = catalogById.get(laneModelId);
      if (!catalogModel) {
        errors.push(`${targetId}: lane ${laneModelId} is missing from models/catalog.json`);
        continue;
      }
      if (catalogModel.family !== 'gemma4') {
        errors.push(`${targetId}: lane ${laneModelId} is not a gemma4 catalog model`);
      }
      if (!GEMMA4_CLAIM_STATUS.has(lane?.claimStatus)) {
        errors.push(`${targetId}: lane ${laneModelId} has invalid claimStatus`);
      }
      if (lane.claimStatus === 'verified' && catalogModel.lifecycle?.status?.tested !== 'verified') {
        errors.push(`${targetId}: lane ${laneModelId} is marked verified but catalog lifecycle is not verified`);
      }
    }
    const evidenceLaneIds = new Set();
    for (const receipt of [...(runtimeReceipts || []), ...(benchmarkReceipts || [])]) {
      const receiptPath = typeof receipt?.path === 'string' ? receipt.path.trim() : '';
      const receiptModelId = normalizeModelId(receipt?.modelId);
      const receiptSurface = normalizeText(receipt?.surface);
      const receiptStatus = normalizeText(receipt?.status);
      if (!receiptPath) {
        errors.push(`${targetId}: evidence receipt path is required`);
      } else if (path.isAbsolute(receiptPath) || receiptPath.includes('\\') || receiptPath.split('/').includes('..')) {
        errors.push(`${targetId}: evidence receipt path must be a repo-relative JSON path`);
      } else if (!receiptPath.endsWith('.json')) {
        errors.push(`${targetId}: evidence receipt ${receiptPath} must be a JSON file`);
      }
      if (!laneIds.has(receiptModelId)) {
        errors.push(`${targetId}: evidence receipt model ${receiptModelId} must be listed in currentLanes`);
      }
      if (!GEMMA4_EVIDENCE_SURFACE.has(receiptSurface)) {
        errors.push(`${targetId}: evidence receipt ${receiptPath || receiptModelId} has invalid surface`);
      }
      if (!GEMMA4_EVIDENCE_STATUS.has(receiptStatus)) {
        errors.push(`${targetId}: evidence receipt ${receiptPath || receiptModelId} has invalid status`);
      }
      if (receiptStatus === 'pass') {
        evidenceLaneIds.add(receiptModelId);
      }
    }
    const servedLaneIds = new Set((servedLanes || []).map((laneId) => normalizeModelId(laneId)));
    for (const receipt of serveReceipts || []) {
      const receiptPath = typeof receipt?.path === 'string' ? receipt.path.trim() : '';
      const receiptModelId = normalizeModelId(receipt?.modelId);
      const receiptSurface = normalizeText(receipt?.surface);
      const receiptStatus = normalizeText(receipt?.status);
      if (!receiptPath) {
        errors.push(`${targetId}: serve receipt path is required`);
      } else if (path.isAbsolute(receiptPath) || receiptPath.includes('\\') || receiptPath.split('/').includes('..')) {
        errors.push(`${targetId}: serve receipt path must be a repo-relative JSON path`);
      } else if (!receiptPath.endsWith('.json')) {
        errors.push(`${targetId}: serve receipt ${receiptPath} must be a JSON file`);
      }
      if (!laneIds.has(receiptModelId)) {
        errors.push(`${targetId}: serve receipt model ${receiptModelId} must be listed in currentLanes`);
      }
      if (!servedLaneIds.has(receiptModelId)) {
        errors.push(`${targetId}: serve receipt model ${receiptModelId} must be listed in servedLanes`);
      }
      if (receiptSurface !== 'serve') {
        errors.push(`${targetId}: serve receipt ${receiptPath || receiptModelId} must use surface "serve"`);
      }
      if (!GEMMA4_SERVE_EVIDENCE_STATUS.has(receiptStatus)) {
        errors.push(`${targetId}: serve receipt ${receiptPath || receiptModelId} has invalid status`);
      }
    }
    for (const receipt of preflightReceipts || []) {
      const receiptPath = typeof receipt?.path === 'string' ? receipt.path.trim() : '';
      const receiptModelId = normalizeModelId(receipt?.modelId);
      const receiptSurface = normalizeText(receipt?.surface);
      const receiptStatus = normalizeText(receipt?.status);
      if (!receiptPath) {
        errors.push(`${targetId}: preflight receipt path is required`);
      } else if (path.isAbsolute(receiptPath) || receiptPath.includes('\\') || receiptPath.split('/').includes('..')) {
        errors.push(`${targetId}: preflight receipt path must be a repo-relative JSON path`);
      } else if (!receiptPath.endsWith('.json')) {
        errors.push(`${targetId}: preflight receipt ${receiptPath} must be a JSON file`);
      }
      if (!laneIds.has(receiptModelId)) {
        errors.push(`${targetId}: preflight receipt model ${receiptModelId} must be listed in currentLanes`);
      }
      if (!GEMMA4_EVIDENCE_SURFACE.has(receiptSurface)) {
        errors.push(`${targetId}: preflight receipt ${receiptPath || receiptModelId} has invalid surface`);
      }
      if (!GEMMA4_PREFLIGHT_EVIDENCE_STATUS.has(receiptStatus)) {
        errors.push(`${targetId}: preflight receipt ${receiptPath || receiptModelId} has invalid status`);
      }
    }
    for (const lane of lanes) {
      const laneModelId = normalizeModelId(lane?.modelId);
      if ((lane?.claimStatus === 'verified' || lane?.claimStatus === 'verified-local') && !evidenceLaneIds.has(laneModelId)) {
        errors.push(`${targetId}: verified lane ${laneModelId} must have passing receipt evidence`);
      }
    }
    for (const surface of ['browser', 'node']) {
      if (target?.surfaceStatus?.[surface] !== 'verified') {
        continue;
      }
      if (!hasGemma4VerifiedSurfaceEvidence(runtimeReceipts, surface)) {
        errors.push(`${targetId}: verified ${surface} surface must have same-surface runtime pass evidence`);
      }
    }
    if (servedLanes) {
      const seenServedLaneIds = new Set();
      for (const servedLaneIdValue of servedLanes) {
        const servedLaneId = normalizeModelId(servedLaneIdValue);
        if (seenServedLaneIds.has(servedLaneId)) {
          errors.push(`${targetId}: duplicate served lane ${servedLaneId}`);
        }
        seenServedLaneIds.add(servedLaneId);
        if (!laneIds.has(servedLaneId)) {
          errors.push(`${targetId}: served lane ${servedLaneId} must be listed in currentLanes`);
        }
        const quickstartModel = quickstartById.get(servedLaneId);
        if (!quickstartModel) {
          errors.push(`${targetId}: served lane ${servedLaneId} is missing from src/client/doppler-registry.json`);
          continue;
        }
        const modes = Array.isArray(quickstartModel?.modes) ? quickstartModel.modes : [];
        if (!modes.includes('text')) {
          errors.push(`${targetId}: served lane ${servedLaneId} must include text mode in quickstart registry`);
        }
      }
    }
  }
  for (const targetId of GEMMA4_REQUIRED_TARGET_IDS) {
    if (!seenTargetIds.has(targetId)) {
      errors.push(`missing Gemma 4 targetId: ${targetId}`);
    }
  }

  return errors;
}

function validateGemma4RuntimeReceiptPayload(targetId, receipt, payload) {
  const errors = [];
  const receiptPath = typeof receipt?.path === 'string' ? receipt.path.trim() : '';
  const receiptModelId = normalizeModelId(receipt?.modelId);
  const payloadModelId = resolveReceiptModelId(payload);
  if (payloadModelId !== receiptModelId) {
    errors.push(`${targetId}: runtime receipt ${receiptPath} modelId mismatch (${payloadModelId} != ${receiptModelId})`);
  }
  if (receipt?.status === 'pass' && !resolveRuntimeContractEvidence(payload)) {
    errors.push(`${targetId}: runtime receipt ${receiptPath} pass status requires execution-contract or program-bundle evidence`);
  }
  if (receipt?.status === 'pass' && !resolveRuntimeOutputEvidence(payload)) {
    errors.push(`${targetId}: runtime receipt ${receiptPath} pass status requires output or transcript evidence`);
  }
  const hasAdapter = isObject(payload?.deviceInfo?.adapterInfo)
    || isObject(payload?.captureProfile?.adapter?.deviceInfo?.adapterInfo);
  if (receipt?.status === 'pass' && !hasAdapter) {
    errors.push(`${targetId}: runtime receipt ${receiptPath} pass status requires adapter identity evidence`);
  }
  return errors;
}

function validateGemma4BenchmarkReceiptPayload(targetId, receipt, payload) {
  const errors = [];
  const receiptPath = typeof receipt?.path === 'string' ? receipt.path.trim() : '';
  const receiptModelId = normalizeModelId(receipt?.modelId);
  const fields = resolveBenchmarkFields(payload);
  if (fields.modelId !== receiptModelId) {
    errors.push(`${targetId}: benchmark receipt ${receiptPath} modelId mismatch (${fields.modelId} != ${receiptModelId})`);
  }
  const timing = isObject(fields.timing) ? fields.timing : {};
  for (const metric of ['decodeTokensPerSec', 'firstTokenMs', 'totalRunMs']) {
    if (!hasFiniteNumber(timing[metric])) {
      errors.push(`${targetId}: benchmark receipt ${receiptPath} missing numeric timing.${metric}`);
    }
  }
  const memoryStats = isObject(fields.memoryStats) ? fields.memoryStats : {};
  if (!hasFiniteNumber(memoryStats.used)) {
    errors.push(`${targetId}: benchmark receipt ${receiptPath} missing numeric memoryStats.used`);
  }
  if (!isObject(memoryStats.kvCache)) {
    errors.push(`${targetId}: benchmark receipt ${receiptPath} missing memoryStats.kvCache`);
  }
  if (!hasText(fields.cacheMode)) {
    errors.push(`${targetId}: benchmark receipt ${receiptPath} missing cacheMode evidence`);
  }
  if (!hasText(fields.loadMode)) {
    errors.push(`${targetId}: benchmark receipt ${receiptPath} missing loadMode evidence`);
  }
  if (!hasText(fields.outputText)) {
    errors.push(`${targetId}: benchmark receipt ${receiptPath} missing generated output evidence`);
  }
  if (receipt?.status === 'performance_evidence') {
    if (!hasText(fields.correctnessStatus)) {
      errors.push(`${targetId}: benchmark receipt ${receiptPath} performance evidence requires correctness status`);
    }
    if (fields.decodeValidityOk !== true) {
      errors.push(`${targetId}: benchmark receipt ${receiptPath} performance evidence requires decode validity`);
    }
  }
  return errors;
}

function validateGemma4ServeReceiptPayload(targetId, receipt, payload) {
  const errors = [];
  const receiptPath = typeof receipt?.path === 'string' ? receipt.path.trim() : '';
  const receiptModelId = normalizeModelId(receipt?.modelId);
  const payloadModelId = resolveServeReceiptModelId(payload);
  if (payloadModelId !== receiptModelId) {
    errors.push(`${targetId}: serve receipt ${receiptPath} modelId mismatch (${payloadModelId} != ${receiptModelId})`);
  }
  if (payload?.receiptVersion !== 'doppler_serve_receipt_v1' || payload?.schemaVersion !== 1) {
    errors.push(`${targetId}: serve receipt ${receiptPath} requires doppler_serve_receipt_v1 schema`);
  }
  if (normalizeText(payload?.surface) !== 'serve' || normalizeText(payload?.status) !== normalizeText(receipt?.status)) {
    errors.push(`${targetId}: serve receipt ${receiptPath} requires matching surface=serve and status`);
  }
  if (normalizeText(payload?.runtime) !== 'doppler-gpu' || normalizeText(payload?.runtimePath) !== 'doppler-gpu.chattext') {
    errors.push(`${targetId}: serve receipt ${receiptPath} requires doppler-gpu chatText runtime path evidence`);
  }
  if (!isObject(payload?.artifact)
    || !hasText(payload.artifact?.sourceCheckpointId)
    || !hasText(payload.artifact?.weightPackId)
    || !hasText(payload.artifact?.manifestVariantId)) {
    errors.push(`${targetId}: serve receipt ${receiptPath} requires artifact identity evidence`);
  }
  if (!resolveServeRequestEvidence(payload)) {
    errors.push(`${targetId}: serve receipt ${receiptPath} requires request digest evidence`);
  }
  if (receipt?.status !== 'pass') {
    if (!resolveServeFailureEvidence(payload)) {
      errors.push(`${targetId}: serve receipt ${receiptPath} diagnostic status requires failure evidence`);
    }
    const failure = payload?.failure;
    const failureStage = normalizeText(failure?.stage);
    if (failureStage === 'loadweights') {
      const weightFailure = failure?.weightLoadFailure;
      if (!isObject(weightFailure)
        || !hasText(weightFailure?.tensorName)
        || !hasText(weightFailure?.tensorRole)
        || !hasText(weightFailure?.tensorDtype)
        || !Array.isArray(weightFailure?.tensorShape)
        || !hasFiniteNumber(weightFailure?.tensorSizeBytes)
        || !hasText(weightFailure?.tensorLoadStage)) {
        errors.push(`${targetId}: serve receipt ${receiptPath} loadWeights diagnostic requires tensor failure evidence`);
      }
      if (normalizeText(weightFailure?.tensorLoadStage) === 'gpuresidentembeddinglimitpreflight') {
        const limitFailure = weightFailure?.deviceLimitFailure;
        if (!isObject(limitFailure)
          || normalizeText(limitFailure?.kind) !== 'gpu_resident_embedding_exceeds_device_limit'
          || !hasFiniteNumber(limitFailure?.maxGpuResidentBytes)
          || !hasFiniteNumber(limitFailure?.maxStorageBufferBindingSize)
          || !hasFiniteNumber(limitFailure?.maxBufferSize)
          || !hasFiniteNumber(limitFailure?.largeWeightMaxBytes)
          || !hasFiniteNumber(limitFailure?.requiredSplitSections)
          || !isObject(limitFailure?.embeddingKernel)
          || !hasText(limitFailure.embeddingKernel?.kernel)
          || !hasText(limitFailure.embeddingKernel?.entry)) {
          errors.push(`${targetId}: serve receipt ${receiptPath} embedding limit diagnostic requires device limit evidence`);
        }
      }
    }
    return errors;
  }
  if (!resolveServeRuntimeModelSourceEvidence(payload)) {
    errors.push(`${targetId}: serve receipt ${receiptPath} pass status requires runtime model source evidence`);
  }
  if (!resolveServeOutputEvidence(payload)) {
    errors.push(`${targetId}: serve receipt ${receiptPath} pass status requires assistant output digest evidence`);
  }
  if (!resolveServeTranscriptEvidence(payload)) {
    errors.push(`${targetId}: serve receipt ${receiptPath} pass status requires transcript digest evidence`);
  }
  if (!isObject(payload?.usage)
    || !hasFiniteNumber(payload.usage?.promptTokens)
    || !hasFiniteNumber(payload.usage?.completionTokens)
    || !hasFiniteNumber(payload.usage?.totalTokens)) {
    errors.push(`${targetId}: serve receipt ${receiptPath} pass status requires token usage evidence`);
  }
  return errors;
}

async function validateGemma4PreflightReceiptPayload(targetId, receipt, payload) {
  const errors = [];
  const receiptPath = typeof receipt?.path === 'string' ? receipt.path.trim() : '';
  const receiptModelId = normalizeModelId(receipt?.modelId);
  const payloadModelId = normalizeModelId(payload?.modelId);
  if (payloadModelId !== receiptModelId) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} modelId mismatch (${payloadModelId} != ${receiptModelId})`);
  }
  if (payload?.receiptVersion !== 'doppler_gemma4_preflight_receipt_v1' || payload?.schemaVersion !== 1) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires doppler_gemma4_preflight_receipt_v1 schema`);
  }
  if (normalizeText(payload?.surface) !== normalizeText(receipt?.surface) || normalizeText(payload?.status) !== normalizeText(receipt?.status)) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires matching surface and status`);
  }
  if (normalizeText(payload?.runtime) !== 'doppler-gpu') {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires doppler-gpu runtime evidence`);
  }
  if (normalizeText(payload?.target) !== 'gpuresidentembeddinglimit') {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires gpuResidentEmbeddingLimit target`);
  }
  if (!isObject(payload?.manifest)
    || !hasText(payload.manifest?.path)
    || !/^sha256:[0-9a-f]{64}$/.test(String(payload.manifest?.sha256 || ''))) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires manifest path and sha256`);
  } else if (!isRepoRelativeJsonPath(payload.manifest.path)) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} manifest path must be a repo-relative JSON path`);
  } else {
    const manifestPath = payload.manifest.path.trim();
    const manifestRaw = await fs.readFile(path.join(REPO_ROOT, manifestPath), 'utf8');
    const manifestSha256 = `sha256:${sha256Hex(manifestRaw)}`;
    if (manifestSha256 !== payload.manifest.sha256) {
      errors.push(`${targetId}: preflight receipt ${receiptPath} manifest sha256 mismatch (${payload.manifest.sha256} != ${manifestSha256})`);
    }
    const manifest = JSON.parse(manifestRaw);
    const manifestModelId = normalizeModelId(manifest?.modelId);
    if (manifestModelId !== receiptModelId) {
      errors.push(`${targetId}: preflight receipt ${receiptPath} manifest modelId mismatch (${manifestModelId} != ${receiptModelId})`);
    }
    const embedding = payload?.embedding;
    const tensorName = typeof embedding?.tensorName === 'string' ? embedding.tensorName.trim() : '';
    const tensorLocation = tensorName ? manifest?.tensors?.[tensorName] : null;
    if (!tensorLocation || typeof tensorLocation !== 'object') {
      errors.push(`${targetId}: preflight receipt ${receiptPath} embedding tensor ${tensorName || '(missing)'} is absent from manifest`);
    } else {
      if (normalizeText(tensorLocation?.dtype) !== normalizeText(embedding?.dtype)) {
        errors.push(`${targetId}: preflight receipt ${receiptPath} embedding dtype mismatch (${embedding?.dtype} != ${tensorLocation?.dtype})`);
      }
      if (JSON.stringify(tensorLocation?.shape ?? null) !== JSON.stringify(embedding?.shape ?? null)) {
        errors.push(`${targetId}: preflight receipt ${receiptPath} embedding shape mismatch`);
      }
      if (hasFiniteNumber(tensorLocation?.size) && tensorLocation.size !== embedding?.tensorSizeBytes) {
        errors.push(`${targetId}: preflight receipt ${receiptPath} embedding tensorSizeBytes mismatch (${embedding?.tensorSizeBytes} != ${tensorLocation.size})`);
      }
      const overrides = manifest?.inference?.largeWeights?.gpuResidentOverrides;
      if (!Array.isArray(overrides) || !overrides.includes(tensorName)) {
        errors.push(`${targetId}: preflight receipt ${receiptPath} manifest must force the preflight embedding through gpuResidentOverrides`);
      }
    }
    const manifestKernel = manifest?.inference?.execution?.kernels?.embed;
    if (!isObject(manifestKernel)) {
      errors.push(`${targetId}: preflight receipt ${receiptPath} manifest is missing inference.execution.kernels.embed`);
    } else {
      const receiptKernel = payload?.embedding?.kernel;
      for (const field of ['kernel', 'entry', 'digest']) {
        if (manifestKernel?.[field] !== receiptKernel?.[field]) {
          errors.push(`${targetId}: preflight receipt ${receiptPath} embedding kernel ${field} mismatch (${receiptKernel?.[field] ?? 'missing'} != ${manifestKernel?.[field] ?? 'missing'})`);
        }
      }
    }
  }
  if (!isObject(payload?.adapterInfo)
    || !hasText(payload.adapterInfo?.vendor)
    || !hasText(payload.adapterInfo?.architecture)) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires adapter identity evidence`);
  }
  if (!isObject(payload?.deviceLimits)
    || !hasFiniteNumber(payload.deviceLimits?.maxStorageBufferBindingSize)
    || !hasFiniteNumber(payload.deviceLimits?.maxBufferSize)
    || !hasFiniteNumber(payload.deviceLimits?.maxStorageBuffersPerShaderStage)) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires numeric device limit evidence`);
  }
  const embedding = payload?.embedding;
  if (!isObject(embedding)
    || !hasText(embedding?.tensorName)
    || !hasText(embedding?.dtype)
    || !Array.isArray(embedding?.shape)
    || !hasFiniteNumber(embedding?.tensorSizeBytes)
    || !isObject(embedding?.kernel)
    || !hasText(embedding.kernel?.kernel)
    || !hasText(embedding.kernel?.entry)
    || !/^sha256:[0-9a-f]{64}$/.test(String(embedding.kernel?.digest || ''))) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires embedding tensor and kernel identity evidence`);
  }
  const preflight = payload?.preflight;
  if (!isObject(preflight)
    || typeof preflight?.ok !== 'boolean'
    || preflight?.splitKernelExpected !== true
    || !hasFiniteNumber(preflight?.activeSplitKernelMaxSections)
    || !hasFiniteNumber(preflight?.maxSplitEmbeddingSections)
    || !hasFiniteNumber(preflight?.requiredSplitSections)) {
    errors.push(`${targetId}: preflight receipt ${receiptPath} requires split-kernel preflight evidence`);
    return errors;
  }
  if (receipt?.status === 'pass') {
    if (preflight.ok !== true) {
      errors.push(`${targetId}: preflight receipt ${receiptPath} pass status requires preflight.ok=true`);
    }
    if (preflight.requiredSplitSections > preflight.maxSplitEmbeddingSections) {
      errors.push(`${targetId}: preflight receipt ${receiptPath} pass status requires requiredSplitSections <= maxSplitEmbeddingSections`);
    }
  }
  return errors;
}

export async function validateGemma4EvidenceFiles(payload) {
  const errors = [];
  const targets = Array.isArray(payload?.targets) ? payload.targets : [];
  for (const target of targets) {
    const targetId = normalizeModelId(target?.targetId);
    const evidence = target?.evidence && typeof target.evidence === 'object' && !Array.isArray(target.evidence)
      ? target.evidence
      : {};
    const receiptGroups = [
      {
        kind: 'runtime',
        entries: Array.isArray(evidence.runtimeReceipts) ? evidence.runtimeReceipts : [],
        validate: validateGemma4RuntimeReceiptPayload,
      },
      {
        kind: 'benchmark',
        entries: Array.isArray(evidence.benchmarkReceipts) ? evidence.benchmarkReceipts : [],
        validate: validateGemma4BenchmarkReceiptPayload,
      },
      {
        kind: 'serve',
        entries: Array.isArray(evidence.serveReceipts) ? evidence.serveReceipts : [],
        validate: validateGemma4ServeReceiptPayload,
      },
      {
        kind: 'preflight',
        entries: Array.isArray(evidence.preflightReceipts) ? evidence.preflightReceipts : [],
        validate: validateGemma4PreflightReceiptPayload,
      },
    ];
    for (const group of receiptGroups) {
      for (const receipt of group.entries) {
        const receiptPath = typeof receipt?.path === 'string' ? receipt.path.trim() : '';
        if (!receiptPath || path.isAbsolute(receiptPath) || receiptPath.includes('\\') || receiptPath.split('/').includes('..')) {
          continue;
        }
        try {
          const fullPath = path.join(REPO_ROOT, receiptPath);
          const stat = await fs.stat(fullPath);
          if (!stat.isFile()) {
            errors.push(`${targetId}: evidence receipt ${receiptPath} is not a file`);
            continue;
          }
          const payload = await readJson(fullPath);
          errors.push(...await group.validate(targetId, receipt, payload));
        } catch (error) {
          if (error && error.code === 'ENOENT') {
            errors.push(`${targetId}: evidence receipt ${receiptPath} is missing`);
            continue;
          }
          throw error;
        }
      }
    }
  }
  return errors;
}

function relativePath(filePath) {
  return path.relative(REPO_ROOT, filePath).replace(/\\/g, '/');
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function collectJsonFiles(rootDir) {
  const entries = await fs.readdir(rootDir, { withFileTypes: true });
  const files = [];
  const ordered = entries.sort((left, right) => left.name.localeCompare(right.name));
  for (const entry of ordered) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      const nested = await collectJsonFiles(fullPath);
      files.push(...nested);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(fullPath);
    }
  }
  return files;
}

function inferFamilyFromModelId(modelId) {
  const normalized = normalizeText(modelId);
  if (!normalized) return null;
  if (normalized.startsWith('google-embeddinggemma-') || normalized.startsWith('embeddinggemma-')) return 'embeddinggemma';
  if (normalized.startsWith('translategemma-')) return 'translategemma';
  if (normalized.startsWith('gemma-4-')) return 'gemma4';
  if (normalized.startsWith('gemma-3-')) return 'gemma3';
  if (normalized.startsWith('qwen-3-')) return 'qwen3';
  if (normalized.startsWith('lfm2')) return 'lfm2';
  if (normalized.startsWith('gpt-oss-')) return 'gpt_oss';
  if (normalized.startsWith('janus-')) return 'janus_text';
  return null;
}

function inferFamilyFromConversionConfig(payload, filePath) {
  const modelBaseId = normalizeText(payload?.output?.modelBaseId);
  const modelIdFamily = inferFamilyFromModelId(modelBaseId);
  if (modelIdFamily) {
    return modelIdFamily;
  }

  const relative = path.relative(CONVERSION_CONFIG_DIR, filePath).replace(/\\/g, '/');
  const [head] = relative.split('/');
  const normalizedHead = normalizeText(head.replace(/\.json$/i, ''));
  if (normalizedHead === 'janus') return 'janus_text';
  return normalizedHead || 'unknown';
}

function inferRuntimeModelTypeFromConversionConfig(payload) {
  const explicit = normalizeText(payload?.manifest?.modelType);
  if (explicit) {
    return explicit;
  }
  const modelBaseId = normalizeText(payload?.output?.modelBaseId);
  if (modelBaseId.startsWith('google-embeddinggemma-') || modelBaseId.startsWith('embeddinggemma-')) {
    return 'embedding';
  }
  return 'transformer';
}

function resolveRuntimeStatus(modelType) {
  return RUNTIME_BLOCKED_MODEL_TYPES.has(modelType) ? 'blocked' : 'active';
}

function createEmptyLifecycleAggregate() {
  return {
    hosted: false,
    demo: 'none',
    tested: 'unknown',
    testedAt: null,
    catalogCount: 0,
    verifiedCount: 0,
    failedCount: 0,
  };
}

function normalizeTestedState(value) {
  const normalized = normalizeText(value);
  if (normalized === 'verified' || normalized === 'pass' || normalized === 'passed') return 'verified';
  if (normalized === 'failed' || normalized === 'fail' || normalized === 'failing') return 'failed';
  return 'unknown';
}

function resolveCatalogLifecycle(model) {
  const lifecycle = model?.lifecycle && typeof model.lifecycle === 'object' ? model.lifecycle : {};
  const availability = lifecycle?.availability && typeof lifecycle.availability === 'object'
    ? lifecycle.availability
    : {};
  const status = lifecycle?.status && typeof lifecycle.status === 'object' ? lifecycle.status : {};
  const tested = lifecycle?.tested && typeof lifecycle.tested === 'object' ? lifecycle.tested : {};

  const baseUrl = typeof model?.baseUrl === 'string' ? model.baseUrl.trim() : '';
  const fallbackDemo = baseUrl.startsWith('./local/') || baseUrl.startsWith('local/')
    ? 'local'
    : 'none';
  const demo = normalizeText(status.demo) || fallbackDemo;
  const hosted = typeof availability.hf === 'boolean'
    ? availability.hf
    : (model?.hf && typeof model.hf === 'object');
  const testedState = normalizeTestedState(tested.result || status.tested);
  const testedAt = typeof tested.lastVerifiedAt === 'string' && tested.lastVerifiedAt.trim()
    ? tested.lastVerifiedAt.trim()
    : null;

  return {
    hosted,
    demo,
    tested: testedState,
    testedAt,
    catalogCount: 1,
    verifiedCount: testedState === 'verified' ? 1 : 0,
    failedCount: testedState === 'failed' ? 1 : 0,
  };
}

function mergeLifecycleAggregate(left, right) {
  const tested = left.tested === 'failed' || right.tested === 'failed'
    ? 'failed'
    : (left.tested === 'verified' || right.tested === 'verified' ? 'verified' : 'unknown');
  const testedAt = [left.testedAt, right.testedAt]
    .filter((value) => typeof value === 'string' && value.length > 0)
    .sort((a, b) => b.localeCompare(a))[0] || null;
  const demo = left.demo === 'curated' || right.demo === 'curated'
    ? 'curated'
    : (left.demo === 'local' || right.demo === 'local' ? 'local' : 'none');
  return {
    hosted: left.hosted || right.hosted,
    demo,
    tested,
    testedAt,
    catalogCount: left.catalogCount + right.catalogCount,
    verifiedCount: left.verifiedCount + right.verifiedCount,
    failedCount: left.failedCount + right.failedCount,
  };
}

export function resolveRowStatus(row) {
  if (row.conversionCount === 0) return 'missing-conversion';
  if (row.runtimeStatus === 'blocked') return 'blocked-runtime';
  if (row.catalogCount > 0) {
    if (row.lifecycleTested === 'verified') return 'verified';
    if (row.lifecycleTested === 'failed') return 'verification-failed';
    return 'verification-pending';
  }
  return 'conversion-ready';
}

function compareCatalogModels(left, right) {
  const leftOrder = Number.isFinite(left?.sortOrder) ? left.sortOrder : Number.POSITIVE_INFINITY;
  const rightOrder = Number.isFinite(right?.sortOrder) ? right.sortOrder : Number.POSITIVE_INFINITY;
  if (leftOrder !== rightOrder) {
    return leftOrder - rightOrder;
  }
  return normalizeModelId(left?.modelId).localeCompare(normalizeModelId(right?.modelId));
}

function buildCatalogModelStatusEntry(model) {
  const lifecycle = resolveCatalogLifecycle(model);
  const typeCluster = resolveModelTypeCluster(model.classification);
  const status = model?.lifecycle && typeof model.lifecycle === 'object' && model.lifecycle.status && typeof model.lifecycle.status === 'object'
    ? model.lifecycle.status
    : {};
  const tested = model?.lifecycle && typeof model.lifecycle === 'object' && model.lifecycle.tested && typeof model.lifecycle.tested === 'object'
    ? model.lifecycle.tested
    : {};
  return {
    modelId: normalizeModelId(model?.modelId),
    family: normalizeFamily(model?.family),
    typeCluster: typeCluster.label,
    modes: summarizeModes(model),
    runtimeStatus: normalizeText(status.runtime) || 'unknown',
    tested: lifecycle.tested,
    testedAt: lifecycle.testedAt,
    surface: Array.isArray(tested.surface)
      ? tested.surface.join(', ')
      : (typeof tested.surface === 'string' && tested.surface.trim() ? tested.surface.trim() : null),
    notes: typeof tested.notes === 'string' && tested.notes.trim() ? tested.notes.trim() : null,
  };
}

function buildFamilyCoverageEntry(row) {
  const notes = [];
  if (row.runtimeStatus === 'blocked') {
    notes.push('runtime path is fail-closed');
  } else {
    notes.push('conversion configs exist, but there is no cataloged model entry yet');
  }
  return {
    entry: row.family,
    type: 'model family',
    status: row.status,
    notes: notes.join('; '),
  };
}

export function buildCurrentInferenceStatusBuckets({ catalogModels, quickStartModelIds, rows }) {
  const verified = [];
  const loadsButUnverified = [];
  const knownFailing = [];
  const everythingElseCatalog = [];
  const sortedCatalogModels = Array.isArray(catalogModels)
    ? catalogModels.slice().sort(compareCatalogModels)
    : [];
  const catalogModelIds = new Set();

  for (const model of sortedCatalogModels) {
    const entry = buildCatalogModelStatusEntry(model);
    catalogModelIds.add(entry.modelId);
    if (entry.tested === 'verified') {
      verified.push(entry);
      continue;
    }
    if (entry.tested === 'failed') {
      knownFailing.push(entry);
      continue;
    }
    if (entry.runtimeStatus === 'active') {
      loadsButUnverified.push(entry);
      continue;
    }
    everythingElseCatalog.push({
      entry: entry.modelId,
      type: 'catalog model',
      status: entry.runtimeStatus || 'unknown',
      notes: entry.notes || 'Cataloged model without a verified or failing inference lifecycle result.',
    });
  }

  const quickstartOnly = Array.isArray(quickStartModelIds)
    ? quickStartModelIds
      .filter((modelId) => typeof modelId === 'string' && modelId.trim() && !catalogModelIds.has(modelId.trim()))
      .sort((left, right) => left.localeCompare(right))
      .map((modelId) => ({
        modelId: modelId.trim(),
        source: 'quickstart registry',
        notes: 'Downloadable through the quickstart path, but not yet represented in models/catalog.json.',
      }))
    : [];

  const everythingElseFamilies = Array.isArray(rows)
    ? rows
      .filter((row) => row.catalogCount === 0)
      .sort((left, right) => compareFamilies(left.family, right.family))
      .map((row) => buildFamilyCoverageEntry(row))
    : [];

  return {
    verified,
    loadsButUnverified,
    knownFailing,
    quickstartOnly,
    everythingElse: [...everythingElseCatalog, ...everythingElseFamilies],
  };
}

function summarizeList(values, maxItems = 3) {
  if (values.length === 0) return '0';
  if (values.length <= maxItems) {
    return `${values.length} (${values.join(', ')})`;
  }
  const visible = values.slice(0, maxItems).join(', ');
  return `${values.length} (${visible}, +${values.length - maxItems} more)`;
}

function formatCell(value) {
  if (value === null || value === undefined || value === '') return '-';
  return String(value).replace(/\|/g, '\\|');
}

function pushTable(lines, headers, rows) {
  lines.push(`| ${headers.join(' | ')} |`);
  lines.push(`| ${headers.map(() => '---').join(' | ')} |`);
  for (const row of rows) {
    lines.push(`| ${row.map((value) => formatCell(value)).join(' | ')} |`);
  }
  lines.push('');
}

function isVerifiedCatalogModel(model) {
  return normalizeText(model?.lifecycle?.status?.runtime) === 'active'
    && normalizeText(model?.lifecycle?.status?.tested) === 'verified'
    && normalizeText(model?.lifecycle?.tested?.result) === 'pass';
}

function summarizeTypeInterface(models) {
  const inputs = new Set();
  const outputs = new Set();
  for (const model of models) {
    for (const input of model?.classification?.inputs || []) inputs.add(input);
    for (const output of model?.classification?.outputs || []) outputs.add(output);
  }
  return `${[...inputs].sort().join(' + ')} → ${[...outputs].sort().join(' + ')}`;
}

function summarizeModelIds(models, limit = 3) {
  const ids = models.map((model) => normalizeModelId(model?.modelId));
  if (ids.length <= limit) return ids.join('<br>');
  return `${ids.slice(0, limit).join('<br>')}<br>+${ids.length - limit} more`;
}

function renderModelTypeClusters(lines, typeClusters) {
  lines.push('## Model Type Clusters');
  lines.push('');
  lines.push('These clusters describe each RDRR artifact by domain, task, architecture role, and input/output contract. They are independent of model family, runtime `modelType`, and artifact-size tier.');
  lines.push('');
  pushTable(lines,
    ['Type', 'Interface', 'Verified lanes', 'Other cataloged lanes'],
    typeClusters.map((cluster) => {
      const verified = cluster.models.filter(isVerifiedCatalogModel);
      const other = cluster.models.filter((model) => !isVerifiedCatalogModel(model));
      return [
        cluster.label,
        summarizeTypeInterface(cluster.models),
        verified.length > 0 ? verified.map((model) => model.modelId).join('<br>') : 'none',
        other.length > 0 ? other.map((model) => model.modelId).join('<br>') : 'none',
      ];
    }));
}

function renderReadmeModelTypeBlock(typeClusters) {
  const lines = [
    README_MODEL_TYPES_START,
    '',
    '## Supported RDRR model types',
    '',
    'Doppler classifies artifacts by what they consume and produce. This is',
    'separate from lineage (`family`), runtime implementation (`modelType`), and',
    'artifact-size tier.',
    '',
    '| Type | Input → output | Runtime-verified / cataloged | Representative lanes |',
    '| --- | --- | --- | --- |',
  ];
  for (const cluster of typeClusters) {
    const verified = cluster.models.filter(isVerifiedCatalogModel);
    lines.push(`| ${cluster.label} | ${summarizeTypeInterface(cluster.models)} | ${verified.length} / ${cluster.models.length} | ${summarizeModelIds(cluster.models)} |`);
  }
  lines.push('');
  lines.push('The [full model-support matrix](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md)');
  lines.push('lists every lane and its lifecycle evidence. Classification says what an');
  lines.push('artifact is shaped to do; only lifecycle receipts establish what is');
  lines.push('verified, and a runtime pass does not by itself qualify every declared input');
  lines.push('modality.');
  lines.push('');
  lines.push(README_MODEL_TYPES_END);
  return lines.join('\n');
}

function replaceReadmeModelTypeBlock(readme, block) {
  const startIndex = readme.indexOf(README_MODEL_TYPES_START);
  const endIndex = readme.indexOf(README_MODEL_TYPES_END);
  if (startIndex >= 0 || endIndex >= 0) {
    if (startIndex < 0 || endIndex < startIndex) {
      throw new Error('README model type cluster markers are malformed');
    }
    const afterEnd = endIndex + README_MODEL_TYPES_END.length;
    return `${readme.slice(0, startIndex).trimEnd()}\n\n${block}\n\n${readme.slice(afterEnd).trimStart()}`;
  }
  const evidenceHeading = '\n## Evidence\n';
  const insertionIndex = readme.indexOf(evidenceHeading);
  if (insertionIndex < 0) {
    throw new Error('README is missing the Evidence heading required for model type cluster insertion');
  }
  return `${readme.slice(0, insertionIndex).trimEnd()}\n\n${block}\n\n${readme.slice(insertionIndex).trimStart()}`;
}

function summarizeGemma4Evidence(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return 'none';
  }
  return entries
    .map((entry) => `${entry.modelId} (${entry.surface}, ${entry.status})`)
    .join('<br>');
}

function summarizeGemma4Blockers(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return 'none';
  }
  return entries
    .map((entry) => `${entry.code} (${entry.surface}, ${entry.state})`)
    .join('<br>');
}

function summarizeGemma4SourcePackages(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return 'none';
  }
  return entries
    .map((entry) => `${entry.id} (${entry.status})`)
    .join('<br>');
}

function renderCurrentInferenceStatus(lines, buckets) {
  lines.push('## Current Inference Status');
  lines.push('');
  lines.push('This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.');
  lines.push('');

  lines.push('### 1. Verified');
  lines.push('');
  if (buckets.verified.length === 0) {
    lines.push('None.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Model ID', 'Type', 'Family', 'Modes', 'Last verified', 'Surface', 'Notes'],
      buckets.verified.map((entry) => [
        entry.modelId,
        entry.typeCluster,
        entry.family,
        entry.modes,
        entry.testedAt || null,
        entry.surface || null,
        entry.notes || null,
      ]));
  }

  lines.push('### 2. Loads But Unverified');
  lines.push('');
  if (buckets.loadsButUnverified.length === 0) {
    lines.push('None right now.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Model ID', 'Type', 'Family', 'Modes', 'Runtime', 'Notes'],
      buckets.loadsButUnverified.map((entry) => [
        entry.modelId,
        entry.typeCluster,
        entry.family,
        entry.modes,
        entry.runtimeStatus,
        entry.notes || 'Cataloged model without a passing or failing verification result yet.',
      ]));
  }

  lines.push('### 3. Known Failing');
  lines.push('');
  if (buckets.knownFailing.length === 0) {
    lines.push('None right now.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Model ID', 'Type', 'Family', 'Modes', 'Last checked', 'Surface', 'Notes'],
      buckets.knownFailing.map((entry) => [
        entry.modelId,
        entry.typeCluster,
        entry.family,
        entry.modes,
        entry.testedAt || null,
        entry.surface || null,
        entry.notes || null,
      ]));
  }

  lines.push('### 4. Quickstart-Supported Only');
  lines.push('');
  if (buckets.quickstartOnly.length === 0) {
    lines.push('None right now.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Model ID', 'Source', 'Notes'],
      buckets.quickstartOnly.map((entry) => [
        entry.modelId,
        entry.source,
        entry.notes,
      ]));
  }

  lines.push('### 5. Everything Else');
  lines.push('');
  if (buckets.everythingElse.length === 0) {
    lines.push('None.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Entry', 'Type', 'Status', 'Notes'],
      buckets.everythingElse.map((entry) => [
        entry.entry,
        entry.type,
        entry.status,
        entry.notes,
      ]));
  }
}

function renderGemma4TargetCoverage(lines, targetMatrix) {
  if (!targetMatrix || !Array.isArray(targetMatrix.targets)) {
    return;
  }
  lines.push('## Gemma 4 Target Coverage');
  lines.push('');
  lines.push('Generated from `models/gemma4-targets.json`. This section tracks the latest official Gemma 4 target set separately from the catalog, so unsupported or unverified targets stay visible.');
  lines.push('');
  pushTable(lines,
    [
      'Target',
      'Doppler status',
      'Browser',
      'Electron',
      'Node',
      'Serve',
      'Official MTP',
      'Doppler MTP',
      'Runtime receipts',
      'Benchmark receipts',
      'Serve receipts',
      'Preflight receipts',
      'Current lanes',
      'Source packages',
      'Missing',
      'Blockers',
    ],
    targetMatrix.targets.map((target) => [
      target.officialName || target.targetId,
      target.dopplerStatus,
      target.surfaceStatus?.browser,
      target.surfaceStatus?.electron,
      target.surfaceStatus?.node,
      target.serveStatus,
      target.officialMtp === true ? 'yes' : 'no',
      target.mtpStatus,
      summarizeGemma4Evidence(target.evidence?.runtimeReceipts),
      summarizeGemma4Evidence(target.evidence?.benchmarkReceipts),
      summarizeGemma4Evidence(target.evidence?.serveReceipts),
      summarizeGemma4Evidence(target.evidence?.preflightReceipts),
      target.currentLanes.length === 0
        ? 'none'
        : target.currentLanes.map((lane) => `${lane.modelId} (${lane.claimStatus})`).join('<br>'),
      summarizeGemma4SourcePackages(target.sourcePackages),
      Array.isArray(target.missing) && target.missing.length > 0
        ? target.missing.join('<br>')
        : null,
      summarizeGemma4Blockers(target.blockers),
    ]));
}

function renderMatrix(rows, metadata, buckets, gemma4TargetMatrix, typeClusters) {
  const lines = [];
  lines.push('# Model Support Matrix');
  lines.push('');
  lines.push('Auto-generated from conversion configs (`src/config/conversion/**`), `models/catalog.json`, `models/model-type-taxonomy.json`, and `models/gemma4-targets.json`.');
  lines.push('Run `npm run support:matrix:sync` after editing the catalog, taxonomy, Gemma 4 targets, or conversion configs.');
  lines.push('');
  lines.push(`Updated at: ${metadata.generatedAt}`);
  lines.push('');
  renderModelTypeClusters(lines, typeClusters);
  renderCurrentInferenceStatus(lines, buckets);
  renderGemma4TargetCoverage(lines, gemma4TargetMatrix);
  lines.push('## Family Coverage Matrix');
  lines.push('');
  lines.push('| Family | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const row of rows) {
    const notes = [];
    if (row.runtimeStatus === 'blocked') {
      notes.push('fail-closed runtime path');
    }
    if (row.catalogCount === 0) {
      notes.push('not in local catalog');
    }
    if (row.lifecycleTested === 'unknown') {
      notes.push('not verified in catalog lifecycle');
    }
    if (row.catalogCount > 0 && row.conversionCount > row.catalogCount) {
      notes.push(`catalog verification applies only to cataloged models (${row.catalogCount}/${row.conversionCount} conversion configs cataloged)`);
    }
    if (row.catalogCount > 0 && row.lifecycleVerifiedCount > 0 && row.lifecycleVerifiedCount < row.catalogCount) {
      notes.push(`partial verification (${row.lifecycleVerifiedCount}/${row.catalogCount} catalog models verified)`);
    }
    if (row.catalogCount > 0 && row.lifecycleFailedCount > 0 && row.lifecycleFailedCount < row.catalogCount) {
      notes.push(`mixed verification state (${row.lifecycleFailedCount}/${row.catalogCount} catalog models failing)`);
    }
    const noteText = notes.length > 0 ? notes.join('; ') : '-';
    let testedLabel = row.lifecycleTested;
    if (row.lifecycleTested === 'verified' && row.lifecycleVerifiedCount > 0 && row.lifecycleVerifiedCount < row.catalogCount) {
      testedLabel = `partially verified (${row.lifecycleVerifiedCount}/${row.catalogCount})`;
    } else if (row.lifecycleTested === 'verified' && row.lifecycleTestedAt) {
      testedLabel = `verified (${row.lifecycleTestedAt})`;
    } else if (row.lifecycleTested === 'failed' && row.lifecycleFailedCount > 0 && row.lifecycleFailedCount < row.catalogCount) {
      testedLabel = `partially failing (${row.lifecycleFailedCount}/${row.catalogCount})`;
    }
    lines.push(
      `| ${row.family} | ${row.runtimeModelType} | ${row.runtimeStatus} | ` +
      `${summarizeList(row.conversionFiles)} | ${summarizeList(row.catalogModels)} | ` +
      `${row.lifecycleHosted ? 'yes' : 'no'} | ${row.lifecycleDemo} | ${testedLabel} | ${row.status} | ${noteText} |`
    );
  }
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Families tracked: ${metadata.familyCount}`);
  lines.push(`- Families with conversion configs: ${metadata.familiesWithConversion}`);
  lines.push(`- Families present in catalog: ${metadata.familiesInCatalog}`);
  lines.push(`- Verified families (active runtime + conversion + catalog + passing verification): ${metadata.verifiedReadyCount}`);
  lines.push(`- Cataloged families pending verification: ${metadata.verificationPendingCount}`);
  lines.push(`- Families with HF-hosted catalog entries: ${metadata.hostedCount}`);
  lines.push(`- Families with verified catalog lifecycle: ${metadata.verifiedCount}`);
  lines.push(`- Families with failed catalog verification: ${metadata.failedVerificationCount}`);
  lines.push(`- Blocked runtime families: ${metadata.blockedCount}`);
  lines.push(`- Catalog entries: ${metadata.catalogCount}`);
  return `${lines.join('\n')}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const conversionFiles = await collectJsonFiles(CONVERSION_CONFIG_DIR);
  const familyIds = [];
  const familySet = new Set();
  const conversionByFamily = new Map();
  const runtimeModelTypeByFamily = new Map();
  for (const filePath of conversionFiles) {
    const payload = await readJson(filePath);
    const family = inferFamilyFromConversionConfig(payload, filePath);
    if (!familySet.has(family)) {
      familySet.add(family);
      familyIds.push(family);
      conversionByFamily.set(family, []);
    }
    conversionByFamily.get(family).push(relativePath(filePath));
    if (!runtimeModelTypeByFamily.has(family)) {
      runtimeModelTypeByFamily.set(family, inferRuntimeModelTypeFromConversionConfig(payload));
    }
  }
  familyIds.sort(compareFamilies);
  for (const family of familyIds) {
    conversionByFamily.get(family).sort((left, right) => left.localeCompare(right));
  }

  const catalogPayload = await readJson(CATALOG_PATH);
  const catalogInputErrors = validateCatalogMatrixInputs(catalogPayload);
  if (catalogInputErrors.length > 0) {
    throw new Error(`Catalog lifecycle metadata is invalid:\n${catalogInputErrors.join('\n')}`);
  }
  const catalogModels = Array.isArray(catalogPayload?.models) ? catalogPayload.models : [];
  const typeClusters = buildModelTypeClusters(catalogModels);
  const quickstartRegistry = await readJson(QUICKSTART_REGISTRY_PATH);
  const quickstartModels = Array.isArray(quickstartRegistry?.models) ? quickstartRegistry.models : [];
  const gemma4TargetMatrix = await readJson(GEMMA4_TARGETS_PATH);
  const gemma4TargetErrors = validateGemma4TargetMatrixInputs(gemma4TargetMatrix, catalogModels, quickstartModels);
  if (gemma4TargetErrors.length > 0) {
    throw new Error(`Gemma 4 target matrix is invalid:\n${gemma4TargetErrors.join('\n')}`);
  }
  const gemma4EvidenceErrors = await validateGemma4EvidenceFiles(gemma4TargetMatrix);
  if (gemma4EvidenceErrors.length > 0) {
    throw new Error(`Gemma 4 target matrix evidence is invalid:\n${gemma4EvidenceErrors.join('\n')}`);
  }
  const catalogByFamily = new Map(familyIds.map((family) => [family, []]));
  const lifecycleByFamily = new Map(familyIds.map((family) => [family, createEmptyLifecycleAggregate()]));
  const unmappedCatalogModels = [];

  for (const model of catalogModels) {
    const family = normalizeText(model?.family);
    if (!family) {
      const modelId = typeof model?.modelId === 'string' && model.modelId.trim() ? model.modelId.trim() : 'unknown-model';
      unmappedCatalogModels.push(modelId);
      continue;
    }
    if (!familySet.has(family)) {
      const modelId = typeof model?.modelId === 'string' && model.modelId.trim() ? model.modelId.trim() : 'unknown-model';
      throw new Error(
        `Catalog model "${modelId}" has family "${family}" which is not represented by any conversion config. ` +
        `Valid families: ${familyIds.join(', ')}`
      );
    }
    const modelId = typeof model?.modelId === 'string' && model.modelId.trim() ? model.modelId.trim() : 'unknown-model';
    catalogByFamily.get(family).push(modelId);
    lifecycleByFamily.set(
      family,
      mergeLifecycleAggregate(
        lifecycleByFamily.get(family) || createEmptyLifecycleAggregate(),
        resolveCatalogLifecycle(model)
      )
    );
  }
  for (const family of familyIds) {
    catalogByFamily.get(family).sort((left, right) => left.localeCompare(right));
  }

  if (unmappedCatalogModels.length > 0) {
    throw new Error(`Catalog entries missing family mapping: ${unmappedCatalogModels.join(', ')}`);
  }

  const rows = familyIds.map((family) => {
    const runtimeModelType = runtimeModelTypeByFamily.get(family) || 'transformer';
    const runtimeStatus = resolveRuntimeStatus(runtimeModelType);
    const conversionFilesForFamily = conversionByFamily.get(family) || [];
    const catalogModelsForFamily = catalogByFamily.get(family) || [];
    const lifecycleForFamily = lifecycleByFamily.get(family) || createEmptyLifecycleAggregate();
    const row = {
      family,
      runtimeModelType,
      runtimeStatus,
      conversionFiles: conversionFilesForFamily,
      conversionCount: conversionFilesForFamily.length,
      catalogModels: catalogModelsForFamily,
      catalogCount: catalogModelsForFamily.length,
      lifecycleHosted: lifecycleForFamily.hosted === true,
      lifecycleDemo: lifecycleForFamily.demo,
      lifecycleTested: lifecycleForFamily.tested,
      lifecycleTestedAt: lifecycleForFamily.testedAt,
      lifecycleCatalogCount: lifecycleForFamily.catalogCount,
      lifecycleVerifiedCount: lifecycleForFamily.verifiedCount,
      lifecycleFailedCount: lifecycleForFamily.failedCount,
    };
    return {
      ...row,
      status: resolveRowStatus(row),
    };
  });

  const quickStartModelIds = Array.isArray(quickstartRegistry?.models)
    ? quickstartRegistry.models
      .map((entry) => normalizeText(entry?.modelId) ? String(entry.modelId).trim() : null)
      .filter((entry) => typeof entry === 'string' && entry.length > 0)
    : [];

  const buckets = buildCurrentInferenceStatusBuckets({
    catalogModels,
    quickStartModelIds,
    rows,
  });

  const metadata = {
    generatedAt: typeof catalogPayload?.updatedAt === 'string' && catalogPayload.updatedAt.trim()
      ? catalogPayload.updatedAt.trim()
      : 'unknown',
    familyCount: rows.length,
    familiesWithConversion: rows.filter((row) => row.conversionCount > 0).length,
    familiesInCatalog: rows.filter((row) => row.catalogCount > 0).length,
    verifiedReadyCount: rows.filter((row) => row.status === 'verified').length,
    verificationPendingCount: rows.filter((row) => row.status === 'verification-pending').length,
    hostedCount: rows.filter((row) => row.lifecycleHosted).length,
    verifiedCount: rows.filter((row) => row.lifecycleTested === 'verified').length,
    failedVerificationCount: rows.filter((row) => row.lifecycleTested === 'failed').length,
    blockedCount: rows.filter((row) => row.runtimeStatus === 'blocked').length,
    catalogCount: catalogModels.length,
  };
  const nextContent = renderMatrix(rows, metadata, buckets, gemma4TargetMatrix, typeClusters);
  const readme = await fs.readFile(README_PATH, 'utf8');
  const nextReadme = replaceReadmeModelTypeBlock(readme, renderReadmeModelTypeBlock(typeClusters));

  if (args.check) {
    let currentContent;
    try {
      currentContent = await fs.readFile(args.outputPath, 'utf8');
    } catch (error) {
      if (error && error.code === 'ENOENT') {
        throw new Error(`Missing ${relativePath(args.outputPath)}. Run npm run support:matrix:sync`);
      }
      throw error;
    }
    if (currentContent !== nextContent) {
      throw new Error(
        `Model support matrix is out of date at ${relativePath(args.outputPath)}. ` +
        'Run npm run support:matrix:sync'
      );
    }
    if (readme !== nextReadme) {
      throw new Error('README model type clusters are out of date. Run npm run support:matrix:sync');
    }
    console.log(`[support-matrix] up to date (${rows.length} families)`);
    return;
  }

  await fs.writeFile(args.outputPath, nextContent, 'utf8');
  await fs.writeFile(README_PATH, nextReadme, 'utf8');
  console.log(`[support-matrix] wrote ${relativePath(args.outputPath)}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
