#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'release-claim-policy.json');
const DEFAULT_CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');
const DEFAULT_QUICKSTART_REGISTRY_PATH = path.join(REPO_ROOT, 'src', 'client', 'doppler-registry.json');
const DEFAULT_SUBSYSTEMS_PATH = path.join(REPO_ROOT, 'src', 'config', 'support-tiers', 'subsystems.json');
const CLAIM_MODES = new Set(['text', 'embedding', 'rerank', 'translate']);
const CLAIM_SURFACES = new Set(['browser', 'node', 'bun', 'serve', 'electron']);
const EVIDENCE_KINDS = new Set([
  'browser-node-webgpu-smoke',
  'browser-webgpu-smoke',
  'local-node-webgpu-verify',
  'manual-report',
  'manifest-variant-identity',
  'runtime-verify',
]);
const PERFORMANCE_EVIDENCE_KINDS = new Set([
  'embedding-runtime-report',
  'rerank-runtime-report',
  'runtime-report',
]);

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeLower(value) {
  return normalizeText(value).toLowerCase();
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function uniqueNormalizedList(values) {
  if (!Array.isArray(values)) return [];
  const out = [];
  const seen = new Set();
  for (const value of values) {
    const normalized = normalizeLower(value);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function parseArgs(argv) {
  const args = {
    policyPath: DEFAULT_POLICY_PATH,
    catalogPath: DEFAULT_CATALOG_PATH,
    quickstartRegistryPath: DEFAULT_QUICKSTART_REGISTRY_PATH,
    subsystemsPath: DEFAULT_SUBSYSTEMS_PATH,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    const nextValue = () => {
      const candidate = argv[i + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${token}`);
      }
      i += 1;
      return path.resolve(REPO_ROOT, String(candidate).trim());
    };
    if (token === '--policy') {
      args.policyPath = nextValue();
      continue;
    }
    if (token === '--catalog') {
      args.catalogPath = nextValue();
      continue;
    }
    if (token === '--quickstart-registry') {
      args.quickstartRegistryPath = nextValue();
      continue;
    }
    if (token === '--subsystems') {
      args.subsystemsPath = nextValue();
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return args;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function isRepoRelativeJsonPath(value) {
  const candidate = normalizeText(value);
  return Boolean(
    candidate
    && !path.isAbsolute(candidate)
    && !candidate.includes('\\')
    && !candidate.split('/').includes('..')
    && candidate.endsWith('.json')
  );
}

function valueAtPath(payload, metricPath) {
  const parts = normalizeText(metricPath).split('.').filter(Boolean);
  let current = payload;
  for (const part of parts) {
    if (!isPlainObject(current) || !Object.prototype.hasOwnProperty.call(current, part)) {
      return undefined;
    }
    current = current[part];
  }
  return current;
}

function resolvePayloadModelId(payload) {
  return normalizeText(payload?.modelId || payload?.dopplerModelId || payload?.model?.modelId);
}

function hasAdapterEvidence(payload) {
  return isPlainObject(payload?.deviceInfo?.adapterInfo)
    || isPlainObject(payload?.captureProfile?.adapter?.deviceInfo?.adapterInfo)
    || isPlainObject(payload?.runtime?.adapterInfo);
}

function hasExecutionContractEvidence(payload) {
  if (payload?.metrics?.executionContractArtifact?.ok === true) return true;
  if (payload?.executionContractArtifact?.ok === true) return true;
  if (payload?.schema === 'doppler.program-bundle/v1') {
    return normalizeText(payload?.sources?.manifest?.hash)
      && normalizeText(payload?.sources?.executionGraph?.hash)
      && normalizeText(payload?.execution?.graphHash)
      && Array.isArray(payload?.execution?.steps)
      && payload.execution.steps.length > 0;
  }
  return false;
}

function hasTextOutputEvidence(payload) {
  if (normalizeText(payload?.output)) return true;
  if (normalizeText(payload?.metrics?.generatedText)) return true;
  if (Number.isFinite(payload?.referenceTranscript?.output?.tokensGenerated)) {
    return payload.referenceTranscript.output.tokensGenerated > 0;
  }
  return false;
}

function hasEmbeddingEvidence(payload) {
  const metrics = isPlainObject(payload?.metrics) ? payload.metrics : {};
  const semanticEmbeddingPassed = Number.isFinite(metrics.embeddingDim)
    && metrics.embeddingDim > 0
    && Number.isFinite(metrics.finiteRatio)
    && metrics.finiteRatio === 1
    && metrics.semanticPassed === true;
  if (semanticEmbeddingPassed) return true;
  if (payload?.schema !== 'doppler.sequenceModelQualification.v1' || payload?.passed !== true) {
    return false;
  }
  const checks = Array.isArray(payload?.result?.checks) ? payload.result.checks : [];
  const passedCheckIds = new Set(
    checks
      .filter((check) => check?.passed === true)
      .map((check) => normalizeText(check?.id))
      .filter(Boolean)
  );
  const required = [
    'model.identity',
    'sequence.contract',
    'tokenizer.ids',
    'pooledEmbedding.finite',
    'pooledEmbedding.parity',
  ];
  if (payload?.model?.sequence?.tokenEmbeddings === true) {
    required.push('tokenEmbeddings.finite', 'tokenEmbeddings.parity');
  }
  return required.every((id) => passedCheckIds.has(id));
}

function hasRerankEvidence(payload) {
  const metrics = isPlainObject(payload?.metrics) ? payload.metrics : {};
  return metrics.semanticPassed === true
    && Number.isFinite(metrics.semanticPairAcc)
    && Number.isFinite(metrics.topDocumentIndex)
    && Number.isFinite(metrics.rerankMs);
}

function expectedClaimMode(model) {
  const modes = uniqueNormalizedList(model?.modes);
  if (modes.includes('embedding')) return 'embedding';
  if (modes.includes('rerank')) return 'rerank';
  if (modes.includes('translate') && !modes.includes('text')) return 'translate';
  if (modes.includes('text')) return 'text';
  return null;
}

function validateReleaseClaimPolicyShape(policy) {
  const errors = [];
  if (!isPlainObject(policy)) {
    return ['release-claim policy must be an object'];
  }
  if (policy.schemaVersion !== 1) {
    errors.push('release-claim policy schemaVersion must be 1');
  }
  if (!normalizeText(policy.updatedAt)) {
    errors.push('release-claim policy updatedAt is required');
  }
  if (!Array.isArray(policy.claims)) {
    errors.push('release-claim policy claims must be an array');
    return errors;
  }
  const seen = new Set();
  for (const claim of policy.claims) {
    const modelId = normalizeText(claim?.modelId);
    const mode = normalizeLower(claim?.mode);
    if (!modelId) {
      errors.push('release claim is missing modelId');
    }
    if (!CLAIM_MODES.has(mode)) {
      errors.push(`${modelId || 'unknown-model'}: release claim mode must be one of ${Array.from(CLAIM_MODES).join(', ')}`);
    }
    const key = `${modelId}:${mode}`;
    if (seen.has(key)) {
      errors.push(`${modelId}: duplicate release claim for mode ${mode}`);
    }
    seen.add(key);
    const surfaces = uniqueNormalizedList(claim?.surface);
    if (surfaces.length === 0) {
      errors.push(`${modelId || 'unknown-model'}: release claim surface must be a non-empty array`);
    }
    for (const surface of surfaces) {
      if (!CLAIM_SURFACES.has(surface)) {
        errors.push(`${modelId || 'unknown-model'}: unsupported release claim surface ${surface}`);
      }
    }
    if (!normalizeText(claim?.verificationSource)) {
      errors.push(`${modelId || 'unknown-model'}: verificationSource is required`);
    }
    if (!normalizeText(claim?.lastVerifiedAt)) {
      errors.push(`${modelId || 'unknown-model'}: lastVerifiedAt is required`);
    }
    if (!normalizeText(claim?.artifactFormat)) {
      errors.push(`${modelId || 'unknown-model'}: artifactFormat is required`);
    }
    if (!EVIDENCE_KINDS.has(normalizeLower(claim?.evidence?.kind))) {
      errors.push(`${modelId || 'unknown-model'}: evidence.kind is invalid`);
    }
    if (!isRepoRelativeJsonPath(claim?.evidence?.reportPath)) {
      errors.push(`${modelId || 'unknown-model'}: evidence.reportPath must be a repo-relative JSON path`);
    }
    if (!PERFORMANCE_EVIDENCE_KINDS.has(normalizeLower(claim?.performanceEvidence?.kind))) {
      errors.push(`${modelId || 'unknown-model'}: performanceEvidence.kind is invalid`);
    }
    if (!isRepoRelativeJsonPath(claim?.performanceEvidence?.reportPath)) {
      errors.push(`${modelId || 'unknown-model'}: performanceEvidence.reportPath must be a repo-relative JSON path`);
    }
    if (!normalizeText(claim?.performanceEvidence?.metricPath)) {
      errors.push(`${modelId || 'unknown-model'}: performanceEvidence.metricPath is required`);
    }
    if (!Number.isFinite(claim?.performanceEvidence?.minValue)) {
      errors.push(`${modelId || 'unknown-model'}: performanceEvidence.minValue must be numeric`);
    }
    if (!normalizeText(claim?.performanceEvidence?.unit)) {
      errors.push(`${modelId || 'unknown-model'}: performanceEvidence.unit is required`);
    }
  }
  return errors;
}

function validateClaimCatalogContract(policy, catalog, quickstartRegistry) {
  const errors = [];
  const models = Array.isArray(catalog?.models) ? catalog.models : [];
  const claims = Array.isArray(policy?.claims) ? policy.claims : [];
  const catalogById = new Map(models.map((model) => [normalizeText(model?.modelId), model]));
  const claimByModelId = new Map(claims.map((claim) => [normalizeText(claim?.modelId), claim]));
  const quickstartIds = new Set(
    (Array.isArray(quickstartRegistry?.models) ? quickstartRegistry.models : [])
      .map((model) => normalizeText(model?.modelId))
      .filter(Boolean)
  );

  for (const model of models) {
    const modelId = normalizeText(model?.modelId);
    if (!modelId) continue;
    const tested = model?.lifecycle?.tested;
    const isVerified = model?.lifecycle?.status?.tested === 'verified'
      && tested?.result === 'pass';
    if (!isVerified) continue;
    const expectedMode = expectedClaimMode(model);
    if (!expectedMode) {
      errors.push(`${modelId}: verified catalog model has no claimable text, embedding, or translate mode`);
      continue;
    }
    const claim = claimByModelId.get(modelId);
    if (!claim) {
      errors.push(`${modelId}: verified catalog model is missing from release-claim policy`);
      continue;
    }
    if (normalizeLower(claim.mode) !== expectedMode) {
      errors.push(`${modelId}: release claim mode ${claim.mode} does not match expected ${expectedMode}`);
    }
    const claimSurfaces = uniqueNormalizedList(claim.surface);
    const testedSurfaces = uniqueNormalizedList(tested.surface);
    if (JSON.stringify(claimSurfaces) !== JSON.stringify(testedSurfaces)) {
      errors.push(`${modelId}: release claim surfaces must match lifecycle.tested.surface`);
    }
  }

  for (const quickstartId of quickstartIds) {
    if (!claimByModelId.has(quickstartId)) {
      errors.push(`${quickstartId}: quickstart registry model is missing from release-claim policy`);
    }
  }

  for (const claim of claims) {
    const modelId = normalizeText(claim?.modelId);
    const model = catalogById.get(modelId);
    if (!model) {
      errors.push(`${modelId || 'unknown-model'}: release claim model is missing from models/catalog.json`);
      continue;
    }
    const tested = model?.lifecycle?.tested;
    if (model?.lifecycle?.status?.tested !== 'verified' || tested?.result !== 'pass') {
      errors.push(`${modelId}: release claim requires catalog lifecycle verified/pass`);
    }
    if (normalizeText(claim.verificationSource) !== normalizeText(tested?.source)) {
      errors.push(`${modelId}: release claim verificationSource must match lifecycle.tested.source`);
    }
    if (normalizeText(claim.lastVerifiedAt) !== normalizeText(tested?.lastVerifiedAt)) {
      errors.push(`${modelId}: release claim lastVerifiedAt must match lifecycle.tested.lastVerifiedAt`);
    }
    if (normalizeLower(claim.artifactFormat) !== normalizeLower(model?.artifact?.format)) {
      errors.push(`${modelId}: release claim artifactFormat must match catalog artifact.format`);
    }
    const modes = uniqueNormalizedList(model?.modes);
    const mode = normalizeLower(claim?.mode);
    if (mode === 'text' && !modes.includes('text')) {
      errors.push(`${modelId}: text release claim requires catalog text mode`);
    }
    if (mode === 'embedding' && !modes.includes('embedding')) {
      errors.push(`${modelId}: embedding release claim requires catalog embedding mode`);
    }
    if (mode === 'rerank' && !modes.includes('rerank')) {
      errors.push(`${modelId}: rerank release claim requires catalog rerank mode`);
    }
    if (mode === 'translate' && !modes.includes('translate')) {
      errors.push(`${modelId}: translate release claim requires catalog translate mode`);
    }
  }
  return errors;
}

async function validateClaimEvidenceFiles(policy) {
  const errors = [];
  const claims = Array.isArray(policy?.claims) ? policy.claims : [];
  for (const claim of claims) {
    const modelId = normalizeText(claim?.modelId) || 'unknown-model';
    const evidencePath = normalizeText(claim?.evidence?.reportPath);
    const performancePath = normalizeText(claim?.performanceEvidence?.reportPath);
    const paths = new Set([evidencePath, performancePath].filter(Boolean));
    const payloadByPath = new Map();
    for (const relative of paths) {
      if (!isRepoRelativeJsonPath(relative)) continue;
      try {
        payloadByPath.set(relative, await readJson(path.join(REPO_ROOT, relative)));
      } catch (error) {
        if (error?.code === 'ENOENT') {
          errors.push(`${modelId}: evidence file is missing: ${relative}`);
          continue;
        }
        throw error;
      }
    }
    const evidencePayload = payloadByPath.get(evidencePath);
    if (evidencePayload) {
      const payloadModelId = resolvePayloadModelId(evidencePayload);
      if (payloadModelId !== modelId) {
        errors.push(`${modelId}: evidence report modelId mismatch (${payloadModelId || 'missing'})`);
      }
      if (!hasAdapterEvidence(evidencePayload)) {
        errors.push(`${modelId}: evidence report requires adapter identity`);
      }
      const mode = normalizeLower(claim.mode);
      if ((mode === 'text' || mode === 'translate' || mode === 'rerank') && !hasExecutionContractEvidence(evidencePayload)) {
        errors.push(`${modelId}: ${mode} evidence requires execution contract evidence`);
      }
      if ((mode === 'text' || mode === 'translate') && !hasTextOutputEvidence(evidencePayload)) {
        errors.push(`${modelId}: ${mode} evidence requires generated output evidence`);
      }
      if (mode === 'embedding' && !hasEmbeddingEvidence(evidencePayload)) {
        errors.push(`${modelId}: embedding evidence requires finite semantic or sequence-parity evidence`);
      }
      if (mode === 'rerank' && !hasRerankEvidence(evidencePayload)) {
        errors.push(`${modelId}: rerank evidence requires semantic rerank evidence`);
      }
    }
    const performancePayload = payloadByPath.get(performancePath);
    if (performancePayload) {
      const metricValue = valueAtPath(performancePayload, claim?.performanceEvidence?.metricPath);
      const minValue = claim?.performanceEvidence?.minValue;
      if (!Number.isFinite(metricValue)) {
        errors.push(`${modelId}: performance metric ${claim.performanceEvidence.metricPath} is missing or non-numeric`);
      } else if (Number.isFinite(minValue) && metricValue <= minValue) {
        errors.push(`${modelId}: performance metric ${claim.performanceEvidence.metricPath} must be greater than ${minValue}`);
      }
    }
  }
  return errors;
}

function validateSubsystemPublicClaimBoundaries(subsystemsPayload) {
  const errors = [];
  const subsystems = Array.isArray(subsystemsPayload?.subsystems) ? subsystemsPayload.subsystems : [];
  for (const subsystem of subsystems) {
    const id = normalizeText(subsystem?.id) || 'unknown-subsystem';
    const tier = normalizeLower(subsystem?.tier);
    const visibility = normalizeLower(subsystem?.claimVisibility);
    if (!['tier1', 'experimental', 'internal-only'].includes(tier)) {
      errors.push(`${id}: invalid subsystem tier`);
    }
    if (tier === 'tier1' && visibility === 'none') {
      errors.push(`${id}: tier1 subsystem must have public claim visibility`);
    }
    if (tier !== 'tier1' && subsystem?.demoDefault === true) {
      errors.push(`${id}: only tier1 subsystems may be demoDefault`);
    }
    if (!Array.isArray(subsystem?.docs) || subsystem.docs.length === 0) {
      errors.push(`${id}: subsystem docs must be listed`);
    }
    if (!Array.isArray(subsystem?.entrypoints) || subsystem.entrypoints.length === 0) {
      errors.push(`${id}: subsystem entrypoints must be listed`);
    }
  }
  return errors;
}

export async function checkReleaseClaims({
  policyPath = DEFAULT_POLICY_PATH,
  catalogPath = DEFAULT_CATALOG_PATH,
  quickstartRegistryPath = DEFAULT_QUICKSTART_REGISTRY_PATH,
  subsystemsPath = DEFAULT_SUBSYSTEMS_PATH,
} = {}) {
  const [policy, catalog, quickstartRegistry, subsystems] = await Promise.all([
    readJson(policyPath),
    readJson(catalogPath),
    readJson(quickstartRegistryPath),
    readJson(subsystemsPath),
  ]);
  const errors = [
    ...validateReleaseClaimPolicyShape(policy),
    ...validateClaimCatalogContract(policy, catalog, quickstartRegistry),
    ...await validateClaimEvidenceFiles(policy),
    ...validateSubsystemPublicClaimBoundaries(subsystems),
  ];
  return {
    ok: errors.length === 0,
    errors,
    claimCount: Array.isArray(policy?.claims) ? policy.claims.length : 0,
  };
}

export {
  parseArgs,
  validateReleaseClaimPolicyShape,
  validateClaimCatalogContract,
  validateSubsystemPublicClaimBoundaries,
};

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const result = await checkReleaseClaims(args);
  if (!result.ok) {
    for (const error of result.errors) {
      console.error(`release-claims: ${error}`);
    }
    process.exitCode = 1;
    return;
  }
  console.log(`release-claims: ok (${result.claimCount} claims)`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
