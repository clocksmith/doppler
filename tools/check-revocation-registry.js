#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  findResolutionRevocation,
  validateRevocationRegistry,
} from '../src/config/revocation-policy.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_PATHS = Object.freeze({
  registry: 'src/config/revocation-registry.json',
  catalog: 'models/catalog.json',
  adapterCatalog: 'models/adapters/catalog.json',
  quickstart: 'src/client/doppler-registry.json',
  claimMatrix: 'benchmarks/vendors/local-inference-claim-matrix.json',
  releaseMatrix: 'benchmarks/vendors/release-matrix.json',
  productIntegrations: 'tools/policies/product-integration-qualification.json',
  providerConformance: 'tools/policies/provider-conformance.json',
  runtimeOwnership: 'benchmarks/vendors/runtime-ownership-decisions.json',
});
const CLAIMABLE_LANE_STATES = new Set(['active', 'promoted', 'qualified', 'product-supported']);
const HASH_PATTERN = /^(?:sha256:)?[0-9a-f]{64}$/i;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function artifactVariantId(value) {
  const normalized = normalizeText(value).toLowerCase();
  if (!HASH_PATTERN.test(normalized)) return null;
  return normalized.startsWith('sha256:') ? normalized : `sha256:${normalized}`;
}

function resolutionIdentity(value = {}) {
  return {
    logicalModelId: normalizeText(value.logicalModelId) || null,
    modelId: normalizeText(value.modelId) || null,
    sourceCheckpointId: normalizeText(value.sourceCheckpointId) || null,
    weightPackId: normalizeText(value.weightPackId) || null,
    manifestVariantId: normalizeText(value.manifestVariantId) || null,
    artifactVariantId: artifactVariantId(value.artifactVariantId),
    adapterId: normalizeText(value.adapterId) || null,
    adapterDigest: artifactVariantId(value.adapterDigest),
  };
}

function matchIdentity(identity, registry) {
  const match = findResolutionRevocation(resolutionIdentity(identity), registry);
  return match ? [match] : [];
}

function matchIdentityAndAliases(identity, aliases, registry) {
  const matches = [...matchIdentity(identity, registry)];
  for (const alias of Array.isArray(aliases) ? aliases : []) {
    matches.push(...matchIdentity({ ...identity, logicalModelId: alias }, registry));
  }
  return matches;
}

function addConflict(errors, label, matches, problem) {
  if (matches.length === 0) return;
  const ids = [...new Set(matches.map((match) => match.revocation.id))].join(', ');
  errors.push(`${label} conflicts with revocation ${ids}: ${problem}`);
}

function validateCatalog(catalog, registry, errors) {
  for (const entry of Array.isArray(catalog?.models) ? catalog.models : []) {
    const modelId = normalizeText(entry?.modelId) || '<missing-model>';
    const matches = matchIdentityAndAliases({
      logicalModelId: modelId,
      modelId,
      sourceCheckpointId: entry?.sourceCheckpointId,
      weightPackId: entry?.weightPackId,
      manifestVariantId: entry?.manifestVariantId,
    }, entry?.aliases, registry);
    if (matches.length === 0) continue;
    if (entry?.quickstart === true) addConflict(errors, `catalog ${modelId}`, matches, 'quickstart must be false');
    if (entry?.demoVisible === true) addConflict(errors, `catalog ${modelId}`, matches, 'demoVisible must be false');
    if (entry?.lifecycle?.status?.runtime !== 'revoked') {
      addConflict(errors, `catalog ${modelId}`, matches, 'lifecycle.status.runtime must be "revoked"');
    }
  }
}

function validateAdapterCatalog(catalog, registry, errors) {
  for (const entry of Array.isArray(catalog?.artifacts) ? catalog.artifacts : []) {
    const adapterId = normalizeText(entry?.artifactId) || '<missing-adapter>';
    const matches = matchIdentity({
      adapterId,
      adapterDigest: entry?.weights?.sha256,
    }, registry);
    if (matches.length > 0 && entry?.lifecycle !== 'revoked') {
      addConflict(errors, `adapter catalog ${adapterId}`, matches, 'lifecycle must be "revoked"');
    }
  }
}

function validateQuickstart(quickstart, registry, errors) {
  for (const entry of Array.isArray(quickstart?.models) ? quickstart.models : []) {
    const modelId = normalizeText(entry?.modelId) || '<missing-model>';
    const matches = matchIdentityAndAliases({
      logicalModelId: modelId,
      modelId,
      sourceCheckpointId: entry?.sourceCheckpointId,
      weightPackId: entry?.weightPackId,
      manifestVariantId: entry?.manifestVariantId,
    }, entry?.aliases, registry);
    addConflict(errors, `quickstart ${modelId}`, matches, 'revoked entries must not resolve');
  }
}

function claimLaneIdentity(lane) {
  return {
    logicalModelId: lane?.model?.dopplerModelId,
    modelId: lane?.model?.dopplerModelId,
    sourceCheckpointId: lane?.model?.sourceCheckpointId,
    weightPackId: lane?.artifact?.weightPackId,
    manifestVariantId: lane?.artifact?.manifestVariantId,
    artifactVariantId: lane?.artifact?.manifestSha256,
  };
}

function validateClaimMatrix(matrix, registry, errors) {
  for (const lane of Array.isArray(matrix?.lanes) ? matrix.lanes : []) {
    if (!CLAIMABLE_LANE_STATES.has(lane?.status) && lane?.claimAllowed !== true) continue;
    addConflict(
      errors,
      `claim lane ${normalizeText(lane?.id) || '<missing-lane>'}`,
      matchIdentity(claimLaneIdentity(lane), registry),
      'claimable lane must be revoked or non-claimable'
    );
  }
}

function validateReleaseMatrix(matrix, registry, errors) {
  for (const lane of Array.isArray(matrix?.localClaimLanes) ? matrix.localClaimLanes : []) {
    if (lane?.claimReady !== true && !CLAIMABLE_LANE_STATES.has(lane?.status)) continue;
    addConflict(
      errors,
      `release lane ${normalizeText(lane?.laneId) || '<missing-lane>'}`,
      matchIdentity({
        logicalModelId: lane?.dopplerModelId,
        modelId: lane?.dopplerModelId,
      }, registry),
      'claimReady must be false and status must not be claimable'
    );
  }
}

function validateProductIntegrations(policy, registry, errors) {
  for (const integration of Array.isArray(policy?.integrations) ? policy.integrations : []) {
    if (integration?.claimAllowed !== true && integration?.lifecycle === 'revoked') continue;
    addConflict(
      errors,
      `product integration ${normalizeText(integration?.id) || '<missing-integration>'}`,
      matchIdentity({
        logicalModelId: integration?.logicalModelId,
        artifactVariantId: integration?.resolvedArtifactVariantId,
      }, registry),
      'lifecycle must be revoked and claimAllowed must be false'
    );
  }
}

function validateProviderConformance(policy, registry, errors) {
  for (const suite of Array.isArray(policy?.suites) ? policy.suites : []) {
    const suiteMatches = matchIdentity({
      logicalModelId: suite?.logicalModelId,
      manifestVariantId: suite?.manifestVariantId,
      artifactVariantId: suite?.resolvedArtifactVariantId,
    }, registry);
    if (suite?.claimAllowed === true) {
      addConflict(errors, `provider suite ${normalizeText(suite?.id) || '<missing-suite>'}`, suiteMatches, 'claimAllowed must be false');
    }
    for (const provider of Array.isArray(suite?.providers) ? suite.providers : []) {
      if (provider?.claimAllowed !== true) continue;
      addConflict(
        errors,
        `provider result ${normalizeText(suite?.id) || '<missing-suite>'}/${normalizeText(provider?.laneId) || '<missing-provider>'}`,
        matchIdentity({
          logicalModelId: provider?.logicalModelId,
          manifestVariantId: provider?.manifestVariantId,
          artifactVariantId: provider?.resolvedArtifactVariantId,
        }, registry),
        'claimAllowed must be false'
      );
    }
  }
}

function validateRuntimeOwnership(policy, registry, errors) {
  for (const decision of Array.isArray(policy?.decisions) ? policy.decisions : []) {
    if (decision?.claimAllowed !== true) continue;
    addConflict(
      errors,
      `runtime ownership ${normalizeText(decision?.id) || '<missing-decision>'}`,
      matchIdentity({
        logicalModelId: decision?.logicalModelId,
        manifestVariantId: decision?.manifestVariantId,
        artifactVariantId: decision?.resolvedArtifactVariantId,
      }, registry),
      'claimAllowed must be false'
    );
  }
}

async function validateEvidencePaths(registry, repoRoot, errors) {
  for (const revocation of registry.revocations) {
    for (const evidencePath of revocation.evidencePaths) {
      if (path.isAbsolute(evidencePath) || evidencePath.includes('\\') || evidencePath.split('/').includes('..')) {
        errors.push(`${revocation.id}: evidence path must be repo-relative: ${evidencePath}`);
        continue;
      }
      try {
        await fs.stat(path.join(repoRoot, evidencePath));
      } catch {
        errors.push(`${revocation.id}: evidence path does not exist: ${evidencePath}`);
      }
    }
  }
}

export async function validateRevocationPropagation(
  registryValue,
  surfaces,
  { repoRoot = REPO_ROOT, requireBundledTrust = false } = {}
) {
  const errors = [];
  let registry;
  try {
    registry = validateRevocationRegistry(registryValue);
  } catch (error) {
    return {
      ok: false,
      activeRevocations: 0,
      signatureVerification: null,
      errors: [error.message],
    };
  }
  if (requireBundledTrust && registry.trust.distribution !== 'bundled-package') {
    errors.push('the package revocation registry must use bundled-package trust');
  }
  await validateEvidencePaths(registry, repoRoot, errors);
  validateCatalog(surfaces.catalog, registry, errors);
  validateAdapterCatalog(surfaces.adapterCatalog, registry, errors);
  validateQuickstart(surfaces.quickstart, registry, errors);
  validateClaimMatrix(surfaces.claimMatrix, registry, errors);
  validateReleaseMatrix(surfaces.releaseMatrix, registry, errors);
  validateProductIntegrations(surfaces.productIntegrations, registry, errors);
  validateProviderConformance(surfaces.providerConformance, registry, errors);
  validateRuntimeOwnership(surfaces.runtimeOwnership, registry, errors);
  return {
    ok: errors.length === 0,
    activeRevocations: registry.revocations.length,
    signatureVerification: registry.trust.signatureVerification,
    errors,
  };
}

async function readJson(repoRoot, relativePath) {
  return JSON.parse(await fs.readFile(path.join(repoRoot, relativePath), 'utf8'));
}

export async function buildRevocationPropagationReport({ repoRoot = REPO_ROOT } = {}) {
  const values = await Promise.all(Object.values(DEFAULT_PATHS).map((filePath) => readJson(repoRoot, filePath)));
  const [registry, catalog, adapterCatalog, quickstart, claimMatrix, releaseMatrix, productIntegrations, providerConformance, runtimeOwnership] = values;
  return validateRevocationPropagation(registry, {
    catalog,
    adapterCatalog,
    quickstart,
    claimMatrix,
    releaseMatrix,
    productIntegrations,
    providerConformance,
    runtimeOwnership,
  }, { repoRoot, requireBundledTrust: true });
}

export async function main() {
  const report = await buildRevocationPropagationReport();
  if (!report.ok) {
    for (const error of report.errors) console.error(`[revocations] ${error}`);
    process.exitCode = 1;
    return;
  }
  console.log(
    `[revocations] valid (${report.activeRevocations} active; ` +
    `signature verification ${report.signatureVerification})`
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`[revocations] ${error.message}`);
    process.exitCode = 1;
  });
}
