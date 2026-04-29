#!/usr/bin/env node

import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const DEFAULT_ROOTS = [
  'models/local',
  '/Volumes/models/rdrr',
  '/media/x/models/rdrr',
];

function parseArgs(argv) {
  const options = {
    roots: [],
    catalogPath: 'models/catalog.json',
    json: false,
    check: false,
    pretty: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const nextValue = () => {
      const value = argv[i + 1];
      if (value == null || String(value).startsWith('--')) {
        throw new Error(`Missing value for ${arg}.`);
      }
      i += 1;
      return value;
    };
    if (arg === '--root') {
      options.roots.push(nextValue());
      continue;
    }
    if (arg === '--catalog') {
      options.catalogPath = nextValue();
      continue;
    }
    if (arg === '--json') {
      options.json = true;
      continue;
    }
    if (arg === '--pretty') {
      options.pretty = true;
      continue;
    }
    if (arg === '--check') {
      options.check = true;
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      options.help = true;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }
  return options;
}

function printHelp() {
  console.log(`Usage: node tools/artifact-identity-inventory.js [options]

Options:
  --root <path>       Artifact root to scan. May be repeated.
  --catalog <path>    Catalog path. Defaults to models/catalog.json.
  --json              Emit JSON only.
  --pretty            Pretty-print JSON.
  --check             Exit non-zero when blocking findings are present.
  --help              Show this help.
`);
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

function isManifestFilename(filename) {
  return /^manifest(?:[-_].*)?\.json$/u.test(filename);
}

function isPrimaryManifestFilename(filename) {
  return filename === 'manifest.json';
}

async function collectManifestFiles(root) {
  const manifests = [];
  const pending = [root];
  while (pending.length > 0) {
    const dir = pending.pop();
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const entryPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        pending.push(entryPath);
        continue;
      }
      if (entry.isFile() && isManifestFilename(entry.name)) {
        manifests.push(entryPath);
      }
    }
  }
  manifests.sort();
  return manifests;
}

function resolveShardFilename(shard) {
  if (typeof shard === 'string') {
    return shard;
  }
  if (!shard || typeof shard !== 'object') {
    return null;
  }
  for (const key of ['filename', 'file', 'path', 'name']) {
    if (typeof shard[key] === 'string' && shard[key].trim().length > 0) {
      return shard[key].trim();
    }
  }
  return null;
}

function normalizeRelativePath(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : null;
}

function resolveArtifactRootFromWeightsRef(weightsRef, manifestDir, repoRoot) {
  const artifactRoot = normalizeRelativePath(weightsRef?.artifactRoot)
    ?? normalizeRelativePath(weightsRef?.root)
    ?? normalizeRelativePath(weightsRef?.modelRoot);
  if (!artifactRoot) {
    return null;
  }
  if (path.isAbsolute(artifactRoot)) {
    return artifactRoot;
  }
  const fromRepo = path.resolve(repoRoot, artifactRoot);
  if (fsSync.existsSync(fromRepo)) {
    return fromRepo;
  }
  return path.resolve(manifestDir, artifactRoot);
}

async function inspectShardSet(manifestDir, shards) {
  const shardFiles = [];
  const missingShards = [];
  const presentShards = [];
  const shardSizes = [];
  for (const shard of shards) {
    const filename = resolveShardFilename(shard);
    if (!filename) {
      continue;
    }
    shardFiles.push(filename);
    const shardPath = path.resolve(manifestDir, filename);
    try {
      const stat = await fs.stat(shardPath);
      if (stat.isFile()) {
        presentShards.push(filename);
        shardSizes.push(stat.size);
        continue;
      }
      missingShards.push(filename);
    } catch {
      missingShards.push(filename);
    }
  }
  return {
    shardFiles,
    presentShards,
    missingShards,
    shardSizes,
  };
}

async function readOrigin(manifestDir) {
  const originPath = path.join(manifestDir, 'origin.json');
  if (!(await pathExists(originPath))) {
    return null;
  }
  try {
    return await readJson(originPath);
  } catch {
    return null;
  }
}

function resolveSourceCheckpointId(manifest, origin) {
  const identity = manifest?.artifactIdentity;
  if (typeof identity?.sourceCheckpointId === 'string') {
    return identity.sourceCheckpointId;
  }
  const repo = identity?.sourceRepo ?? origin?.sourceRepo ?? origin?.sourceModel ?? null;
  const revision = identity?.sourceRevision ?? origin?.sourceRevision ?? null;
  if (typeof repo === 'string' && repo.length > 0 && typeof revision === 'string' && revision.length > 0) {
    return `${repo}@${revision}`;
  }
  if (typeof repo === 'string' && repo.length > 0) {
    return repo;
  }
  return null;
}

function resolveWeightPackId(manifest) {
  const identity = manifest?.artifactIdentity;
  if (typeof identity?.weightPackId === 'string' && identity.weightPackId.length > 0) {
    return identity.weightPackId;
  }
  return null;
}

function resolveManifestVariantId(manifest) {
  const identity = manifest?.artifactIdentity;
  if (typeof identity?.manifestVariantId === 'string' && identity.manifestVariantId.length > 0) {
    return identity.manifestVariantId;
  }
  return null;
}

function classifyArtifact({ manifestFile, manifest, declaredShardCount, presentShardCount, missingShardCount, hasWeightsRef, weightsRefStatus }) {
  if (!isPrimaryManifestFilename(manifestFile)) {
    return 'sidecar-manifest';
  }
  if (declaredShardCount === 0) {
    return hasWeightsRef ? 'weights-ref-manifest' : 'metadata-only-no-shards';
  }
  if (missingShardCount === 0) {
    return 'complete-local-shards';
  }
  if (hasWeightsRef && weightsRefStatus === 'resolved') {
    return 'incomplete-local-shards-with-resolved-weights-ref';
  }
  if (hasWeightsRef) {
    return 'incomplete-local-shards-with-unresolved-weights-ref';
  }
  return presentShardCount === 0
    ? 'invalid-manifest-only'
    : 'invalid-incomplete-shards';
}

function artifactCompletenessFromClassification(classification) {
  if (classification === 'complete-local-shards') {
    return 'complete';
  }
  if (classification === 'weights-ref-manifest' || classification === 'incomplete-local-shards-with-resolved-weights-ref') {
    return 'weights-ref';
  }
  if (classification === 'sidecar-manifest') {
    return 'sidecar';
  }
  return 'incomplete';
}

async function inspectWeightsRef(weightsRef, manifestDir, repoRoot) {
  if (!weightsRef || typeof weightsRef !== 'object') {
    return {
      hasWeightsRef: false,
      weightsRefStatus: 'absent',
      weightsRefArtifactRoot: null,
    };
  }
  const weightsRefArtifactRoot = resolveArtifactRootFromWeightsRef(weightsRef, manifestDir, repoRoot);
  if (!weightsRefArtifactRoot) {
    return {
      hasWeightsRef: true,
      weightsRefStatus: 'missing-artifact-root',
      weightsRefArtifactRoot: null,
    };
  }
  const targetManifestPath = path.join(weightsRefArtifactRoot, 'manifest.json');
  if (!(await pathExists(targetManifestPath))) {
    return {
      hasWeightsRef: true,
      weightsRefStatus: 'missing-target-manifest',
      weightsRefArtifactRoot,
    };
  }
  try {
    const targetManifest = await readJson(targetManifestPath);
    const expectedWeightPackId = normalizeRelativePath(weightsRef.weightPackId);
    const targetWeightPackId = resolveWeightPackId(targetManifest);
    if (expectedWeightPackId && targetWeightPackId && expectedWeightPackId !== targetWeightPackId) {
      return {
        hasWeightsRef: true,
        weightsRefStatus: 'weight-pack-mismatch',
        weightsRefArtifactRoot,
      };
    }
    return {
      hasWeightsRef: true,
      weightsRefStatus: 'resolved',
      weightsRefArtifactRoot,
    };
  } catch {
    return {
      hasWeightsRef: true,
      weightsRefStatus: 'invalid-target-manifest',
      weightsRefArtifactRoot,
    };
  }
}

async function inspectManifest(filePath, root, repoRoot) {
  const manifestDir = path.dirname(filePath);
  const manifestFile = path.basename(filePath);
  try {
    const manifest = await readJson(filePath);
    const origin = await readOrigin(manifestDir);
    const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
    const shardSet = await inspectShardSet(manifestDir, shards);
    const weightsRef = await inspectWeightsRef(manifest?.weightsRef, manifestDir, repoRoot);
    const declaredShardCount = shardSet.shardFiles.length;
    const presentShardCount = shardSet.presentShards.length;
    const missingShardCount = shardSet.missingShards.length;
    const classification = classifyArtifact({
      manifestFile,
      manifest,
      declaredShardCount,
      presentShardCount,
      missingShardCount,
      hasWeightsRef: weightsRef.hasWeightsRef,
      weightsRefStatus: weightsRef.weightsRefStatus,
    });
    const shardLayoutKey = missingShardCount === 0 && declaredShardCount > 0
      ? `${declaredShardCount}:${shardSet.shardSizes.join(',')}`
      : null;
    return {
      artifactRoot: path.relative(repoRoot, manifestDir) || '.',
      absoluteArtifactRoot: manifestDir,
      scanRoot: root,
      manifestPath: path.relative(repoRoot, filePath) || filePath,
      manifestFile,
      modelId: manifest?.modelId ?? path.basename(manifestDir),
      sourceCheckpointId: resolveSourceCheckpointId(manifest, origin),
      weightPackId: resolveWeightPackId(manifest),
      manifestVariantId: resolveManifestVariantId(manifest),
      declaredShardCount,
      presentShardCount,
      missingShardCount,
      missingShards: shardSet.missingShards,
      hasWeightsRef: weightsRef.hasWeightsRef,
      weightsRefStatus: weightsRef.weightsRefStatus,
      weightsRefArtifactRoot: weightsRef.weightsRefArtifactRoot
        ? path.relative(repoRoot, weightsRef.weightsRefArtifactRoot) || weightsRef.weightsRefArtifactRoot
        : null,
      artifactCompleteness: artifactCompletenessFromClassification(classification),
      classification,
      shardLayoutKey,
      hasArtifactIdentity: manifest?.artifactIdentity != null,
      originSourceRepo: origin?.sourceRepo ?? null,
      originSourceRevision: origin?.sourceRevision ?? null,
    };
  } catch (error) {
    return {
      artifactRoot: path.relative(repoRoot, manifestDir) || '.',
      absoluteArtifactRoot: manifestDir,
      scanRoot: root,
      manifestPath: path.relative(repoRoot, filePath) || filePath,
      manifestFile,
      modelId: path.basename(manifestDir),
      sourceCheckpointId: null,
      weightPackId: null,
      manifestVariantId: null,
      declaredShardCount: 0,
      presentShardCount: 0,
      missingShardCount: 0,
      missingShards: [],
      hasWeightsRef: false,
      weightsRefStatus: 'absent',
      weightsRefArtifactRoot: null,
      artifactCompleteness: 'invalid',
      classification: 'invalid-json',
      shardLayoutKey: null,
      hasArtifactIdentity: false,
      originSourceRepo: null,
      originSourceRevision: null,
      error: error?.message ?? String(error),
    };
  }
}

async function loadCatalog(catalogPath, repoRoot) {
  const absoluteCatalogPath = path.resolve(repoRoot, catalogPath);
  if (!(await pathExists(absoluteCatalogPath))) {
    return {
      catalogPath,
      entries: [],
      error: 'missing-catalog',
    };
  }
  try {
    const catalog = await readJson(absoluteCatalogPath);
    const entries = Array.isArray(catalog?.models) ? catalog.models : [];
    return {
      catalogPath,
      entries: entries.map((entry) => ({
        modelId: entry?.modelId ?? null,
        quickstart: entry?.quickstart ?? null,
        sourceCheckpointId: entry?.sourceCheckpointId ?? null,
        weightPackId: entry?.weightPackId ?? null,
        manifestVariantId: entry?.manifestVariantId ?? null,
        artifactCompleteness: entry?.artifactCompleteness ?? null,
        runtimePromotionState: entry?.runtimePromotionState ?? null,
        weightsRefAllowed: entry?.weightsRefAllowed ?? null,
        lifecycle: {
          availability: {
            hf: entry?.lifecycle?.availability?.hf ?? null,
          },
        },
      })).filter((entry) => typeof entry.modelId === 'string' && entry.modelId.length > 0),
      error: null,
    };
  } catch (error) {
    return {
      catalogPath,
      entries: [],
      error: error?.message ?? String(error),
    };
  }
}

function groupBy(artifacts, key) {
  const groups = new Map();
  for (const artifact of artifacts) {
    const value = artifact[key];
    if (typeof value !== 'string' || value.length === 0) {
      continue;
    }
    const group = groups.get(value) ?? [];
    group.push(artifact);
    groups.set(value, group);
  }
  return Array.from(groups.entries())
    .filter(([, group]) => group.length > 1)
    .map(([value, group]) => ({
      value,
      artifacts: group.map((artifact) => ({
        modelId: artifact.modelId,
        manifestPath: artifact.manifestPath,
        classification: artifact.classification,
      })),
    }));
}

function findCatalogEntry(catalog, modelId) {
  const needle = typeof modelId === 'string' ? modelId.trim() : '';
  if (!needle) return null;
  return catalog.entries.find((entry) => entry.modelId === needle) ?? null;
}

function isKnownUnpromotedIncomplete(entry) {
  return entry?.artifactCompleteness === 'incomplete'
    && entry?.runtimePromotionState === 'unpromoted'
    && entry?.quickstart !== true
    && entry?.lifecycle?.availability?.hf !== true;
}

function incompleteArtifactSeverity(artifact, catalog) {
  const entry = findCatalogEntry(catalog, artifact.modelId);
  return isKnownUnpromotedIncomplete(entry) ? 'warning' : 'error';
}

function buildFindings(artifacts, catalog) {
  const findings = [];
  for (const artifact of artifacts) {
    if (artifact.classification === 'invalid-json') {
      findings.push({
        severity: 'error',
        code: 'invalid_manifest_json',
        manifestPath: artifact.manifestPath,
        message: `Invalid manifest JSON: ${artifact.error}`,
      });
    }
    if (artifact.classification === 'invalid-manifest-only') {
      findings.push({
        severity: incompleteArtifactSeverity(artifact, catalog),
        code: 'manifest_only_without_weights_ref',
        manifestPath: artifact.manifestPath,
        modelId: artifact.modelId,
        message: `${artifact.modelId} declares ${artifact.declaredShardCount} shards, but none are present and no weightsRef is declared.`,
      });
    }
    if (artifact.classification === 'invalid-incomplete-shards') {
      findings.push({
        severity: incompleteArtifactSeverity(artifact, catalog),
        code: 'missing_declared_shards',
        manifestPath: artifact.manifestPath,
        modelId: artifact.modelId,
        message: `${artifact.modelId} is missing ${artifact.missingShardCount}/${artifact.declaredShardCount} declared shards.`,
      });
    }
    if (artifact.classification === 'incomplete-local-shards-with-unresolved-weights-ref') {
      findings.push({
        severity: incompleteArtifactSeverity(artifact, catalog),
        code: 'unresolved_weights_ref',
        manifestPath: artifact.manifestPath,
        modelId: artifact.modelId,
        message: `${artifact.modelId} has missing local shards and unresolved weightsRef status "${artifact.weightsRefStatus}".`,
      });
    }
    if (artifact.classification === 'metadata-only-no-shards') {
      findings.push({
        severity: 'warning',
        code: 'metadata_only_without_weights_ref',
        manifestPath: artifact.manifestPath,
        modelId: artifact.modelId,
        message: `${artifact.modelId} declares no shards and no weightsRef.`,
      });
    }
    if (artifact.classification === 'sidecar-manifest') {
      findings.push({
        severity: 'warning',
        code: 'sidecar_manifest_variant',
        manifestPath: artifact.manifestPath,
        modelId: artifact.modelId,
        message: `${artifact.manifestFile} is a sidecar manifest and should become a named manifestVariant or be removed.`,
      });
    }
  }

  const artifactsByModelId = new Map();
  for (const artifact of artifacts) {
    if (!isPrimaryManifestFilename(artifact.manifestFile)) {
      continue;
    }
    const group = artifactsByModelId.get(artifact.modelId) ?? [];
    group.push(artifact);
    artifactsByModelId.set(artifact.modelId, group);
  }
  for (const entry of catalog.entries) {
    const matchingArtifacts = artifactsByModelId.get(entry.modelId) ?? [];
    const incomplete = matchingArtifacts.filter((artifact) => {
      return artifact.artifactCompleteness === 'incomplete' || artifact.artifactCompleteness === 'invalid';
    });
    if (incomplete.length > 0) {
      findings.push({
        severity: isKnownUnpromotedIncomplete(entry) ? 'warning' : 'error',
        code: 'catalog_points_to_incomplete_local_artifact',
        modelId: entry.modelId,
        message: `Catalog entry ${entry.modelId} has ${incomplete.length} incomplete local artifact(s).`,
      });
    }
    if (
      entry.sourceCheckpointId == null
      || entry.weightPackId == null
      || entry.manifestVariantId == null
      || entry.artifactCompleteness == null
      || entry.runtimePromotionState == null
      || entry.weightsRefAllowed == null
    ) {
      findings.push({
        severity: 'warning',
        code: 'catalog_missing_artifact_identity',
        modelId: entry.modelId,
        message: `Catalog entry ${entry.modelId} has not been backfilled with artifact identity fields.`,
      });
    }
    if (entry.quickstart === true && entry.artifactCompleteness !== 'complete') {
      findings.push({
        severity: 'error',
        code: 'quickstart_catalog_artifact_incomplete',
        modelId: entry.modelId,
        message: `Quickstart entry ${entry.modelId} must have artifactCompleteness="complete".`,
      });
    }
    if (entry.lifecycle?.availability?.hf === true && entry.runtimePromotionState !== 'manifest-owned') {
      findings.push({
        severity: 'error',
        code: 'hosted_catalog_artifact_not_manifest_owned',
        modelId: entry.modelId,
        message: `Hosted entry ${entry.modelId} must have runtimePromotionState="manifest-owned".`,
      });
    }
  }
  return findings;
}

function summarize(artifacts, findings, catalog) {
  return {
    artifactCount: artifacts.length,
    primaryManifestCount: artifacts.filter((artifact) => isPrimaryManifestFilename(artifact.manifestFile)).length,
    sidecarManifestCount: artifacts.filter((artifact) => !isPrimaryManifestFilename(artifact.manifestFile)).length,
    completeCount: artifacts.filter((artifact) => artifact.artifactCompleteness === 'complete').length,
    incompleteCount: artifacts.filter((artifact) => artifact.artifactCompleteness === 'incomplete').length,
    weightsRefCount: artifacts.filter((artifact) => artifact.hasWeightsRef).length,
    catalogEntryCount: catalog.entries.length,
    errorCount: findings.filter((finding) => finding.severity === 'error').length,
    warningCount: findings.filter((finding) => finding.severity === 'warning').length,
  };
}

function renderText(report) {
  const lines = [];
  lines.push(`Artifact identity inventory: ${report.summary.artifactCount} manifests`);
  lines.push(`Complete: ${report.summary.completeCount}`);
  lines.push(`Incomplete: ${report.summary.incompleteCount}`);
  lines.push(`Sidecars: ${report.summary.sidecarManifestCount}`);
  lines.push(`Findings: ${report.summary.errorCount} error(s), ${report.summary.warningCount} warning(s)`);
  for (const finding of report.findings) {
    lines.push(`[${finding.severity}] ${finding.code}: ${finding.message}`);
  }
  return lines.join('\n');
}

async function buildReport(options) {
  const repoRoot = process.cwd();
  const configuredRoots = options.roots.length > 0 ? options.roots : DEFAULT_ROOTS;
  const existingRoots = [];
  for (const root of configuredRoots) {
    const absoluteRoot = path.resolve(repoRoot, root);
    if (await pathExists(absoluteRoot)) {
      existingRoots.push(absoluteRoot);
    }
  }

  const manifestFiles = [];
  for (const root of existingRoots) {
    const files = await collectManifestFiles(root);
    manifestFiles.push(...files.map((file) => ({ file, root })));
  }

  const artifacts = [];
  for (const entry of manifestFiles) {
    artifacts.push(await inspectManifest(entry.file, entry.root, repoRoot));
  }
  artifacts.sort((a, b) => a.manifestPath.localeCompare(b.manifestPath));

  const catalog = await loadCatalog(options.catalogPath, repoRoot);
  const findings = buildFindings(artifacts, catalog);
  const duplicateGroups = {
    sourceCheckpoint: groupBy(artifacts, 'sourceCheckpointId'),
    weightPack: groupBy(artifacts, 'weightPackId'),
    shardLayout: groupBy(artifacts, 'shardLayoutKey'),
  };

  return {
    schemaVersion: 1,
    generatedAtUtc: new Date().toISOString(),
    roots: existingRoots.map((root) => path.relative(repoRoot, root) || root),
    missingRoots: configuredRoots
      .map((root) => path.resolve(repoRoot, root))
      .filter((root) => !existingRoots.includes(root))
      .map((root) => path.relative(repoRoot, root) || root),
    catalog,
    summary: summarize(artifacts, findings, catalog),
    artifacts,
    duplicateGroups,
    findings,
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  const report = await buildReport(options);
  if (options.json || options.check) {
    console.log(JSON.stringify(report, null, options.pretty ? 2 : 0));
  } else {
    console.log(renderText(report));
  }
  if (options.check && report.summary.errorCount > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(`[artifact-identity] ${error?.stack ?? error?.message ?? String(error)}`);
  process.exit(1);
});
