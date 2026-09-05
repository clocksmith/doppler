import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expandExecutionV1 } from '../config/schema/execution-v1.schema.js';
import {
  PROGRAM_BUNDLE_SCHEMA_ID,
  PROGRAM_BUNDLE_SCHEMA_VERSION,
  PROGRAM_BUNDLE_HOST_SCHEMA_ID,
  PROGRAM_BUNDLE_HOST_JS_SUBSET,
  PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID,
  PROGRAM_BUNDLE_PACKAGE_SCHEMA_ID,
  validateProgramBundle,
} from '../config/schema/program-bundle.schema.js';
import { sha256BytesHex, sha256Hex } from '../formats/sha256.js';
import { computeHash } from '../storage/shard-manager.js';
import {
  buildReferenceTranscript,
  createPackageSourceFile,
  fileArtifact,
  hashStableJson,
  readJsonFile,
  readTextFile,
  toRepoRelativePath,
} from './program-bundle/materialize.js';
import {
  normalizeDigest,
  requirePlainObject,
  requireString,
} from './program-bundle/validation.js';
import {
  buildExecutionStepMetadata,
  buildWgslClosure,
} from './program-bundle/wgsl-closure.js';

const DEFAULT_HOST_ENTRYPOINTS = Object.freeze([
  Object.freeze({
    id: 'text-generation',
    module: 'src/tooling/program-bundle-host.js',
    export: 'createTextGenerationProgram',
    role: 'model-orchestration',
  }),
]);

async function shardArtifact(shard, modelDir, repoRoot, hashAlgorithm) {
  const filename = requireString(shard?.filename ?? shard?.path, 'shard filename');
  const shardPath = path.resolve(modelDir, filename);
  if (hashAlgorithm === 'sha256') {
    return {
      role: 'weight-shard',
      path: toRepoRelativePath(shardPath, repoRoot),
      hash: normalizeDigest(shard.hash ?? shard.sha256, `shard ${filename} hash`),
      sizeBytes: Number.isFinite(shard.size) ? Number(shard.size) : null,
    };
  }
  if (hashAlgorithm !== 'sha256' && hashAlgorithm !== 'blake3') {
    throw new Error(`program bundle export: weight shard ${filename} requires an explicit supported hashAlgorithm.`);
  }
  const bytes = await fs.readFile(shardPath);
  const expected = requireString(shard.hash, `shard ${filename} hash`).replace(`${hashAlgorithm}:`, '');
  const observed = await computeHash(bytes, hashAlgorithm);
  if (observed !== expected || bytes.byteLength !== shard.size) {
    throw new Error(`program bundle export: source shard ${filename} hash/size mismatch.`);
  }
  return {
    role: 'weight-shard',
    path: toRepoRelativePath(shardPath, repoRoot),
    hash: `sha256:${sha256BytesHex(bytes)}`,
    sizeBytes: Number.isFinite(shard.size) ? Number(shard.size) : null,
  };
}

function resolveWeightsRefArtifactRoot(modelDir, artifactRoot) {
  const root = typeof artifactRoot === 'string' ? artifactRoot.trim() : '';
  if (!root) return null;
  if (/^file:\/\//i.test(root)) {
    return fileURLToPath(root);
  }
  if (/^[a-z][a-z0-9+.-]*:\/\//iu.test(root)) {
    return null;
  }
  return path.resolve(modelDir, root);
}

function normalizeOptionalDigest(value) {
  if (typeof value !== 'string' || !value.trim()) return null;
  return normalizeDigest(value, 'weightsRef digest');
}

function assertWeightsRefMatchesStorageManifest(manifest, storageManifest, storageManifestRaw, modelId) {
  const weightsRef = manifest?.weightsRef;
  if (!weightsRef) return;

  const expectedManifestDigest = normalizeOptionalDigest(weightsRef.manifestDigest);
  if (expectedManifestDigest) {
    const actualManifestDigest = `sha256:${sha256Hex(storageManifestRaw)}`;
    if (actualManifestDigest !== expectedManifestDigest) {
      throw new Error(
        `program bundle export: ${modelId} weightsRef.manifestDigest ${expectedManifestDigest} ` +
        `does not match target manifest ${actualManifestDigest}.`
      );
    }
  }

  const expectedWeightPackId = typeof weightsRef.weightPackId === 'string'
    ? weightsRef.weightPackId.trim()
    : '';
  if (expectedWeightPackId) {
    const actualWeightPackId = typeof storageManifest?.artifactIdentity?.weightPackId === 'string'
      ? storageManifest.artifactIdentity.weightPackId.trim()
      : '';
    if (actualWeightPackId !== expectedWeightPackId) {
      throw new Error(
        `program bundle export: ${modelId} weightsRef.weightPackId "${expectedWeightPackId}" ` +
        `does not match target artifactIdentity.weightPackId "${actualWeightPackId}".`
      );
    }
  }

  const expectedShardSetHash = normalizeOptionalDigest(weightsRef.shardSetHash);
  if (expectedShardSetHash) {
    const actualShardSetHash = normalizeOptionalDigest(storageManifest?.artifactIdentity?.shardSetHash)
      || normalizeOptionalDigest(storageManifest?.artifactIdentity?.weightPackHash);
    if (actualShardSetHash !== expectedShardSetHash) {
      throw new Error(
        `program bundle export: ${modelId} weightsRef.shardSetHash ${expectedShardSetHash} ` +
        `does not match target artifact identity ${actualShardSetHash}.`
      );
    }
  }
}

export async function resolveProgramBundleStorageArtifact(manifest, modelDir) {
  const weightsRef = manifest?.weightsRef;
  if (!weightsRef) {
    return {
      manifest,
      modelDir,
      manifestPath: null,
      manifestRaw: null,
    };
  }

  const storageModelDir = resolveWeightsRefArtifactRoot(modelDir, weightsRef.artifactRoot);
  if (!storageModelDir) {
    return {
      manifest,
      modelDir,
      manifestPath: null,
      manifestRaw: null,
    };
  }

  const storageManifestPath = path.join(storageModelDir, 'manifest.json');
  const { raw, json: storageManifest } = await readJsonFile(
    storageManifestPath,
    `weightsRef target manifest ${storageManifestPath}`
  );
  const modelId = typeof manifest?.modelId === 'string' && manifest.modelId.trim()
    ? manifest.modelId.trim()
    : 'unknown-model';
  assertWeightsRefMatchesStorageManifest(manifest, storageManifest, raw, modelId);

  return {
    manifest: storageManifest,
    modelDir: storageModelDir,
    manifestPath: storageManifestPath,
    manifestRaw: raw,
  };
}

function scanHostEntrypointSource(source, label) {
  if (/\bimport\s*\(/.test(source)) {
    throw new Error(`program bundle export: host entrypoint ${label} uses dynamic import().`);
  }
  if (/\b(document|window|localStorage|sessionStorage|XMLHttpRequest)\b/.test(source)) {
    throw new Error(`program bundle export: host entrypoint ${label} references DOM-only globals.`);
  }
  if (/^\s*(?:import|export)\s+(?:[^'";]+?\s+from\s+)?['"]/m.test(source)) {
    throw new Error(`program bundle export: host entrypoint ${label} uses a static module dependency.`);
  }
  if (/\b(process|require|module|__dirname|__filename|Deno|Bun)\b/.test(source)) {
    throw new Error(`program bundle export: host entrypoint ${label} references a runtime-specific global.`);
  }
  if (/\b(fetch|WebSocket|EventSource)\b/.test(source)) {
    throw new Error(`program bundle export: host entrypoint ${label} references a network API.`);
  }
  if (/\b(?:eval|Function)\s*\(/.test(source)) {
    throw new Error(`program bundle export: host entrypoint ${label} references dynamic code evaluation.`);
  }
  return {
    dynamicImport: 'none-detected',
    staticImport: 'none-detected',
    dom: 'none-detected',
    runtimeGlobals: 'none-detected',
    network: 'none-detected',
    dynamicCode: 'none-detected',
  };
}

async function buildHostContract(host = {}, repoRoot) {
  const rawEntrypoints = Array.isArray(host.entrypoints) && host.entrypoints.length > 0
    ? host.entrypoints
    : DEFAULT_HOST_ENTRYPOINTS.map((entry) => ({ ...entry }));
  const entrypoints = [];
  const packageFiles = [];
  for (const entrypoint of rawEntrypoints) {
    const modulePath = requireString(entrypoint.module, 'host.entrypoint.module');
    const sourcePath = path.resolve(repoRoot, modulePath);
    const source = await readTextFile(sourcePath, `host entrypoint ${modulePath}`);
    const packageFile = createPackageSourceFile({
      role: 'host-source',
      id: entrypoint.id,
      extension: 'js',
      source,
    });
    packageFiles.push(packageFile);
    entrypoints.push({
      ...entrypoint,
      module: packageFile.path,
      sourceHash: packageFile.hash,
      validation: scanHostEntrypointSource(source, `${modulePath}#${entrypoint.export}`),
    });
  }
  return {
    contract: {
      schema: PROGRAM_BUNDLE_HOST_SCHEMA_ID,
      jsSubset: PROGRAM_BUNDLE_HOST_JS_SUBSET,
      entrypoints,
      constraints: {
        dynamicImport: 'disallowed',
        staticImport: 'disallowed',
        dom: 'disallowed-in-model-path',
        runtimeGlobals: 'disallowed',
        dynamicCode: 'disallowed',
        filesystem: 'declared-artifacts-only',
        network: 'declared-artifacts-only',
        ...(host.constraints && typeof host.constraints === 'object' ? host.constraints : {}),
      },
    },
    packageFiles,
  };
}

function buildCaptureProfile(captureProfile = {}, context = {}) {
  const profile = {
    schema: PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID,
    deterministic: true,
    phases: Array.isArray(captureProfile.phases) && captureProfile.phases.length > 0
      ? captureProfile.phases
      : ['prefill', 'decode'],
    surfaces: Array.isArray(captureProfile.surfaces) && captureProfile.surfaces.length > 0
      ? captureProfile.surfaces
      : [context.adapter?.surface ?? 'unknown-webgpu'],
    adapter: context.adapter ?? {
      source: 'not-captured',
      surface: null,
      deviceInfoHash: hashStableJson(null),
    },
    hashPolicy: {
      graph: 'stable-json-sha256',
      dispatch: 'stable-json-sha256',
      transcript: 'stable-json-sha256',
    },
  };
  return {
    ...profile,
    captureHash: hashStableJson({
      phases: profile.phases,
      surfaces: profile.surfaces,
      adapter: profile.adapter,
      executionGraphHash: context.executionGraphHash ?? null,
      expandedStepHash: context.expandedStepHash ?? null,
      wgslModuleDigests: context.wgslModuleDigests ?? [],
      hostSourceHashes: context.hostSourceHashes ?? [],
    }),
  };
}

async function collectArtifacts(options, manifest, manifestArtifact, referenceReportArtifact, programArtifacts) {
  const artifacts = [manifestArtifact, ...programArtifacts];
  const storageArtifact = await resolveProgramBundleStorageArtifact(manifest, options.modelDir);
  const storageManifest = storageArtifact.manifest;
  const storageModelDir = storageArtifact.modelDir;
  if (storageArtifact.manifestPath && storageArtifact.manifestRaw != null) {
    artifacts.push({
      role: 'source',
      path: toRepoRelativePath(storageArtifact.manifestPath, options.repoRoot),
      hash: `sha256:${sha256Hex(storageArtifact.manifestRaw)}`,
      sizeBytes: Buffer.byteLength(storageArtifact.manifestRaw),
    });
  }

  if (Array.isArray(storageManifest.shards)) {
    for (const shard of storageManifest.shards) {
      artifacts.push(await shardArtifact(
        shard,
        storageModelDir,
        options.repoRoot,
        storageManifest.hashAlgorithm
      ));
    }
  }

  const tokenizerFile = storageManifest.tokenizer?.file;
  if (typeof tokenizerFile === 'string' && tokenizerFile.trim()) {
    artifacts.push(await fileArtifact({
      role: 'tokenizer',
      filePath: path.resolve(storageModelDir, tokenizerFile),
      repoRoot: options.repoRoot,
    }));
  }

  if (options.conversionConfigPath) {
    artifacts.push(await fileArtifact({
      role: 'conversion-config',
      filePath: path.resolve(options.conversionConfigPath),
      repoRoot: options.repoRoot,
    }));
  }

  if (options.runtimeConfigPath) {
    artifacts.push(await fileArtifact({
      role: 'runtime-config',
      filePath: path.resolve(options.runtimeConfigPath),
      repoRoot: options.repoRoot,
    }));
  }

  artifacts.push(referenceReportArtifact);
  return artifacts.sort((left, right) => `${left.role}:${left.path}`.localeCompare(`${right.role}:${right.path}`));
}

function resolveProgramBundleOptions(options = {}) {
  const repoRoot = path.resolve(options.repoRoot || process.cwd());
  const manifestPath = options.manifestPath
    ? path.resolve(options.manifestPath)
    : (options.modelDir ? path.resolve(options.modelDir, 'manifest.json') : null);
  if (!manifestPath) {
    throw new Error('program bundle export: manifestPath or modelDir is required.');
  }
  const modelDir = path.resolve(options.modelDir || path.dirname(manifestPath));
  const referenceReportPath = options.referenceReportPath
    ? path.resolve(options.referenceReportPath)
    : null;
  if (!referenceReportPath) {
    throw new Error('program bundle export: referenceReportPath is required.');
  }
  return {
    ...options,
    repoRoot,
    manifestPath,
    modelDir,
    referenceReportPath,
  };
}

async function buildProgramBundle(options = {}) {
  const resolvedOptions = resolveProgramBundleOptions(options);
  const { raw: manifestRaw, json: manifest } = await readJsonFile(resolvedOptions.manifestPath, 'manifest');
  requirePlainObject(manifest, 'manifest');
  const modelId = requireString(manifest.modelId, 'manifest.modelId');
  const execution = manifest.inference?.execution;
  requirePlainObject(execution, 'manifest.inference.execution');

  const expandedSteps = expandExecutionV1(execution);
  const executionGraphHash = hashStableJson(execution);
  const expandedStepHash = hashStableJson(expandedSteps);
  const closure = await buildWgslClosure(execution, expandedSteps, resolvedOptions);
  const executionMetadata = buildExecutionStepMetadata(execution, expandedSteps, closure.modules);
  const hostResult = await buildHostContract(resolvedOptions.host ?? (
    manifest.inference?.supportsSequence === true ? { entrypoints: [{
      id: 'sequence-encoding', module: 'src/tooling/program-bundle-host.js',
      export: 'createSequenceProgram', role: 'model-orchestration',
    }] } : undefined
  ), resolvedOptions.repoRoot);
  const host = hostResult.contract;
  const packageFiles = [...closure.packageFiles, ...hostResult.packageFiles]
    .sort((left, right) => left.path.localeCompare(right.path));
  const manifestArtifact = {
    role: 'manifest',
    path: toRepoRelativePath(resolvedOptions.manifestPath, resolvedOptions.repoRoot),
    hash: `sha256:${sha256Hex(manifestRaw)}`,
    sizeBytes: Buffer.byteLength(manifestRaw),
  };
  const reference = await buildReferenceTranscript(
    resolvedOptions.referenceReportPath,
    resolvedOptions.repoRoot,
    executionGraphHash
  );
  if (reference.transcript.operation === 'encodeSequence') {
    if (manifest.inference?.supportsSequence !== true
      || reference.transcript.modelId !== modelId
      || reference.transcript.manifestHash !== manifestArtifact.hash
      || reference.transcript.reference.source.checkpointId !== manifest.artifactIdentity?.sourceCheckpointId) {
      throw new Error('program bundle export: sequence qualification does not bind this exact manifest.');
    }
  }
  const artifacts = await collectArtifacts(
    resolvedOptions,
    manifest,
    manifestArtifact,
    reference.artifact,
    packageFiles.map((file) => file.artifact),
  );
  const weightArtifacts = artifacts.filter((artifact) => artifact.role === 'weight-shard');
  const conversionConfig = resolvedOptions.conversionConfigPath
    ? artifacts.find((artifact) => artifact.role === 'conversion-config') ?? null
    : null;

  const bundle = {
    schema: PROGRAM_BUNDLE_SCHEMA_ID,
    schemaVersion: PROGRAM_BUNDLE_SCHEMA_VERSION,
    bundleId: resolvedOptions.bundleId || `${modelId}-${executionGraphHash.slice('sha256:'.length, 'sha256:'.length + 12)}`,
    modelId,
    createdAtUtc: resolvedOptions.createdAtUtc || new Date().toISOString(),
    package: {
      schema: PROGRAM_BUNDLE_PACKAGE_SCHEMA_ID,
      root: '.',
      files: packageFiles.map((file) => file.artifact),
      fileSetHash: hashStableJson(packageFiles.map((file) => file.artifact)),
    },
    sources: {
      manifest: {
        path: manifestArtifact.path,
        hash: manifestArtifact.hash,
      },
      conversionConfig: conversionConfig
        ? {
          path: conversionConfig.path,
          hash: conversionConfig.hash,
        }
        : null,
      executionGraph: {
        schema: manifest.inference?.schema ?? null,
        hash: executionGraphHash,
        expandedStepHash,
      },
      weightSetHash: hashStableJson(weightArtifacts.map((artifact) => ({
        path: artifact.path,
        hash: artifact.hash,
        sizeBytes: artifact.sizeBytes,
      }))),
      artifactSetHash: hashStableJson(artifacts.map((artifact) => ({
        role: artifact.role,
        path: artifact.path,
        hash: artifact.hash,
        sizeBytes: artifact.sizeBytes,
      }))),
    },
    host,
    wgslModules: closure.modules,
    execution: {
      graphHash: executionGraphHash,
      stepMetadataHash: executionMetadata.stepMetadataHash,
      kernelClosure: closure.kernelClosure,
      steps: executionMetadata.steps,
    },
    captureProfile: buildCaptureProfile(resolvedOptions.captureProfile, {
      adapter: reference.adapter,
      executionGraphHash,
      expandedStepHash,
      wgslModuleDigests: closure.modules.map((module) => ({
        id: module.id,
        digest: module.digest,
        metadataHash: module.metadata.sourceMetadataHash,
      })),
      hostSourceHashes: host.entrypoints.map((entrypoint) => ({
        id: entrypoint.id,
        sourceHash: entrypoint.sourceHash,
      })),
    }),
    artifacts,
    referenceTranscript: reference.transcript,
  };

  return {
    bundle: validateProgramBundle(bundle),
    packageFiles,
  };
}

export async function exportProgramBundle(options = {}) {
  return (await buildProgramBundle(options)).bundle;
}

export async function writeProgramBundle(options = {}) {
  const outputPath = options.outputPath ? path.resolve(options.outputPath) : null;
  if (!outputPath) {
    throw new Error('program bundle export: outputPath is required.');
  }
  const { bundle, packageFiles } = await buildProgramBundle(options);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  for (const file of packageFiles) {
    const destination = path.resolve(path.dirname(outputPath), file.path);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.writeFile(destination, file.contents, 'utf8');
  }
  await fs.writeFile(outputPath, `${JSON.stringify(bundle, null, 2)}\n`, 'utf8');
  await verifyClosedProgramBundle(outputPath, bundle);
  return {
    outputPath,
    bundle,
  };
}

export async function loadProgramBundle(bundlePath) {
  const resolvedPath = path.resolve(bundlePath);
  const { json } = await readJsonFile(resolvedPath, 'program bundle');
  return validateProgramBundle(json);
}

export async function verifyClosedProgramBundle(bundlePath, providedBundle = null) {
  const resolvedPath = path.resolve(bundlePath);
  const bundle = providedBundle ?? await loadProgramBundle(resolvedPath);
  const bundleRoot = path.dirname(resolvedPath);
  const files = [];
  for (const file of bundle.package.files) {
    const filePath = path.resolve(bundleRoot, file.path);
    const relative = path.relative(bundleRoot, filePath);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      throw new Error(`program bundle: packaged file escapes bundle root: ${file.path}.`);
    }
    let bytes;
    try {
      bytes = await fs.readFile(filePath);
    } catch (error) {
      throw new Error(`program bundle: packaged file is unavailable: ${file.path}: ${error.message}`);
    }
    const hash = `sha256:${sha256BytesHex(bytes)}`;
    if (hash !== file.hash || bytes.byteLength !== file.sizeBytes) {
      throw new Error(`program bundle: packaged file hash/size mismatch: ${file.path}.`);
    }
    files.push({ ...file, absolutePath: filePath });
  }
  return {
    ok: true,
    bundle,
    bundlePath: resolvedPath,
    files,
  };
}
export async function checkProgramBundleFile(bundlePath) {
  const closed = await verifyClosedProgramBundle(bundlePath);
  const bundle = closed.bundle;
  return {
    ok: true,
    path: path.resolve(bundlePath),
    modelId: bundle.modelId,
    bundleId: bundle.bundleId,
    artifactCount: bundle.artifacts.length,
    wgslModuleCount: bundle.wgslModules.length,
    packagedFileCount: closed.files.length,
    executionGraphHash: bundle.sources.executionGraph.hash,
  };
}

export function createProgramBundleCliDefaults(metaUrl) {
  return {
    repoRoot: path.resolve(path.dirname(fileURLToPath(metaUrl)), '..'),
  };
}
