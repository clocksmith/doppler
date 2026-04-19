import path from 'node:path';

import {
  DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE,
  DIRECT_SOURCE_RUNTIME_MODE,
  DIRECT_SOURCE_RUNTIME_SCHEMA,
  DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION,
  getSourceRuntimeMetadata,
} from './source-runtime-bundle.js';
import { cloneJsonValue } from '../utils/clone-json.js';

function toRelativeArtifactPath(value, artifactDir, label) {
  const raw = String(value || '').trim();
  if (!raw) {
    throw new Error(`${label} path is required.`);
  }
  const resolvedArtifactDir = path.resolve(artifactDir);
  const resolvedTarget = path.isAbsolute(raw)
    ? path.resolve(raw)
    : path.resolve(resolvedArtifactDir, raw);
  const relativePath = path.relative(resolvedArtifactDir, resolvedTarget).replace(/\\/g, '/');
  if (!relativePath || relativePath.startsWith('../') || relativePath === '..') {
    throw new Error(
      `${label} "${raw}" must live inside artifactDir "${resolvedArtifactDir}" for a persisted direct-source manifest.`
    );
  }
  return relativePath;
}

export function materializeSourceRuntimeManifest(manifest, artifactDir) {
  const sourceRuntime = getSourceRuntimeMetadata(manifest);
  if (!sourceRuntime) {
    throw new Error('materializeSourceRuntimeManifest requires manifest.metadata.sourceRuntime.');
  }
  const resolvedArtifactDir = String(artifactDir || '').trim();
  if (!resolvedArtifactDir) {
    throw new Error('materializeSourceRuntimeManifest requires artifactDir.');
  }

  const nextManifest = cloneJsonValue(manifest);
  if (!nextManifest.metadata || typeof nextManifest.metadata !== 'object') {
    nextManifest.metadata = {};
  }
  const sourceMetadata = nextManifest.metadata.sourceRuntime && typeof nextManifest.metadata.sourceRuntime === 'object'
    ? cloneJsonValue(nextManifest.metadata.sourceRuntime)
    : {};

  sourceMetadata.mode = DIRECT_SOURCE_RUNTIME_MODE;
  sourceMetadata.schema = DIRECT_SOURCE_RUNTIME_SCHEMA;
  sourceMetadata.schemaVersion = DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION;
  sourceMetadata.hashAlgorithm = sourceRuntime.hashAlgorithm;
  sourceMetadata.pathSemantics = DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE;
  sourceMetadata.sourceFiles = sourceRuntime.sourceFiles.map((entry) => ({
    index: entry.index,
    filename: entry.filename ?? null,
    path: toRelativeArtifactPath(
      entry.path,
      resolvedArtifactDir,
      `source runtime source file ${entry.index}`
    ),
    size: entry.size,
    hash: entry.hash,
    hashAlgorithm: entry.hashAlgorithm,
  }));
  sourceMetadata.auxiliaryFiles = sourceRuntime.auxiliaryFiles.map((entry) => ({
    path: toRelativeArtifactPath(
      entry.path,
      resolvedArtifactDir,
      `source runtime auxiliary file ${entry.kind}`
    ),
    size: entry.size,
    hash: entry.hash,
    hashAlgorithm: entry.hashAlgorithm,
    kind: entry.kind,
  }));
  sourceMetadata.tokenizer = {
    jsonPath: sourceRuntime.tokenizer.jsonPath
      ? toRelativeArtifactPath(sourceRuntime.tokenizer.jsonPath, resolvedArtifactDir, 'source runtime tokenizer json')
      : null,
    configPath: sourceRuntime.tokenizer.configPath
      ? toRelativeArtifactPath(sourceRuntime.tokenizer.configPath, resolvedArtifactDir, 'source runtime tokenizer config')
      : null,
    modelPath: sourceRuntime.tokenizer.modelPath
      ? toRelativeArtifactPath(sourceRuntime.tokenizer.modelPath, resolvedArtifactDir, 'source runtime tokenizer model')
      : null,
  };
  nextManifest.metadata.sourceRuntime = sourceMetadata;
  return nextManifest;
}
