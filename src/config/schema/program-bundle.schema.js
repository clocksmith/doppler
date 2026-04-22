import { isExecutionV1Digest } from './execution-v1.schema.js';

export const PROGRAM_BUNDLE_SCHEMA_VERSION = 1;
export const PROGRAM_BUNDLE_SCHEMA_ID = 'doppler.program-bundle/v1';
export const PROGRAM_BUNDLE_HOST_SCHEMA_ID = 'doppler.host-js/v1';
export const PROGRAM_BUNDLE_HOST_JS_SUBSET = 'doppler-webgpu-host/v1';
export const PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID = 'doppler.capture-profile/v1';
export const PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID = 'doppler.reference-transcript/v1';

const ARTIFACT_ROLE_SET = new Set([
  'manifest',
  'weight-shard',
  'tokenizer',
  'conversion-config',
  'runtime-config',
  'reference-report',
  'source',
  'other',
]);

function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`program bundle: ${label} must be a non-null object.`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`program bundle: ${label} must be a non-empty string.`);
  }
}

function assertArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`program bundle: ${label} must be an array.`);
  }
}

function assertNonEmptyArray(value, label) {
  assertArray(value, label);
  if (value.length === 0) {
    throw new Error(`program bundle: ${label} must contain at least one entry.`);
  }
}

function assertDigest(value, label) {
  if (!isExecutionV1Digest(value)) {
    throw new Error(`program bundle: ${label} must match sha256:<64 hex chars>.`);
  }
}

function assertNullableFiniteNumber(value, label) {
  if (value === null) return;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`program bundle: ${label} must be a non-negative finite number or null.`);
  }
}

function assertNullablePlainObject(value, label) {
  if (value === null) return;
  assertPlainObject(value, label);
}

function assertDigestOrNull(value, label) {
  if (value === null) return;
  assertDigest(value, label);
}

function assertNullableString(value, label) {
  if (value === null) return;
  assertString(value, label);
}

function validateArtifact(artifact, index) {
  assertPlainObject(artifact, `artifacts[${index}]`);
  assertString(artifact.role, `artifacts[${index}].role`);
  if (!ARTIFACT_ROLE_SET.has(artifact.role)) {
    throw new Error(
      `program bundle: artifacts[${index}].role must be one of ${[...ARTIFACT_ROLE_SET].join(', ')}.`
    );
  }
  assertString(artifact.path, `artifacts[${index}].path`);
  assertDigest(artifact.hash, `artifacts[${index}].hash`);
  assertNullableFiniteNumber(artifact.sizeBytes, `artifacts[${index}].sizeBytes`);
}

function validateHost(host) {
  assertPlainObject(host, 'host');
  if (host.schema !== PROGRAM_BUNDLE_HOST_SCHEMA_ID) {
    throw new Error(`program bundle: host.schema must be "${PROGRAM_BUNDLE_HOST_SCHEMA_ID}".`);
  }
  if (host.jsSubset !== PROGRAM_BUNDLE_HOST_JS_SUBSET) {
    throw new Error(`program bundle: host.jsSubset must be "${PROGRAM_BUNDLE_HOST_JS_SUBSET}".`);
  }

  assertNonEmptyArray(host.entrypoints, 'host.entrypoints');
  for (let index = 0; index < host.entrypoints.length; index += 1) {
    const entrypoint = host.entrypoints[index];
    assertPlainObject(entrypoint, `host.entrypoints[${index}]`);
    assertString(entrypoint.id, `host.entrypoints[${index}].id`);
    assertString(entrypoint.module, `host.entrypoints[${index}].module`);
    assertString(entrypoint.export, `host.entrypoints[${index}].export`);
    assertString(entrypoint.role, `host.entrypoints[${index}].role`);
    if (entrypoint.sourceHash !== undefined) {
      assertDigest(entrypoint.sourceHash, `host.entrypoints[${index}].sourceHash`);
    }
    if (entrypoint.validation !== undefined) {
      assertPlainObject(entrypoint.validation, `host.entrypoints[${index}].validation`);
      if (entrypoint.validation.dynamicImport !== 'none-detected') {
        throw new Error(`program bundle: host.entrypoints[${index}].validation.dynamicImport must be "none-detected".`);
      }
      if (entrypoint.validation.dom !== 'none-detected') {
        throw new Error(`program bundle: host.entrypoints[${index}].validation.dom must be "none-detected".`);
      }
    }
  }

  assertPlainObject(host.constraints, 'host.constraints');
  if (host.constraints.dynamicImport !== 'disallowed') {
    throw new Error('program bundle: host.constraints.dynamicImport must be "disallowed".');
  }
  if (host.constraints.dom !== 'disallowed-in-model-path') {
    throw new Error('program bundle: host.constraints.dom must be "disallowed-in-model-path".');
  }
  if (host.constraints.filesystem !== 'declared-artifacts-only') {
    throw new Error('program bundle: host.constraints.filesystem must be "declared-artifacts-only".');
  }
  if (host.constraints.network !== 'declared-artifacts-only') {
    throw new Error('program bundle: host.constraints.network must be "declared-artifacts-only".');
  }
}

function validateWgslModules(modules) {
  assertNonEmptyArray(modules, 'wgslModules');
  const ids = new Set();
  const fileEntries = new Map();
  for (let index = 0; index < modules.length; index += 1) {
    const module = modules[index];
    assertPlainObject(module, `wgslModules[${index}]`);
    assertString(module.id, `wgslModules[${index}].id`);
    assertString(module.file, `wgslModules[${index}].file`);
    assertString(module.entry, `wgslModules[${index}].entry`);
    assertDigest(module.digest, `wgslModules[${index}].digest`);
    assertNullablePlainObject(module.metadata ?? null, `wgslModules[${index}].metadata`);
    if (module.metadata) {
      assertDigest(module.metadata.sourceMetadataHash, `wgslModules[${index}].metadata.sourceMetadataHash`);
      assertArray(module.metadata.bindings, `wgslModules[${index}].metadata.bindings`);
      assertArray(module.metadata.overrides, `wgslModules[${index}].metadata.overrides`);
      assertArray(module.metadata.workgroupSize, `wgslModules[${index}].metadata.workgroupSize`);
      for (let bindingIndex = 0; bindingIndex < module.metadata.bindings.length; bindingIndex += 1) {
        const binding = module.metadata.bindings[bindingIndex];
        assertPlainObject(binding, `wgslModules[${index}].metadata.bindings[${bindingIndex}]`);
        if (!Number.isInteger(binding.group) || binding.group < 0) {
          throw new Error(`program bundle: wgslModules[${index}].metadata.bindings[${bindingIndex}].group must be a non-negative integer.`);
        }
        if (!Number.isInteger(binding.binding) || binding.binding < 0) {
          throw new Error(`program bundle: wgslModules[${index}].metadata.bindings[${bindingIndex}].binding must be a non-negative integer.`);
        }
        assertString(binding.name, `wgslModules[${index}].metadata.bindings[${bindingIndex}].name`);
      }
    }
    if (ids.has(module.id)) {
      throw new Error(`program bundle: duplicate wgslModules id "${module.id}".`);
    }
    ids.add(module.id);

    const fileKey = `${module.file}#${module.entry}`;
    const previousDigest = fileEntries.get(fileKey);
    if (previousDigest && previousDigest !== module.digest) {
      throw new Error(
        `program bundle: WGSL module "${fileKey}" has conflicting digests.`
      );
    }
    fileEntries.set(fileKey, module.digest);
  }
  return ids;
}

function validateExecutionSteps(steps, moduleIds) {
  assertNonEmptyArray(steps, 'execution.steps');
  const ids = new Set();
  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    assertPlainObject(step, `execution.steps[${index}]`);
    assertString(step.id, `execution.steps[${index}].id`);
    if (ids.has(step.id)) {
      throw new Error(`program bundle: duplicate execution step id "${step.id}".`);
    }
    ids.add(step.id);
    assertString(step.op, `execution.steps[${index}].op`);
    assertString(step.phase, `execution.steps[${index}].phase`);
    assertString(step.section, `execution.steps[${index}].section`);
    assertString(step.src, `execution.steps[${index}].src`);
    assertString(step.dst, `execution.steps[${index}].dst`);
    assertString(step.kernelId, `execution.steps[${index}].kernelId`);
    if (!moduleIds.has(step.kernelId)) {
      throw new Error(`program bundle: execution step "${step.id}" references missing kernel "${step.kernelId}".`);
    }
    assertDigest(step.kernelDigest, `execution.steps[${index}].kernelDigest`);
    assertPlainObject(step.dispatch, `execution.steps[${index}].dispatch`);
    assertString(step.dispatch.phase, `execution.steps[${index}].dispatch.phase`);
    assertString(step.dispatch.workgroups, `execution.steps[${index}].dispatch.workgroups`);
    assertArray(step.dispatch.bindings, `execution.steps[${index}].dispatch.bindings`);
  }
}

function validateKernelClosure(kernelClosure, moduleIds) {
  assertPlainObject(kernelClosure, 'execution.kernelClosure');
  assertNonEmptyArray(kernelClosure.declaredKernelIds, 'execution.kernelClosure.declaredKernelIds');
  assertNonEmptyArray(kernelClosure.reachableKernelIds, 'execution.kernelClosure.reachableKernelIds');
  assertArray(kernelClosure.excludedKernelIds, 'execution.kernelClosure.excludedKernelIds');
  assertArray(kernelClosure.undeclaredKernelRefs, 'execution.kernelClosure.undeclaredKernelRefs');
  if (kernelClosure.undeclaredKernelRefs.length > 0) {
    throw new Error(
      `program bundle: execution.kernelClosure.undeclaredKernelRefs must be empty; got ${kernelClosure.undeclaredKernelRefs.join(', ')}.`
    );
  }
  if (!Number.isInteger(kernelClosure.expandedStepCount) || kernelClosure.expandedStepCount < 1) {
    throw new Error('program bundle: execution.kernelClosure.expandedStepCount must be a positive integer.');
  }
  for (const id of kernelClosure.reachableKernelIds) {
    assertString(id, `execution.kernelClosure.reachableKernelIds entry`);
    if (!moduleIds.has(id)) {
      throw new Error(
        `program bundle: reachable kernel "${id}" is missing from wgslModules.`
      );
    }
  }
}

function validateCaptureProfile(captureProfile) {
  assertPlainObject(captureProfile, 'captureProfile');
  if (captureProfile.schema !== PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID) {
    throw new Error(
      `program bundle: captureProfile.schema must be "${PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID}".`
    );
  }
  if (captureProfile.deterministic !== true) {
    throw new Error('program bundle: captureProfile.deterministic must be true.');
  }
  assertNonEmptyArray(captureProfile.phases, 'captureProfile.phases');
  assertNonEmptyArray(captureProfile.surfaces, 'captureProfile.surfaces');
  assertDigest(captureProfile.captureHash, 'captureProfile.captureHash');
  assertPlainObject(captureProfile.adapter, 'captureProfile.adapter');
  assertString(captureProfile.adapter.source, 'captureProfile.adapter.source');
  assertPlainObject(captureProfile.hashPolicy, 'captureProfile.hashPolicy');
  assertString(captureProfile.hashPolicy.graph, 'captureProfile.hashPolicy.graph');
  assertString(captureProfile.hashPolicy.dispatch, 'captureProfile.hashPolicy.dispatch');
}

function validateReferenceTranscript(referenceTranscript, expectedGraphHash) {
  assertPlainObject(referenceTranscript, 'referenceTranscript');
  if (referenceTranscript.schema !== PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID) {
    throw new Error(
      `program bundle: referenceTranscript.schema must be "${PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID}".`
    );
  }
  if (referenceTranscript.executionGraphHash !== expectedGraphHash) {
    throw new Error('program bundle: referenceTranscript.executionGraphHash must match sources.executionGraph.hash.');
  }

  assertPlainObject(referenceTranscript.source, 'referenceTranscript.source');
  assertString(referenceTranscript.source.kind, 'referenceTranscript.source.kind');
  assertString(referenceTranscript.source.path, 'referenceTranscript.source.path');
  assertDigest(referenceTranscript.source.hash, 'referenceTranscript.source.hash');

  assertPlainObject(referenceTranscript.prompt, 'referenceTranscript.prompt');
  assertDigest(referenceTranscript.prompt.hash, 'referenceTranscript.prompt.hash');
  assertString(referenceTranscript.prompt.identity, 'referenceTranscript.prompt.identity');
  assertDigestOrNull(referenceTranscript.prompt.tokenIdsHash ?? null, 'referenceTranscript.prompt.tokenIdsHash');
  assertNullableFiniteNumber(referenceTranscript.prompt.tokenCount ?? null, 'referenceTranscript.prompt.tokenCount');

  assertPlainObject(referenceTranscript.output, 'referenceTranscript.output');
  assertDigest(referenceTranscript.output.textHash, 'referenceTranscript.output.textHash');
  assertString(referenceTranscript.output.stopReason, 'referenceTranscript.output.stopReason');
  assertNullableFiniteNumber(referenceTranscript.output.stopTokenId ?? null, 'referenceTranscript.output.stopTokenId');
  if (!Number.isInteger(referenceTranscript.output.tokensGenerated) || referenceTranscript.output.tokensGenerated < 0) {
    throw new Error('program bundle: referenceTranscript.output.tokensGenerated must be a non-negative integer.');
  }

  assertPlainObject(referenceTranscript.tokens, 'referenceTranscript.tokens');
  assertDigest(referenceTranscript.tokens.generatedTokenIdsHash, 'referenceTranscript.tokens.generatedTokenIdsHash');
  assertDigest(referenceTranscript.tokens.generatedTextHash, 'referenceTranscript.tokens.generatedTextHash');
  assertArray(referenceTranscript.tokens.perStep, 'referenceTranscript.tokens.perStep');
  if (referenceTranscript.tokens.perStep.length !== referenceTranscript.output.tokensGenerated) {
    throw new Error('program bundle: referenceTranscript.tokens.perStep length must match output.tokensGenerated.');
  }
  for (let index = 0; index < referenceTranscript.tokens.perStep.length; index += 1) {
    const step = referenceTranscript.tokens.perStep[index];
    assertPlainObject(step, `referenceTranscript.tokens.perStep[${index}]`);
    if (!Number.isInteger(step.index) || step.index !== index) {
      throw new Error(`program bundle: referenceTranscript.tokens.perStep[${index}].index must equal ${index}.`);
    }
    assertDigest(step.tokenHash, `referenceTranscript.tokens.perStep[${index}].tokenHash`);
  }

  assertPlainObject(referenceTranscript.phase, 'referenceTranscript.phase');
  for (const field of ['prefillMs', 'decodeMs', 'prefillTokens', 'decodeTokens']) {
    assertNullableFiniteNumber(referenceTranscript.phase[field], `referenceTranscript.phase.${field}`);
  }
  assertPlainObject(referenceTranscript.kvCache, 'referenceTranscript.kvCache');
  assertString(referenceTranscript.kvCache.mode, 'referenceTranscript.kvCache.mode');
  assertDigest(referenceTranscript.kvCache.stateHash, 'referenceTranscript.kvCache.stateHash');
  assertNullableString(referenceTranscript.kvCache.layout, 'referenceTranscript.kvCache.layout');
  assertNullableString(referenceTranscript.kvCache.kvDtype ?? null, 'referenceTranscript.kvCache.kvDtype');
  assertNullableFiniteNumber(referenceTranscript.kvCache.seqLen, 'referenceTranscript.kvCache.seqLen');
  assertNullableFiniteNumber(referenceTranscript.kvCache.maxSeqLen, 'referenceTranscript.kvCache.maxSeqLen');
  assertNullableFiniteNumber(referenceTranscript.kvCache.usedBytes, 'referenceTranscript.kvCache.usedBytes');
  assertNullableFiniteNumber(referenceTranscript.kvCache.allocatedBytes, 'referenceTranscript.kvCache.allocatedBytes');
  if (referenceTranscript.kvCache.byteDigest !== undefined && referenceTranscript.kvCache.byteDigest !== null) {
    assertDigest(referenceTranscript.kvCache.byteDigest, 'referenceTranscript.kvCache.byteDigest');
  }
  if (referenceTranscript.kvCache.byteDigestMode !== undefined && referenceTranscript.kvCache.byteDigestMode !== null) {
    assertString(referenceTranscript.kvCache.byteDigestMode, 'referenceTranscript.kvCache.byteDigestMode');
  }
  if (referenceTranscript.kvCache.byteDigests !== undefined && referenceTranscript.kvCache.byteDigests !== null) {
    assertArray(referenceTranscript.kvCache.byteDigests, 'referenceTranscript.kvCache.byteDigests');
    for (let index = 0; index < referenceTranscript.kvCache.byteDigests.length; index += 1) {
      const entry = referenceTranscript.kvCache.byteDigests[index];
      assertPlainObject(entry, `referenceTranscript.kvCache.byteDigests[${index}]`);
      assertDigest(entry.keyDigest, `referenceTranscript.kvCache.byteDigests[${index}].keyDigest`);
      assertDigest(entry.valueDigest, `referenceTranscript.kvCache.byteDigests[${index}].valueDigest`);
      assertNullableFiniteNumber(entry.seqLen, `referenceTranscript.kvCache.byteDigests[${index}].seqLen`);
      assertNullableFiniteNumber(entry.keyBytes, `referenceTranscript.kvCache.byteDigests[${index}].keyBytes`);
      assertNullableFiniteNumber(entry.valueBytes, `referenceTranscript.kvCache.byteDigests[${index}].valueBytes`);
    }
  }

  assertPlainObject(referenceTranscript.logits, 'referenceTranscript.logits');
  if (referenceTranscript.logits.mode !== 'not-captured' && referenceTranscript.logits.mode !== 'sha256-per-step') {
    throw new Error('program bundle: referenceTranscript.logits.mode must be "not-captured" or "sha256-per-step".');
  }
  if (referenceTranscript.logits.perStepDigests !== null) {
    assertArray(referenceTranscript.logits.perStepDigests, 'referenceTranscript.logits.perStepDigests');
    for (const digest of referenceTranscript.logits.perStepDigests) {
      assertDigest(digest, 'referenceTranscript.logits.perStepDigests entry');
    }
  }

  assertPlainObject(referenceTranscript.tolerance, 'referenceTranscript.tolerance');
  assertString(referenceTranscript.tolerance.tokenPolicy, 'referenceTranscript.tolerance.tokenPolicy');
  assertString(referenceTranscript.tolerance.logitsPolicy, 'referenceTranscript.tolerance.logitsPolicy');
}

export function validateProgramBundle(bundle) {
  assertPlainObject(bundle, 'bundle');
  if (bundle.schema !== PROGRAM_BUNDLE_SCHEMA_ID) {
    throw new Error(`program bundle: schema must be "${PROGRAM_BUNDLE_SCHEMA_ID}".`);
  }
  if (bundle.schemaVersion !== PROGRAM_BUNDLE_SCHEMA_VERSION) {
    throw new Error(`program bundle: schemaVersion must be ${PROGRAM_BUNDLE_SCHEMA_VERSION}.`);
  }
  assertString(bundle.bundleId, 'bundleId');
  assertString(bundle.modelId, 'modelId');
  assertString(bundle.createdAtUtc, 'createdAtUtc');

  assertPlainObject(bundle.sources, 'sources');
  assertPlainObject(bundle.sources.manifest, 'sources.manifest');
  assertString(bundle.sources.manifest.path, 'sources.manifest.path');
  assertDigest(bundle.sources.manifest.hash, 'sources.manifest.hash');
  assertPlainObject(bundle.sources.executionGraph, 'sources.executionGraph');
  assertDigest(bundle.sources.executionGraph.hash, 'sources.executionGraph.hash');
  assertDigest(bundle.sources.executionGraph.expandedStepHash, 'sources.executionGraph.expandedStepHash');
  assertDigest(bundle.sources.weightSetHash, 'sources.weightSetHash');
  assertDigest(bundle.sources.artifactSetHash, 'sources.artifactSetHash');

  validateHost(bundle.host);
  const moduleIds = validateWgslModules(bundle.wgslModules);

  assertPlainObject(bundle.execution, 'execution');
  if (bundle.execution.graphHash !== bundle.sources.executionGraph.hash) {
    throw new Error('program bundle: execution.graphHash must match sources.executionGraph.hash.');
  }
  assertDigest(bundle.execution.stepMetadataHash, 'execution.stepMetadataHash');
  validateKernelClosure(bundle.execution.kernelClosure, moduleIds);
  validateExecutionSteps(bundle.execution.steps, moduleIds);
  validateCaptureProfile(bundle.captureProfile);

  assertNonEmptyArray(bundle.artifacts, 'artifacts');
  const artifactKeys = new Set();
  for (let index = 0; index < bundle.artifacts.length; index += 1) {
    const artifact = bundle.artifacts[index];
    validateArtifact(artifact, index);
    const key = `${artifact.role}:${artifact.path}`;
    if (artifactKeys.has(key)) {
      throw new Error(`program bundle: duplicate artifact "${key}".`);
    }
    artifactKeys.add(key);
  }

  validateReferenceTranscript(bundle.referenceTranscript, bundle.sources.executionGraph.hash);
  return bundle;
}
