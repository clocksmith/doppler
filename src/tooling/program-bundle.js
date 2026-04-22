import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expandExecutionV1 } from '../config/schema/execution-v1.schema.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../config/kernels/kernel-ref-digests.js';
import {
  PROGRAM_BUNDLE_SCHEMA_ID,
  PROGRAM_BUNDLE_SCHEMA_VERSION,
  PROGRAM_BUNDLE_HOST_SCHEMA_ID,
  PROGRAM_BUNDLE_HOST_JS_SUBSET,
  PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID,
  PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID,
  validateProgramBundle,
} from '../config/schema/program-bundle.schema.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

const DEFAULT_HOST_ENTRYPOINTS = Object.freeze([
  Object.freeze({
    id: 'text-generation',
    module: 'src/tooling/program-bundle-host.js',
    export: 'createTextGenerationProgram',
    role: 'model-orchestration',
  }),
]);

function stableJson(value) {
  return JSON.stringify(stableSortObject(value)) ?? 'null';
}

function hashStableJson(value) {
  return `sha256:${sha256Hex(stableJson(value))}`;
}

function normalizeSlash(value) {
  return String(value || '').replace(/\\/g, '/');
}

function normalizeDigest(value, label) {
  const raw = String(value || '').trim().toLowerCase();
  const digest = raw.startsWith('sha256:') ? raw : `sha256:${raw}`;
  if (!/^sha256:[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`program bundle export: ${label} must be a sha256 digest.`);
  }
  return digest;
}

function digestText(value) {
  return `sha256:${sha256Hex(String(value ?? ''))}`;
}

function toRepoRelativePath(filePath, repoRoot) {
  return normalizeSlash(path.relative(repoRoot, filePath));
}

function requirePlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`program bundle export: ${label} must be a non-null object.`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`program bundle export: ${label} must be a non-empty string.`);
  }
  return value.trim();
}

async function readTextFile(filePath, label) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch (error) {
    throw new Error(`program bundle export: failed to read ${label} at ${filePath}: ${error.message}`);
  }
}

async function tryReadTextFile(filePath) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch {
    return null;
  }
}

async function readJsonFile(filePath, label) {
  const raw = await readTextFile(filePath, label);
  try {
    return {
      raw,
      json: JSON.parse(raw),
    };
  } catch (error) {
    throw new Error(`program bundle export: ${label} must contain valid JSON: ${error.message}`);
  }
}

async function fileArtifact({ role, filePath, repoRoot, artifactPath = null }) {
  const raw = await readTextFile(filePath, role);
  const stat = await fs.stat(filePath);
  return {
    role,
    path: artifactPath ?? toRepoRelativePath(filePath, repoRoot),
    hash: `sha256:${sha256Hex(raw)}`,
    sizeBytes: stat.size,
  };
}

function shardArtifact(shard, modelDir, repoRoot) {
  const filename = requireString(shard?.filename ?? shard?.path, 'shard filename');
  const shardPath = path.resolve(modelDir, filename);
  return {
    role: 'weight-shard',
    path: toRepoRelativePath(shardPath, repoRoot),
    hash: normalizeDigest(shard.hash ?? shard.sha256, `shard ${filename} hash`),
    sizeBytes: Number.isFinite(shard.size) ? Number(shard.size) : null,
  };
}

function collectKernelRefsFromEntries(entries, section, refs) {
  if (!Array.isArray(entries)) return;
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    if (Array.isArray(entry)) {
      if (typeof entry[1] === 'string' && entry[1].trim()) {
        refs.push({ id: entry[1].trim(), section, index });
      }
      continue;
    }
    if (entry && typeof entry === 'object' && Array.isArray(entry.steps)) {
      collectKernelRefsFromEntries(entry.steps, section, refs);
    }
  }
}

function collectReachableKernelRefs(execution) {
  const refs = [];
  collectKernelRefsFromEntries(execution.preLayer, 'preLayer', refs);
  collectKernelRefsFromEntries(execution.decode, 'decode', refs);
  collectKernelRefsFromEntries(execution.prefill, 'prefill', refs);
  collectKernelRefsFromEntries(execution.postLayer, 'postLayer', refs);
  return refs;
}

function countExpandedStepsByPhase(expandedSteps) {
  const phases = {
    prefill: 0,
    decode: 0,
    preLayer: 0,
    postLayer: 0,
  };
  for (const step of expandedSteps) {
    if (step.section === 'preLayer') {
      phases.preLayer += 1;
    } else if (step.section === 'postLayer') {
      phases.postLayer += 1;
    } else if (step.phase === 'prefill') {
      phases.prefill += 1;
    } else if (step.phase === 'decode') {
      phases.decode += 1;
    }
  }
  return phases;
}

function buildKernelIdLookup(execution, modules = []) {
  const lookup = new Map();
  const reachableIds = new Set(modules.map((module) => module.id));
  for (const [id, decl] of Object.entries(execution.kernels || {})) {
    const key = `${decl.kernel}#${decl.entry}#${normalizeDigest(decl.digest, `execution.kernels.${id}.digest`)}`;
    const current = lookup.get(key);
    if (!current || (!reachableIds.has(current) && reachableIds.has(id))) {
      lookup.set(key, id);
    }
  }
  return lookup;
}

function normalizeLayersForStep(layers) {
  if (layers === 'all') return 'all';
  return Array.isArray(layers) ? layers : [];
}

function buildExecutionStepMetadata(execution, expandedSteps, modules) {
  const kernelIdLookup = buildKernelIdLookup(execution, modules);
  const moduleById = new Map(modules.map((module) => [module.id, module]));
  const steps = expandedSteps.map((step, index) => {
    const digest = normalizeDigest(step.digest, `expandedSteps[${index}].digest`);
    const lookupKey = `${step.kernel}#${step.entry}#${digest}`;
    const kernelId = kernelIdLookup.get(lookupKey);
    if (!kernelId) {
      throw new Error(
        `program bundle export: expanded step ${index} ${step.op} cannot be mapped to a declared kernel id.`
      );
    }
    const module = moduleById.get(kernelId);
    const bindings = Array.isArray(module?.metadata?.bindings)
      ? module.metadata.bindings.map((binding) => ({
        group: binding.group,
        binding: binding.binding,
        name: binding.name,
        addressSpace: binding.addressSpace,
        access: binding.access,
      }))
      : [];
    return {
      id: `${step.section}_${step.phase}_${index}_${step.op}`,
      index,
      op: step.op,
      phase: step.phase,
      section: step.section,
      layers: normalizeLayersForStep(step.layers),
      src: typeof step.src === 'string' && step.src.trim() ? step.src : 'state',
      dst: typeof step.dst === 'string' && step.dst.trim() ? step.dst : 'state',
      kernelId,
      kernel: step.kernel,
      entry: step.entry,
      kernelDigest: digest,
      weights: step.weights ?? null,
      constants: step.constants ?? null,
      precision: step.precision ?? null,
      dispatch: {
        phase: step.phase,
        workgroups: `symbolic:${step.section}:${step.phase}:${step.op}`,
        bindings,
      },
    };
  });
  return {
    steps,
    stepMetadataHash: hashStableJson(steps),
  };
}

async function resolveKernelSourceDigest(kernel, entry, kernelSourceRoot) {
  const digestKey = `${kernel}#${entry}`;
  const registryDigest = KERNEL_REF_CONTENT_DIGESTS[digestKey];
  const kernelPath = path.resolve(kernelSourceRoot, kernel);
  const source = await tryReadTextFile(kernelPath);
  if (registryDigest) {
    return {
      digest: `sha256:${registryDigest}`,
      sourcePath: normalizeSlash(path.join(kernelSourceRoot, kernel)),
      sourceText: source,
      source: 'registry',
    };
  }

  if (source != null) {
    const normalizedSource = source.replace(/\r\n/g, '\n');
    return {
      digest: `sha256:${sha256Hex(`${normalizedSource}\n@@entry:${entry}`)}`,
      sourcePath: normalizeSlash(path.relative(process.cwd(), kernelPath)),
      sourceText: normalizedSource,
      source: 'file',
    };
  }

  return {
    digest: null,
    sourcePath: null,
    sourceText: null,
    source: 'unresolved',
  };
}

function parseWgslBindings(sourceText) {
  if (typeof sourceText !== 'string') return [];
  const bindings = [];
  const bindingRe = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]+)>)?\s+([A-Za-z_]\w*)/g;
  let match;
  while ((match = bindingRe.exec(sourceText)) !== null) {
    const address = typeof match[3] === 'string'
      ? match[3].split(',').map((part) => part.trim()).filter(Boolean)
      : [];
    bindings.push({
      group: Number(match[1]),
      binding: Number(match[2]),
      addressSpace: address[0] ?? null,
      access: address[1] ?? null,
      name: match[4],
    });
  }
  return bindings.sort((left, right) => (
    left.group - right.group
      || left.binding - right.binding
      || left.name.localeCompare(right.name)
  ));
}

function parseWgslOverrides(sourceText) {
  if (typeof sourceText !== 'string') return [];
  const overrides = [];
  const overrideRe = /\boverride\s+([A-Za-z_]\w*)(?:\s*:\s*([^=;]+))?(?:\s*=\s*([^;]+))?;/g;
  let match;
  while ((match = overrideRe.exec(sourceText)) !== null) {
    overrides.push({
      name: match[1],
      type: typeof match[2] === 'string' ? match[2].trim() : null,
      defaultValue: typeof match[3] === 'string' ? match[3].trim() : null,
    });
  }
  return overrides.sort((left, right) => left.name.localeCompare(right.name));
}

function parseWgslWorkgroupSize(sourceText) {
  if (typeof sourceText !== 'string') return [];
  const match = /@workgroup_size\(([^)]*)\)/.exec(sourceText);
  if (!match) return [];
  return match[1].split(',').map((part) => part.trim()).filter(Boolean);
}

function buildWgslMetadata(sourceText, entry) {
  const metadata = {
    entry,
    bindings: parseWgslBindings(sourceText),
    overrides: parseWgslOverrides(sourceText),
    workgroupSize: parseWgslWorkgroupSize(sourceText),
    requiresSubgroups: typeof sourceText === 'string' && /\b(subgroup|enable\s+subgroups)\b/.test(sourceText),
  };
  return {
    ...metadata,
    sourceMetadataHash: hashStableJson(metadata),
  };
}

async function buildWgslClosure(execution, expandedSteps, options) {
  const declaredKernelIds = Object.keys(execution.kernels || {}).sort();
  const reachableRefs = collectReachableKernelRefs(execution);
  const reachableKernelIds = [...new Set(reachableRefs.map((ref) => ref.id))].sort();
  const undeclaredKernelRefs = reachableRefs
    .filter((ref) => !execution.kernels?.[ref.id])
    .map((ref) => `${ref.section}[${ref.index}]:${ref.id}`);
  if (undeclaredKernelRefs.length > 0) {
    throw new Error(
      `program bundle export: execution graph references undeclared kernels: ${undeclaredKernelRefs.join(', ')}.`
    );
  }

  const kernelSourceRoot = path.resolve(options.repoRoot, options.kernelSourceRoot || 'src/gpu/kernels');
  const modules = [];
  for (const id of reachableKernelIds) {
    const decl = execution.kernels[id];
    const file = requireString(decl.kernel, `execution.kernels.${id}.kernel`);
    const entry = requireString(decl.entry, `execution.kernels.${id}.entry`);
    const declaredDigest = normalizeDigest(decl.digest, `execution.kernels.${id}.digest`);
    const sourceDigest = await resolveKernelSourceDigest(file, entry, kernelSourceRoot);
    if (sourceDigest.digest && sourceDigest.digest !== declaredDigest) {
      throw new Error(
        `program bundle export: kernel digest mismatch for ${file}#${entry}. ` +
        `execution declares ${declaredDigest}, source has ${sourceDigest.digest}.`
      );
    }
    modules.push({
      id,
      file,
      entry,
      digest: declaredDigest,
      sourcePath: sourceDigest.sourcePath
        ? toRepoRelativePath(path.resolve(sourceDigest.sourcePath), options.repoRoot)
        : null,
      reachable: true,
      metadata: buildWgslMetadata(sourceDigest.sourceText, entry),
    });
  }

  return {
    modules,
    kernelClosure: {
      declaredKernelIds,
      reachableKernelIds,
      excludedKernelIds: declaredKernelIds.filter((id) => !reachableKernelIds.includes(id)),
      undeclaredKernelRefs,
      expandedStepCount: expandedSteps.length,
      phases: countExpandedStepsByPhase(expandedSteps),
    },
  };
}

function scanHostEntrypointSource(source, label) {
  if (/\bimport\s*\(/.test(source)) {
    throw new Error(`program bundle export: host entrypoint ${label} uses dynamic import().`);
  }
  if (/\b(document|window|localStorage|sessionStorage|XMLHttpRequest)\b/.test(source)) {
    throw new Error(`program bundle export: host entrypoint ${label} references DOM-only globals.`);
  }
  return {
    dynamicImport: 'none-detected',
    dom: 'none-detected',
  };
}

async function buildHostContract(host = {}, repoRoot) {
  const rawEntrypoints = Array.isArray(host.entrypoints) && host.entrypoints.length > 0
    ? host.entrypoints
    : DEFAULT_HOST_ENTRYPOINTS.map((entry) => ({ ...entry }));
  const entrypoints = [];
  for (const entrypoint of rawEntrypoints) {
    const modulePath = requireString(entrypoint.module, 'host.entrypoint.module');
    const sourcePath = path.resolve(repoRoot, modulePath);
    const source = await readTextFile(sourcePath, `host entrypoint ${modulePath}`);
    entrypoints.push({
      ...entrypoint,
      sourceHash: digestText(source),
      validation: scanHostEntrypointSource(source, `${modulePath}#${entrypoint.export}`),
    });
  }
  return {
    schema: PROGRAM_BUNDLE_HOST_SCHEMA_ID,
    jsSubset: PROGRAM_BUNDLE_HOST_JS_SUBSET,
    entrypoints,
    constraints: {
      dynamicImport: 'disallowed',
      dom: 'disallowed-in-model-path',
      filesystem: 'declared-artifacts-only',
      network: 'declared-artifacts-only',
      ...(host.constraints && typeof host.constraints === 'object' ? host.constraints : {}),
    },
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
      : ['browser-webgpu'],
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

function extractGenerationPreview(report) {
  const preview = report?.metrics?.generationDiagnostics?.preview;
  return Array.isArray(preview) ? preview : [];
}

function resolvePromptPayload(report) {
  const metrics = report?.metrics || {};
  if (metrics.promptInput !== undefined) {
    return {
      identity: typeof metrics.prompt === 'string' && metrics.prompt.trim()
        ? metrics.prompt
        : 'metrics.promptInput',
      payload: metrics.promptInput,
    };
  }
  return {
    identity: typeof metrics.prompt === 'string' && metrics.prompt.trim()
      ? metrics.prompt
      : 'unknown-prompt',
    payload: metrics.prompt ?? null,
  };
}

function buildPerStepTokenProof(tokenIds) {
  return tokenIds.map((tokenId, index) => ({
    index,
    tokenId,
    tokenHash: hashStableJson({ index, tokenId }),
  }));
}

function normalizeNullableHash(value, label) {
  if (typeof value !== 'string' || !value.trim()) return null;
  return normalizeDigest(value, label);
}

function buildKvCacheTranscript(metrics, transcriptSeed) {
  const seedKv = transcriptSeed?.kvCache && typeof transcriptSeed.kvCache === 'object'
    ? transcriptSeed.kvCache
    : null;
  const metricKv = metrics.kvCache && typeof metrics.kvCache === 'object'
    ? metrics.kvCache
    : null;
  const source = seedKv ?? metricKv ?? null;
  const kvCache = {
    mode: source ? 'stats' : 'not-captured',
    layout: typeof source?.layout === 'string' ? source.layout : null,
    kvDtype: typeof source?.kvDtype === 'string' ? source.kvDtype : null,
    seqLen: Number.isFinite(source?.seqLen) ? source.seqLen : null,
    maxSeqLen: Number.isFinite(source?.maxSeqLen) ? source.maxSeqLen : null,
    usedBytes: Number.isFinite(source?.usedBytes)
      ? source.usedBytes
      : (Number.isFinite(source?.used) ? source.used : null),
    allocatedBytes: Number.isFinite(source?.allocatedBytes)
      ? source.allocatedBytes
      : (Number.isFinite(source?.allocated) ? source.allocated : null),
    counters: source?.counters && typeof source.counters === 'object' ? source.counters : null,
  };
  return {
    ...kvCache,
    stateHash: normalizeNullableHash(source?.stateHash, 'referenceTranscript.kvCache.stateHash')
      ?? hashStableJson(kvCache),
  };
}

function buildReferenceAdapterStamp(report) {
  return {
    source: 'reference-report',
    surface: report.env?.runtime ?? (report.mode ? `browser-${report.mode}` : null),
    deviceInfoHash: hashStableJson(report.deviceInfo ?? null),
    deviceInfo: report.deviceInfo ?? null,
  };
}

async function buildReferenceTranscript(referenceReportPath, repoRoot, executionGraphHash) {
  const resolvedReportPath = path.resolve(referenceReportPath);
  const reportArtifact = await fileArtifact({
    role: 'reference-report',
    filePath: resolvedReportPath,
    repoRoot,
  });
  const { json: report } = await readJsonFile(resolvedReportPath, 'reference report');
  requirePlainObject(report, 'reference report');
  const metrics = report.metrics || {};
  if (!metrics || typeof metrics !== 'object' || Array.isArray(metrics)) {
    throw new Error('program bundle export: reference report must include metrics for transcript identity.');
  }
  const transcriptSeed = metrics.referenceTranscript && typeof metrics.referenceTranscript === 'object'
    ? metrics.referenceTranscript
    : null;
  const prompt = transcriptSeed?.prompt?.hash && transcriptSeed?.prompt?.identity
    ? {
      identity: transcriptSeed.prompt.identity,
      payload: transcriptSeed.prompt,
      hash: transcriptSeed.prompt.hash,
    }
    : resolvePromptPayload(report);
  const outputText = typeof report.output === 'string'
    ? report.output
    : (typeof metrics.generatedText === 'string' ? metrics.generatedText : '');
  const preview = Array.isArray(transcriptSeed?.tokens?.preview)
    ? transcriptSeed.tokens.preview
    : extractGenerationPreview(report);
  const fullTokenIds = Array.isArray(transcriptSeed?.tokens?.ids)
    ? transcriptSeed.tokens.ids.map((value) => Number(value)).filter((value) => Number.isInteger(value))
    : null;
  const previewTokenIds = preview
    .map((entry) => Number(entry?.id))
    .filter((value) => Number.isInteger(value));
  const tokenIdsForHash = fullTokenIds ?? previewTokenIds;
  const tokensGenerated = Number.isInteger(transcriptSeed?.output?.tokensGenerated)
    ? transcriptSeed.output.tokensGenerated
    : Number.isInteger(metrics.tokensGenerated)
    ? metrics.tokensGenerated
    : (Number.isInteger(metrics.generationDiagnostics?.total) ? metrics.generationDiagnostics.total : previewTokenIds.length);
  if (prompt.identity === 'unknown-prompt') {
    throw new Error('program bundle export: reference report must include metrics.prompt or metrics.promptInput.');
  }
  if (tokensGenerated < 1 || tokenIdsForHash.length < 1 || !outputText) {
    throw new Error(
      'program bundle export: reference report must include generated output and token diagnostics.'
    );
  }

  return {
    artifact: reportArtifact,
    adapter: buildReferenceAdapterStamp(report),
    transcript: {
      schema: PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID,
      source: {
        kind: 'browser-report',
        path: reportArtifact.path,
        hash: reportArtifact.hash,
      },
      executionGraphHash,
      surface: report.env?.runtime ?? (report.mode ? `browser-${report.mode}` : null),
      prompt: {
        identity: prompt.identity,
        hash: transcriptSeed?.prompt?.hash
          ? normalizeDigest(transcriptSeed.prompt.hash, 'referenceTranscript.prompt.hash')
          : hashStableJson(prompt.payload),
        tokenIdsHash: normalizeNullableHash(transcriptSeed?.prompt?.tokenIdsHash, 'referenceTranscript.prompt.tokenIdsHash'),
        tokenCount: Number.isFinite(transcriptSeed?.prompt?.tokenCount)
          ? transcriptSeed.prompt.tokenCount
          : null,
      },
      output: {
        textHash: transcriptSeed?.output?.textHash
          ? normalizeDigest(transcriptSeed.output.textHash, 'referenceTranscript.output.textHash')
          : `sha256:${sha256Hex(outputText)}`,
        tokensGenerated,
        stopReason: typeof transcriptSeed?.output?.stopReason === 'string'
          ? transcriptSeed.output.stopReason
          : (typeof metrics.stopReason === 'string' ? metrics.stopReason : 'unknown'),
        stopTokenId: Number.isInteger(transcriptSeed?.output?.stopTokenId)
          ? transcriptSeed.output.stopTokenId
          : (Number.isInteger(metrics.stopTokenId) ? metrics.stopTokenId : null),
      },
      tokens: {
        generatedTokenIdsHash: transcriptSeed?.tokens?.generatedTokenIdsHash
          ? normalizeDigest(transcriptSeed.tokens.generatedTokenIdsHash, 'referenceTranscript.tokens.generatedTokenIdsHash')
          : hashStableJson(tokenIdsForHash),
        generatedTextHash: transcriptSeed?.tokens?.generatedTextHash
          ? normalizeDigest(transcriptSeed.tokens.generatedTextHash, 'referenceTranscript.tokens.generatedTextHash')
          : `sha256:${sha256Hex(outputText)}`,
        preview,
        perStep: Array.isArray(transcriptSeed?.tokens?.perStep) && transcriptSeed.tokens.perStep.length > 0
          ? transcriptSeed.tokens.perStep
          : buildPerStepTokenProof(tokenIdsForHash),
        ...(fullTokenIds ? { ids: fullTokenIds } : {}),
        coverage: transcriptSeed?.tokens?.coverage && typeof transcriptSeed.tokens.coverage === 'object'
          ? transcriptSeed.tokens.coverage
          : {
            mode: metrics.generationDiagnostics?.omitted > 0 ? 'preview' : 'complete-preview',
            omitted: Number.isInteger(metrics.generationDiagnostics?.omitted)
              ? metrics.generationDiagnostics.omitted
              : 0,
          },
      },
      phase: {
        prefillMs: Number.isFinite(transcriptSeed?.phase?.prefillMs)
          ? transcriptSeed.phase.prefillMs
          : (Number.isFinite(metrics.prefillMs) ? metrics.prefillMs : null),
        decodeMs: Number.isFinite(transcriptSeed?.phase?.decodeMs)
          ? transcriptSeed.phase.decodeMs
          : (Number.isFinite(metrics.decodeMs) ? metrics.decodeMs : null),
        prefillTokens: Number.isFinite(transcriptSeed?.phase?.prefillTokens)
          ? transcriptSeed.phase.prefillTokens
          : (Number.isFinite(metrics.prefillTokens) ? metrics.prefillTokens : null),
        decodeTokens: Number.isFinite(transcriptSeed?.phase?.decodeTokens)
          ? transcriptSeed.phase.decodeTokens
          : (Number.isFinite(metrics.decodeTokens) ? metrics.decodeTokens : null),
      },
      kvCache: buildKvCacheTranscript(metrics, transcriptSeed),
      logits: transcriptSeed?.logits && typeof transcriptSeed.logits === 'object' ? transcriptSeed.logits : {
        mode: 'not-captured',
        reason: 'Browser reports do not persist per-step logits digests yet.',
        perStepDigests: null,
      },
      tolerance: transcriptSeed?.tolerance && typeof transcriptSeed.tolerance === 'object' ? transcriptSeed.tolerance : {
        tokenPolicy: 'exact generated token IDs when a full-token transcript is present; preview IDs are diagnostic only',
        logitsPolicy: 'not captured in current browser report fixtures',
      },
    },
  };
}

async function collectArtifacts(options, manifest, manifestArtifact, referenceReportArtifact) {
  const artifacts = [manifestArtifact];
  if (Array.isArray(manifest.shards)) {
    for (const shard of manifest.shards) {
      artifacts.push(shardArtifact(shard, options.modelDir, options.repoRoot));
    }
  }

  const tokenizerFile = manifest.tokenizer?.file;
  if (typeof tokenizerFile === 'string' && tokenizerFile.trim()) {
    artifacts.push(await fileArtifact({
      role: 'tokenizer',
      filePath: path.resolve(options.modelDir, tokenizerFile),
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

export async function exportProgramBundle(options = {}) {
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
  const host = await buildHostContract(resolvedOptions.host, resolvedOptions.repoRoot);
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
  const artifacts = await collectArtifacts(
    resolvedOptions,
    manifest,
    manifestArtifact,
    reference.artifact
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

  return validateProgramBundle(bundle);
}

export async function writeProgramBundle(options = {}) {
  const outputPath = options.outputPath ? path.resolve(options.outputPath) : null;
  if (!outputPath) {
    throw new Error('program bundle export: outputPath is required.');
  }
  const bundle = await exportProgramBundle(options);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(bundle, null, 2)}\n`, 'utf8');
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

export async function checkProgramBundleFile(bundlePath) {
  const bundle = await loadProgramBundle(bundlePath);
  return {
    ok: true,
    path: path.resolve(bundlePath),
    modelId: bundle.modelId,
    bundleId: bundle.bundleId,
    artifactCount: bundle.artifacts.length,
    wgslModuleCount: bundle.wgslModules.length,
    executionGraphHash: bundle.sources.executionGraph.hash,
  };
}

export function createProgramBundleCliDefaults(metaUrl) {
  return {
    repoRoot: path.resolve(path.dirname(fileURLToPath(metaUrl)), '..'),
  };
}
