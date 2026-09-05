import fs from 'node:fs/promises';
import path from 'node:path';
import { PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID } from '../../config/schema/program-bundle.schema.js';
import { sha256Hex } from '../../formats/sha256.js';
import { stableSortObject } from '../../formats/stable-sort-object.js';
import { normalizeDigest, requirePlainObject } from './validation.js';
import { buildSequenceReferenceTranscript } from './sequence-reference.js';
import { buildRerankReferenceTranscript } from './rerank-reference.js';

function stableJson(value) {
  return JSON.stringify(stableSortObject(value)) ?? 'null';
}

export function hashStableJson(value) {
  return `sha256:${sha256Hex(stableJson(value))}`;
}

export function normalizeSlash(value) {
  return String(value || '').replace(/\\/g, '/');
}

function digestText(value) {
  return `sha256:${sha256Hex(String(value ?? ''))}`;
}

function safePackageName(value) {
  const normalized = String(value ?? '')
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  if (!normalized) {
    throw new Error('program bundle export: package source id must contain a portable filename character.');
  }
  return normalized;
}

export function createPackageSourceFile({ role, id, extension, source }) {
  const contents = String(source);
  const hash = digestText(contents);
  const digestPrefix = hash.slice('sha256:'.length, 'sha256:'.length + 16);
  const packagePath = `program/${role === 'wgsl-source' ? 'wgsl' : 'host'}/${safePackageName(id)}-${digestPrefix}.${extension}`;
  const artifact = {
    role,
    path: packagePath,
    hash,
    sizeBytes: Buffer.byteLength(contents),
  };
  return { ...artifact, contents, artifact };
}

export function toRepoRelativePath(filePath, repoRoot) {
  return normalizeSlash(path.relative(repoRoot, filePath));
}

export async function readTextFile(filePath, label) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch (error) {
    throw new Error(`program bundle export: failed to read ${label} at ${filePath}: ${error.message}`);
  }
}

export async function tryReadTextFile(filePath) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch {
    return null;
  }
}

export async function readJsonFile(filePath, label) {
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

export async function fileArtifact({ role, filePath, repoRoot, artifactPath = null }) {
  const raw = await readTextFile(filePath, role);
  const stat = await fs.stat(filePath);
  return {
    role,
    path: artifactPath ?? toRepoRelativePath(filePath, repoRoot),
    hash: `sha256:${sha256Hex(raw)}`,
    sizeBytes: stat.size,
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
  const byteProof = {
    byteDigestMode: typeof source?.byteDigestMode === 'string' ? source.byteDigestMode : null,
    byteDigest: normalizeNullableHash(source?.byteDigest, 'referenceTranscript.kvCache.byteDigest'),
    byteDigests: Array.isArray(source?.byteDigests) ? source.byteDigests : null,
  };
  const hasByteProof = byteProof.byteDigestMode || byteProof.byteDigest || byteProof.byteDigests;
  return {
    ...kvCache,
    ...(hasByteProof ? byteProof : {}),
    ...(hasByteProof && kvCache.mode === 'stats' ? { mode: 'stats+sha256-layer-kv-bytes' } : {}),
    stateHash: normalizeNullableHash(source?.stateHash, 'referenceTranscript.kvCache.stateHash')
      ?? hashStableJson({ ...kvCache, ...(hasByteProof ? byteProof : {}) }),
  };
}

function buildReferenceAdapterStamp(report) {
  const declaredSurface = report.surface === 'node' || report.surface === 'browser'
    ? `${report.surface}-webgpu`
    : (report.metrics?.referenceTranscript?.surface ?? report.env?.runtime ?? null);
  return {
    source: 'reference-report',
    surface: declaredSurface,
    deviceInfoHash: hashStableJson(report.deviceInfo ?? null),
    deviceInfo: report.deviceInfo ?? null,
  };
}

export async function buildReferenceTranscript(referenceReportPath, repoRoot, executionGraphHash) {
  const resolvedReportPath = path.resolve(referenceReportPath);
  const reportArtifact = await fileArtifact({
    role: 'reference-report',
    filePath: resolvedReportPath,
    repoRoot,
  });
  const { json: report } = await readJsonFile(resolvedReportPath, 'reference report');
  requirePlainObject(report, 'reference report');
  if (report.schema === 'doppler.rerankModelQualification.v1') {
    return buildRerankReferenceTranscript(report, reportArtifact, executionGraphHash);
  }
  if (report.schema === 'doppler.sequenceModelQualification.v1') {
    return buildSequenceReferenceTranscript(report, reportArtifact, executionGraphHash);
  }
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

  const adapter = buildReferenceAdapterStamp(report);
  return {
    artifact: reportArtifact,
    adapter,
    transcript: {
      schema: PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID,
      source: {
        kind: 'browser-report',
        path: reportArtifact.path,
        hash: reportArtifact.hash,
      },
      executionGraphHash,
      surface: adapter.surface,
      ...((transcriptSeed?.generationConfig ?? metrics.generationConfig) != null
        ? { generationConfig: transcriptSeed?.generationConfig ?? metrics.generationConfig }
        : {}),
      sourceParity: metrics.sourceParity ?? null,
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
