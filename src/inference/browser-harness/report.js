import { computeSampleStats } from '../../debug/stats.js';
import {
  resolveDeviceInfo,
  resolveKernelPathForModel,
} from './model-resolution.js';
import {
  normalizeDecodeRecordOpLabels,
  normalizeUniformCacheStats,
  buildDecodeRecordTopOps,
  buildDecodeRecordTopOpGroups,
  isCoherentOutput,
} from './text-evidence.js';
import { sha256Hex } from '../../formats/sha256.js';
import {
  buildTokenCostLedger,
  isExecutionObservationRequested,
} from '../../tooling/execution-cost-ledger.js';
import { createUnsupportedWorkloadError, hashStableJson, resolveDecodeCadence, resolveDispatchSuite, resolveExecutionGraphHash, resolveHarnessContext, resolveHarnessMode, resolveWorkload } from './request.js';

export function resolvePipelineLoadTimings(pipeline) {
  if (!pipeline || typeof pipeline.getStats !== 'function') {
    return { loadTiming: null, pipelineLoadTiming: null };
  }
  try {
    const stats = pipeline.getStats() ?? {};
    return {
      loadTiming: stats.loadTiming ?? null,
      pipelineLoadTiming: stats.pipelineLoadTiming ?? null,
    };
  } catch {
    return { loadTiming: null, pipelineLoadTiming: null };
  }
}

export function serializeSequenceProbeRows(tokenEmbeddings, embeddingDim, positions) {
  if (!ArrayBuffer.isView(tokenEmbeddings) || !Number.isInteger(embeddingDim) || embeddingDim <= 0) {
    return [];
  }
  const uniquePositions = Array.isArray(positions) ? [...new Set(positions)] : [];
  return uniquePositions.map((position) => {
    const start = position * embeddingDim;
    const end = start + embeddingDim;
    if (!Number.isInteger(position) || position < 0 || end > tokenEmbeddings.length) {
      throw new Error(`Sequence probe position ${position} is outside the returned token embedding range.`);
    }
    return {
      position,
      values: Array.from(tokenEmbeddings.subarray(start, end)),
    };
  });
}

export function getNestedSampleValue(sample, key) {
  if (!sample || typeof sample !== 'object' || typeof key !== 'string' || key.length === 0) {
    return null;
  }
  let current = sample;
  for (const part of key.split('.')) {
    if (!current || typeof current !== 'object') {
      return null;
    }
    current = current[part];
  }
  return current;
}

export function getNestedPhaseValue(sample, key) {
  const value = getNestedSampleValue(sample, key);
  return Number.isFinite(value) ? value : null;
}

export function summarizeTimingPhaseSamples(samples, keys) {
  const rows = Array.isArray(samples) ? samples.filter((sample) => sample && typeof sample === 'object') : [];
  const summary = {};
  for (const key of keys) {
    const values = rows
      .map((sample) => getNestedPhaseValue(sample, key))
      .filter((value) => Number.isFinite(value));
    summary[key] = computeSampleStats(values);
  }
  return summary;
}

export function summarizePrefillRecordOps(samples) {
  const rows = Array.isArray(samples) ? samples.filter((sample) => sample && typeof sample === 'object') : [];
  const labelSums = {};
  let labelSampleCount = 0;
  for (const sample of rows) {
    const labelCounts = normalizeDecodeRecordOpLabels(getNestedSampleValue(sample, 'prefillRecordOpLabels'));
    if (!labelCounts) {
      continue;
    }
    labelSampleCount += 1;
    for (const [label, count] of Object.entries(labelCounts)) {
      labelSums[label] = (labelSums[label] ?? 0) + count;
    }
  }
  if (labelSampleCount === 0) {
    return {};
  }
  const meanLabels = {};
  for (const [label, count] of Object.entries(labelSums)) {
    meanLabels[label] = count / labelSampleCount;
  }
  const prefillRecordOps = summarizeTimingPhaseSamples(rows, ['prefillRecordOps']).prefillRecordOps;
  return {
    prefillRecordUniqueOpLabels: Object.keys(meanLabels).length,
    prefillRecordTopOps: buildDecodeRecordTopOps(meanLabels, prefillRecordOps?.mean),
    prefillRecordTopOpGroups: buildDecodeRecordTopOpGroups(meanLabels, prefillRecordOps?.mean),
  };
}

export function resolveArtifactHash(manifest) {
  const declaredDigest = manifest?.artifactIdentity?.artifactDigest;
  if (typeof declaredDigest === 'string' && declaredDigest.trim()) {
    return declaredDigest;
  }
  if (!Array.isArray(manifest?.shards) || manifest.shards.length === 0) {
    return null;
  }
  return hashStableJson({
    modelId: manifest.modelId ?? null,
    artifactIdentity: manifest.artifactIdentity ?? null,
    hashAlgorithm: manifest.hashAlgorithm ?? null,
    totalSize: manifest.totalSize ?? null,
    shards: manifest.shards.map((shard) => ({
      index: shard?.index ?? null,
      filename: shard?.filename ?? null,
      size: shard?.size ?? null,
      hash: shard?.hash ?? null,
      blake3: shard?.blake3 ?? null,
      offset: shard?.offset ?? null,
    })),
    tokenizer: manifest.tokenizer ?? null,
  });
}

export function resolveWrapperHash(manifest, metrics) {
  const execution = manifest?.inference?.execution;
  if (!execution || typeof execution !== 'object') {
    return null;
  }
  return hashStableJson({
    wrapperContract: 'doppler.text-execution-wrapper/v1',
    kernelPathId: metrics?.kernelPathId ?? null,
    kernelPathSource: metrics?.kernelPathSource ?? null,
    executionPlan: metrics?.executionPlan ?? null,
    preLayer: execution.preLayer ?? null,
    prefill: execution.prefill ?? null,
    decode: execution.decode ?? null,
    postLayer: execution.postLayer ?? null,
    policies: execution.policies ?? null,
  });
}

export function attachExecutionCostLedger(metrics, runtimeConfig, manifest, options = {}) {
  if (options.force !== true && !isExecutionObservationRequested(runtimeConfig)) {
    return metrics;
  }
  const device = resolveDeviceInfo();
  const browser = {
    userAgent: typeof navigator !== 'undefined' ? (navigator.userAgent || null) : null,
    platform: typeof navigator !== 'undefined' ? (navigator.platform || null) : null,
    vendor: typeof navigator !== 'undefined' ? (navigator.vendor || null) : null,
  };
  return {
    ...metrics,
    tokenCostLedger: buildTokenCostLedger({
      metrics,
      identity: {
        artifactDigest: resolveArtifactHash(manifest),
        manifestDigest: hashStableJson(manifest),
        executionGraphDigest: resolveExecutionGraphHash(manifest),
        runtimeConfigDigest: hashStableJson(runtimeConfig),
        kernelSetDigest: manifest?.inference?.execution?.kernels
          ? hashStableJson(manifest.inference.execution.kernels)
          : null,
        wrapperDigest: resolveWrapperHash(manifest, metrics),
      },
      device,
      browser,
    }),
  };
}

export function buildPerStepTokenProof(tokenIds) {
  return tokenIds.map((tokenId, index) => ({
    index,
    tokenId,
    tokenHash: hashStableJson({ index, tokenId }),
  }));
}

export function buildKvCacheTranscriptSeed(kvCache, byteProof = null) {
  const source = kvCache && typeof kvCache === 'object' ? kvCache : null;
  const proof = byteProof && typeof byteProof === 'object' ? byteProof : null;
  const seed = {
    mode: proof ? 'stats+sha256-layer-kv-bytes' : (source ? 'stats' : 'not-captured'),
    layout: typeof source?.layout === 'string' ? source.layout : null,
    kvDtype: typeof source?.kvDtype === 'string' ? source.kvDtype : null,
    seqLen: Number.isFinite(source?.seqLen) ? source.seqLen : null,
    maxSeqLen: Number.isFinite(source?.maxSeqLen) ? source.maxSeqLen : null,
    usedBytes: Number.isFinite(source?.usedBytes) ? source.usedBytes : null,
    allocatedBytes: Number.isFinite(source?.allocatedBytes) ? source.allocatedBytes : null,
    counters: source?.counters ?? null,
    byteDigestMode: typeof proof?.mode === 'string' ? proof.mode : null,
    byteDigest: typeof proof?.digest === 'string' ? proof.digest : null,
    byteDigests: Array.isArray(proof?.layers) ? proof.layers : null,
  };
  return {
    ...seed,
    stateHash: hashStableJson(seed),
  };
}

export function buildReferenceTranscriptSeed(run, context = {}) {
  const promptPayload = run.promptInput ?? run.prompt ?? null;
  const outputText = typeof run.output === 'string' ? run.output : '';
  const tokenIds = Array.isArray(run.tokenIds)
    ? run.tokenIds.map((value) => Number(value)).filter((value) => Number.isInteger(value))
    : [];
  const promptTokenIds = Array.isArray(run.promptTokenIds)
    ? run.promptTokenIds.map((value) => Number(value)).filter((value) => Number.isInteger(value))
    : null;
  const logitsDigests = Array.isArray(run.logitsDigests)
    ? run.logitsDigests
    : [];
  const hasCompleteLogitsDigests = logitsDigests.length === tokenIds.length && tokenIds.length > 0;
  const transcript = {
    schema: 'doppler.reference-transcript/v1',
    source: {
      kind: 'inline-browser-suite',
      path: 'inline',
      hash: 'sha256:' + '0'.repeat(64),
    },
    executionGraphHash: context.executionGraphHash ?? null,
    surface: context.surface ?? 'browser-webgpu', generationConfig: run.generationConfig ?? null,
    prompt: {
      identity: typeof run.prompt === 'string' && run.prompt.trim() ? run.prompt : 'promptInput',
      hash: hashStableJson(promptPayload),
      ids: promptTokenIds,
      tokenIdsHash: promptTokenIds ? hashStableJson(promptTokenIds) : null,
      tokenCount: promptTokenIds ? promptTokenIds.length : null,
    },
    output: {
      textHash: `sha256:${sha256Hex(outputText)}`,
      tokensGenerated: tokenIds.length,
      stopReason: typeof run.phase?.stopReason === 'string' ? run.phase.stopReason : 'unknown',
      stopTokenId: Number.isInteger(run.phase?.stopTokenId) ? run.phase.stopTokenId : null,
    },
    tokens: {
      ids: tokenIds,
      generatedTokenIdsHash: hashStableJson(tokenIds),
      generatedTextHash: `sha256:${sha256Hex(outputText)}`,
      preview: Array.isArray(run.tokenDiagnostics?.preview) ? run.tokenDiagnostics.preview : [],
      perStep: buildPerStepTokenProof(tokenIds),
      coverage: {
        mode: 'full-token-ids',
        omitted: 0,
      },
    },
    phase: {
      prefillMs: Number.isFinite(run.phase?.prefillMs) ? run.phase.prefillMs : null,
      decodeMs: Number.isFinite(run.phase?.decodeMs) ? run.phase.decodeMs : null,
      prefillTokens: Number.isFinite(run.phase?.prefillTokens) ? run.phase.prefillTokens : null,
      decodeTokens: Number.isFinite(run.phase?.decodeTokens) ? run.phase.decodeTokens : null,
    },
    kvCache: buildKvCacheTranscriptSeed(
      run.phase?.kvCache ?? context.kvCache ?? null,
      run.kvCacheByteProof ?? null
    ),
    logits: hasCompleteLogitsDigests ? {
      mode: 'sha256-per-step',
      perStepDigests: logitsDigests.map((entry) => entry.digest),
      steps: logitsDigests,
    } : {
      mode: 'not-captured',
      reason: logitsDigests.length > 0
        ? `Per-step logits digest count did not match generated token count (${logitsDigests.length}/${tokenIds.length}).`
        : 'Per-step logits digests were not requested for this browser harness run.',
      perStepDigests: null,
    },
    tolerance: {
      tokenPolicy: 'exact generated token IDs',
      logitsPolicy: hasCompleteLogitsDigests
        ? 'exact sha256 digest per generated step over finalized f32 logits before sampling'
        : 'not captured',
      kvPolicy: run.kvCacheByteProof
        ? 'exact sha256 digest over used KV cache bytes by layer/key/value'
        : 'metadata hash only; KV tensor bytes are not read back by default',
    },
  };
  return {
    ...transcript,
    source: {
      ...transcript.source,
      hash: hashStableJson({
        prompt: transcript.prompt,
        output: transcript.output,
        tokens: {
          generatedTokenIdsHash: transcript.tokens.generatedTokenIdsHash,
          generatedTextHash: transcript.tokens.generatedTextHash,
        },
        generationConfig: transcript.generationConfig, phase: transcript.phase,
        kvCache: transcript.kvCache,
        logits: transcript.logits,
      }),
    },
  };
}
