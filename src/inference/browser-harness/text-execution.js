import { isPlainObject } from '../../formats/plain-object.js';

import { observeInitialExecutionIdentity } from '../../config/initial-execution-identity.js';

import { isExecutionObservationRequested } from '../../tooling/execution-cost-ledger.js';

import {

  assertRerankLogitsVector,

  buildRerankScoreRecord,

  formatRerankPrompt,

  resolveRerankScoringConfig,

  scoreRerankDocument,

} from '../rerank.js';

import {

  DEFAULT_IMAGE_TRANSCRIPTION_PROMPT,

  DEFAULT_IMAGE_TRANSCRIPTION_SOFT_TOKEN_BUDGET,

  describePromptInput,

  formatEmbeddingSemanticText,

  resolveAutomaticGenerationDiagnostics,

  resolveEmbeddingSemanticFixtures,

  resolveEmbeddingSemanticStyle,

  resolveGenerationPromptInput,

  resolveInferenceImagePayload,

  resolveMaxTokens,

  resolvePrompt,

  resolveRerankInput,

  resolveRerankSemanticFixtures,

} from './text-input.js';

import {

  buildGenerationPhaseFromStats,

  captureKvCacheByteProof,

  cosineSimilarity,

  digestLogitsForTranscript,

  normalizeTokenIdArray,

  resolveGenerationUseChatTemplate,

  resolvePromptTokenIdsForTranscript,

  shouldCaptureReferenceKvBytes,

  shouldCaptureReferenceLogits,

  shouldEnableReferenceTranscriptDiagnostics,

  summarizeEmbeddingValues,

  summarizeGenerationTokens,

  summarizeRerankScores,

  top1Index,

} from './text-evidence.js';

function getPipelineTokenizer(pipeline) {
  const tokenizer = pipeline?.tokenizer;
  return tokenizer && typeof tokenizer.encode === 'function' ? tokenizer : null;
}

function longestCommonTokenPrefixLength(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return 0;
  }
  const first = rows[0];
  let length = Array.isArray(first) ? first.length : 0;
  for (let rowIndex = 1; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    if (!Array.isArray(row)) {
      return 0;
    }
    length = Math.min(length, row.length);
    for (let index = 0; index < length; index += 1) {
      if (row[index] !== first[index]) {
        length = index;
        break;
      }
    }
  }
  return length;
}

async function createRerankPrefixContext(pipeline, query, documents, config, options = {}) {
  const totalStart = performance.now();
  if (
    typeof pipeline?.prefillKVOnly !== 'function'
    || typeof pipeline?.prefillWithTokenLogits !== 'function'
    || typeof pipeline?.resetToSeqLen !== 'function'
  ) {
    return null;
  }
  const tokenizer = getPipelineTokenizer(pipeline);
  if (!tokenizer) {
    return null;
  }
  const prompts = documents.map((document) => formatRerankPrompt(query, document, config));
  const tokenRows = prompts.map((prompt, index) => normalizeTokenIdArray(
    tokenizer.encode(prompt),
    `rerank.prompt[${index}].tokenIds`
  ));
  const prefixLength = longestCommonTokenPrefixLength(tokenRows);
  if (prefixLength <= 0 || tokenRows.some((row) => row.length <= prefixLength)) {
    return null;
  }
  const prefixTokens = tokenRows[0].slice(0, prefixLength);
  pipeline.reset?.();
  const prefillStart = performance.now();
  const prefix = await pipeline.prefillKVOnly('', {
    useChatTemplate: false,
    inputIds: prefixTokens,
    benchmark: options.benchmark === true,
  });
  const prefillMs = performance.now() - prefillStart;
  return {
    prefix,
    seqLen: prefix.seqLen,
    prefixTokens,
    prompts,
    tokenRows,
    suffixRows: tokenRows.map((row) => row.slice(prefixLength)),
    phase: {
      totalMs: performance.now() - totalStart,
      prefillMs,
      prefixTokens: prefixTokens.length,
      documentCount: documents.length,
    },
  };
}

async function scoreRerankDocumentFromPrefix(pipeline, query, document, config, prefixContext, index, options = {}) {
  const suffixTokens = prefixContext.suffixRows[index];
  const totalStart = performance.now();
  try {
    const prefillCallStart = performance.now();
    const result = await pipeline.prefillWithTokenLogits(
      '',
      [config.trueTokenId, config.falseTokenId],
      {
        useChatTemplate: false,
        inputIds: suffixTokens,
        benchmark: options.benchmark === true,
      }
    );
    const prefillCallMs = performance.now() - prefillCallStart;
    const logits = assertRerankLogitsVector(result?.logits);
    const trueLogit = Number(logits[0]);
    const falseLogit = Number(logits[1]);
    return buildRerankScoreRecord(
      query,
      document,
      prefixContext.prompts[index],
      prefixContext.prefixTokens.length + suffixTokens.length,
      trueLogit,
      falseLogit,
      config,
      'prefix-selected-token-logits',
      {
        ...(isPlainObject(result?.phase) ? result.phase : {}),
        totalMs: performance.now() - totalStart,
        prefillCallMs,
        suffixTokens: suffixTokens.length,
      }
    );
  } finally {
    pipeline.resetToSeqLen(prefixContext.seqLen);
  }
}

export async function runRerank(pipeline, runtimeConfig, runOverrides = null) {
  const input = resolveRerankInput(runtimeConfig, runOverrides);
  const config = resolveRerankScoringConfig(pipeline);
  const start = performance.now();
  const scores = [];
  const prefixStart = performance.now();
  const prefixContext = await createRerankPrefixContext(pipeline, input.query, input.documents, config, {
    benchmark: runOverrides?.benchmark === true,
  });
  const prefixMs = performance.now() - prefixStart;
  for (let i = 0; i < input.documents.length; i++) {
    const scored = prefixContext
      ? await scoreRerankDocumentFromPrefix(
        pipeline,
        input.query,
        input.documents[i],
        config,
        prefixContext,
        i,
        {
          benchmark: runOverrides?.benchmark === true,
        }
      )
      : await scoreRerankDocument(
        pipeline,
        input.query,
        input.documents[i],
        config,
        {
          benchmark: runOverrides?.benchmark === true,
        }
      );
    scores.push({
      index: i,
      ...scored,
    });
  }
  const summary = summarizeRerankScores(scores);
  const durationMs = Math.max(1, performance.now() - start);
  const documentDurations = scores
    .map((entry) => Number(entry.phase?.totalMs))
    .filter((value) => Number.isFinite(value));
  const documentTotalMs = documentDurations.reduce((sum, value) => sum + value, 0);
  return {
    query: input.query,
    documents: input.documents,
    documentCount: input.documents.length,
    scores,
    ranking: summary.ranking,
    topDocument: summary.top
      ? {
        index: summary.top.index,
        document: summary.top.document,
        score: summary.top.score,
        probability: summary.top.probability,
      }
      : null,
    phase: {
      totalMs: durationMs,
      prefixMs,
      prefixApplied: prefixContext != null,
      prefixTokens: Number.isFinite(prefixContext?.prefixTokens?.length)
        ? prefixContext.prefixTokens.length
        : 0,
      prefix: prefixContext?.phase ?? null,
      documentCount: input.documents.length,
      documentTotalMs,
      maxDocumentMs: documentDurations.length > 0 ? Math.max(...documentDurations) : 0,
      avgDocumentMs: documentDurations.length > 0 ? documentTotalMs / documentDurations.length : 0,
      documents: scores.map((entry) => ({
        index: entry.index,
        scoringPath: entry.scoringPath,
        tokenCount: entry.tokenCount,
        phase: entry.phase ?? null,
      })),
    },
    durationMs,
  };
}

export async function runRerankSemanticChecks(pipeline, options = null) {
  const fixture = resolveRerankSemanticFixtures(pipeline?.runtimeConfig ?? {}, options);
  const config = resolveRerankScoringConfig(pipeline);
  const start = performance.now();
  const pairs = [];
  let pairPassed = 0;
  for (const testCase of fixture.cases) {
    const positive = await scoreRerankDocument(
      pipeline,
      testCase.query,
      testCase.positive,
      config
    );
    const negative = await scoreRerankDocument(
      pipeline,
      testCase.query,
      testCase.negative,
      config
    );
    const margin = positive.score - negative.score;
    const passed = Number.isFinite(margin) && margin > fixture.minScoreMargin;
    if (passed) pairPassed++;
    pairs.push({
      id: testCase.id,
      query: testCase.query,
      positive: testCase.positive,
      negative: testCase.negative,
      passed,
      positiveScore: Number(positive.score.toFixed(6)),
      negativeScore: Number(negative.score.toFixed(6)),
      positiveProbability: Number(positive.probability.toFixed(6)),
      negativeProbability: Number(negative.probability.toFixed(6)),
      margin: Number.isFinite(margin) ? Number(margin.toFixed(6)) : null,
    });
  }
  const pairAcc = pairs.length > 0 ? pairPassed / pairs.length : 0;
  const passed = pairAcc >= fixture.minPairAcc;
  return {
    passed,
    pairAcc,
    pairPassed,
    pairTotal: pairs.length,
    minPairAcc: Number(fixture.minPairAcc.toFixed(4)),
    minScoreMargin: Number(fixture.minScoreMargin.toFixed(4)),
    failedCaseIds: pairs.filter((item) => !item.passed).map((item) => item.id),
    pairs,
    durationMs: Math.max(1, performance.now() - start),
  };
}

async function embedStandaloneText(pipeline, text) {
  pipeline.reset?.();
  const result = await pipeline.embed(text);
  const embedding = result?.embedding;
  if (!embedding || !Number.isFinite(embedding.length) || embedding.length <= 0) {
    throw new Error('Semantic check embedding is missing.');
  }
  return embedding;
}

export async function runEmbeddingSemanticChecks(pipeline, options = null) {
  const config = resolveEmbeddingSemanticFixtures(
    pipeline?.runtimeConfig ?? {},
    options
  );
  const start = performance.now();
  const semanticStyle = resolveEmbeddingSemanticStyle(pipeline);
  const retrieval = [];
  let retrievalPassed = 0;

  for (const testCase of config.retrievalCases) {
    const formattedQuery = formatEmbeddingSemanticText(testCase.query, 'query', semanticStyle);
    const queryEmbedding = await embedStandaloneText(
      pipeline,
      formattedQuery
    );
    const docEmbeddings = [];
    const docs = [];
    for (const doc of testCase.docs) {
      const formattedDoc = formatEmbeddingSemanticText(doc, 'document', semanticStyle);
      docEmbeddings.push(await embedStandaloneText(
        pipeline,
        formattedDoc
      ));
      docs.push({
        text: doc,
        formattedText: formattedDoc,
      });
    }
    const sims = docEmbeddings.map((docEmbedding) => cosineSimilarity(queryEmbedding, docEmbedding));
    const topDoc = top1Index(sims);
    const passed = topDoc === testCase.expectedDoc;
    if (passed) retrievalPassed++;
    retrieval.push({
      id: testCase.id,
      query: testCase.query,
      formattedQuery,
      docs,
      passed,
      expectedDoc: testCase.expectedDoc,
      topDoc,
      sims: sims.map((v) => (Number.isFinite(v) ? Number(v.toFixed(6)) : null)),
    });
  }

  const pairs = [];
  let pairPassed = 0;
  for (const testCase of config.pairCases) {
    const formattedAnchor = formatEmbeddingSemanticText(testCase.anchor, 'query', semanticStyle);
    const anchor = await embedStandaloneText(
      pipeline,
      formattedAnchor
    );
    const formattedPositive = formatEmbeddingSemanticText(testCase.positive, 'query', semanticStyle);
    const positive = await embedStandaloneText(
      pipeline,
      formattedPositive
    );
    const formattedNegative = formatEmbeddingSemanticText(testCase.negative, 'query', semanticStyle);
    const negative = await embedStandaloneText(
      pipeline,
      formattedNegative
    );
    const simPos = cosineSimilarity(anchor, positive);
    const simNeg = cosineSimilarity(anchor, negative);
    const margin = simPos - simNeg;
    const passed = Number.isFinite(margin) && margin > config.pairMargin;
    if (passed) pairPassed++;
    pairs.push({
      id: testCase.id,
      anchor: testCase.anchor,
      formattedAnchor,
      positive: testCase.positive,
      formattedPositive,
      negative: testCase.negative,
      formattedNegative,
      passed,
      simPos: Number.isFinite(simPos) ? Number(simPos.toFixed(6)) : null,
      simNeg: Number.isFinite(simNeg) ? Number(simNeg.toFixed(6)) : null,
      margin: Number.isFinite(margin) ? Number(margin.toFixed(6)) : null,
    });
  }

  const lengthStability = [];
  let lengthStabilityPassed = 0;
  for (const testCase of config.lengthStabilityCases) {
    const shortEmb = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.short, 'document', semanticStyle)
    );
    const mediumEmb = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.medium, 'document', semanticStyle)
    );
    const longEmb = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.long, 'document', semanticStyle)
    );
    const simShortMedium = cosineSimilarity(shortEmb, mediumEmb);
    const simShortLong = cosineSimilarity(shortEmb, longEmb);
    const simMediumLong = cosineSimilarity(mediumEmb, longEmb);
    const minSim = Math.min(
      Number.isFinite(simShortMedium) ? simShortMedium : -1,
      Number.isFinite(simShortLong) ? simShortLong : -1,
      Number.isFinite(simMediumLong) ? simMediumLong : -1
    );
    const maxDrift = 1 - minSim;
    const passed = Number.isFinite(maxDrift) && maxDrift <= testCase.maxCosineDrift;
    if (passed) lengthStabilityPassed++;
    lengthStability.push({
      id: testCase.id,
      passed,
      simShortMedium: Number.isFinite(simShortMedium) ? Number(simShortMedium.toFixed(6)) : null,
      simShortLong: Number.isFinite(simShortLong) ? Number(simShortLong.toFixed(6)) : null,
      simMediumLong: Number.isFinite(simMediumLong) ? Number(simMediumLong.toFixed(6)) : null,
      maxDrift: Number.isFinite(maxDrift) ? Number(maxDrift.toFixed(6)) : null,
      maxCosineDrift: testCase.maxCosineDrift,
    });
  }

  let throughput = null;
  if (config.throughputCorpus.length > 0) {
    const corpusStart = performance.now();
    for (const text of config.throughputCorpus) {
      await embedStandaloneText(
        pipeline,
        formatEmbeddingSemanticText(text, 'document', semanticStyle)
      );
    }
    const corpusDurationMs = Math.max(1, performance.now() - corpusStart);
    throughput = {
      corpusSize: config.throughputCorpus.length,
      durationMs: Number(corpusDurationMs.toFixed(1)),
      docsPerSecond: Number((config.throughputCorpus.length / (corpusDurationMs / 1000)).toFixed(2)),
    };
  }

  const retrievalTop1Acc = retrieval.length > 0 ? retrievalPassed / retrieval.length : 0;
  const pairAcc = pairs.length > 0 ? pairPassed / pairs.length : 0;
  const lengthStabilityAcc = lengthStability.length > 0
    ? lengthStabilityPassed / lengthStability.length : 1;
  const passed = retrievalTop1Acc >= config.minRetrievalTop1Acc
    && pairAcc >= config.minPairAcc;
  const failedCaseIds = [
    ...retrieval.filter((item) => !item.passed).map((item) => `retrieval:${item.id}`),
    ...pairs.filter((item) => !item.passed).map((item) => `pair:${item.id}`),
    ...lengthStability.filter((item) => !item.passed).map((item) => `length:${item.id}`),
  ];

  return {
    passed,
    style: semanticStyle,
    retrievalTop1Acc,
    pairAcc,
    lengthStabilityAcc,
    retrievalPassed,
    retrievalTotal: retrieval.length,
    pairPassed,
    pairTotal: pairs.length,
    lengthStabilityPassed,
    lengthStabilityTotal: lengthStability.length,
    minRetrievalTop1Acc: Number(config.minRetrievalTop1Acc.toFixed(4)),
    minPairAcc: Number(config.minPairAcc.toFixed(4)),
    pairMarginThreshold: Number(config.pairMargin.toFixed(4)),
    failedCaseIds,
    retrieval,
    pairs,
    lengthStability,
    throughput,
    durationMs: Math.max(1, performance.now() - start),
  };
}

export async function runGeneration(pipeline, runtimeConfig, runOverrides = null) {
  const initialExecutionIdentity = pipeline?.resolvedRuntimeSession
    ? observeInitialExecutionIdentity(pipeline.resolvedRuntimeSession)
    : null;
  const tokens = [];
  const tokenIds = [];
  const tokenRecords = [];
  const logitsDigests = [];
  const promptInput = resolveGenerationPromptInput(runtimeConfig, runOverrides, pipeline);
  const promptLabel = describePromptInput(promptInput);
  const useChatTemplate = resolveGenerationUseChatTemplate(pipeline, runtimeConfig, runOverrides, promptInput);
  const promptTokenIds = resolvePromptTokenIdsForTranscript(pipeline, promptInput, useChatTemplate);
  const maxTokens = Number.isFinite(runOverrides?.maxTokens)
    ? Math.max(1, Math.floor(runOverrides.maxTokens))
    : resolveMaxTokens(runtimeConfig);
  const sampling = {
    ...(runtimeConfig.inference?.sampling || {}),
    ...(isPlainObject(runOverrides?.sampling) ? runOverrides.sampling : {}),
  };
  const seed = Number.isFinite(runOverrides?.seed)
    ? Math.max(0, Math.floor(runOverrides.seed))
    : Number.isFinite(sampling.seed)
      ? Math.max(0, Math.floor(sampling.seed))
      : null;
  const generationConfig = {
    maxTokens,
    temperature: sampling.temperature,
    topP: sampling.topP,
    topK: sampling.topK,
    repetitionPenalty: sampling.repetitionPenalty,
    repetitionPenaltyWindow: sampling.repetitionPenaltyWindow,
    greedyThreshold: sampling.greedyThreshold,
    suppressSpecialTokens: sampling.suppressSpecialTokens,
    suppressSpecialLikeTokens: sampling.suppressSpecialLikeTokens,
    suppressTokenIds: Array.isArray(sampling.suppressTokenIds) ? [...sampling.suppressTokenIds] : [],
    seed,
    useChatTemplate,
  };
  const debugProbes = runtimeConfig.shared?.debug?.probes || [];
  const executionObserverEnabled = isExecutionObservationRequested(runtimeConfig);
  const profile = executionObserverEnabled;
  const explicitDiagnosticsEnabled = runtimeConfig.shared?.harness?.mode === 'diagnose'
    || shouldEnableReferenceTranscriptDiagnostics(runOverrides, runtimeConfig);
  const disableCommandBatchingForDiagnostics = explicitDiagnosticsEnabled
    || (Array.isArray(debugProbes) && debugProbes.length > 0);
  const start = performance.now();
  const diagnostics = resolveAutomaticGenerationDiagnostics(runtimeConfig, runOverrides);
  const captureLogits = shouldCaptureReferenceLogits(runOverrides, runtimeConfig);

  for await (const tokenText of pipeline.generate(promptInput, {
    maxTokens,
    ...(Number.isFinite(seed) ? { seed } : {}),
    temperature: sampling.temperature,
    topP: sampling.topP,
    topK: sampling.topK,
    repetitionPenalty: sampling.repetitionPenalty,
    repetitionPenaltyWindow: sampling.repetitionPenaltyWindow,
    greedyThreshold: sampling.greedyThreshold,
    suppressSpecialTokens: sampling.suppressSpecialTokens,
    suppressSpecialLikeTokens: sampling.suppressSpecialLikeTokens,
    suppressTokenIds: generationConfig.suppressTokenIds,
    useChatTemplate,
    benchmark: runOverrides?.benchmark === true,
    profile,
    executionObserver: executionObserverEnabled,
    ...(disableCommandBatchingForDiagnostics ? { disableCommandBatching: true } : {}),
    diagnostics,
    ...(captureLogits ? {
      onLogits: (logits, context) => {
        logitsDigests.push(digestLogitsForTranscript(logits, {
          ...context,
          index: logitsDigests.length,
          decodeToken: (tokenId) => pipeline?.tokenizer?.decode?.([tokenId], false, false) ?? null,
        }));
      },
    } : {}),
    onToken: (tokenId, tokenText) => {
      tokenIds.push(tokenId);
      tokenRecords.push({
        id: tokenId,
        text: typeof tokenText === 'string' ? tokenText : '',
        fallbackText: pipeline?.tokenizer?.decode?.([tokenId], false, false) ?? '',
      });
    },
  })) {
    if (typeof tokenText === 'string') {
      tokens.push(tokenText);
    }
  }

  const durationMs = Math.max(1, performance.now() - start);
  const tokensPerSec = (tokens.length / durationMs) * 1000;
  const { phase } = buildGenerationPhaseFromStats(pipeline, durationMs, tokenIds.length);
  const kvCacheByteProof = await captureKvCacheByteProof(
    pipeline,
    shouldCaptureReferenceKvBytes(runOverrides, runtimeConfig)
  );

  return {
    ...(Number.isFinite(seed) ? { seed } : {}),
    prompt: promptLabel,
    promptInput,
    promptTokenIds,
    initialExecutionIdentity,
    maxTokens,
    generationConfig,
    tokens,
    tokenIds,
    tokenDiagnostics: summarizeGenerationTokens(tokenRecords),
    logitsDigests,
    kvCacheByteProof,
    output: tokens.join(''),
    durationMs,
    tokensPerSec,
    phase,
  };
}

export async function runImageTranscription(pipeline, runtimeConfig, runOverrides = null) {
  const imageInput = runOverrides?.image;
  if (!isPlainObject(imageInput)) {
    throw new Error('Image transcription requires inferenceInput.image.');
  }
  const prompt = typeof runOverrides?.prompt === 'string' && runOverrides.prompt.trim()
    ? runOverrides.prompt.trim()
    : DEFAULT_IMAGE_TRANSCRIPTION_PROMPT;
  const maxTokens = Number.isFinite(runOverrides?.maxTokens)
    ? Math.max(1, Math.floor(runOverrides.maxTokens))
    : resolveMaxTokens(runtimeConfig);
  const softTokenBudget = Number.isFinite(runOverrides?.softTokenBudget)
    ? Math.max(1, Math.floor(runOverrides.softTokenBudget))
    : DEFAULT_IMAGE_TRANSCRIPTION_SOFT_TOKEN_BUDGET;
  const {
    imageBytes,
    width,
    height,
    descriptor,
  } = await resolveInferenceImagePayload(imageInput);
  const start = performance.now();
  const result = await pipeline.transcribeImage({
    imageBytes,
    width,
    height,
    prompt,
    maxTokens,
    softTokenBudget,
  });
  const durationMs = Math.max(1, performance.now() - start);
  const tokenIds = Array.isArray(result?.tokens)
    ? result.tokens.map((value) => Number(value)).filter((value) => Number.isInteger(value))
    : [];
  const tokenRecords = tokenIds.map((tokenId) => {
    const decoded = pipeline?.tokenizer?.decode?.([tokenId], false, false) ?? '';
    return {
      id: tokenId,
      text: decoded,
      fallbackText: decoded,
    };
  });
  const { phase } = buildGenerationPhaseFromStats(pipeline, durationMs, tokenIds.length);
  return {
    inputMode: 'image_to_text',
    prompt: `image ${width}x${height}: ${prompt}`,
    promptInput: {
      prompt,
      image: descriptor,
    },
    maxTokens,
    tokens: tokenRecords.map((record) => record.text),
    tokenIds,
    tokenDiagnostics: summarizeGenerationTokens(tokenRecords),
    output: typeof result?.text === 'string' ? result.text : tokenRecords.map((record) => record.text).join(''),
    durationMs,
    tokensPerSec: tokenIds.length > 0 ? (tokenIds.length / durationMs) * 1000 : 0,
    phase,
  };
}

export async function runTextInference(pipeline, runtimeConfig, runOverrides = null) {
  if (isPlainObject(runOverrides?.image)) {
    return runImageTranscription(pipeline, runtimeConfig, runOverrides);
  }
  return runGeneration(pipeline, runtimeConfig, runOverrides);
}

export async function runEmbedding(pipeline, runtimeConfig, runOverrides = null) {
  const prompt = typeof runOverrides?.prompt === 'string' && runOverrides.prompt.trim()
    ? runOverrides.prompt.trim()
    : resolvePrompt(runtimeConfig);
  const start = performance.now();
  const result = await pipeline.embed(prompt, {
    benchmark: runOverrides?.benchmark === true,
  });
  const durationMs = Math.max(1, performance.now() - start);
  const tokenCount = Number.isFinite(result?.tokens?.length) ? result.tokens.length : 0;
  const stats = summarizeEmbeddingValues(result?.embedding);
  return {
    prompt,
    tokenCount,
    durationMs,
    phase: result?.phase ?? null,
    ...stats,
  };
}

export async function runSequenceEncoding(pipeline, runOverrides = null) {
  const sequence = typeof runOverrides?.sequence === 'string' ? runOverrides.sequence.trim() : '';
  if (!sequence) {
    throw new Error('Sequence qualification requires inferenceInput.sequence.');
  }
  const start = performance.now();
  const abortMode = runOverrides?.sequenceQualificationAbort ?? null;
  const staleAfterStart = runOverrides?.sequenceQualificationStaleAfterStart === true;
  const controller = abortMode ? new AbortController() : null;
  let requestGeneration = 1;
  if (abortMode === 'before_execution') {
    controller.abort('Sequence qualification cancelled before execution.');
  } else if (abortMode === 'after_start') {
    // Queue the abort only after encodeSequence has entered its async path. The
    // pipeline checks the signal at GPU synchronization boundaries and must
    // drop the late result instead of publishing an output.
    queueMicrotask(() => controller.abort('Sequence qualification cancelled after start.'));
  }
  if (staleAfterStart) {
    // Model results must stay bound to the generation that initiated them.
    // This qualification-only path supersedes the generation at the first
    // asynchronous boundary and verifies that a late result never reaches the
    // serialized command output.
    queueMicrotask(() => {
      requestGeneration = 2;
    });
  }
  const result = await pipeline.encodeSequence(sequence, {
    includeTokenEmbeddings: runOverrides?.includeTokenEmbeddings === true,
    includeLogits: runOverrides?.includeLogits === true,
    signal: controller?.signal,
  });
  if (staleAfterStart && requestGeneration !== 1) {
    const error = new Error('Sequence qualification result was superseded before publication.');
    error.name = 'StaleResultError';
    throw error;
  }
  const durationMs = Math.max(1, performance.now() - start);
  return {
    sequence,
    durationMs,
    ...result,
  };
}
