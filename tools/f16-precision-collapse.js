#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { initializeInference } from '../src/inference/test-harness.js';
import { applyRuntimeProfile } from '../src/inference/browser-harness-runtime-helpers.js';
import { getTopK } from '../src/inference/pipelines/text/sampling.js';
import { decodeFloatWeights } from '../src/inference/pipelines/text/generator-runtime.js';
import { readBufferSlice } from '../src/memory/buffer-pool.js';
import { isWeightBuffer } from '../src/gpu/weight-buffer.js';
import { installNodeFileFetchShim } from '../src/tooling/node-file-fetch.js';
import { bootstrapNodeWebGPU } from '../src/tooling/node-webgpu.js';
import {
  PRECISION_REPLAY_MODES,
  buildModeScoreMaps,
  compareTokenSequences,
  computeInversionCount,
  summarizeRanking,
} from '../src/tooling/precision-replay-math.js';

const DEFAULT_MODEL_DIR = 'models/local/gemma-3-270m-it-q4k-ehf16-af32';
const DEFAULT_RUNTIME_PROFILE = 'profiles/production';
const DEFAULT_TOP_K = 64;
const DEFAULT_SUMMARY_TOP_K = 8;
const DEFAULT_CONTINUATION_TOKENS = 8;
const TOOL_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_PROMPT_PACK = path.join(TOOL_DIR, 'data', 'f16-precision-collapse-curated-prompts.json');

function cloneValue(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function parsePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function parseArgs(argv) {
  const parsed = {
    modelDir: DEFAULT_MODEL_DIR,
    modelId: null,
    promptPack: null,
    runtimeProfile: DEFAULT_RUNTIME_PROFILE,
    topK: DEFAULT_TOP_K,
    summaryTopK: DEFAULT_SUMMARY_TOP_K,
    continuationTokens: DEFAULT_CONTINUATION_TOKENS,
    outDir: null,
    maxPrompts: null,
    useChatTemplate: false,
    help: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      parsed.help = true;
      continue;
    }
    if (arg === '--model-dir') {
      const value = argv[i + 1];
      if (!value) throw new Error('--model-dir requires a path.');
      parsed.modelDir = value;
      i += 1;
      continue;
    }
    if (arg === '--model-id') {
      const value = argv[i + 1];
      if (!value) throw new Error('--model-id requires a value.');
      parsed.modelId = value;
      i += 1;
      continue;
    }
    if (arg === '--prompt-pack') {
      const value = argv[i + 1];
      if (!value) throw new Error('--prompt-pack requires a path.');
      parsed.promptPack = value;
      i += 1;
      continue;
    }
    if (arg === '--runtime-profile') {
      const value = argv[i + 1];
      if (!value) throw new Error('--runtime-profile requires a value.');
      parsed.runtimeProfile = value;
      i += 1;
      continue;
    }
    if (arg === '--top-k') {
      const value = argv[i + 1];
      if (!value) throw new Error('--top-k requires a value.');
      parsed.topK = parsePositiveInteger(value, '--top-k');
      i += 1;
      continue;
    }
    if (arg === '--summary-top-k') {
      const value = argv[i + 1];
      if (!value) throw new Error('--summary-top-k requires a value.');
      parsed.summaryTopK = parsePositiveInteger(value, '--summary-top-k');
      i += 1;
      continue;
    }
    if (arg === '--continuation-tokens') {
      const value = argv[i + 1];
      if (!value) throw new Error('--continuation-tokens requires a value.');
      parsed.continuationTokens = parsePositiveInteger(value, '--continuation-tokens');
      i += 1;
      continue;
    }
    if (arg === '--out-dir') {
      const value = argv[i + 1];
      if (!value) throw new Error('--out-dir requires a path.');
      parsed.outDir = value;
      i += 1;
      continue;
    }
    if (arg === '--max-prompts') {
      const value = argv[i + 1];
      if (!value) throw new Error('--max-prompts requires a value.');
      parsed.maxPrompts = parsePositiveInteger(value, '--max-prompts');
      i += 1;
      continue;
    }
    if (arg === '--use-chat-template') {
      parsed.useChatTemplate = true;
      continue;
    }
    if (arg === '--no-chat-template') {
      parsed.useChatTemplate = false;
      continue;
    }
    throw new Error(`Unknown argument "${arg}".`);
  }

  return parsed;
}

function printHelp() {
  console.log(
    [
      'Usage: node tools/f16-precision-collapse.js [options]',
      '',
      'Options:',
      `  --model-dir <path>           Local RDRR artifact directory (default: ${DEFAULT_MODEL_DIR})`,
      '  --model-id <id>              Optional modelId label for the report',
      `  --runtime-profile <id>       Runtime profile to apply (default: ${DEFAULT_RUNTIME_PROFILE})`,
      '  --prompt-pack <path>         JSON prompt pack or Doe-style fixture with promptCandidates',
      `  --top-k <n>                  Candidate slice width for replay (default: ${DEFAULT_TOP_K})`,
      `  --summary-top-k <n>          Top entries to keep in per-mode summaries (default: ${DEFAULT_SUMMARY_TOP_K})`,
      `  --continuation-tokens <n>    Forced-branch decode length including step 0 (default: ${DEFAULT_CONTINUATION_TOKENS})`,
      '  --out-dir <path>             Output directory (default: reports/f16-precision-collapse/<timestamp>)',
      '  --max-prompts <n>            Limit the built-in prompt pack',
      '  --use-chat-template          Enable chat template expansion',
      '  --no-chat-template           Disable chat template expansion (default)',
      '  --help, -h                   Show this help',
      '',
      'This is a repo-only evidence tool. Output is written under reports/f16-precision-collapse/.',
    ].join('\n')
  );
}

function timestampLabel() {
  const now = new Date();
  const yyyy = now.getUTCFullYear();
  const mm = String(now.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(now.getUTCDate()).padStart(2, '0');
  const hh = String(now.getUTCHours()).padStart(2, '0');
  const mi = String(now.getUTCMinutes()).padStart(2, '0');
  const ss = String(now.getUTCSeconds()).padStart(2, '0');
  return `${yyyy}${mm}${dd}T${hh}${mi}${ss}Z`;
}

function resolveModelUrl(modelDir) {
  const absolutePath = path.resolve(modelDir);
  return pathToFileURL(absolutePath).href;
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function decodeToken(tokenizer, tokenId) {
  try {
    return tokenizer?.decode?.([tokenId], true, false) ?? `[${tokenId}]`;
  } catch {
    return `[${tokenId}]`;
  }
}

function decodeTokenIds(tokenizer, tokenIds) {
  try {
    return tokenizer?.decode?.(tokenIds, true, false) ?? null;
  } catch {
    return null;
  }
}

function resolveSingleTokenId(tokenizer, tokenText) {
  const raw = tokenizer?.encode?.(tokenText);
  const tokenIds = Array.isArray(raw)
    ? raw
    : (ArrayBuffer.isView(raw) ? Array.from(raw) : null);
  if (!Array.isArray(tokenIds) || tokenIds.length !== 1) {
    return null;
  }
  const tokenId = Number(tokenIds[0]);
  if (!Number.isInteger(tokenId) || tokenId < 0) {
    return null;
  }
  return tokenId;
}

function normalizeTokenText(text) {
  return typeof text === 'string' ? text : String(text ?? '');
}

function bytesPerElementForDtype(dtype) {
  const normalized = typeof dtype === 'string' ? dtype.trim().toLowerCase() : '';
  if (normalized === 'f16' || normalized === 'fp16') return 2;
  if (normalized === 'f32' || normalized === 'fp32') return 4;
  throw new Error(`Unsupported dtype "${dtype}" for LM-head replay.`);
}

function summarizeResolvedWatches(prompt, tokenizer) {
  const resolved = [];
  const unresolved = [];
  const seenIds = new Set();
  for (const tokenText of prompt.watchTokenTexts ?? []) {
    const tokenId = resolveSingleTokenId(tokenizer, tokenText);
    if (tokenId == null) {
      unresolved.push(tokenText);
      continue;
    }
    resolved.push({
      tokenText,
      tokenId,
      decodedText: decodeToken(tokenizer, tokenId),
    });
    seenIds.add(tokenId);
  }
  return {
    resolved,
    unresolved,
    tokenIds: [...seenIds],
  };
}

function normalizePromptRecord(record, index) {
  if (!record || typeof record !== 'object') {
    throw new Error(`Prompt record ${index} must be an object.`);
  }
  const id = typeof record.id === 'string' && record.id.trim() !== ''
    ? record.id.trim()
    : `prompt-${index + 1}`;
  if (typeof record.text !== 'string') {
    throw new Error(`Prompt record "${id}" is missing text.`);
  }
  const text = record.text;
  return {
    id,
    text,
    watchTokenTexts: Array.isArray(record.watchTokenTexts) ? record.watchTokenTexts.map(normalizeTokenText) : [],
    watchPairs: Array.isArray(record.watchPairs)
      ? record.watchPairs
        .filter((pair) => Array.isArray(pair) && pair.length === 2)
        .map((pair) => [normalizeTokenText(pair[0]), normalizeTokenText(pair[1])])
      : [],
    persistSlice: record.persistSlice === true,
  };
}

function loadPromptPack(promptPackPath) {
  const raw = JSON.parse(fs.readFileSync(promptPackPath, 'utf8'));
  const prompts = Array.isArray(raw)
    ? raw
    : (Array.isArray(raw?.promptCandidates) ? raw.promptCandidates : null);
  if (!Array.isArray(prompts) || prompts.length === 0) {
    throw new Error(`Prompt pack "${promptPackPath}" did not contain an array or promptCandidates array.`);
  }
  return prompts.map(normalizePromptRecord);
}

function resolveWatchTokensFromEntries(watchSummary, stableTopEntries) {
  const resolvedByText = new Map(watchSummary.resolved.map((entry) => [entry.tokenText, entry]));
  const stillUnresolved = [];
  for (const tokenText of watchSummary.unresolved) {
    const matchedEntry = stableTopEntries.find((entry) => entry.text === tokenText);
    if (!matchedEntry) {
      stillUnresolved.push(tokenText);
      continue;
    }
    resolvedByText.set(tokenText, {
      tokenText,
      tokenId: matchedEntry.token,
      decodedText: matchedEntry.text,
    });
  }
  return {
    resolved: [...resolvedByText.values()],
    unresolved: stillUnresolved,
    tokenIds: [...new Set([...watchSummary.tokenIds, ...[...resolvedByText.values()].map((entry) => entry.tokenId)])],
  };
}

function summarizeWatchPairs(prompt, watchLookup, perModeScores) {
  const pairs = [];
  for (const pair of prompt.watchPairs ?? []) {
    const left = watchLookup.get(pair[0]);
    const right = watchLookup.get(pair[1]);
    if (!left || !right) {
      pairs.push({
        pair,
        present: false,
      });
      continue;
    }
    const modes = {};
    for (const mode of PRECISION_REPLAY_MODES) {
      const leftScore = perModeScores[mode].get(left.tokenId);
      const rightScore = perModeScores[mode].get(right.tokenId);
      modes[mode] = {
        winnerTokenId: leftScore >= rightScore ? left.tokenId : right.tokenId,
        winnerText: leftScore >= rightScore ? left.decodedText : right.decodedText,
        leftScore,
        rightScore,
        gap: leftScore - rightScore,
      };
    }
    pairs.push({
      pair,
      present: true,
      left,
      right,
      modes,
      f16VsF32WinnerChanged: modes.f16_forward.winnerTokenId !== modes.f32_forward.winnerTokenId,
    });
  }
  return pairs;
}

async function readLmHeadRow(lmHead, hiddenSize, tokenId, rowCache) {
  if (rowCache.has(tokenId)) {
    return rowCache.get(tokenId);
  }
  if (!isWeightBuffer(lmHead)) {
    throw new Error('LM-head weight is not a readable WeightBuffer.');
  }
  if (lmHead.layout !== 'row') {
    throw new Error(`LM-head layout "${lmHead.layout}" is not supported by this tool.`);
  }
  const bytesPerElement = bytesPerElementForDtype(lmHead.dtype);
  const rowBytes = hiddenSize * bytesPerElement;
  const offset = tokenId * rowBytes;
  const bytes = await readBufferSlice(lmHead.buffer, offset, rowBytes);
  const decoded = decodeFloatWeights(bytes, lmHead.dtype, hiddenSize, `lm_head row ${tokenId}`);
  rowCache.set(tokenId, decoded);
  return decoded;
}

async function collectCandidateRows(lmHead, hiddenSize, tokenIds, rowCache) {
  const rows = new Map();
  for (const tokenId of tokenIds) {
    rows.set(tokenId, await readLmHeadRow(lmHead, hiddenSize, tokenId, rowCache));
  }
  return rows;
}

function buildReplayValidation(stableTopEntries, replayF32Scores) {
  let maxAbsError = 0;
  let meanAbsError = 0;
  for (const entry of stableTopEntries) {
    const replay = replayF32Scores.get(entry.token);
    const error = Math.abs((entry.logit ?? 0) - (replay ?? 0));
    maxAbsError = Math.max(maxAbsError, error);
    meanAbsError += error;
  }
  return {
    comparedTokens: stableTopEntries.length,
    maxAbsError,
    meanAbsError: stableTopEntries.length > 0 ? meanAbsError / stableTopEntries.length : 0,
  };
}

async function generateForcedBranch(pipeline, prefixSnapshot, tokenizer, forcedTokenId, continuationTokens) {
  pipeline.reset();
  pipeline.applyKVCacheSnapshot(prefixSnapshot);
  const tailLength = Math.max(continuationTokens - 1, 0);
  const result = await pipeline.generateTokenIds('', {
    useChatTemplate: false,
    inputIds: [forcedTokenId],
    maxTokens: tailLength,
    temperature: 0,
    topP: 1,
    topK: 1,
    repetitionPenalty: 1,
  });
  const tokenIds = [forcedTokenId, ...(result?.tokenIds ?? [])];
  return {
    tokenIds,
    decodedText: decodeTokenIds(tokenizer, tokenIds),
    steps: tokenIds.map((tokenId, index) => ({
      step: index,
      tokenId,
      text: decodeToken(tokenizer, tokenId),
    })),
    stats: result?.stats ?? null,
  };
}

function relativeReportPath(rootDir, filePath) {
  return path.relative(rootDir, filePath).split(path.sep).join('/');
}

function buildMarkdownSummary(report) {
  const lines = [];
  lines.push('# f16 precision collapse summary');
  lines.push('');
  lines.push(`- model: \`${report.modelId}\``);
  lines.push(`- gpu: \`${report.gpu.provider ?? 'unknown'}\` / \`${report.gpu.adapter ?? 'unknown'}\``);
  lines.push(`- prompts: ${report.aggregate.promptCount}`);
  lines.push(`- prompts with f16 vs f32 top-1 flip: ${report.aggregate.f16VsF32FlipCount}`);
  lines.push(`- persistent winner flips through ${report.continuationTokens} forced steps: ${report.aggregate.persistentFlipCount}`);
  lines.push(`- healed winner flips within ${report.continuationTokens} forced steps: ${report.aggregate.healedFlipCount}`);
  lines.push(`- prompts with watched-pair order swap: ${report.aggregate.watchPairSwapCount}`);
  lines.push(`- total top-${report.topK} inversions (f16 vs f32): ${report.aggregate.totalF16VsF32Inversions}`);
  lines.push(`- max replay abs error vs live f32 logits: ${report.aggregate.maxReplayAbsError.toExponential(6)}`);
  lines.push(`- mean replay abs error vs live f32 logits: ${report.aggregate.meanReplayAbsError.toExponential(6)}`);
  lines.push('');
  lines.push('## Winner flips');
  lines.push('');
  if (report.highlights.flips.length === 0) {
    lines.push('No top-1 winner flips were observed in this prompt pack.');
  } else {
    for (const prompt of report.highlights.flips) {
      lines.push(`- \`${prompt.id}\`: f32=\`${prompt.modes.f32_forward.winnerText}\`, f16=\`${prompt.modes.f16_forward.winnerText}\`, gap=${prompt.modes.f32_forward.winnerGap?.toFixed(6) ?? 'n/a'}, branchDiffSteps=${prompt.branchComparison?.differingStepCount ?? 0}`);
    }
  }
  lines.push('');
  lines.push('## Strong watched-pair examples');
  lines.push('');
  if (report.highlights.watchPairs.length === 0) {
    lines.push('No watched pairs swapped order in this run.');
  } else {
    for (const pair of report.highlights.watchPairs) {
      lines.push(`- \`${pair.promptId}\` \`${pair.left}\` vs \`${pair.right}\`: f32=${pair.f32Winner}, f16=${pair.f16Winner}, f32Gap=${pair.f32Gap.toFixed(6)}, f16Gap=${pair.f16Gap.toFixed(6)}`);
    }
  }
  lines.push('');
  lines.push('## Artifacts');
  lines.push('');
  lines.push('- `summary.json` contains the full prompt-by-prompt replay report.');
  if (report.highlights.sliceArtifacts.length > 0) {
    lines.push(`- slice artifacts: ${report.highlights.sliceArtifacts.map((entry) => `\`${entry}\``).join(', ')}`);
  }
  return `${lines.join('\n')}\n`;
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    printHelp();
    process.exit(0);
  }

  installNodeFileFetchShim();
  const originalRuntime = cloneValue(getRuntimeConfig());
  const stamp = timestampLabel();
  const outputDir = path.resolve(args.outDir ?? path.join('reports', 'f16-precision-collapse', stamp));
  const slicesDir = path.join(outputDir, 'slices');
  ensureDir(outputDir);
  ensureDir(slicesDir);

  let harness = null;
  try {
    if (args.runtimeProfile) {
      await applyRuntimeProfile(args.runtimeProfile);
    }

    const bootstrap = await bootstrapNodeWebGPU();
    if (!bootstrap?.ok) {
      throw new Error(`WebGPU bootstrap failed: ${bootstrap?.detail ?? 'unknown error'}`);
    }

    const modelUrl = resolveModelUrl(args.modelDir);
    const runtimeConfig = cloneValue(getRuntimeConfig());
    harness = await initializeInference(modelUrl, {
      modelId: args.modelId,
      runtime: { runtimeConfig },
    });

    const promptPackPath = path.resolve(args.promptPack ?? DEFAULT_PROMPT_PACK);
    const promptPack = loadPromptPack(promptPackPath);
    const prompts = args.maxPrompts == null
      ? promptPack
      : promptPack.slice(0, Math.min(args.maxPrompts, promptPack.length));
    const { pipeline, manifest, capabilities } = harness;
    const tokenizer = pipeline.tokenizer;
    const lmHead = pipeline.weights.get('lm_head');
    const hiddenSize = Number(Array.isArray(lmHead?.shape) ? lmHead.shape[1] : 0);
    if (!Number.isInteger(hiddenSize) || hiddenSize < 1) {
      throw new Error('Unable to resolve LM-head hidden size for replay.');
    }

    const rowCache = new Map();
    const promptReports = [];
    const sliceArtifacts = [];

    for (const prompt of prompts) {
      pipeline.reset();
      const prefill = await pipeline.prefillWithLogits(prompt.text, {
        useChatTemplate: args.useChatTemplate,
      });
      const stableLogits = prefill?.logits instanceof Float32Array
        ? prefill.logits
        : Float32Array.from(prefill?.logits ?? []);
      const stableTopEntries = getTopK(stableLogits, args.topK, (tokenIds) => decodeTokenIds(tokenizer, tokenIds));
      const promptTokenIds = Array.isArray(prefill?.tokens)
        ? prefill.tokens
        : (ArrayBuffer.isView(prefill?.tokens) ? Array.from(prefill.tokens) : []);
      const prefixSnapshot = {
        cache: prefill.cache,
        seqLen: prefill.seqLen,
        tokens: promptTokenIds,
        linearAttention: prefill.linearAttention,
      };
      const stableTopTokenIds = stableTopEntries.map((entry) => entry.token);
      const watchSummary = resolveWatchTokensFromEntries(
        summarizeResolvedWatches(prompt, tokenizer),
        stableTopEntries
      );
      const watchLookup = new Map(watchSummary.resolved.map((entry) => [entry.tokenText, entry]));
      const candidateTokenIds = [...new Set([...stableTopTokenIds, ...watchSummary.tokenIds])];

      pipeline.reset();
      const embeddingResult = await pipeline.prefillWithEmbedding(prompt.text, {
        useChatTemplate: args.useChatTemplate,
        embeddingMode: 'last',
      });
      const hidden = Array.isArray(embeddingResult.embedding)
        ? Float32Array.from(embeddingResult.embedding)
        : Float32Array.from(embeddingResult.embedding ?? []);
      const rows = await collectCandidateRows(lmHead, hiddenSize, candidateTokenIds, rowCache);
      const perModeScores = buildModeScoreMaps(hidden, rows);
      const replayValidation = buildReplayValidation(stableTopEntries, perModeScores.f32_forward);

      const modes = {};
      for (const mode of PRECISION_REPLAY_MODES) {
        modes[mode] = summarizeRanking(
          candidateTokenIds,
          perModeScores[mode],
          (tokenId) => decodeToken(tokenizer, tokenId),
          args.summaryTopK
        );
      }

      const watchPairs = summarizeWatchPairs(prompt, watchLookup, perModeScores);
      const inversionCount = computeInversionCount(candidateTokenIds, perModeScores.f32_forward, perModeScores.f16_forward);
      const f16Winner = modes.f16_forward.winnerTokenId;
      const f32Winner = modes.f32_forward.winnerTokenId;
      let branches = null;
      let branchComparison = null;
      if (f16Winner != null && f32Winner != null && f16Winner !== f32Winner) {
        const f32Branch = await generateForcedBranch(
          pipeline,
          prefixSnapshot,
          tokenizer,
          f32Winner,
          args.continuationTokens
        );
        const f16Branch = await generateForcedBranch(
          pipeline,
          prefixSnapshot,
          tokenizer,
          f16Winner,
          args.continuationTokens
        );
        branches = {
          f32_forward: f32Branch,
          f16_forward: f16Branch,
        };
        branchComparison = compareTokenSequences(f32Branch.tokenIds, f16Branch.tokenIds);
      }

      const promptReport = {
        id: prompt.id,
        text: prompt.text,
        promptTokenCount: promptTokenIds.length,
        promptTailText: decodeTokenIds(tokenizer, promptTokenIds.slice(Math.max(0, promptTokenIds.length - 16))),
        stableTopK: stableTopEntries.slice(0, args.summaryTopK).map((entry, index) => ({
          rank: index + 1,
          tokenId: entry.token,
          text: entry.text,
          logit: entry.logit,
          prob: entry.prob,
        })),
        candidateTokenCount: candidateTokenIds.length,
        watchTokens: {
          resolved: watchSummary.resolved,
          unresolved: watchSummary.unresolved,
        },
        replayValidation,
        modes,
        inversionCountF16VsF32: inversionCount,
        watchPairs,
        f16VsF32WinnerChanged: f16Winner !== f32Winner,
        branches,
        branchComparison,
      };

      const persistSlice = prompt.persistSlice || promptReport.f16VsF32WinnerChanged;
      if (persistSlice) {
        const slicePath = path.join(slicesDir, `${prompt.id}.json`);
        writeJson(slicePath, {
          prompt: {
            id: prompt.id,
            text: prompt.text,
          },
          embedding: Array.from(hidden),
          candidateRows: candidateTokenIds.map((tokenId) => ({
            tokenId,
            text: decodeToken(tokenizer, tokenId),
            row: Array.from(rows.get(tokenId) ?? []),
          })),
        });
        promptReport.sliceArtifact = relativeReportPath(outputDir, slicePath);
        sliceArtifacts.push(promptReport.sliceArtifact);
      }

      promptReports.push(promptReport);
    }

    const watchPairHighlights = [];
    const flippedPrompts = promptReports.filter((entry) => entry.f16VsF32WinnerChanged);
    for (const prompt of promptReports) {
      for (const pair of prompt.watchPairs) {
        if (!pair.present || !pair.f16VsF32WinnerChanged) {
          continue;
        }
        watchPairHighlights.push({
          promptId: prompt.id,
          left: pair.left.decodedText,
          right: pair.right.decodedText,
          f32Winner: pair.modes.f32_forward.winnerText,
          f16Winner: pair.modes.f16_forward.winnerText,
          f32Gap: pair.modes.f32_forward.gap,
          f16Gap: pair.modes.f16_forward.gap,
        });
      }
    }
    const persistentFlipCount = flippedPrompts.filter((entry) => entry.branchComparison?.persistsThroughEnd === true).length;
    const healedFlipCount = flippedPrompts.filter((entry) => entry.branchComparison?.healedAtStep != null).length;
    const replayMaxErrors = promptReports.map((entry) => entry.replayValidation.maxAbsError);
    const replayMeanErrors = promptReports.map((entry) => entry.replayValidation.meanAbsError);

    const report = {
      schemaVersion: 1,
      generatedAtUtc: new Date().toISOString(),
      modelId: manifest?.modelId ?? args.modelId ?? path.basename(args.modelDir),
      modelUrl,
      runtimeProfile: args.runtimeProfile,
      promptPackPath,
      topK: args.topK,
      summaryTopK: args.summaryTopK,
      continuationTokens: args.continuationTokens,
      promptPack: prompts.map(({ id, text }) => ({ id, text })),
      gpu: {
        provider: bootstrap.provider ?? null,
        adapter: bootstrap.adapter ?? null,
        capabilities,
      },
      aggregate: {
        promptCount: promptReports.length,
        f16VsF32FlipCount: flippedPrompts.length,
        persistentFlipCount,
        healedFlipCount,
        watchPairSwapCount: watchPairHighlights.length,
        totalF16VsF32Inversions: promptReports.reduce((sum, entry) => sum + entry.inversionCountF16VsF32, 0),
        maxReplayAbsError: Math.max(...replayMaxErrors, 0),
        meanReplayAbsError: replayMeanErrors.length > 0
          ? replayMeanErrors.reduce((sum, value) => sum + value, 0) / replayMeanErrors.length
          : 0,
      },
      prompts: promptReports,
      highlights: {
        flips: flippedPrompts.map((entry) => ({
          id: entry.id,
          text: entry.text,
          modes: {
            exact: entry.modes.exact,
            f32_forward: entry.modes.f32_forward,
            f16_forward: entry.modes.f16_forward,
          },
          branchComparison: entry.branchComparison,
          sliceArtifact: entry.sliceArtifact ?? null,
        })),
        watchPairs: watchPairHighlights,
        sliceArtifacts,
      },
    };

    writeJson(path.join(outputDir, 'summary.json'), report);
    fs.writeFileSync(path.join(outputDir, 'summary.md'), buildMarkdownSummary(report), 'utf8');
    console.log(outputDir);
  } finally {
    try {
      await harness?.pipeline?.unload?.();
    } catch {
      // Best-effort cleanup for a repo-only analysis tool.
    }
    try {
      harness?.pipeline?.releaseGPUResources?.();
    } catch {
      // Best-effort cleanup for a repo-only analysis tool.
    }
    setRuntimeConfig(originalRuntime);
  }
}

main()
  .then(() => {
    process.exit(0);
  })
  .catch((error) => {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exit(1);
  });
