#!/usr/bin/env node

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

import { bootstrapNodeWebGPU } from '../src/tooling/node-webgpu.js';
import { releaseBuffer } from '../src/memory/buffer-pool.js';
import {
  initDevice,
  getKernelCapabilities,
} from '../src/gpu/device.js';
import { parseManifest } from '../src/formats/rdrr/parsing.js';
import { createPipeline } from '../src/inference/pipelines/text.js';
import { createDopplerConfig } from '../src/config/schema/index.js';
import { computeSampleStats } from '../src/debug/stats.js';
import { applyRepetitionPenalty, sample } from '../src/inference/pipelines/text/sampling.js';

const DEFAULT_MODEL_DIR = 'models/local/gemma-3-1b-it-f16-af32';
const DEFAULT_PROMPT = 'The weather is nice today.';
const DEFAULT_MAX_TOKENS = 32;
const DEFAULT_WARMUP_RUNS = 1;
const DEFAULT_TIMED_RUNS = 3;
const DEFAULT_TRACE_TOP_K = 5;

function releaseLogitsStepResult(stepResult) {
  if (stepResult?.logitsBuffer) {
    releaseBuffer(stepResult.logitsBuffer);
  }
}

function parsePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return parsed;
}

function parseSeed(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error('seed must be a non-negative number');
  }
  return parsed;
}

function parseArgs(argv) {
  const parsed = {
    modelDir: DEFAULT_MODEL_DIR,
    prompt: DEFAULT_PROMPT,
    maxTokens: DEFAULT_MAX_TOKENS,
    warmupRuns: DEFAULT_WARMUP_RUNS,
    timedRuns: DEFAULT_TIMED_RUNS,
    seed: 0,
    json: false,
    traceParity: false,
    teacherForced: false,
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
      if (!value) {
        throw new Error('--model-dir requires a path');
      }
      parsed.modelDir = value;
      i += 1;
      continue;
    }
    if (arg === '--prompt') {
      const value = argv[i + 1];
      if (value == null) {
        throw new Error('--prompt requires a string');
      }
      parsed.prompt = value;
      i += 1;
      continue;
    }
    if (arg === '--max-tokens') {
      parsed.maxTokens = parsePositiveInteger(argv[i + 1], '--max-tokens');
      i += 1;
      continue;
    }
    if (arg === '--warmup-runs') {
      parsed.warmupRuns = parseNonNegativeInteger(argv[i + 1], '--warmup-runs');
      i += 1;
      continue;
    }
    if (arg === '--timed-runs') {
      parsed.timedRuns = parseNonNegativeInteger(argv[i + 1], '--timed-runs');
      i += 1;
      continue;
    }
    if (arg === '--seed') {
      parsed.seed = parseSeed(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--json') {
      parsed.json = true;
      continue;
    }
    if (arg === '--trace-parity') {
      parsed.traceParity = true;
      continue;
    }
    if (arg === '--teacher-forced') {
      parsed.teacherForced = true;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }

  return parsed;
}

function usage() {
  return [
    'Usage:',
    '  node tools/bench-text-decode-paths.js [flags]',
    '',
    'Flags:',
    '  --model-dir <dir>        Model directory containing manifest.json (default: models/local/gemma-3-1b-it-f16-af32)',
    '  --prompt <text>          Prompt text (default: The weather is nice today.)',
    '  --max-tokens <int>       Tokens to generate per run (default: 32)',
    '  --warmup-runs <int>      Warmup runs per mode, >= 0 (default: 1)',
    '  --timed-runs <int>       Timed runs per mode, >= 0 (default: 3)',
    '  --seed <number>          Seed for non-deterministic sampling modes (default: 0)',
    '  --json                   Print JSON payload only',
    '  --trace-parity           Emit per-step token/logit trace for token-path vs logits-path',
    '  --teacher-forced         Force both paths to decode the same teacher token sequence',
  ].join('\n');
}

function fileBytesToArrayBuffer(filePath) {
  const bytes = readFileSync(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function createLocalStorageContext(modelDir, manifest) {
  const tokenizerPath = manifest?.tokenizer?.file
    ? path.join(modelDir, manifest.tokenizer.file)
    : null;

  return {
    verifyHashes: true,
    async loadShard(index) {
      const shard = manifest?.shards?.[index];
      if (!shard?.filename) {
        throw new Error(`Manifest shard ${index} is missing filename.`);
      }
      return fileBytesToArrayBuffer(path.join(modelDir, shard.filename));
    },
    async loadTokenizerJson() {
      if (!tokenizerPath) {
        throw new Error('Manifest tokenizer.file is required for local RDRR benchmark execution.');
      }
      return JSON.parse(readFileSync(tokenizerPath, 'utf8'));
    },
  };
}

function isStopToken(tokenId, tokenizer, modelConfig) {
  const eos = tokenizer?.getSpecialTokens?.()?.eos;
  if (Number.isFinite(eos) && tokenId === eos) {
    return true;
  }
  if (Array.isArray(modelConfig?.stopTokenIds)) {
    return modelConfig.stopTokenIds.includes(tokenId);
  }
  return false;
}

function toDecodeText(tokenizer, tokenId) {
  try {
    return tokenizer?.decode?.([tokenId], true, false) ?? `[${tokenId}]`;
  } catch {
    return `[${tokenId}]`;
  }
}

function buildRuntimeConfig(maxTokens) {
  return createDopplerConfig({
    runtime: {
      inference: {
        batching: {
          maxTokens,
          batchSize: 1,
          readbackInterval: 1,
        },
        generation: {
          disableCommandBatching: true,
          disableMultiTokenDecode: true,
        },
        sampling: {
          temperature: 0,
          topK: 1,
          topP: 1,
          repetitionPenalty: 1.0,
        },
        chatTemplate: {
          enabled: false,
        },
      },
    },
  });
}

function summarizeTimings(runs, metric) {
  const samples = runs.map((entry) => entry[metric]).filter((value) => Number.isFinite(value));
  const stats = computeSampleStats(samples, { outlierIqrMultiplier: 1.5 });
  const median = stats.median;
  return {
    count: samples.length,
    meanMs: stats.mean,
    medianMs: median,
    p95Ms: stats.p95,
    p99Ms: stats.p99,
    minMs: stats.min,
    maxMs: stats.max,
  };
}

async function createBenchmarkPipeline(modelDir, manifest, runtimeConfig) {
  return createPipeline(manifest, {
    runtimeConfig,
    gpu: { device: await initDevice() },
    storage: createLocalStorageContext(modelDir, manifest),
  });
}

async function runWithFreshPipeline(modelDir, manifest, runtimeConfig, runFn) {
  const pipeline = await createBenchmarkPipeline(modelDir, manifest, runtimeConfig);
  try {
    return await runFn(pipeline);
  } finally {
    pipeline?.releaseGPUResources?.();
    await pipeline?.unload?.();
  }
}

function formatRunResult(modeResult) {
  return [
    `${modeResult.label}:`,
    `  tokens: ${modeResult.tokensGenerated}`,
    `  output: ${JSON.stringify(modeResult.outputText)}`,
    `  prefillMs: ${modeResult.prefillMs.toFixed(2)}`,
    `  decodeMs: ${modeResult.decodeMs.toFixed(2)}`,
    `  firstTokenMs: ${modeResult.firstTokenMs.toFixed(2)}`,
    `  totalMs: ${modeResult.totalMs.toFixed(2)}`,
    `  tokens/s: ${modeResult.tokensPerSec.toFixed(2)}`,
  ];
}

function buildSamplingOptions(tokenizer, seed) {
  return {
    temperature: 0,
    topK: 1,
    topP: 1,
    padTokenId: tokenizer?.getSpecialTokens?.()?.pad,
    seed,
  };
}

function getTopKSummary(logits, tokenizer, k = DEFAULT_TRACE_TOP_K) {
  const entries = [];
  const limit = Math.min(k, logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    const value = logits[i];
    if (!Number.isFinite(value)) {
      continue;
    }
    if (entries.length < limit) {
      entries.push({ tokenId: i, logit: value });
      entries.sort((a, b) => b.logit - a.logit || a.tokenId - b.tokenId);
      continue;
    }
    const tail = entries[entries.length - 1];
    if (value > tail.logit || (value === tail.logit && i < tail.tokenId)) {
      entries[entries.length - 1] = { tokenId: i, logit: value };
      entries.sort((a, b) => b.logit - a.logit || a.tokenId - b.tokenId);
    }
  }
  return entries.map((entry) => ({
    tokenId: entry.tokenId,
    logit: entry.logit,
    text: toDecodeText(tokenizer, entry.tokenId),
  }));
}

async function traceParity(pipeline, prompt, options) {
  const tokenizer = pipeline.tokenizer;
  const samplingOptions = buildSamplingOptions(tokenizer, options.seed);
  const penalty = Number.isFinite(options.repetitionPenalty)
    ? options.repetitionPenalty
    : 1.0;

  pipeline.reset();
  const prefill = await pipeline.prefillWithLogits(prompt, { useChatTemplate: false });
  const manualContext = Array.isArray(prefill.tokens) ? [...prefill.tokens] : [];
  const trace = [];

  for (let stepIndex = 0; stepIndex < options.maxTokens; stepIndex += 1) {
    const stepSource = stepIndex === 0
      ? { logits: prefill.logits }
      : await pipeline.decodeStepLogits(manualContext, { useChatTemplate: false, seed: options.seed });
    const logits = Float32Array.from(stepSource.logits || []);
    releaseLogitsStepResult(stepSource);
    applyRepetitionPenalty(logits, manualContext, penalty);
    const sampledToken = sample(logits, samplingOptions);
    trace.push({
      step: stepIndex + 1,
      manualContextTail: manualContext.slice(Math.max(0, manualContext.length - 8)),
      manualToken: sampledToken,
      manualText: toDecodeText(tokenizer, sampledToken),
      manualTopK: getTopKSummary(logits, tokenizer),
    });
    manualContext.push(sampledToken);
    if (isStopToken(sampledToken, tokenizer, pipeline.modelConfig)) {
      break;
    }
  }

  pipeline.reset();
  let generateIndex = 0;
  for await (const _chunk of pipeline.generate(prompt, {
    useChatTemplate: false,
    maxTokens: options.maxTokens,
    temperature: 0,
    topK: 1,
    topP: 1,
    repetitionPenalty: penalty,
    seed: options.seed,
    onToken(tokenId) {
      if (!trace[generateIndex]) {
        trace[generateIndex] = { step: generateIndex + 1 };
      }
      trace[generateIndex].generateToken = tokenId;
      trace[generateIndex].generateText = toDecodeText(tokenizer, tokenId);
      generateIndex += 1;
    },
  })) {
    void _chunk;
  }

  let firstMismatchStep = null;
  for (const entry of trace) {
    if (entry && entry.generateToken !== entry.manualToken) {
      firstMismatchStep = entry.step;
      break;
    }
  }

  return {
    firstMismatchStep,
    steps: trace,
  };
}

async function runTokenPath(pipeline, prompt, options) {
  const generationOptions = {
    useChatTemplate: false,
    maxTokens: options.maxTokens,
    temperature: 0,
    topK: 1,
    topP: 1,
    repetitionPenalty: Number.isFinite(options.repetitionPenalty) ? options.repetitionPenalty : 1.0,
    seed: options.seed,
  };
  pipeline.reset();
  const startedAt = performance.now();
  const result = await pipeline.generateTokenIds(prompt, generationOptions);
  const totalMs = performance.now() - startedAt;
  const tokenIds = result.tokenIds;
  const stats = result.stats ?? pipeline.getStats();
  const tokensGenerated = Number.isInteger(stats?.tokensGenerated) ? stats.tokensGenerated : tokenIds.length;
  const prefillMs = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : totalMs;
  const decodeMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : Math.max(0, totalMs - prefillMs);
  const firstTokenMs = Number.isFinite(stats.ttftMs) ? stats.ttftMs : prefillMs;
  const firstResponseMs = Number.isFinite(stats.firstResponseMs) ? stats.firstResponseMs : firstTokenMs;
  const decodedText = tokenIds.map((tokenId) => toDecodeText(pipeline.tokenizer, tokenId)).join('');
  return {
    label: 'token-path',
    tokensGenerated,
    outputText: decodedText,
    tokenIds,
    prefillMs,
    decodeMs,
    firstTokenMs,
    firstResponseMs,
    totalMs,
    tokensPerSec: tokensGenerated > 0 ? (tokensGenerated / totalMs) * 1000 : 0,
  };
}

async function runLogitsPath(pipeline, prompt, options) {
  const maxTokens = options.maxTokens;
  const seed = options.seed;
  const penalty = Number.isFinite(options.repetitionPenalty)
    ? options.repetitionPenalty
    : 1.0;

  pipeline.reset();
  const prefillStart = performance.now();
  const prefillResult = await pipeline.prefillWithLogits(prompt, { useChatTemplate: false });
  const prefillMs = performance.now() - prefillStart;

  const tokenIds = [];
  const tokenTexts = [];
  const modelConfig = pipeline.modelConfig;
  const tokenizer = pipeline.tokenizer;
  const padTokenId = tokenizer?.getSpecialTokens?.()?.pad;
  const samplingOptsBase = {
    temperature: 0,
    topK: 1,
    topP: 1,
    padTokenId,
  };

  const contextTokens = Array.isArray(prefillResult.tokens)
    ? [...prefillResult.tokens]
    : [];
  const firstLogits = Float32Array.from(prefillResult.logits || []);
  if (firstLogits.length === 0) {
    const now = performance.now();
    return {
      label: 'logits-path',
      tokensGenerated: 0,
      outputText: '',
      tokenIds: [],
      prefillMs,
      decodeMs: 0,
      firstTokenMs: prefillMs,
      firstResponseMs: prefillMs,
      totalMs: now - prefillStart,
      tokensPerSec: 0,
      error: 'first decode returned empty logits',
    };
  }
  const firstLogitsCopy = Float32Array.from(firstLogits);
  applyRepetitionPenalty(firstLogitsCopy, contextTokens, penalty);
  if (seed != null) {
    samplingOptsBase.seed = seed;
  }
  let firstToken;
  try {
    firstToken = sample(firstLogitsCopy, samplingOptsBase);
  } catch (error) {
    const now = performance.now();
    return {
      label: 'logits-path',
      tokensGenerated: 0,
      outputText: '',
      tokenIds: [],
      prefillMs,
      decodeMs: 0,
      firstTokenMs: now - prefillStart,
      firstResponseMs: now - prefillStart,
      totalMs: now - prefillStart,
      tokensPerSec: 0,
      error: error.message,
    };
  }

  tokenIds.push(firstToken);
  tokenTexts.push(toDecodeText(tokenizer, firstToken));
  contextTokens.push(firstToken);

  const firstTokenTs = performance.now();
  const totalStart = prefillStart;
  let decodeMs = 0;

  while (tokenIds.length < maxTokens && !isStopToken(firstToken, tokenizer, modelConfig)) {
    const decodeStepStart = performance.now();
    let stepResult;
    try {
      stepResult = await pipeline.decodeStepLogits(contextTokens, {
        useChatTemplate: false,
      });
    } catch (error) {
      const elapsedMs = performance.now() - prefillStart;
      return {
        label: 'logits-path',
        tokensGenerated: tokenIds.length,
        outputText: tokenTexts.join(''),
        tokenIds,
        prefillMs,
        decodeMs,
        firstTokenMs: firstTokenTs - prefillStart,
        firstResponseMs: firstTokenTs - prefillStart,
        totalMs: elapsedMs,
        tokensPerSec: tokenIds.length > 0 ? (tokenIds.length / elapsedMs) * 1000 : 0,
        error: error.message,
      };
    }
    const stepLogits = stepResult?.logits;
    const stepLogitsCopy = Float32Array.from(stepLogits || []);
    releaseLogitsStepResult(stepResult);
    applyRepetitionPenalty(stepLogitsCopy, contextTokens, penalty);
    let nextToken;
    try {
      nextToken = sample(stepLogitsCopy, samplingOptsBase);
    } catch (error) {
      const elapsedMs = performance.now() - prefillStart;
      return {
        label: 'logits-path',
        tokensGenerated: tokenIds.length,
        outputText: tokenTexts.join(''),
        tokenIds,
        prefillMs,
        decodeMs,
        firstTokenMs: firstTokenTs - prefillStart,
        firstResponseMs: firstTokenTs - prefillStart,
        totalMs: elapsedMs,
        tokensPerSec: tokenIds.length > 0 ? (tokenIds.length / elapsedMs) * 1000 : 0,
        error: error.message,
      };
    }
    decodeMs += performance.now() - decodeStepStart;
    tokenIds.push(nextToken);
    tokenTexts.push(toDecodeText(tokenizer, nextToken));
    contextTokens.push(nextToken);
    if (isStopToken(nextToken, tokenizer, modelConfig)) {
      break;
    }
  }

  const totalMs = performance.now() - totalStart;
  const firstTokenMs = firstTokenTs - prefillStart;
  const firstResponseMs = firstTokenMs;
  const decodedText = tokenTexts.join('');
  const decodeStepsMs = tokenIds.length > 1 ? decodeMs : 0;
  return {
    label: 'logits-path',
    tokensGenerated: tokenIds.length,
    outputText: decodedText,
    tokenIds,
    prefillMs,
    decodeMs: decodeStepsMs,
    firstTokenMs,
    firstResponseMs,
    totalMs,
    tokensPerSec: tokenIds.length > 0 ? (tokenIds.length / totalMs) * 1000 : 0,
  };
}

async function buildTeacherSequence(modelDir, manifest, runtimeConfig, prompt, options) {
  const run = await runWithFreshPipeline(modelDir, manifest, runtimeConfig, (pipeline) => runTokenPath(pipeline, prompt, {
    maxTokens: options.maxTokens,
    repetitionPenalty: options.repetitionPenalty,
    seed: options.seed,
  }));
  return {
    source: 'token-path',
    tokenIds: run.tokenIds,
    outputText: run.outputText,
  };
}

async function runTeacherForcedTokenPath(pipeline, prompt, options) {
  const teacherIds = options.teacherIds ?? [];
  const run = await runTokenPath(pipeline, prompt, {
    maxTokens: teacherIds.length,
    repetitionPenalty: options.repetitionPenalty,
    seed: options.seed,
  });
  for (let i = 0; i < teacherIds.length; i += 1) {
    if (run.tokenIds[i] !== teacherIds[i]) {
      throw new Error(
        `Teacher-forced token path diverged at step ${i + 1}: expected=${teacherIds[i]}, actual=${run.tokenIds[i]}`
      );
    }
  }
  return {
    ...run,
    label: 'token-path-teacher-forced',
    teacherForced: true,
  };
}

async function runTeacherForcedLogitsPath(pipeline, prompt, options) {
  const teacherIds = options.teacherIds ?? [];
  const penalty = Number.isFinite(options.repetitionPenalty)
    ? options.repetitionPenalty
    : 1.0;

  pipeline.reset();
  const prefillStart = performance.now();
  const prefillResult = await pipeline.prefillWithLogits(prompt, { useChatTemplate: false });
  const prefillMs = performance.now() - prefillStart;

  const tokenizer = pipeline.tokenizer;
  const samplingOptions = buildSamplingOptions(tokenizer, options.seed);
  const contextTokens = Array.isArray(prefillResult.tokens) ? [...prefillResult.tokens] : [];
  const tokenTexts = [];
  let decodeMs = 0;

  if (teacherIds.length > 0) {
    const firstLogits = Float32Array.from(prefillResult.logits || []);
    applyRepetitionPenalty(firstLogits, contextTokens, penalty);
    const actualFirstToken = sample(firstLogits, samplingOptions);
    if (actualFirstToken !== teacherIds[0]) {
      throw new Error(
        `Teacher-forced logits path diverged at step 1: expected=${teacherIds[0]}, actual=${actualFirstToken}`
      );
    }
    tokenTexts.push(toDecodeText(tokenizer, teacherIds[0]));
    contextTokens.push(teacherIds[0]);
  }

  const firstTokenMs = prefillMs;
  for (let i = 1; i < teacherIds.length; i += 1) {
    const expectedToken = teacherIds[i];
    const stepStart = performance.now();
    const stepResult = await pipeline.decodeStepLogits(contextTokens, {
      useChatTemplate: false,
      seed: options.seed,
    });
    const stepLogits = Float32Array.from(stepResult?.logits || []);
    releaseLogitsStepResult(stepResult);
    applyRepetitionPenalty(stepLogits, contextTokens, penalty);
    const actualToken = sample(stepLogits, samplingOptions);
    decodeMs += performance.now() - stepStart;
    if (actualToken !== expectedToken) {
      throw new Error(
        `Teacher-forced logits path diverged at step ${i + 1}: expected=${expectedToken}, actual=${actualToken}`
      );
    }
    tokenTexts.push(toDecodeText(tokenizer, expectedToken));
    contextTokens.push(expectedToken);
  }

  const totalMs = prefillMs + decodeMs;
  return {
    label: 'logits-path-teacher-forced',
    tokensGenerated: teacherIds.length,
    outputText: tokenTexts.join(''),
    tokenIds: [...teacherIds],
    prefillMs,
    decodeMs,
    firstTokenMs,
    firstResponseMs: firstTokenMs,
    totalMs,
    tokensPerSec: teacherIds.length > 0 ? (teacherIds.length / totalMs) * 1000 : 0,
    teacherForced: true,
  };
}

function runAggregateSummary(runs) {
  return {
    tokensGenerated: summarizeTimings(runs, 'tokensGenerated'),
    prefillMs: summarizeTimings(runs, 'prefillMs'),
    decodeMs: summarizeTimings(runs, 'decodeMs'),
    firstTokenMs: summarizeTimings(runs, 'firstTokenMs'),
    totalMs: summarizeTimings(runs, 'totalMs'),
    tokensPerSec: summarizeTimings(runs, 'tokensPerSec'),
  };
}

async function main() {
  let args;
  try {
    args = parseArgs(process.argv);
  } catch (error) {
    console.error(error.message);
    console.log(usage());
    process.exit(1);
  }

  if (args.help) {
    console.log(usage());
    return;
  }

  const bootstrapResult = await bootstrapNodeWebGPU();
  if (!bootstrapResult.ok) {
    throw new Error('WebGPU bootstrap failed. Install/enable a WebGPU runtime.');
  }

  const modelDir = path.resolve(args.modelDir);
  const manifest = parseManifest(
    readFileSync(path.join(modelDir, 'manifest.json'), 'utf8'),
  );
  const runtimeConfig = buildRuntimeConfig(args.maxTokens);
  await initDevice();
  const kernelCaps = getKernelCapabilities();
  const runConfig = {
    label: 'text-decode-paths',
    modelDir,
    prompt: args.prompt,
    maxTokens: args.maxTokens,
    warmupRuns: args.warmupRuns,
    timedRuns: args.timedRuns,
    seed: args.seed,
    mode: args.teacherForced ? 'teacher-forced' : 'free-generate',
    config: runtimeConfig,
  };

  const tokenPathRuns = [];
  const logitsPathRuns = [];
  let teacherSequence = null;

  if (args.teacherForced) {
    teacherSequence = await buildTeacherSequence(modelDir, manifest, runtimeConfig, args.prompt, {
      maxTokens: args.maxTokens,
    });
  }

  for (let i = 0; i < args.warmupRuns; i += 1) {
    const warmupTokenRun = args.teacherForced ? runTeacherForcedTokenPath : runTokenPath;
    const warmupLogitsRun = args.teacherForced ? runTeacherForcedLogitsPath : runLogitsPath;
    await runWithFreshPipeline(modelDir, manifest, runtimeConfig, (pipeline) => warmupTokenRun(pipeline, args.prompt, {
      maxTokens: args.maxTokens,
      teacherIds: teacherSequence?.tokenIds,
      repetitionPenalty: runtimeConfig.runtime.inference.sampling.repetitionPenalty,
      seed: args.seed,
    }));
    await runWithFreshPipeline(modelDir, manifest, runtimeConfig, (pipeline) => warmupLogitsRun(pipeline, args.prompt, {
      maxTokens: args.maxTokens,
      teacherIds: teacherSequence?.tokenIds,
      repetitionPenalty: runtimeConfig.runtime.inference.sampling.repetitionPenalty,
      seed: args.seed,
    }));
  }

  for (let i = 0; i < args.timedRuns; i += 1) {
    const timedTokenRun = args.teacherForced ? runTeacherForcedTokenPath : runTokenPath;
    const timedLogitsRun = args.teacherForced ? runTeacherForcedLogitsPath : runLogitsPath;
    tokenPathRuns.push(await runWithFreshPipeline(modelDir, manifest, runtimeConfig, (pipeline) => timedTokenRun(pipeline, args.prompt, {
      maxTokens: args.maxTokens,
      teacherIds: teacherSequence?.tokenIds,
      repetitionPenalty: runtimeConfig.runtime.inference.sampling.repetitionPenalty,
      seed: args.seed,
    })));
    logitsPathRuns.push(await runWithFreshPipeline(modelDir, manifest, runtimeConfig, (pipeline) => timedLogitsRun(pipeline, args.prompt, {
      maxTokens: args.maxTokens,
      teacherIds: teacherSequence?.tokenIds,
      repetitionPenalty: runtimeConfig.runtime.inference.sampling.repetitionPenalty,
      seed: args.seed,
    })));
  }

  const tokenSummary = runAggregateSummary(tokenPathRuns);
  const logitsSummary = runAggregateSummary(logitsPathRuns);

  const finalResult = {
    config: runConfig,
    env: {
      provider: bootstrapResult.provider,
      adapterInfo: kernelCaps.adapterInfo,
      hasSubgroups: kernelCaps.hasSubgroups,
      hasF16: kernelCaps.hasF16,
    },
    output: {
      tokenPath: {
        runs: tokenPathRuns,
        summary: tokenSummary,
      },
      logitsPath: {
        runs: logitsPathRuns,
        summary: logitsSummary,
      },
      sampleRuns: {
        tokenPath: tokenPathRuns[0]?.outputText ?? '',
        logitsPath: logitsPathRuns[0]?.outputText ?? '',
      },
    },
  };

  if (teacherSequence) {
    finalResult.output.teacherSequence = teacherSequence;
  }

  if (args.traceParity) {
    finalResult.output.parityTrace = await runWithFreshPipeline(
      modelDir,
      manifest,
      runtimeConfig,
      (pipeline) => traceParity(pipeline, args.prompt, {
        maxTokens: args.maxTokens,
        repetitionPenalty: runtimeConfig.runtime.inference.sampling.repetitionPenalty,
        seed: args.seed,
      })
    );
  }

  if (args.json) {
    console.log(JSON.stringify(finalResult, null, 2));
    return;
  }

  console.log(`Benchmark: ${runConfig.label}`);
  console.log(`Model dir: ${runConfig.modelDir}`);
  console.log(`Prompt: ${runConfig.prompt}`);
  console.log(`Token mode runs: ${tokenPathRuns.length}, Logits mode runs: ${logitsPathRuns.length}`);
  console.log('');
  console.log(`Sample token-path output: ${JSON.stringify(finalResult.output.sampleRuns.tokenPath)}`);
  console.log(formatRunResult(tokenPathRuns[0]).join('\n'));
  console.log('');
  console.log(`Sample logits-path output: ${JSON.stringify(finalResult.output.sampleRuns.logitsPath)}`);
  console.log(formatRunResult(logitsPathRuns[0]).join('\n'));
  console.log('');
  console.log('Averages:');
  console.log(`  token path decodeMs(mean): ${tokenSummary.decodeMs.meanMs.toFixed(2)}ms (p95=${tokenSummary.decodeMs.p95Ms.toFixed(2)}ms)`);
  console.log(`  logits path decodeMs(mean): ${logitsSummary.decodeMs.meanMs.toFixed(2)}ms (p95=${logitsSummary.decodeMs.p95Ms.toFixed(2)}ms)`);
  console.log(`  token path totalMs(mean): ${tokenSummary.totalMs.meanMs.toFixed(2)}ms (p95=${tokenSummary.totalMs.p95Ms.toFixed(2)}ms)`);
  console.log(`  logits path totalMs(mean): ${logitsSummary.totalMs.meanMs.toFixed(2)}ms (p95=${logitsSummary.totalMs.p95Ms.toFixed(2)}ms)`);
  console.log(`  token path tokens/s(mean): ${tokenSummary.tokensPerSec.meanMs.toFixed(2)}`);
  console.log(`  logits path tokens/s(mean): ${logitsSummary.tokensPerSec.meanMs.toFixed(2)}`);
}

await main().catch((error) => {
  console.error(error);
  process.exit(1);
});
