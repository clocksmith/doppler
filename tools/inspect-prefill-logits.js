#!/usr/bin/env node

import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { initializeInference } from '../src/inference/test-harness.js';
import { applyRuntimeProfile } from '../src/inference/browser-harness-runtime-helpers.js';
import { getTopK } from '../src/inference/pipelines/text/sampling.js';
import { installNodeFileFetchShim } from '../src/tooling/node-file-fetch.js';
import { bootstrapNodeWebGPU } from '../src/tooling/node-webgpu.js';

const DEFAULT_PROMPT = 'What color is the sky on a clear day? Answer in one word.';
const DEFAULT_RUNTIME_PROFILE = 'profiles/production';
const DEFAULT_TOP_K = 10;
const DEFAULT_WATCH_TOKENS = Object.freeze([1, 50, 105, 106, 107]);
const DEFAULT_PROMPT_TAIL = 16;

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

function parseTokenId(value) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`watch token id must be a non-negative integer, got "${value}".`);
  }
  return parsed;
}

function parseArgs(argv) {
  const parsed = {
    modelDir: null,
    modelId: null,
    prompt: DEFAULT_PROMPT,
    runtimeProfile: DEFAULT_RUNTIME_PROFILE,
    topK: DEFAULT_TOP_K,
    promptTail: DEFAULT_PROMPT_TAIL,
    useChatTemplate: true,
    watchTokens: [],
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
        throw new Error('--model-dir requires a path.');
      }
      parsed.modelDir = value;
      i += 1;
      continue;
    }
    if (arg === '--model-id') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--model-id requires a value.');
      }
      parsed.modelId = value;
      i += 1;
      continue;
    }
    if (arg === '--prompt') {
      const value = argv[i + 1];
      if (value === undefined) {
        throw new Error('--prompt requires a value.');
      }
      parsed.prompt = value;
      i += 1;
      continue;
    }
    if (arg === '--runtime-profile') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--runtime-profile requires a value.');
      }
      parsed.runtimeProfile = value;
      i += 1;
      continue;
    }
    if (arg === '--top-k') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--top-k requires a value.');
      }
      parsed.topK = parsePositiveInteger(value, '--top-k');
      i += 1;
      continue;
    }
    if (arg === '--prompt-tail') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--prompt-tail requires a value.');
      }
      parsed.promptTail = parsePositiveInteger(value, '--prompt-tail');
      i += 1;
      continue;
    }
    if (arg === '--watch-token') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--watch-token requires a value.');
      }
      parsed.watchTokens.push(parseTokenId(value));
      i += 1;
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
      'Usage: node tools/inspect-prefill-logits.js --model-dir <path> [options]',
      '',
      'Options:',
      '  --model-dir <path>         Local RDRR artifact directory (required)',
      '  --model-id <id>            Optional modelId label for the report',
      '  --prompt <text>            Prompt to prefill',
      `  --runtime-profile <id>     Runtime profile to apply (default: ${DEFAULT_RUNTIME_PROFILE})`,
      `  --top-k <n>                Number of top logits to print (default: ${DEFAULT_TOP_K})`,
      `  --prompt-tail <n>          Number of prompt tail tokens to include (default: ${DEFAULT_PROMPT_TAIL})`,
      '  --watch-token <id>         Additional token id to rank/logit-check (repeatable)',
      '  --no-chat-template         Disable chat template expansion for the probe',
      '  --help, -h                 Show this help',
      '',
      'Output is JSON.',
      '',
      'Example:',
      '  node tools/inspect-prefill-logits.js \\',
      '    --model-dir models/local/gemma-4-e2b-it-q4k-ehf16-af32 \\',
      '    --watch-token 106',
    ].join('\n')
  );
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

function summarizePromptTail(tokenizer, tokenIds, limit) {
  const normalized = Array.isArray(tokenIds)
    ? tokenIds
    : (ArrayBuffer.isView(tokenIds) ? Array.from(tokenIds) : []);
  const tail = normalized.slice(-limit);
  return tail.map((tokenId) => ({
    tokenId,
    text: decodeToken(tokenizer, tokenId),
  }));
}

function computeSoftmaxStats(logits) {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const value = logits[i];
    if (Number.isFinite(value) && value > maxLogit) {
      maxLogit = value;
    }
  }
  if (!Number.isFinite(maxLogit)) {
    return {
      maxLogit: null,
      sumExp: null,
    };
  }
  let sumExp = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = logits[i];
    if (!Number.isFinite(value)) {
      continue;
    }
    sumExp += Math.exp(value - maxLogit);
  }
  return {
    maxLogit,
    sumExp: Number.isFinite(sumExp) && sumExp > 0 ? sumExp : null,
  };
}

function summarizeWatchedToken(logits, tokenizer, tokenId, softmaxStats) {
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= logits.length) {
    return {
      tokenId,
      present: false,
      text: decodeToken(tokenizer, tokenId),
      rank: null,
      logit: null,
      prob: null,
    };
  }
  const logit = logits[tokenId];
  if (!Number.isFinite(logit)) {
    return {
      tokenId,
      present: true,
      text: decodeToken(tokenizer, tokenId),
      rank: null,
      logit,
      prob: null,
    };
  }

  let rank = 1;
  for (let i = 0; i < logits.length; i += 1) {
    if (i === tokenId) {
      continue;
    }
    const value = logits[i];
    if (!Number.isFinite(value)) {
      continue;
    }
    if (value > logit || (value === logit && i < tokenId)) {
      rank += 1;
    }
  }

  const prob = softmaxStats.maxLogit == null || softmaxStats.sumExp == null
    ? null
    : Math.exp(logit - softmaxStats.maxLogit) / softmaxStats.sumExp;

  return {
    tokenId,
    present: true,
    text: decodeToken(tokenizer, tokenId),
    rank,
    logit,
    prob,
  };
}

function summarizeSpecialTokens(tokenizer) {
  try {
    const specialTokens = tokenizer?.getSpecialTokens?.();
    return specialTokens && typeof specialTokens === 'object'
      ? specialTokens
      : null;
  } catch {
    return null;
  }
}

function resolveModelUrl(modelDir) {
  const absolutePath = path.resolve(modelDir);
  return pathToFileURL(absolutePath).href;
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help || !args.modelDir) {
    printHelp();
    process.exit(args.help ? 0 : 1);
  }

  installNodeFileFetchShim();
  const originalRuntime = cloneValue(getRuntimeConfig());

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

    const { pipeline, manifest, capabilities } = harness;
    const prefill = await pipeline.prefillWithLogits(args.prompt, {
      useChatTemplate: args.useChatTemplate,
    });
    const logits = prefill?.logits instanceof Float32Array
      ? prefill.logits
      : Float32Array.from(prefill?.logits ?? []);
    const promptTokenIds = Array.isArray(prefill?.tokens)
      ? prefill.tokens
      : (ArrayBuffer.isView(prefill?.tokens) ? Array.from(prefill.tokens) : []);
    const tokenizer = pipeline.tokenizer;
    const softmaxStats = computeSoftmaxStats(logits);
    const watchedTokenIds = Array.from(new Set([
      ...DEFAULT_WATCH_TOKENS,
      ...args.watchTokens,
    ])).sort((a, b) => a - b);
    const decode = (tokenIds) => decodeTokenIds(tokenizer, tokenIds);
    const stats = pipeline.getStats?.() ?? {};

    console.log(JSON.stringify({
      ok: true,
      modelId: manifest?.modelId ?? args.modelId ?? path.basename(args.modelDir),
      modelUrl,
      runtimeProfile: args.runtimeProfile,
      prompt: args.prompt,
      useChatTemplate: args.useChatTemplate,
      gpu: {
        provider: bootstrap.provider ?? null,
        capabilities,
      },
      kernelPathId: stats.kernelPathId ?? pipeline.resolvedKernelPath?.id ?? null,
      kernelPathSource: stats.kernelPathSource ?? pipeline.kernelPathSource ?? null,
      executionPlan: stats.executionPlan ?? null,
      promptTokens: {
        count: promptTokenIds.length,
        tail: summarizePromptTail(tokenizer, promptTokenIds, args.promptTail),
      },
      promptTailText: decodeTokenIds(
        tokenizer,
        promptTokenIds.slice(Math.max(0, promptTokenIds.length - args.promptTail))
      ),
      specialTokens: summarizeSpecialTokens(tokenizer),
      encodedMarkers: {
        bos: resolveSingleTokenId(tokenizer, '<bos>'),
        eos: resolveSingleTokenId(tokenizer, '<eos>'),
        turnOpen: resolveSingleTokenId(tokenizer, '<|turn>'),
        turnClose: resolveSingleTokenId(tokenizer, '<turn|>'),
      },
      topK: getTopK(logits, args.topK, decode).map((entry) => ({
        tokenId: entry.token,
        logit: entry.logit,
        prob: entry.prob,
        text: entry.text,
      })),
      watchedTokens: watchedTokenIds.map((tokenId) => (
        summarizeWatchedToken(logits, tokenizer, tokenId, softmaxStats)
      )),
      stats: {
        prefillMs: stats.prefillMs ?? null,
        firstTokenMs: stats.firstTokenMs ?? null,
        modelLoadMs: stats.modelLoadMs ?? null,
        prefillTokens: stats.prefillTokens ?? null,
        prefillTokensPerSec: stats.prefillTokensPerSec ?? null,
      },
    }, null, 2));
  } finally {
    try {
      await harness?.pipeline?.unload?.();
    } catch {
      // Best-effort cleanup for an internal diagnostic tool.
    }
    try {
      harness?.pipeline?.releaseGPUResources?.();
    } catch {
      // Best-effort cleanup for an internal diagnostic tool.
    }
    setRuntimeConfig(originalRuntime);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
