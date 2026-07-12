#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { activateLoRAFromTrainingOutputForPipeline } from '../src/client/runtime/lora.js';
import { destroyDevice, resetDeviceState } from '../src/gpu/device.js';
import { initializeInference } from '../src/inference/test-harness.js';
import { applyRuntimeProfile } from '../src/inference/browser-harness-runtime-helpers.js';
import { getTopK } from '../src/inference/pipelines/text/sampling.js';
import { destroyBufferPool } from '../src/memory/buffer-pool.js';
import { installNodeFileFetchShim } from '../src/tooling/node-file-fetch.js';
import { bootstrapNodeWebGPU } from '../src/tooling/node-webgpu.js';

const ROOT = path.resolve(import.meta.dirname, '..');
const DEFAULT_PROMPT = 'What color is the sky on a clear day? Answer in one word.';
const DEFAULT_RUNTIME_PROFILE = 'profiles/production';
const DEFAULT_TOP_K = 10;
const DEFAULT_WATCH_TOKENS = Object.freeze([1, 50, 105, 106, 107]);
const DEFAULT_PROMPT_TAIL = 16;
const DEFAULT_MAX_TOKENS = 16;
const DIAGNOSTIC_LEVELS = new Set(['none', 'metadata', 'slice', 'full']);
const DETERMINISTIC_SAMPLING = Object.freeze({
  temperature: 0,
  topP: 1,
  topK: 1,
  repetitionPenalty: 1,
  greedyThreshold: 0,
});
const HOST_BYTE_ORDER = new Uint8Array(new Uint16Array([0x00ff]).buffer)[0] === 0xff
  ? 'little-endian'
  : 'big-endian';

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
    maxTokens: DEFAULT_MAX_TOKENS,
    useChatTemplate: true,
    watchTokens: [],
    outputPath: null,
    logitsOutputPath: null,
    adapterManifestPath: null,
    diagnosticsLevel: 'none',
    diagnosticsLayers: [],
    diagnosticsOpIds: [],
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
    if (arg === '--max-tokens') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--max-tokens requires a value.');
      }
      parsed.maxTokens = parsePositiveInteger(value, '--max-tokens');
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
    if (arg === '--out') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--out requires a path.');
      }
      parsed.outputPath = value;
      i += 1;
      continue;
    }
    if (arg === '--logits-out') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--logits-out requires a path.');
      }
      parsed.logitsOutputPath = value;
      i += 1;
      continue;
    }
    if (arg === '--adapter-manifest') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--adapter-manifest requires a path.');
      }
      parsed.adapterManifestPath = value;
      i += 1;
      continue;
    }
    if (arg === '--diagnostics-level') {
      const value = argv[i + 1];
      if (!DIAGNOSTIC_LEVELS.has(value)) {
        throw new Error('--diagnostics-level must be one of: none, metadata, slice, full.');
      }
      parsed.diagnosticsLevel = value;
      i += 1;
      continue;
    }
    if (arg === '--diagnostics-layer') {
      const value = argv[i + 1];
      const layer = Number(value);
      if (!Number.isInteger(layer) || layer < 0) {
        throw new Error('--diagnostics-layer must be a non-negative integer.');
      }
      parsed.diagnosticsLayers.push(layer);
      i += 1;
      continue;
    }
    if (arg === '--diagnostics-op') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('--diagnostics-op requires an operator id.');
      }
      parsed.diagnosticsOpIds.push(value);
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
      `  --max-tokens <n>           Deterministic generation token limit (default: ${DEFAULT_MAX_TOKENS})`,
      '  --watch-token <id>         Additional token id to rank/logit-check (repeatable)',
      '  --out <path>               Write the JSON receipt to a file',
      '  --logits-out <path>        Write the full first-token Float32 logits',
      '  --adapter-manifest <path>  Activate a local Doppler LoRA manifest before inference',
      '  --diagnostics-level <level> Capture operator boundaries: none, metadata, slice, or full',
      '  --diagnostics-layer <n>     Restrict detailed captures to a layer (repeatable)',
      '  --diagnostics-op <id>       Restrict detailed captures to an operator id (repeatable)',
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

function summarizeLogits(logits) {
  const bytes = Buffer.from(logits.buffer, logits.byteOffset, logits.byteLength);
  let finiteCount = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const value of logits) {
    if (!Number.isFinite(value)) continue;
    finiteCount += 1;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return {
    elementCount: logits.length,
    byteLength: logits.byteLength,
    dtype: 'float32',
    byteOrder: HOST_BYTE_ORDER,
    finiteCount,
    nonFiniteCount: logits.length - finiteCount,
    min: finiteCount > 0 ? min : null,
    max: finiteCount > 0 ? max : null,
    sha256: createHash('sha256').update(bytes).digest('hex'),
  };
}

async function writeLogitsCapture(logits, outputPath) {
  if (!outputPath) return null;
  const absolutePath = path.resolve(outputPath);
  await mkdir(path.dirname(absolutePath), { recursive: true });
  const bytes = Buffer.from(logits.buffer, logits.byteOffset, logits.byteLength);
  await writeFile(absolutePath, bytes);
  return {
    path: outputPath,
  };
}

function sha256Bytes(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function inspectArtifactFiles(modelDir) {
  const absoluteDir = path.resolve(modelDir);
  const files = {};
  let manifestSummary = null;
  for (const filename of ['manifest.json', 'tokenizer.json', 'origin.json']) {
    try {
      const bytes = await readFile(path.join(absoluteDir, filename));
      files[filename] = {
        bytes: bytes.byteLength,
        sha256: sha256Bytes(bytes),
      };
      if (filename === 'manifest.json') {
        const manifest = JSON.parse(bytes.toString('utf8'));
        manifestSummary = {
          modelId: manifest.modelId ?? null,
          artifactIdentity: manifest.artifactIdentity ?? null,
          shardCount: Array.isArray(manifest.shards) ? manifest.shards.length : null,
          totalSize: manifest.totalSize ?? null,
        };
      }
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      files[filename] = null;
    }
  }
  return { files, manifestSummary };
}

async function inspectAdapterManifest(manifestPath) {
  const absoluteManifestPath = path.resolve(manifestPath);
  const manifestBytes = await readFile(absoluteManifestPath);
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const weightsPath = typeof manifest.weightsPath === 'string' && manifest.weightsPath
    ? path.resolve(path.dirname(absoluteManifestPath), manifest.weightsPath)
    : null;
  const weightsBytes = weightsPath ? await readFile(weightsPath) : null;
  const weightsSha256 = weightsBytes ? sha256Bytes(weightsBytes) : null;
  return {
    manifestPath,
    manifestSha256: sha256Bytes(manifestBytes),
    weightsPath: weightsPath
      ? path.join(path.dirname(manifestPath), manifest.weightsPath)
      : null,
    weightsBytes: weightsBytes?.byteLength ?? null,
    weightsSha256,
    declaredWeightsSha256: manifest.checksum ?? null,
    declaredWeightsMatch: manifest.checksum
      ? manifest.checksum === weightsSha256
      : null,
    identity: {
      id: manifest.id ?? null,
      name: manifest.name ?? null,
      baseModel: manifest.baseModel ?? null,
      rank: manifest.rank ?? null,
      alpha: manifest.alpha ?? null,
      targetModules: manifest.targetModules ?? null,
    },
  };
}

function resolveDopplerCommit() {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: ROOT,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return null;
  }
}

function runHostCommand(command, args) {
  try {
    return execFileSync(command, args, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return null;
  }
}

function inspectMacMetal() {
  if (process.platform !== 'darwin') return null;
  const raw = runHostCommand('system_profiler', ['SPDisplaysDataType', '-json']);
  let adapter = null;
  try {
    const entry = JSON.parse(raw)?.SPDisplaysDataType?.[0] ?? null;
    if (entry) {
      adapter = {
        name: entry._name ?? null,
        model: entry.sppci_model ?? null,
        gpuCoreCount: Number(entry.sppci_cores) || null,
        metalFamilySupport: entry.spdisplays_mtlgpufamilysupport ?? null,
      };
    }
  } catch {
    adapter = null;
  }
  return {
    productVersion: runHostCommand('sw_vers', ['-productVersion']),
    buildVersion: runHostCommand('sw_vers', ['-buildVersion']),
    adapter,
  };
}

function inspectHost() {
  const cpus = os.cpus();
  return {
    platform: process.platform,
    arch: process.arch,
    osRelease: os.release(),
    nodeVersion: process.version,
    cpuModel: cpus?.[0]?.model ?? null,
    logicalCpuCount: cpus?.length ?? null,
    totalMemoryBytes: os.totalmem(),
    metal: inspectMacMetal(),
  };
}

function buildDiagnosticsOptions(args) {
  if (args.diagnosticsLevel === 'none') return null;
  const hasTargets = args.diagnosticsLayers.length > 0 || args.diagnosticsOpIds.length > 0;
  return {
    enabled: true,
    captureConfig: {
      enabled: true,
      defaultLevel: hasTargets ? 'none' : args.diagnosticsLevel,
      targetLevel: args.diagnosticsLevel,
      targetLayers: args.diagnosticsLayers,
      targetOpIds: args.diagnosticsOpIds,
    },
  };
}

async function generateDeterministically(pipeline, args) {
  pipeline.reset();
  const tokenIds = [];
  const chunks = [];
  for await (const chunk of pipeline.generate(args.prompt, {
    ...DETERMINISTIC_SAMPLING,
    maxTokens: args.maxTokens,
    useChatTemplate: args.useChatTemplate,
    benchmark: false,
    diagnostics: buildDiagnosticsOptions(args) ?? undefined,
    onToken(tokenId) {
      tokenIds.push(tokenId);
    },
  })) {
    chunks.push(String(chunk));
  }
  const outputRaw = chunks.join('');
  const stats = pipeline.getStats?.() ?? {};
  return {
    maxTokens: args.maxTokens,
    sampling: DETERMINISTIC_SAMPLING,
    tokenIds,
    firstTokenId: tokenIds[0] ?? null,
    outputRaw,
    output: outputRaw.trim(),
    stopReason: stats.stopReason ?? null,
    stopTokenId: stats.stopTokenId ?? null,
    stats: {
      firstTokenMs: stats.firstTokenMs ?? stats.ttftMs ?? null,
      prefillMs: stats.prefillMs ?? stats.prefillTimeMs ?? null,
      decodeMs: stats.decodeMs ?? stats.decodeTimeMs ?? null,
      totalMs: stats.totalMs ?? stats.totalTimeMs ?? null,
      tokensGenerated: stats.tokensGenerated ?? tokenIds.length,
    },
    operatorDiagnostics: stats.operatorDiagnostics ?? null,
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

async function writeReport(report, outputPath) {
  const json = `${JSON.stringify(report, null, 2)}\n`;
  if (!outputPath) {
    console.log(json.trimEnd());
    return null;
  }
  const absoluteOutputPath = path.resolve(outputPath);
  await mkdir(path.dirname(absoluteOutputPath), { recursive: true });
  await writeFile(absoluteOutputPath, json, 'utf8');
  console.log(absoluteOutputPath);
  return absoluteOutputPath;
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help || !args.modelDir) {
    printHelp();
    process.exit(args.help ? 0 : 1);
  }

  installNodeFileFetchShim();
  const originalRuntime = cloneValue(getRuntimeConfig());
  const modelUrl = resolveModelUrl(args.modelDir);
  const artifactInspection = await inspectArtifactFiles(args.modelDir);
  const baseReport = {
    artifactKind: 'first_token_inference_receipt',
    schemaVersion: 1,
    recordedAt: new Date().toISOString(),
    invocation: {
      workingDirectory: 'repository root',
      executable: 'node',
      script: path.relative(ROOT, path.resolve(process.argv[1])),
      argv: process.argv.slice(2),
      command: [
        'node',
        path.relative(ROOT, path.resolve(process.argv[1])),
        ...process.argv.slice(2),
      ],
    },
    dopplerCommit: resolveDopplerCommit(),
    host: inspectHost(),
    modelId: artifactInspection.manifestSummary?.modelId
      ?? args.modelId
      ?? path.basename(args.modelDir),
    modelSource: {
      kind: 'local_directory',
      path: args.modelDir,
    },
    artifactIdentity: artifactInspection.manifestSummary?.artifactIdentity ?? null,
    artifactFiles: artifactInspection.files,
    artifactSummary: artifactInspection.manifestSummary,
    runtimeProfile: args.runtimeProfile,
  };

  let harness = null;
  let phase = 'runtime-profile';
  let runtimeConfig = cloneValue(originalRuntime);
  let bootstrap = null;
  let adapter = null;
  try {
    if (args.runtimeProfile) {
      await applyRuntimeProfile(args.runtimeProfile);
    }

    runtimeConfig = cloneValue(getRuntimeConfig());
    phase = 'webgpu-bootstrap';
    bootstrap = await bootstrapNodeWebGPU();
    if (!bootstrap?.ok) {
      throw new Error(`WebGPU bootstrap failed: ${bootstrap?.detail ?? 'unknown error'}`);
    }

    phase = 'model-load';
    harness = await initializeInference(modelUrl, {
      modelId: args.modelId,
      runtime: { runtimeConfig },
    });

    const { pipeline, manifest, capabilities } = harness;
    if (args.adapterManifestPath) {
      phase = 'adapter-inspection';
      const artifact = await inspectAdapterManifest(args.adapterManifestPath);
      baseReport.adapter = { artifact, activation: null };
      if (artifact.declaredWeightsMatch === false) {
        throw new Error(
          `Adapter checksum mismatch: declared ${artifact.declaredWeightsSha256}, got ${artifact.weightsSha256}.`
        );
      }
      phase = 'adapter-activation';
      const activation = await activateLoRAFromTrainingOutputForPipeline(pipeline, {
        adapterManifestPath: artifact.manifestPath,
      });
      adapter = { artifact, activation };
      baseReport.adapter = adapter;
      if (activation.activated !== true) {
        throw new Error(`Adapter activation failed: ${activation.reason ?? 'unknown reason'}.`);
      }
    }
    phase = 'prefill';
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
    const topK = getTopK(logits, args.topK, decode).map((entry) => ({
      tokenId: entry.token,
      logit: entry.logit,
      prob: entry.prob,
      text: entry.text,
    }));
    const logitsCapture = {
      ...summarizeLogits(logits),
      ...(await writeLogitsCapture(logits, args.logitsOutputPath)),
    };
    const prefillStats = {
      prefillMs: stats.prefillMs ?? stats.prefillTimeMs ?? null,
      firstTokenMs: stats.firstTokenMs ?? stats.ttftMs ?? null,
      modelLoadMs: stats.modelLoadMs ?? null,
      prefillTokens: stats.prefillTokens ?? null,
      prefillTokensPerSec: stats.prefillTokensPerSec ?? null,
    };
    phase = 'generation';
    const generation = await generateDeterministically(pipeline, args);
    const selectedToken = topK[0] ?? null;

    const report = {
      ...baseReport,
      ok: true,
      modelId: manifest?.modelId ?? args.modelId ?? path.basename(args.modelDir),
      artifactIdentity: manifest?.artifactIdentity ?? null,
      adapter,
      resolvedRuntimeConfig: runtimeConfig,
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
        ids: promptTokenIds,
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
      logitsCapture,
      topK,
      selectedToken,
      watchedTokens: watchedTokenIds.map((tokenId) => (
        summarizeWatchedToken(logits, tokenizer, tokenId, softmaxStats)
      )),
      prefillStats,
      generation,
      parity: {
        selectedTokenMatchesGeneratedFirstToken:
          selectedToken?.tokenId === generation.firstTokenId,
      },
    };
    phase = 'receipt-write';
    await writeReport(report, args.outputPath);
  } catch (error) {
    const failure = {
      ...baseReport,
      ok: false,
      failurePhase: phase,
      resolvedRuntimeConfig: runtimeConfig,
      gpu: bootstrap ? {
        provider: bootstrap.provider ?? null,
        detail: bootstrap.detail ?? null,
      } : null,
      error: {
        name: error?.name ?? 'Error',
        message: error instanceof Error ? error.message : String(error),
        code: error?.code ?? null,
      },
    };
    if (args.outputPath) {
      await writeReport(failure, args.outputPath);
    }
    throw error;
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
    try {
      destroyBufferPool();
    } finally {
      try {
        destroyDevice();
      } finally {
        resetDeviceState();
      }
    }
    setRuntimeConfig(originalRuntime);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
