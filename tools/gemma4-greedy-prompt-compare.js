#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { performance } from 'node:perf_hooks';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { getRuntimeConfig, setRuntimeConfig } from '../src/config/runtime.js';
import { mergeRuntimeValues } from '../src/config/runtime-merge.js';
import { initializeInference } from '../src/inference/test-harness.js';
import { installNodeFileFetchShim } from '../src/tooling/node-file-fetch.js';
import { bootstrapNodeWebGPU } from '../src/tooling/node-webgpu.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_MODEL_DIR = 'models/local/gemma-4-e2b-it-q4k-ehf16-af32';
const DEFAULT_MODEL_ID = 'gemma-4-e2b-it-q4k-ehf16-af32';
const DEFAULT_TJS_MODEL_ID = 'onnx-community/gemma-4-E2B-it-ONNX';
const DEFAULT_PROMPT_PACK = path.join('tools', 'data', 'gemma4-e2b-blog-prompts-512.json');
const DEFAULT_RUNTIME_PROFILE = 'profiles/production';
const DEFAULT_MAX_TOKENS = 1;
const DEFAULT_TIMEOUT_MS = 600_000;
const DEFAULT_TJS_DTYPE = 'q4f16';
const DEFAULT_TJS_FORMAT = 'onnx';
const DEFAULT_BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-dawn-features=disallow_unsafe_apis',
  '--ignore-gpu-blocklist',
  '--disable-breakpad',
  '--disable-gpu-sandbox',
  '--no-sandbox',
  '--use-angle=vulkan',
  '--enable-features=Vulkan',
  '--disable-vulkan-surface',
]);

function cloneValue(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
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

function parsePositiveInteger(value, label, fallback = null) {
  if (value == null) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function normalizePromptPackEntry(entry, index) {
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    throw new Error(`promptPack[${index}] must be an object.`);
  }
  const id = typeof entry.id === 'string' && entry.id.trim() !== ''
    ? entry.id.trim()
    : `prompt-${index + 1}`;
  if (typeof entry.text !== 'string' || entry.text.length === 0) {
    throw new Error(`promptPack[${index}] must include non-empty text.`);
  }
  return {
    id,
    text: entry.text,
  };
}

function parseArgs(argv) {
  const parsed = {
    modelDir: DEFAULT_MODEL_DIR,
    modelId: DEFAULT_MODEL_ID,
    tjsModelId: DEFAULT_TJS_MODEL_ID,
    promptPack: DEFAULT_PROMPT_PACK,
    runtimeProfile: DEFAULT_RUNTIME_PROFILE,
    maxTokens: DEFAULT_MAX_TOKENS,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    useChatTemplate: true,
    tjsDtype: DEFAULT_TJS_DTYPE,
    tjsFormat: DEFAULT_TJS_FORMAT,
    outDir: null,
    maxPrompts: null,
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
    if (arg === '--tjs-model-id') {
      const value = argv[i + 1];
      if (!value) throw new Error('--tjs-model-id requires a value.');
      parsed.tjsModelId = value;
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
    if (arg === '--max-tokens') {
      parsed.maxTokens = parsePositiveInteger(argv[i + 1], '--max-tokens');
      i += 1;
      continue;
    }
    if (arg === '--timeout-ms') {
      parsed.timeoutMs = parsePositiveInteger(argv[i + 1], '--timeout-ms');
      i += 1;
      continue;
    }
    if (arg === '--max-prompts') {
      parsed.maxPrompts = parsePositiveInteger(argv[i + 1], '--max-prompts');
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
    if (arg === '--tjs-dtype') {
      const value = argv[i + 1];
      if (!value) throw new Error('--tjs-dtype requires a value.');
      parsed.tjsDtype = value;
      i += 1;
      continue;
    }
    if (arg === '--tjs-format') {
      const value = argv[i + 1];
      if (!value) throw new Error('--tjs-format requires a value.');
      parsed.tjsFormat = value;
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
      'Usage: node tools/gemma4-greedy-prompt-compare.js [options]',
      '',
      'Options:',
      `  --model-dir <path>         Doppler local artifact directory (default: ${DEFAULT_MODEL_DIR})`,
      `  --model-id <id>            Doppler model id label (default: ${DEFAULT_MODEL_ID})`,
      `  --tjs-model-id <id>        Transformers.js model id (default: ${DEFAULT_TJS_MODEL_ID})`,
      `  --prompt-pack <path>       Prompt pack JSON path (default: ${DEFAULT_PROMPT_PACK})`,
      `  --runtime-profile <id>     Doppler runtime profile (default: ${DEFAULT_RUNTIME_PROFILE})`,
      `  --max-tokens <n>           Greedy decode length per prompt (default: ${DEFAULT_MAX_TOKENS})`,
      `  --timeout-ms <n>           Browser/page timeout (default: ${DEFAULT_TIMEOUT_MS})`,
      '  --max-prompts <n>          Optional prompt-count cap for smoke/debug runs',
      '  --out-dir <path>           Output directory (default: reports/gemma4-greedy-compare/<timestamp>)',
      `  --tjs-dtype <dtype>        Transformers.js dtype (default: ${DEFAULT_TJS_DTYPE})`,
      `  --tjs-format <fmt>         Transformers.js format (default: ${DEFAULT_TJS_FORMAT})`,
      '  --no-chat-template         Disable chat template expansion on both engines',
      '  --help, -h                 Show this help',
    ].join('\n')
  );
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function loadPromptPack(promptPackPath, maxPrompts = null) {
  const raw = JSON.parse(fs.readFileSync(promptPackPath, 'utf8'));
  const prompts = Array.isArray(raw)
    ? raw
    : (Array.isArray(raw?.promptCandidates) ? raw.promptCandidates : null);
  if (!Array.isArray(prompts) || prompts.length < 1) {
    throw new Error(`Prompt pack "${promptPackPath}" must contain a non-empty array.`);
  }
  const normalized = prompts.map(normalizePromptPackEntry);
  if (maxPrompts == null) {
    return normalized;
  }
  return normalized.slice(0, Math.min(maxPrompts, normalized.length));
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

function resolveModelUrl(modelDir) {
  return pathToFileURL(path.resolve(modelDir)).href;
}

function resolveRuntimeProfilePath(profileId) {
  const normalized = String(profileId ?? '').trim().replace(/\.json$/u, '');
  if (!normalized) {
    throw new Error('runtime profile id is required');
  }
  return path.join(REPO_ROOT, 'src', 'config', 'runtime', `${normalized}.json`);
}

function loadRuntimeProfileConfig(profileId, stack = []) {
  const profilePath = resolveRuntimeProfilePath(profileId);
  const resolvedPath = path.resolve(profilePath);
  if (stack.includes(resolvedPath)) {
    throw new Error(`Runtime profile extends cycle: ${[...stack, resolvedPath].join(' -> ')}`);
  }
  const parsed = JSON.parse(fs.readFileSync(resolvedPath, 'utf8'));
  const runtime = parsed?.runtime;
  if (!runtime || typeof runtime !== 'object' || Array.isArray(runtime)) {
    throw new Error(`Runtime profile "${profileId}" is missing runtime.`);
  }
  const extendsRefs = Array.isArray(parsed.extends)
    ? parsed.extends
    : (typeof parsed.extends === 'string' ? [parsed.extends] : []);
  let mergedRuntime = null;
  for (const ref of extendsRefs) {
    const baseRuntime = loadRuntimeProfileConfig(ref, [...stack, resolvedPath]);
    mergedRuntime = mergedRuntime == null ? baseRuntime : mergeRuntimeValues(mergedRuntime, baseRuntime);
  }
  return mergedRuntime == null ? runtime : mergeRuntimeValues(mergedRuntime, runtime);
}

async function runDopplerPromptPack(args, promptPack) {
  installNodeFileFetchShim();
  const bootstrap = await bootstrapNodeWebGPU();
  if (!bootstrap?.ok) {
    throw new Error(`WebGPU bootstrap failed: ${bootstrap?.detail ?? 'unknown error'}`);
  }

  const runtimeConfig = cloneValue(getRuntimeConfig());
  const harness = await initializeInference(resolveModelUrl(args.modelDir), {
    modelId: args.modelId,
    runtime: { runtimeConfig },
  });
  const { pipeline, capabilities, manifest } = harness;
  const tokenizer = pipeline.tokenizer;
  const startedAtMs = performance.now();

  try {
    const results = [];
    for (const prompt of promptPack) {
      pipeline.reset();
      const generated = await pipeline.generateTokenIds(prompt.text, {
        useChatTemplate: args.useChatTemplate,
        maxTokens: args.maxTokens,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
      });
      const tokenIds = Array.isArray(generated?.tokenIds)
        ? generated.tokenIds.map((tokenId) => Number(tokenId)).filter((tokenId) => Number.isInteger(tokenId))
        : [];
      results.push({
        id: prompt.id,
        text: prompt.text,
        generatedTokenIds: tokenIds,
        firstTokenId: tokenIds[0] ?? null,
        tokenTexts: tokenIds.map((tokenId) => decodeToken(tokenizer, tokenId)),
        generatedText: decodeTokenIds(tokenizer, tokenIds),
      });
    }

    return {
      modelId: args.modelId,
      runtimeProfile: args.runtimeProfile,
      maxTokens: args.maxTokens,
      useChatTemplate: args.useChatTemplate,
      durationMs: performance.now() - startedAtMs,
      capabilities,
      manifestModelType: manifest?.modelType ?? null,
      results,
      tokenizer,
    };
  } finally {
    try {
      await pipeline?.unload?.();
    } catch { /* ignore */ }
  }
}

async function createStaticServer(root) {
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.onnx': 'application/octet-stream',
  };

  const serveRequest = async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const safePath = decodeURIComponent(url.pathname).replace(/^\/+/, '');
    const filePath = path.resolve(root, safePath || '');
    const rootPath = path.resolve(root);
    const normalizedRoot = rootPath.endsWith(path.sep) ? rootPath : `${rootPath}${path.sep}`;

    try {
      if (!filePath.startsWith(normalizedRoot)) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }
      const stat = await fs.promises.stat(filePath);
      if (!stat.isFile()) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      const ext = path.extname(filePath);
      const contentType = mimeTypes[ext] || 'application/octet-stream';
      const data = await fs.promises.readFile(filePath);
      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': data.byteLength,
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'credentialless',
      });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end('Not found');
    }
  };

  return await new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      serveRequest(req, res).catch((error) => {
        res.writeHead(500);
        res.end(String(error?.message ?? error));
      });
    });
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = Number.isFinite(address?.port) ? address.port : 0;
      resolve({
        server,
        baseUrl: `http://127.0.0.1:${port}`,
      });
    });
  });
}

function closeStaticServer(server) {
  if (!server) return;
  try {
    server.close();
  } catch { /* ignore */ }
}

async function runTjsPromptPack(args, promptPack) {
  const { chromium } = await import('playwright');
  const staticServer = await createStaticServer(REPO_ROOT);
  const browser = await chromium.launch({
    headless: true,
    args: [...DEFAULT_BROWSER_ARGS],
  });
  const context = await browser.newContext();
  const page = await context.newPage();
  page.setDefaultTimeout(args.timeoutMs);
  page.setDefaultNavigationTimeout(args.timeoutMs);

  try {
    const runnerUrl = new URL('/benchmarks/runners/transformersjs-runner.html', staticServer.baseUrl);
    runnerUrl.search = new URLSearchParams({ v: '4' }).toString();
    await page.goto(runnerUrl.toString(), { timeout: args.timeoutMs });
    await page.waitForFunction(() => window.__tfjsReady === true, { timeout: args.timeoutMs });
    return await page.evaluate(
      async (config) => window.__runGreedyPromptPack(config),
      {
        modelId: args.tjsModelId,
        promptPack,
        maxNewTokens: args.maxTokens,
        useChatTemplate: args.useChatTemplate,
        sampling: {
          temperature: 0,
          topK: 1,
          topP: 1,
        },
        dtype: args.tjsDtype,
        format: args.tjsFormat,
      }
    );
  } finally {
    try {
      await context.close();
    } catch { /* ignore */ }
    try {
      await browser.close();
    } catch { /* ignore */ }
    closeStaticServer(staticServer.server);
  }
}

function compareTokenSequences(left, right) {
  const maxLength = Math.max(left.length, right.length);
  let firstMismatchStep = null;
  let samePrefixLength = 0;
  for (let i = 0; i < maxLength; i += 1) {
    if (left[i] !== right[i]) {
      firstMismatchStep = i;
      break;
    }
    samePrefixLength += 1;
  }
  return {
    sameFirstToken: left[0] != null && left[0] === right[0],
    sameAllTokens: left.length === right.length && left.every((tokenId, index) => tokenId === right[index]),
    firstMismatchStep,
    samePrefixLength,
  };
}

function buildMarkdownSummary(report) {
  const lines = [];
  lines.push('# gemma4 greedy prompt compare');
  lines.push('');
  lines.push(`- prompt pack: \`${report.promptPackPath}\``);
  lines.push(`- prompt count: ${report.aggregate.promptCount}`);
  lines.push(`- max tokens: ${report.maxTokens}`);
  lines.push(`- chat template: ${report.useChatTemplate}`);
  lines.push(`- same first token: ${report.aggregate.sameFirstTokenCount}`);
  lines.push(`- same full token sequence: ${report.aggregate.sameFullTokenSequenceCount}`);
  lines.push(`- first-token mismatches: ${report.aggregate.firstTokenMismatchCount}`);
  lines.push('');
  lines.push('## Sample mismatches');
  lines.push('');
  if (report.highlights.mismatches.length === 0) {
    lines.push('No mismatches observed.');
  } else {
    for (const entry of report.highlights.mismatches) {
      lines.push(`- \`${entry.id}\``);
      lines.push(`  prompt: ${entry.text}`);
      lines.push(`  doppler: ${entry.dopplerText}`);
      lines.push(`  tjs: ${entry.tjsText}`);
    }
  }
  return `${lines.join('\n')}\n`;
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    printHelp();
    return;
  }

  const originalRuntimeConfig = cloneValue(getRuntimeConfig());
  const outputDir = path.resolve(args.outDir ?? path.join('reports', 'gemma4-greedy-compare', timestampLabel()));
  ensureDir(outputDir);

  try {
    if (args.runtimeProfile) {
      const profileRuntime = loadRuntimeProfileConfig(args.runtimeProfile);
      setRuntimeConfig(mergeRuntimeValues(getRuntimeConfig(), profileRuntime));
    }

    const promptPackPath = path.resolve(args.promptPack);
    const promptPack = loadPromptPack(promptPackPath, args.maxPrompts);
    const doppler = await runDopplerPromptPack(args, promptPack);
    const tjs = await runTjsPromptPack(args, promptPack);
    const tjsResultsById = new Map((tjs.results ?? []).map((entry) => [entry.id, entry]));

    const comparedPrompts = doppler.results.map((dopplerEntry) => {
      const tjsEntry = tjsResultsById.get(dopplerEntry.id) ?? null;
      const tjsTokenIds = Array.isArray(tjsEntry?.generatedTokenIds)
        ? tjsEntry.generatedTokenIds.map((tokenId) => Number(tokenId)).filter((tokenId) => Number.isInteger(tokenId))
        : [];
      const compare = compareTokenSequences(dopplerEntry.generatedTokenIds, tjsTokenIds);
      return {
        id: dopplerEntry.id,
        text: dopplerEntry.text,
        doppler: dopplerEntry,
        tjs: tjsEntry == null
          ? null
          : {
              ...tjsEntry,
              tokenTexts: tjsTokenIds.map((tokenId) => decodeToken(doppler.tokenizer, tokenId)),
              generatedText: decodeTokenIds(doppler.tokenizer, tjsTokenIds),
            },
        compare,
      };
    });

    const sameFirstTokenCount = comparedPrompts.filter((entry) => entry.compare.sameFirstToken).length;
    const sameFullTokenSequenceCount = comparedPrompts.filter((entry) => entry.compare.sameAllTokens).length;
    const mismatches = comparedPrompts.filter((entry) => entry.compare.sameFirstToken === false);

    const report = {
      schemaVersion: 1,
      source: 'gemma4-greedy-prompt-compare',
      promptPackPath,
      promptPack,
      maxTokens: args.maxTokens,
      useChatTemplate: args.useChatTemplate,
      doppler: {
        modelId: doppler.modelId,
        runtimeProfile: doppler.runtimeProfile,
        durationMs: doppler.durationMs,
        capabilities: doppler.capabilities,
        manifestModelType: doppler.manifestModelType,
      },
      tjs: {
        modelId: args.tjsModelId,
        env: tjs.env ?? null,
        generationConfig: tjs.generationConfig ?? null,
      },
      aggregate: {
        promptCount: comparedPrompts.length,
        sameFirstTokenCount,
        sameFullTokenSequenceCount,
        firstTokenMismatchCount: mismatches.length,
      },
      highlights: {
        mismatches: mismatches.slice(0, 40).map((entry) => ({
          id: entry.id,
          text: entry.text,
          dopplerText: entry.doppler.generatedText,
          tjsText: entry.tjs?.generatedText ?? null,
        })),
      },
      prompts: comparedPrompts,
    };

    writeJson(path.join(outputDir, 'summary.json'), report);
    fs.writeFileSync(path.join(outputDir, 'summary.md'), buildMarkdownSummary(report), 'utf8');
    console.log(outputDir);
  } finally {
    setRuntimeConfig(originalRuntimeConfig);
  }
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
