#!/usr/bin/env node

import http from 'node:http';
import { realpathSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { DOPPLER_VERSION, doppler } from '../index.js';
import { getCachedModel } from '../client/doppler-api.js';
import { listQuickstartModels, resolveQuickstartModel } from '../client/doppler-registry.js';
import {
  bindLogicalRuntimeModel,
  buildServeFailureReceipt,
  buildServeReceipt,
  generateCompletionId,
} from './serve/receipt.js';
import {
  DEFAULT_PREFIX_CACHE_POLICY,
  createServePrefixCache,
  deriveServePrefixKey,
} from './serve/prefix-cache.js';

export { buildServeFailureReceipt, buildServeReceipt } from './serve/receipt.js';
export {
  DEFAULT_PREFIX_CACHE_POLICY,
  createServePrefixCache,
  deriveServePrefixKey,
} from './serve/prefix-cache.js';

const DEFAULT_PORT = 8080;
const DEFAULT_HOST = '127.0.0.1';

export function isLoopbackHost(host) {
  if (typeof host !== 'string') return false;
  const normalized = host.trim().toLowerCase();
  return normalized === '127.0.0.1'
    || normalized === 'localhost'
    || normalized === '::1'
    || normalized === '[::1]';
}

export class ServeRequestError extends Error {
  constructor(message, statusCode = 400, type = 'invalid_request_error') {
    super(message);
    this.name = 'ServeRequestError';
    this.statusCode = statusCode;
    this.type = type;
  }
}

function usage() {
  return [
    'Usage:',
    '  doppler-serve [--model <id>] [--port <n>] [--host <addr>] [--prefix-cache | --no-prefix-cache] [--prefix-cache-bytes <n>]',
    '',
    'Options:',
    '  --model <id>              Pre-load a model at startup (optional, lazy-loads on request otherwise)',
    '  --model-url <url>         Explicit local or remote RDRR artifact URL for --model',
    '  --port <n>                Port to listen on (default: 8080)',
    '  --host <addr>             Host to bind to (default: 127.0.0.1)',
    '  --prefix-cache            Explicitly enable multi-turn KV prefix reuse (default: on for loopback hosts only)',
    '  --prefix-cache-bytes <n>  Total GPU/host byte budget for multi-turn KV prefix reuse (default: 268435456)',
    '  --no-prefix-cache         Disable automatic multi-turn KV prefix reuse',
    '  --help                    Show this help',
    '',
    'Endpoints:',
    '  POST /v1/chat/completions   OpenAI-compatible chat completions',
    '  GET  /v1/models             List available models',
    '  GET  /health                Readiness, retained prefix tokens, and reasons checkpoints were dropped',
    '',
    'Examples:',
    '  node src/cli/doppler-serve.js --model qwen3-0.8b',
    '  node src/cli/doppler-serve.js --model gemma4-e2b --port 3000',
    '',
    'Then use with any OpenAI-compatible client:',
    '  curl http://localhost:8080/v1/chat/completions \\',
    '    -H "Content-Type: application/json" \\',
    '    -d \'{"model":"qwen3-0.8b","messages":[{"role":"user","content":"Hello"}]}\'',
  ].join('\n');
}

export function parseServeArgs(argv) {
  let explicitPrefixCache = null;
  const flags = {
    port: DEFAULT_PORT,
    host: DEFAULT_HOST,
    model: null,
    modelUrl: null,
    prefixCache: true,
    prefixCacheBytes: DEFAULT_PREFIX_CACHE_POLICY.maxBytes,
    help: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (token === '--prefix-cache') {
      explicitPrefixCache = true;
      continue;
    }
    if (token === '--no-prefix-cache') {
      explicitPrefixCache = false;
      continue;
    }
    if (!token.startsWith('--')) {
      throw new Error(`Unexpected positional argument: ${token}`);
    }
    if (
      token !== '--port'
      && token !== '--host'
      && token !== '--model'
      && token !== '--model-url'
      && token !== '--prefix-cache-bytes'
    ) {
      throw new Error(`Unknown flag ${token}.`);
    }
    const nextValue = argv[i + 1];
    if (nextValue === undefined || nextValue.startsWith('--')) {
      throw new Error(`Missing value for ${token}.`);
    }
    if (token === '--port') {
      const parsed = Number(nextValue);
      if (!Number.isFinite(parsed) || parsed < 0 || parsed > 65535 || parsed !== Math.floor(parsed)) {
        throw new Error('--port must be a valid port number (0-65535).');
      }
      flags.port = parsed;
      i += 1;
      continue;
    }
    if (token === '--host') {
      flags.host = nextValue.trim();
      i += 1;
      continue;
    }
    if (token === '--model') {
      flags.model = nextValue.trim();
      i += 1;
      continue;
    }
    if (token === '--model-url') {
      flags.modelUrl = normalizeModelUrl(nextValue.trim());
      i += 1;
      continue;
    }
    if (token === '--prefix-cache-bytes') {
      const parsed = Number(nextValue);
      if (!Number.isSafeInteger(parsed) || parsed <= 0) {
        throw new Error('--prefix-cache-bytes must be a positive integer byte budget.');
      }
      flags.prefixCacheBytes = parsed;
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag ${token}.`);
  }
  if (flags.modelUrl && !flags.model) {
    throw new Error('--model-url requires --model so the served model identity remains explicit.');
  }
  // Without an authenticated principal, only enable the prefix cache by default on loopback binds.
  flags.prefixCache = explicitPrefixCache === null ? isLoopbackHost(flags.host) : explicitPrefixCache;
  return flags;
}

function normalizeModelUrl(value) {
  if (!value) {
    throw new Error('--model-url must be non-empty.');
  }
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value)) {
    return value;
  }
  return pathToFileURL(path.resolve(value)).href;
}

function jsonError(res, statusCode, message, type, extra = null) {
  const payload = {
    error: {
      message,
      type: type ?? 'invalid_request_error',
      param: null,
      code: null,
    },
  };
  if (extra && typeof extra === 'object' && !Array.isArray(extra)) {
    Object.assign(payload, extra);
  }
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(body);
}

function jsonResponse(res, statusCode, data) {
  const body = JSON.stringify(data);
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(body);
}

function requestError(message, statusCode = 400, type = 'invalid_request_error') {
  return new ServeRequestError(message, statusCode, type);
}

async function readRequestBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  if (!raw || raw.trim().length === 0) {
    throw requestError('Request body is empty.');
  }
  try {
    return JSON.parse(raw);
  } catch {
    throw requestError('Request body is not valid JSON.');
  }
}

function validateMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw requestError('"messages" must be a non-empty array.');
  }
  for (let i = 0; i < messages.length; i += 1) {
    const msg = messages[i];
    if (!msg || typeof msg !== 'object') {
      throw requestError(`messages[${i}] must be an object.`);
    }
    if (typeof msg.role !== 'string' || msg.role.trim().length === 0) {
      throw requestError(`messages[${i}].role must be a non-empty string.`);
    }
    if (typeof msg.content !== 'string') {
      throw requestError(`messages[${i}].content must be a string.`);
    }
  }
  return messages.map((msg) => ({ role: msg.role.trim(), content: msg.content }));
}

function extractGenerationOptions(body) {
  const options = {};
  if (body.max_tokens != null) {
    const n = Number(body.max_tokens);
    if (!Number.isFinite(n) || n < 1 || Math.floor(n) !== n) {
      throw requestError('"max_tokens" must be a positive integer.');
    }
    options.maxTokens = n;
  }
  if (body.temperature != null) {
    const t = Number(body.temperature);
    if (!Number.isFinite(t) || t < 0) {
      throw requestError('"temperature" must be a non-negative number.');
    }
    options.temperature = t;
  }
  if (body.top_p != null) {
    const p = Number(body.top_p);
    if (!Number.isFinite(p) || p <= 0 || p > 1) {
      throw requestError('"top_p" must be a number between 0 (exclusive) and 1 (inclusive).');
    }
    options.topP = p;
  }
  if (body.top_k != null) {
    const k = Number(body.top_k);
    if (!Number.isFinite(k) || k < 1 || Math.floor(k) !== k) {
      throw requestError('"top_k" must be a positive integer.');
    }
    options.topK = k;
  }
  return options;
}

function resolveIncludeReceipt(body) {
  const includeReceipt = body.include_receipt;
  const dopplerReceipt = body.doppler_receipt;
  if (includeReceipt == null && dopplerReceipt == null) {
    return false;
  }
  if (includeReceipt != null && typeof includeReceipt !== 'boolean') {
    throw requestError('"include_receipt" must be a boolean when provided.');
  }
  if (dopplerReceipt != null && typeof dopplerReceipt !== 'boolean') {
    throw requestError('"doppler_receipt" must be a boolean when provided.');
  }
  if (includeReceipt != null && dopplerReceipt != null && includeReceipt !== dopplerReceipt) {
    throw requestError('"include_receipt" and "doppler_receipt" must agree when both are provided.');
  }
  return includeReceipt ?? dopplerReceipt;
}

async function resolveTextModelEntry(modelId, resolveModel) {
  let entry;
  try {
    entry = await resolveModel(modelId);
  } catch {
    throw requestError(`Unknown model "${modelId}". Use GET /v1/models for supported chat models.`);
  }
  if (!entry.modes.includes('text')) {
    throw requestError(`Model "${modelId}" is not text-generative. Use GET /v1/models for supported chat models.`);
  }
  return entry;
}

function resolveServeDependencies(dependencies = {}) {
  const dopplerClient = dependencies.dopplerClient ?? doppler;
  const prefixCache = dependencies.prefixCache !== undefined
    ? dependencies.prefixCache
    : createServePrefixCache({
      policy: dependencies.prefixCachePolicy ?? false,
    });
  return {
    dopplerClient,
    getCachedModel: dependencies.getCachedModel ?? (dopplerClient === doppler ? getCachedModel : null),
    prefixCache,
    listModels: dependencies.listModels ?? listQuickstartModels,
    resolveModel: dependencies.resolveModel ?? resolveQuickstartModel,
    resolveRuntimeModel: typeof dependencies.resolveRuntimeModel === 'function'
      ? dependencies.resolveRuntimeModel
      : (registryEntry) => registryEntry.modelId,
  };
}

async function acquireServePrefixLease(dependencies, registryEntry, runtimeModel, messages) {
  const { prefixCache } = dependencies;
  if (!prefixCache || !prefixCache.policy?.enabled || typeof dependencies.getCachedModel !== 'function') {
    return { modelHandle: null, lease: null };
  }
  const modelHandle = await dependencies.getCachedModel(runtimeModel);
  if (
    modelHandle?.supportsGenerationContext === true
    && typeof modelHandle.createGenerationContext === 'function'
  ) {
    const key = deriveServePrefixKey(registryEntry.modelId, messages);
    const lease = prefixCache.acquire(key, (options) => modelHandle.createGenerationContext(options),
      modelHandle, modelHandle.generationContextMinimumBytes ?? 0);
    return { modelHandle, lease };
  }
  return { modelHandle, lease: null };
}

async function handleChatCompletions(req, res, dependencies) {
  const body = await readRequestBody(req);
  if (typeof body.model !== 'string' || body.model.trim().length === 0) {
    return jsonError(res, 400, '"model" is required and must be a non-empty string.');
  }
  const modelId = body.model.trim();
  const messages = validateMessages(body.messages);
  const generationOptions = extractGenerationOptions(body);
  const includeReceipt = resolveIncludeReceipt(body);
  const stream = body.stream === true;
  if (includeReceipt && stream) {
    throw requestError('"include_receipt" is not supported with streaming responses.');
  }
  const registryEntry = await resolveTextModelEntry(modelId, dependencies.resolveModel);
  const completionId = generateCompletionId();
  const created = Math.floor(Date.now() / 1000);

  if (stream) {
    return handleStreamingCompletion(
      res,
      modelId,
      registryEntry,
      messages,
      generationOptions,
      completionId,
      created,
      dependencies
    );
  }
  return handleNonStreamingCompletion(
    res,
    modelId,
    registryEntry,
    messages,
    generationOptions,
    completionId,
    created,
    includeReceipt,
    dependencies
  );
}

async function handleNonStreamingCompletion(
  res,
  requestedModel,
  registryEntry,
  messages,
  generationOptions,
  completionId,
  created,
  includeReceipt,
  dependencies
) {
  let result;
  const runtimeModel = bindLogicalRuntimeModel(
    dependencies.resolveRuntimeModel(registryEntry, requestedModel),
    requestedModel
  );
  let lease = null;
  let completed = false;
  try {
    const acquired = await acquireServePrefixLease(dependencies, registryEntry, runtimeModel, messages);
    lease = acquired.lease;
    if (acquired.modelHandle && typeof acquired.modelHandle.chatText === 'function') {
      const options = lease
        ? { ...generationOptions, generationContext: lease.context }
        : { ...generationOptions };
      result = await acquired.modelHandle.chatText(messages, options);
    } else {
      result = await dependencies.dopplerClient.chatText(messages, { model: runtimeModel, ...generationOptions });
    }
    completed = true;
  } catch (error) {
    if (includeReceipt) {
      return jsonError(
        res,
        500,
        error?.message || String(error),
        'server_error',
        {
          doppler_receipt: buildServeFailureReceipt({
            requestedModel,
            registryEntry,
            messages,
            generationOptions,
            error,
            runtimeModel,
          }),
        }
      );
    }
    throw error;
  } finally {
    lease?.release({ discard: !completed });
  }
  const cachedTokens = result.evidence?.stats?.reusedPrefixTokens;
  const usage = {
    prompt_tokens: result.usage.promptTokens,
    completion_tokens: result.usage.completionTokens,
    total_tokens: result.usage.totalTokens,
  };
  if (Number.isSafeInteger(cachedTokens) && cachedTokens > 0) {
    usage.prompt_tokens_details = { cached_tokens: cachedTokens };
  }
  const body = {
    id: completionId,
    object: 'chat.completion',
    created,
    model: registryEntry.modelId,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: result.content,
        },
        finish_reason: 'stop',
      },
    ],
    usage,
  };
  if (includeReceipt) {
    body.doppler_receipt = buildServeReceipt({
      requestedModel,
      registryEntry,
      messages,
      generationOptions,
      outputContent: result.content,
      usage: result.usage,
      runtimeModel,
      generationEvidence: result.evidence ?? null,
    });
  }
  jsonResponse(res, 200, body);
}

async function handleStreamingCompletion(
  res,
  requestedModel,
  registryEntry,
  messages,
  generationOptions,
  completionId,
  created,
  dependencies
) {
  const runtimeModel = bindLogicalRuntimeModel(
    dependencies.resolveRuntimeModel(registryEntry, requestedModel),
    requestedModel
  );
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
  });

  function sendChunk(data) {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  }

  sendChunk({
    id: completionId,
    object: 'chat.completion.chunk',
    created,
    model: registryEntry.modelId,
    choices: [
      {
        index: 0,
        delta: { role: 'assistant' },
        finish_reason: null,
      },
    ],
  });

  let lease = null;
  let completed = false;
  try {
    const acquired = await acquireServePrefixLease(dependencies, registryEntry, runtimeModel, messages);
    lease = acquired.lease;
    const stream = acquired.modelHandle && typeof acquired.modelHandle.chat === 'function'
      ? acquired.modelHandle.chat(
        messages,
        lease ? { ...generationOptions, generationContext: lease.context } : { ...generationOptions }
      )
      : dependencies.dopplerClient.chat(messages, { model: runtimeModel, ...generationOptions });
    for await (const token of stream) {
      if (res.destroyed) {
        break;
      }
      sendChunk({
        id: completionId,
        object: 'chat.completion.chunk',
        created,
        model: registryEntry.modelId,
        choices: [
          {
            index: 0,
            delta: { content: token },
            finish_reason: null,
          },
        ],
      });
    }
    if (!res.destroyed) {
      completed = true;
    }
  } finally {
    lease?.release({ discard: !completed });
  }

  sendChunk({
    id: completionId,
    object: 'chat.completion.chunk',
    created,
    model: registryEntry.modelId,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: 'stop',
      },
    ],
  });
  res.write('data: [DONE]\n\n');
  res.end();
}

async function handleListModels(res, dependencies) {
  const models = await dependencies.listModels();
  const textModels = models.filter((entry) => entry.modes.includes('text'));
  jsonResponse(res, 200, {
    object: 'list',
    data: textModels.map((entry) => ({
      id: entry.modelId,
      object: 'model',
      created: 0,
      owned_by: 'doppler',
      doppler: {
        sourceCheckpointId: entry.sourceCheckpointId,
        weightCapsuleId: entry.weightCapsuleId,
        manifestVariantId: entry.manifestVariantId,
        artifactCompleteness: entry.artifactCompleteness,
        runtimePromotionState: entry.runtimePromotionState,
        weightsRefAllowed: entry.weightsRefAllowed,
        modes: entry.modes,
      },
    })),
  });
}

function handleHealth(res, dependencies) {
  const payload = {
    status: 'ok',
    version: DOPPLER_VERSION,
  };
  if (dependencies?.prefixCache && typeof dependencies.prefixCache.getStats === 'function') {
    payload.prefix_cache = dependencies.prefixCache.getStats();
  }
  jsonResponse(res, 200, payload);
}

function handleCors(req, res) {
  res.writeHead(204, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
  });
  res.end();
}

export function createServeHandler(dependencies = {}) {
  const resolvedDependencies = resolveServeDependencies(dependencies);
  return async (req, res) => {
    if (req.method === 'OPTIONS') {
      return handleCors(req, res);
    }

    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    const pathname = url.pathname.replace(/\/+$/, '') || '/';

    try {
      if (pathname === '/v1/chat/completions' && req.method === 'POST') {
        return await handleChatCompletions(req, res, resolvedDependencies);
      }
      if (pathname === '/v1/models' && req.method === 'GET') {
        return await handleListModels(res, resolvedDependencies);
      }
      if ((pathname === '/health' || pathname === '/') && req.method === 'GET') {
        return handleHealth(res, resolvedDependencies);
      }
      jsonError(res, 404, `Unknown endpoint: ${req.method} ${pathname}`, 'not_found');
    } catch (error) {
      const message = error?.message || String(error);
      console.error(`[doppler-serve] ${req.method} ${pathname}: ${message}`);
      if (!res.headersSent) {
        if (error instanceof ServeRequestError) {
          jsonError(res, error.statusCode, message, error.type);
        } else {
          jsonError(res, 500, message, 'server_error');
        }
      }
    }
  };
}

function createRuntimeModelResolver(localModelSourceByModelId = new Map()) {
  return (registryEntry) => localModelSourceByModelId.get(registryEntry.modelId) ?? registryEntry.modelId;
}

async function resolveLocalModelSourceMap(settings) {
  const localModelSourceByModelId = new Map();
  if (!settings.modelUrl) {
    return localModelSourceByModelId;
  }
  const registryEntry = await resolveQuickstartModel(settings.model);
  localModelSourceByModelId.set(registryEntry.modelId, {
    ...(registryEntry.hostedSource ? { registryId: registryEntry.modelId } : {}),
    url: settings.modelUrl,
  });
  return localModelSourceByModelId;
}

async function startServer(settings) {
  const localModelSourceByModelId = await resolveLocalModelSourceMap(settings);
  const resolveRuntimeModel = createRuntimeModelResolver(localModelSourceByModelId);
  if (settings.model) {
    console.error(`[doppler-serve] pre-loading model: ${settings.model}`);
    const registryEntry = await resolveQuickstartModel(settings.model);
    await getCachedModel(bindLogicalRuntimeModel(resolveRuntimeModel(registryEntry), settings.model));
    console.error(`[doppler-serve] model ready: ${settings.model}`);
  }

  const prefixCache = createServePrefixCache({
    policy: settings.prefixCache
      ? { enabled: true, maxBytes: settings.prefixCacheBytes }
      : { enabled: false },
  });
  const handler = createServeHandler({ resolveRuntimeModel, prefixCache });
  const server = http.createServer(handler);
  server.on('close', () => prefixCache.close());

  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(settings.port, settings.host, () => {
      const addr = server.address();
      console.error(`[doppler-serve] listening on http://${addr.address}:${addr.port}`);
      console.error(`[doppler-serve] OpenAI-compatible: POST http://${addr.address}:${addr.port}/v1/chat/completions`);
      resolve(server);
    });
  });
}

export async function main(argv = process.argv.slice(2)) {
  const settings = parseServeArgs(argv);
  if (settings.help) {
    console.log(usage());
    return;
  }
  await startServer(settings);
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  if (!entryPath) {
    return false;
  }
  // npm starts bins through a symbolic link, so compare real paths.
  return realpathSync(path.resolve(fileURLToPath(metaUrl))) === realpathSync(path.resolve(entryPath));
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[doppler-serve] ${error?.message || String(error)}`);
    process.exit(1);
  });
}
