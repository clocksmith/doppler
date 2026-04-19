#!/usr/bin/env node

import http from 'node:http';
import crypto from 'node:crypto';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { DOPPLER_VERSION, doppler } from '../index.js';
import { listQuickstartModels } from '../client/doppler-registry.js';

const DEFAULT_PORT = 8080;
const DEFAULT_HOST = '127.0.0.1';

function usage() {
  return [
    'Usage:',
    '  doppler-serve [--model <id>] [--port <n>] [--host <addr>]',
    '',
    'Options:',
    '  --model <id>     Pre-load a model at startup (optional, lazy-loads on request otherwise)',
    '  --port <n>       Port to listen on (default: 8080)',
    '  --host <addr>    Host to bind to (default: 127.0.0.1)',
    '  --help           Show this help',
    '',
    'Endpoints:',
    '  POST /v1/chat/completions   OpenAI-compatible chat completions',
    '  GET  /v1/models             List available models',
    '  GET  /health                Health check',
    '',
    'Examples:',
    '  node src/cli/doppler-serve.js --model gemma3-270m',
    '  node src/cli/doppler-serve.js --model qwen3-0.8b --port 3000',
    '',
    'Then use with any OpenAI-compatible client:',
    '  curl http://localhost:8080/v1/chat/completions \\',
    '    -H "Content-Type: application/json" \\',
    '    -d \'{"model":"gemma3-270m","messages":[{"role":"user","content":"Hello"}]}\'',
  ].join('\n');
}

export function parseServeArgs(argv) {
  const flags = {
    port: DEFAULT_PORT,
    host: DEFAULT_HOST,
    model: null,
    help: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (!token.startsWith('--')) {
      throw new Error(`Unexpected positional argument: ${token}`);
    }
    if (token !== '--port' && token !== '--host' && token !== '--model') {
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
    throw new Error(`Unknown flag ${token}.`);
  }
  return flags;
}

function generateCompletionId() {
  return `chatcmpl-${crypto.randomBytes(12).toString('base64url')}`;
}

function jsonError(res, statusCode, message, type) {
  const body = JSON.stringify({
    error: {
      message,
      type: type ?? 'invalid_request_error',
      param: null,
      code: null,
    },
  });
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

async function readRequestBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  if (!raw || raw.trim().length === 0) {
    throw new Error('Request body is empty.');
  }
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error('Request body is not valid JSON.');
  }
}

function validateMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error('"messages" must be a non-empty array.');
  }
  for (let i = 0; i < messages.length; i += 1) {
    const msg = messages[i];
    if (!msg || typeof msg !== 'object') {
      throw new Error(`messages[${i}] must be an object.`);
    }
    if (typeof msg.role !== 'string' || msg.role.trim().length === 0) {
      throw new Error(`messages[${i}].role must be a non-empty string.`);
    }
    if (typeof msg.content !== 'string') {
      throw new Error(`messages[${i}].content must be a string.`);
    }
  }
  return messages.map((msg) => ({ role: msg.role.trim(), content: msg.content }));
}

function extractGenerationOptions(body) {
  const options = {};
  if (body.max_tokens != null) {
    const n = Number(body.max_tokens);
    if (!Number.isFinite(n) || n < 1) {
      throw new Error('"max_tokens" must be a positive integer.');
    }
    options.maxTokens = Math.floor(n);
  }
  if (body.temperature != null) {
    const t = Number(body.temperature);
    if (!Number.isFinite(t) || t < 0) {
      throw new Error('"temperature" must be a non-negative number.');
    }
    options.temperature = t;
  }
  if (body.top_p != null) {
    const p = Number(body.top_p);
    if (!Number.isFinite(p) || p <= 0 || p > 1) {
      throw new Error('"top_p" must be a number between 0 (exclusive) and 1 (inclusive).');
    }
    options.topP = p;
  }
  if (body.top_k != null) {
    const k = Number(body.top_k);
    if (!Number.isFinite(k) || k < 1) {
      throw new Error('"top_k" must be a positive integer.');
    }
    options.topK = Math.floor(k);
  }
  return options;
}

async function handleChatCompletions(req, res) {
  const body = await readRequestBody(req);
  if (typeof body.model !== 'string' || body.model.trim().length === 0) {
    return jsonError(res, 400, '"model" is required and must be a non-empty string.');
  }
  const modelId = body.model.trim();
  const messages = validateMessages(body.messages);
  const generationOptions = extractGenerationOptions(body);
  const stream = body.stream === true;
  const completionId = generateCompletionId();
  const created = Math.floor(Date.now() / 1000);

  if (stream) {
    return handleStreamingCompletion(res, modelId, messages, generationOptions, completionId, created);
  }
  return handleNonStreamingCompletion(res, modelId, messages, generationOptions, completionId, created);
}

async function handleNonStreamingCompletion(res, modelId, messages, generationOptions, completionId, created) {
  const result = await doppler.chatText(messages, { model: modelId, ...generationOptions });
  jsonResponse(res, 200, {
    id: completionId,
    object: 'chat.completion',
    created,
    model: modelId,
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
    usage: {
      prompt_tokens: result.usage.promptTokens,
      completion_tokens: result.usage.completionTokens,
      total_tokens: result.usage.totalTokens,
    },
  });
}

async function handleStreamingCompletion(res, modelId, messages, generationOptions, completionId, created) {
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
    model: modelId,
    choices: [
      {
        index: 0,
        delta: { role: 'assistant' },
        finish_reason: null,
      },
    ],
  });

  const stream = doppler.chat(messages, { model: modelId, ...generationOptions });
  for await (const token of stream) {
    if (res.destroyed) {
      break;
    }
    sendChunk({
      id: completionId,
      object: 'chat.completion.chunk',
      created,
      model: modelId,
      choices: [
        {
          index: 0,
          delta: { content: token },
          finish_reason: null,
        },
      ],
    });
  }

  sendChunk({
    id: completionId,
    object: 'chat.completion.chunk',
    created,
    model: modelId,
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

async function handleListModels(res) {
  const models = await listQuickstartModels();
  const textModels = models.filter((entry) => entry.modes.includes('text'));
  jsonResponse(res, 200, {
    object: 'list',
    data: textModels.map((entry) => ({
      id: entry.modelId,
      object: 'model',
      created: 0,
      owned_by: 'doppler',
    })),
  });
}

function handleHealth(res) {
  jsonResponse(res, 200, {
    status: 'ok',
    version: DOPPLER_VERSION,
  });
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

export function createServeHandler() {
  return async (req, res) => {
    if (req.method === 'OPTIONS') {
      return handleCors(req, res);
    }

    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    const pathname = url.pathname.replace(/\/+$/, '') || '/';

    try {
      if (pathname === '/v1/chat/completions' && req.method === 'POST') {
        return await handleChatCompletions(req, res);
      }
      if (pathname === '/v1/models' && req.method === 'GET') {
        return await handleListModels(res);
      }
      if ((pathname === '/health' || pathname === '/') && req.method === 'GET') {
        return handleHealth(res);
      }
      jsonError(res, 404, `Unknown endpoint: ${req.method} ${pathname}`, 'not_found');
    } catch (error) {
      const message = error?.message || String(error);
      console.error(`[doppler-serve] ${req.method} ${pathname}: ${message}`);
      if (!res.headersSent) {
        jsonError(res, 500, message, 'server_error');
      }
    }
  };
}

async function startServer(settings) {
  if (settings.model) {
    console.error(`[doppler-serve] pre-loading model: ${settings.model}`);
    await doppler.load(settings.model);
    console.error(`[doppler-serve] model ready: ${settings.model}`);
  }

  const handler = createServeHandler();
  const server = http.createServer(handler);

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
  return path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[doppler-serve] ${error?.message || String(error)}`);
    process.exit(1);
  });
}
