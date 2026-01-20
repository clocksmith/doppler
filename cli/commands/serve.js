#!/usr/bin/env node


import http from 'http';
import { stat, readdir, rm, mkdir } from 'fs/promises';
import { createReadStream } from 'fs';
import path, { extname, resolve, join } from 'path';
import os from 'os';
import { exec } from 'child_process';
import { URL } from 'url';

import { loadConfig } from '../config/index.js';
import { createConverterConfig } from '../../src/config/index.js';
import { convertGGUF } from '../../src/converter/node-converter/converter.js';


function parseArgs(argv) {
  const opts = { config: null, help: false };
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      i += 1;
      continue;
    }
    if (arg === '--config' || arg === '-c') {
      opts.config = argv[i + 1] || null;
      i += 2;
      continue;
    }
    if (!arg.startsWith('-') && !opts.config) {
      opts.config = arg;
      i += 1;
      continue;
    }
    console.error(`Unknown argument: ${arg}`);
    opts.help = true;
    break;
  }
  return opts;
}


function printHelp() {
  console.log(`
DOPPLER Serve - Convert + Serve models for the DOPPLER provider

Usage:
  doppler --config <ref>

Config requirements:
  cli.command = "tool"
  cli.tool = "serve"
  tools.serve.input (string, required)
  tools.serve.port (number, optional; default 8765)
  tools.serve.output (string|null, optional; null uses temp dir)
  tools.serve.keep (boolean, optional; default false)
  tools.serve.open (boolean, optional; default true)
  tools.serve.dopplerUrl (string, optional; default http://localhost:5173)

Optional converter overrides:
  converter.quantization, converter.sharding, converter.weightLayout,
  converter.manifest, converter.output, converter.presets, converter.verbose

Examples:
  doppler --config ./tmp-serve-gguf.json
  doppler --config ./tmp-serve-rdrr.json
`);
}


function openBrowser(url) {
  const openCmd =
    process.platform === 'darwin'
      ? 'open'
      : process.platform === 'win32'
        ? 'start'
        : 'xdg-open';

  exec(`${openCmd} "${url}"`, (err) => {
    if (err) {
      console.log(`Could not open browser automatically.`);
      console.log(`Please open: ${url}`);
    }
  });
}


async function detectInputType(inputPath) {
  const stats = await stat(inputPath);
  if (stats.isDirectory()) {
    const files = await readdir(inputPath);
    if (files.some(f => f === 'manifest.json')) return 'rdrr';
    throw new Error(`Directory ${inputPath} does not look like an .rdrr pack (missing manifest.json)`);
  }
  if (extname(inputPath).toLowerCase() === '.gguf') return 'gguf';
  throw new Error(`Unsupported input: ${inputPath} (must be .gguf or .rdrr folder)`);
}


function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertStringOrNull(value, label) {
  if (value === null) return;
  assertString(value, label);
}

function assertBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be a boolean`);
  }
}

function assertNumber(value, label) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`${label} must be a number`);
  }
}

function resolveServeConfig(raw) {
  assertObject(raw, 'config');
  const tool = raw.tools?.serve ?? null;
  if (!tool) {
    throw new Error('tools.serve is required in config');
  }
  assertObject(tool, 'tools.serve');
  assertString(tool.input, 'tools.serve.input');

  if (tool.port !== undefined) assertNumber(tool.port, 'tools.serve.port');
  if (tool.output !== undefined) assertStringOrNull(tool.output, 'tools.serve.output');
  if (tool.keep !== undefined) assertBoolean(tool.keep, 'tools.serve.keep');
  if (tool.open !== undefined) assertBoolean(tool.open, 'tools.serve.open');
  if (tool.dopplerUrl !== undefined) assertString(tool.dopplerUrl, 'tools.serve.dopplerUrl');

  return {
    input: tool.input,
    port: tool.port ?? 8765,
    output: tool.output ?? null,
    keep: tool.keep ?? false,
    open: tool.open ?? true,
    dopplerUrl: tool.dopplerUrl ?? 'http://localhost:5173',
  };
}

function resolveConverterConfig(raw) {
  const converter = raw?.converter ?? null;
  if (!converter) {
    return {
      converterConfig: createConverterConfig(),
      verbose: false,
    };
  }
  assertObject(converter, 'converter');
  return {
    converterConfig: createConverterConfig({
      quantization: converter.quantization,
      sharding: converter.sharding,
      weightLayout: converter.weightLayout,
      manifest: converter.manifest,
      output: converter.output,
      presets: converter.presets,
    }),
    verbose: converter.verbose === true,
  };
}

async function runConvert(inputPath, outputDir, converterConfig, verbose) {
  await mkdir(outputDir, { recursive: true });
  console.log(`[serve-cli] Converting GGUF -> .rdrr at ${outputDir}`);
  await convertGGUF(inputPath, outputDir, { converterConfig, verbose });
  return outputDir;
}


async function validateRDRR(dir) {
  const manifestPath = join(dir, 'manifest.json');
  try {
    await stat(manifestPath);
  } catch {
    throw new Error(`manifest.json not found in ${dir}`);
  }
  const files = await readdir(dir);
  const hasShard = files.some(f => f.startsWith('shard_') && f.endsWith('.bin'));
  if (!hasShard) {
    throw new Error(`No shard_*.bin files found in ${dir}`);
  }
}


function contentTypeFor(ext) {
  switch (ext) {
    case '.json': return 'application/json';
    case '.bin': return 'application/octet-stream';
    default: return 'application/octet-stream';
  }
}


function startServer(serveDir, args) {
  const { port, open, dopplerUrl } = args;

  const server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
    res.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    try {
      const reqUrl = new URL(req.url || '/', `http://localhost:${port}`);
      let relativePath = decodeURIComponent(reqUrl.pathname);
      if (relativePath === '/') relativePath = '/';

      const safePath = path.normalize(relativePath).replace(/^(\.\.[/\\])+/, '');
      let target = join(serveDir, safePath);
      const rel = path.relative(serveDir, target);
      if (rel.startsWith('..')) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }

      let stats;
      try {
        stats = await stat(target);
      } catch {
        res.writeHead(404);
        res.end('Not found');
        return;
      }

      if (stats.isDirectory()) {
        target = join(target, 'manifest.json');
      }

      const ext = path.extname(target).toLowerCase();
      res.setHeader('Content-Type', contentTypeFor(ext));
      res.writeHead(200);
      const stream = createReadStream(target);
      stream.pipe(res);
      stream.on('error', (err) => {
        console.error('[serve-cli] Stream error:', err);
        res.destroy(err);
      });
      return; // Stream handles response, explicitly return void
    } catch (err) {
      console.error('[serve-cli] Request error:', err);
      if (!res.headersSent) {
        res.writeHead(500);
      }
      res.end('Internal server error');
      return;
    }
  });

  server.listen(port, () => {
    const modelUrl = `http://localhost:${port}`;
    console.log(`\n${'─'.repeat(50)}`);
    console.log(`Model ready at: ${modelUrl}`);
    console.log(`${'─'.repeat(50)}\n`);

    if (open) {
      const fullDopplerUrl = `${dopplerUrl}/?provider=doppler&modelUrl=${encodeURIComponent(modelUrl)}`;
      console.log(`Opening DOPPLER: ${fullDopplerUrl}\n`);
      openBrowser(fullDopplerUrl);
    } else {
      console.log('Paste this URL into the DOPPLER boot screen (Model URL field).\n');
    }

    console.log('Press Ctrl+C to stop the server.\n');
  });

  return server;
}


async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    process.exit(0);
  }
  if (!args.config) {
    console.error('Error: --config is required');
    process.exit(1);
  }

  let loaded;
  try {
    loaded = await loadConfig(args.config);
  } catch (err) {
    console.error(`Failed to load config "${args.config}": ${err.message}`);
    process.exit(1);
  }

  const raw = loaded.raw ?? {};
  const serveConfig = resolveServeConfig(raw);
  const converter = resolveConverterConfig(raw);

  const inputPath = resolve(serveConfig.input);
  const inputType = await detectInputType(inputPath);

  let serveDir = inputPath;

  let tempDir = null;

  if (inputType === 'gguf') {
    tempDir = serveConfig.output
      ? resolve(serveConfig.output)
      : path.join(os.tmpdir(), `doppler-rdrr-${Date.now()}`);
    serveDir = await runConvert(inputPath, tempDir, converter.converterConfig, converter.verbose);
  }

  await validateRDRR(serveDir);
  startServer(serveDir, serveConfig);

  const cleanup = async () => {
    if (!serveConfig.keep && tempDir) {
      try {
        await rm(tempDir, { recursive: true, force: true });
        console.log(`[serve-cli] Removed temp directory ${tempDir}`);
      } catch (err) {
        const error =  (err);
        console.warn('[serve-cli] Failed to remove temp directory:', error.message);
      }
    }
    process.exit(0);
  };

  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);
}

main().catch((err) => {
  console.error('[serve-cli] Error:', err.message);
  process.exit(1);
});

export { parseArgs, detectInputType, validateRDRR, startServer };
