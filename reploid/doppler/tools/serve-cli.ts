#!/usr/bin/env node
/**
 * Serve CLI - Convert + Serve models for DOPPLER
 */

import http, { IncomingMessage, ServerResponse } from 'http';
import { stat, readdir, rm, mkdir } from 'fs/promises';
import { createReadStream } from 'fs';
import path, { extname, resolve, join, dirname } from 'path';
import os from 'os';
import { spawn, exec } from 'child_process';
import { fileURLToPath } from 'url';
import { URL } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export type InputType = 'gguf' | 'rdrr';

export interface ServeOptions {
  input: string | null;
  port: number;
  output: string | null;
  keep: boolean;
  open: boolean;
  dopplerUrl: string;
  help: boolean;
}

function parseArgs(argv: string[]): ServeOptions {
  const args: ServeOptions = {
    input: null,
    port: 8765,
    output: null,
    keep: false,
    open: true,
    dopplerUrl: 'http://localhost:5173',
    help: false,
  };

  const tokens = [...argv];
  while (tokens.length) {
    const arg = tokens.shift()!;
    if (arg === '--help' || arg === '-h') args.help = true;
    else if (arg === '--port' || arg === '-p') args.port = parseInt(tokens.shift() || '8765', 10);
    else if (arg === '--output' || arg === '-o') args.output = tokens.shift() || null;
    else if (arg === '--keep') args.keep = true;
    else if (arg === '--no-open') args.open = false;
    else if (arg === '--doppler-url') args.dopplerUrl = tokens.shift() || args.dopplerUrl;
    else if (!args.input) args.input = arg;
  }
  return args;
}

function printHelp(): void {
  console.log(`
DOPPLER Serve - Convert + Serve models for the DOPPLER provider

Usage:
  node serve-cli.js <input> [options]

Input:
  - GGUF file (e.g., model.gguf)
  - .rdrr folder containing manifest.json and shard_*.bin

Options:
  --port, -p <num>   Port to serve on (default: 8765)
  --output, -o <dir> Output directory for converted .rdrr (default: temp)
  --keep             Keep converted .rdrr directory after exit
  --no-open          Don't auto-open browser
  --doppler-url      Base URL for DOPPLER (default: http://localhost:5173)
  --help             Show this help

Examples:
  node serve-cli.js model.gguf
  node serve-cli.js ./model-rdrr-folder --port 9000
  node serve-cli.js model.gguf --no-open
`);
}

function openBrowser(url: string): void {
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

async function detectInputType(inputPath: string): Promise<InputType> {
  const stats = await stat(inputPath);
  if (stats.isDirectory()) {
    const files = await readdir(inputPath);
    if (files.some(f => f === 'manifest.json')) return 'rdrr';
    throw new Error(`Directory ${inputPath} does not look like an .rdrr pack (missing manifest.json)`);
  }
  if (extname(inputPath).toLowerCase() === '.gguf') return 'gguf';
  throw new Error(`Unsupported input: ${inputPath} (must be .gguf or .rdrr folder)`);
}

async function runConvert(inputPath: string, outputDir: string): Promise<string> {
  await mkdir(outputDir, { recursive: true });
  console.log(`[serve-cli] Converting GGUF -> .rdrr at ${outputDir}`);

  // Detect if running from TypeScript source or compiled JavaScript
  const isTypeScript = __filename.endsWith('.ts');
  const cliBasename = isTypeScript ? 'convert-cli.ts' : 'convert-cli.js';
  const cliPath = resolve(__dirname, `./${cliBasename}`);

  // When running TypeScript directly, use tsx or ts-node loader
  const spawnArgs = isTypeScript
    ? ['--import', 'tsx', cliPath, inputPath, outputDir]
    : [cliPath, inputPath, outputDir];

  await new Promise<void>((resolvePromise, rejectPromise) => {
    const proc = spawn(process.execPath, spawnArgs, {
      stdio: 'inherit',
    });
    proc.on('exit', (code) => {
      if (code === 0) resolvePromise();
      else rejectPromise(new Error(`convert-cli exited with code ${code}`));
    });
  });
  return outputDir;
}

async function validateRDRR(dir: string): Promise<void> {
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

function contentTypeFor(ext: string): string {
  switch (ext) {
    case '.json': return 'application/json';
    case '.bin': return 'application/octet-stream';
    default: return 'application/octet-stream';
  }
}

function startServer(serveDir: string, args: ServeOptions): http.Server {
  const { port, open, dopplerUrl } = args;

  const server = http.createServer(async (req: IncomingMessage, res: ServerResponse) => {
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

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.input) {
    printHelp();
    process.exit(args.help ? 0 : 1);
  }

  const inputPath = resolve(args.input);
  const inputType = await detectInputType(inputPath);

  let serveDir = inputPath;
  let tempDir: string | null = null;

  if (inputType === 'gguf') {
    tempDir = args.output ? resolve(args.output) : path.join(os.tmpdir(), `doppler-rdrr-${Date.now()}`);
    serveDir = await runConvert(inputPath, tempDir);
  }

  await validateRDRR(serveDir);
  startServer(serveDir, args);

  const cleanup = async (): Promise<void> => {
    if (!args.keep && tempDir) {
      try {
        await rm(tempDir, { recursive: true, force: true });
        console.log(`[serve-cli] Removed temp directory ${tempDir}`);
      } catch (err) {
        const error = err as Error;
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
