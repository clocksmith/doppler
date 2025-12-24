#!/usr/bin/env node
/**
 * DOPPLER Development Server - Serves the app UI and model files.
 * Standalone server for the DOPPLER WebGPU inference engine.
 *
 * Repository: https://github.com/clocksmith/doppler
 */

import http, { IncomingMessage, ServerResponse } from 'http';
import { readFile, stat, readdir } from 'fs/promises';
import { createReadStream, Stats } from 'fs';
import { extname, join, dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MIME_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.wgsl': 'text/plain; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
};

export interface ServerOptions {
  port: number;
  open: boolean;
  help: boolean;
}

export interface ModelInfo {
  path: string;
  name: string;
  architecture?: string | null;
  quantization?: string | null;
  size?: string | null;
  downloadSize?: number;
  vocabSize?: number | null;
  numLayers?: number | null;
}

function parseArgs(argv: string[]): ServerOptions {
  const args: ServerOptions = {
    port: 8080,
    open: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      args.help = true;
    } else if (arg === '--port' || arg === '-p') {
      args.port = parseInt(argv[++i], 10);
    } else if (arg === '--open' || arg === '-o') {
      args.open = true;
    }
  }

  return args;
}

function printHelp(): void {
  console.log(`
DOPPLER Development Server

Serves the DOPPLER WebGPU inference engine demo UI and model files.

Repository: https://github.com/clocksmith/doppler

Usage:
  npm start [options]
  npx tsx serve.ts [options]

Options:
  --port, -p <num>     Port to serve on (default: 8080)
  --open, -o           Open browser automatically
  --help, -h           Show this help

URLs:
  http://localhost:<port>/           DOPPLER app
  http://localhost:<port>/models/    Model files

Models are served from:
  ./models/<model-name>/
`);
}

function openBrowser(url: string): void {
  const platform = process.platform;
  let cmd: string;

  if (platform === 'darwin') {
    cmd = `open "${url}"`;
  } else if (platform === 'win32') {
    cmd = `start "" "${url}"`;
  } else {
    cmd = `xdg-open "${url}"`;
  }

  exec(cmd, (err) => {
    if (err) {
      console.log(`Could not open browser automatically. Visit: ${url}`);
    }
  });
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printHelp();
    process.exit(0);
  }

  const { port } = args;
  // Serve from doppler directory (standalone mode)
  const dopplerDir = __dirname;
  const rootDir = dopplerDir;

  function serveFile(filePath: string, stats: Stats, req: IncomingMessage, res: ServerResponse): void {
    const ext = extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    const rangeHeader = req.headers.range;
    if (rangeHeader) {
      const match = rangeHeader.match(/bytes=(\d*)-(\d*)/);
      if (match) {
        const parts = rangeHeader.replace(/bytes=/, '').split('-');
        const start = parseInt(parts[0], 10) || 0;
        const end = parts[1] ? parseInt(parts[1], 10) : stats.size - 1;
        const chunkSize = end - start + 1;

        res.writeHead(206, {
          'Content-Range': `bytes ${start}-${end}/${stats.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': chunkSize,
          'Content-Type': contentType,
        });

        const stream = createReadStream(filePath, { start, end });
        stream.pipe(res);
        return;
      }
    }

    const headers: Record<string, string | number> = {
      'Content-Type': contentType,
      'Content-Length': stats.size,
      'Accept-Ranges': 'bytes',
    };

    if (ext === '.js' || ext === '.mjs') {
      headers['Cache-Control'] = 'no-store, no-cache, must-revalidate';
      headers['Pragma'] = 'no-cache';
      headers['Expires'] = '0';
    }

    res.writeHead(200, headers);

    if (req.method === 'HEAD') {
      res.end();
      return;
    }

    const stream = createReadStream(filePath);
    stream.pipe(res);
  }

  const server = http.createServer(async (req: IncomingMessage, res: ServerResponse) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
    res.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      return res.end();
    }

    try {
      const url = new URL(req.url || '/', `http://localhost:${port}`);
      let pathname = decodeURIComponent(url.pathname);

      // API endpoint for models (always from doppler/models)
      if (pathname === '/api/models') {
        const modelsDir = join(dopplerDir, 'models');
        try {
          const entries = await readdir(modelsDir, { withFileTypes: true });
          const models: ModelInfo[] = [];
          for (const entry of entries) {
            if (!entry.isDirectory()) continue;
            const modelPath = `models/${entry.name}`;
            const manifestPath = join(modelsDir, entry.name, 'manifest.json');
            try {
              const manifestData = await readFile(manifestPath, 'utf-8');
              const manifest = JSON.parse(manifestData);
              const config = manifest.config || {};
              const textConfig = config.text_config || config;
              const totalSize = (manifest.shards || []).reduce((sum: number, s: { size?: number }) => sum + (s.size || 0), 0);
              models.push({
                path: modelPath,
                name: entry.name,
                architecture: manifest.architecture || config.architectures?.[0] || null,
                quantization: manifest.quantization || null,
                size: textConfig.hidden_size ? `${textConfig.num_hidden_layers || 0}L/${textConfig.hidden_size}H` : null,
                downloadSize: totalSize,
                vocabSize: textConfig.vocab_size || null,
                numLayers: textConfig.num_hidden_layers || null,
              });
            } catch {
              models.push({ path: modelPath, name: entry.name });
            }
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          return res.end(JSON.stringify(models));
        } catch {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          return res.end('[]');
        }
      }

      // Standalone mode: serve index.html at /
      // Also strip /doppler/ prefix for compatibility with old paths
      if (pathname.startsWith('/doppler/')) {
        pathname = pathname.replace('/doppler/', '/');
      } else if (pathname === '/doppler') {
        pathname = '/index.html';
      }
      if (pathname === '/' || pathname === '') {
        pathname = '/index.html';
      }

      // Serve JS files from dist/ (TypeScript is compiled there)
      // This handles: tests/benchmark/, inference/, gpu/, etc.
      if (pathname.endsWith('.js') && !pathname.includes('node_modules')) {
        // Strip /doppler/ prefix if present (added by /d/* rewrite)
        const jsPath = pathname.startsWith('/doppler/') ? pathname.slice(8) : pathname;
        const distPath = join(dopplerDir, 'dist', jsPath);
        try {
          const distStats = await stat(distPath);
          return serveFile(distPath, distStats, req, res);
        } catch {
          // Fall through to normal resolution (for vendor JS, etc.)
        }
      }

      const safePath = pathname.replace(/^(\.\.[/\\])+/, '').replace(/\.\./g, '');
      const filePath = join(rootDir, safePath);

      const resolved = resolve(filePath);
      const resolvedRoot = resolve(rootDir);
      if (!resolved.startsWith(resolvedRoot)) {
        res.writeHead(403);
        return res.end('Forbidden');
      }

      let stats: Stats;
      try {
        stats = await stat(filePath);
      } catch {
        res.writeHead(404);
        return res.end('Not found');
      }

      if (stats.isDirectory()) {
        const indexPath = join(filePath, 'index.html');
        try {
          stats = await stat(indexPath);
          return serveFile(indexPath, stats, req, res);
        } catch {
          res.writeHead(404);
          return res.end('Not found');
        }
      }

      return serveFile(filePath, stats, req, res);
    } catch (err) {
      console.error('Server error:', err);
      res.writeHead(500);
      res.end('Internal server error');
    }
  });

  server.listen(port, () => {
    const baseUrl = `http://localhost:${port}`;
    console.log(`
DOPPLER Development Server
==========================
App:    ${baseUrl}/
Models: ${baseUrl}/models/

Repository: https://github.com/clocksmith/doppler

Press Ctrl+C to stop
`);
    if (args.open) {
      openBrowser(baseUrl);
    }
  });

  process.on('SIGINT', () => {
    console.log('\nShutting down...');
    server.close(() => process.exit(0));
  });

  process.on('SIGTERM', () => {
    server.close(() => process.exit(0));
  });
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});

export { parseArgs };
