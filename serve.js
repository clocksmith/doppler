#!/usr/bin/env node


import http from 'http';
import { readFile, stat, readdir } from 'fs/promises';
import { createReadStream } from 'fs';
import { extname, join, dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LOG_404 = process.env.DOPPLER_LOG_404 === '1';


const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
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


function parseArgs(argv) {

  const args = {
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


function printHelp() {
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


function openBrowser(url) {
  const platform = process.platform;

  let cmd;

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


async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printHelp();
    process.exit(0);
  }

  const { port } = args;
  // Serve from doppler directory (standalone mode)
  const dopplerDir = __dirname;
  const rootDir = dopplerDir;


  function serveFile(filePath, stats, req, res) {
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


    const headers = {
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

  const server = http.createServer(async (req, res) => {
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

          const models = [];
          for (const entry of entries) {
            if (!entry.isDirectory()) continue;
            const modelPath = `models/${entry.name}`;
            const manifestPath = join(modelsDir, entry.name, 'manifest.json');
            try {
              const manifestData = await readFile(manifestPath, 'utf-8');
              const manifest = JSON.parse(manifestData);
              const config = manifest.config || {};
              const textConfig = config.text_config || config;
              const totalSize = (manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0);
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

<<<<<<< Updated upstream
<<<<<<< HEAD
=======
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
>>>>>>> a18bca1 (.)
      // Serve demo directly at root.
      if (pathname === '/' || pathname === '') {
        pathname = '/demo/index.html';
      }

      // Shortcut: /d redirects to root.
      if (pathname === '/d' || pathname === '/d/') {
        res.writeHead(302, { 'Location': '/' });
        return res.end();
      }

      // Legacy /dr redirects to root.
      if (pathname === '/dr' || pathname === '/dr/') {
        res.writeHead(302, { 'Location': '/' });
        return res.end();
      }

      // Legacy /doppler path redirects to canonical root paths.
      if (pathname === '/doppler' || pathname === '/doppler/') {
        res.writeHead(302, { 'Location': '/' });
        return res.end();
      } else if (pathname.startsWith('/doppler/')) {
        const stripped = pathname.replace(/^\/doppler/, '') || '/';
        res.writeHead(302, { 'Location': stripped });
        return res.end();
      }

      // Legacy /dr/<path> redirects to canonical root paths.
      if (pathname.startsWith('/dr/')) {
        const stripped = pathname.replace(/^\/dr/, '') || '/';
        res.writeHead(302, { 'Location': stripped });
        return res.end();
      }

      // Backward compatibility for the removed legacy app surface.
      if (pathname === '/app' || pathname === '/app/' || pathname === '/app/index.html') {
        pathname = '/demo/index.html';
<<<<<<< HEAD
=======
=======
      // Canonical app entry point.
      if (pathname === '/' || pathname === '') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Shortcut: /d redirects to the demo app
      if (pathname === '/d' || pathname === '/d/') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Legacy /dr path maps to /doppler (compatibility).
      if (pathname === '/dr' || pathname === '/dr/') {
=======
      // Canonical app entry point.
      if (pathname === '/' || pathname === '') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Shortcut: /d redirects to the demo app
      if (pathname === '/d' || pathname === '/d/') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Legacy /dr path maps to /doppler (compatibility).
      if (pathname === '/dr' || pathname === '/dr/') {
>>>>>>> Stashed changes
=======
      // Canonical app entry point.
      if (pathname === '/' || pathname === '') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Shortcut: /d redirects to the demo app
      if (pathname === '/d' || pathname === '/d/') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Legacy /dr path maps to /doppler (compatibility).
      if (pathname === '/dr' || pathname === '/dr/') {
>>>>>>> Stashed changes
=======
      // Canonical app entry point.
      if (pathname === '/' || pathname === '') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Shortcut: /d redirects to the demo app
      if (pathname === '/d' || pathname === '/d/') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Legacy /dr path maps to /doppler (compatibility).
      if (pathname === '/dr' || pathname === '/dr/') {
>>>>>>> Stashed changes
=======
      // Canonical app entry point.
      if (pathname === '/' || pathname === '') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Shortcut: /d redirects to the demo app
      if (pathname === '/d' || pathname === '/d/') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Legacy /dr path maps to /doppler (compatibility).
      if (pathname === '/dr' || pathname === '/dr/') {
>>>>>>> Stashed changes
=======
      // Canonical app entry point.
      if (pathname === '/' || pathname === '') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Shortcut: /d redirects to the demo app
      if (pathname === '/d' || pathname === '/d/') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Legacy /dr path maps to /doppler (compatibility).
      if (pathname === '/dr' || pathname === '/dr/') {
>>>>>>> Stashed changes
=======
      // Canonical app entry point.
      if (pathname === '/' || pathname === '') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Shortcut: /d redirects to the demo app
      if (pathname === '/d' || pathname === '/d/') {
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      }

      // Legacy /dr path maps to /doppler (compatibility).
      if (pathname === '/dr' || pathname === '/dr/') {
>>>>>>> Stashed changes
        res.writeHead(302, { 'Location': '/doppler' });
        return res.end();
      } else if (pathname.startsWith('/dr/')) {
        pathname = pathname.replace('/dr/', '/doppler/');
      }

      // Standalone mode: serve app/index.html at /
      // Strip /doppler/ prefix for prefixed routing.
      if (pathname.startsWith('/doppler/')) {
        pathname = pathname.replace('/doppler/', '/');
      } else if (pathname === '/doppler') {
        pathname = '/app/index.html';
>>>>>>> Stashed changes
>>>>>>> a18bca1 (.)
      }
      if (pathname === '/rd.css') {
        pathname = '/styles/rd.css';
      }
      // Backward compatibility for older app shell references.
      if (pathname === '/app/rd.css') {
        pathname = '/styles/rd.css';
      }
      if (pathname === '/kernel-tests/browser/registry.json') {
        pathname = '/config/kernels/registry.json';
      }
      if (
        pathname === '/favicon.ico' ||
        pathname === '/site.webmanifest' ||
        pathname === '/browserconfig.xml' ||
        pathname === '/apple-touch-icon.png' ||
        pathname === '/apple-touch-icon-precomposed.png' ||
        pathname === '/mstile-150x150.png' ||
        pathname === '/android-chrome-192x192.png' ||
        pathname === '/android-chrome-512x512.png'
      ) {
        res.writeHead(204);
        return res.end();
      }
      if (pathname === '/' || pathname === '') {
        pathname = '/demo/index.html';
      }

      // Serve JS and JSON files from dist/ (TypeScript is compiled there)
      // This handles: tests/benchmark/, inference/, gpu/, etc.
      // Note: /doppler/ prefix was already stripped above, so pathname is like /dist/config/...
      if ((pathname.endsWith('.js') || pathname.endsWith('.json')) && !pathname.includes('node_modules')) {
        const jsPath = pathname.startsWith('/') ? pathname.slice(1) : pathname;
        const pathWithoutDist = jsPath.replace(/^dist\//, '');
        // Try multiple locations in order:
        // 1. dist/src/<path> - where tsc outputs src/** files
        // 2. dist/<path> - for test/benchmark outputs
        // 3. src/<path> - for raw JS files not processed by tsc (kernels, platforms)
        const candidates = [
          join(dopplerDir, 'dist', 'src', pathWithoutDist),
          join(dopplerDir, 'dist', jsPath),
          join(dopplerDir, 'src', pathWithoutDist),
        ];
        for (const candidate of candidates) {
          try {
            const stats = await stat(candidate);
            return serveFile(candidate, stats, req, res);
          } catch {
            // Try next candidate
          }
        }
        // Fall through to normal resolution (for vendor JS, etc.)
      }

      // Serve WGSL shader files from src/gpu/kernels/
      // Requested at /gpu/kernels/*.wgsl but files are in src/gpu/kernels/
      if (pathname.endsWith('.wgsl')) {
        const wgslPath = pathname.startsWith('/') ? pathname.slice(1) : pathname;
        const srcPath = join(dopplerDir, 'src', wgslPath);
        try {
          const stats = await stat(srcPath);
          return serveFile(srcPath, stats, req, res);
        } catch {
          // Fall through to normal resolution
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

      let stats;
      try {
        stats = await stat(filePath);
      } catch {
        if (LOG_404) {
          console.log(`[404] ${pathname}`);
        }
        res.writeHead(404);
        return res.end('Not found');
      }

      if (stats.isDirectory()) {
        const indexPath = join(filePath, 'index.html');
        try {
          stats = await stat(indexPath);
          return serveFile(indexPath, stats, req, res);
        } catch {
          if (LOG_404) {
            console.log(`[404] ${pathname}`);
          }
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
