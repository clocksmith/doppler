import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import http from 'node:http';

// Shared byte/range owner for probe and resident comparison realms.
export async function createProbeServer({ entry, config, routes, files = {} }) {
  const server = http.createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://localhost');
      if (url.pathname === '/') {
        response.setHeader('Content-Type', 'text/html');
        response.end(`<script type="module" src="/adapter/${entry}"></script>`);
        return;
      }
      if (url.pathname === '/probe-config.json') {
        response.setHeader('Content-Type', 'application/json');
        response.end(JSON.stringify(config));
        return;
      }
      let file = files[url.pathname];
      if (!file) for (const [prefix, root] of routes) if (url.pathname.startsWith(prefix)) {
        const relative = decodeURIComponent(url.pathname.slice(prefix.length));
        file = path.resolve(root, relative);
        if (!file.startsWith(path.resolve(root) + path.sep)) throw new Error('Path outside served root');
        break;
      }
      if (!file) { response.writeHead(404); response.end(); return; }
      const stat = await fs.stat(file);
      if (!stat.isFile()) { response.writeHead(404); response.end(); return; }
      const range = request.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
      if (request.headers.range && !range) throw new Error('Unsupported byte range');
      const start = range ? Number(range[1]) : 0;
      const end = range ? Math.min(Number(range[2]), stat.size - 1) : stat.size - 1;
      if (start > end || end >= stat.size) throw new Error('Invalid byte extent');
      const mime = path.extname(file) === '.js' ? 'text/javascript'
        : path.extname(file) === '.json' ? 'application/json'
        : path.extname(file) === '.wasm' ? 'application/wasm' : 'application/octet-stream';
      response.writeHead(range ? 206 : 200, {
        'Content-Type': mime, 'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes',
        ...(range ? { 'Content-Range': `bytes ${start}-${end}/${stat.size}` } : {}),
      });
      createReadStream(file, { start, end }).on('error', error => response.destroy(error)).pipe(response);
    } catch (error) {
      if (!response.headersSent) response.writeHead(500);
      response.end(String(error));
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return server;
}

export const probeOrigin = server => `http://127.0.0.1:${server.address().port}`;
export const closeProbeServer = server => new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
