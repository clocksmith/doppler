import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const MIME_TYPES = { '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json',
  '.css': 'text/css', '.txt': 'text/plain', '.md': 'text/plain', '.wgsl': 'text/plain' };
const inside = (root, filename) => filename.startsWith(root + path.sep);

export function createServer({ root = DIR, fetch: acquire = globalThis.fetch } = {}) {
  root = path.resolve(root);
  return http.createServer(async (req, res) => {
    try {
      if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
      const target = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
      if (target.split('/').some(part => part === '..' || part === '.') || target.includes('\\')) {
        res.writeHead(400).end('Invalid path'); return;
      }
      const shard = /^capsules\/(embedding|reranker)\/artifacts\/model\/(shard_\d+\.bin)$/.exec(target);
      if (shard) {
        // Exact signed descriptors, never repository caches or guessed dtype routes.
        const sources = JSON.parse(await fsp.readFile(path.join(root, 'shard-sources.json'), 'utf8'));
        const source = sources.artifacts[target];
        if (!source?.url) { res.writeHead(503).end('No published source for declared artifact: ' + target); return; }
        const capsule = JSON.parse(await fsp.readFile(path.join(root, 'capsules', shard[1], 'capsule-v3.json'), 'utf8'));
        const artifact = capsule.artifacts.find(entry => entry.path === 'artifacts/model/' + shard[2]);
        if (!artifact || source.hash !== artifact.hash || source.sizeBytes !== artifact.sizeBytes) {
          throw new Error('Shard route does not match the accepted Capsule.');
        }
        const controller = new AbortController();
        res.on('close', () => { if (!res.writableEnded) controller.abort(); });
        const upstream = await acquire(source.url, { method: req.method, signal: controller.signal,
          headers: req.headers.range ? { range: req.headers.range } : {} });
        if (!upstream.ok) { res.writeHead(upstream.status).end('Declared artifact acquisition failed.'); return; }
        const headers = { 'Content-Type': 'application/octet-stream' };
        for (const key of ['content-length', 'content-range', 'accept-ranges']) {
          if (upstream.headers.has(key)) headers[key] = upstream.headers.get(key);
        }
        res.writeHead(upstream.status, headers);
        if (req.method === 'HEAD') res.end();
        else await pipeline(Readable.fromWeb(upstream.body), res);
        return;
      }
      // Runtime URLs resolve ONLY inside this starter's installed package.
      const runtime = target.startsWith('runtime/');
      const base = runtime ? path.join(root, 'node_modules/doppler-gpu') : root;
      const filename = path.resolve(base, runtime ? target.slice('runtime/'.length) : target);
      if (!inside(base, filename) || target.startsWith('node_modules/') || target.startsWith('vendor/')) {
        res.writeHead(404).end(); return;
      }
      const real = await fsp.realpath(filename);
      if (!inside(base, real)) { res.writeHead(404).end(); return; }
      const stat = await fsp.stat(real);
      if (!stat.isFile()) { res.writeHead(404).end(); return; }
      const headers = { 'Content-Type': MIME_TYPES[path.extname(real)] ?? 'application/octet-stream',
        'Content-Length': stat.size, 'Accept-Ranges': 'bytes' };
      let range;
      if (req.headers.range) {
        const match = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range);
        const start = Number(match?.[1]);
        const end = match?.[2] ? Number(match[2]) : stat.size - 1;
        if (!match || !Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= stat.size) {
          res.writeHead(416, { 'Content-Range': 'bytes */' + stat.size }).end(); return;
        }
        range = { start, end };
        headers['Content-Range'] = 'bytes ' + start + '-' + end + '/' + stat.size;
        headers['Content-Length'] = end - start + 1;
      }
      res.writeHead(range ? 206 : 200, headers);
      if (req.method === 'HEAD') res.end();
      else await pipeline(fs.createReadStream(real, range), res);
    } catch (error) {
      if (!res.headersSent) res.writeHead(error.code === 'ENOENT' ? 404 : 500, { 'Content-Type': 'text/plain' }).end(error.message);
      else res.destroy(error);
    }
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT) || 8080;
  const host = process.env.HOST || '127.0.0.1';
  const server = createServer();
  server.listen(port, host, () => console.log('Local document search: http://' + host + ':' + port + '/index.html'));
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close(() => process.exit(0)));
}
