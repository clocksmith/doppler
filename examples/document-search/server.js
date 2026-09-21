import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 8080;
const HOST = process.env.HOST || '127.0.0.1';

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.wgsl': 'text/plain; charset=utf-8',
};

const UPSTREAM_SHARDS = {
  embedding: {
    repo: '049000f49325dca7db2ed2c9de2c8881bd0f4603/models/qwen-3-embedding-0-6b-q4k-ehf16-af32',
    localDir: path.resolve(DIR, '../../models/local/qwen-3-embedding-0-6b-q4k-ehf16-af32'),
  },
  reranker: {
    repo: 'f86fe245b9bbc275cd69af46b1d45d47ea685a55/models/qwen-3-reranker-0-6b-q4k-ehf16-af32',
    localDir: path.resolve(DIR, '../../models/local/qwen-3-reranker-0-6b-q4k-ehf16-af32'),
  },
};

async function resolveFilePath(urlPath) {
  const cleanPath = urlPath.split('?')[0].replace(/^\/+/, '');
  const target = cleanPath === '' ? 'index.html' : cleanPath;

  // 1. Direct file in application directory
  const localFile = path.resolve(DIR, target);
  if (localFile.startsWith(DIR) && fs.existsSync(localFile) && fs.statSync(localFile).isFile()) {
    return localFile;
  }

  // 2. Runtime files: ./runtime/src/... -> check node_modules or repo root src
  if (target.startsWith('runtime/src/')) {
    const subPath = target.slice('runtime/src/'.length);
    const inNodeModules = path.resolve(DIR, 'node_modules/doppler-gpu/src', subPath);
    if (fs.existsSync(inNodeModules) && fs.statSync(inNodeModules).isFile()) {
      return inNodeModules;
    }
    const inRepo = path.resolve(DIR, '../../src', subPath);
    if (fs.existsSync(inRepo) && fs.statSync(inRepo).isFile()) {
      return inRepo;
    }
  }

  // 3. Shard files: check local model cache if present in repo
  const shardMatch = target.match(/^capsules\/(embedding|reranker)\/artifacts\/model\/(shard_\d+\.bin)$/);
  if (shardMatch) {
    const [, role, shardName] = shardMatch;
    const localModelFile = path.resolve(UPSTREAM_SHARDS[role].localDir, shardName);
    if (fs.existsSync(localModelFile) && fs.statSync(localModelFile).isFile()) {
      return localModelFile;
    }
  }

  return null;
}

export function createServer() {
  return http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', '*');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405, { 'Content-Type': 'text/plain' });
      res.end('Method Not Allowed');
      return;
    }

    const filePath = await resolveFilePath(req.url);

    if (filePath) {
      const ext = path.extname(filePath).toLowerCase();
      const contentType = MIME_TYPES[ext] || 'application/octet-stream';
      const stat = await fsp.stat(filePath);
      const range = req.headers.range;

      if (range) {
        const parts = range.replace(/bytes=/, '').split('-');
        const start = parseInt(parts[0], 10);
        const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
        const chunkSize = end - start + 1;

        res.writeHead(206, {
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': chunkSize,
          'Content-Type': contentType,
        });

        if (req.method === 'HEAD') {
          res.end();
          return;
        }

        fs.createReadStream(filePath, { start, end }).pipe(res);
        return;
      }

      res.writeHead(200, {
        'Content-Length': stat.size,
        'Content-Type': contentType,
        'Accept-Ranges': 'bytes',
      });

      if (req.method === 'HEAD') {
        res.end();
        return;
      }

      fs.createReadStream(filePath).pipe(res);
      return;
    }

    // 4. Shard proxy from Hugging Face if not present locally
    const shardMatch = req.url.split('?')[0].replace(/^\/+/, '').match(/^capsules\/(embedding|reranker)\/artifacts\/model\/(shard_\d+\.bin)$/);
    if (shardMatch) {
      const [, role, shardName] = shardMatch;
      const hfUrl = `https://huggingface.co/clocksmith/rdrr/resolve/${UPSTREAM_SHARDS[role].repo}/${shardName}`;
      try {
        const hfRes = await fetch(hfUrl, {
          headers: req.headers.range ? { range: req.headers.range } : {},
        });

        if (!hfRes.ok && hfRes.status !== 206) {
          res.writeHead(hfRes.status, { 'Content-Type': 'text/plain' });
          res.end(`Upstream fetch failed: ${hfRes.status}`);
          return;
        }

        const headers = {
          'Content-Type': 'application/octet-stream',
          'Accept-Ranges': 'bytes',
        };
        if (hfRes.headers.has('content-length')) headers['Content-Length'] = hfRes.headers.get('content-length');
        if (hfRes.headers.has('content-range')) headers['Content-Range'] = hfRes.headers.get('content-range');

        res.writeHead(hfRes.status, headers);
        if (req.method === 'HEAD') {
          res.end();
          return;
        }

        const stream = hfRes.body;
        if (stream) {
          for await (const chunk of stream) {
            res.write(chunk);
          }
        }
        res.end();
        return;
      } catch (err) {
        res.writeHead(502, { 'Content-Type': 'text/plain' });
        res.end(`Upstream proxy error: ${err.message}`);
        return;
      }
    }

    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end(`Not Found: ${req.url}`);
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const server = createServer();
  server.listen(PORT, HOST, () => {
    console.log(`Local document search running at: http://${HOST}:${PORT}/index.html`);
  });

  const stop = () => {
    server.close(() => process.exit(0));
  };
  process.on('SIGINT', stop);
  process.on('SIGTERM', stop);
}
