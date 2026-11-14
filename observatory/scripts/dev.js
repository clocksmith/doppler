// Local server for the Observatory. It mirrors the hosted layout: the pages at
// the root, the flags and protocol files under /config/, and the Doppler runtime
// under /doppler/ (served from this repository).
import http from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { handleApi } from '../server/api.js';
import { digestFiles, offlineShell, readDemoShell } from './offline-shell.js';
import { readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dopplerRoot = path.resolve(process.env.DOPPLER_ROOT ?? path.join(root, '..'));
// The preview runtime is always the sibling checkout unless told otherwise. To mirror
// the hosted site, point DOPPLER_ROOT at d4da's hosted copy and leave this alone.
const previewRoot = path.resolve(process.env.DOPPLER_PREVIEW_ROOT ?? path.join(root, '..'));
// A fixed port keeps one origin, so a model saved by the browser is found again.
const port = Number(process.env.PORT ?? 8791);

const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8', '.css': 'text/css; charset=utf-8', '.wgsl': 'text/plain; charset=utf-8', '.wasm': 'application/wasm', '.svg': 'image/svg+xml' };

const MOUNTS = [
  { prefix: '/doppler/src/', dir: path.join(dopplerRoot, 'src') },
  { prefix: '/doppler/demo/', dir: path.join(dopplerRoot, 'demo') },
  { prefix: '/doppler/models/', dir: path.join(dopplerRoot, 'models') },
  { prefix: '/runtime-preview/src/', dir: path.join(previewRoot, 'src') },
  { prefix: '/runtime-preview/models/', dir: path.join(previewRoot, 'models') },
  { prefix: '/config/', dir: path.join(root, 'config') },
  { prefix: '/', dir: path.join(root, 'public') },
];

const git = (cwd, ...args) => promisify(execFile)('git', ['-C', cwd, ...args]).then((out) => out.stdout.trim());

// A hosted copy of Doppler carries release.json. A source checkout does not, so
// the same facts are read from the checkout: evidence then names what ran.
async function runtimeRelease(checkout, { preview }) {
  const hosted = await readFile(path.join(checkout, 'release.json'), 'utf8').catch(() => null);
  if (hosted) return hosted;
  const [sourceCommit, changes, described, pkg] = await Promise.all([
    git(checkout, 'rev-parse', 'HEAD'),
    git(checkout, 'status', '--porcelain', '--', 'src'),
    git(checkout, 'describe', '--tags', '--always').catch(() => null),
    readFile(path.join(checkout, 'package.json'), 'utf8').then(JSON.parse),
  ]);
  return JSON.stringify({
    ...(preview ? { channel: 'preview', qualified: false } : {}),
    runtime: { sourceCommit, version: pkg.version, label: previewLabel(pkg.version, described), uncommittedSourceChanges: changes.length > 0 },
    servedFrom: 'source checkout',
  });
}

// "0.7.0 plus 9 commits" for a checkout that has moved past its release tag.
export function previewLabel(version, described) {
  const ahead = described?.match(/^v?(.+)-(\d+)-g[0-9a-f]+$/);
  return ahead && ahead[1] === version ? `${version} plus ${ahead[2]} commit${ahead[2] === '1' ? '' : 's'}` : version;
}

function resolveFile(pathname) {
  const mount = MOUNTS.find((item) => pathname.startsWith(item.prefix));
  const relative = pathname.slice(mount.prefix.length) || 'index.html';
  const file = path.resolve(mount.dir, relative);
  return file === mount.dir || file.startsWith(mount.dir + path.sep) ? file : null;
}

export function startServer({ listenPort = port } = {}) {
  const server = http.createServer(async (request, response) => {
    let pathname;
    try { pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname); } catch { response.writeHead(400).end(); return; }
    if (await handleApi(request, response, { switchesFile: path.join(root, 'server', 'switches.json'), protocolFile: path.join(root, 'config', 'protocols', 'device-benchmark.json') })) return;
    // What the offline worker keeps: the served demo's shell and this site's files.
    if (pathname === '/offline-shell.json') {
      try {
        const publicDir = path.join(root, 'public');
        const siteFiles = (await readdir(publicDir, { recursive: true })).map((file) => file.split(path.sep).join('/')).filter((file) => /\.(js|css|html)$/.test(file) && file !== 'sw.js');
        const shell = offlineShell({ ...(await readDemoShell(dopplerRoot)), siteFiles: [...siteFiles, 'app-manifest.json', 'config/flags.json', 'config/runtimes.json'], siteDigest: await digestFiles(publicDir, siteFiles) });
        response.writeHead(200, { 'content-type': TYPES['.json'], 'cache-control': 'no-store' }).end(JSON.stringify(shell));
      } catch (error) {
        response.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' }).end(`The offline file list could not be built: ${error.message}`);
      }
      return;
    }
    if (pathname === '/doppler/release.json' || pathname === '/runtime-preview/release.json') {
      const preview = pathname.startsWith('/runtime-preview/');
      const body = await runtimeRelease(preview ? previewRoot : dopplerRoot, { preview }).catch(() => null);
      response.writeHead(body ? 200 : 404, { 'content-type': TYPES['.json'], 'cache-control': 'no-store' }).end(body ?? '{}');
      return;
    }
    let file = resolveFile(pathname);
    let info = file ? await stat(file).catch(() => null) : null;
    if (info?.isDirectory()) { file = path.join(file, 'index.html'); info = await stat(file).catch(() => null); }
    if (!info?.isFile()) { response.writeHead(404, { 'content-type': 'text/plain' }).end('Not found'); return; }
    response.writeHead(200, { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream', 'content-length': info.size, 'cache-control': 'no-store' });
    if (request.method === 'HEAD') { response.end(); return; }
    createReadStream(file).pipe(response);
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(listenPort, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${listenPort}/` }));
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { url } = await startServer();
  console.log(`Doppler observatory prototype: ${url}`);
  console.log(`Doppler runtime served from ${dopplerRoot}`);
  console.log(`Preview runtime served from ${previewRoot} (add ?preview=runtime.preview&runtime=preview)`);
}
