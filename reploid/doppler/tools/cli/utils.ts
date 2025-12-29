/**
 * CLI Utilities - Server management, browser setup, and build helpers
 */

import type { Page, BrowserContext } from 'playwright';
import { resolve, dirname, join, extname } from 'path';
import { fileURLToPath } from 'url';
import { spawn, type ChildProcess } from 'child_process';
import { open, readFile, readdir, stat } from 'fs/promises';
import type { CLIOptions } from './types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Track spawned server process for cleanup
let serverProcess: ChildProcess | null = null;

// ============================================================================
// Build Management
// ============================================================================

/**
 * Run TypeScript compilation for kernel tests
 */
export async function runBuild(verbose: boolean): Promise<void> {
  console.log('Building kernel tests...');
  const projectRoot = resolve(__dirname, '../..');

  return new Promise((resolve, reject) => {
    const build = spawn('npx', ['tsc', '--project', 'tsconfig.json'], {
      cwd: projectRoot,
      stdio: verbose ? 'inherit' : ['ignore', 'pipe', 'pipe'],
      shell: true,
    });

    let stderr = '';
    if (!verbose && build.stderr) {
      build.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });
    }

    build.on('error', (err) => {
      reject(new Error(`Build failed to start: ${err.message}`));
    });

    build.on('exit', (code) => {
      if (code === 0) {
        console.log('Build complete.');
        resolve();
      } else {
        reject(new Error(`Build failed with code ${code}${stderr ? `: ${stderr}` : ''}`));
      }
    });
  });
}

/**
 * Run the benchmark build (esbuild bundle)
 */
export async function runBenchmarkBuild(verbose: boolean): Promise<void> {
  console.log('Building benchmark bundle...');
  const projectRoot = resolve(__dirname, '../..');

  return new Promise((resolve, reject) => {
    const build = spawn('npm', ['run', 'build:benchmark'], {
      cwd: projectRoot,
      stdio: verbose ? 'inherit' : ['ignore', 'pipe', 'pipe'],
      shell: true,
    });

    let stderr = '';
    if (!verbose && build.stderr) {
      build.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });
    }

    build.on('error', (err) => {
      reject(new Error(`Build failed to start: ${err.message}`));
    });

    build.on('exit', (code) => {
      if (code === 0) {
        console.log('Benchmark build complete.');
        resolve();
      } else {
        reject(new Error(`Build failed with code ${code}${stderr ? `: ${stderr}` : ''}`));
      }
    });
  });
}

// ============================================================================
// Server Management
// ============================================================================

/**
 * Check if the dev server is already running
 */
export async function isServerRunning(baseUrl: string): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const response = await fetch(baseUrl, { signal: controller.signal });
    clearTimeout(timeout);
    return response.ok || response.status === 304;
  } catch {
    return false;
  }
}

/**
 * Start the dev server and wait for it to be ready
 */
export async function ensureServerRunning(baseUrl: string, verbose: boolean): Promise<void> {
  if (await isServerRunning(baseUrl)) {
    if (verbose) {
      console.log('Server already running at', baseUrl);
    }
    return;
  }

  console.log('Starting dev server...');

  const projectRoot = resolve(__dirname, '../..');
  serverProcess = spawn('npm', ['run', 'start'], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
    shell: true,
  });

  if (verbose && serverProcess.stdout) {
    serverProcess.stdout.on('data', (data: Buffer) => {
      console.log(`[server] ${data.toString().trim()}`);
    });
  }
  if (serverProcess.stderr) {
    serverProcess.stderr.on('data', (data: Buffer) => {
      const msg = data.toString().trim();
      if (msg && !msg.includes('ExperimentalWarning')) {
        console.error(`[server] ${msg}`);
      }
    });
  }

  serverProcess.on('error', (err) => {
    console.error('Failed to start server:', err.message);
    serverProcess = null;
  });

  serverProcess.on('exit', (code) => {
    if (code !== null && code !== 0) {
      console.error(`Server exited with code ${code}`);
    }
    serverProcess = null;
  });

  // Wait for server to be ready
  const maxWait = 30000;
  const pollInterval = 500;
  const startTime = Date.now();

  while (Date.now() - startTime < maxWait) {
    if (await isServerRunning(baseUrl)) {
      console.log('Server ready at', baseUrl);
      return;
    }
    await new Promise((r) => setTimeout(r, pollInterval));
  }

  throw new Error(`Server failed to start within ${maxWait / 1000}s`);
}

// ============================================================================
// Local Static Routing (No-Server Mode)
// ============================================================================

const MIME_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.wgsl': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function parseByteRange(rangeHeader: string | undefined, size: number): { start: number; end: number } | null {
  if (!rangeHeader) return null;
  const match = rangeHeader.match(/bytes=(\d*)-(\d*)/);
  if (!match) return null;

  const start = match[1] ? Number.parseInt(match[1], 10) : 0;
  const end = match[2] ? Number.parseInt(match[2], 10) : size - 1;

  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  if (start < 0 || end < start || start >= size) return null;

  return { start, end: Math.min(end, size - 1) };
}

async function buildModelsApiResponse(modelsDir: string): Promise<string> {
  try {
    const entries = await readdir(modelsDir, { withFileTypes: true });
    const models: any[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const manifestPath = join(modelsDir, entry.name, 'manifest.json');
      try {
        const manifestData = await readFile(manifestPath, 'utf-8');
        const manifest = JSON.parse(manifestData);
        const config = manifest.config || {};
        const textConfig = config.text_config || config;
        const totalSize = (manifest.shards || []).reduce((sum: number, s: { size?: number }) => sum + (s.size || 0), 0);
        models.push({
          path: `models/${entry.name}`,
          name: entry.name,
          architecture: manifest.architecture || config.architectures?.[0] || null,
          quantization: manifest.quantization || null,
          size: textConfig.hidden_size ? `${textConfig.num_hidden_layers || 0}L/${textConfig.hidden_size}H` : null,
          downloadSize: totalSize,
          vocabSize: textConfig.vocab_size || null,
          numLayers: textConfig.num_hidden_layers || null,
        });
      } catch {
        models.push({ path: `models/${entry.name}`, name: entry.name });
      }
    }
    return JSON.stringify(models);
  } catch {
    return '[]';
  }
}

/**
 * Install Playwright routes to serve DOPPLER static assets from disk.
 *
 * This enables running tests/benchmarks in environments where binding a local
 * dev server is not permitted (e.g., sandboxed runners).
 */
export async function installLocalDopplerRoutes(page: Page, opts: CLIOptions): Promise<void> {
  const baseOrigin = new URL(opts.baseUrl).origin;

  const projectRoot = resolve(__dirname, '../..');
  const modelsDir = join(projectRoot, 'models');

  const pattern = `${baseOrigin}/**`;
  if (opts.verbose) {
    console.log(`[LocalServer] Installing static routes for ${pattern}`);
  }

  await page.route(pattern, async (route) => {
    const request = route.request();
    const url = new URL(request.url());

    let pathname = decodeURIComponent(url.pathname);

    // API endpoint: /api/models
    if (pathname === '/api/models') {
      const body = await buildModelsApiResponse(modelsDir);
      return route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'application/json; charset=utf-8',
          'Access-Control-Allow-Origin': '*',
        },
        body,
      });
    }

    // Firebase-style rewrites (serve.ts default mode)
    if (pathname === '/') {
      pathname = '/landing.html';
    } else if (pathname === '/d') {
      pathname = '/doppler/index.html';
    } else if (pathname.startsWith('/d/')) {
      pathname = '/doppler' + pathname.slice(2);
    } else if (pathname.startsWith('/models/')) {
      pathname = '/doppler' + pathname;
    }

    // Prefer JS from dist/ when available (matches serve.ts behavior).
    let resolvedPathname = pathname;
    if (pathname.endsWith('.js') && !pathname.includes('node_modules')) {
      const jsPath = pathname.startsWith('/doppler/') ? pathname.slice(8) : pathname;
      const distPath = join(projectRoot, 'dist', jsPath);
      try {
        await stat(distPath);
        resolvedPathname = `/__dist__${jsPath}`;
      } catch {
        // fall through
      }
    }

    // Resolve to filesystem path with traversal guard
    const safePath = resolvedPathname.replace(/^(\.\.[/\\])+/, '').replace(/\.\./g, '');

    let filePath: string;
    if (safePath.startsWith('/__dist__/')) {
      filePath = join(projectRoot, 'dist', safePath.slice('/__dist__/'.length));
    } else {
      filePath = join(projectRoot, safePath);
    }

    const resolved = resolve(filePath);
    const allowedRoot = resolve(safePath.startsWith('/__dist__/') ? join(projectRoot, 'dist') : projectRoot);
    if (!resolved.startsWith(allowedRoot)) {
      return route.fulfill({ status: 403, body: 'Forbidden' });
    }

    let fileStats: any;
    try {
      fileStats = await stat(resolved);
    } catch {
      return route.fulfill({ status: 404, body: 'Not found' });
    }

    if (fileStats.isDirectory()) {
      const indexPath = join(resolved, 'index.html');
      try {
        const indexStats = await stat(indexPath);
        fileStats = indexStats;
        filePath = indexPath;
      } catch {
        return route.fulfill({ status: 404, body: 'Not found' });
      }
    } else {
      filePath = resolved;
    }

    const ext = extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    const headers: Record<string, string> = {
      'Content-Type': contentType,
      'Access-Control-Allow-Origin': '*',
      'Accept-Ranges': 'bytes',
    };

    // Disable caching for JS to make iteration less confusing.
    if (ext === '.js' || ext === '.mjs') {
      headers['Cache-Control'] = 'no-store, no-cache, must-revalidate';
      headers['Pragma'] = 'no-cache';
      headers['Expires'] = '0';
    }

    const range = parseByteRange(request.headers()['range'], fileStats.size);
    if (range) {
      const length = range.end - range.start + 1;
      const handle = await open(filePath, 'r');
      try {
        const buf = Buffer.alloc(length);
        const { bytesRead } = await handle.read(buf, 0, length, range.start);
        headers['Content-Range'] = `bytes ${range.start}-${range.start + bytesRead - 1}/${fileStats.size}`;
        headers['Content-Length'] = String(bytesRead);
        return route.fulfill({ status: 206, headers, body: buf.subarray(0, bytesRead) });
      } finally {
        await handle.close();
      }
    }

    const body = await readFile(filePath);
    headers['Content-Length'] = String(body.byteLength);
    return route.fulfill({ status: 200, headers, body });
  });
}

/**
 * Stop the server if we started it
 */
export function stopServer(): void {
  if (serverProcess) {
    console.log('Stopping dev server...');
    serverProcess.kill('SIGTERM');
    serverProcess = null;
  }
}

// Clean up server on exit
process.on('exit', stopServer);
process.on('SIGINT', () => {
  stopServer();
  process.exit(130);
});
process.on('SIGTERM', () => {
  stopServer();
  process.exit(143);
});

// ============================================================================
// Browser Setup
// ============================================================================

export type BrowserProfileScope = 'test' | 'bench';

export async function createBrowserContext(
  opts: CLIOptions,
  options: { scope?: BrowserProfileScope; devtools?: boolean } = {}
): Promise<BrowserContext> {
  // Playwright host platform detection breaks in some sandboxed environments
  // because `os.cpus()` can be empty, which causes Playwright to assume mac-x64
  // even on arm64. Force arm64 so the installed browser binaries are found.
  if (process.platform === 'darwin' && process.arch === 'arm64' && !process.env.PLAYWRIGHT_HOST_PLATFORM_OVERRIDE) {
    const os = await import('os');
    const ver = os.release().split('.').map((a) => Number.parseInt(a, 10));
    let macVersion = 'mac15-arm64';
    if (ver[0] < 18) macVersion = 'mac10.13-arm64';
    else if (ver[0] === 18) macVersion = 'mac10.14-arm64';
    else if (ver[0] === 19) macVersion = 'mac10.15-arm64';
    else {
      const LAST_STABLE_MACOS_MAJOR_VERSION = 15;
      macVersion = `mac${Math.min(ver[0] - 9, LAST_STABLE_MACOS_MAJOR_VERSION)}-arm64`;
    }
    process.env.PLAYWRIGHT_HOST_PLATFORM_OVERRIDE = macVersion;
  }

  const dopplerRoot = resolve(__dirname, '../..');
  const defaultDirName = options.scope === 'bench' ? '.benchmark-cache' : '.test-cache';
  const userDataDir = opts.profileDir
    ? resolve(dopplerRoot, opts.profileDir)
    : resolve(dopplerRoot, defaultDirName);

  const args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan'];
  if (!opts.headless && options.devtools) {
    args.push('--auto-open-devtools-for-tabs');
  }
  // Position window off-screen to avoid focus stealing in headed mode
  // Use --minimized flag in CLI to enable this behavior
  if (!opts.headless && opts.minimized) {
    args.push('--window-position=-2000,-2000');
  }

  const { chromium } = await import('playwright');

  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: opts.headless,
    devtools: Boolean(!opts.headless && options.devtools),
    args,
  });

  // Set default timeout from CLI options (default 120s, vs Playwright's 30s)
  context.setDefaultTimeout(opts.timeout);

  return context;
}

export async function setupPage(context: BrowserContext, opts: CLIOptions): Promise<Page> {
  const page = context.pages()[0] || await context.newPage();

  // Console logging
  const relevantTags = ['[Test]', '[Benchmark]', '[GPU]', 'ERROR', 'PASS', 'FAIL', 'Failed', 'error', 'WebGPU'];
  page.on('console', (msg) => {
    const text = msg.text();
    const isRelevant = relevantTags.some((tag) => text.includes(tag));
    if (opts.verbose || isRelevant) {
      console.log(`  [browser] ${text}`);
    }
  });

  page.on('pageerror', (err) => {
    console.error(`  [browser error] ${err.message}`);
  });

  // Log all network failures
  page.on('requestfailed', (req) => {
    console.log(`  [network 404] ${req.url()}`);
  });

  if (opts.noServer) {
    await installLocalDopplerRoutes(page, opts);
  }

  return page;
}

// ============================================================================
// Result Utilities
// ============================================================================

export function generateResultFilename(result: any): string {
  const suite = result.suite || 'pipeline';
  const model = result.model?.modelName || result.model?.modelId || 'unknown';
  const modelSlug = model.replace(/[^a-zA-Z0-9-_]/g, '-').toLowerCase();

  const gpu = result.env?.gpu?.description || result.env?.gpu?.device || '';
  const gpuSlug = gpu
    .replace(/[^a-zA-Z0-9-_]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase()
    .slice(0, 30);

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

  if (gpuSlug) {
    return `${suite}_${modelSlug}_${gpuSlug}_${timestamp}.json`;
  }
  return `${suite}_${modelSlug}_${timestamp}.json`;
}
