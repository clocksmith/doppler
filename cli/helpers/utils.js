/**
 * CLI Utilities - Server management, browser setup, and build helpers
 */

import { resolve, dirname, join, extname } from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { open, readFile, readdir, stat } from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/** @type {import('child_process').ChildProcess | null} */
let serverProcess = null;

// ============================================================================
// Build Management
// ============================================================================

/**
 * Run TypeScript compilation for kernel tests
 * @param {boolean} verbose
 * @returns {Promise<void>}
 */
export async function runBuild(verbose) {
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
      build.stderr.on('data', (/** @type {Buffer} */ data) => {
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
 * @param {boolean} verbose
 * @returns {Promise<void>}
 */
export async function runBenchmarkBuild(verbose) {
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
      build.stderr.on('data', (/** @type {Buffer} */ data) => {
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
 * @param {string} baseUrl
 * @returns {Promise<boolean>}
 */
export async function isServerRunning(baseUrl) {
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
 * @param {string} baseUrl
 * @param {boolean} verbose
 * @returns {Promise<void>}
 */
export async function ensureServerRunning(baseUrl, verbose) {
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
    serverProcess.stdout.on('data', (/** @type {Buffer} */ data) => {
      console.log(`[server] ${data.toString().trim()}`);
    });
  }
  if (serverProcess.stderr) {
    serverProcess.stderr.on('data', (/** @type {Buffer} */ data) => {
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

/** @type {Record<string, string>} */
const MIME_TYPES = {
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

/**
 * @param {string | undefined} rangeHeader
 * @param {number} size
 * @returns {{ start: number; end: number } | null}
 */
function parseByteRange(rangeHeader, size) {
  if (!rangeHeader) return null;
  const match = rangeHeader.match(/bytes=(\d*)-(\d*)/);
  if (!match) return null;

  const start = match[1] ? Number.parseInt(match[1], 10) : 0;
  const end = match[2] ? Number.parseInt(match[2], 10) : size - 1;

  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  if (start < 0 || end < start || start >= size) return null;

  return { start, end: Math.min(end, size - 1) };
}

/**
 * @param {Record<string, unknown>} manifest
 * @returns {string | null}
 */
function formatQuantizationSummary(manifest) {
  const info = /** @type {Record<string, unknown> | undefined} */ (manifest.quantizationInfo);
  if (info && typeof info === 'object') {
    const variantTag = typeof info.variantTag === 'string' ? info.variantTag : null;
    if (variantTag) return variantTag;
    const weights = typeof info.weights === 'string' ? info.weights : null;
    const embeddings = typeof info.embeddings === 'string' ? info.embeddings : null;
    if (weights && embeddings) return `w${weights}-emb${embeddings}`;
    if (weights) return `w${weights}`;
  }
  return typeof manifest.quantization === 'string' ? manifest.quantization : null;
}

/**
 * @param {string} modelsDir
 * @returns {Promise<string>}
 */
async function buildModelsApiResponse(modelsDir) {
  try {
    const entries = await readdir(modelsDir, { withFileTypes: true });
    /** @type {any[]} */
    const models = [];
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const manifestPath = join(modelsDir, entry.name, 'manifest.json');
      try {
        const manifestData = await readFile(manifestPath, 'utf-8');
        const manifest = JSON.parse(manifestData);
        const archInfo = manifest.architecture;
        const archLabel = typeof archInfo === 'string' ? archInfo : (manifest.modelType ?? null);
        const archConfig = archInfo && typeof archInfo === 'object' ? archInfo : null;
        const totalSize = (manifest.shards || []).reduce((/** @type {number} */ sum, /** @type {{ size?: number }} */ s) => sum + (s.size || 0), 0);
        const numLayers = archConfig?.numLayers ?? null;
        const hiddenSize = archConfig?.hiddenSize ?? null;
        const vocabSize = archConfig?.vocabSize ?? null;
        models.push({
          path: `models/${entry.name}`,
          name: entry.name,
          architecture: archLabel,
          quantization: formatQuantizationSummary(manifest) || null,
          size: hiddenSize && numLayers ? `${numLayers}L/${hiddenSize}H` : null,
          downloadSize: totalSize,
          vocabSize,
          numLayers,
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
 * @param {import('playwright').Page} page
 * @param {import('./types.js').CLIOptions} opts
 * @returns {Promise<void>}
 */
export async function installLocalDopplerRoutes(page, opts) {
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
    if (pathname.startsWith('/doppler/')) {
      pathname = pathname.slice('/doppler'.length) || '/';
    }
    if (pathname === '/') {
      pathname = '/landing.html';
    } else if (pathname === '/d') {
      pathname = '/doppler/index.html';
    } else if (pathname.startsWith('/d/')) {
      pathname = '/doppler' + pathname.slice(2);
    }

    // Prefer JS/JSON from dist/src when available (matches serve.ts behavior).
    let resolvedPathname = pathname;
    if ((pathname.endsWith('.js') || pathname.endsWith('.json')) && !pathname.includes('node_modules')) {
      const assetPath = pathname.startsWith('/doppler/') ? pathname.slice(8) : pathname;
      const distRelative = assetPath.replace(/^dist\//, '');
      const distSrcPath = join(projectRoot, 'dist', 'src', distRelative);
      try {
        await stat(distSrcPath);
        resolvedPathname = `/__dist_src__/${distRelative}`;
      } catch {
        const distPath = join(projectRoot, 'dist', distRelative);
        try {
          await stat(distPath);
          resolvedPathname = `/__dist__/${distRelative}`;
        } catch {
          // fall through
        }
      }
    }

    // Resolve to filesystem path with traversal guard
    const safePath = resolvedPathname.replace(/^(\.\.[/\\])+/, '').replace(/\.\./g, '');

    /** @type {string} */
    let filePath;
    if (safePath.startsWith('/__dist_src__/')) {
      filePath = join(projectRoot, 'dist', 'src', safePath.slice('/__dist_src__/'.length));
    } else if (safePath.startsWith('/__dist__/')) {
      filePath = join(projectRoot, 'dist', safePath.slice('/__dist__/'.length));
    } else {
      filePath = join(projectRoot, safePath);
    }

    const resolved = resolve(filePath);
    const allowedRoot = safePath.startsWith('/__dist_src__/')
      ? resolve(join(projectRoot, 'dist', 'src'))
      : safePath.startsWith('/__dist__/')
        ? resolve(join(projectRoot, 'dist'))
        : resolve(projectRoot);
    if (!resolved.startsWith(allowedRoot)) {
      return route.fulfill({ status: 403, body: 'Forbidden' });
    }

    /** @type {any} */
    let fileStats;
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

    /** @type {Record<string, string>} */
    const headers = {
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
 * @returns {void}
 */
export function stopServer() {
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

/**
 * Check if Chrome is running with CDP enabled at the given endpoint
 * @param {string} endpoint
 * @returns {Promise<boolean>}
 */
async function isCDPAvailable(endpoint) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1000);
    const response = await fetch(`${endpoint}/json/version`, { signal: controller.signal });
    clearTimeout(timeout);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * @param {import('./types.js').CLIOptions} opts
 * @param {{ scope?: import('./utils.js').BrowserProfileScope; devtools?: boolean }} [options]
 * @returns {Promise<import('playwright').BrowserContext>}
 */
export async function createBrowserContext(opts, options = {}) {
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

  const dopplerRoot = resolve(__dirname, '../../..');
  const defaultDirName = options.scope === 'bench' ? '.benchmark-cache' : '.test-cache';
  const userDataDir = opts.profileDir
    ? resolve(dopplerRoot, opts.profileDir)
    : resolve(dopplerRoot, defaultDirName);

  const args = [
    '--enable-unsafe-webgpu',
    '--disable-crash-reporter',
    '--disable-crashpad',
  ];

  // For headless mode with real GPU (not SwiftShader), use --headless=new
  // See: https://developer.chrome.com/blog/supercharge-web-ai-testing
  if (opts.headless) {
    args.push('--headless=new');  // New headless mode (supports GPU)

    if (process.platform === 'darwin') {
      // macOS: Chrome uses Metal directly for WebGPU, not Vulkan
      // Just need --headless=new, Metal backend is automatic
      args.push('--use-angle=metal');  // Explicit Metal backend
    } else {
      // Linux/Windows: Use Vulkan backend
      args.push(
        '--enable-features=Vulkan',
        '--use-angle=vulkan',       // Use Vulkan backend for ANGLE
        '--disable-vulkan-surface', // Required for headless (no display surface)
        '--no-sandbox',             // Often needed in containerized environments
      );
    }
  } else {
    // Headed mode: enable Vulkan features for better performance
    args.push('--enable-features=Vulkan');
  }

  if (!opts.headless && options.devtools) {
    args.push('--auto-open-devtools-for-tabs');
  }
  // Position window off-screen to avoid focus stealing in headed mode
  // Use --minimized flag in CLI to enable this behavior
  if (!opts.headless && opts.minimized) {
    args.push('--window-position=-2000,-2000');
  }

  const { chromium } = await import('playwright');

  // Try to connect to existing Chrome via CDP to avoid focus stealing
  if (opts.reuseBrowser && !opts.headless) {
    const cdpAvailable = await isCDPAvailable(opts.cdpEndpoint);
    if (cdpAvailable) {
      try {
        console.log(`Connecting to existing Chrome at ${opts.cdpEndpoint}...`);
        const browser = await chromium.connectOverCDP(opts.cdpEndpoint);
        const contexts = browser.contexts();
        const context = contexts[0] || await browser.newContext();
        context.setDefaultTimeout(opts.timeout);
        console.log('Connected to existing browser (no focus steal)');
        return context;
      } catch (err) {
        console.log(`CDP connection failed: ${/** @type {Error} */ (err).message}`);
        console.log('Falling back to launching new browser...');
      }
    } else if (opts.verbose) {
      console.log(`No Chrome with CDP at ${opts.cdpEndpoint}, launching new browser...`);
      console.log('TIP: To avoid focus stealing, start Chrome with:');
      console.log('  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222');
    }
  }

  // Fall back to launching persistent context
  // Note: When using --headless=new in args for GPU support, set headless: false
  // to prevent Playwright from adding its own (old) headless flag
  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: false,  // We handle headless via args (--headless=new for GPU support)
    devtools: Boolean(!opts.headless && options.devtools),
    args,
  });

  // Set default timeout from CLI options (default 120s, vs Playwright's 30s)
  context.setDefaultTimeout(opts.timeout);

  return context;
}

/**
 * @param {import('playwright').BrowserContext} context
 * @param {import('./types.js').CLIOptions} opts
 * @returns {Promise<import('playwright').Page>}
 */
export async function setupPage(context, opts) {
  const page = context.pages()[0] || await context.newPage();

  // Console logging
  const relevantTags = ['[Test]', '[Benchmark]', '[GPU]', 'ERROR', 'PASS', 'FAIL', 'Failed', 'error', 'WebGPU'];
  page.on('console', (msg) => {
    const text = msg.text();
    const isRelevant = relevantTags.some((tag) => text.includes(tag));
    const isError = /error|fail/i.test(text);
    if (opts.quiet) {
      if (isError) {
        console.log(`  [browser] ${text}`);
      }
      return;
    }
    if (opts.verbose || isRelevant) {
      console.log(`  [browser] ${text}`);
    }
  });

  page.on('pageerror', (err) => {
    console.error(`  [browser error] ${err.message}`);
  });

  // Log all network failures
  page.on('requestfailed', (req) => {
    if (!opts.quiet) {
      console.log(`  [network 404] ${req.url()}`);
    }
  });

  if (opts.noServer) {
    await installLocalDopplerRoutes(page, opts);
  }

  return page;
}

// ============================================================================
// Result Utilities
// ============================================================================

/**
 * @param {any} result
 * @returns {string}
 */
export function generateResultFilename(result) {
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
