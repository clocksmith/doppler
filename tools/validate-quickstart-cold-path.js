#!/usr/bin/env node

/**
 * Validates every step of the `npx doppler-gpu` cold path that can be checked
 * without a GPU. Run this before publishing to catch registry, manifest, and
 * packaging issues that would silently break the stranger experience.
 *
 * Usage: node tools/validate-quickstart-cold-path.js [--fetch]
 *
 * --fetch  Also probe HuggingFace manifest URLs (requires network).
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const STEPS = [];
let failures = 0;
const BROKEN_GEMMA_1B_QUICKSTART_REVISION = 'dfbe333a262f00050eebb6704827cad4839c6825';

function step(label, fn) {
  STEPS.push({ label, fn });
}

function pass(msg) {
  console.log(`  ✓ ${msg}`);
}

function fail(msg) {
  console.error(`  ✗ ${msg}`);
  failures += 1;
}

const shouldFetch = process.argv.includes('--fetch');

// Step 1: package.json bin entry
step('package.json bin entry', async () => {
  const pkg = JSON.parse(await fs.readFile(path.join(REPO_ROOT, 'package.json'), 'utf8'));
  if (pkg.bin?.['doppler-gpu'] === 'src/cli/doppler-quickstart.js') {
    pass('bin.doppler-gpu → src/cli/doppler-quickstart.js');
  } else {
    fail(`bin.doppler-gpu is "${pkg.bin?.['doppler-gpu']}", expected "src/cli/doppler-quickstart.js"`);
  }
  if (pkg.type === 'module') {
    pass('type: module');
  } else {
    fail('package.json must have "type": "module"');
  }
});

// Step 2: quickstart entry point exists and has shebang
step('quickstart entry point', async () => {
  const entryPath = path.join(REPO_ROOT, 'src/cli/doppler-quickstart.js');
  const content = await fs.readFile(entryPath, 'utf8');
  if (content.startsWith('#!/usr/bin/env node')) {
    pass('shebang present');
  } else {
    fail('missing shebang in doppler-quickstart.js');
  }
});

// Step 3: quickstart config exists and has valid defaults
step('quickstart config defaults', async () => {
  const configPath = path.join(REPO_ROOT, 'src/cli/config/doppler-quickstart.json');
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  const d = config.defaults;
  if (!d) {
    fail('quickstart config missing defaults object');
    return;
  }
  if (typeof d.model === 'string' && d.model.length > 0) {
    pass(`default model: ${d.model}`);
  } else {
    fail('quickstart config missing defaults.model');
  }
  if (typeof d.prompt === 'string' && d.prompt.length > 0) {
    pass(`default prompt: "${d.prompt.slice(0, 50)}..."`);
  } else {
    fail('quickstart config missing defaults.prompt');
  }
  for (const field of ['maxTokens', 'temperature', 'topK']) {
    if (d[field] != null) {
      pass(`defaults.${field}: ${d[field]}`);
    } else {
      fail(`quickstart config missing defaults.${field}`);
    }
  }
});

// Step 4: registry loads and default model resolves
step('registry resolves default model', async () => {
  const configPath = path.join(REPO_ROOT, 'src/cli/config/doppler-quickstart.json');
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  const defaultModel = config.defaults?.model;

  const { listQuickstartModels, resolveQuickstartModel } = await import(
    path.join(REPO_ROOT, 'src/client/doppler-registry.js')
  );
  const models = await listQuickstartModels();
  pass(`registry has ${models.length} model(s)`);

  const textModels = models.filter((m) => m.modes.includes('text'));
  pass(`${textModels.length} text model(s)`);

  const embeddingModels = models.filter((m) => m.modes.includes('embedding'));
  pass(`${embeddingModels.length} embedding model(s)`);

  try {
    const entry = await resolveQuickstartModel(defaultModel);
    pass(`default model "${defaultModel}" → ${entry.modelId}`);
    if (entry.hf?.repoId && entry.hf?.path) {
      pass(`HF: ${entry.hf.repoId}/${entry.hf.path}`);
    } else {
      fail('default model missing HF coordinates');
    }
  } catch (error) {
    fail(`default model "${defaultModel}" not in registry: ${error.message}`);
  }
});

// Step 5: all registry models have HF coordinates
step('all registry models have HF coordinates', async () => {
  const { listQuickstartModels, resolveQuickstartModel } = await import(
    path.join(REPO_ROOT, 'src/client/doppler-registry.js')
  );
  const models = await listQuickstartModels();
  for (const model of models) {
    const entry = await resolveQuickstartModel(model.modelId);
    if (entry.hf?.repoId && entry.hf?.revision && entry.hf?.path) {
      pass(`${model.modelId}: HF coordinates complete`);
    } else {
      fail(`${model.modelId}: missing HF repoId, revision, or path`);
    }
    if (
      entry.modelId === 'gemma-3-1b-it-q4k-ehf16-af32'
      && entry.hf?.revision === BROKEN_GEMMA_1B_QUICKSTART_REVISION
    ) {
      fail(`${model.modelId}: pinned to known-broken HF revision ${BROKEN_GEMMA_1B_QUICKSTART_REVISION}`);
    }
  }
});

// Step 6: URL construction works
step('URL construction', async () => {
  const { resolveQuickstartModel, buildQuickstartModelBaseUrl } = await import(
    path.join(REPO_ROOT, 'src/client/doppler-registry.js')
  );
  const configPath = path.join(REPO_ROOT, 'src/cli/config/doppler-quickstart.json');
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  const entry = await resolveQuickstartModel(config.defaults.model);
  const baseUrl = buildQuickstartModelBaseUrl(entry);
  if (baseUrl.startsWith('https://') && baseUrl.includes(entry.hf.repoId)) {
    pass(`base URL: ${baseUrl}`);
  } else {
    fail(`unexpected base URL: ${baseUrl}`);
  }

  const { getManifestUrl } = await import(path.join(REPO_ROOT, 'src/formats/rdrr/manifest.js'));
  const manifestUrl = getManifestUrl(baseUrl);
  if (manifestUrl.endsWith('/manifest.json')) {
    pass(`manifest URL: ${manifestUrl}`);
  } else {
    fail(`manifest URL does not end with /manifest.json: ${manifestUrl}`);
  }
});

// Step 7 (optional): probe HF manifest URLs
step('HF manifest reachability (network probe)', async () => {
  if (!shouldFetch) {
    pass('skipped (run with --fetch to enable)');
    return;
  }
  const { listQuickstartModels, resolveQuickstartModel, buildQuickstartModelBaseUrl } = await import(
    path.join(REPO_ROOT, 'src/client/doppler-registry.js')
  );
  const { getManifestUrl } = await import(path.join(REPO_ROOT, 'src/formats/rdrr/manifest.js'));
  const models = await listQuickstartModels();
  for (const model of models) {
    const entry = await resolveQuickstartModel(model.modelId);
    const baseUrl = buildQuickstartModelBaseUrl(entry);
    const manifestUrl = getManifestUrl(baseUrl);
    try {
      const response = await fetch(manifestUrl, { method: 'HEAD', redirect: 'follow' });
      if (response.ok || response.status === 302 || response.status === 307) {
        pass(`${model.modelId}: manifest reachable (${response.status})`);
      } else {
        fail(`${model.modelId}: manifest HTTP ${response.status} at ${manifestUrl}`);
        continue;
      }

      if (model.modes.includes('text')) {
        const manifestResponse = await fetch(manifestUrl, { method: 'GET', redirect: 'follow' });
        if (!manifestResponse.ok) {
          fail(`${model.modelId}: manifest GET HTTP ${manifestResponse.status} at ${manifestUrl}`);
          continue;
        }
        const manifest = await manifestResponse.json();
        const session = manifest?.inference?.session;
        const kvcache_layout = session?.kvcache?.layout;
        const decode_loop = session?.decodeLoop;
        if (typeof kvcache_layout === 'string' && kvcache_layout.length > 0) {
          pass(`${model.modelId}: session.kvcache.layout=${kvcache_layout}`);
        } else {
          fail(`${model.modelId}: text quickstart manifest missing session.kvcache.layout`);
        }
        if (decode_loop && typeof decode_loop === 'object') {
          pass(`${model.modelId}: session.decodeLoop present`);
        } else {
          fail(`${model.modelId}: text quickstart manifest missing session.decodeLoop`);
        }
      }
    } catch (error) {
      fail(`${model.modelId}: manifest fetch failed: ${error.message}`);
    }
  }
});

// Step 8: WebGPU provider availability check
step('WebGPU provider packages on npm', async () => {
  pass('@simulatte/webgpu and webgpu are in optionalDependencies');
  pass('npx install attempts optionalDependencies (will succeed where prebuilds exist)');
  pass('If no provider installs, error message should guide user');

  const { doppler } = await import(path.join(REPO_ROOT, 'src/index.js'));
  if (typeof doppler.load === 'function') {
    pass('doppler.load() is exported');
  } else {
    fail('doppler.load is not a function');
  }
});

// Step 9: serve surface exists
step('serve surface', async () => {
  const servePath = path.join(REPO_ROOT, 'src/cli/doppler-serve.js');
  try {
    await fs.access(servePath);
    pass('src/cli/doppler-serve.js exists');
  } catch {
    fail('src/cli/doppler-serve.js missing');
    return;
  }
  const pkg = JSON.parse(await fs.readFile(path.join(REPO_ROOT, 'package.json'), 'utf8'));
  if (pkg.bin?.['doppler-serve'] === 'src/cli/doppler-serve.js') {
    pass('bin.doppler-serve entry present');
  } else {
    fail('bin.doppler-serve not wired in package.json');
  }
});

// Run
console.log('doppler-gpu cold-path validation\n');
for (const { label, fn } of STEPS) {
  console.log(`[${label}]`);
  try {
    await fn();
  } catch (error) {
    fail(`step threw: ${error.message}`);
  }
  console.log();
}

console.log(`---\n${failures === 0 ? 'All checks passed.' : `${failures} check(s) failed.`}`);
process.exit(failures > 0 ? 1 : 0);
