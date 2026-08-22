#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_PACK = 'reports/pack-v0/gemma-3-270m-it-q4k-ehf16-af32/model.pack.json';
const DEFAULT_REFERENCE = 'reports/gemma-3-270m-it-q4k-ehf16-af32/2026-08-22T18-14-30.244Z.json';
const DEFAULT_OUT = 'reports/pack-v0/gemma-3-270m-it-q4k-ehf16-af32/browser-qualification.json';

function parseArgs(argv) {
  const options = { pack: DEFAULT_PACK, reference: DEFAULT_REFERENCE, out: DEFAULT_OUT, channel: 'chrome', maxTokens: null };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!['--pack', '--reference', '--out', '--channel', '--max-tokens'].includes(token)) throw new Error(`Unknown argument "${token}".`);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${token}.`);
    if (token === '--max-tokens') {
      const maxTokens = Number(value);
      if (!Number.isInteger(maxTokens) || maxTokens < 1) throw new Error('--max-tokens must be a positive integer.');
      options.maxTokens = maxTokens;
    } else {
      options[token.slice(2)] = value;
    }
    index += 1;
  }
  return options;
}

function browserArgs() {
  const common = [
    '--enable-unsafe-webgpu',
    '--enable-webgpu-developer-features',
    '--disable-dawn-features=disallow_unsafe_apis',
    '--ignore-gpu-blocklist',
  ];
  if (process.platform === 'darwin') return [...common, '--use-angle=metal'];
  if (process.platform === 'linux') return [...common, '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'];
  return common;
}

function isSoftwareAdapter(profile) {
  if (profile?.isFallbackAdapter === true) return true;
  const identity = [profile?.vendor, profile?.architecture, profile?.device, profile?.description]
    .map((value) => String(value ?? '').toLowerCase()).join(' ');
  return ['swiftshader', 'llvmpipe', 'software rasterizer'].some((marker) => identity.includes(marker));
}

async function launchPhysicalBrowser(channel) {
  const options = { headless: true, args: browserArgs(), timeout: 180_000 };
  try {
    return await chromium.launch({ ...options, channel });
  } catch (firstError) {
    if (channel !== 'chrome') throw firstError;
    return chromium.launch(options);
  }
}

export async function qualifyPackV0Browser(options = parseArgs(process.argv.slice(2))) {
  const server = await createStaticFileServer({ rootDir: REPO_ROOT, port: 0 });
  const browser = await launchPhysicalBrowser(options.channel);
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(180_000);
    page.on('console', (message) => console.error(`[browser:${message.type()}] ${message.text()}`));
    page.on('pageerror', (error) => console.error(`[browser:pageerror] ${error?.stack || error?.message || String(error)}`));
    const url = new URL('/tests/pack/browser-pack-qualification.html', server.baseUrl).href;
    await page.goto(url, { waitUntil: 'load' });
    const result = await page.evaluate(async ({ packPath, referencePath, maxTokens }) => {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('Physical browser qualification could not acquire a WebGPU adapter.');
      const adapterInfo = adapter.info ?? await adapter.requestAdapterInfo?.() ?? {};
      const [{ openPack }, { PACK_V0_TRUSTED_SIGNERS }] = await Promise.all([
        import('/src/index-browser.js'),
        import('/src/config/pack-v0-trusted-signers.js'),
      ]);
      const referenceResponse = await fetch(`/${referencePath}`);
      if (!referenceResponse.ok) throw new Error(`Reference fetch failed: ${referenceResponse.status}`);
      const reference = await referenceResponse.json();
      const expected = reference.metrics.referenceTranscript.tokens.ids;
      const generationConfig = reference.metrics.referenceTranscript.generationConfig;
      const session = await openPack(new URL(`/${packPath}`, location.href).href, {
        trustedSigners: PACK_V0_TRUSTED_SIGNERS,
      });
      const digestBefore = session.selectedTargetPlanDigest;
      try {
        const generated = await session.generateText({
          ...generationConfig,
          maxTokens: maxTokens ?? generationConfig.maxTokens,
          prompt: reference.metrics.prompt,
          maxSeqLen: 4096,
          stopSequences: [],
        });
        const firstMismatch = generated.tokenIds.findIndex((tokenId, index) => tokenId !== expected[index]);
        return {
          schema: 'doppler.pack-v0-browser-qualification/v1',
          passed: generated.tokenIds.length === (maxTokens ?? expected.length)
            && firstMismatch === -1,
          generatedTokens: generated.tokenIds.length,
          expectedTokens: expected.length,
          firstMismatch,
          packId: session.packId,
          semanticRoot: session.semanticRoot,
          targetId: session.selectedTargetId,
          targetPlanDigestBefore: digestBefore,
          targetPlanDigestAfter: session.selectedTargetPlanDigest,
          adapter: {
            vendor: adapterInfo.vendor ?? null,
            architecture: adapterInfo.architecture ?? null,
            device: adapterInfo.device ?? null,
            description: adapterInfo.description ?? null,
            isFallbackAdapter: adapter.isFallbackAdapter ?? adapterInfo.isFallbackAdapter ?? null,
          },
        };
      } finally {
        await session.close();
      }
    }, { packPath: options.pack, referencePath: options.reference, maxTokens: options.maxTokens });
    if (isSoftwareAdapter(result.adapter)) throw new Error(`Browser qualification selected a software adapter: ${JSON.stringify(result.adapter)}`);
    if (!result.passed) throw new Error(`Browser Pack parity failed at token ${result.firstMismatch}.`);
    const receipt = { ...result, surface: 'browser-webgpu', capturedAtUtc: new Date().toISOString() };
    const outputPath = path.resolve(REPO_ROOT, options.out);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
    console.log(JSON.stringify({ ...receipt, outputPath: path.relative(REPO_ROOT, outputPath) }, null, 2));
    return receipt;
  } finally {
    await browser.close();
    await server.close();
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  qualifyPackV0Browser().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
