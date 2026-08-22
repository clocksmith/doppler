#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { openPack } from '../src/index.js';
import { destroyDevice } from '../src/gpu/device.js';
import { releaseNodeWebGPU } from '../src/tooling/node-webgpu.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_PACK = 'reports/pack-v0/gemma-3-270m-it-q4k-ehf16-af32/model.pack.json';
const DEFAULT_REFERENCE = 'reports/gemma-3-270m-it-q4k-ehf16-af32/2026-08-22T18-14-30.244Z.json';
const DEFAULT_OUT = 'reports/pack-v0/gemma-3-270m-it-q4k-ehf16-af32/node-qualification.json';

function parseArgs(argv) {
  const options = { pack: DEFAULT_PACK, reference: DEFAULT_REFERENCE, out: DEFAULT_OUT };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!['--pack', '--reference', '--out'].includes(token)) throw new Error(`Unknown argument "${token}".`);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${token}.`);
    options[token.slice(2)] = value;
    index += 1;
  }
  return options;
}

function isSoftwareAdapter(profile) {
  const adapter = profile?.adapter ?? {};
  const identity = [adapter.vendor, adapter.architecture, adapter.device, adapter.description]
    .map((value) => String(value ?? '').toLowerCase()).join(' ');
  return ['swiftshader', 'llvmpipe', 'software rasterizer'].some((marker) => identity.includes(marker));
}

export async function qualifyPackV0Node(options = parseArgs(process.argv.slice(2))) {
  const packPath = path.resolve(REPO_ROOT, options.pack);
  const reference = JSON.parse(await fs.readFile(path.resolve(REPO_ROOT, options.reference), 'utf8'));
  const expected = reference.metrics.referenceTranscript.tokens.ids;
  const generationConfig = reference.metrics.referenceTranscript.generationConfig;
  const session = await openPack(packPath);
  const digestBefore = session.selectedTargetPlanDigest;
  try {
    if (session.deviceProfile.surface !== 'node-webgpu') {
      throw new Error(`Node qualification selected unexpected surface "${session.deviceProfile.surface}".`);
    }
    if (isSoftwareAdapter(session.deviceProfile)) {
      throw new Error(`Node qualification selected a software adapter: ${JSON.stringify(session.deviceProfile.adapter)}`);
    }
    const generated = await session.generateText({
      ...generationConfig,
      prompt: reference.metrics.prompt,
      maxSeqLen: 4096,
      stopSequences: [],
    });
    const firstMismatch = generated.tokenIds.findIndex((tokenId, index) => tokenId !== expected[index]);
    const receipt = {
      schema: 'doppler.pack-v0-node-qualification/v1',
      passed: generated.tokenIds.length === expected.length && firstMismatch === -1,
      generatedTokens: generated.tokenIds.length,
      expectedTokens: expected.length,
      firstMismatch,
      packId: session.packId,
      semanticRoot: session.semanticRoot,
      targetId: session.selectedTargetId,
      targetPlanDigestBefore: digestBefore,
      targetPlanDigestAfter: session.selectedTargetPlanDigest,
      adapter: session.deviceProfile.adapter,
      surface: session.deviceProfile.surface,
      capturedAtUtc: new Date().toISOString(),
    };
    if (!receipt.passed) throw new Error(`Node Pack parity failed at token ${receipt.firstMismatch}.`);
    const outputPath = path.resolve(REPO_ROOT, options.out);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
    console.log(JSON.stringify({ ...receipt, outputPath: path.relative(REPO_ROOT, outputPath) }, null, 2));
    return receipt;
  } finally {
    await session.close();
    try {
      destroyDevice();
    } finally {
      await releaseNodeWebGPU();
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  qualifyPackV0Node().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
