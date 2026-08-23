#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { openPack } from '../src/index.js';
import { destroyDevice } from '../src/gpu/device.js';
import { releaseNodeWebGPU } from '../src/tooling/node-webgpu.js';
import { PACK_V0_TRUSTED_SIGNERS } from '../src/config/pack-v0-trusted-signers.js';

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

export function isSoftwareAdapter(profile) {
  const adapter = profile?.adapter ?? {};
  const identity = [adapter.vendor, adapter.architecture, adapter.device, adapter.description]
    .map((value) => String(value ?? '').toLowerCase()).join(' ');
  return ['swiftshader', 'llvmpipe', 'software rasterizer'].some((marker) => identity.includes(marker));
}

export function createPackNodeQualificationReceipt({
  pack,
  session,
  generatedTokenIds,
  expectedTokenIds,
  targetPlanDigestBefore,
  capturedAtUtc,
}) {
  const firstMismatch = generatedTokenIds.findIndex((tokenId, index) => tokenId !== expectedTokenIds[index]);
  const declaredIdentity = session.selectedPlan?.initialExecutionIdentity ?? null;
  const observedIdentity = session.observedInitialExecutionIdentity ?? null;
  if (!declaredIdentity?.digest || !observedIdentity?.digest) {
    throw new Error('Node Pack qualification requires declared and observed initial execution identities.');
  }
  if (declaredIdentity.digest !== observedIdentity.digest) {
    throw new Error(
      `Node Pack initial execution identity mismatch: declared ${declaredIdentity.digest}, ` +
      `observed ${observedIdentity.digest}.`
    );
  }
  const digestAfter = session.selectedTargetPlanDigest;
  return {
    schema: 'doppler.pack-node-qualification/v2',
    passed: generatedTokenIds.length === expectedTokenIds.length
      && firstMismatch === -1
      && targetPlanDigestBefore === digestAfter,
    generatedTokens: generatedTokenIds.length,
    expectedTokens: expectedTokenIds.length,
    firstMismatch,
    packId: session.packId,
    semanticRoot: session.semanticRoot,
    signature: {
      authority: pack.signature?.authority ?? null,
      algorithm: pack.signature?.algorithm ?? null,
      publicKeyDigest: pack.signature?.publicKeyDigest ?? null,
      disposition: 'explicitly-trusted-for-qualification',
    },
    closure: {
      artifacts: pack.artifacts.length,
      verifiedArtifacts: session.verification?.artifactReceipts?.length ?? 0,
      wgslModules: pack.wgslModules.length,
      qualifiedEntryPoints: [...(pack.modelIR?.supportScope?.qualifiedEntryPoints ?? [])],
    },
    targetId: session.selectedTargetId,
    targetPlanDigestBefore,
    targetPlanDigestAfter: digestAfter,
    targetPlanImmutable: targetPlanDigestBefore === digestAfter,
    initialExecutionIdentity: {
      schema: observedIdentity.schema ?? null,
      declaredDigest: declaredIdentity.digest,
      observedDigest: observedIdentity.digest,
      boundBeforePrefill: true,
      executionGraphHash: observedIdentity.executionGraphHash,
      kernelClosureHash: observedIdentity.kernelClosureHash,
      runtimeEngineDigest: observedIdentity.runtimeEngineDigest,
      programLoadPolicyHash: observedIdentity.programLoadPolicyHash ?? null,
    },
    adapter: session.deviceProfile.adapter,
    surface: session.deviceProfile.surface,
    softwareAdapter: false,
    capturedAtUtc,
  };
}

export async function withNodePackQualificationLifecycle(openSession, runSession, lifecycle = {}) {
  const destroy = lifecycle.destroyDevice ?? destroyDevice;
  const release = lifecycle.releaseNodeWebGPU ?? releaseNodeWebGPU;
  let session = null;
  try {
    session = await openSession();
    return await runSession(session);
  } finally {
    try {
      await session?.close?.();
    } finally {
      try {
        destroy();
      } finally {
        await release();
      }
    }
  }
}

export async function qualifyPackV0Node(options = parseArgs(process.argv.slice(2))) {
  const packPath = path.resolve(REPO_ROOT, options.pack);
  const pack = JSON.parse(await fs.readFile(packPath, 'utf8'));
  const reference = JSON.parse(await fs.readFile(path.resolve(REPO_ROOT, options.reference), 'utf8'));
  const expected = reference.metrics.referenceTranscript.tokens.ids;
  const generationConfig = reference.metrics.referenceTranscript.generationConfig;
  return withNodePackQualificationLifecycle(
    () => openPack(packPath, { trustedSigners: PACK_V0_TRUSTED_SIGNERS }),
    async (session) => {
      const digestBefore = session.selectedTargetPlanDigest;
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
      const receipt = createPackNodeQualificationReceipt({
        pack,
        session,
        generatedTokenIds: generated.tokenIds,
        expectedTokenIds: expected,
        targetPlanDigestBefore: digestBefore,
        capturedAtUtc: new Date().toISOString(),
      });
      if (!receipt.passed) throw new Error(`Node Pack parity failed at token ${receipt.firstMismatch}.`);
      const outputPath = path.resolve(REPO_ROOT, options.out);
      await fs.mkdir(path.dirname(outputPath), { recursive: true });
      await fs.writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
      console.log(JSON.stringify({ ...receipt, outputPath: path.relative(REPO_ROOT, outputPath) }, null, 2));
      return receipt;
    }
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  qualifyPackV0Node().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
