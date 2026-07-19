#!/usr/bin/env node

import { createReadStream } from 'node:fs';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

import { initializeInference } from '../src/inference/test-harness.js';
import {
  bootstrapNodeWebGPU,
  releaseNodeWebGPU,
} from '../src/tooling/node-webgpu.js';
import {
  destroyDevice,
  getDevice,
  resetDeviceState,
} from '../src/gpu/device.js';
import { destroyBufferPool } from '../src/memory/buffer-pool.js';
import { sha256BytesHex } from '../src/utils/sha256.js';
import {
  getActiveLoRAForPipeline,
  loadLoRAAdapterForPipeline,
  unloadLoRAAdapterForPipeline,
} from '../src/client/runtime/lora.js';
import {
  createSyntheticSequenceLoRAManifest,
  evaluateSequenceLoRAQualification,
  evaluateSequenceReference,
  validateSequenceReference,
} from './lib/sequence-model-qualification.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_REFERENCE = resolve(ROOT, 'tools/data/amplify-120m-sequence-reference.json');

function readFlag(argv, index) {
  const value = argv[index + 1];
  if (!value || value.startsWith('--')) throw new Error(`${argv[index]} requires a value.`);
  return value;
}

export function parseArgs(argv) {
  const args = {
    modelDir: null,
    modelUrl: null,
    reference: DEFAULT_REFERENCE,
    output: null,
    diagnoseLayers: [],
    diagnoseOps: [],
    qualifyLoRA: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--model-dir') args.modelDir = resolve(readFlag(argv, index++));
    else if (arg === '--model-url') args.modelUrl = readFlag(argv, index++).replace(/\/+$/u, '');
    else if (arg === '--reference') args.reference = resolve(readFlag(argv, index++));
    else if (arg === '--output') args.output = resolve(readFlag(argv, index++));
    else if (arg === '--diagnose-layer') {
      const layer = Number(readFlag(argv, index++));
      if (!Number.isInteger(layer) || layer < 0) {
        throw new Error('--diagnose-layer requires a non-negative integer.');
      }
      args.diagnoseLayers.push(layer);
    }
    else if (arg === '--diagnose-op') {
      const opId = readFlag(argv, index++).trim();
      if (!opId) throw new Error('--diagnose-op requires a non-empty operator id.');
      args.diagnoseOps.push(opId);
    }
    else if (arg === '--qualify-lora') args.qualifyLoRA = true;
    else throw new Error(`Unknown argument "${arg}".`);
  }
  if (Boolean(args.modelDir) === Boolean(args.modelUrl)) {
    throw new Error('Exactly one of --model-dir or --model-url is required.');
  }
  if (args.modelUrl && !/^https?:\/\//u.test(args.modelUrl)) {
    throw new Error('--model-url must be an HTTP(S) URL.');
  }
  return args;
}

function contentType(path) {
  if (extname(path) === '.json') return 'application/json';
  if (extname(path) === '.wgsl') return 'text/plain; charset=utf-8';
  return 'application/octet-stream';
}

function parseByteRange(value, size) {
  const match = /^bytes=(\d+)-(\d*)$/u.exec(String(value ?? ''));
  if (!match) return null;
  const start = Number(match[1]);
  const requestedEnd = match[2] ? Number(match[2]) : size - 1;
  const end = Math.min(requestedEnd, size - 1);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start > end) return null;
  return { start, end };
}

async function createModelServer(modelDir) {
  const server = createServer(async (request, response) => {
    try {
      const relative = decodeURIComponent(new URL(request.url, 'http://localhost').pathname)
        .replace(/^\/+model\/?/u, '');
      const path = resolve(modelDir, relative);
      if (path !== modelDir && !path.startsWith(`${modelDir}${sep}`)) throw new Error('Path escapes model root.');
      const file = await stat(path);
      if (!file.isFile()) throw new Error('Artifact path is not a file.');
      const range = parseByteRange(request.headers.range, file.size);
      const headers = {
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-store',
        'Content-Type': contentType(path),
      };
      if (range) {
        response.writeHead(206, {
          ...headers,
          'Content-Length': range.end - range.start + 1,
          'Content-Range': `bytes ${range.start}-${range.end}/${file.size}`,
        });
        createReadStream(path, range).pipe(response);
        return;
      }
      response.writeHead(200, { ...headers, 'Content-Length': file.size });
      if (request.method === 'HEAD') response.end();
      else createReadStream(path).pipe(response);
    } catch (error) {
      response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      response.end(error?.message ?? String(error));
    }
  });
  await new Promise((accept, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', accept);
  });
  return server;
}

async function closeModelServer(server) {
  await new Promise((accept, reject) => {
    server.close((error) => (error ? reject(error) : accept()));
    server.closeAllConnections?.();
  });
}

async function readModelManifest(args) {
  if (args.modelDir) {
    return JSON.parse(await readFile(resolve(args.modelDir, 'manifest.json'), 'utf8'));
  }
  const manifestUrl = `${args.modelUrl}/manifest.json`;
  const response = await fetch(manifestUrl, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to load hosted manifest (${response.status}) from ${manifestUrl}.`);
  }
  return response.json();
}

function gitValue(args) {
  try {
    return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

function repositoryPath(path) {
  const candidate = relative(ROOT, path);
  if (!candidate.startsWith(`..${sep}`) && candidate !== '..') {
    return candidate.split(sep).join('/');
  }
  return path;
}

function summarizeCapabilities(capabilities) {
  return {
    hasF16: capabilities?.hasF16 === true,
    hasSubgroups: capabilities?.hasSubgroups === true,
    hasSubgroupsF16: capabilities?.hasSubgroupsF16 === true,
    maxBufferSize: capabilities?.maxBufferSize ?? null,
  };
}

function errorMessage(error) {
  return error?.message ?? String(error);
}

async function encodeQualificationSequence(pipeline, reference, options = {}) {
  return pipeline.encodeSequence(reference.input.sequence, {
    includeTokenEmbeddings: true,
    includeLogits: reference.outputs?.logits !== false,
    ...options,
  });
}

async function qualifySequenceLoRA(pipeline, manifest, reference, baseResult) {
  const adapterManifest = createSyntheticSequenceLoRAManifest(manifest);
  let wrongBaseError = null;
  let invalidLayerError = null;
  try {
    await loadLoRAAdapterForPipeline(pipeline, {
      ...adapterManifest,
      baseModel: `${manifest.modelId}-wrong-base`,
    });
  } catch (error) {
    wrongBaseError = errorMessage(error);
  }
  try {
    await loadLoRAAdapterForPipeline(
      pipeline,
      createSyntheticSequenceLoRAManifest(manifest, {
        layerIndex: Number(manifest.architecture?.numLayers),
      })
    );
  } catch (error) {
    invalidLayerError = errorMessage(error);
  }
  await loadLoRAAdapterForPipeline(pipeline, adapterManifest);
  const activeAdapterName = getActiveLoRAForPipeline(pipeline);
  const adaptedResult = await encodeQualificationSequence(pipeline, reference);
  await unloadLoRAAdapterForPipeline(pipeline);
  const unloadedAdapterName = getActiveLoRAForPipeline(pipeline);
  const restoredResult = await encodeQualificationSequence(pipeline, reference);
  const evaluation = evaluateSequenceLoRAQualification({
    baseResult,
    adaptedResult,
    restoredResult,
    expectedAdapterName: adapterManifest.name,
    activeAdapterName,
    unloadedAdapterName,
    wrongBaseError,
    invalidLayerError,
  });
  return {
    ...evaluation,
    adapter: {
      id: adapterManifest.id,
      name: adapterManifest.name,
      baseModel: adapterManifest.baseModel,
      rank: adapterManifest.rank,
      alpha: adapterManifest.alpha,
      targetModules: adapterManifest.targetModules,
      layerIndices: [0],
      synthetic: true,
    },
    claimBoundary: 'Synthetic rank-1 q_proj qualification proves exact base binding, target validation, WebGPU activation, output influence, unload, and base-output restoration. It does not prove trained-adapter quality or arbitrary PEFT compatibility.',
  };
}

export async function qualifySequenceModel(args) {
  const referenceBytes = await readFile(args.reference);
  const reference = validateSequenceReference(JSON.parse(referenceBytes.toString('utf8')));
  const manifest = await readModelManifest(args);
  if (manifest.modelId !== reference.modelId) {
    throw new Error(`Reference modelId "${reference.modelId}" does not match manifest "${manifest.modelId}".`);
  }

  const webgpuBootstrap = await bootstrapNodeWebGPU();
  const releaseBootstrappedProvider = webgpuBootstrap.provider !== 'pre-installed';
  const server = args.modelDir ? await createModelServer(args.modelDir) : null;
  let pipeline = null;
  try {
    const baseUrl = server
      ? `http://127.0.0.1:${server.address().port}/model`
      : args.modelUrl;
    const harness = await initializeInference(baseUrl, {
      modelId: reference.modelId,
      loadMode: 'http',
      log: () => {},
      runtime: {
        runtimeConfig: {
          inference: {
            session: manifest.inference?.session ?? {},
          },
        },
      },
    });
    pipeline = harness.pipeline;
    const includeLogits = reference.outputs?.logits !== false;
    const result = await encodeQualificationSequence(pipeline, reference, {
      diagnostics: args.diagnoseLayers.length > 0 || args.diagnoseOps.length > 0 ? {
        enabled: true,
        captureConfig: {
          defaultLevel: 'metadata',
          targetLayers: args.diagnoseLayers,
          targetOpIds: args.diagnoseOps,
          targetLevel: 'full',
        },
      } : undefined,
    });
    const evaluation = evaluateSequenceReference({ manifest: harness.manifest, result, reference });
    const loraQualification = args.qualifyLoRA
      ? await qualifySequenceLoRA(pipeline, manifest, reference, result)
      : null;
    const pipelineStats = pipeline.getStats?.() ?? null;
    const device = getDevice();
    const receipt = {
      schema: 'doppler.sequenceModelQualification.v1',
      passed: evaluation.passed && (loraQualification?.passed ?? true),
      generatedAt: new Date().toISOString(),
      model: {
        modelId: manifest.modelId,
        artifactIdentity: manifest.artifactIdentity ?? null,
        quantization: manifest.quantization ?? null,
        architecture: manifest.architecture ?? null,
        sequence: manifest.inference?.sequence ?? null,
        session: manifest.inference?.session ?? null,
        execution: {
          schema: manifest.inference?.schema ?? null,
          kernelPathId: pipeline.resolvedKernelPath?.id ?? null,
          kernels: manifest.inference?.execution?.kernels ?? null,
        },
      },
      reference: {
        path: repositoryPath(args.reference),
        digest: `sha256:${sha256BytesHex(new Uint8Array(
          referenceBytes.buffer,
          referenceBytes.byteOffset,
          referenceBytes.byteLength
        ))}`,
        source: reference.source,
        input: reference.input,
        tolerances: reference.tolerances,
      },
      runtime: {
        surface: 'node-webgpu',
        artifactSource: {
          kind: args.modelDir ? 'local-directory' : 'hosted-url',
          baseUrl,
        },
        capabilities: summarizeCapabilities(harness.capabilities),
        adapterInfo: device?.adapterInfo ?? null,
        sourceRevision: gitValue(['rev-parse', 'HEAD']),
        sourceDirty: Boolean(gitValue(['status', '--short'])),
      },
      result: {
        checks: evaluation.checks,
        outputDigests: evaluation.outputDigests,
        loraQualification,
        phase: result.phase ?? null,
        operatorDiagnostics: args.diagnoseLayers.length > 0 || args.diagnoseOps.length > 0
          ? pipelineStats?.operatorDiagnostics ?? null
          : null,
      },
    };
    if (args.output) {
      await mkdir(dirname(args.output), { recursive: true });
      await writeFile(args.output, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
    }
    return receipt;
  } finally {
    try {
      if (pipeline?.unload) await pipeline.unload();
    } finally {
      try {
        if (server) await closeModelServer(server);
      } finally {
        try {
          destroyBufferPool();
        } finally {
          try {
            destroyDevice();
          } finally {
            try {
              resetDeviceState();
            } finally {
              if (releaseBootstrappedProvider) {
                releaseNodeWebGPU();
              }
            }
          }
        }
      }
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const receipt = await qualifySequenceModel(args);
  console.log(JSON.stringify(receipt, null, 2));
  if (!receipt.passed) process.exitCode = 1;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack ?? error);
    process.exitCode = 1;
  });
}
