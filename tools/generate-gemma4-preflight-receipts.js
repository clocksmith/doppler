#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { DTYPE_SIZES } from '../src/config/schema/index.js';
import { destroyDevice, getDevice, getKernelCapabilities, initDevice } from '../src/gpu/device.js';
import {
  expectsSplitGpuEmbeddingKernel,
  getEmbeddingFloatDtype,
  getMaxSplitGpuEmbeddingSectionsForDevice,
  getSplitGpuEmbeddingKernelSectionCount,
  getSplitGpuEmbeddingRequiredStorageBuffers,
  resolveManifestGpuResidentEmbeddingLimitError,
} from '../src/loader/embedding-limit-preflight.js';
import { getLargeWeightMaxBytes } from '../src/loader/manifest-config.js';
import { bootstrapNodeWebGPU } from '../src/tooling/node-webgpu.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const DEFAULT_OUTPUT_ROOT = path.join(REPO_ROOT, 'reports', 'gemma4-preflight');
const DEFAULT_TARGET_MATRIX_PATH = path.join(REPO_ROOT, 'models', 'gemma4-targets.json');
const DEFAULT_MANIFEST_PATHS = Object.freeze([
  'models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest.json',
  'models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/manifest.json',
  'models/local/gemma-4-e2b-it-q4k-ehf16-af16-int4ple/manifest.json',
]);

function usage() {
  return [
    'Usage:',
    '  node tools/generate-gemma4-preflight-receipts.js [--manifest <manifest.json> ...] [--generated-at <iso>] [--output-root <dir>]',
    '  node tools/generate-gemma4-preflight-receipts.js --check [--target-matrix <models/gemma4-targets.json>]',
    '',
    'Options:',
    '  --manifest <path>       Manifest to generate. Repeatable. Defaults to current Gemma 4 E2B local manifests.',
    '  --generated-at <iso>    Timestamp to stamp into written receipts.',
    '  --output-root <dir>     Receipt output root. Defaults to reports/gemma4-preflight.',
    '  --target-matrix <path>  Gemma 4 target matrix used by --check.',
    '  --check                 Rebuild listed preflight receipts and fail on drift.',
  ].join('\n');
}

export function parseArgs(argv) {
  const args = {
    check: false,
    generatedAt: null,
    manifestPaths: [],
    outputRoot: DEFAULT_OUTPUT_ROOT,
    targetMatrixPath: DEFAULT_TARGET_MATRIX_PATH,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    const nextValue = () => {
      const value = argv[i + 1];
      if (value == null || String(value).startsWith('--')) {
        throw new Error(`Missing value for ${token}`);
      }
      i += 1;
      return String(value).trim();
    };
    if (token === '--help' || token === '-h') {
      args.help = true;
      continue;
    }
    if (token === '--check') {
      args.check = true;
      continue;
    }
    if (token === '--generated-at') {
      args.generatedAt = normalizeIsoTimestamp(nextValue(), '--generated-at');
      continue;
    }
    if (token === '--manifest') {
      args.manifestPaths.push(resolveRepoPath(nextValue()));
      continue;
    }
    if (token === '--output-root') {
      args.outputRoot = path.resolve(REPO_ROOT, nextValue());
      continue;
    }
    if (token === '--target-matrix') {
      args.targetMatrixPath = path.resolve(REPO_ROOT, nextValue());
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  if (args.check && args.manifestPaths.length > 0) {
    throw new Error('--check reads preflight receipts from the target matrix; omit --manifest.');
  }
  return args;
}

function normalizeIsoTimestamp(value, flagName) {
  const candidate = typeof value === 'string' ? value.trim() : '';
  if (!candidate) {
    throw new Error(`${flagName} requires a non-empty ISO timestamp`);
  }
  const timestamp = new Date(candidate);
  if (!Number.isFinite(timestamp.getTime())) {
    throw new Error(`${flagName} must be a valid ISO timestamp`);
  }
  return timestamp.toISOString();
}

function resolveRepoPath(value) {
  const candidate = typeof value === 'string' ? value.trim() : '';
  if (!candidate) {
    throw new Error('Path value must be non-empty');
  }
  return path.isAbsolute(candidate) ? candidate : path.resolve(REPO_ROOT, candidate);
}

function repoRelativePath(filePath) {
  const relative = path.relative(REPO_ROOT, filePath).split(path.sep).join('/');
  if (!relative || relative.startsWith('../') || relative === '..') {
    throw new Error(`${filePath} is outside the repository`);
  }
  return relative;
}

function sha256Hex(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function normalizeDtype(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

function alignByteLength(byteLength) {
  return Math.ceil(byteLength / 4) * 4;
}

function resolveEmbeddingTensor(manifest) {
  const tensors = isObject(manifest?.tensors) ? manifest.tensors : null;
  const overrides = manifest?.inference?.largeWeights?.gpuResidentOverrides;
  if (!tensors || !Array.isArray(overrides)) {
    throw new Error(`${manifest?.modelId || 'unknown-model'}: manifest must include tensors and inference.largeWeights.gpuResidentOverrides`);
  }
  for (const tensorName of overrides) {
    const location = tensors[tensorName];
    if (location?.role === 'embedding') {
      return { tensorName, location };
    }
  }
  throw new Error(`${manifest?.modelId || 'unknown-model'}: gpuResidentOverrides does not include an embedding tensor`);
}

function resolveEmbeddingEstimate(location) {
  const shape = Array.isArray(location?.shape) ? location.shape : null;
  if (!shape || shape.length !== 2) {
    throw new Error('Embedding tensor requires a two-dimensional shape');
  }
  const [rows, hidden] = shape;
  if (!Number.isFinite(rows) || rows <= 0 || !Number.isFinite(hidden) || hidden <= 0) {
    throw new Error('Embedding tensor shape must contain positive finite row and hidden dimensions');
  }
  const floatDtype = getEmbeddingFloatDtype(location);
  const bytesPerElement = DTYPE_SIZES[normalizeDtype(floatDtype)];
  if (!Number.isFinite(bytesPerElement) || bytesPerElement <= 0) {
    throw new Error(`Unsupported embedding dtype: ${floatDtype}`);
  }
  const computedTensorSizeBytes = rows * hidden * bytesPerElement;
  const tensorSizeBytes = hasFiniteNumber(location?.size)
    ? location.size
    : alignByteLength(computedTensorSizeBytes);
  return {
    rows,
    hidden,
    bytesPerElement,
    tensorSizeBytes,
    rowBytes: hidden * bytesPerElement,
  };
}

function resolveAdapterInfo(capabilities) {
  const adapterInfo = capabilities?.adapterInfo;
  return {
    vendor: adapterInfo?.vendor || 'unknown',
    architecture: adapterInfo?.architecture || 'unknown',
    device: adapterInfo?.device || 'unknown',
    description: adapterInfo?.description || '',
  };
}

function resolveDeviceLimits(device) {
  const limits = device?.limits;
  if (!limits) {
    throw new Error('A WebGPU device with limits is required to build a preflight receipt');
  }
  return {
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxBufferSize: limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
  };
}

function buildPreflightMath({ location, embeddingKernel, device, largeWeightMaxBytes, limitError }) {
  const estimate = resolveEmbeddingEstimate(location);
  const rowsPerSplitSection = largeWeightMaxBytes
    ? Math.floor(largeWeightMaxBytes / estimate.rowBytes)
    : null;
  const requiredSplitSections = rowsPerSplitSection && rowsPerSplitSection > 0
    ? Math.ceil(estimate.rows / rowsPerSplitSection)
    : null;
  const activeSplitKernelMaxSections = getSplitGpuEmbeddingKernelSectionCount(embeddingKernel);
  const maxSplitEmbeddingSections = getMaxSplitGpuEmbeddingSectionsForDevice(embeddingKernel, device);
  return {
    ok: !limitError,
    splitKernelExpected: expectsSplitGpuEmbeddingKernel(embeddingKernel),
    activeSplitKernelMaxSections,
    maxSplitEmbeddingSections,
    requiredSplitSections,
    requiredStorageBuffers: activeSplitKernelMaxSections > 0
      ? getSplitGpuEmbeddingRequiredStorageBuffers(activeSplitKernelMaxSections)
      : null,
    largeWeightMaxBytes,
    rowsPerSplitSection,
  };
}

function buildFailureEvidence(limitError) {
  const failure = limitError?.details?.weightLoadFailure;
  if (!failure) {
    return limitError ? { message: limitError.message || String(limitError) } : null;
  }
  return failure;
}

export function buildGemma4PreflightReceipt({
  manifest,
  manifestPath,
  manifestRaw,
  generatedAt,
  surface = 'node',
  adapterInfo = null,
  capabilities,
  device,
  largeWeightMaxBytes,
  limitError = null,
}) {
  if (!isObject(manifest)) {
    throw new Error('manifest must be an object');
  }
  const modelId = typeof manifest.modelId === 'string' && manifest.modelId.trim()
    ? manifest.modelId.trim()
    : null;
  if (!modelId) {
    throw new Error('manifest.modelId is required');
  }
  const manifestText = typeof manifestRaw === 'string'
    ? manifestRaw
    : `${JSON.stringify(manifest, null, 2)}\n`;
  const manifestRepoPath = repoRelativePath(resolveRepoPath(manifestPath));
  const { tensorName, location } = resolveEmbeddingTensor(manifest);
  const embeddingKernel = manifest?.inference?.execution?.kernels?.embed;
  if (!isObject(embeddingKernel)) {
    throw new Error(`${modelId}: manifest is missing inference.execution.kernels.embed`);
  }
  const estimate = resolveEmbeddingEstimate(location);
  const effectiveLargeWeightMaxBytes = largeWeightMaxBytes ?? getLargeWeightMaxBytes();
  const preflight = buildPreflightMath({
    location,
    embeddingKernel,
    device,
    largeWeightMaxBytes: effectiveLargeWeightMaxBytes,
    limitError,
  });
  const resolvedCapabilities = capabilities ?? {};
  const receipt = {
    receiptVersion: 'doppler_gemma4_preflight_receipt_v1',
    schemaVersion: 1,
    surface,
    runtime: 'doppler-gpu',
    target: 'gpuResidentEmbeddingLimit',
    status: limitError ? 'diagnostic' : 'pass',
    modelId,
    generatedAt: normalizeIsoTimestamp(generatedAt, 'generatedAt'),
    observedWith: 'node WebGPU provider plus resolveManifestGpuResidentEmbeddingLimitError',
    manifest: {
      path: manifestRepoPath,
      sha256: `sha256:${sha256Hex(manifestText)}`,
    },
    adapterInfo: adapterInfo ?? resolveAdapterInfo(resolvedCapabilities),
    capabilities: {
      hasF16: resolvedCapabilities.hasF16 === true,
      hasSubgroups: resolvedCapabilities.hasSubgroups === true,
    },
    deviceLimits: resolveDeviceLimits(device),
    embedding: {
      tensorName,
      dtype: location.dtype,
      shape: [...location.shape],
      tensorSizeBytes: estimate.tensorSizeBytes,
      kernel: {
        kernel: embeddingKernel.kernel,
        entry: embeddingKernel.entry,
        digest: embeddingKernel.digest,
      },
    },
    preflight,
  };
  const failure = buildFailureEvidence(limitError);
  if (failure) {
    receipt.failure = failure;
  }
  return receipt;
}

export function formatReceiptTimestamp(generatedAt) {
  return normalizeIsoTimestamp(generatedAt, 'generatedAt')
    .replace(/\.\d{3}Z$/, 'Z')
    .replace(/:/g, '');
}

export function buildGemma4PreflightReceiptPath({ outputRoot = DEFAULT_OUTPUT_ROOT, modelId, generatedAt }) {
  return path.join(outputRoot, modelId, `${formatReceiptTimestamp(generatedAt)}.preflight.json`);
}

export function serializeReceipt(receipt) {
  return `${JSON.stringify(receipt, null, 2)}\n`;
}

async function readJsonWithRaw(filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  return {
    raw,
    payload: JSON.parse(raw),
  };
}

async function loadRuntimeFacts() {
  const bootstrap = await bootstrapNodeWebGPU();
  if (!bootstrap.ok) {
    throw new Error(`Node WebGPU provider unavailable: ${bootstrap.detail || bootstrap.provider || 'unknown failure'}`);
  }
  const device = await initDevice();
  const capabilities = getKernelCapabilities();
  return { device, capabilities };
}

async function buildReceiptForManifest({ manifestPath, generatedAt, surface, runtimeFacts }) {
  const { raw, payload } = await readJsonWithRaw(manifestPath);
  const limitError = resolveManifestGpuResidentEmbeddingLimitError(payload);
  return buildGemma4PreflightReceipt({
    manifest: payload,
    manifestPath,
    manifestRaw: raw,
    generatedAt,
    surface,
    capabilities: runtimeFacts.capabilities,
    device: runtimeFacts.device,
    largeWeightMaxBytes: getLargeWeightMaxBytes(),
    limitError,
  });
}

async function writeReceipts(args, runtimeFacts) {
  const generatedAt = args.generatedAt ?? new Date().toISOString();
  const manifestPaths = args.manifestPaths.length > 0
    ? args.manifestPaths
    : DEFAULT_MANIFEST_PATHS.map(resolveRepoPath);
  const written = [];
  for (const manifestPath of manifestPaths) {
    const receipt = await buildReceiptForManifest({
      manifestPath,
      generatedAt,
      surface: 'node',
      runtimeFacts,
    });
    const receiptPath = buildGemma4PreflightReceiptPath({
      outputRoot: args.outputRoot,
      modelId: receipt.modelId,
      generatedAt: receipt.generatedAt,
    });
    await fs.mkdir(path.dirname(receiptPath), { recursive: true });
    await fs.writeFile(receiptPath, serializeReceipt(receipt), 'utf8');
    written.push(receiptPath);
  }
  for (const receiptPath of written) {
    console.log(`[gemma4-preflight] wrote ${repoRelativePath(receiptPath)}`);
  }
}

function collectPreflightReceiptEntries(targetMatrix) {
  const entries = [];
  for (const target of targetMatrix?.targets || []) {
    const evidence = isObject(target?.evidence) ? target.evidence : {};
    for (const receipt of evidence.preflightReceipts || []) {
      entries.push({
        targetId: target.targetId || 'unknown-target',
        receipt,
      });
    }
  }
  return entries;
}

async function checkReceipts(args, runtimeFacts) {
  const targetMatrix = (await readJsonWithRaw(args.targetMatrixPath)).payload;
  const entries = collectPreflightReceiptEntries(targetMatrix);
  const failures = [];
  for (const entry of entries) {
    const receiptPath = resolveRepoPath(entry.receipt.path);
    const existing = await readJsonWithRaw(receiptPath);
    const existingReceipt = existing.payload;
    const manifestPath = resolveRepoPath(existingReceipt?.manifest?.path);
    const rebuilt = await buildReceiptForManifest({
      manifestPath,
      generatedAt: existingReceipt.generatedAt,
      surface: entry.receipt.surface || existingReceipt.surface || 'node',
      runtimeFacts,
    });
    const nextText = serializeReceipt(rebuilt);
    if (existing.raw !== nextText) {
      failures.push(`${entry.targetId}: ${repoRelativePath(receiptPath)} is not reproducible from current manifest and adapter`);
    }
  }
  if (failures.length > 0) {
    throw new Error(failures.join('\n'));
  }
  console.log(`[gemma4-preflight] checked ${entries.length} receipts`);
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(usage());
    return;
  }
  const runtimeFacts = await loadRuntimeFacts();
  try {
    if (args.check) {
      await checkReceipts(args, runtimeFacts);
      return;
    }
    await writeReceipts(args, runtimeFacts);
  } finally {
    destroyDevice();
  }
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  return entryPath && path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[gemma4-preflight] ${error.message}`);
    process.exitCode = 1;
  });
}
