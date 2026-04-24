import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runNodeCommand } from './node-command-runner.js';
import { runBrowserCommandInNode } from './node-browser-command-runner.js';

const DEFAULT_BROWSER_TIMEOUT_MS = 600000;
const PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID = 'doppler.reference-transcript/v1';

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function mergePlainObjects(base, patch) {
  const output = { ...(isPlainObject(base) ? base : {}) };
  for (const [key, value] of Object.entries(patch)) {
    if (isPlainObject(value) && isPlainObject(output[key])) {
      output[key] = mergePlainObjects(output[key], value);
    } else {
      output[key] = value;
    }
  }
  return output;
}

export async function normalizeRuntimeConfigInput(input) {
  if (input == null || input === '') return {};
  if (isPlainObject(input)) {
    return { runtimeConfig: input };
  }
  const raw = String(input).trim();
  if (!raw) return {};
  if (raw.startsWith('{')) {
    return { runtimeConfig: JSON.parse(raw) };
  }
  return { runtimeConfigUrl: pathToFileURL(path.resolve(raw)).href };
}

export function withReferenceTranscriptRuntimeConfig(runtimeInput) {
  const proofRuntimeConfig = {
    shared: {
      harness: {
        referenceTranscript: {
          enabled: true,
          captureLogits: true,
          captureKvBytes: true,
        },
      },
    },
  };
  return {
    ...runtimeInput,
    runtimeConfig: mergePlainObjects(runtimeInput.runtimeConfig ?? {}, proofRuntimeConfig),
  };
}

function localModelDirFromUrl(modelUrl) {
  if (typeof modelUrl !== 'string' || !modelUrl.startsWith('file://')) {
    return null;
  }
  return fileURLToPath(modelUrl);
}

export async function assertLocalModelArtifactsReadable({ modelUrl, manifest }) {
  const modelDir = localModelDirFromUrl(modelUrl);
  if (!modelDir) return;
  const missing = [];
  const tokenizerFile = manifest?.tokenizer?.file;
  if (typeof tokenizerFile === 'string' && tokenizerFile.trim()) {
    const tokenizerPath = path.resolve(modelDir, tokenizerFile);
    try {
      await fs.access(tokenizerPath);
    } catch {
      missing.push(path.relative(process.cwd(), tokenizerPath));
    }
  }
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  for (const shard of shards) {
    const filename = typeof shard?.filename === 'string'
      ? shard.filename
      : (typeof shard?.path === 'string' ? shard.path : null);
    if (!filename) continue;
    const shardPath = path.resolve(modelDir, filename);
    try {
      await fs.access(shardPath);
    } catch {
      missing.push(path.relative(process.cwd(), shardPath));
      if (missing.length >= 5) break;
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `reference capture: local model artifacts are incomplete under ${modelDir}. `
      + `Missing: ${missing.join(', ')}${missing.length >= 5 ? ', ...' : ''}. `
      + 'Pass --model-url to a complete hosted/local artifact or restore the shard files before running the capture lane.'
    );
  }
}

export function normalizeModelUrl(value, modelDir) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) {
    return pathToFileURL(path.resolve(modelDir)).href;
  }
  if (/^[a-z][a-z0-9+.-]*:\/\//u.test(raw)) {
    return raw;
  }
  return pathToFileURL(path.resolve(raw)).href;
}

export async function runReferenceCapture({
  manifest,
  modelId,
  modelUrl,
  surface,
  prompt,
  maxTokens,
  runtimeConfig,
  repoRoot,
  browserTimeoutMs,
}) {
  await assertLocalModelArtifactsReadable({ modelUrl, manifest });
  const runtimeInput = withReferenceTranscriptRuntimeConfig(
    await normalizeRuntimeConfigInput(runtimeConfig)
  );
  const request = {
    command: 'verify',
    workload: 'inference',
    modelId,
    modelUrl,
    loadMode: modelUrl.startsWith('file://') ? 'http' : null,
    inferenceInput: {
      prompt,
      maxTokens,
    },
    ...runtimeInput,
  };
  if (surface === 'node') {
    return runNodeCommand(request);
  }
  return runBrowserCommandInNode(request, {
    opfsCache: false,
    timeoutMs: browserTimeoutMs ?? DEFAULT_BROWSER_TIMEOUT_MS,
    staticRootDir: repoRoot,
  });
}

export function extractReferenceReport(response) {
  const report = response?.result?.report;
  if (!isPlainObject(report)) {
    throw new Error(
      'reference capture: verify response did not include result.report. '
      + 'Use a command runner that returns the full report object.'
    );
  }
  return report;
}

export function extractReferenceTranscriptSeed(report) {
  const results = Array.isArray(report?.results) ? report.results : [];
  for (const entry of results) {
    const seed = entry?.runReport?.referenceTranscript
      ?? entry?.referenceTranscript
      ?? null;
    if (isPlainObject(seed) && seed.schema === PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID) {
      return seed;
    }
  }
  const inlineSeed = report?.referenceTranscript ?? null;
  if (isPlainObject(inlineSeed) && inlineSeed.schema === PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID) {
    return inlineSeed;
  }
  throw new Error(
    'reference capture: verify report did not contain a doppler.reference-transcript/v1 seed. '
    + 'Confirm runtime.shared.harness.referenceTranscript.enabled was honored by the surface.'
  );
}

export async function writeReferenceReport(report, reportPath) {
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  return reportPath;
}

export async function writeReferenceTranscript(transcript, transcriptPath) {
  await fs.mkdir(path.dirname(transcriptPath), { recursive: true });
  await fs.writeFile(transcriptPath, `${JSON.stringify(transcript, null, 2)}\n`, 'utf8');
  return transcriptPath;
}

export { PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID };
