import { log } from '../../debug/index.js';
import { getManifestUrl, parseManifest } from '../../formats/rdrr/index.js';
import { buildQuickstartModelBaseUrl, resolveQuickstartModel } from '../doppler-registry.js';

export function createDefaultNodeLoadProgressLogger() {
  return (event) => {
    const message = typeof event?.message === 'string' ? event.message.trim() : '';
    if (!message) return;
    log.info('doppler', message);
  };
}

export function resolveLoadProgressHandlers(options = {}, defaultLoadProgressLogger = null) {
  const onProgress = typeof options?.onProgress === 'function' ? options.onProgress : null;
  if (onProgress) {
    return {
      userProgress: onProgress,
      pipelineProgress: onProgress,
    };
  }
  if (typeof defaultLoadProgressLogger === 'function') {
    return {
      userProgress: defaultLoadProgressLogger,
      pipelineProgress: null,
    };
  }
  log.debug('doppler', 'resolveLoadProgressHandlers: no progress handler configured, returning null handlers');
  return {
    userProgress: null,
    pipelineProgress: null,
  };
}

export async function fetchManifestPayloadFromBaseUrl(baseUrl) {
  const response = await fetch(getManifestUrl(baseUrl));
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest from ${baseUrl}: ${response.status}`);
  }
  const text = await response.text();
  return {
    text,
    manifest: parseManifest(text),
  };
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeDigest(value) {
  const normalized = normalizeText(value).toLowerCase();
  if (!normalized) return null;
  return normalized.startsWith('sha256:') ? normalized.slice('sha256:'.length) : normalized;
}

async function sha256Text(value) {
  if (typeof crypto === 'undefined' || !crypto?.subtle) {
    throw new Error('weightsRef manifestDigest verification requires crypto.subtle.');
  }
  const bytes = new TextEncoder().encode(String(value));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

function resolveWeightsRefBaseUrl(baseUrl, artifactRoot) {
  const root = normalizeText(artifactRoot);
  if (!root) {
    throw new Error('weightsRef.artifactRoot is required.');
  }
  if (/^(https?|file):\/\//i.test(root)) {
    return root.replace(/\/+$/, '');
  }
  if (root.startsWith('/')) {
    return root.replace(/\/+$/, '');
  }
  const base = normalizeText(baseUrl);
  if (!base) {
    throw new Error('Relative weightsRef.artifactRoot requires a model baseUrl.');
  }
  if (/^https?:\/\//i.test(base) || /^file:\/\//i.test(base)) {
    return new URL(root, base.endsWith('/') ? base : `${base}/`).toString().replace(/\/+$/, '');
  }
  return `${base.replace(/\/+$/, '')}/${root.replace(/^\/+/, '')}`.replace(/\/+$/, '');
}

function assertWeightsRefIdentity(variantManifest, weightsManifest, weightsRef, storageBaseUrl) {
  const modelId = variantManifest?.modelId ?? 'unknown';
  const expectedWeightPackId = normalizeText(weightsRef?.weightPackId);
  if (!expectedWeightPackId) {
    throw new Error(`${modelId}: weightsRef.weightPackId is required.`);
  }
  const variantWeightPackId = normalizeText(variantManifest?.artifactIdentity?.weightPackId);
  if (variantWeightPackId && variantWeightPackId !== expectedWeightPackId) {
    throw new Error(
      `${modelId}: weightsRef.weightPackId "${expectedWeightPackId}" does not match ` +
      `manifest artifactIdentity.weightPackId "${variantWeightPackId}".`
    );
  }
  const storageWeightPackId = normalizeText(weightsManifest?.artifactIdentity?.weightPackId);
  if (storageWeightPackId !== expectedWeightPackId) {
    throw new Error(
      `${modelId}: weightsRef target ${storageBaseUrl} has artifactIdentity.weightPackId ` +
      `"${storageWeightPackId}", expected "${expectedWeightPackId}".`
    );
  }
  const expectedShardSetHash = normalizeText(weightsRef?.shardSetHash);
  if (expectedShardSetHash) {
    const actualShardSetHash = normalizeText(weightsManifest?.artifactIdentity?.shardSetHash);
    if (actualShardSetHash !== expectedShardSetHash) {
      throw new Error(
        `${modelId}: weightsRef.shardSetHash "${expectedShardSetHash}" does not match ` +
        `target artifactIdentity.shardSetHash "${actualShardSetHash}".`
      );
    }
  }
}

export async function resolveManifestArtifactSource(resolved, manifestPayload) {
  const manifest = manifestPayload?.manifest;
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('Manifest payload is required before artifact source resolution.');
  }
  const weightsRef = manifest.weightsRef;
  if (weightsRef == null) {
    return {
      ...resolved,
      manifest,
      manifestText: manifestPayload?.text ?? JSON.stringify(manifest),
      storageManifest: manifest,
      storageManifestText: manifestPayload?.text ?? JSON.stringify(manifest),
      storageBaseUrl: resolved?.baseUrl ?? null,
      variantBaseUrl: resolved?.baseUrl ?? null,
    };
  }

  const storageBaseUrl = resolveWeightsRefBaseUrl(resolved?.baseUrl, weightsRef.artifactRoot);
  const storageManifestPayload = await fetchManifestPayloadFromBaseUrl(storageBaseUrl);
  const expectedManifestDigest = normalizeDigest(weightsRef.manifestDigest);
  if (expectedManifestDigest) {
    const actualManifestDigest = await sha256Text(storageManifestPayload.text);
    if (actualManifestDigest !== expectedManifestDigest) {
      throw new Error(
        `${manifest.modelId ?? 'unknown'}: weightsRef.manifestDigest mismatch for ${storageBaseUrl}. ` +
        `Expected ${expectedManifestDigest}, got ${actualManifestDigest}.`
      );
    }
  }
  assertWeightsRefIdentity(manifest, storageManifestPayload.manifest, weightsRef, storageBaseUrl);
  return {
    ...resolved,
    baseUrl: storageBaseUrl,
    manifest,
    manifestText: manifestPayload?.text ?? JSON.stringify(manifest),
    storageManifest: storageManifestPayload.manifest,
    storageManifestText: storageManifestPayload.text,
    storageBaseUrl,
    variantBaseUrl: resolved?.baseUrl ?? null,
    trace: [
      ...(Array.isArray(resolved?.trace) ? resolved.trace : []),
      { source: 'weightsRef', id: storageBaseUrl, outcome: 'resolved' },
    ],
  };
}

export async function resolveModelSource(model) {
  const trace = [];

  if (typeof model === 'string') {
    try {
      const entry = await resolveQuickstartModel(model);
      trace.push({ source: 'quickstart-registry', id: model, outcome: 'resolved' });
      log.debug('doppler', `Model resolved via quickstart-registry: ${entry.modelId}`, { trace });
      return {
        modelId: entry.modelId,
        baseUrl: buildQuickstartModelBaseUrl(entry),
        manifest: null,
        trace,
      };
    } catch (registryError) {
      trace.push({
        source: 'quickstart-registry',
        id: model,
        outcome: registryError?.message || 'not-found',
      });
    }
  }

  if (model && typeof model === 'object' && typeof model.url === 'string' && model.url.trim().length > 0) {
    trace.push({ source: 'url', id: model.url.trim(), outcome: 'resolved' });
    log.debug('doppler', `Model resolved via explicit url: ${model.url.trim()}`, { trace });
    return {
      modelId: model.url.trim(),
      baseUrl: model.url.trim(),
      manifest: null,
      trace,
    };
  }
  if (model && typeof model === 'object' && typeof model.url === 'string') {
    trace.push({ source: 'url', id: String(model.url), outcome: 'empty-url' });
  }

  if (model && typeof model === 'object' && model.manifest && typeof model.manifest === 'object') {
    const manifest = model.manifest;
    const modelId = typeof manifest.modelId === 'string' && manifest.modelId.length > 0
      ? manifest.modelId
      : 'manifest';
    trace.push({ source: 'inline-manifest', id: modelId, outcome: 'resolved' });
    log.debug('doppler', `Model resolved via inline manifest: ${modelId}`, { trace });
    return {
      modelId,
      baseUrl: typeof model.baseUrl === 'string' && model.baseUrl.length > 0 ? model.baseUrl : null,
      manifest,
      trace,
    };
  }

  const traceDescription = trace.length > 0
    ? trace.map((entry) => `${entry.source} (${entry.outcome})`).join(', ')
    : 'no sources attempted';
  throw new Error(
    `Model not found. Attempted: ${traceDescription}. ` +
    'doppler.load expects a quickstart registry id, { url }, or { manifest, baseUrl? }.'
  );
}
