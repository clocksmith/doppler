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

export async function fetchManifestFromBaseUrl(baseUrl) {
  const response = await fetch(getManifestUrl(baseUrl));
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest from ${baseUrl}: ${response.status}`);
  }
  return parseManifest(await response.text());
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
