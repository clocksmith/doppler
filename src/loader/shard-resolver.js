import { loadTensorsFromStore } from '../storage/shard-manager.js';
import { parseTensorMap } from '../formats/rdrr/index.js';
import { log, trace as debugTrace } from '../debug/index.js';

function normalizeLocationSpans(spans, name, sourceLabel) {
  if (spans === undefined) {
    return undefined;
  }
  if (!Array.isArray(spans)) {
    throw new Error(`Tensor "${name}" has invalid spans in ${sourceLabel}`);
  }
  return spans.map((span, spanIndex) => {
    const shardIndex = typeof span?.shardIndex === 'number'
      ? span.shardIndex
      : span?.shard;
    if (typeof shardIndex !== 'number') {
      throw new Error(`Tensor "${name}" span[${spanIndex}] missing shard index in ${sourceLabel}`);
    }
    return {
      shardIndex,
      offset: span.offset,
      size: span.size,
    };
  });
}

/**
 * Builds the tensor name-to-location map from the manifest.
 *
 * Tensor map resolution order (first successful source wins):
 *   1. External tensors.json via OPFS store — tried when the model was
 *      previously downloaded and no custom loader overrides storage.
 *   2. External tensors.json via HTTP — tried when `options.tensorsJsonUrl`
 *      is set (HTTP-based testing / direct serve).
 *   3. Inline manifest.tensors (legacy) — used when `manifest.tensorsFile`
 *      is absent and tensor entries are embedded in the manifest JSON.
 *
 * On failure, the resolution trace is included in the warning/error so
 * callers can see which sources were attempted and why each was skipped.
 */
export async function buildTensorLocations(manifest, options = {}) {
  const locations = new Map();
  const trace = [];

  // v1 format: load external tensors.json
  if (manifest.tensorsFile) {
    debugTrace.loader(`Loading external tensor map: ${manifest.tensorsFile}`);

    let tensorsJsonRaw = null;

    // 1. Try OPFS first (for downloaded models)
    if (!options.hasCustomLoader) {
      tensorsJsonRaw = await loadTensorsFromStore();
      if (tensorsJsonRaw) {
        trace.push({ source: 'opfs-store', outcome: 'resolved' });
      } else {
        trace.push({ source: 'opfs-store', outcome: 'not-found' });
      }
    } else {
      trace.push({ source: 'opfs-store', outcome: 'skipped (custom-loader)' });
    }

    // 2. Try HTTP if we have a tensors URL set (for HTTP-based testing)
    if (!tensorsJsonRaw && typeof options.loadTensorsJson === 'function') {
      try {
        const payload = await options.loadTensorsJson();
        if (payload != null) {
          tensorsJsonRaw = typeof payload === 'string' ? payload : JSON.stringify(payload);
          trace.push({ source: 'storage-context', outcome: 'resolved' });
        } else {
          trace.push({ source: 'storage-context', outcome: 'not-found' });
        }
      } catch (e) {
        trace.push({ source: 'storage-context', outcome: `error: ${e.message}` });
        log.warn('Loader', `Failed to load tensors.json from storage context: ${e.message}`);
      }
    }

    if (!tensorsJsonRaw && options.tensorsJsonUrl) {
      try {
        const resp = await fetch(options.tensorsJsonUrl);
        if (resp.ok) {
          tensorsJsonRaw = await resp.text();
          trace.push({ source: 'http', url: options.tensorsJsonUrl, outcome: 'resolved' });
          debugTrace.loader(`Loaded tensors.json via HTTP: ${options.tensorsJsonUrl}`);
        } else {
          trace.push({ source: 'http', url: options.tensorsJsonUrl, outcome: `http-${resp.status}` });
        }
      } catch (e) {
        trace.push({ source: 'http', url: options.tensorsJsonUrl, outcome: `error: ${e.message}` });
        log.warn('Loader', `Failed to load tensors.json from ${options.tensorsJsonUrl}: ${e.message}`);
      }
    } else if (!tensorsJsonRaw && !options.tensorsJsonUrl) {
      trace.push({ source: 'http', outcome: 'skipped (no tensorsJsonUrl)' });
    }

    if (tensorsJsonRaw) {
      const tensorsJson = parseTensorMap(tensorsJsonRaw);
      for (const [name, rdrrInfo] of Object.entries(tensorsJson)) {
        const info = rdrrInfo;
        if (!info.role) {
          throw new Error(`Tensor "${name}" missing role in tensors.json`);
        }
        locations.set(name, {
          shardIndex: info.shardIndex ?? info.shard,
          offset: info.offset,
          size: info.size,
          shape: info.shape,
          dtype: info.dtype,
          role: info.role,
          group: info.group,
          spans: normalizeLocationSpans(info.spans, name, 'tensors.json'),
          layout: info.layout,
          originalShape: info.originalShape,
          sourceTransform: info.sourceTransform,
        });
      }
      const resolvedSource = trace.find((entry) => entry.outcome === 'resolved')?.source ?? 'unknown';
      debugTrace.loader(`Loaded ${locations.size} tensors from tensors.json (source: ${resolvedSource})`);
      log.debug('Loader', `Tensor map resolved via ${resolvedSource}`, { trace });
      return locations;
    }

    // External tensors.json required but not found from any source
    const traceDescription = trace.map((entry) => `${entry.source} (${entry.outcome})`).join(', ');
    log.warn('Loader', `tensors.json not found from any source. Attempted: ${traceDescription}`);
  }

  // 3. Legacy format: inline tensors in manifest
  if (!manifest.tensors) {
    const traceDescription = trace.length > 0
      ? trace.map((entry) => `${entry.source} (${entry.outcome})`).join(', ')
      : 'no external sources attempted';
    log.warn('Loader', `No tensor locations in manifest. Resolution trace: ${traceDescription}`);
    return locations;
  }

  trace.push({ source: 'inline-manifest', outcome: 'resolved' });
  for (const [name, info] of Object.entries(manifest.tensors)) {
    const tensorInfo = info;
    if (!tensorInfo.role) {
      throw new Error(`Tensor "${name}" missing role in manifest.tensors`);
    }
    locations.set(name, {
      shardIndex: tensorInfo.shardIndex ?? tensorInfo.shard ?? 0,
      offset: tensorInfo.offset,
      size: tensorInfo.size,
      shape: tensorInfo.shape,
      dtype: tensorInfo.dtype,
      role: tensorInfo.role,
      group: tensorInfo.group,
      spans: normalizeLocationSpans(tensorInfo.spans, name, 'manifest.tensors'),
      layout: tensorInfo.layout,
      originalShape: tensorInfo.originalShape,
      sourceTransform: tensorInfo.sourceTransform,
    });
  }
  debugTrace.loader(`Tensor map: ${locations.size} tensors (inline)`);
  log.debug('Loader', 'Tensor map resolved via inline manifest', { trace });
  return locations;
}
