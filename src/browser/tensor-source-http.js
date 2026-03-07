

function normalizeHeaderValue(value) {
  return value ? value.toLowerCase().trim() : '';
}

function parseContentLength(value) {
  if (!value) return null;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function inferNameFromUrl(url) {
  try {
    const baseHref = typeof globalThis.location !== 'undefined' ? globalThis.location.href : undefined;
    const parsed = new URL(url, baseHref);
    const pathname = parsed.pathname || '';
    const part = pathname.split('/').filter(Boolean).pop();
    return part || 'remote';
  } catch {
    const parts = String(url).split('/');
    return parts[parts.length - 1] || 'remote';
  }
}

export async function probeHttpRange(url, options = {}) {
  const { headers, signal } = options;
  try {
    const response = await fetch(url, {
      method: 'HEAD',
      headers,
      signal,
    });

    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        supportsRange: false,
        size: null,
        acceptRanges: null,
        contentEncoding: null,
      };
    }

    const acceptRanges = normalizeHeaderValue(response.headers.get('accept-ranges'));
    const contentEncoding = normalizeHeaderValue(response.headers.get('content-encoding'));
    const contentLength = parseContentLength(response.headers.get('content-length'));

    const supportsRange = Boolean(
      acceptRanges.includes('bytes') &&
      contentLength !== null &&
      (!contentEncoding || contentEncoding === 'identity')
    );

    return {
      ok: true,
      status: response.status,
      supportsRange,
      size: contentLength,
      acceptRanges,
      contentEncoding,
    };
  } catch (error) {
    return {
      ok: false,
      status: 0,
      supportsRange: false,
      size: null,
      acceptRanges: null,
      contentEncoding: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function createHttpTensorSource(url, options = {}) {
  const { headers, signal, name: overrideName } = options;
  const probe = await probeHttpRange(url, { headers, signal });
  if (!probe.ok && probe.status === 0 && probe.error) {
    throw new Error(`HTTP tensor source probe failed for "${url}": ${probe.error}`);
  }
  if (!probe.supportsRange || probe.size == null) {
    throw new Error('HTTP range requests not supported for tensor source');
  }

  const name = overrideName || inferNameFromUrl(url);
  const size = probe.size;

  return {
    sourceType: 'http',
    name,
    size,
    url,
    async readRange(offset, length) {
      if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
        return new ArrayBuffer(0);
      }
      const start = Math.max(0, offset);
      const end = Math.min(start + length - 1, size - 1);
      if (end < start) {
        return new ArrayBuffer(0);
      }

      const rangeHeader = `bytes=${start}-${end}`;
      const response = await fetch(url, {
        headers: {
          ...(headers || {}),
          Range: rangeHeader,
        },
        signal,
      });

      if (response.status !== 206) {
        throw new Error(`HTTP range request failed: ${response.status}`);
      }

      return response.arrayBuffer();
    },
    async readAll() {
      return this.readRange(0, size);
    },
    async close() {
      return;
    },
    async getAuxFiles() {
      return {};
    },
  };
}
