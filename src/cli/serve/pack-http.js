import { createHash, timingSafeEqual } from 'node:crypto';

export class PackServeError extends Error {
  constructor(code, message, statusCode) {
    super(message);
    this.name = 'PackServeError';
    this.code = code;
    this.statusCode = statusCode;
  }
}

const credentialHash = value => createHash('sha256').update(value).digest();

export function createPackRequestAuthorization(token, policy) {
  if (typeof token !== 'string' || !token.length || /\s/.test(token)) {
    throw new Error('Pack serving requires an explicit non-empty bearer token without whitespace.');
  }
  const expected = credentialHash(`Bearer ${token}`);
  return (req, res) => {
    const origin = req.headers.origin;
    if (origin !== undefined && !policy.allowedOrigins.includes(origin)) {
      throw new PackServeError('ORIGIN_DENIED', 'Origin is not authorized for this Pack session.', 403);
    }
    if (origin !== undefined) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Vary', 'Origin');
    }
    if (req.method === 'OPTIONS') return;
    const supplied = req.headers.authorization;
    if (typeof supplied !== 'string' || !timingSafeEqual(expected, credentialHash(supplied))) {
      throw new PackServeError('AUTH_REQUIRED', 'A valid bearer token is required.', 401);
    }
  };
}

export function readPackRequest(req, maxBytes, signal) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let bytes = 0;
    const cleanup = () => {
      req.off('data', data);
      req.off('end', end);
      req.off('error', fail);
      signal.removeEventListener('abort', abort);
    };
    const fail = error => { cleanup(); req.pause(); reject(error); };
    const abort = () => fail(signal.reason);
    const data = chunk => {
      bytes += chunk.length;
      if (bytes > maxBytes) {
        fail(new PackServeError('REQUEST_TOO_LARGE', 'Request exceeds maxRequestBytes.', 413));
      } else { chunks.push(chunk); }
    };
    const end = () => {
      cleanup();
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch { reject(new PackServeError('INVALID_JSON', 'Request must contain JSON.', 400)); }
    };
    req.on('data', data);
    req.once('end', end);
    req.once('error', fail);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
  });
}

export async function writePackEvent(res, event, signal) {
  signal.throwIfAborted();
  if (res.write(event)) return;
  await new Promise((resolve, reject) => {
    const cleanup = () => {
      res.off('drain', drain);
      res.off('error', fail);
      signal.removeEventListener('abort', abort);
    };
    const drain = () => { cleanup(); resolve(); };
    const fail = error => { cleanup(); reject(error); };
    const abort = () => fail(signal.reason);
    res.once('drain', drain);
    res.once('error', fail);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
  });
}

export function endPackError(res, error, remainingBytes) {
  if (res.destroyed || res.writableEnded) return;
  const payload = JSON.stringify({ error: {
    code: error?.code ?? 'PACK_EXECUTION_FAILED', message: error?.message ?? String(error),
  } });
  // Preserve the response ceiling even when no diagnostic envelope will fit.
  const body = Buffer.byteLength(payload) + 1 <= remainingBytes ? payload : '';
  if (!res.headersSent) {
    res.writeHead(error instanceof PackServeError ? error.statusCode : 500, {
      'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Connection': 'close',
    });
    res.end(body);
  } else {
    // This is transport failure, not a completed or accepted operation event.
    res.end(body ? `${body}\n` : '');
  }
}
