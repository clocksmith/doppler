import { normalizeText } from './catalog-io.js';

export async function probeUrl(url, options = {}) {
  const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : 30000;
  const methods = Array.isArray(options.methods) && options.methods.length > 0
    ? options.methods
    : ['HEAD', 'GET'];
  let lastError = null;

  for (const method of methods) {
    try {
      const headers = method === 'GET'
        ? { Connection: 'close', Range: 'bytes=0-0' }
        : { Connection: 'close' };
      const response = await fetch(url, {
        method,
        headers,
        redirect: 'follow',
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (response.ok || response.status === 206) {
        return {
          ok: true,
          status: response.status,
          url: response.url,
          method,
        };
      }
      lastError = new Error(`HTTP ${response.status}: ${url}`);
    } catch (error) {
      lastError = error;
    }
  }

  return {
    ok: false,
    status: null,
    url,
    method: methods[methods.length - 1] || 'HEAD',
    error: lastError,
  };
}

export async function fetchJson(url, options = {}) {
  const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : 30000;
  const response = await fetch(url, {
    headers: {
      Connection: 'close',
    },
    redirect: 'follow',
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${url}`);
  }
  return response.json();
}

export async function fetchRepoHeadSha(repoId, options = {}) {
  const normalizedRepoId = normalizeText(repoId);
  if (!normalizedRepoId) {
    throw new Error('repoId is required to fetch Hugging Face repo head SHA.');
  }
  const payload = await fetchJson(
    `https://huggingface.co/api/models/${normalizedRepoId}`,
    options
  );
  const sha = normalizeText(payload?.sha).toLowerCase();
  if (!/^[a-f0-9]{40}$/.test(sha)) {
    throw new Error(`Could not resolve HEAD commit SHA for Hugging Face repo "${normalizedRepoId}".`);
  }
  return sha;
}
