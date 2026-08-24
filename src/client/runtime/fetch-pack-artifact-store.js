function resolveUrl(baseUrl, artifact) {
  return new URL(artifact.path, baseUrl).href;
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Pack artifact fetch failed (${response.status}) for ${url}.`);
  return new Uint8Array(await response.arrayBuffer());
}

async function hashBytes(bytes) {
  if (!globalThis.crypto?.subtle) throw new Error('Pack artifact verification requires WebCrypto.');
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (value) => value.toString(16).padStart(2, '0')).join('');
}

export function createFetchPackArtifactStore(packUrl) {
  const baseUrl = new URL('.', packUrl).href;
  return {
    async hashArtifact(artifact) {
      const bytes = await fetchBytes(resolveUrl(baseUrl, artifact));
      return { hash: `sha256:${await hashBytes(bytes)}`, sizeBytes: bytes.byteLength };
    },
    readArtifact(artifact) {
      return fetchBytes(resolveUrl(baseUrl, artifact));
    },
    resolveArtifactUrl(artifact) {
      return resolveUrl(baseUrl, artifact);
    },
  };
}
