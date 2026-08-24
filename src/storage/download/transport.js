import {
  createFileWriter,
  createStreamingHasher,
  deleteFileFromStore,
} from '../shard-manager.js';
import { getDistributionConfig } from '../download-types.js';
import { normalizeSourceArtifactPath } from '../source-artifact-store.js';
import { downloadDistributedShard } from '../distribution-transport.js';
import { fetchWithRetry } from './retry.js';

export function joinArtifactUrl(baseUrl, relativePath) {
  const root = String(baseUrl || '').trim();
  const rel = normalizeSourceArtifactPath(relativePath);
  if (!root || !rel) {
    throw new Error('joinArtifactUrl requires baseUrl and relativePath.');
  }
  return new URL(rel, root.endsWith('/') ? root : `${root}/`).href;
}

export async function downloadSourceAsset(url, asset, options = {}) {
  const response = await fetchWithRetry(url, { signal: options.signal });
  const expectedSize = Number.isFinite(asset?.size) ? Math.max(0, Math.floor(Number(asset.size))) : null;
  const expectedHash = typeof asset?.hash === 'string' && asset.hash.trim() ? asset.hash.trim().toLowerCase() : null;
  const hashAlgorithm = typeof asset?.hashAlgorithm === 'string' && asset.hashAlgorithm.trim()
    ? asset.hashAlgorithm.trim().toLowerCase()
    : 'sha256';
  const writer = await createFileWriter(asset.path);
  const hasher = expectedHash ? await createStreamingHasher(hashAlgorithm) : null;
  let receivedBytes = 0;
  try {
    if (response.body && typeof response.body.getReader === 'function') {
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (!(value instanceof Uint8Array) || value.byteLength <= 0) {
          continue;
        }
        await writer.write(value);
        hasher?.update(value);
        receivedBytes += value.byteLength;
        options.onProgress?.(receivedBytes);
      }
    } else {
      const bytes = new Uint8Array(await response.arrayBuffer());
      await writer.write(bytes);
      hasher?.update(bytes);
      receivedBytes += bytes.byteLength;
      options.onProgress?.(receivedBytes);
    }
    await writer.close();

    if (expectedSize != null && receivedBytes !== expectedSize) {
      throw new Error(
        `Asset size mismatch for ${asset.path}: expected ${expectedSize}, got ${receivedBytes}`
      );
    }

    if (hasher && expectedHash) {
      const computedHashBytes = await hasher.finalize();
      const computedHash = Array.from(computedHashBytes)
        .map((byte) => byte.toString(16).padStart(2, '0'))
        .join('');
      if (computedHash !== expectedHash) {
        throw new Error(
          `Asset hash mismatch for ${asset.path}: expected ${expectedHash}, got ${computedHash}`
        );
      }
    }

    return {
      source: 'http',
      path: asset.path,
      bytes: receivedBytes,
    };
  } catch (error) {
    try {
      await writer.abort();
    } catch {}
    try {
      await deleteFileFromStore(asset.path);
    } catch {}
    throw error;
  }
}

export async function downloadShard(
  baseUrl,
  shardIndex,
  shardInfo,
  options = {}
) {
  return downloadDistributedShard(baseUrl, shardIndex, shardInfo, {
    ...options,
    distributionConfig: getDistributionConfig(),
  });
}
