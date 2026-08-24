import {
  buildTensorBlockMerkleRoot,
  getShardInfo,
  parseTensorMap,
} from '../../formats/rdrr/index.js';
import { log } from '../../debug/index.js';
import { createHasher as createBlake3Hasher, hash as blake3Hash } from '../blake3.js';
import { isRequestedRangeInsideTensor } from './index.js';

let blake3Module = null;
let hashAlgorithm = null;

async function initBlake3(requiredAlgorithm = null) {
  if (blake3Module && hashAlgorithm) return;

  try {
    blake3Module = {
      hash: blake3Hash,
      createHasher: createBlake3Hasher,
    };
    hashAlgorithm = 'blake3';
    return;
  } catch (error) {
    log.warn('ShardManager', `BLAKE3 module not available: ${error.message}`);
  }

  if (requiredAlgorithm === 'blake3') {
    throw new Error(
      'BLAKE3 required by manifest but not available. '
      + 'Install the JS blake3 module or re-convert model with SHA-256.'
    );
  }

  log.warn(
    'ShardManager',
    'BLAKE3 unavailable; falling back to SHA-256 for hash verification. '
      + 'Hashes will not match BLAKE3-based manifests.'
  );
  hashAlgorithm = 'sha256';
  blake3Module = {
    hash: async (data) => {
      const hashBuffer = await crypto.subtle.digest(
        'SHA-256',
        data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
      );
      return new Uint8Array(hashBuffer);
    },
    createHasher: () => {
      const chunks = [];
      return {
        update(data) {
          if (!(data instanceof Uint8Array) && !(data instanceof ArrayBuffer)) {
            throw new Error('SHA-256 fallback hasher: update() requires Uint8Array or ArrayBuffer');
          }
          chunks.push(new Uint8Array(data));
        },
        async finalize() {
          const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
          const combined = new Uint8Array(totalLength);
          let offset = 0;
          for (const chunk of chunks) {
            combined.set(chunk, offset);
            offset += chunk.length;
          }
          const hashBuffer = await crypto.subtle.digest('SHA-256', combined);
          return new Uint8Array(hashBuffer);
        },
      };
    },
  };
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

export function getHashAlgorithm() {
  return hashAlgorithm;
}

export function hexToBytes(hex) {
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = parseInt(hex.substring(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

export async function computeBlake3(data) {
  await initBlake3('blake3');
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  return bytesToHex(await blake3Module.hash(bytes));
}

export async function computeSHA256(data) {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return bytesToHex(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)));
}

export async function computeHash(data, algorithm) {
  if (!algorithm) {
    throw new Error('computeHash requires an explicit hash algorithm.');
  }
  return algorithm === 'sha256' ? computeSHA256(data) : computeBlake3(data);
}

export async function createStreamingHasher(algorithm) {
  if (!algorithm) {
    throw new Error('createStreamingHasher requires an explicit hash algorithm.');
  }
  if (algorithm === 'sha256') {
    const chunks = [];
    return {
      update(data) {
        chunks.push(new Uint8Array(data));
      },
      async finalize() {
        const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const combined = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
          combined.set(chunk, offset);
          offset += chunk.length;
        }
        return new Uint8Array(await crypto.subtle.digest('SHA-256', combined));
      },
    };
  }
  await initBlake3('blake3');
  return blake3Module.createHasher();
}

export function requireManifestHashAlgorithm(manifest, context) {
  const algorithm = manifest?.hashAlgorithm;
  if (!algorithm) {
    throw new Error(
      `Manifest missing hashAlgorithm for ${context}. `
      + 'Re-convert the model to include a manifest hash algorithm.'
    );
  }
  return algorithm;
}

export function createTensorIntegrityController({
  readBackendFileRange,
  loadTensorsFromStore,
}) {
  let cachedTensorMap = null;
  const verifiedTensorRootCache = new Map();

  function reset() {
    cachedTensorMap = null;
    verifiedTensorRootCache.clear();
  }

  async function loadTensorMap(manifest) {
    if (cachedTensorMap) return cachedTensorMap;
    if (manifest?.tensors && typeof manifest.tensors === 'object' && !Array.isArray(manifest.tensors)) {
      cachedTensorMap = manifest.tensors;
      return cachedTensorMap;
    }
    const tensorsJson = await loadTensorsFromStore();
    if (!tensorsJson) {
      throw new Error('Tensor integrity verification requires inline tensors or tensors.json to be present.');
    }
    cachedTensorMap = parseTensorMap(tensorsJson);
    return cachedTensorMap;
  }

  async function verifyTensorRoot(manifest, tensorId) {
    const roots = manifest?.integrityExtensions?.blockMerkle?.roots;
    if (!roots || typeof roots !== 'object' || Array.isArray(roots)) {
      throw new Error('Manifest is missing integrityExtensions.blockMerkle for tensor integrity verification.');
    }
    const normalizedTensorId = typeof tensorId === 'string' ? tensorId.trim() : '';
    if (!normalizedTensorId) {
      throw new Error('Tensor integrity verification requires a non-empty tensorId.');
    }
    const expectedRoot = roots[normalizedTensorId];
    if (typeof expectedRoot !== 'string' || !expectedRoot.trim()) {
      throw new Error(`Manifest is missing a block Merkle root for tensor "${normalizedTensorId}".`);
    }

    const tensorMap = await loadTensorMap(manifest);
    const location = tensorMap?.[normalizedTensorId];
    if (!location || typeof location !== 'object') {
      throw new Error(`Tensor "${normalizedTensorId}" is missing from the tensor map.`);
    }

    const cacheKey = `${normalizedTensorId}:${expectedRoot}`;
    if (verifiedTensorRootCache.get(cacheKey) !== true) {
      const built = await buildTensorBlockMerkleRoot(normalizedTensorId, location, {
        blockSize: manifest?.integrityExtensions?.blockMerkle?.blockSize,
        async readShardRange(shardIndex, offset, length) {
          const shardInfo = getShardInfo(shardIndex);
          if (!shardInfo) {
            throw new Error(`Invalid shard index during tensor integrity verification: ${shardIndex}`);
          }
          return readBackendFileRange(shardInfo.filename, offset, length);
        },
      });
      if (built.root !== expectedRoot) {
        throw new Error(
          `Tensor integrity mismatch for "${normalizedTensorId}": expected ${expectedRoot}, got ${built.root}.`
        );
      }
      verifiedTensorRootCache.set(cacheKey, true);
    }
    return { tensorId: normalizedTensorId, location, expectedRoot };
  }

  async function verifyTensorRange(manifest, shardIndex, offset, length, tensorId) {
    const verified = await verifyTensorRoot(manifest, tensorId);
    if (!isRequestedRangeInsideTensor(verified.location, shardIndex, offset, length)) {
      throw new Error(
        `Requested shard range ${shardIndex}:${offset}+${length ?? 'all'} is outside tensor "${verified.tensorId}".`
      );
    }
  }

  return Object.freeze({ reset, verifyTensorRoot, verifyTensorRange });
}
