import { computeNamespacedCanonicalSha256, hashBytesSha256 } from '../../utils/canonical-hash.js';
import { buildMerkleTreeFromLeafHashes, DEFAULT_MERKLE_BLOCK_SIZE } from './merkle.js';

export { DEFAULT_MERKLE_BLOCK_SIZE } from './merkle.js';

function asPositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`[RDRRIntegrity] ${label} must be a positive integer`);
  }
  return parsed;
}

function normalizeShardIndex(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`[RDRRIntegrity] ${label} must be a non-negative integer`);
  }
  return parsed;
}

function normalizePrimarySpans(location, tensorId) {
  if (!location || typeof location !== 'object') {
    throw new Error(`[RDRRIntegrity] tensor ${tensorId} is missing a location object`);
  }

  if (Array.isArray(location.spans) && location.spans.length > 0) {
    return location.spans.map((span, index) => {
      if (!span || typeof span !== 'object') {
        throw new Error(`[RDRRIntegrity] tensor ${tensorId} span ${index} must be an object`);
      }
      const shardIndex = normalizeShardIndex(
        span.shardIndex ?? span.shard,
        `tensor ${tensorId} span ${index}.shardIndex`
      );
      const byteStart = asPositiveInteger(
        (span.offset ?? 0) + 1,
        `tensor ${tensorId} span ${index}.offset`
      ) - 1;
      const size = asPositiveInteger(span.size, `tensor ${tensorId} span ${index}.size`);
      return {
        shardIndex,
        byteStart,
        size,
      };
    });
  }

  const shardIndex = normalizeShardIndex(
    location.shardIndex ?? location.shard,
    `tensor ${tensorId}.shardIndex`
  );
  const byteStart = asPositiveInteger(
    (location.offset ?? 0) + 1,
    `tensor ${tensorId}.offset`
  ) - 1;
  const size = asPositiveInteger(location.size, `tensor ${tensorId}.size`);
  return [{
    shardIndex,
    byteStart,
    size,
  }];
}

function resolveHashBlockBytes(options) {
  return typeof options?.hashBlockBytesSha256 === 'function'
    ? options.hashBlockBytesSha256
    : hashBytesSha256;
}

function finalizeLeafHashes(state, hashBlockBytes) {
  if (state.totalBytes === 0) {
    return [hashBlockBytes(new Uint8Array(0))];
  }
  if (state.blockFill === 0) {
    return state.leafHashes;
  }
  state.leafHashes.push(hashBlockBytes(state.blockBuffer.subarray(0, state.blockFill)));
  return state.leafHashes;
}

async function readLeafHashesForTensor(location, tensorId, options) {
  const readShardRange = options?.readShardRange;
  if (typeof readShardRange !== 'function') {
    throw new Error('[RDRRIntegrity] readShardRange(shardIndex, offset, length) is required');
  }
  const blockSize = asPositiveInteger(
    options?.blockSize ?? DEFAULT_MERKLE_BLOCK_SIZE,
    'blockSize'
  );
  const spans = normalizePrimarySpans(location, tensorId);
  const hashBlockBytes = resolveHashBlockBytes(options);
  const state = {
    blockBuffer: new Uint8Array(blockSize),
    blockFill: 0,
    leafHashes: [],
    totalBytes: 0,
  };

  for (const span of spans) {
    let cursor = 0;
    while (cursor < span.size) {
      const take = Math.min(blockSize - state.blockFill, span.size - cursor);
      const chunk = await readShardRange(
        span.shardIndex,
        span.byteStart + cursor,
        take,
        { tensorId }
      );
      const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
      if (bytes.byteLength !== take) {
        throw new Error(
          `[RDRRIntegrity] expected ${take} bytes for ${tensorId} from shard ${span.shardIndex}, got ${bytes.byteLength}`
        );
      }
      state.blockBuffer.set(bytes, state.blockFill);
      state.blockFill += take;
      state.totalBytes += take;
      cursor += take;

      if (state.blockFill === blockSize) {
        state.leafHashes.push(hashBlockBytes(state.blockBuffer));
        state.blockBuffer = new Uint8Array(blockSize);
        state.blockFill = 0;
      }
    }
  }

  return {
    blockSize,
    leafHashes: finalizeLeafHashes(state, hashBlockBytes),
    totalBytes: state.totalBytes,
  };
}

export async function buildTensorBlockMerkleRoot(tensorId, location, options = {}) {
  const normalizedTensorId = String(tensorId || '').trim();
  if (!normalizedTensorId) {
    throw new Error('[RDRRIntegrity] tensorId is required');
  }
  const { blockSize, leafHashes, totalBytes } = await readLeafHashesForTensor(
    location,
    normalizedTensorId,
    options
  );
  const tree = buildMerkleTreeFromLeafHashes(leafHashes, { blockSize });
  return {
    tensorId: normalizedTensorId,
    blockSize,
    blockCount: tree.blockCount,
    totalBytes,
    root: tree.root,
  };
}

export async function buildIntegrityExtensions(manifest, options = {}) {
  const tensorMap = options?.tensorMap ?? manifest?.tensors;
  if (!tensorMap || typeof tensorMap !== 'object' || Array.isArray(tensorMap)) {
    throw new Error('[RDRRIntegrity] buildIntegrityExtensions requires a tensor map object');
  }

  const blockSize = asPositiveInteger(
    options?.blockSize
      ?? manifest?.integrityExtensions?.blockMerkle?.blockSize
      ?? DEFAULT_MERKLE_BLOCK_SIZE,
    'blockSize'
  );
  const roots = {};
  const entries = Object.entries(tensorMap).sort(([left], [right]) => left.localeCompare(right));
  for (let index = 0; index < entries.length; index += 1) {
    const [tensorId, location] = entries[index];
    options?.onProgress?.({
      tensorId,
      current: index + 1,
      total: entries.length,
    });
    const root = await buildTensorBlockMerkleRoot(tensorId, location, {
      ...options,
      blockSize,
    });
    roots[tensorId] = root.root;
  }

  const integrityExtensions = {
    contractVersion: 1,
    blockMerkle: {
      blockSize,
      roots,
    },
  };

  return {
    integrityExtensions,
    integrityExtensionsHash: computeNamespacedCanonicalSha256('integrity', integrityExtensions),
  };
}
