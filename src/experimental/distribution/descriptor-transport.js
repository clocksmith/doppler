import {
  P2P_TRANSPORT_ERROR_CODES,
  createP2PTransportError,
} from './p2p-transport-contract.js';

export const DESCRIPTOR_TRANSPORT_CONTRACT_VERSION = 1;

function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be an object.`,
      { label }
    );
  }
  return value;
}

function normalizePositiveInteger(value, label, allowZero = false) {
  const parsed = Number(value);
  const min = allowZero ? 0 : 1;
  if (!Number.isInteger(parsed) || parsed < min) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a ${allowZero ? 'non-negative' : 'positive'} integer.`,
      { label }
    );
  }
  return parsed;
}

function normalizeNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a finite number.`,
      { label }
    );
  }
  return parsed;
}

function normalizeStringArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a non-empty string array.`,
      { label }
    );
  }
  const out = value.map((entry, index) => {
    if (typeof entry !== 'string' || entry.trim().length === 0) {
      throw createP2PTransportError(
        P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
        `${label}[${index}] must be a non-empty string.`,
        { label, index }
      );
    }
    return entry.trim();
  });
  return [...new Set(out)];
}

function normalizeOptionalHash(value, label) {
  if (value == null) return null;
  if (typeof value !== 'string' || !/^sha256:[a-f0-9]{64}$/i.test(value.trim())) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be sha256:<64 hex chars> when provided.`,
      { label }
    );
  }
  return value.trim().toLowerCase();
}

function getComponentShardFile(component, label) {
  const file = component?.shard_file;
  if (typeof file !== 'string' || file.trim().length === 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `descriptorManifest.components.${label}.shard_file is required.`,
      { label }
    );
  }
  return file.trim();
}

function getExpectedShardHash(shardHashes, role, file, component) {
  return normalizeOptionalHash(
    component?.shard_hash
      ?? shardHashes?.[role]
      ?? shardHashes?.[file]
      ?? null,
    `descriptor shard hash ${role}`
  );
}

function getPeerShardCacheEntry(peerDescriptorCache, file) {
  const shards = peerDescriptorCache?.shards;
  if (!shards || typeof shards !== 'object') return null;
  const entry = shards[file];
  if (typeof entry === 'string') {
    return { hash: entry.trim().toLowerCase() };
  }
  if (entry && typeof entry === 'object') {
    return {
      hash: typeof entry.hash === 'string' ? entry.hash.trim().toLowerCase() : null,
    };
  }
  return null;
}

function toByteLength(payload, label) {
  if (payload instanceof ArrayBuffer) return payload.byteLength;
  if (ArrayBuffer.isView(payload)) return payload.byteLength;
  throw createP2PTransportError(
    P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
    `${label} must be ArrayBuffer or Uint8Array.`,
    { label }
  );
}

function normalizeOptionalVramRequirement(value, label) {
  if (value == null) {
    return null;
  }
  return normalizePositiveInteger(value, label);
}

export function normalizePeerCapabilityProfile(value, label = 'peer capability profile') {
  const profile = assertPlainObject(value, label);
  const reliabilityScore = normalizeNumber(profile.reliability_score, `${label}.reliability_score`);
  if (reliabilityScore < 0 || reliabilityScore > 1) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label}.reliability_score must be between 0 and 1.`,
      { label }
    );
  }
  const latencyMs = normalizeNumber(profile.latency_ms, `${label}.latency_ms`);
  if (latencyMs < 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label}.latency_ms must be non-negative.`,
      { label }
    );
  }
  return {
    available_vram_bytes: normalizePositiveInteger(profile.available_vram_bytes, `${label}.available_vram_bytes`),
    backends: normalizeStringArray(profile.backends, `${label}.backends`),
    supported_generators: normalizeStringArray(profile.supported_generators, `${label}.supported_generators`),
    bandwidth_bps: normalizePositiveInteger(profile.bandwidth_bps, `${label}.bandwidth_bps`),
    latency_ms: latencyMs,
    reliability_score: reliabilityScore,
  };
}

export function getDescriptorRequiredGenerators(descriptorManifest) {
  const manifest = assertPlainObject(descriptorManifest, 'descriptorManifest');
  const components = assertPlainObject(manifest.components, 'descriptorManifest.components');
  const generators = [];
  const prngAlgorithm = components.prng_substrate?.algorithm;
  if (typeof prngAlgorithm === 'string' && prngAlgorithm.trim()) {
    generators.push(prngAlgorithm.trim());
  }
  const inrType = components.coordinate_inr?.type;
  if (typeof inrType === 'string' && inrType.trim().toLowerCase() === 'siren') {
    generators.push('siren_f16_v1');
  }
  return [...new Set(generators)];
}

export function assertPeerSupportsDescriptor(peerCapabilityProfile, descriptorManifest) {
  const profile = normalizePeerCapabilityProfile(peerCapabilityProfile);
  const requiredGenerators = getDescriptorRequiredGenerators(descriptorManifest);
  const supported = new Set(profile.supported_generators);
  const missingGenerators = requiredGenerators.filter((generator) => !supported.has(generator));
  if (missingGenerators.length > 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.policyDenied,
      `Peer capability profile is missing descriptor generators: ${missingGenerators.join(', ')}.`,
      { missingGenerators }
    );
  }
  return {
    profile,
    requiredGenerators,
  };
}

export function getDescriptorRequiredShards(descriptorManifest, descriptorShardHashes = {}) {
  const manifest = assertPlainObject(descriptorManifest, 'descriptorManifest');
  const components = assertPlainObject(manifest.components, 'descriptorManifest.components');
  return [
    {
      role: 'kronecker_sum',
      file: getComponentShardFile(components.kronecker_sum, 'kronecker_sum'),
      hash: getExpectedShardHash(descriptorShardHashes, 'kronecker_sum', components.kronecker_sum?.shard_file, components.kronecker_sum),
    },
    {
      role: 'coordinate_inr',
      file: getComponentShardFile(components.coordinate_inr, 'coordinate_inr'),
      hash: getExpectedShardHash(descriptorShardHashes, 'coordinate_inr', components.coordinate_inr?.shard_file, components.coordinate_inr),
    },
    {
      role: 'sparse_outliers',
      file: getComponentShardFile(components.sparse_outliers, 'sparse_outliers'),
      hash: getExpectedShardHash(descriptorShardHashes, 'sparse_outliers', components.sparse_outliers?.shard_file, components.sparse_outliers),
    },
  ];
}

export function negotiateDescriptorShardCache(options = {}) {
  const descriptorManifest = assertPlainObject(options.descriptorManifest, 'descriptorManifest');
  const descriptorHash = normalizeOptionalHash(
    descriptorManifest.descriptor_hash,
    'descriptorManifest.descriptor_hash'
  );
  const peerDescriptorCache = options.peerDescriptorCache && typeof options.peerDescriptorCache === 'object'
    ? options.peerDescriptorCache
    : {};
  const peerDescriptorHash = normalizeOptionalHash(
    peerDescriptorCache.descriptorHash ?? peerDescriptorCache.descriptor_hash ?? null,
    'peerDescriptorCache.descriptorHash'
  );
  const descriptorHashMatches = descriptorHash != null && peerDescriptorHash === descriptorHash;
  const requiredShards = getDescriptorRequiredShards(
    descriptorManifest,
    options.descriptorShardHashes ?? {}
  );
  const missingShards = [];

  for (const shard of requiredShards) {
    const cached = getPeerShardCacheEntry(peerDescriptorCache, shard.file);
    const hashMatches = shard.hash == null || cached?.hash === shard.hash;
    if (!descriptorHashMatches || !cached || !hashMatches) {
      missingShards.push({
        ...shard,
        reason: !descriptorHashMatches
          ? 'descriptor_hash_mismatch'
          : (!cached ? 'not_cached' : 'shard_hash_mismatch'),
      });
    }
  }

  return {
    contractVersion: DESCRIPTOR_TRANSPORT_CONTRACT_VERSION,
    descriptorHash,
    peerDescriptorHash,
    ready: missingShards.length === 0,
    requiredShards,
    missingShards,
  };
}

export function validateActivationTransportPayload(payload, options = {}) {
  const modelDim = normalizePositiveInteger(options.modelDim, 'activation payload modelDim');
  const tokenCount = normalizePositiveInteger(options.tokenCount, 'activation payload tokenCount');
  const bytesPerToken = modelDim * 2;
  const expectedBytes = bytesPerToken * tokenCount;
  const actualBytes = toByteLength(payload, 'activation payload');
  if (actualBytes !== expectedBytes) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `Activation payload byte length mismatch: expected ${expectedBytes}, got ${actualBytes}.`,
      {
        modelDim,
        tokenCount,
        bytesPerToken,
        expectedBytes,
        actualBytes,
      }
    );
  }
  return {
    contractVersion: DESCRIPTOR_TRANSPORT_CONTRACT_VERSION,
    modelDim,
    tokenCount,
    bytesPerToken,
    expectedBytes,
    actualBytes,
  };
}

export function createDescriptorPeerAssignment(options = {}) {
  const descriptorManifest = assertPlainObject(options.descriptorManifest, 'descriptorManifest');
  const support = assertPeerSupportsDescriptor(
    options.peerCapabilityProfile,
    descriptorManifest
  );
  const cache = negotiateDescriptorShardCache({
    descriptorManifest,
    descriptorShardHashes: options.descriptorShardHashes ?? {},
    peerDescriptorCache: options.peerDescriptorCache ?? null,
  });
  const activation = options.activationPayload == null
    ? null
    : validateActivationTransportPayload(options.activationPayload, {
      modelDim: options.modelDim,
      tokenCount: options.tokenCount,
    });
  const requiredVramBytes = normalizeOptionalVramRequirement(
    options.requiredVramBytes,
    'descriptor assignment requiredVramBytes'
  );
  const blockers = [];

  if (!cache.ready) {
    blockers.push({
      code: 'descriptor_shards_missing',
      missingShards: cache.missingShards,
    });
  }
  if (requiredVramBytes != null && requiredVramBytes > support.profile.available_vram_bytes) {
    blockers.push({
      code: 'insufficient_vram',
      requiredVramBytes,
      availableVramBytes: support.profile.available_vram_bytes,
    });
  }

  if (options.failClosed === true && blockers.length > 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.policyDenied,
      `Descriptor peer assignment blocked: ${blockers.map((blocker) => blocker.code).join(', ')}.`,
      { blockers }
    );
  }

  return {
    contractVersion: DESCRIPTOR_TRANSPORT_CONTRACT_VERSION,
    assignable: blockers.length === 0,
    blockers,
    profile: support.profile,
    requiredGenerators: support.requiredGenerators,
    cache,
    requiredDownloads: cache.missingShards,
    activation,
    requiredVramBytes,
  };
}
