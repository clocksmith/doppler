import { Q4K_BLOCK_BYTES, QK_K } from '../../config/schema/quantization.schema.js';

const SUPPORTED_PACKINGS = new Set(['dense', 'q4k', 'q4_0', 'gguf-block-v2']);
const Q4_0_BLOCK_BYTES = 18;
const Q4_0_BLOCK_VALUES = 32;

function fail(label, message) {
  throw new Error(`[RDRRStorage] ${label}: ${message}`);
}

function normalizePositiveInteger(value, label, field) {
  const normalized = Number(value);
  if (!Number.isInteger(normalized) || normalized <= 0) {
    fail(label, `invalid ${field}`);
  }
  return normalized;
}

function normalizeNonNegativeInteger(value, label, field) {
  const normalized = Number(value);
  if (!Number.isInteger(normalized) || normalized < 0) {
    fail(label, `invalid ${field}`);
  }
  return normalized;
}

function normalizePacking(packing, label) {
  const normalized = String(packing || '').trim().toLowerCase();
  if (!SUPPORTED_PACKINGS.has(normalized)) {
    fail(label, `unsupported packing "${packing}"`);
  }
  return normalized;
}

function normalizeCompanions(companions, label) {
  if (companions === undefined) {
    return [];
  }
  if (!Array.isArray(companions)) {
    fail(label, 'companions must be an array');
  }
  return companions.map((companion, index) => {
    if (!companion || typeof companion !== 'object' || Array.isArray(companion)) {
      fail(label, `companions[${index}] must be an object`);
    }
    if (typeof companion.role !== 'string' || companion.role.trim().length === 0) {
      fail(label, `companions[${index}].role must be non-empty`);
    }
    if (typeof companion.tensorId !== 'string' || companion.tensorId.trim().length === 0) {
      fail(label, `companions[${index}].tensorId must be non-empty`);
    }
    return {
      role: companion.role.trim(),
      tensorId: companion.tensorId.trim(),
    };
  });
}

function normalizeShardSpans(shardSpans, label) {
  if (shardSpans === undefined) {
    return undefined;
  }
  if (!Array.isArray(shardSpans) || shardSpans.length === 0) {
    fail(label, 'shardSpans must be a non-empty array when present');
  }
  return shardSpans.map((span, index) => {
    if (!span || typeof span !== 'object' || Array.isArray(span)) {
      fail(label, `shardSpans[${index}] must be an object`);
    }
    const shardIndex = normalizeNonNegativeInteger(span.shardIndex ?? span.shard, label, `shardSpans[${index}].shardIndex`);
    const byteStart = normalizeNonNegativeInteger(span.byteStart ?? span.offset, label, `shardSpans[${index}].byteStart`);
    const rawEnd = span.byteEnd ?? (
      span.size == null
        ? undefined
        : byteStart + normalizePositiveInteger(span.size, label, `shardSpans[${index}].size`)
    );
    const byteEnd = normalizePositiveInteger(rawEnd, label, `shardSpans[${index}].byteEnd`);
    if (byteEnd <= byteStart) {
      fail(label, `shardSpans[${index}] has non-positive length`);
    }
    return { shardIndex, byteStart, byteEnd };
  });
}

function normalizeBlockShape(blockShape, label, packing) {
  if (packing === 'dense') {
    return blockShape == null
      ? undefined
      : (Array.isArray(blockShape)
        ? blockShape.map((value, index) => normalizePositiveInteger(value, label, `blockShape[${index}]`))
        : fail(label, 'blockShape must be an array'));
  }
  if (!Array.isArray(blockShape) || blockShape.length === 0) {
    fail(label, `packing "${packing}" requires blockShape`);
  }
  return blockShape.map((value, index) => normalizePositiveInteger(value, label, `blockShape[${index}]`));
}

export function normalizeTensorStorageDescriptor(storage, label) {
  if (storage == null) {
    return null;
  }
  if (typeof storage !== 'object' || Array.isArray(storage)) {
    fail(label, 'storage must be an object');
  }
  const packing = normalizePacking(storage.packing, label);
  const blockShape = normalizeBlockShape(storage.blockShape, label, packing);
  const companions = normalizeCompanions(storage.companions, label);
  const shardSpans = normalizeShardSpans(storage.shardSpans, label);
  const blockBytes = storage.blockBytes == null
    ? undefined
    : normalizePositiveInteger(storage.blockBytes, label, 'blockBytes');
  if (packing === 'gguf-block-v2' && blockBytes == null) {
    fail(label, 'packing "gguf-block-v2" requires blockBytes');
  }
  return {
    ...storage,
    packing,
    ...(blockShape ? { blockShape } : {}),
    ...(blockBytes == null ? {} : { blockBytes }),
    ...(companions.length === 0 ? {} : { companions }),
    ...(shardSpans ? { shardSpans } : {}),
  };
}

export function validateTensorStorageDescriptor(storage, label, errors) {
  try {
    normalizeTensorStorageDescriptor(storage, label);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
  }
}

export function getTensorPhysicalSpans(location, label = 'tensor') {
  const storage = normalizeTensorStorageDescriptor(location?.storage ?? null, label);
  if (storage?.shardSpans) {
    return storage.shardSpans;
  }
  if (Array.isArray(location?.spans) && location.spans.length > 0) {
    return location.spans.map((span, index) => {
      const shardIndex = normalizeNonNegativeInteger(span.shardIndex ?? span.shard, label, `spans[${index}].shardIndex`);
      const byteStart = normalizeNonNegativeInteger(span.offset, label, `spans[${index}].offset`);
      const size = normalizePositiveInteger(span.size, label, `spans[${index}].size`);
      return { shardIndex, byteStart, byteEnd: byteStart + size };
    });
  }
  const shardIndex = location?.shardIndex ?? location?.shard;
  if (Number.isInteger(shardIndex) && Number.isInteger(location?.offset) && Number.isInteger(location?.size)) {
    return [{
      shardIndex,
      byteStart: location.offset,
      byteEnd: location.offset + location.size,
    }];
  }
  fail(label, 'missing physical shard spans');
}

export function getPackingByteLayout(storage, tensor) {
  const descriptor = normalizeTensorStorageDescriptor(storage ?? { packing: 'dense' }, tensor?.name ?? 'tensor');
  const blockElementCount = descriptor.blockShape == null
    ? 1
    : descriptor.blockShape.reduce((product, value) => product * value, 1);
  if (descriptor.packing === 'dense') {
    return {
      blockElementCount,
      blockBytes: null,
    };
  }
  if (descriptor.packing === 'q4k') {
    return {
      blockElementCount: blockElementCount || QK_K,
      blockBytes: descriptor.blockBytes ?? Q4K_BLOCK_BYTES,
    };
  }
  if (descriptor.packing === 'q4_0') {
    return {
      blockElementCount: blockElementCount || Q4_0_BLOCK_VALUES,
      blockBytes: descriptor.blockBytes ?? Q4_0_BLOCK_BYTES,
    };
  }
  return {
    blockElementCount,
    blockBytes: descriptor.blockBytes,
  };
}

