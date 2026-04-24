import { getTensorPhysicalSpans, getPackingByteLayout, normalizeTensorStorageDescriptor } from './storage-descriptor.js';

function dtypeBytes(dtype, label) {
  const normalized = String(dtype || '').trim().toUpperCase();
  if (normalized === 'F16' || normalized === 'BF16') return 2;
  if (normalized === 'F32' || normalized === 'I32' || normalized === 'U32') return 4;
  if (normalized === 'I16' || normalized === 'U16') return 2;
  if (normalized === 'I8' || normalized === 'U8') return 1;
  throw new Error(`[RDRRSlice] ${label}: unsupported dtype "${dtype}"`);
}

function normalizeRange(shape, axis, start, end, label) {
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new Error(`[RDRRSlice] ${label}: tensor is missing shape`);
  }
  if (!Number.isInteger(axis) || axis < 0 || axis >= shape.length) {
    throw new Error(`[RDRRSlice] ${label}: invalid axis ${axis}`);
  }
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > shape[axis]) {
    throw new Error(`[RDRRSlice] ${label}: invalid range [${start}, ${end})`);
  }
  return { axis, start, end };
}

function mapPhysicalRangesToShardRanges(spans, ranges) {
  const totalLength = spans.reduce((sum, span) => sum + (span.byteEnd - span.byteStart), 0);
  const results = [];
  for (const range of ranges) {
    if (range.end <= range.start) {
      continue;
    }
    if (range.start < 0 || range.end > totalLength) {
      throw new Error(`[RDRRSlice] physical range [${range.start}, ${range.end}) exceeds tensor storage`);
    }
    let cursor = 0;
    let remainingStart = range.start;
    let remainingEnd = range.end;
    for (const span of spans) {
      const spanLength = span.byteEnd - span.byteStart;
      const spanGlobalStart = cursor;
      const spanGlobalEnd = cursor + spanLength;
      cursor = spanGlobalEnd;
      if (remainingEnd <= spanGlobalStart || remainingStart >= spanGlobalEnd) {
        continue;
      }
      const sliceStart = Math.max(remainingStart, spanGlobalStart);
      const sliceEnd = Math.min(remainingEnd, spanGlobalEnd);
      results.push({
        shardIndex: span.shardIndex,
        byteStart: span.byteStart + (sliceStart - spanGlobalStart),
        byteEnd: span.byteStart + (sliceEnd - spanGlobalStart),
      });
    }
  }
  return results;
}

function getTensorShape(tensor) {
  return Array.isArray(tensor?.originalShape) && tensor.originalShape.length > 0
    ? tensor.originalShape
    : tensor?.shape;
}

function compileDenseRanges(tensor, range, label) {
  const shape = getTensorShape(tensor);
  const elementBytes = dtypeBytes(tensor?.dtype, label);
  if (shape.length === 1) {
    if (range.axis !== 0) {
      throw new Error(`[RDRRSlice] ${label}: axis ${range.axis} unsupported for 1D tensor`);
    }
    return [{
      start: range.start * elementBytes,
      end: range.end * elementBytes,
    }];
  }
  if (shape.length !== 2) {
    throw new Error(`[RDRRSlice] ${label}: dense slice compiler currently supports 1D or 2D tensors only`);
  }
  const [rows, cols] = shape;
  const rowBytes = cols * elementBytes;
  if (range.axis === 0) {
    return [{
      start: range.start * rowBytes,
      end: range.end * rowBytes,
    }];
  }
  if (range.axis !== 1) {
    throw new Error(`[RDRRSlice] ${label}: unsupported axis ${range.axis}`);
  }
  const ranges = [];
  for (let row = 0; row < rows; row += 1) {
    ranges.push({
      start: row * rowBytes + (range.start * elementBytes),
      end: row * rowBytes + (range.end * elementBytes),
    });
  }
  return ranges;
}

function compileBlocked2DRanges(tensor, range, label) {
  const shape = getTensorShape(tensor);
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error(`[RDRRSlice] ${label}: packed slice compiler currently supports 2D tensors only`);
  }
  const [rows, cols] = shape;
  const descriptor = normalizeTensorStorageDescriptor(tensor?.storage, label);
  const layout = getPackingByteLayout(descriptor, tensor);
  const blockValues = layout.blockElementCount;
  const blockBytes = layout.blockBytes;
  const blocksPerRow = Math.ceil(cols / blockValues);
  const rowBytes = blocksPerRow * blockBytes;
  if (range.axis === 0) {
    return [{
      start: range.start * rowBytes,
      end: range.end * rowBytes,
    }];
  }
  if (range.axis !== 1) {
    throw new Error(`[RDRRSlice] ${label}: unsupported axis ${range.axis}`);
  }
  const blockStart = Math.floor(range.start / blockValues);
  const blockEnd = Math.ceil(range.end / blockValues);
  const ranges = [];
  for (let row = 0; row < rows; row += 1) {
    ranges.push({
      start: row * rowBytes + (blockStart * blockBytes),
      end: row * rowBytes + (blockEnd * blockBytes),
    });
  }
  return ranges;
}

function compileFullTensorRanges(tensor, label) {
  const spans = getTensorPhysicalSpans(tensor, label);
  return spans.map((span) => ({
    shardIndex: span.shardIndex,
    byteStart: span.byteStart,
    byteEnd: span.byteEnd,
  }));
}

export function compileTensorSlice(options) {
  const tensorMap = options?.tensorMap;
  const tensorId = String(options?.tensorId || '').trim();
  const axis = Number(options?.axis);
  const rangeStart = Number(options?.rangeStart);
  const rangeEnd = Number(options?.rangeEnd);
  if (!tensorMap || typeof tensorMap !== 'object') {
    throw new Error('[RDRRSlice] tensorMap is required');
  }
  if (!tensorId) {
    throw new Error('[RDRRSlice] tensorId is required');
  }
  const tensor = tensorMap[tensorId];
  if (!tensor) {
    throw new Error(`[RDRRSlice] tensor "${tensorId}" not found`);
  }
  const label = `tensor "${tensorId}"`;
  const shape = getTensorShape(tensor);
  const range = normalizeRange(shape, axis, rangeStart, rangeEnd, label);
  const storage = normalizeTensorStorageDescriptor(tensor?.storage ?? { packing: 'dense' }, label);
  const spans = getTensorPhysicalSpans(tensor, label);
  const physicalRanges = storage.packing === 'dense'
    ? compileDenseRanges(tensor, range, label)
    : compileBlocked2DRanges(tensor, range, label);
  const byteRanges = mapPhysicalRangesToShardRanges(spans, physicalRanges);
  const companionRanges = [];
  for (const companion of (storage.companions ?? [])) {
    const companionTensor = tensorMap[companion.tensorId];
    if (!companionTensor) {
      throw new Error(`[RDRRSlice] ${label}: missing companion tensor "${companion.tensorId}" for role "${companion.role}"`);
    }
    companionRanges.push({
      role: companion.role,
      tensorId: companion.tensorId,
      byteRanges: compileFullTensorRanges(companionTensor, `companion "${companion.tensorId}"`),
    });
  }
  return {
    tensorId,
    axis: range.axis,
    rangeStart: range.start,
    rangeEnd: range.end,
    byteRanges,
    companionRanges,
  };
}

