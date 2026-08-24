import { float32ToFloat16 } from '../../converter/quantizer.js';
import { normalizeTensorSourceTransform } from '../../formats/rdrr/source-transform-contract.js';
import {
  computeElementCount,
  computeSourceByteLength,
  getLiteRTCompanionByteLength,
  getPackedValuesPerByte,
  validateLiteRTStorageEncoding,
  validateLiteRTStorageLaneOrder,
  validateLiteRTTransformTarget,
} from './source-transform/plan.js';
import {
  computeStoredQuantizedSum,
  readQuantizedValue,
  readStoredQuantizedValue,
  resolveLiteRTScaleValue,
} from './source-transform/executor.js';

export function getSourceTransformSpec(location, tensorName) {
  const transform = normalizeTensorSourceTransform(location, tensorName);
  if (!transform) {
    return null;
  }
  if (transform.kind === 'affine_dequant') {
    const elementCount = computeElementCount(location.shape, tensorName);
    return {
      transform,
      kind: 'affine_dequant',
      elementCount,
      outputByteLength: elementCount * 2,
      sourceByteLength: computeSourceByteLength(
        elementCount,
        transform.sourceDtype,
        tensorName
      ),
    };
  }
  if (!Array.isArray(location?.shape) || location.shape.length !== 2) {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" LiteRT sourceTransform requires a 2D shape.`
    );
  }
  const rows = Number(location.shape[0]);
  const cols = Number(location.shape[1]);
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(cols) || cols <= 0) {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT shape ${JSON.stringify(location.shape)}.`
    );
  }

  if (transform.kind === 'litert_rowwise_dequant') {
    if (transform.scheme !== 'per_row_affine') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has unsupported sourceTransform.scheme "${transform.scheme}".`
      );
    }
    validateLiteRTTransformTarget(location, tensorName, transform, 'LiteRT row-wise sourceTransform');
    validateLiteRTStorageEncoding(transform.storageEncoding, tensorName);
    const rawRowBytes = computeSourceByteLength(cols, transform.sourceDtype, tensorName);
    const sourceByteLength = rows * rawRowBytes;
    const scaleSource = transform.scaleSource;
    if (!scaleSource || typeof scaleSource !== 'object') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise transform is missing scaleSource.`
      );
    }
    getLiteRTCompanionByteLength(scaleSource, tensorName, 'row-wise scale', rows * 4);
    if (transform.rowSumSource != null && typeof transform.rowSumSource !== 'object') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise transform has invalid rowSumSource.`
      );
    }
    if (transform.rowSumSource != null) {
      getLiteRTCompanionByteLength(transform.rowSumSource, tensorName, 'row-wise row-sum', rows * 4);
    }
    if (transform.scaleSemantics !== 'step' && transform.scaleSemantics !== 'qmax_abs') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise transform has unsupported scaleSemantics "${transform.scaleSemantics}".`
      );
    }
    if (transform.scaleSemantics === 'qmax_abs') {
      const scaleDivisor = Number(transform.scaleDivisor);
      if (!Number.isFinite(scaleDivisor) || scaleDivisor <= 0) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise transform requires scaleDivisor > 0 for scaleSemantics="qmax_abs".`
        );
      }
    }
    return {
      transform,
      kind: 'litert_rowwise_dequant',
      rows,
      cols,
      rawRowBytes,
      scaleRowBytes: 4,
      targetRowBytes: cols * 2,
      outputByteLength: rows * cols * 2,
      sourceByteLength,
      storageValuesPerByte: getPackedValuesPerByte(transform.sourceDtype, tensorName),
    };
  }

  if (transform.kind === 'litert_axis_dequant') {
    if (transform.scheme !== 'per_axis_affine') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has unsupported sourceTransform.scheme "${transform.scheme}".`
      );
    }
    validateLiteRTTransformTarget(location, tensorName, transform, 'LiteRT axis sourceTransform');
    validateLiteRTStorageEncoding(transform.storageEncoding, tensorName);
    if (!Array.isArray(transform.storageShape) || transform.storageShape.length !== 2) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform requires storageShape=[rows, cols].`
      );
    }
    const storageRows = Number(transform.storageShape[0]);
    const storageCols = Number(transform.storageShape[1]);
    if (
      !Number.isInteger(storageRows)
      || storageRows <= 0
      || !Number.isInteger(storageCols)
      || storageCols <= 0
    ) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT storageShape ${JSON.stringify(transform.storageShape)}.`
      );
    }
    if (transform.quantAxis !== 0 && transform.quantAxis !== 1) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT quantAxis ${transform.quantAxis}.`
      );
    }
    if (transform.quantAxis === 0) {
      if (storageRows !== cols || storageCols !== rows) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform expects storageShape ` +
          `[${cols}, ${rows}] for quantAxis=0, got [${storageRows}, ${storageCols}].`
        );
      }
    } else if (storageRows !== rows || storageCols !== cols) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform expects storageShape ` +
        `[${rows}, ${cols}] for quantAxis=1, got [${storageRows}, ${storageCols}].`
      );
    }
    const scaleSource = transform.scaleSource;
    if (!scaleSource || typeof scaleSource !== 'object') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform is missing scaleSource.`
      );
    }
    getLiteRTCompanionByteLength(scaleSource, tensorName, 'axis scale', rows * 4);
    if (transform.sumSource != null && typeof transform.sumSource !== 'object') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform has invalid sumSource.`
      );
    }
    if (transform.sumSource != null) {
      getLiteRTCompanionByteLength(transform.sumSource, tensorName, 'axis sum', rows * 4);
    }
    if (transform.scaleSemantics !== 'step' && transform.scaleSemantics !== 'qmax_abs') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform has unsupported scaleSemantics "${transform.scaleSemantics}".`
      );
    }
    if (transform.scaleSemantics === 'qmax_abs') {
      const scaleDivisor = Number(transform.scaleDivisor);
      if (!Number.isFinite(scaleDivisor) || scaleDivisor <= 0) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform requires scaleDivisor > 0 for scaleSemantics="qmax_abs".`
        );
      }
    }
    const scaleCompanionDtype = String(transform?.scaleCompanionDtype || '').toUpperCase();
    const scaleCompanionDequant = scaleCompanionDtype === 'UINT8'
      ? transform.scaleCompanionDequant
      : null;
    if (scaleCompanionDtype === 'UINT8') {
      if (!scaleCompanionDequant || typeof scaleCompanionDequant !== 'object') {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform has UINT8 scale companion without scaleCompanionDequant metadata.`
        );
      }
      const companionScale = Number(scaleCompanionDequant.scale);
      const companionZeroPoint = Number(scaleCompanionDequant.zeroPoint);
      if (!Number.isFinite(companionScale) || companionScale <= 0) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform has invalid scaleCompanionDequant.scale ${scaleCompanionDequant.scale}.`
        );
      }
      if (!Number.isSafeInteger(companionZeroPoint)) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform has invalid scaleCompanionDequant.zeroPoint ${scaleCompanionDequant.zeroPoint}.`
        );
      }
    } else if (scaleCompanionDtype !== '') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis transform has unsupported scaleCompanionDtype "${scaleCompanionDtype}".`
      );
    }
    const rawStorageRowBytes = computeSourceByteLength(storageCols, transform.sourceDtype, tensorName);
    const sourceByteLength = computeSourceByteLength(
      storageRows * storageCols,
      transform.sourceDtype,
      tensorName
    );
    return {
      transform,
      kind: 'litert_axis_dequant',
      rows,
      cols,
      storageRows,
      storageCols,
      rawStorageRowBytes,
      scaleRowBytes: scaleCompanionDtype === 'UINT8' ? 4 : 4,
      targetRowBytes: cols * 2,
      outputByteLength: rows * cols * 2,
      sourceByteLength,
      quantAxis: transform.quantAxis,
      storageValuesPerByte: getPackedValuesPerByte(transform.sourceDtype, tensorName),
    };
  }

  if (transform.kind === 'litert_axis_blocked_dequant') {
    if (transform.scheme !== 'per_axis_affine') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has unsupported sourceTransform.scheme "${transform.scheme}".`
      );
    }
    validateLiteRTTransformTarget(location, tensorName, transform, 'LiteRT blocked axis sourceTransform');
    validateLiteRTStorageEncoding(transform.storageEncoding, tensorName);
    if (!Array.isArray(transform.storageShape) || transform.storageShape.length !== 2) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis transform requires storageShape=[rows, cols].`
      );
    }
    const storageRows = Number(transform.storageShape[0]);
    const storageCols = Number(transform.storageShape[1]);
    const storageBlockSize = Number(transform.storageBlockSize);
    if (
      !Number.isInteger(storageRows)
      || storageRows <= 0
      || !Number.isInteger(storageCols)
      || storageCols <= 0
      || !Number.isInteger(storageBlockSize)
      || storageBlockSize <= 0
    ) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT blocked storage shape ` +
        `${JSON.stringify({ storageShape: transform.storageShape, storageBlockSize: transform.storageBlockSize })}.`
      );
    }
    if (transform.quantAxis !== 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis transform only supports quantAxis=0.`
      );
    }
    validateLiteRTStorageLaneOrder(transform.storageLaneOrder, storageBlockSize, tensorName);
    if (storageCols !== rows || storageRows * storageBlockSize !== cols) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis transform expects storageShape ` +
        `[${cols / storageBlockSize}, ${rows}] for blockSize=${storageBlockSize}, got [${storageRows}, ${storageCols}].`
      );
    }
    const scaleSource = transform.scaleSource;
    if (!scaleSource || typeof scaleSource !== 'object') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis transform is missing scaleSource.`
      );
    }
    getLiteRTCompanionByteLength(scaleSource, tensorName, 'blocked axis scale', rows * 4);
    if (transform.sumSource != null && typeof transform.sumSource !== 'object') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis transform has invalid sumSource.`
      );
    }
    if (transform.sumSource != null) {
      getLiteRTCompanionByteLength(transform.sumSource, tensorName, 'blocked axis sum', rows * 4);
    }
    if (transform.scaleSemantics !== 'step' && transform.scaleSemantics !== 'qmax_abs') {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis transform has unsupported scaleSemantics "${transform.scaleSemantics}".`
      );
    }
    if (transform.scaleSemantics === 'qmax_abs') {
      const scaleDivisor = Number(transform.scaleDivisor);
      if (!Number.isFinite(scaleDivisor) || scaleDivisor <= 0) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis transform requires scaleDivisor > 0 for scaleSemantics="qmax_abs".`
        );
      }
    }
    const storageElementBytes = computeSourceByteLength(storageBlockSize, transform.sourceDtype, tensorName);
    const rawStorageRowBytes = storageCols * storageElementBytes;
    return {
      transform,
      kind: 'litert_axis_blocked_dequant',
      rows,
      cols,
      storageRows,
      storageCols,
      rawStorageRowBytes,
      storageElementBytes,
      scaleRowBytes: 4,
      targetRowBytes: cols * 2,
      outputByteLength: rows * cols * 2,
      sourceByteLength: storageRows * rawStorageRowBytes,
      quantAxis: transform.quantAxis,
      storageBlockSize,
      storageLaneOrder: transform.storageLaneOrder.map((value) => Number(value)),
    };
  }

  throw new Error(
    `[DopplerLoader] Tensor "${tensorName}" has unsupported sourceTransform.kind "${transform.kind}".`
  );
}

export function hasSourceTransform(location) {
  return Boolean(location?.sourceTransform && typeof location.sourceTransform === 'object');
}

export function materializeTensorSourceTransform(rawBytes, location, tensorName, options = {}) {
  if (!(rawBytes instanceof Uint8Array)) {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" sourceTransform requires Uint8Array input bytes.`
    );
  }
  if (!hasSourceTransform(location)) {
    return rawBytes;
  }

  const spec = getSourceTransformSpec(location, tensorName);
  const transform = spec.transform;
  if (transform.kind === 'litert_rowwise_dequant') {
    const scaleBytes = options.scaleBytes;
    if (!(scaleBytes instanceof Uint8Array)) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise sourceTransform requires scaleBytes.`
      );
    }
    const rowSumBytes = options.rowSumBytes ?? null;
    const hasRowSumSource = transform.rowSumSource != null;
    if (hasRowSumSource && !(rowSumBytes instanceof Uint8Array)) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise sourceTransform requires rowSumBytes.`
      );
    }
    const rowStart = options.rowStart == null ? 0 : Number(options.rowStart);
    const rowCount = options.rowCount == null ? spec.rows : Number(options.rowCount);
    if (!Number.isInteger(rowStart) || rowStart < 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT rowStart ${options.rowStart}.`
      );
    }
    if (!Number.isInteger(rowCount) || rowCount <= 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT rowCount ${options.rowCount}.`
      );
    }
    if (rowStart + rowCount > spec.rows) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row range (${rowStart}..${rowStart + rowCount}) exceeds ${spec.rows} rows.`
      );
    }
    const expectedRawBytes = rowCount * spec.rawRowBytes;
    const expectedScaleBytes = rowCount * spec.scaleRowBytes;
    if (rawBytes.byteLength !== expectedRawBytes) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise raw byte size mismatch. ` +
        `Expected ${expectedRawBytes}, got ${rawBytes.byteLength}.`
      );
    }
    if (scaleBytes.byteLength !== expectedScaleBytes) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise scale byte size mismatch. ` +
        `Expected ${expectedScaleBytes}, got ${scaleBytes.byteLength}.`
      );
    }
    if (hasRowSumSource && rowSumBytes.byteLength !== rowCount * 4) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT row-wise row-sum byte size mismatch. ` +
        `Expected ${rowCount * 4}, got ${rowSumBytes.byteLength}.`
      );
    }
    const scaleView = new DataView(scaleBytes.buffer, scaleBytes.byteOffset, scaleBytes.byteLength);
    const rowSumView = hasRowSumSource
      ? new DataView(rowSumBytes.buffer, rowSumBytes.byteOffset, rowSumBytes.byteLength)
      : null;
    const out = new Uint16Array(rowCount * spec.cols);
    for (let row = 0; row < rowCount; row++) {
      const scale = scaleView.getFloat32(row * spec.scaleRowBytes, true);
      if (!Number.isFinite(scale) || scale <= 0) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT row ${rowStart + row} has invalid scale ${scale}.`
        );
      }
      const resolvedScale = resolveLiteRTScaleValue(
        scale,
        transform,
        tensorName,
        `LiteRT row ${rowStart + row}`
      );
      const rawRowOffset = row * spec.rawRowBytes;
      const outRowOffset = row * spec.cols;
      const rowBytes = rawBytes.subarray(rawRowOffset, rawRowOffset + spec.rawRowBytes);
      const rowZeroPoint = rowSumView
        ? (
          computeStoredQuantizedSum(
            rowBytes,
            transform.sourceDtype,
            transform.storageEncoding
          ) - rowSumView.getInt32(row * 4, true)
        ) / spec.cols
        : null;
      if (rowZeroPoint != null && !Number.isFinite(rowZeroPoint)) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT row ${rowStart + row} has invalid derived rowZeroPoint ${rowZeroPoint}.`
        );
      }
      for (let col = 0; col < spec.cols; col++) {
        const quantized = rowZeroPoint == null
          ? readQuantizedValue(
            rowBytes,
            col,
            transform.sourceDtype,
            transform.storageEncoding
          )
          : (
            readStoredQuantizedValue(
              rowBytes,
              col,
              transform.sourceDtype,
              transform.storageEncoding
            ) - rowZeroPoint
            );
        out[outRowOffset + col] = float32ToFloat16(quantized * resolvedScale);
      }
    }
    return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
  }

  if (transform.kind === 'litert_axis_dequant') {
    const scaleBytes = options.scaleBytes;
    if (!(scaleBytes instanceof Uint8Array)) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis sourceTransform requires scaleBytes.`
      );
    }
    const sumBytes = options.sumBytes ?? null;
    const hasSumSource = transform.sumSource != null;
    if (hasSumSource && !(sumBytes instanceof Uint8Array)) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis sourceTransform requires sumBytes.`
      );
    }
    const rowStart = options.rowStart == null ? 0 : Number(options.rowStart);
    const rowCount = options.rowCount == null ? spec.rows : Number(options.rowCount);
    if (!Number.isInteger(rowStart) || rowStart < 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT axis rowStart ${options.rowStart}.`
      );
    }
    if (!Number.isInteger(rowCount) || rowCount <= 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT axis rowCount ${options.rowCount}.`
      );
    }
    if (rowStart + rowCount > spec.rows) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis row range (${rowStart}..${rowStart + rowCount}) exceeds ${spec.rows} rows.`
      );
    }
    if (scaleBytes.byteLength !== rowCount * spec.scaleRowBytes) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis scale byte size mismatch. ` +
        `Expected ${rowCount * spec.scaleRowBytes}, got ${scaleBytes.byteLength}.`
      );
    }
    if (hasSumSource && sumBytes.byteLength !== rowCount * 4) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis sum byte size mismatch. ` +
        `Expected ${rowCount * 4}, got ${sumBytes.byteLength}.`
      );
    }

    const scaleView = new DataView(scaleBytes.buffer, scaleBytes.byteOffset, scaleBytes.byteLength);
    const sumView = hasSumSource
      ? new DataView(sumBytes.buffer, sumBytes.byteOffset, sumBytes.byteLength)
      : null;
    const out = new Uint16Array(rowCount * spec.cols);
    const scaleCompanionDtype = String(transform.scaleCompanionDtype || '').toUpperCase();
    const scaleCompanion = scaleCompanionDtype === 'UINT8'
      ? {
        scale: Number(transform.scaleCompanionDequant?.scale),
        zeroPoint: Number(transform.scaleCompanionDequant?.zeroPoint),
      }
      : null;
    const scaleSubCount = scaleCompanion ? spec.scaleRowBytes : 1;
    const scaleSubBlockWidth = scaleCompanion
      ? Math.floor(spec.cols / scaleSubCount)
      : spec.cols;
    if (!Number.isInteger(scaleSubCount) || scaleSubCount <= 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT axis uint8 companion block count ${scaleSubCount}.`
      );
    }
    if (!Number.isInteger(scaleSubBlockWidth) || scaleSubBlockWidth <= 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT axis uint8 companion layout for width ${spec.cols} and sub-count ${scaleSubCount}.`
      );
    }

    // scaleBytes was loaded for the `[rowStart, rowStart + rowCount)` window,
    // so its byte offsets are relative to `rowStart`. Subtract `rowStart`
    // before indexing into scaleView; keep the caller contract of passing
    // absolute row indices so error messages still name the real row.
    const resolveScale = (logicalRow, col = 0) => {
      const localRow = logicalRow - rowStart;
      const rowScaleOffset = localRow * spec.scaleRowBytes;
      const storedScale = scaleCompanion
        ? (scaleView.getUint8(rowScaleOffset + Math.min(Math.floor(col / scaleSubBlockWidth), scaleSubCount - 1))
          - scaleCompanion.zeroPoint) * scaleCompanion.scale
        : scaleView.getFloat32(rowScaleOffset, true);
      if (!Number.isFinite(storedScale) || storedScale <= 0) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT axis row ${logicalRow} has invalid scale ${storedScale}.`
        );
      }
      return resolveLiteRTScaleValue(
        storedScale,
        transform,
        tensorName,
        `LiteRT axis row ${logicalRow}`
      );
    };

    if (spec.quantAxis === 1) {
      const expectedRawBytes = rowCount * spec.rawStorageRowBytes;
      if (rawBytes.byteLength !== expectedRawBytes) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT axis raw byte size mismatch. ` +
          `Expected ${expectedRawBytes}, got ${rawBytes.byteLength}.`
        );
      }
      for (let row = 0; row < rowCount; row++) {
        const rawRowOffset = row * spec.rawStorageRowBytes;
        const rowBytes = rawBytes.subarray(rawRowOffset, rawRowOffset + spec.rawStorageRowBytes);
        const storedRowSum = hasSumSource
          ? sumView.getInt32(row * 4, true)
          : null;
        const rowZeroPoint = storedRowSum == null
          ? null
          : (
            computeStoredQuantizedSum(
              rowBytes,
              transform.sourceDtype,
              transform.storageEncoding
            ) - storedRowSum
          ) / spec.cols;
        if (rowZeroPoint != null && !Number.isFinite(rowZeroPoint)) {
          throw new Error(
            `[DopplerLoader] Tensor "${tensorName}" LiteRT axis row ${rowStart + row} has invalid derived rowZeroPoint ${rowZeroPoint}.`
          );
        }
        const outRowOffset = row * spec.cols;
        for (let col = 0; col < spec.cols; col++) {
          const scaleForCol = resolveScale(rowStart + row, col);
          const quantized = rowZeroPoint == null
            ? readQuantizedValue(
              rowBytes,
              col,
              transform.sourceDtype,
              transform.storageEncoding
            )
            : (
              readStoredQuantizedValue(
                rowBytes,
                col,
                transform.sourceDtype,
                transform.storageEncoding
              ) - rowZeroPoint
            );
          out[outRowOffset + col] = float32ToFloat16(quantized * scaleForCol);
        }
      }
      return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
    }

    const storageColumnStart = options.storageColumnStart == null
      ? 0
      : Number(options.storageColumnStart);
    if (!Number.isInteger(storageColumnStart) || storageColumnStart < 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT axis storageColumnStart ${options.storageColumnStart}.`
      );
    }
    if (rawBytes.byteLength % spec.storageRows !== 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis raw byte size ${rawBytes.byteLength} ` +
        `is not divisible by storageRows=${spec.storageRows}.`
      );
    }
    const rawSliceRowBytes = rawBytes.byteLength / spec.storageRows;
    const windowColumns = rawSliceRowBytes * spec.storageValuesPerByte;
    if (
      rowStart < storageColumnStart
      || rowStart + rowCount > storageColumnStart + windowColumns
    ) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT axis raw window [${storageColumnStart}, ${storageColumnStart + windowColumns}) ` +
        `does not cover requested rows [${rowStart}, ${rowStart + rowCount}).`
      );
    }
    const localColumnBase = rowStart - storageColumnStart;
    const rowSlices = new Array(spec.storageRows);
    for (let storageRow = 0; storageRow < spec.storageRows; storageRow++) {
      const sliceOffset = storageRow * rawSliceRowBytes;
      rowSlices[storageRow] = rawBytes.subarray(sliceOffset, sliceOffset + rawSliceRowBytes);
    }

    for (let row = 0; row < rowCount; row++) {
      const logicalRow = rowStart + row;
      const localColumnIndex = localColumnBase + row;
      const storedValues = new Array(spec.cols);
      let storedSum = 0;
      for (let col = 0; col < spec.cols; col++) {
        const stored = readStoredQuantizedValue(
          rowSlices[col],
          localColumnIndex,
          transform.sourceDtype,
          transform.storageEncoding
        );
        storedValues[col] = stored;
        storedSum += stored;
      }

      const storedRowSum = hasSumSource
        ? sumView.getInt32(row * 4, true)
        : null;
      const rowZeroPoint = storedRowSum == null
        ? null
        : (storedSum - storedRowSum) / spec.cols;
      if (rowZeroPoint != null && !Number.isFinite(rowZeroPoint)) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT axis row ${logicalRow} has invalid derived rowZeroPoint ${rowZeroPoint}.`
        );
      }

      const outRowOffset = row * spec.cols;
      for (let col = 0; col < spec.cols; col++) {
        const scaleForCol = resolveScale(logicalRow, col);
        const quantized = rowZeroPoint == null
          ? readQuantizedValue(
            rowSlices[col],
            localColumnIndex,
            transform.sourceDtype,
            transform.storageEncoding
          )
          : (storedValues[col] - rowZeroPoint);
        out[outRowOffset + col] = float32ToFloat16(quantized * scaleForCol);
      }
    }
    return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
  }

  if (transform.kind === 'litert_axis_blocked_dequant') {
    const scaleBytes = options.scaleBytes;
    if (!(scaleBytes instanceof Uint8Array)) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis sourceTransform requires scaleBytes.`
      );
    }
    const sumBytes = options.sumBytes ?? null;
    const hasSumSource = transform.sumSource != null;
    if (hasSumSource && !(sumBytes instanceof Uint8Array)) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis sourceTransform requires sumBytes.`
      );
    }
    const rowStart = options.rowStart == null ? 0 : Number(options.rowStart);
    const rowCount = options.rowCount == null ? spec.rows : Number(options.rowCount);
    if (!Number.isInteger(rowStart) || rowStart < 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT blocked axis rowStart ${options.rowStart}.`
      );
    }
    if (!Number.isInteger(rowCount) || rowCount <= 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT blocked axis rowCount ${options.rowCount}.`
      );
    }
    if (rowStart + rowCount > spec.rows) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis row range (${rowStart}..${rowStart + rowCount}) exceeds ${spec.rows} rows.`
      );
    }
    if (scaleBytes.byteLength !== rowCount * spec.scaleRowBytes) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis scale byte size mismatch. ` +
        `Expected ${rowCount * spec.scaleRowBytes}, got ${scaleBytes.byteLength}.`
      );
    }
    if (hasSumSource && sumBytes.byteLength !== rowCount * 4) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis sum byte size mismatch. ` +
        `Expected ${rowCount * 4}, got ${sumBytes.byteLength}.`
      );
    }
    const expectedRawBytes = spec.storageRows * rowCount * spec.storageElementBytes;
    if (rawBytes.byteLength !== expectedRawBytes) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis raw byte size mismatch. ` +
        `Expected ${expectedRawBytes}, got ${rawBytes.byteLength}.`
      );
    }
    const scaleView = new DataView(scaleBytes.buffer, scaleBytes.byteOffset, scaleBytes.byteLength);
    const sumView = hasSumSource
      ? new DataView(sumBytes.buffer, sumBytes.byteOffset, sumBytes.byteLength)
      : null;
    const out = new Uint16Array(rowCount * spec.cols);
    for (let row = 0; row < rowCount; row++) {
      const logicalRow = rowStart + row;
      const scale = scaleView.getFloat32(row * spec.scaleRowBytes, true);
      if (!Number.isFinite(scale) || scale <= 0) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis row ${logicalRow} has invalid scale ${scale}.`
        );
      }
      const resolvedScale = resolveLiteRTScaleValue(
        scale,
        transform,
        tensorName,
        `LiteRT blocked axis row ${logicalRow}`
      );
      const storedRowSum = hasSumSource
        ? sumView.getInt32(row * 4, true)
        : null;
      let storedSum = 0;
      if (storedRowSum != null) {
        for (let storageRow = 0; storageRow < spec.storageRows; storageRow++) {
          const baseOffset = storageRow * rowCount * spec.storageElementBytes + row * spec.storageElementBytes;
          const blockBytes = rawBytes.subarray(baseOffset, baseOffset + spec.storageElementBytes);
          for (let lane = 0; lane < spec.storageBlockSize; lane++) {
            storedSum += readStoredQuantizedValue(
              blockBytes,
              lane,
              transform.sourceDtype,
              transform.storageEncoding
            );
          }
        }
      }
      const rowZeroPoint = storedRowSum == null
        ? null
        : (storedSum - storedRowSum) / spec.cols;
      if (rowZeroPoint != null && !Number.isFinite(rowZeroPoint)) {
        throw new Error(
          `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis row ${logicalRow} has invalid derived rowZeroPoint ${rowZeroPoint}.`
        );
      }
      const outRowOffset = row * spec.cols;
      for (let storageRow = 0; storageRow < spec.storageRows; storageRow++) {
        const baseOffset = storageRow * rowCount * spec.storageElementBytes + row * spec.storageElementBytes;
        const blockBytes = rawBytes.subarray(baseOffset, baseOffset + spec.storageElementBytes);
        for (let lane = 0; lane < spec.storageBlockSize; lane++) {
          const stored = readStoredQuantizedValue(
            blockBytes,
            lane,
            transform.sourceDtype,
            transform.storageEncoding
          );
          const quantized = rowZeroPoint == null
            ? readQuantizedValue(
              blockBytes,
              lane,
              transform.sourceDtype,
              transform.storageEncoding
            )
            : (stored - rowZeroPoint);
          const logicalCol = storageRow * spec.storageBlockSize + spec.storageLaneOrder[lane];
          out[outRowOffset + logicalCol] = float32ToFloat16(quantized * resolvedScale);
        }
      }
    }
    return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
  }

  const affineTransform = transform;
  const elementCount = computeElementCount(location.shape, tensorName);
  const expectedBytes = computeSourceByteLength(
    elementCount,
    affineTransform.sourceDtype,
    tensorName
  );
  if (rawBytes.byteLength !== expectedBytes) {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" sourceTransform byte size mismatch. ` +
      `Expected ${expectedBytes} bytes for ${elementCount} elements, got ${rawBytes.byteLength}.`
    );
  }

  const out = new Uint16Array(elementCount);
  for (let index = 0; index < elementCount; index++) {
    const quantized = readQuantizedValue(rawBytes, index, affineTransform.sourceDtype);
    const value = (quantized - affineTransform.zeroPoint) * affineTransform.scale;
    out[index] = float32ToFloat16(value);
  }
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}
