export function computeElementCount(shape, tensorName) {
  if (!Array.isArray(shape)) {
    throw new Error(`[DopplerLoader] Tensor "${tensorName}" shape must be an array.`);
  }
  let total = 1;
  for (let index = 0; index < shape.length; index++) {
    const value = Number(shape[index]);
    if (!Number.isFinite(value) || Math.floor(value) !== value || value < 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid shape[${index}] (${shape[index]}).`
      );
    }
    total *= value;
  }
  return total;
}

export function computeSourceByteLength(elementCount, sourceDtype, tensorName) {
  if (sourceDtype === 'INT8' || sourceDtype === 'UINT8') {
    return elementCount;
  }
  if (sourceDtype === 'INT4') {
    return Math.ceil(elementCount / 2);
  }
  if (sourceDtype === 'INT2') {
    return Math.ceil(elementCount / 4);
  }
  throw new Error(
    `[DopplerLoader] Tensor "${tensorName}" has unsupported sourceTransform.sourceDtype "${sourceDtype}".`
  );
}

export function getPackedValuesPerByte(sourceDtype, tensorName) {
  if (sourceDtype === 'INT8' || sourceDtype === 'UINT8') {
    return 1;
  }
  if (sourceDtype === 'INT4') {
    return 2;
  }
  if (sourceDtype === 'INT2') {
    return 4;
  }
  throw new Error(
    `[DopplerLoader] Tensor "${tensorName}" has unsupported sourceTransform.sourceDtype "${sourceDtype}".`
  );
}

export function validateLiteRTTransformTarget(location, tensorName, transform, label) {
  if (transform.targetDtype !== 'F16') {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" has unsupported ${label} targetDtype "${transform.targetDtype}".`
    );
  }
  if (String(location?.dtype || '').toUpperCase() !== transform.targetDtype) {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" ${label} targetDtype "${transform.targetDtype}" ` +
      `does not match location.dtype "${location?.dtype}".`
    );
  }
}

export function validateLiteRTStorageEncoding(storageEncoding, tensorName) {
  if (storageEncoding !== 'signed' && storageEncoding !== 'offset_binary') {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" has unsupported LiteRT storageEncoding "${storageEncoding}".`
    );
  }
}

export function validateLiteRTStorageLaneOrder(storageLaneOrder, storageBlockSize, tensorName) {
  if (!Array.isArray(storageLaneOrder) || storageLaneOrder.length !== storageBlockSize) {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" LiteRT blocked axis transform requires storageLaneOrder ` +
      `with ${storageBlockSize} entries.`
    );
  }
  const seen = new Set();
  for (let index = 0; index < storageLaneOrder.length; index++) {
    const value = Number(storageLaneOrder[index]);
    if (!Number.isInteger(value) || value < 0 || value >= storageBlockSize || seen.has(value)) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" has invalid LiteRT storageLaneOrder ` +
        `${JSON.stringify(storageLaneOrder)}.`
      );
    }
    seen.add(value);
  }
}

export function getLiteRTCompanionByteLength(
  companionSource,
  tensorName,
  label,
  expectedByteLength
) {
  if (!companionSource || typeof companionSource !== 'object') {
    return null;
  }
  const byteLength = Number(companionSource.size);
  if (!Number.isInteger(byteLength) || byteLength !== expectedByteLength) {
    throw new Error(
      `[DopplerLoader] Tensor "${tensorName}" LiteRT ${label} bytes must equal ${expectedByteLength}. ` +
      `Got ${companionSource?.size}.`
    );
  }
  return byteLength;
}
