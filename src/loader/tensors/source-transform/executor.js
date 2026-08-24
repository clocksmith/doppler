export function readStoredQuantizedValue(
  bytes,
  index,
  sourceDtype,
  storageEncoding = 'signed'
) {
  if (sourceDtype === 'INT8') {
    if (storageEncoding === 'offset_binary') {
      return bytes[index];
    }
    return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)[index];
  }
  if (sourceDtype === 'UINT8') {
    return bytes[index];
  }
  if (sourceDtype === 'INT2') {
    const packed = bytes[index >> 2];
    const shift = (index & 3) * 2;
    const dibit = (packed >> shift) & 0x03;
    if (storageEncoding === 'offset_binary') {
      return dibit;
    }
    return dibit >= 2 ? dibit - 4 : dibit;
  }
  const packed = bytes[index >> 1];
  const nibble = (index & 1) === 0 ? (packed & 0x0f) : (packed >> 4);
  if (storageEncoding === 'offset_binary') {
    return nibble;
  }
  return nibble >= 8 ? nibble - 16 : nibble;
}

export function readQuantizedValue(bytes, index, sourceDtype, storageEncoding = 'signed') {
  const rawValue = readStoredQuantizedValue(bytes, index, sourceDtype, storageEncoding);
  if (sourceDtype === 'INT8' || sourceDtype === 'UINT8') {
    if (storageEncoding === 'offset_binary') {
      return rawValue - 128;
    }
    return rawValue;
  }
  if (sourceDtype === 'INT2') {
    if (storageEncoding === 'offset_binary') {
      return rawValue - 2;
    }
    return rawValue;
  }
  if (storageEncoding === 'offset_binary') {
    return rawValue - 8;
  }
  return rawValue;
}

export function computeStoredQuantizedSum(bytes, sourceDtype, storageEncoding = 'signed') {
  let total = 0;
  if (sourceDtype === 'INT8') {
    if (storageEncoding === 'offset_binary') {
      for (let index = 0; index < bytes.byteLength; index++) {
        total += bytes[index];
      }
      return total;
    }
    const signed = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let index = 0; index < signed.length; index++) {
      total += signed[index];
    }
    return total;
  }
  if (sourceDtype === 'UINT8') {
    for (let index = 0; index < bytes.byteLength; index++) {
      total += bytes[index];
    }
    return total;
  }
  if (sourceDtype === 'INT2') {
    for (let index = 0; index < bytes.byteLength; index++) {
      const packed = bytes[index];
      const a = packed & 0x03;
      const b = (packed >> 2) & 0x03;
      const c = (packed >> 4) & 0x03;
      const d = (packed >> 6) & 0x03;
      if (storageEncoding === 'offset_binary') {
        total += a + b + c + d;
      } else {
        total += (a >= 2 ? a - 4 : a);
        total += (b >= 2 ? b - 4 : b);
        total += (c >= 2 ? c - 4 : c);
        total += (d >= 2 ? d - 4 : d);
      }
    }
    return total;
  }
  for (let index = 0; index < bytes.byteLength; index++) {
    const packed = bytes[index];
    const a = packed & 0x0f;
    const b = packed >> 4;
    if (storageEncoding === 'offset_binary') {
      total += a + b;
    } else {
      total += (a >= 8 ? a - 16 : a);
      total += (b >= 8 ? b - 16 : b);
    }
  }
  return total;
}

export function resolveLiteRTScaleValue(storedScale, transform, tensorName, rowLabel) {
  const scaleSemantics = String(transform?.scaleSemantics || '').trim().toLowerCase();
  if (scaleSemantics === 'step') {
    return storedScale;
  }
  if (scaleSemantics === 'qmax_abs') {
    const scaleDivisor = Number(transform?.scaleDivisor);
    if (!Number.isFinite(scaleDivisor) || scaleDivisor <= 0) {
      throw new Error(
        `[DopplerLoader] Tensor "${tensorName}" ${rowLabel} is missing a valid LiteRT scaleDivisor for scaleSemantics="qmax_abs".`
      );
    }
    return storedScale / scaleDivisor;
  }
  throw new Error(
    `[DopplerLoader] Tensor "${tensorName}" ${rowLabel} has unsupported LiteRT scaleSemantics "${transform?.scaleSemantics}".`
  );
}
