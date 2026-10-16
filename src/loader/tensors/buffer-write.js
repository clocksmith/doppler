/** @type {import('./buffer-write.js').alignTo4} */
export function alignTo4(size) {
  return Math.ceil(size / 4) * 4;
}

/** @type {import('./buffer-write.js').toUint8View} */
export function toUint8View(data) {
  if (data instanceof Uint8Array) return data;
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  return new Uint8Array(data);
}

/** @type {import('./buffer-write.js').writeBufferAligned} */
export function writeBufferAligned(device, buffer, data) {
  const bytes = toUint8View(data);
  const alignedSize = alignTo4(bytes.byteLength);
  if (alignedSize === bytes.byteLength) {
    device.queue.writeBuffer(buffer, 0, bytes);
    return;
  }
  const padded = new Uint8Array(alignedSize);
  padded.set(bytes);
  device.queue.writeBuffer(buffer, 0, padded);
}
