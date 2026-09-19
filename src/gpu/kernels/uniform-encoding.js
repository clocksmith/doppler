export function writeU32(view, offset, value, kernel, field) {
  if (!Number.isInteger(value) || value < 0 || value > 0xFFFFFFFF) {
    throw new Error(`Kernel "${kernel}" uniform "${field}" requires a u32; got ${String(value)}.`);
  }
  view.setUint32(offset, value, true);
}

export function writeI32(view, offset, value, kernel, field) {
  if (!Number.isInteger(value) || value < -0x80000000 || value > 0x7FFFFFFF) {
    throw new Error(`Kernel "${kernel}" uniform "${field}" requires an i32; got ${String(value)}.`);
  }
  view.setInt32(offset, value, true);
}

export function writeF32(view, offset, value, kernel, field) {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
    throw new Error(`Kernel "${kernel}" uniform "${field}" requires a finite f32; got ${String(value)}.`);
  }
  view.setFloat32(offset, value, true);
}
