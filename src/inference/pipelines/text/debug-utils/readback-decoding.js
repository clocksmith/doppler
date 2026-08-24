import { f16ToF32 } from '../../../../loader/dtype-utils.js';

export { f16ToF32 };

export function decodeReadback(buffer, dtype) {
  if (dtype === 'f32') {
    return new Float32Array(buffer);
  }
  const src = new Uint16Array(buffer);
  const out = new Float32Array(src.length);
  if (dtype === 'bf16') {
    const tmp = new Uint32Array(1);
    const f32View = new Float32Array(tmp.buffer);
    for (let i = 0; i < src.length; i++) {
      tmp[0] = src[i] << 16;
      out[i] = f32View[0];
    }
    return out;
  }
  for (let i = 0; i < src.length; i++) {
    out[i] = f16ToF32(src[i]);
  }
  return out;
}
