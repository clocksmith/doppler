import formats from './quantized-formats.json' with { type: 'json' };

/** @type {Readonly<Record<string, Omit<import('./block-storage.js').GGUFBlockFormat, 'variant'> & { variant: string } | { alias: string }>>} */
const declared = formats;

/** @type {typeof import('./block-storage.js').ggufBlockFormat} */
export function ggufBlockFormat(dtype) {
  const entry = declared[dtype.toLowerCase()];
  const format = entry && 'alias' in entry ? declared[entry.alias] : entry;
  if (!format || 'alias' in format) throw new Error(`Unsupported packed GGUF dtype: ${dtype}`);
  const { variant } = format;
  if (variant !== 'q4_0' && variant !== 'q4_1' && variant !== 'q8_0' && variant !== 'q6k') {
    throw new Error(`Unsupported packed GGUF kernel variant: ${variant}`);
  }
  return { ...format, variant };
}
/** @type {typeof import('./block-storage.js').ggufBlockBytes} */
export function ggufBlockBytes(shape,dtype) {
  const format=ggufBlockFormat(dtype);
  if (!Array.isArray(shape) || ![2,3].includes(shape.length)
    || shape.some(value=>!Number.isSafeInteger(value)||value<1) || shape.at(-1)%format.values) {
    throw new Error('Packed GGUF matrix/bank requires explicit geometry and complete quantization blocks.');
  }
  const bytes=shape.reduce((a,b)=>a*b,1)/format.values*format.bytes;
  if (!Number.isSafeInteger(bytes)||bytes>0xffffffff) throw new Error('Packed GGUF storage exceeds addressable byte extents.');
  return bytes;
}
