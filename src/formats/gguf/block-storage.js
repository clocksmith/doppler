import formats from './quantized-formats.json' with { type: 'json' };

/** @type {Readonly<Record<string, Omit<import('./block-storage.js').GGUFBlockFormat, 'variant'> & { variant: string; source: string } | { alias: string }>>} */
const declared = formats;

function declaredFormats() {
  /** @type {Array<Omit<import('./block-storage.js').GGUFBlockFormat, 'variant'> & { variant: string; source: string }>} */
  const out = [];
  for (const entry of Object.values(declared)) if (!('alias' in entry)) out.push(entry);
  return out;
}

/** Runtime dtypes of every declared packed GGUF block format; the JSON is the single owner. */
/** @type {typeof import('./block-storage.js').GGUF_BLOCK_DTYPES} */
export const GGUF_BLOCK_DTYPES = Object.freeze(declaredFormats()
  .map(entry => /** @type {import('./block-storage.js').GGUFBlockDtype} */ (entry.variant)));

/** GGUF source type names of the same formats, e.g. Q5_0. */
/** @type {typeof import('./block-storage.js').GGUF_BLOCK_SOURCE_DTYPES} */
export const GGUF_BLOCK_SOURCE_DTYPES = Object.freeze(declaredFormats().map(entry => entry.source));

/** @type {typeof import('./block-storage.js').isGgufBlockSourceDtype} */
export function isGgufBlockSourceDtype(dtype) {
  return typeof dtype === 'string' && GGUF_BLOCK_SOURCE_DTYPES.includes(dtype);
}

/** @type {typeof import('./block-storage.js').isGgufBlockDtype} */
export function isGgufBlockDtype(dtype) {
  return typeof dtype === 'string' && /** @type {readonly string[]} */ (GGUF_BLOCK_DTYPES).includes(dtype);
}

/** @type {typeof import('./block-storage.js').ggufBlockFormat} */
export function ggufBlockFormat(dtype) {
  const entry = declared[dtype.toLowerCase()];
  const format = entry && 'alias' in entry ? declared[entry.alias] : entry;
  if (!format || 'alias' in format) throw new Error(`Unsupported packed GGUF dtype: ${dtype}`);
  const { variant } = format;
  if (!isGgufBlockDtype(variant)) throw new Error(`Unsupported packed GGUF kernel variant: ${variant}`);
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
