import type { FP8Encoding } from '../config/schema/fp8.schema.js';
/** Exact SafeTensors E4M3FN/E5M2 code decoding, including signed zero, NaN and infinity. */
export function float8ToFloat32(code: number, encoding: FP8Encoding): number;
