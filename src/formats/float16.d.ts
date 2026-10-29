export declare function float32ToFloat16(value: number): number;

export declare function float16ToFloat32(h: number): number;

/** Decodes binary16 bit patterns element-wise (exact). */
export declare function decodeFloat16Array(values: ArrayLike<number>): Float32Array;
