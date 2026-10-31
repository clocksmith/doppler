export type GGUFBlockDtype = 'q4_0' | 'q4_1' | 'q5_0' | 'q8_0' | 'q6k';
export interface GGUFBlockFormat { values: number; bytes: number; code: number; variant: GGUFBlockDtype; source: string }
export declare const GGUF_BLOCK_SOURCE_DTYPES: readonly string[];
export declare function isGgufBlockSourceDtype(dtype: unknown): boolean;
export declare const GGUF_BLOCK_DTYPES: readonly GGUFBlockDtype[];
export declare function isGgufBlockDtype(dtype: unknown): dtype is GGUFBlockDtype;
export declare function ggufBlockFormat(dtype: string): GGUFBlockFormat;
export declare function ggufBlockBytes(shape: readonly number[], dtype: string): number;
