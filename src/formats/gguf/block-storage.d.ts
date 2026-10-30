export interface GGUFBlockFormat { values: number; bytes: number; code: number; variant: 'q4_0' | 'q4_1' | 'q8_0' | 'q6k' }
export declare function ggufBlockFormat(dtype: string): GGUFBlockFormat;
export declare function ggufBlockBytes(shape: readonly number[], dtype: string): number;
