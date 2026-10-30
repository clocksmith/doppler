export declare function q40StorageBytes(shape: readonly number[]): number;
export declare function validateQ40Storage(shape: readonly number[], storage: {
  packing?: string; blockBytes?: number; blockShape?: readonly number[];
} | null | undefined, layout: string): number;
