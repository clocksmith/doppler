export declare const update: any;
export declare const updateFromGPU: any;
export declare const recordUpdateFromGPU: any;
export declare const recordUpdateF32ToF16FromGPU: any;
export declare const recordF16UpdateAlreadyWrittenFromGPU: any;
export declare const _updateContiguous: any;
export declare const _updatePaged: any;
/** Advance sequence bookkeeping for rows already written on the GPU (contiguous base caches only). */
export declare function markAppended(this: import('./base.js').KVCache, layerIdx: number, startPos: number, numTokens: number): void;
