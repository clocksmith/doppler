export interface FinitenessStatus {
  triggered: boolean;
  layer: number;
  step: number;
  metadata: string;
}

export declare function parseFinitenessStatusWords(
  words: ArrayLike<number>,
  offset?: number
): FinitenessStatus;

/** Zero status words written to the finiteness buffer before a guarded step. */
export declare const FINITENESS_RESET_WORDS: Uint32Array;

/** Raised when the finiteness guard reports a non-finite or out-of-range activation. */
export declare class FinitenessError extends Error {
  constructor(message: string);
}
