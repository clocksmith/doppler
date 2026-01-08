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
