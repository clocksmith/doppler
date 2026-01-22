export interface TextPair {
  prompt: string;
  completion: string;
}

export interface TokenizedSample {
  inputIds: number[];
  targetIds: number[];
  text?: string;
}

export declare function buildCausalPair(tokens: number[]): {
  inputIds: number[];
  targetIds: number[];
};

export declare function tokenizeTextPairs(
  tokenizer: { encode: (text: string) => number[] },
  pairs: TextPair[],
  options?: { maxLength?: number | null; joinWith?: string }
): Promise<TokenizedSample[]>;
