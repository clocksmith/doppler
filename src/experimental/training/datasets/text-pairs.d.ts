export interface TextPair {
  id?: string;
  rowId?: string;
  prompt?: string;
  completion?: string;
  source?: string;
  target?: string;
  input?: string;
  output?: string;
}

export interface NormalizedTextPair {
  id: string;
  prompt: string;
  completion: string;
  promptField: string;
  completionField: string;
}

export interface TokenizedSample {
  id?: string;
  inputIds: number[];
  targetIds: number[];
  text?: string;
  prompt?: string;
  completion?: string;
}

export interface TextPairsDataset {
  sourceLabel: string;
  rowCount: number;
  rows: NormalizedTextPair[];
}

export interface LoadedTextPairsDataset extends TextPairsDataset {
  absolutePath: string;
  raw: string;
}

export declare function buildCausalPair(tokens: number[]): {
  inputIds: number[];
  targetIds: number[];
};

export declare function normalizeTextPair(record: TextPair, index?: number): NormalizedTextPair;

export declare function mapTextPairs(records: TextPair[]): NormalizedTextPair[];

export declare function parseTextPairsDataset(
  text: string,
  options?: { sourceLabel?: string }
): TextPairsDataset;

export declare function loadTextPairsDataset(
  datasetPath: string,
  options?: {
    fetch?: (url: string) => Promise<string>;
    readFile?: (path: string) => Promise<string>;
  }
): Promise<LoadedTextPairsDataset>;

export declare function tokenizeTextPairs(
  tokenizer: { encode: (text: string) => number[] },
  pairs: TextPair[],
  options?: { maxLength?: number | null; joinWith?: string }
): Promise<TokenizedSample[]>;
