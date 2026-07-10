export interface JsonGrammarMaskTokenizer {
  decode(ids: number[], skipSpecial?: boolean, skipBos?: boolean): string;
}

export interface JsonGrammarMaskOptions {
  tokenizer?: JsonGrammarMaskTokenizer | null;
  cacheBudget?: number;
}

export interface JsonGrammarMaskContext {
  generatedIds: number[];
  tokenizer?: JsonGrammarMaskTokenizer;
  vocabSize?: number;
}

export function createJsonGrammarMask(
  options?: JsonGrammarMaskOptions
): (logits: Float32Array, context: JsonGrammarMaskContext) => void;
