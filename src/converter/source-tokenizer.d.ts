export interface CharacterVocabSourceTokenizerPolicy {
  kind: 'character_vocab';
  vocabFile: string;
  specialTokens: {
    pad: string;
    bos: string;
    eos: string;
    unk: string;
    mask?: string | null;
  };
  addBosToken: boolean;
  addEosToken: boolean;
}

export interface GreedyVocabSourceTokenizerPolicy {
  kind: 'greedy_vocab';
  vocabFile: string;
  specialTokens: {
    pad: string;
    bos: string;
    eos?: string | null;
    unk: string;
    mask?: string | null;
  };
  addBosToken: boolean;
  addEosToken: boolean;
}

export type SourceTokenizerPolicy =
  | CharacterVocabSourceTokenizerPolicy
  | GreedyVocabSourceTokenizerPolicy;

export declare function validateSourceTokenizerPolicy(
  policy: SourceTokenizerPolicy | null | undefined
): SourceTokenizerPolicy | null;

export declare function buildCharacterTokenizerJson(
  policy: CharacterVocabSourceTokenizerPolicy,
  vocabText: string
): Record<string, unknown>;

export declare function buildSourceTokenizerJson(
  policy: SourceTokenizerPolicy,
  vocabText: string
): Record<string, unknown>;
