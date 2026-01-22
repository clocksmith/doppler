export declare function resolveEosTokenId(options: {
  config?: Record<string, unknown> | null;
  tokenizer?: {
    eosTokenId?: number;
    eos_token_id?: number;
  } | null;
  tokenizerJson?: {
    specialTokens?: { eos?: number; eos_token_id?: number };
    special_tokens?: { eos?: number; eos_token_id?: number };
  } | null;
}): number | number[];
