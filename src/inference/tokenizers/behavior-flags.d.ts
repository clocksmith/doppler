export interface BundledTokenizerBehaviorFlags {
  addBosToken: boolean | null;
  addEosToken: boolean | null;
}

export function inferBundledTokenizerBehaviorFlags(
  tokenizerJson: unknown,
  specialTokens?: { bos?: number | null, eos?: number | null } | null
): BundledTokenizerBehaviorFlags;
