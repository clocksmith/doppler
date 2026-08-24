export interface BundledTokenizerBehaviorFlags {
  addBosToken: boolean | null;
  addEosToken: boolean | null;
}

export function isBosLikeLabel(value: unknown): boolean;
export function isEosLikeLabel(value: unknown): boolean;

export function inferBundledTokenizerBehaviorFlags(
  tokenizerJson: unknown,
  specialTokens?: { bos?: number | null, eos?: number | null } | null
): BundledTokenizerBehaviorFlags;
