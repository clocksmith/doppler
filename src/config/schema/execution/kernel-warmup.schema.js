export const DEFAULT_KERNEL_WARMUP_CONFIG = {
  prewarm: false,
  prewarmMode: 'parallel',
  // { prompt, maxTokens }: one greedy generation at the end of a text pipeline load
  // (counted in load time), then a sequence reset. null runs none.
  warmGeneration: null,
};
