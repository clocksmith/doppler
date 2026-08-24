export async function moeFeedForwardCPU(
  hiddenStates,
  numTokens,
  config,
  moeRouter,
  expertWeights,
  expertLoader,
  layerIdx
) {
  void hiddenStates;
  void numTokens;
  void config;
  void moeRouter;
  void expertWeights;
  void expertLoader;
  void layerIdx;
  throw new Error(
    'moeFeedForwardCPU is unavailable: production MoE execution requires the WebGPU gather, expert, and scatter path.'
  );
}
