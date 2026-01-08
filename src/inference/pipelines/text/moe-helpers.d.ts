export declare function ensureExpertLoaded(
  layerIdx: number,
  expertIdx: number,
  expertWeights: Map<string, unknown>,
  expertLoader: { loadExpert: (layerIdx: number, expertIdx: number) => Promise<unknown> }
): Promise<void>;

export declare function gatherTokens(
  hiddenStates: Float32Array,
  indices: Uint32Array,
  hiddenSize: number
): Float32Array;
