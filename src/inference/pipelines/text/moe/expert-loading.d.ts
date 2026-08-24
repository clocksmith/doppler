export declare function ensureExpertLoaded(
  layerIdx: number,
  expertIdx: number,
  expertWeights: Map<string, unknown>,
  expertLoader: { loadExpert: (layerIdx: number, expertIdx: number) => Promise<unknown> }
): Promise<void>;
