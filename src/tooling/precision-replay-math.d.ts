export type PrecisionReplayMode =
  | 'exact'
  | 'f32_forward'
  | 'f32_reverse'
  | 'f32_tree64'
  | 'f16_forward';

export declare const PRECISION_REPLAY_MODES: readonly PrecisionReplayMode[];
export declare function roundF32(value: number): number;
export declare function roundF16(value: number): number;
export declare function exactDot(hidden: ArrayLike<number>, weights: ArrayLike<number>): number;
export declare function forwardDotF32(hidden: ArrayLike<number>, weights: ArrayLike<number>): number;
export declare function reverseDotF32(hidden: ArrayLike<number>, weights: ArrayLike<number>): number;
export declare function tree64DotF32(hidden: ArrayLike<number>, weights: ArrayLike<number>): number;
export declare function forwardDotF16(hidden: ArrayLike<number>, weights: ArrayLike<number>): number;
export declare function scoreDotProductModes(
  hidden: ArrayLike<number>,
  weights: ArrayLike<number>
): Record<PrecisionReplayMode, number>;
export declare function buildModeScoreMaps(
  hidden: ArrayLike<number>,
  rows: Map<number, ArrayLike<number>>
): Record<PrecisionReplayMode, Map<number, number>>;
export declare function sortTokenIdsByScore(
  tokenIds: readonly number[],
  scoreMap: Map<number, number>
): number[];
export declare function summarizeRanking(
  tokenIds: readonly number[],
  scoreMap: Map<number, number>,
  decodeToken: (tokenId: number) => string,
  limit: number
): Record<string, unknown>;
export declare function computeInversionCount(
  tokenIds: readonly number[],
  leftScoreMap: Map<number, number>,
  rightScoreMap: Map<number, number>
): number;
export declare function compareTokenSequences(
  left: readonly number[],
  right: readonly number[]
): Record<string, unknown>;
