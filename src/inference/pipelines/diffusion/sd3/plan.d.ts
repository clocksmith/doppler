export interface SD3ModulationOffset {
  scale: number;
  shift: number;
  gate: number;
}

export interface SD3TransformerPlan {
  hiddenSize: number;
  numHeads: number;
  headDim: number;
  patchSize: number;
  layerNormEps: number;
  latentChannels: number;
  latentHeight: number;
  latentWidth: number;
  gridHeight: number;
  gridWidth: number;
  tokenCount: number;
  numLayers: number;
  dualAttentionLayers: readonly number[];
  attn2Layers: readonly number[] | null;
}

export interface SD3PositionPlan {
  maxTokens: number;
  maxGrid: number;
  square: boolean;
  indices: readonly number[];
}

export declare function resolveSD3EmbeddingDtype(
  weightDtype: string | null | undefined,
  locationDtype: string | null | undefined,
  runtime: Record<string, unknown>
): string | null;
export declare function resolveSD3MatmulDtype(
  weightDtype: string | null | undefined,
  locationDtype: string | null | undefined
): string | null;
export declare function resolveSD3BiasDtype(
  weightDtype: string | null | undefined,
  locationDtype: string | null | undefined
): string;
export declare function resolveSD3LayerNormEps(
  config: Record<string, unknown>,
  runtime: Record<string, unknown>
): number;
export declare function resolveSD3ModulationSegments(
  shape: readonly number[] | null | undefined,
  hiddenSize: number,
  fallbackSegments: number,
  name: string | null | undefined
): number;
export declare function resolveSD3ModulationOffsets(
  segments: number,
  hiddenSize: number
): Readonly<{
  attn: Readonly<SD3ModulationOffset>;
  attn2: Readonly<SD3ModulationOffset>;
  ff: Readonly<SD3ModulationOffset>;
}>;
export declare function createSD3TransformerPlan(
  config: Record<string, unknown>,
  runtime: Record<string, unknown>,
  latentShape: readonly number[]
): Readonly<SD3TransformerPlan>;
export declare function createSD3PositionPlan(
  gridHeight: number,
  gridWidth: number,
  maxTokens: number
): Readonly<SD3PositionPlan>;

