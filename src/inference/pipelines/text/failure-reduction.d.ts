/**
 * Failure reduction hooks for operator-level differential debugging.
 *
 * Shrinks failures to minimal reproducible cases by prompt, token count,
 * or graph slice.
 */

export type ReducerMode = 'prompt' | 'tokens' | 'graph';

export declare const REDUCER_MODES: Readonly<{
  PROMPT: 'prompt';
  TOKENS: 'tokens';
  GRAPH: 'graph';
}>;

export interface ReductionConfig {
  mode: ReducerMode | null;
  enabled: boolean;
  prompt: {
    minLength: number;
    strategy: 'binary_search' | 'linear';
  };
  tokens: {
    maxTokens: number;
    step: number;
  };
  graph: {
    startLayer: number | null;
    endLayer: number | null;
    targetOpIds: string[];
    targetStages: string[];
  };
  divergenceOpId: string | null;
}

export interface GraphSlice {
  startLayer: number | null;
  endLayer: number | null;
  targetOpIds: string[];
  targetStages: string[];
  shouldProcessLayer(layerIdx: number): boolean;
  shouldProcessOp(opId: string, stageName: string): boolean;
  isSliced(): boolean;
}

export interface ReductionReport {
  mode: ReducerMode | null;
  originalSize: number | null;
  reducedSize: number | null;
  stepsAttempted: number;
  divergenceReproduced: boolean;
  divergenceOpId: string | null;
  minimalPromptLength: number | null;
  minimalTokenCount: number | null;
  graphSlice: GraphSlice | null;
}

export declare function createReductionConfig(options?: {
  mode?: ReducerMode | null;
  enabled?: boolean;
  promptMinLength?: number;
  promptStrategy?: 'binary_search' | 'linear';
  maxTokens?: number;
  tokenStep?: number;
  startLayer?: number | null;
  endLayer?: number | null;
  targetOpIds?: string[];
  targetStages?: string[];
  divergenceOpId?: string | null;
}): ReductionConfig;

export declare function computePromptReductionSteps(
  tokenIds: number[],
  config: ReductionConfig
): number[][];

export declare function computeTokenReductionPlan(
  originalMaxTokens: number,
  config: ReductionConfig
): number[];

export declare function createGraphSlice(config: ReductionConfig): GraphSlice;

export declare function createReductionReport(options?: Partial<ReductionReport>): ReductionReport;
