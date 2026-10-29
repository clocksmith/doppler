export interface GreedyHeadState {
  runtimeConfig?: {
    shared?: { debug?: { probes?: readonly unknown[] | null } };
    inference?: { session?: { useGreedyLmHeadArgmaxFusion?: boolean } };
  };
  modelConfig?: { logitOutputScale?: number };
  operatorDiagnostics?: unknown;
}
export interface GreedyHeadOptions {
  temperature: number;
  topK: number;
  presencePenalty: number;
  suppressTokenIds?: readonly number[];
  logitMaskFn?: unknown;
}
/** The fused head bypasses materialized logits, so downstream transforms and
 * diagnostic consumers must keep the ordinary head and sampling path.
 */
export function shouldUseGreedyLmHeadArgmaxFusion(
  state: GreedyHeadState, opts: GreedyHeadOptions,
  samplingDefaults: { greedyThreshold: number }, repetitionPenalty: number
): boolean;
