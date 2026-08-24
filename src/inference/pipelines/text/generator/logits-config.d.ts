import type { LogitsConfig, LogitsWeights } from '../logits/index.js';
import type { PipelineState } from '../state.js';

export declare function getLogitsWeights(state: PipelineState): LogitsWeights;
export declare function getLogitsConfig(state: PipelineState): LogitsConfig;
