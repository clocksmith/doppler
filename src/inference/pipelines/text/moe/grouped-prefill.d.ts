import type { KernelPathSchema } from '../../../../config/schema/index.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { QwenRouteExpertOptions } from './qwen-route-executor.js';
/** Opt-in execution-v1-derived path only; absent steps preserve the baseline. */
export declare function resolveGroupedPrefill(options: {
  kernelPath: KernelPathSchema | null; phase: 'prefill' | 'decode'; layerIdx: number;
}): boolean;
/** Routed experts only. Existing shared-expert and layer residual owners run afterwards. */
export declare function executeGroupedPrefill(recorder: CommandRecorder | null, options: QwenRouteExpertOptions): Promise<Tensor>;
