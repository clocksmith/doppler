import type { LinearAttentionRuntime, LinearLayerRuntimeState } from '../linear-attention.js';

export const LINEAR_RUNTIME_SCHEMA_VERSION: 4;
export function cloneLayerRuntimeState(state: LinearLayerRuntimeState): LinearLayerRuntimeState;
/** Same-device clone of a settled runtime. The caller pins the source until
 * completion. GPU buffers remain authoritative; CPU mirrors need synchronization
 * before export. The returned runtime owns all of its allocations independently. */
export function cloneLinearAttentionRuntimeOnGPU(runtime: LinearAttentionRuntime | null | undefined): Promise<LinearAttentionRuntime>;
