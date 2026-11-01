import type { LinearAttentionRuntime, LinearLayerRuntimeState } from '../linear-attention.js';

export const LINEAR_RUNTIME_SCHEMA_VERSION: 4;
export function cloneLayerRuntimeState(state: LinearLayerRuntimeState): LinearLayerRuntimeState;
/** Zero conversation state: GPU-resident convolution and recurrent buffers with one
 * batch of GPU clears (no host upload), host-resident ones in place. A GPU-resident
 * field's CPU mirror is left as is; it needs synchronization before export. */
export function clearLayerRuntimeStates(states: Iterable<LinearLayerRuntimeState>): void;
/** Same-device clone of a settled runtime. The caller pins the source until
 * completion. GPU buffers remain authoritative; CPU mirrors need synchronization
 * before export. The returned runtime owns all of its allocations independently. */
export function cloneLinearAttentionRuntimeOnGPU(runtime: LinearAttentionRuntime | null | undefined): Promise<LinearAttentionRuntime>;
