import type { RoPEPrecomputeDispatchPlan } from './rope-precompute.js';
export declare function planRoPEPrecomputeDispatch(device: Pick<GPUDevice, 'limits'> | null | undefined, count: number): RoPEPrecomputeDispatchPlan;
