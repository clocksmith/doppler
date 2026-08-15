import type { RuntimeOptimizationContract } from './runtime-optimization.js';

export declare function validateRuntimeOptimizationCampaign(
  campaign: RuntimeOptimizationContract['campaign'],
  contract: RuntimeOptimizationContract
): void;

export declare function finalizeRuntimeOptimizationReceipt(
  receipt: Omit<import('./runtime-optimization.js').RuntimeOptimizationReceipt, 'promotion' | 'receiptHash'>
): import('./runtime-optimization.js').RuntimeOptimizationReceipt;
