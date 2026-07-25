import type { RefactorReceipt } from '../../../refactor-receipt.js';
import type { SemanticAttentionPlan } from './plan.js';

export function captureAttentionRefactorReceipt(options: {
  state: Record<string, unknown>;
  plan?: SemanticAttentionPlan | null;
  resourceEvents?: ReadonlyArray<Record<string, unknown>>;
  error?: Error | null;
  failureBoundary?: string;
}): RefactorReceipt | null;
