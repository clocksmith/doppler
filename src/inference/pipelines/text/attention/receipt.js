import { createRefactorReceipt } from '../../../refactor-receipt.js';

function normalizeError(error, boundary) {
  if (!error) {
    return null;
  }
  return {
    boundary,
    name: typeof error.name === 'string' ? error.name : 'Error',
    message: typeof error.message === 'string' ? error.message : String(error),
    cleanup: 'completed',
  };
}

export function captureAttentionRefactorReceipt(options = {}) {
  const state = options.state ?? {};
  const policy = state.observationContext?.receiptPolicy ?? 'off';
  const failure = options.error
    ? normalizeError(options.error, options.failureBoundary ?? 'attention')
    : null;
  if (policy === 'off' || (policy === 'on_failure' && !failure)) {
    return null;
  }
  const plan = options.plan ?? null;
  const receipt = createRefactorReceipt({
    commandContext: state.observationContext?.commandContext ?? null,
    resolvedSession: state.resolvedRuntimeSession ?? null,
    observationContext: state.observationContext ?? null,
    operationPlan: plan,
    operations: (plan?.stages ?? []).map((stage, sequence) => ({
      sequence,
      stage,
    })),
    dtypeTransitions: Object.entries(plan?.transitions ?? {})
      .filter(([, transition]) => transition != null)
      .map(([boundary, transition]) => ({
        boundary,
        transition,
      })),
    resourceEvents: options.resourceEvents ?? [],
    failure,
  });
  state.stats ??= {};
  state.stats.refactorReceipts ??= [];
  state.stats.refactorReceipts.push(receipt);
  if (
    failure
    && options.error
    && typeof options.error === 'object'
    && Object.isExtensible(options.error)
  ) {
    options.error.details = {
      ...(options.error.details && typeof options.error.details === 'object'
        ? options.error.details
        : {}),
      refactorReceipt: receipt,
    };
  }
  return receipt;
}
