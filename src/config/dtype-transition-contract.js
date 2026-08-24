function normalizeDtype(value, label) {
  if (value == null) {
    return null;
  }
  const normalized = String(value).trim().toLowerCase();
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(`[DtypeContract] ${label} must be "f16" or "f32"; got "${value}".`);
  }
  return normalized;
}

export function assertImplicitDtypeTransitionAllowed(options = {}) {
  const policy = options.executionPolicies?.dtypeTransition ?? null;
  if (policy !== 'require_cast_step') {
    return;
  }
  if (
    options.transitionDeclaredBy === 'step_precision'
    || options.transitionDeclaredBy === 'explicit_cast_step'
  ) {
    return;
  }

  const fromDtype = normalizeDtype(options.fromDtype, 'fromDtype');
  const toDtype = normalizeDtype(options.toDtype, 'toDtype');
  if (!fromDtype || !toDtype || fromDtype === toDtype) {
    return;
  }

  const op = typeof options.op === 'string' && options.op.trim()
    ? options.op.trim()
    : 'operation';
  const detail = typeof options.detail === 'string' && options.detail.trim()
    ? ` ${options.detail.trim()}`
    : '';
  throw new Error(
    `[ExecutionV1] ${op} requires implicit dtype transition ${fromDtype} -> ${toDtype}.${detail} ` +
    `execution.policies.dtypeTransition="${policy}" requires an explicit cast step in manifest.inference.execution.`
  );
}
