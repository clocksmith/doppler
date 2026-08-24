export interface ImplicitDtypeTransitionOptions {
  executionPolicies?: {
    dtypeTransition?: 'require_cast_step' | null;
  } | null;
  fromDtype?: 'f16' | 'f32' | null;
  toDtype?: 'f16' | 'f32' | null;
  op?: string | null;
  detail?: string | null;
  transitionDeclaredBy?: 'step_precision' | 'explicit_cast_step' | null;
}

export function assertImplicitDtypeTransitionAllowed(
  options?: ImplicitDtypeTransitionOptions
): void;
