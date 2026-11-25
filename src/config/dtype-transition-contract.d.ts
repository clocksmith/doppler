export interface ImplicitDtypeTransitionOptions {
  executionPolicies?: {
    dtypeTransition?: 'require-cast-step' | null;
  } | null;
  fromDtype?: 'f16' | 'f32' | null;
  toDtype?: 'f16' | 'f32' | null;
  op?: string | null;
  detail?: string | null;
  transitionDeclaredBy?: 'step-precision' | 'explicit-cast-step' | null;
}

export function assertImplicitDtypeTransitionAllowed(
  options?: ImplicitDtypeTransitionOptions
): void;
