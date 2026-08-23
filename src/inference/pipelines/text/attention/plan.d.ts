export const ATTENTION_PLAN_SCHEMA: 'doppler.attention-plan/v1';

export interface SemanticAttentionPlan {
  readonly schema: typeof ATTENTION_PLAN_SCHEMA;
  readonly id: string;
  readonly phase: 'prefill' | 'decode';
  readonly geometry: Readonly<Record<string, number>>;
  readonly dtypes: Readonly<Record<string, string>>;
  readonly transitions: Readonly<Record<string, string | null>>;
  readonly fusion: Readonly<Record<string, boolean>>;
  readonly queryTransform: Readonly<{ scale: number; rope: 'enabled' | 'disabled' }>;
  readonly kv: Readonly<Record<string, unknown>>;
  readonly attention: Readonly<Record<string, unknown>>;
  readonly outputGate: Readonly<Record<string, unknown>>;
  readonly outputProjection: Readonly<Record<string, unknown>>;
  readonly observation: Readonly<Record<string, unknown>>;
  readonly lifetimes: Readonly<Record<string, boolean>>;
  readonly stages: readonly string[];
}

export function resolveAttentionPlan(input: Record<string, unknown>): SemanticAttentionPlan;
export function resolveAttentionPlanForDispatch(options: Record<string, unknown>): SemanticAttentionPlan;
export function kernelPathSupportsOutputGateFusion(kernelPath: Record<string, unknown> | null): boolean;
export function bindAttentionPlan<T extends Record<string, unknown>>(
  plan: SemanticAttentionPlan,
  resources: T
): Readonly<{ plan: SemanticAttentionPlan; resources: T }>;
export function executeBoundAttentionPlan<T>(
  boundPlan: Readonly<{ plan: SemanticAttentionPlan; resources: Record<string, unknown> }>,
  executor: {
    executeStage(stage: string, boundPlan: unknown, previous: unknown): Promise<unknown>;
  }
): Promise<T>;
export function createAttentionExecutor(
  mode: 'immediate' | 'recorded',
  stageRunners?: Record<string, (boundPlan: unknown, previous: unknown) => unknown>
): Readonly<Record<string, unknown>>;
