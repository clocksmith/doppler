import type { TargetPlan } from '../../config/target-plan.js';
import type { CommandExecutor } from './command-executor.js';
import type { ResourceBinder } from './resource-binder.js';
import type { GenerationOptions, GenerationInput, GenerationCompletion, ResolvedGenerationOptions } from '../../config/generation-contract.js';

export type GenerationRunOptions = GenerationInput & GenerationOptions & {
  signal?: AbortSignal | null;
};
export interface GenerationResult { sampling: ResolvedGenerationOptions; completion: GenerationCompletion; }

export interface CanonicalGenerationRequest {
  schema: 'doppler.canonical-generation-request/v1';
  /** Effective tokenizer input, already admitted under the execution identity. */
  promptTokens: readonly number[];
  policy: import('../../config/generation-contract.js').CanonicalGenerationPolicy;
  executionIdentity: string;
  signal?: AbortSignal | null;
  modules?: unknown;
}

export interface SessionController {
  /** Unpublished execution transitions; the conversation owner must settle and publish each before requesting the next. */
  generateCanonicalTransitions(targetPlan: TargetPlan, request: CanonicalGenerationRequest): AsyncGenerator<
    Readonly<import('../../inference/resident-target-state.js').ResidentTargetTransition> |
    Readonly<import('../../inference/resident-state-contract.js').AdmissionOnlyResult>, void, void>;
  generateTokens(targetPlan: TargetPlan, options: GenerationRunOptions, control?: { incremental?: boolean }): AsyncGenerator<number, GenerationResult, void>;
  close(): Promise<void>;
}

export { resolveGenerationOptions as requireGenerationOptions } from '../../config/generation-contract.js';

export declare function sampleCapsuleLogits(logits: ArrayLike<number>, contextTokens: number[], options: GenerationRunOptions, tokenContract?: Record<string, unknown>): number;
export declare function createSessionController(commandExecutor: CommandExecutor, resourceBinder: ResourceBinder, program: object): SessionController;
