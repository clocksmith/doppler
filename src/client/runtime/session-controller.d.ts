import type { TargetPlan } from '../../config/target-plan.js';
import type { CommandExecutor } from './command-executor.js';
import type { ResourceBinder } from './resource-binder.js';
import type { GenerationOptions, GenerationInput, GenerationCompletion, ResolvedGenerationOptions } from '../../config/generation-contract.js';

export type GenerationRunOptions = GenerationInput & GenerationOptions & {
  signal?: AbortSignal | null;
};
export interface GenerationResult { sampling: ResolvedGenerationOptions; completion: GenerationCompletion; }

export interface SessionController {
  generateTokens(targetPlan: TargetPlan, options: GenerationRunOptions): AsyncGenerator<number, GenerationResult, void>;
  close(): Promise<void>;
}

export { resolveGenerationOptions as requireGenerationOptions } from '../../config/generation-contract.js';

export declare function sampleCapsuleLogits(logits: ArrayLike<number>, contextTokens: number[], options: GenerationRunOptions, tokenContract?: Record<string, unknown>): number;
export declare function createSessionController(commandExecutor: CommandExecutor, resourceBinder: ResourceBinder, program: object): SessionController;
