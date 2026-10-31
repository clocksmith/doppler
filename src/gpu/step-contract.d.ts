import type { CommandRecorder } from './command-recorder.js';

/**
 * What an operation declares about its recorded commands to a step observer:
 * - `token-input`: the next queue write of `count` u32 words to `buffer` at `offset`
 *   holds the step's input token ids.
 * - `position-range`: the owner's commands are fixed or affine in the step position
 *   for step positions in [min, max] (inclusive; max may be Infinity).
 * - `context-limit`: the owner's commands stay valid while a quantity that is `value`
 *   now and grows by at most one per decode position stays at or below `max`.
 */
export type StepContractDeclaration =
  | { kind: 'token-input'; buffer: GPUBuffer; offset: number; count: number }
  | { kind: 'position-range'; owner: string; min: number; max: number }
  | { kind: 'context-limit'; owner: string; value: number; max: number };

/** Route step-contract declarations made on `recorder` to `observer`; returns a stop function. */
export function observeStepContract(
  recorder: CommandRecorder,
  observer: (declaration: StepContractDeclaration) => void
): () => void;

/** Declare a step contract on `recorder`; does nothing unless an observer is attached. */
export function declareStepContract(
  recorder: CommandRecorder | null | undefined,
  declaration: StepContractDeclaration
): void;
