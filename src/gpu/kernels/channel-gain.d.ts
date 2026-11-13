import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
export function validateChannelTensor(input: Tensor, tokens: number, channels: number): void;
/** Result buffer transfers to the caller. Recorded commands must settle before release. */
export function channelEnergy(recorder: CommandRecorder | null, input: Tensor,
  options: { tokens: number; channels: number; start: number; stop: number }): Promise<Tensor>;
/** First native path: split f32 SwiGLU with immutable, caller-owned positive gains. */
export function channelGainSwiGLU(recorder: CommandRecorder | null, up: Tensor, gate: Tensor,
  gains: Tensor, options: { tokens: number; channels: number; swigluLimit: number | null }): Promise<Tensor>;
/** GPU-derived gains: caller must check finite positive readback and hash before application.
 * Readback is validation/observation only; the original GPU tensor remains the applied value. */
export function deriveChannelGains(recorder: CommandRecorder | null, instruction: Tensor, reference: Tensor,
  options: { channels: number; beta: number; clip: number; epsilon: number }): Promise<Tensor>;
/** Seeds contain independent SHA-derived f32 z values (random-z) or u32 permutation indices.
 * These inputs contain no data computed from model activations or gains. Output transfers to caller. */
export function controlChannelGains(recorder: CommandRecorder | null, gains: Tensor,
  options: { channels: number; mode: 'permuted' | 'random-z'; seeds: GPUBuffer }): Promise<Tensor>;
