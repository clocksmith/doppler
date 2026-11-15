import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
/** Caller owns both GPU buffers. status: 1 valid, 0 degenerate identity, -1 invalid data.
 * Invalid capture rejects immediately or through recorder validation before output consumption.
 * Keep basis/status immutable through all applications; settle recorded work before releasing. */
export interface ChannelRotation { basis: Tensor; status: Tensor; }
/** Derives a rank-two plane from signed means, not activation energies, in one capture pass.
 * Input is explicit post-SwiGLU f32 activations; template tokens must lie outside content.
 * This primitive does not integrate a model, mutate weights, or authorize quality claims. */
export function deriveChannelRotation(recorder: CommandRecorder | null, input: Tensor,
  options: { tokens: number; channels: number; start: number; stop: number;
    contentStart: number; contentStop: number; epsilon: number }): Promise<ChannelRotation>;
/** Applies R(theta); theta zero and degenerate planes are identities; inverse is R(-theta).
 * Output transfers to caller; input and basis are borrowed and unchanged. */
export function applyChannelRotation(recorder: CommandRecorder | null, input: Tensor,
  rotation: ChannelRotation, options: { tokens: number; channels: number; theta: number }): Promise<Tensor>;
