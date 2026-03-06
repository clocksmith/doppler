import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

export interface RepeatChannelsOptions extends OutputBufferOptions {
  inChannels: number;
  height: number;
  width: number;
  repeats: number;
}

export declare function runRepeatChannels(
  input: Tensor,
  options: RepeatChannelsOptions
): Promise<Tensor>;

export declare function recordRepeatChannels(
  recorder: CommandRecorder,
  input: Tensor,
  options: RepeatChannelsOptions
): Promise<Tensor>;
