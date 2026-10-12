import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { SplitQGOptions, SplitQGResult } from './split_qg.js';
import type { SplitQKVOptions, SplitQKVResult } from './split_qkv.js';
export declare function executePairedSplitQG(target: CommandRecorder | null, input: Tensor, options: SplitQGOptions): Promise<SplitQGResult>;
export declare function executePairedSplitQKV(target: CommandRecorder | null, input: Tensor, options: SplitQKVOptions): Promise<SplitQKVResult>;
