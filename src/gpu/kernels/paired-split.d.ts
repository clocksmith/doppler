import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { SplitQGOptions, SplitQGResult } from './split_qg.js';
import type { SplitQKVOptions, SplitQKVResult } from './split_qkv.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';
/** The paired split program; validated inputs are always interleaved two-F32 components. */
export declare function resolvePairedSplitPipeline(split: 'qg' | 'qkv'): KernelPipelineRequest;
export declare function executePairedSplitQG(target: CommandRecorder | null, input: Tensor, options: SplitQGOptions): Promise<SplitQGResult>;
export declare function executePairedSplitQKV(target: CommandRecorder | null, input: Tensor, options: SplitQKVOptions): Promise<SplitQKVResult>;
