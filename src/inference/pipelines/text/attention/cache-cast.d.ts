import type { Tensor } from '../../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { ResourceScope } from '../../../resource-scope.js';
export declare function recordAttentionCacheCast(recorder: CommandRecorder, tensor: Tensor, scope: ResourceScope<GPUBuffer>, label: string): Promise<Tensor>;
