import type { Tensor } from '../../../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../../../gpu/command-recorder.js';
import type { ResourceScope } from '../../../../resource-scope.js';
import type { LayerWeights } from '../../types.js';
import type { LoRAAdapter } from '../../lora.js';
import type { AttentionConfig, AttentionDebugFlags, AttentionResult, AttentionState } from '../types.js';
import type { SemanticAttentionPlan } from '../plan.js';
import type { interpretAttentionWithRecorder } from '../interpreter.js';

export interface RecordedAttentionOutputContext extends Pick<AttentionConfig,
  'layerIdx' | 'numTokens' | 'numHeads' | 'headDim' | 'hiddenSize' | 'kernelPath' | 'residualTensor'> {
  attnForProjection: Tensor;
  layerWeights: LayerWeights;
  getWeightBuffer: Parameters<typeof interpretAttentionWithRecorder>[7];
  oProjInputDtype: 'f16' | 'f32';
  oProjOutputDtype: 'f16' | 'f32';
  state: AttentionState;
  recorder: CommandRecorder;
  lora: LoRAAdapter | null | undefined;
  attentionPlan: SemanticAttentionPlan;
  resourceScope: ResourceScope<GPUBuffer>;
  debugFlags: AttentionDebugFlags;
  kTensor: Tensor | null;
  qGateTensor: Tensor | null;
  qTensor: Tensor;
  vTensor: Tensor | null;
  retainSharedKvBuffers: boolean;
}
export function recordAttentionOutputProjection(context: RecordedAttentionOutputContext): Promise<AttentionResult>;
