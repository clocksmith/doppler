import type { TrainingLossScalingConfigSchema } from '../config/schema/training.schema.js';
import type { Tensor } from '../gpu/tensor.js';

export declare function detectOverflow(grads: Map<Tensor, Tensor>): Promise<boolean>;

export declare class DynamicLossScaler {
  enabled: boolean;
  scale: number;
  minScale: number;
  maxScale: number;
  scaleFactor: number;
  backoffFactor: number;
  growthInterval: number;
  overflowCheck: boolean;

  constructor(config?: TrainingLossScalingConfigSchema);
  shouldScale(): boolean;
  scaleLoss(loss: Tensor): Promise<Tensor>;
  unscaleGradients(grads: Map<Tensor, Tensor>): Promise<Map<Tensor, Tensor>>;
  update(hasOverflow: boolean): void;
}
