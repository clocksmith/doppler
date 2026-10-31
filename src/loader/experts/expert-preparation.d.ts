import type { ExpertPreparationConfigSchema } from '../../config/schema/loading.schema.js';
import type { ModelLoadOwner } from '../model-load.js';
import type { LoadOptions } from '../loader-types.js';

export interface ExpertPreparationReceipt {
  mode: ExpertPreparationConfigSchema['mode'];
  status: 'deferred' | 'pending' | 'running' | 'complete' | 'cancelled' | 'failed';
  layerCount: number | null;
  preparedBanks: number;
  expertsPerBank: number;
}

export declare function createExpertPreparation(
  loader: ModelLoadOwner,
  policy: ExpertPreparationConfigSchema
): (layerRange: readonly [number, number], options: Pick<LoadOptions, 'signal' | 'onProgress'>) => Promise<void>;
