/**
 * model-registry.d.ts - Model discovery and registry
 *
 * @module app/model-registry
 */

import type { ModelInfo, ModelSources } from './model-selector.js';

export interface RegistryModel extends ModelInfo {
  sources: ModelSources;
}

export interface RemoteModelConfig {
  id: string;
  name: string;
  size?: string;
  architecture?: string;
  quantization?: string;
  downloadSize?: number;
  url: string;
}

export declare class ModelRegistry {
  constructor(remoteModels?: RemoteModelConfig[]);

  discover(): Promise<RegistryModel[]>;
  getModels(): RegistryModel[];
  findByKey(key: string): RegistryModel | undefined;
  findByBrowserId(id: string): RegistryModel | undefined;
  isAvailableLocally(model: RegistryModel): boolean;
}
