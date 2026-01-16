export interface ModelRegistryEntry {
  modelId: string;
  totalSize?: number;
  quantization?: string;
  hashAlgorithm?: string;
  backend?: string;
  createdAt?: string;
  [key: string]: unknown;
}

export interface ModelRegistry {
  models: ModelRegistryEntry[];
}

export interface ModelRegistrySaveInfo {
  backend: 'opfs' | 'indexeddb';
  path: string;
}

export declare function loadModelRegistry(): Promise<ModelRegistry>;

export declare function saveModelRegistry(registry: ModelRegistry): Promise<ModelRegistrySaveInfo>;

export declare function listRegisteredModels(): Promise<ModelRegistryEntry[]>;

export declare function registerModel(entry: ModelRegistryEntry): Promise<ModelRegistryEntry>;

export declare function removeRegisteredModel(modelId: string): Promise<ModelRegistryEntry[]>;
