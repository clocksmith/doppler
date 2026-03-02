export interface StorageCallbacks {
  onUnloadActiveModel?: (modelId: string) => Promise<void>;
  onModelsUpdated?: () => Promise<void>;
  onStorageInventoryRefreshed?: (modelIds: string[]) => void;
  onTryModel?: (modelId: string) => Promise<void>;
  onSelectModel?: (modelId: string) => void;
}

export declare function updateStorageInfo(): Promise<void>;
export declare function refreshStorageInspector(callbacks?: StorageCallbacks): Promise<void>;
