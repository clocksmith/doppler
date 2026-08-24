import type { OpfsPathConfigSchema, StorageBackendConfigSchema } from '../config/schema/loading.schema.js';

export interface SelectedStorageBackend {
  type: 'opfs' | 'indexeddb' | 'memory';
  backend: Record<string, unknown> & { init(): Promise<void> };
}

export function selectStorageBackend(
  config: StorageBackendConfigSchema,
  opfsPathConfig: OpfsPathConfigSchema
): SelectedStorageBackend;
