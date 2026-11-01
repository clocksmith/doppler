import type { OpfsPathConfigSchema, StorageBackendConfigSchema } from '../config/schema/loading.schema.js';

export interface SelectedStorageBackend {
  type: 'opfs' | 'indexeddb' | 'memory';
  backend: import('./model-read-session.js').StorageBackend;
}

export function selectStorageBackend(
  config: StorageBackendConfigSchema,
  opfsPathConfig: OpfsPathConfigSchema
): SelectedStorageBackend;
