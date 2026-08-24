import type { DownloadState } from '../download-types.js';

export function getDownloadStateDatabase(): Promise<IDBDatabase | null>;
export function saveDownloadState(state: DownloadState): Promise<void>;
export function loadDownloadState(modelId: string): Promise<DownloadState | null>;
export function deleteDownloadState(modelId: string): Promise<void>;
export function loadAllDownloadStates(): Promise<Array<Record<string, unknown> & { modelId: string }>>;
