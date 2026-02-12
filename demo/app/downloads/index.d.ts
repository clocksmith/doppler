export interface DownloadProgress {
  percent?: number;
  status?: string;
  modelId?: string;
  totalBytes?: number;
  downloadedBytes?: number;
}

export interface DownloadModuleCallbacks {
  onModelRegistered?: (modelId: string) => void | Promise<void>;
  onModelsUpdated?: () => void | Promise<void>;
}

export function configureDownloadCallbacks(nextCallbacks?: DownloadModuleCallbacks): void;
export function updateDownloadStatus(progress: DownloadProgress | Record<string, unknown> | null): void;
export function refreshDownloads(): Promise<void>;
export function startDownload(): Promise<void>;
export function startDownloadFromBaseUrl(baseUrl: string, modelIdOverride?: string): Promise<string | null | undefined>;
export function pauseActiveDownload(): Promise<void>;
export function resumeActiveDownload(): Promise<void>;
export function cancelActiveDownload(): Promise<void>;
