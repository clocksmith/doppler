export interface DownloadProgressUpdate {
  modelId: string;
  percent: number;
  downloadedBytes: number;
  totalBytes: number;
  status: string;
}

export interface DownloadStateUpdate {
  active: boolean;
  modelId: string;
}

export interface DownloadCallbacks {
  onModelRegistered?: ((modelId: string) => Promise<unknown> | unknown) | null;
  onModelsUpdated?: (() => Promise<unknown> | unknown) | null;
  onProgress?: ((update: DownloadProgressUpdate) => void) | null;
  onStateChange?: ((update: DownloadStateUpdate) => void) | null;
}

export declare function configureDownloadCallbacks(callbacks?: DownloadCallbacks): void;
export declare function refreshDownloads(): Promise<void>;
export declare function startDownload(): Promise<boolean>;
export declare function startDownloadFromBaseUrl(baseUrl: string, modelIdOverride?: string): Promise<boolean>;
export declare function pauseActiveDownload(): void;
export declare function resumeActiveDownload(): void;
export declare function cancelActiveDownload(): void;
