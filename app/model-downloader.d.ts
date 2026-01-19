/**
 * model-downloader.d.ts - Browser download controller
 *
 * @module app/model-downloader
 */

import type { ModelInfo, ModelSources } from './model-selector.js';

export interface DownloadProgress {
  percent?: number;
  downloadedBytes?: number;
  totalBytes?: number;
  stage?: string;
  message?: string;
}

export interface ModelDownloaderCallbacks {
  onProgress?: (modelKey: string, percent: number, progress: DownloadProgress | null) => void;
  onStatus?: (status: string) => void;
}

export type DownloadModel = ModelInfo & { sources?: ModelSources };

export declare class ModelDownloader {
  constructor(callbacks?: ModelDownloaderCallbacks);

  download(model: DownloadModel, options?: { runAfter?: boolean } | null): Promise<boolean>;
  delete(model: DownloadModel): Promise<void>;
}
