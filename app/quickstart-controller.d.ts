/**
 * quickstart-controller.d.ts - Quick-start flow controller
 *
 * @module app/quickstart-controller
 */

import type { QuickStartDownloadResult, RemoteModelConfig } from '../src/storage/quickstart-downloader.js';

export interface QuickStartControllerCallbacks {
  onVRAMInsufficient?: (requiredBytes: number, availableBytes: number) => void;
  onStorageConsent?: (modelName: string, requiredBytes: number, availableBytes: number) => Promise<boolean> | boolean;
  onDownloadStart?: () => void;
  onProgress?: (percent: number, downloadedBytes: number, totalBytes: number, speed: number) => void;
  onComplete?: (modelId: string) => void;
  onDeclined?: () => void;
  onError?: (message: string) => void;
}

export declare class QuickStartController {
  constructor(callbacks?: QuickStartControllerCallbacks);

  static getAvailableModels(): Record<string, RemoteModelConfig>;
  static isQuickStartModel(modelId: string): boolean;

  start(modelId: string): Promise<QuickStartDownloadResult>;
}
