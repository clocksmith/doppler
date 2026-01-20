/**
 * tensor-source-download.ts - Download-first tensor source fallback
 *
 * @module browser/tensor-source-download
 */

import type { TensorSource } from './tensor-source-file.js';
import type { HttpTensorSourceOptions } from './tensor-source-http.js';

export interface DownloadTensorSourceResult {
  source: TensorSource;
  size: number;
}

export interface RemoteTensorSourceResult extends DownloadTensorSourceResult {
  supportsRange: boolean;
}

export declare function createDownloadTensorSource(
  url: string,
  options?: HttpTensorSourceOptions
): Promise<DownloadTensorSourceResult>;

export declare function createRemoteTensorSource(
  url: string,
  options?: HttpTensorSourceOptions
): Promise<RemoteTensorSourceResult>;
