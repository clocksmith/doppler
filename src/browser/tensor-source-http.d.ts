/**
 * tensor-source-http.ts - HTTP tensor source with Range support
 *
 * @module browser/tensor-source-http
 */

import type { TensorSource } from './tensor-source-file.js';

export interface HttpRangeProbe {
  ok: boolean;
  status: number;
  supportsRange: boolean;
  size: number | null;
  acceptRanges: string | null;
  contentEncoding: string | null;
}

export interface HttpTensorSourceOptions {
  headers?: Record<string, string>;
  signal?: AbortSignal;
  name?: string;
  allowDownloadFallback?: boolean;
  maxDownloadBytes?: number | null;
}

export declare function probeHttpRange(url: string, options?: HttpTensorSourceOptions): Promise<HttpRangeProbe>;

export declare function createHttpTensorSource(url: string, options?: HttpTensorSourceOptions): Promise<TensorSource>;
