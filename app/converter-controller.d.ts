/**
 * converter-controller.d.ts - In-browser conversion controller
 *
 * @module app/converter-controller
 */

import type { ConverterConfigSchema } from '../src/config/schema/index.js';

export interface ConverterControllerCallbacks {
  onStart?: () => void;
  onProgress?: (percent: number, message: string) => void;
  onComplete?: (modelId: string) => void;
  onCancel?: () => void;
  onError?: (error: Error) => void;
  onFinish?: () => void;
}

export interface ConverterControllerOptions {
  converterConfig?: ConverterConfigSchema;
  modelId?: string;
  signal?: AbortSignal;
  [key: string]: unknown;
}

export declare class ConverterController {
  constructor(callbacks?: ConverterControllerCallbacks);

  static isSupported(): boolean;

  get isConverting(): boolean;

  convert(options?: ConverterControllerOptions): Promise<string | null>;

  convertRemote(urls: string[], options?: ConverterControllerOptions): Promise<string | null>;
}
