/**
 * converter-controller.d.ts - In-browser conversion controller
 *
 * @module app/converter-controller
 */

export interface ConverterControllerCallbacks {
  onStart?: () => void;
  onProgress?: (percent: number, message: string) => void;
  onComplete?: (modelId: string) => void;
  onCancel?: () => void;
  onError?: (error: Error) => void;
  onFinish?: () => void;
}

export declare class ConverterController {
  constructor(callbacks?: ConverterControllerCallbacks);

  static isSupported(): boolean;

  get isConverting(): boolean;

  convert(): Promise<string | null>;
}
