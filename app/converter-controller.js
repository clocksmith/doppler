

import { log } from '../src/debug/index.js';
import {
  convertModel,
  pickModelFiles,
  isConversionSupported,
  ConvertStage,
} from '../src/browser/browser-converter.js';

/**
 * Controls browser-based model conversion.
 */
export class ConverterController {
  /** @type {boolean} */
  #isConverting = false;

  /** @type {ConverterCallbacks} */
  #callbacks;

  /**
   * @param {ConverterCallbacks} callbacks
   */
  constructor(callbacks = {}) {
    this.#callbacks = callbacks;
  }

  /**
   * Check if conversion is supported in this browser.
   * @returns {boolean}
   */
  static isSupported() {
    return isConversionSupported();
  }

  /**
   * Check if currently converting.
   * @returns {boolean}
   */
  get isConverting() {
    return this.#isConverting;
  }

  /**
   * Start the conversion process (file picker + convert).
   * @returns {Promise<string|null>} The model ID if successful, null if cancelled
   */
  async convert() {
    if (this.#isConverting) {
      return null;
    }

    try {
      // Pick files
      const files = await pickModelFiles();
      if (!files || files.length === 0) {
        return null;
      }

      log.info('Converter', `Converting ${files.length} files...`);
      this.#isConverting = true;
      this.#callbacks.onStart?.();
      this.#callbacks.onProgress?.(0, 'Starting conversion...');

      // Convert model
      const modelId = await convertModel(files, {
        onProgress: (progress) => {
          const percent = progress.percent || 0;
          const message = progress.message || progress.stage;
          this.#callbacks.onProgress?.(percent, message);

          if (progress.stage === ConvertStage.ERROR) {
            throw new Error(progress.message);
          }
        },
      });

      log.info('Converter', `Conversion complete: ${modelId}`);
      this.#callbacks.onProgress?.(100, `Done! Model: ${modelId}`);
      this.#callbacks.onComplete?.(modelId);

      return modelId;
    } catch (error) {
      if (error.name === 'AbortError') {
        log.info('Converter', 'Conversion cancelled');
        this.#callbacks.onProgress?.(0, 'Cancelled');
        this.#callbacks.onCancel?.();
        return null;
      } else {
        log.error('Converter', 'Conversion failed:', error);
        this.#callbacks.onProgress?.(0, `Error: ${error.message}`);
        this.#callbacks.onError?.(error);
        throw error;
      }
    } finally {
      this.#isConverting = false;
      this.#callbacks.onFinish?.();
    }
  }
}

/**
 * @typedef {Object} ConverterCallbacks
 * @property {() => void} [onStart]
 * @property {(percent: number, message: string) => void} [onProgress]
 * @property {(modelId: string) => void} [onComplete]
 * @property {() => void} [onCancel]
 * @property {(error: Error) => void} [onError]
 * @property {() => void} [onFinish]
 */
