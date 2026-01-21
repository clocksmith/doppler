

import { log } from '../src/debug/index.js';
import {
  convertModel,
  createRemoteModelSources,
  pickModelFiles,
  isConversionSupported,
  ConvertStage,
} from '../src/browser/browser-converter.js';

export class ConverterController {
  #isConverting = false;

  #callbacks;

  constructor(callbacks = {}) {
    this.#callbacks = callbacks;
  }

  static isSupported() {
    return isConversionSupported();
  }

  get isConverting() {
    return this.#isConverting;
  }

  async #runConversion(sources, options = {}) {
    if (!sources || sources.length === 0) {
      return null;
    }

    log.info('Converter', `Converting ${sources.length} source file(s)...`);
    this.#isConverting = true;
    this.#callbacks.onStart?.();
    this.#callbacks.onProgress?.(0, 'Starting conversion...');

    const modelId = await convertModel(sources, {
      ...options,
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
  }

  async convert(options = {}) {
    if (this.#isConverting) {
      return null;
    }

    try {
      const files = await pickModelFiles();
      return await this.#runConversion(files, options);
    } catch (error) {
      if (error.name === 'AbortError') {
        log.info('Converter', 'Conversion cancelled');
        this.#callbacks.onProgress?.(0, 'Cancelled');
        this.#callbacks.onCancel?.();
        return null;
      }
      log.error('Converter', 'Conversion failed:', error);
      this.#callbacks.onProgress?.(0, `Error: ${error.message}`);
      this.#callbacks.onError?.(error);
      throw error;
    } finally {
      this.#isConverting = false;
      this.#callbacks.onFinish?.();
    }
  }

  async convertRemote(urls, options = {}) {
    if (this.#isConverting) {
      return null;
    }

    try {
      const sources = await createRemoteModelSources(urls, options);
      return await this.#runConversion(sources, options);
    } catch (error) {
      if (error.name === 'AbortError') {
        log.info('Converter', 'Conversion cancelled');
        this.#callbacks.onProgress?.(0, 'Cancelled');
        this.#callbacks.onCancel?.();
        return null;
      }
      log.error('Converter', 'Conversion failed:', error);
      this.#callbacks.onProgress?.(0, `Error: ${error.message}`);
      this.#callbacks.onError?.(error);
      throw error;
    } finally {
      this.#isConverting = false;
      this.#callbacks.onFinish?.();
    }
  }
}
