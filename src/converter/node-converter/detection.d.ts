/**
 * Model format detection for the Node.js Model Converter.
 *
 * @module converter/node-converter/detection
 */

import type { InputFormat } from './types.js';
import type { ModelType } from '../../config/index.js';

export declare function detectInputFormat(inputPath: string): Promise<InputFormat>;

export declare function detectModelTypeFromPreset(
  arch: string,
  config: Record<string, unknown>
): { presetId: string; modelType: ModelType };

export declare function isEmbeddingTensorName(name: string): boolean;

export declare function isLmHeadTensorName(name: string): boolean;

export declare function findTensorDtype(
  tensors: Array<{ name: string; dtype: string }>,
  matcher: (name: string) => boolean
): string | null;
