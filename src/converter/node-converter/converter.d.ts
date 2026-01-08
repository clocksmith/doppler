/**
 * Main conversion orchestration logic for the Node.js Model Converter.
 *
 * @module converter/node-converter/converter
 */

import type { ConvertOptions } from './types.js';
import type { WriteResult } from '../writer.js';

export declare function convertSafetensors(
  inputPath: string,
  outputPath: string,
  opts: ConvertOptions
): Promise<WriteResult>;

export declare function convertGGUF(
  inputPath: string,
  outputPath: string,
  opts: ConvertOptions
): Promise<WriteResult>;
