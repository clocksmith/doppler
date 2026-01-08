/**
 * Progress reporting and logging utilities for the Node.js Model Converter.
 *
 * @module converter/node-converter/progress
 */

export declare function createVerboseLogger(
  verbose: boolean,
  category: string
): (msg: string) => void;

export declare function logConversionStart(
  format: string,
  inputPath: string,
  outputPath: string,
  weightQuant: string | null,
  embedQuant: string | null
): void;

export declare function logConversionComplete(
  manifestPath: string,
  shardCount: number,
  tensorCount: number,
  totalSize: number
): void;

export declare function logTestFixtureComplete(
  manifestPath: string,
  shardCount: number,
  tensorCount: number,
  totalSize: number
): void;

export declare function logConversionError(error: Error, verbose: boolean): void;
