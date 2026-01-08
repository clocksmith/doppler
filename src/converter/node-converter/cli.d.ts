/**
 * Command-line argument parsing for the Node.js Model Converter.
 *
 * @module converter/node-converter/cli
 */

import type { ConvertOptions } from './types.js';

export declare function parseArgs(argv: string[]): ConvertOptions;

export declare function printHelp(): void;
