/**
 * Command-line argument parsing for the Node.js Model Converter (config-only).
 *
 * @module converter/node-converter/cli
 */

export declare function parseArgs(argv: string[]): { config: string | null; help: boolean };

export declare function printHelp(): void;
