#!/usr/bin/env node
/**
 * Serve CLI - Convert + Serve models for DOPPLER
 */

import http from 'http';

export type InputType = 'gguf' | 'rdrr';

export interface ServeOptions {
  input: string | null;
  port: number;
  output: string | null;
  keep: boolean;
  open: boolean;
  dopplerUrl: string;
  help: boolean;
}

export declare function parseArgs(argv: string[]): ServeOptions;
export declare function detectInputType(inputPath: string): Promise<InputType>;
export declare function validateRDRR(dir: string): Promise<void>;
export declare function startServer(serveDir: string, args: ServeOptions): http.Server;
