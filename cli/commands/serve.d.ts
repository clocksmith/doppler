#!/usr/bin/env node
/**
 * Serve CLI - Convert + Serve models for DOPPLER
 */

import http from 'http';

export type InputType = 'gguf' | 'rdrr';

export interface ServeConfig {
  input: string;
  port: number;
  output: string | null;
  keep: boolean;
  open: boolean;
  dopplerUrl: string;
}

export declare function parseArgs(argv: string[]): { config: string | null; help: boolean };
export declare function detectInputType(inputPath: string): Promise<InputType>;
export declare function validateRDRR(dir: string): Promise<void>;
export declare function startServer(serveDir: string, args: ServeConfig): http.Server;
