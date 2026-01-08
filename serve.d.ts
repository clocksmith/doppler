#!/usr/bin/env node
/**
 * DOPPLER Development Server - Serves the app UI and model files.
 * Standalone server for the DOPPLER WebGPU inference engine.
 *
 * Repository: https://github.com/clocksmith/doppler
 */

export interface ServerOptions {
  port: number;
  open: boolean;
  help: boolean;
}

export interface ModelInfo {
  path: string;
  name: string;
  architecture?: string | null;
  quantization?: string | null;
  size?: string | null;
  downloadSize?: number;
  vocabSize?: number | null;
  numLayers?: number | null;
}

export function parseArgs(argv: string[]): ServerOptions;
