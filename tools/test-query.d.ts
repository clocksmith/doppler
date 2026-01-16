#!/usr/bin/env node
/**
 * Quick test query runner - minimal Playwright script for ad-hoc inference testing
 *
 * Usage:
 *   npx tsx tools/test-query.ts "the color of the sky is "
 *   npx tsx tools/test-query.ts --model gemma-3-1b-it-q4 "hello world"
 *   npx tsx tools/test-query.ts --config debug --repl  # Interactive mode with cached model
 */

import type { BrowserContext, Page } from 'playwright';

export interface Options {
  prompt: string;
  model: string;
  baseUrl: string;
  config: string;
  timeout: number;
  repl: boolean;
}

export declare function parseArgs(argv: string[]): Options;
export declare function launchBrowser(): Promise<{ context: BrowserContext; page: Page }>;
export declare function loadModel(page: Page, baseUrl: string, model: string): Promise<void>;
export declare function runQuery(page: Page, prompt: string): Promise<{ output: string; tokenCount: number; elapsedMs: number; tokensPerSec: number }>;
export declare function clearKVCache(page: Page): Promise<void>;
export declare function runRepl(page: Page, opts: Options): Promise<void>;
export declare function main(): Promise<void>;
