#!/usr/bin/env node
/**
 * Quick test query runner - minimal Playwright script for ad-hoc inference testing
 *
 * Usage:
 *   doppler --config <ref>
 */

import type { BrowserContext, Page } from 'playwright';

export interface Options {
  prompt: string | null;
  model: string;
  baseUrl: string;
  repl: boolean;
}

export declare function parseArgs(argv: string[]): { config: string | null; help: boolean };
export declare function launchBrowser(): Promise<{ context: BrowserContext; page: Page }>;
export declare function loadModel(page: Page, baseUrl: string, model: string): Promise<void>;
export declare function runQuery(page: Page, prompt: string): Promise<{ output: string; tokenCount: number; elapsedMs: number; tokensPerSec: number }>;
export declare function clearKVCache(page: Page): Promise<void>;
export declare function runRepl(page: Page, opts: Options): Promise<void>;
export declare function main(): Promise<void>;
