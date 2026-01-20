/**
 * CLI Utilities - Server management and browser setup helpers
 */

import type { Page, BrowserContext } from 'playwright';
import type { CLIOptions } from './types.js';

/**
 * Check if the dev server is already running
 */
export function isServerRunning(baseUrl: string): Promise<boolean>;

/**
 * Start the dev server and wait for it to be ready
 */
export function ensureServerRunning(baseUrl: string, verbose: boolean): Promise<void>;

/**
 * Install Playwright routes to serve DOPPLER static assets from disk.
 *
 * This enables running tests/benchmarks in environments where binding a local
 * dev server is not permitted (e.g., sandboxed runners).
 */
export function installLocalDopplerRoutes(page: Page, opts: CLIOptions): Promise<void>;

/**
 * Stop the server if we started it
 */
export function stopServer(): void;

export type BrowserProfileScope = 'test' | 'bench';

export function createBrowserContext(
  opts: CLIOptions,
  options?: { scope?: BrowserProfileScope; devtools?: boolean }
): Promise<BrowserContext>;

export function setupPage(context: BrowserContext, opts: CLIOptions): Promise<Page>;

export function generateResultFilename(result: any): string;
