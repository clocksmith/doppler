import type {
  ToolingCommandRequest,
  ToolingCommandRequestInput,
} from '../command-api.js';
import type { BrowserCommandRunResult } from '../browser-command-runner.js';

export interface StaticMount {
  urlPrefix: string;
  rootDir: string;
}

export interface BrowserRelayOptions {
  baseUrl?: string;
  staticMounts?: StaticMount[];
}

export interface BrowserRelayLocalModelResolution {
  relayRequest: ToolingCommandRequestInput;
  staticMounts: StaticMount[];
}

export interface StaticFileServerOptions {
  rootDir?: string;
  staticMounts?: StaticMount[];
  host?: string;
  port?: number;
}

export interface StaticFileServerHandle {
  baseUrl: string;
  close: () => Promise<void>;
}

export declare const DEFAULT_CLEANUP_TIMEOUT_MS: number;
export declare const DEFAULT_OPFS_CACHE_DIR: string;
export declare const DEFAULT_OPFS_CACHE_PORT: number;

export declare function createStaticFileServer(
  options?: StaticFileServerOptions
): Promise<StaticFileServerHandle>;
export declare function normalizeHeadless(value: unknown): boolean;
export declare function normalizeTimeoutMs(value: unknown): number;
export declare function normalizeRunnerPath(value: unknown): string;
export declare function isRecoverablePersistentLaunchError(error: unknown): boolean;
export declare function normalizeBaseUrl(value: unknown): string | null;
export declare function normalizeBrowserArgs(value: unknown): string[];
export declare function resolveLocalFileModelUrlForBrowserRelay(
  commandRequest: ToolingCommandRequestInput,
  options?: BrowserRelayOptions
): Promise<BrowserRelayLocalModelResolution>;
export declare function createPersistentContextRequiredError(
  requestedLoadMode: unknown,
  cause?: unknown
): Error;
export declare function finalizeBrowserRelayResponse(
  response: unknown,
  request: ToolingCommandRequest
): BrowserCommandRunResult & { request: ToolingCommandRequest };
export declare function formatBrowserEvaluationError(payload: unknown): Error;
export declare function runBrowserCommandEvaluationWithTimeout<T>(
  operation: () => Promise<T>,
  timeoutMs: number
): Promise<T>;
export declare function runBrowserCleanupWithTimeout(
  operation: () => Promise<unknown> | unknown,
  timeoutMs: number
): Promise<boolean>;
export declare function terminateBrowserProcess(browser: unknown): void;
export declare function browserLaunchArgs(extraArgs?: string[]): string[];
export declare function launchBrowser(
  chromium: unknown,
  launchOptions: Record<string, unknown>,
  options?: Record<string, unknown>
): Promise<unknown>;
export declare function launchPersistentBrowser(
  chromium: unknown,
  userDataDir: string,
  launchOptions: Record<string, unknown>,
  options?: Record<string, unknown>
): Promise<unknown>;
