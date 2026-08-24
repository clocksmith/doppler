import type { RuntimeConfigLoadOptions } from '../../inference/browser-harness.js';
import type {
  ToolingCommandRequest,
  ToolingCommandRequestInput,
} from '../command-api.js';
import type { BrowserCommandRunResult } from '../browser-command-runner.js';
import type { StaticMount } from './transport.js';

export interface NodeBrowserCommandRunOptions {
  staticRootDir?: string;
  staticMounts?: StaticMount[];
  baseUrl?: string;
  host?: string;
  port?: number;
  headless?: boolean | string;
  channel?: string;
  executablePath?: string;
  runnerPath?: string;
  timeoutMs?: number;
  browserArgs?: string[];
  runtimeLoadOptions?: RuntimeConfigLoadOptions;
  opfsCache?: boolean;
  userDataDir?: string;
  wipeCacheBeforeLaunch?: boolean;
  onConsole?: (entry: { type: string; text: string }) => void;
  afterOpfsCachePrime?: (context: {
    page: unknown;
    modelId: string;
    cacheResult: Record<string, unknown>;
  }) => Promise<void> | void;
}

export declare function runBrowserCommandInNode(
  commandRequest: ToolingCommandRequestInput,
  options?: NodeBrowserCommandRunOptions
): Promise<BrowserCommandRunResult>;
export declare function normalizeNodeBrowserCommand(
  commandRequest: ToolingCommandRequestInput
): ToolingCommandRequest;

