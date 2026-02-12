import type { RuntimeConfigLoadOptions } from '../inference/browser-harness.js';
import type {
  ToolingCommandRequest,
  ToolingCommandRequestInput,
} from './command-api.js';
import type { BrowserCommandRunResult } from './browser-command-runner.js';

export interface NodeBrowserCommandRunOptions {
  staticRootDir?: string;
  baseUrl?: string;
  host?: string;
  port?: number;
  headless?: boolean | string;
  channel?: string;
  executablePath?: string;
  runnerPath?: string;
  timeoutMs?: number;
  runtimeLoadOptions?: RuntimeConfigLoadOptions;
  onConsole?: (entry: { type: string; text: string }) => void;
}

export declare function runBrowserCommandInNode(
  commandRequest: ToolingCommandRequestInput,
  options?: NodeBrowserCommandRunOptions
): Promise<BrowserCommandRunResult>;

export declare function normalizeNodeBrowserCommand(
  commandRequest: ToolingCommandRequestInput
): ToolingCommandRequest;
