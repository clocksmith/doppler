import type { RuntimeConfigLoadOptions } from '../inference/browser-harness.js';
import type {
  ToolingCommandRequest,
  ToolingCommandRequestInput,
} from './command-api.js';

export interface BrowserCommandRunOptions {
  runtimeLoadOptions?: RuntimeConfigLoadOptions;
  convertHandler?: (
    request: ToolingCommandRequest
  ) => Promise<unknown> | unknown;
}

export interface BrowserCommandRunResult {
  ok: true;
  surface: 'browser';
  request: ToolingCommandRequest;
  result: unknown;
}

export declare function runBrowserCommand(
  commandRequest: ToolingCommandRequestInput,
  options?: BrowserCommandRunOptions
): Promise<BrowserCommandRunResult>;

export declare function normalizeBrowserCommand(
  commandRequest: ToolingCommandRequestInput
): ToolingCommandRequest;
