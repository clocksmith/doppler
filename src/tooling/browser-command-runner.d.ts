import type { RuntimeConfigLoadOptions } from '../inference/browser-harness.js';
import type {
  ToolingCommandRequest,
  ToolingCommandRequestInput,
} from './command-api.js';
import type { ToolingSuccessEnvelope } from './command-envelope.js';

export interface BrowserCommandRunOptions {
  runtimeLoadOptions?: RuntimeConfigLoadOptions;
  convertHandler?: (
    request: ToolingCommandRequest
  ) => Promise<unknown> | unknown;
}

export interface BrowserCommandRunResult extends ToolingSuccessEnvelope<unknown> {
  ok: true;
  surface: 'browser';
  request: ToolingCommandRequest;
}

export declare function runBrowserCommand(
  commandRequest: ToolingCommandRequestInput,
  options?: BrowserCommandRunOptions
): Promise<BrowserCommandRunResult>;

export declare function normalizeBrowserCommand(
  commandRequest: ToolingCommandRequestInput
): ToolingCommandRequest;
