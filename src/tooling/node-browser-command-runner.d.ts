import type { RuntimeConfigLoadOptions } from '../inference/browser-harness.js';
import type {
  ToolingCommandRequest,
  ToolingCommandRequestInput,
} from './command-api.js';
import type { BrowserCommandRunResult } from './browser-command-runner.js';

export interface NodeBrowserCommandRunOptions {
  staticRootDir?: string;
  staticMounts?: Array<{
    urlPrefix: string;
    rootDir: string;
  }>;
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
  onConsole?: (entry: { type: string; text: string }) => void;
}

export interface BrowserRelayLocalModelResolution {
  relayRequest: ToolingCommandRequestInput;
  staticMounts: Array<{
    urlPrefix: string;
    rootDir: string;
  }>;
}

export declare function resolveLocalFileModelUrlForBrowserRelay(
  commandRequest: ToolingCommandRequestInput,
  options?: Pick<NodeBrowserCommandRunOptions, 'baseUrl' | 'staticMounts'>
): Promise<BrowserRelayLocalModelResolution>;

export declare function runBrowserCommandInNode(
  commandRequest: ToolingCommandRequestInput,
  options?: NodeBrowserCommandRunOptions
): Promise<BrowserCommandRunResult>;

export declare function normalizeNodeBrowserCommand(
  commandRequest: ToolingCommandRequestInput
): ToolingCommandRequest;

/**
 * Validate a browser-relay response envelope and stamp the original
 * request onto it. Throws when the envelope does not match the success
 * shape.
 */
export declare function finalizeBrowserRelayResponse(
  response: unknown,
  request: ToolingCommandRequest
): BrowserCommandRunResult & { request: ToolingCommandRequest };
