export interface ToolRunnerOptions {
  vfs?: {
    list: (prefix: string) => Promise<string[]>;
    readText: (path: string) => Promise<string | null>;
    exists?: (path: string) => Promise<boolean>;
  } | null;
  root?: string;
}

export interface ToolInfo {
  name: string;
  description: string | null;
  inputSchema: unknown | null;
}

export interface ToolRefreshResult {
  tools: string[];
  errors: Array<{ path: string; error: string }>;
}

export declare class ToolRunner {
  constructor(options?: ToolRunnerOptions);
  setVfs(vfs: ToolRunnerOptions['vfs']): void;
  list(): string[];
  getToolInfo(name: string): ToolInfo | null;
  refresh(options?: { force?: boolean }): Promise<ToolRefreshResult>;
  execute(name: string, args?: Record<string, unknown>): Promise<unknown>;
}
