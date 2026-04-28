export interface RuntimeProfileSignals {
  trace: boolean;
  profiler: boolean;
  probes: boolean;
  debugTokens: boolean;
  benchmark: boolean;
}

export interface RuntimeProfileSummary {
  id: string;
  name: string | null;
  description: string | null;
  intent: string | null;
  toolingIntent: string | null;
  stability: string | null;
  owner: string | null;
  createdAtUtc: string | null;
  supersedes: string | null;
  replacementId: string | null;
  deprecatedAtUtc: string | null;
  extends: string | string[] | null;
  modelId: string | null;
  model: string | null;
  path: string;
  runtimePath: string;
  signals: RuntimeProfileSignals;
}

export interface RuntimeProfileListResult {
  ok: true;
  schemaVersion: 1;
  profileRoot: string;
  profiles: RuntimeProfileSummary[];
}

export declare function listRuntimeProfiles(options?: {
  rootDir?: string;
}): Promise<RuntimeProfileListResult>;

export declare function formatRuntimeProfiles(result: RuntimeProfileListResult): string;
