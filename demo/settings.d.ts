export interface DemoSettingsSnapshot {
  temperature?: number;
  topK?: number;
  topP?: number;
  maxTokens?: number;
  runtimeProfile: string | null;
}

export declare function getSettings(): DemoSettingsSnapshot;

export declare function initSettings(options?: {
  requireDefaultProfile?: boolean;
}): Promise<boolean>;
