export interface DiagnosticsProfile {
  id: string;
  suite: string;
  runtimeProfile: string;
  label: string;
}

export declare function storeDiagnosticsSelection(mode: string, updates: any): void;
export declare function decodeDiagnosticsProfileId(profileId: string): { suite: string; runtimeProfile: string } | null;
export declare function syncDiagnosticsModeUI(mode: string): void;
export declare function normalizeRuntimeConfig(raw: any): any;
export declare function getDiagnosticsDefaultSuite(mode: string): string;
export declare function getDiagnosticsRuntimeConfig(): any;
export declare function refreshDiagnosticsRuntimeConfig(runtimeProfile?: string): Promise<any>;
export declare function syncDiagnosticsDefaultsForMode(mode: string): Promise<void>;
export declare function clearDiagnosticsOutput(): void;
export declare function renderDiagnosticsOutput(result: any, fallbackSuite: string, captureOutput: boolean): void;
export declare function updateDiagnosticsStatus(message: string, isError?: boolean): void;
export declare function updateDiagnosticsReport(text: string): void;
export declare function updateDiagnosticsGuidance(): void;
export declare function selectDiagnosticsModel(modelId: string): void;
export declare function updateRuntimeConfigStatus(runtimeProfile?: string): void;
export declare function handleRuntimeConfigFile(file: File): Promise<void>;
export declare function applyRuntimeConfigProfile(runtimeProfile: string): Promise<void>;
export declare function getMergedRuntimeOverride(): any;
export declare function applySelectedRuntimeProfile(): Promise<void>;
