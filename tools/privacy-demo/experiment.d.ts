export interface ExperimentReport {
  schema: string; qualified: false; modelId: string; protocol: string; output: string; matchesReference: boolean;
  referenceElapsedMs: number; steps: number; setupMs: number; stepMs: number[]; elapsedMs: number; clientBytes: number; providerBytes: number;
  minimumNoiseBits: number; maxRssKiB: number; adversary: string; leakage: string[];
  integrityProvided: false; productionBackendAvailable: false;
}
export interface ExperimentFrame { direction: 'client' | 'provider'; packet: Record<string, unknown> }
export declare function runExperiment(options: {seed: string; signal?: AbortSignal | null; observe?: (frame: ExperimentFrame) => void;
  transformResponse?: (packet: Record<string, unknown>) => Record<string, unknown>}): Promise<ExperimentReport>;
