export type EnergyRowHeadModelType =
  | 'energy_row_head'
  | 'energy-row-head'
  | 'dream_energy_head'
  | 'dream-energy-head'
  | 'd1-to2-bridge-diffusion'
  | 'synthesis-mixer-diffusion'
  | 'ebrm-diffusion';

export type EnergyRowHeadId = 'main' | 'local' | 'tree' | 'consistency';
export type EnergyRowHeadActivation = 'sigmoid' | 'linear';
export type EnergyRowHeadBackend = 'auto' | 'gpu' | 'cpu';
export type EnergyRowHeadDtype = 'f32' | 'f16';

export interface EnergyRowHeadScoreRowInput {
  rowId?: string | number;
  candidateId?: string;
  features: number[] | Record<string, number>;
}

export interface EnergyRowHeadScoreRow {
  rowId: string;
  score: number;
  logit: number;
  energy: number;
}

export interface EnergyRowHeadInferRequest {
  rows: EnergyRowHeadScoreRowInput[];
  head?: EnergyRowHeadId;
  activation?: EnergyRowHeadActivation;
  backend?: EnergyRowHeadBackend;
  dtype?: EnergyRowHeadDtype;
  steps?: number;
  stepSize?: number;
  gradientScale?: number;
  energyScale?: number;
}

export interface EnergyRowHeadInferResult {
  modelId: string;
  modelHash: unknown;
  backend: 'gpu' | 'cpu';
  head: EnergyRowHeadId | string;
  activation: EnergyRowHeadActivation | string;
  rows: EnergyRowHeadScoreRow[];
  totalTimeMs: number;
}

export interface EnergyRowHeadStats {
  backend?: 'gpu' | 'cpu';
  rowCount?: number;
  totalTimeMs?: number;
  steps?: number;
  activation?: string;
  head?: string;
}

export declare class EnergyRowHeadPipeline {
  runtimeConfig: Record<string, unknown> | null;
  manifest: Record<string, unknown> | null;
  model: Record<string, unknown> | null;
  stats: EnergyRowHeadStats;
  baseUrl: string | null;
  _onProgress: ((progress: { stage?: string; percent: number; message?: string }) => void) | null;

  initialize(contexts?: Record<string, unknown>): Promise<void>;
  loadModel(manifest: Record<string, unknown>): Promise<void>;
  getStats(): EnergyRowHeadStats;
  getMemoryStats(): { used: number; kvCache: null };
  unload(): Promise<void>;
  scoreRows(request: EnergyRowHeadInferRequest): Promise<EnergyRowHeadInferResult>;
  infer(request: EnergyRowHeadInferRequest): Promise<EnergyRowHeadInferResult>;
}

export declare function createEnergyRowHeadPipeline(
  manifest: Record<string, unknown>,
  contexts?: Record<string, unknown>
): Promise<EnergyRowHeadPipeline>;

export type DreamEnergyHeadModelType = EnergyRowHeadModelType;

export type DreamEnergyHeadId = EnergyRowHeadId;

export type DreamEnergyHeadActivation = EnergyRowHeadActivation;

export type DreamEnergyHeadBackend = EnergyRowHeadBackend;

export type DreamEnergyHeadDtype = EnergyRowHeadDtype;

export type DreamEnergyHeadScoreRowInput = EnergyRowHeadScoreRowInput;

export type DreamEnergyHeadScoreRow = EnergyRowHeadScoreRow;

export type DreamEnergyHeadInferRequest = EnergyRowHeadInferRequest;

export type DreamEnergyHeadInferResult = EnergyRowHeadInferResult;

export type DreamEnergyHeadStats = EnergyRowHeadStats;

export declare class DreamEnergyHeadPipeline extends EnergyRowHeadPipeline {}

export declare const createDreamEnergyHeadPipeline: typeof createEnergyRowHeadPipeline;
