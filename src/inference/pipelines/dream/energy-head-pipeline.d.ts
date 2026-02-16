export type DreamEnergyHeadModelType =
  | 'dream_energy_head'
  | 'dream-energy-head'
  | 'd1-to2-bridge-diffusion'
  | 'synthesis-mixer-diffusion'
  | 'ebrm-diffusion';

export type DreamEnergyHeadId = 'main' | 'local' | 'tree' | 'consistency';
export type DreamEnergyHeadActivation = 'sigmoid' | 'linear';
export type DreamEnergyHeadBackend = 'auto' | 'gpu' | 'cpu';
export type DreamEnergyHeadDtype = 'f32' | 'f16';

export interface DreamEnergyHeadScoreRowInput {
  rowId?: string | number;
  candidateId?: string;
  features: number[] | Record<string, number>;
}

export interface DreamEnergyHeadScoreRow {
  rowId: string;
  score: number;
  logit: number;
  energy: number;
}

export interface DreamEnergyHeadInferRequest {
  rows: DreamEnergyHeadScoreRowInput[];
  head?: DreamEnergyHeadId;
  activation?: DreamEnergyHeadActivation;
  backend?: DreamEnergyHeadBackend;
  dtype?: DreamEnergyHeadDtype;
  steps?: number;
  stepSize?: number;
  gradientScale?: number;
  energyScale?: number;
}

export interface DreamEnergyHeadInferResult {
  modelId: string;
  modelHash: unknown;
  backend: 'gpu' | 'cpu';
  head: DreamEnergyHeadId | string;
  activation: DreamEnergyHeadActivation | string;
  rows: DreamEnergyHeadScoreRow[];
  totalTimeMs: number;
}

export interface DreamEnergyHeadStats {
  backend?: 'gpu' | 'cpu';
  rowCount?: number;
  totalTimeMs?: number;
  steps?: number;
  activation?: string;
  head?: string;
}

export declare class DreamEnergyHeadPipeline {
  runtimeConfig: Record<string, unknown> | null;
  manifest: Record<string, unknown> | null;
  model: Record<string, unknown> | null;
  stats: DreamEnergyHeadStats;
  baseUrl: string | null;
  _onProgress: ((progress: { stage?: string; percent: number; message?: string }) => void) | null;

  initialize(contexts?: Record<string, unknown>): Promise<void>;
  loadModel(manifest: Record<string, unknown>): Promise<void>;
  getStats(): DreamEnergyHeadStats;
  getMemoryStats(): { used: number; kvCache: null };
  unload(): Promise<void>;
  scoreRows(request: DreamEnergyHeadInferRequest): Promise<DreamEnergyHeadInferResult>;
  infer(request: DreamEnergyHeadInferRequest): Promise<DreamEnergyHeadInferResult>;
}

export declare function createDreamEnergyHeadPipeline(
  manifest: Record<string, unknown>,
  contexts?: Record<string, unknown>
): Promise<DreamEnergyHeadPipeline>;
