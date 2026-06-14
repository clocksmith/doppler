export interface DiffusionGemmaRouterContract {
  scaleHiddenStates: boolean;
  normalizeTopK: boolean;
  perExpertScale: boolean;
}

export interface DiffusionGemmaConfig {
  canvasLength: number;
  maxDenoisingSteps: number;
  maxNewTokens: number;
  tMin: number;
  tMax: number;
  entropyBound: number;
  confidenceThreshold: number;
  stabilityThreshold: number;
  padTokenId: number;
  eosTokenIds: number[];
  boiTokenId: number | null;
  eoiTokenId: number | null;
  imageTokenId: number | null;
  selfConditioning: boolean;
  decoderCacheMode: 'encoder_kv_readonly_canvas_concat';
  router: DiffusionGemmaRouterContract;
  vocabSize: number;
}

export function parseDiffusionGemmaConfig(manifest: unknown): DiffusionGemmaConfig;
