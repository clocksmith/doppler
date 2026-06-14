import type { DiffusionGemmaConfig } from './config.js';
import type { DiffusionGemmaLogitsProvider } from './sampling.js';
import type { InferencePipeline } from '../text.js';

export interface DiffusionGemmaPipelineContexts {
  diffusionGemma?: {
    logitsProvider?: DiffusionGemmaLogitsProvider | null;
    corePipeline?: InferencePipeline | null;
    tokenizer?: {
      encode(input: string): ArrayLike<number>;
      decode(tokens: ArrayLike<number>): string;
    } | null;
  } | null;
}

export interface DiffusionGemmaGenerateOptions {
  inputIds?: ArrayLike<number> | null;
  maxNewTokens?: number | null;
  canvasLength?: number | null;
  maxDenoisingSteps?: number | null;
  seed?: number | null;
  random?: (() => number) | null;
  logitsProvider?: DiffusionGemmaLogitsProvider | null;
  initialCanvas?: ArrayLike<number> | null;
  selfConditioningLogits?: ArrayBufferView | null;
  decodeToken?: ((tokenId: number) => string) | null;
}

export interface DiffusionGemmaStats {
  canvasesGenerated: number;
  tokensGenerated: number;
  denoiseSteps: number;
  modelLoadMs: number;
  totalTimeMs: number;
  prefillTimeMs: number;
  decodeTimeMs: number;
  prefillTokens: number;
  decodeTokens: number;
  tokensPerForward: number;
  stopReason: string | null;
  stopTokenId: number | null;
}

export class DiffusionGemmaPipeline {
  manifest: unknown | null;
  config: DiffusionGemmaConfig | null;
  runtimeConfig: unknown | null;
  runtimeOverrides: unknown | null;
  tokenizer: unknown | null;
  logitsProvider: DiffusionGemmaLogitsProvider | null;
  corePipeline: InferencePipeline | null;
  ownsCorePipeline: boolean;
  isLoaded: boolean;
  stats: DiffusionGemmaStats;

  initialize(contexts?: DiffusionGemmaPipelineContexts): Promise<void>;
  loadModel(manifest: unknown): Promise<void>;
  assertReady(): void;
  resolveCoreOptions(options: DiffusionGemmaGenerateOptions): DiffusionGemmaGenerateOptions & {
    __internalGenerate: true;
    useChatTemplate: false;
  };
  resetCoreEncoder(inputIds: ArrayLike<number>, options: DiffusionGemmaGenerateOptions): Promise<void>;
  appendCoreEncoderTokens(tokenIds: ArrayLike<number>, options: DiffusionGemmaGenerateOptions): Promise<void>;
  generateTokenIds(prompt: string | ArrayLike<number>, options?: DiffusionGemmaGenerateOptions): Promise<Int32Array>;
  generate(prompt: string | ArrayLike<number>, options?: DiffusionGemmaGenerateOptions): AsyncGenerator<string>;
  getStats(): DiffusionGemmaStats;
  unload(): Promise<void>;
}
