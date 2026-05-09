export interface WrappedPipelineHandle {
  readonly loaded: boolean;
  readonly modelId: string;
  readonly manifest: Record<string, unknown> | null;
  readonly deviceInfo: Record<string, unknown> | null;
  generateText(prompt: unknown, opts?: Record<string, unknown>): Promise<string>;
  unload(): Promise<void>;
}

export declare function wrapPipelineAsHandle(
  pipeline: {
    generate: (...args: unknown[]) => AsyncIterable<unknown>;
    [key: string]: unknown;
  },
  resolved?: {
    modelId?: string;
    manifest?: Record<string, unknown>;
    deviceInfo?: Record<string, unknown>;
  }
): WrappedPipelineHandle;

export interface DreamProviderAttachLoRARequest {
  adapterId: string;
  version?: string;
  baseModel?: string;
  baseModelId?: string;
  rank: number;
  alpha?: number;
  scale?: number;
  targetModules?: string[];
  layers: Map<number, Record<string, {
    a: Float32Array | number[] | ArrayLike<number>;
    b: Float32Array | number[] | ArrayLike<number>;
  }>>;
}

export interface DreamProviderGenerateRequest {
  prompt: unknown;
  loraAdapterId?: string | null;
  samplingOptions?: Record<string, unknown>;
}

export interface DreamProviderGenerateResult {
  text: string;
  useLora: boolean;
  baseModelId: string;
  loraAdapterId: string | null;
}

export interface DreamCausalLmBaseProvider {
  readonly modelId: string;
  readonly manifest: Record<string, unknown> | null;
  readonly backend: 'doppler';
  readonly device: unknown | null;
  attachLoraAdapter(adapter: DreamProviderAttachLoRARequest): Promise<{
    adapterId: string;
    rank: number;
    alpha: number;
    scale: number;
    targetModules: string[];
    layerCount: number;
  }>;
  detachLoraAdapter(adapterId?: string | null): Promise<{
    detached: true;
    adapterId: string | null;
  }>;
  generate(request: string | DreamProviderGenerateRequest): Promise<DreamProviderGenerateResult>;
}

export declare function wrapPipelineAsDreamProvider(
  pipeline: {
    generate: (...args: unknown[]) => AsyncIterable<unknown>;
    setLoRAAdapter: (adapter: unknown | null) => void;
    getActiveLoRA: () => unknown | null;
    [key: string]: unknown;
  },
  resolved?: {
    modelId?: string;
    manifest?: Record<string, unknown>;
    device?: unknown | null;
  }
): DreamCausalLmBaseProvider;
