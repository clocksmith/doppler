import type { ExtensionBridgeClient } from '../../bridge/index.js';
import type { InferencePipeline, KVCacheSnapshot } from '../../inference/pipeline.js';
import type { LoRAManifest } from '../../adapters/lora-loader.js';
import type { RDRRManifest } from '../../storage/rdrr-format.js';
import type { RuntimeConfigSchema } from '../../config/schema/index.js';
import type { ConverterConfigSchema } from '../../config/schema/converter.schema.js';
import type { ConvertOptions } from '../../browser/browser-converter.js';
import type { TensorSource } from '../../browser/tensor-source-file.js';
import type { HttpTensorSourceOptions } from '../../browser/tensor-source-http.js';
import type {
  BrowserHarnessOptions,
  BrowserSuiteOptions,
  BrowserManifest,
  BrowserHarnessResult,
  BrowserSuiteResult,
  BrowserManifestResult,
  RuntimeConfigLoadOptions,
} from '../../inference/browser-harness.js';
import type { InitializeResult, RuntimeOverrides } from '../../inference/test-harness.js';
import type { SavedReportInfo, SaveReportOptions } from '../../storage/reports.js';

export declare const DOPPLER_PROVIDER_VERSION: string;

export interface TextModelConfig {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  vocabSize: number;
  maxSeqLen: number;
  quantization: string;
}

export interface InferredAttentionParams {
  numHeads: number;
  numKVHeads: number;
  headDim: number;
}

export interface ModelEstimate {
  weightsBytes: number;
  kvCacheBytes: number;
  totalBytes: number;
  modelConfig: TextModelConfig;
}

export interface LoadProgressEvent {
  stage: 'connecting' | 'manifest' | 'estimate' | 'warming' | 'downloading' | 'loading';
  message: string;
  estimate?: ModelEstimate;
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopTokens?: number[];
  stopSequences?: string[];
  useChatTemplate?: boolean;
  onToken?: (token: string) => void;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface DopplerProviderRuntime {
  getRuntimeConfig(): RuntimeConfigSchema;
  setRuntimeConfig(
    overrides?: Partial<RuntimeConfigSchema> | RuntimeConfigSchema
  ): RuntimeConfigSchema;
  resetRuntimeConfig(): RuntimeConfigSchema;
}

export interface DopplerProviderConversion {
  ConvertStage: typeof import('../../browser/browser-converter.js').ConvertStage;
  isConversionSupported(): boolean;
  createRemoteModelSources(
    urls: string[],
    options?: HttpTensorSourceOptions & { converterConfig?: ConverterConfigSchema }
  ): Promise<TensorSource[]>;
  convertModel(files: Array<File | TensorSource>, options?: ConvertOptions): Promise<string>;
  pickModelFiles(): Promise<File[]>;
}

export interface DopplerProviderBench {
  loadRuntimeConfigFromUrl(
    url: string,
    options?: RuntimeConfigLoadOptions
  ): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;
  applyRuntimeConfigFromUrl(
    url: string,
    options?: RuntimeConfigLoadOptions
  ): Promise<Record<string, unknown>>;
  loadRuntimePreset(
    presetId: string,
    options?: RuntimeConfigLoadOptions
  ): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;
  applyRuntimePreset(
    presetId: string,
    options?: RuntimeConfigLoadOptions
  ): Promise<Record<string, unknown>>;
  initializeBrowserHarness(
    options: BrowserHarnessOptions
  ): Promise<InitializeResult & { runtime: RuntimeOverrides }>;
  saveBrowserReport(
    modelId: string,
    report: Record<string, unknown>,
    options?: SaveReportOptions
  ): Promise<SavedReportInfo>;
  runBrowserHarness(options: BrowserHarnessOptions): Promise<BrowserHarnessResult>;
  runBrowserSuite(options: BrowserSuiteOptions): Promise<BrowserSuiteResult>;
  runBrowserManifest(
    manifest: BrowserManifest,
    options?: RuntimeConfigLoadOptions & {
      saveReport?: boolean;
      timestamp?: string | Date;
      onProgress?: (progress: { index: number; total: number; label: string }) => void;
    }
  ): Promise<BrowserManifestResult>;
}

export interface DopplerCapabilitiesType {
  available: boolean;
  HAS_MEMORY64: boolean;
  HAS_SUBGROUPS: boolean;
  HAS_F16: boolean;
  IS_UNIFIED_MEMORY: boolean;
  TIER_LEVEL: number;
  TIER_NAME: string;
  MAX_MODEL_SIZE: number;
  initialized: boolean;
  currentModelId: string | null;
  kernelsWarmed: boolean;
  kernelsTuned: boolean;
  lastModelEstimate: ModelEstimate | null;
  bridgeClient?: ExtensionBridgeClient | null;
  localPath?: string | null;
}

export interface DopplerProviderInterface {
  name: string;
  displayName: string;
  isLocal: boolean;
  init(): Promise<boolean>;
  loadModel(
    modelId: string,
    modelUrl?: string | null,
    onProgress?: ((event: LoadProgressEvent) => void) | null,
    localPath?: string | null
  ): Promise<boolean>;
  chat(messages: ChatMessage[], options?: GenerateOptions): Promise<ChatResponse>;
  stream(messages: ChatMessage[], options?: GenerateOptions): AsyncGenerator<string>;
  prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  generateWithPrefixKV(prefix: KVCacheSnapshot, prompt: string, options?: GenerateOptions): AsyncGenerator<string>;
  loadLoRAAdapter(adapter: LoRAManifest | RDRRManifest | string): Promise<void>;
  unloadLoRAAdapter(): Promise<void>;
  getActiveLoRA(): string | null;
  getPipeline(): InferencePipeline | null;
  getCurrentModelId(): string | null;
  extractTextModelConfig(manifest: RDRRManifest): TextModelConfig;
  readOPFSFile(path: string): Promise<ArrayBuffer>;
  writeOPFSFile(path: string, data: ArrayBuffer): Promise<void>;
  fetchArrayBuffer(url: string): Promise<ArrayBuffer>;
  getCapabilities(): DopplerCapabilitiesType;
  getModels(): Promise<string[]>;
  getAvailableModels(): Promise<string[]>;
  getDopplerStorageInfo(): Promise<unknown>;
  runtime: DopplerProviderRuntime;
  conversion: DopplerProviderConversion;
  bench: DopplerProviderBench;
  destroy(): Promise<void>;
}

export declare const DopplerCapabilities: DopplerCapabilitiesType;
