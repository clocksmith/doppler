# Doppler API Surface

## src/index.ts

### Classes

#### AdapterManager

```typescript
class AdapterManager {
  setDefaultLoadOptions(options: LoRALoadOptions): void;
  setEvents(events: AdapterManagerEvents): void;
  setStackOptions(options: Partial<AdapterStackOptions>): void;
  loadAdapter(id: string, path: string, options?: LoRALoadOptions): Promise<AdapterState>;
  registerAdapter(id: string, adapter: LoRAAdapter, manifest: AdapterManifest): AdapterState;
  enableAdapter(id: string, options?: EnableAdapterOptions): void;
  disableAdapter(id: string): void;
  toggleAdapter(id: string): boolean;
  disableAll(): void;
  enableOnly(id: string, options?: EnableAdapterOptions): void;
  setAdapterWeight(id: string, weight: number): void;
  unloadAdapter(id: string): void;
  unloadAll(): void;
  getActiveAdapter(): LoRAAdapter;
  getActiveAdapterIds(): string[];
  getAdapterState(id: string): AdapterState;
  getAllAdapters(): AdapterState[];
  getEnabledAdapters(): AdapterState[];
  isLoaded(id: string): boolean;
  isEnabled(id: string): boolean;
}
```

#### AdapterRegistry

```typescript
class AdapterRegistry {
  constructor(storage?: RegistryStorage): void;
  register(manifest: AdapterManifest, location: { storageType: "opfs" | "indexeddb" | "url"; manifestPath: string; weightsPath?: string; }): Promise<AdapterRegistryEntry>;
  registerFromUrl(url: string): Promise<AdapterRegistryEntry>;
  unregister(id: string): Promise<boolean>;
  clear(): Promise<void>;
  get(id: string): Promise<AdapterRegistryEntry>;
  list(options?: AdapterQueryOptions): Promise<AdapterRegistryEntry[]>;
  count(options?: Omit<AdapterQueryOptions, "offset" | "sortBy" | "sortOrder" | "limit">): Promise<number>;
  has(id: string): Promise<boolean>;
  getBaseModels(): Promise<string[]>;
  getTags(): Promise<string[]>;
  updateMetadata(id: string, metadata: Partial<AdapterMetadata>): Promise<AdapterRegistryEntry>;
  updateLocation(id: string, location: { storageType?: "opfs" | "indexeddb" | "url"; manifestPath?: string; weightsPath?: string; }): Promise<AdapterRegistryEntry>;
  exportToJSON(): Promise<string>;
  importFromJSON(json: string, options?: { overwrite?: boolean; merge?: boolean; }): Promise<{ imported: number; skipped: number; errors: string[]; }>;
}
```

#### DopplerLoader

```typescript
class DopplerLoader {
  memoryCapabilities: MemoryCapabilities;
  gpuCapabilities: KernelCapabilities;
  isUnifiedMemory: boolean;
  manifest: RDRRManifest;
  modelId: string;
  isMoE: boolean;
  isLoaded: boolean;
  embeddings: any;
  layers: Map<number, LayerWeights>;
  experts: Map<string, ExpertWeights>;
  lmHead: any;
  finalNorm: any;
  heapManager: HeapManager;
  gpuBuffers: Set<GPUBuffer>;
  expertCache: ExpertCache;
  loadedShards: Set<number>;
  tensorLocations: Map<string, TensorLocation>;
  shardCache: ShardCache;
  useFusedQ4K: boolean;
  q4kLayout: "flat" | "row_wise" | "column_wise";
  constructor(loadingConfig?: LoadingConfigSchema): void;
  setCustomShardLoader(loadShardFn: CustomShardLoader, options?: CustomShardLoaderOptions): void;
  init(): Promise<void>;
  setManifest(manifest: RDRRManifest): void;
  loadLoRAWeights(manifest: RDRRManifest): Promise<LoRAAdapter>;
  load(modelId: string, options?: LoadOptions): Promise<ModelConfig>;
  prefetchExperts(nextLayerIdx: number, expertIndices: number[]): void;
  predictNextLayerExperts(currentExperts: number[]): number[];
  loadExpert(layerIdx: number, expertIdx: number): Promise<ExpertWeights>;
  getLayerWeights(layerIdx: number): LayerWeights;
  getConfig(): ModelConfig;
  canRunDense(): boolean;
  getStats(): LoaderStats;
  getExpertCacheStats(): CacheStats;
  unload(): Promise<void>;
}
```

#### ExpertRouter

```typescript
class ExpertRouter {
  constructor(): void;
  registerExpert(profile: ExpertProfile): void;
  removeExpert(id: string): void;
  listExperts(): ExpertProfile[];
  selectByEmbedding(embedding: number[], topK?: number): ExpertProfile[];
}
```

#### InferencePipeline

```typescript
class InferencePipeline {
  tokenizer: Tokenizer;
  kvCache: KVCache | SlidingWindowKVCache;
  moeRouter: MoERouter;
  speculativeDecoder: SpeculativeDecoder;
  manifest: Manifest;
  modelConfig: ParsedModelConfig;
  weights: Map<string, any>;
  expertWeights: Map<string, ExpertWeights>;
  isLoaded: boolean;
  isGenerating: boolean;
  currentSeqLen: number;
  dopplerLoader: DopplerLoader;
  gpuContext: { device?: GPUDevice; };
  useGPU: boolean;
  memoryContext: Record<string, unknown>;
  storageContext: { loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array>; };
  stats: PipelineStats;
  batchingStats: BatchingStats;
  baseUrl: string;
  ropeFreqsCos: any;
  ropeFreqsSin: any;
  ropeLocalCos: any;
  ropeLocalSin: any;
  debug: boolean;
  useTiedEmbeddings: boolean;
  embeddingVocabSize: number;
  embeddingTranspose: boolean;
  layerRouterWeights: Map<number, RouterWeights>;
  loraAdapter: LoRAAdapter;
  constructor(): void;
  initialize(contexts?: PipelineContexts): Promise<void>;
  loadModel(manifest: Manifest): Promise<void>;
  setPreloadedWeights(weights: WeightLoadResult): void;
  generate(prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  prefillKVOnly(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  applyKVCacheSnapshot(snapshot: KVCacheSnapshot): void;
  generateWithPrefixKV(prefix: KVCacheSnapshot, prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  getStats(): PipelineStats;
  getBatchingStats(): BatchingStats;
  unload(): Promise<void>;
  setLoRAAdapter(adapter: LoRAAdapter): void;
  getActiveLoRA(): LoRAAdapter;
  reset(): void;
  releaseGPUResources(): void;
}
```

#### KVCache

```typescript
class KVCache {
  numLayers: number;
  numHeads: number;
  headDim: number;
  maxSeqLen: number;
  layout: "contiguous" | "paged";
  pageSize: number;
  kvDtype: "f16" | "f32";
  bytesPerElem: number;
  kvSize: number;
  windowSize: number;
  useGPU: boolean;
  layers: LayerCache[];
  currentSeqLen: number;
  memoryUsage: number;
  gpuContext: GPUContext;
  constructor(config: KVCacheConfig): void;
  update(layerIdx: number, keys: any, values: any, startPos?: number): void;
  updateFromGPU(layerIdx: number, keysBuffer: GPUBuffer, valuesBuffer: GPUBuffer, startPos: number, numTokens: number): void;
  recordUpdateFromGPU(encoder: GPUCommandEncoder, layerIdx: number, keysBuffer: GPUBuffer, valuesBuffer: GPUBuffer, startPos: number, numTokens: number): void;
  get(layerIdx: number, startPos?: number, endPos?: number): KVGetResult;
  getGPUBuffers(layerIdx: number): GPUBuffersResult;
  hasGPUCache(): boolean;
  clear(): void;
  clone(): KVCache;
  truncate(length: number): void;
  getMemoryStats(): MemoryStats;
  setGPUContext(gpuContext: GPUContext): void;
  syncToCPU(): Promise<void>;
  destroy(): void;
}
```

#### MoERouter

```typescript
class MoERouter {
  numExperts: number;
  topK: number;
  hiddenSize: number;
  normalizeWeights: boolean;
  gateWeight: any;
  gateBias: any;
  activeExperts: Set<number>;
  loadBalanceStats: LoadBalanceStats;
  constructor(config: MoEConfig): void;
  loadWeights(weights: any, bias?: any): void;
  computeRouterLogitsCPU(hiddenStates: Float32Array<ArrayBufferLike>, numTokens: number): Float32Array<ArrayBufferLike>;
  computeRouterLogitsGPU(hiddenStates: GPUBuffer, numTokens: number, gpuContext?: GpuContext): Promise<GPUBuffer>;
  routeGPU(hiddenStates: GPUBuffer, numTokens: number): Promise<ExpertSelection[]>;
  softmax(logits: Float32Array<ArrayBufferLike>, size: number): Float32Array<ArrayBufferLike>;
  selectExpertsForToken(logits: Float32Array<ArrayBufferLike>): ExpertSelection;
  route(hiddenStates: Float32Array<ArrayBufferLike>, numTokens: number): ExpertSelection[];
  getActiveExperts(): number[];
  computeLoadBalanceLoss(): number;
  resetStats(): void;
  getUtilizationStats(): UtilizationStats;
}
```

#### MultiModelLoader

```typescript
class MultiModelLoader {
  baseManifest: Manifest;
  baseWeights: WeightLoadResult;
  adapters: Map<string, LoRAAdapter>;
  loadBase(manifest: Manifest, options?: { storageContext?: { loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array>; }; }): Promise<WeightLoadResult>;
  loadAdapter(name: string, source: AdapterSource): Promise<LoRAAdapter>;
  getAdapter(name: string): LoRAAdapter;
  listAdapters(): string[];
  createSharedPipeline(contexts?: { gpu?: { device?: GPUDevice; }; storage?: { loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array>; }; }): Promise<InferencePipeline>;
}
```

#### MultiModelNetwork

```typescript
class MultiModelNetwork {
  constructor(pipeline: InferencePipeline, loader?: MultiModelLoader, pool?: MultiPipelinePool, recorder?: MultiModelRecorder): void;
  setRecorder(recorder: MultiModelRecorder): void;
  getRecorder(): MultiModelRecorder;
  setPipelinePool(pool: MultiPipelinePool): void;
  registerExpert(node: ExpertNode): void;
  getExpert(id: string): ExpertNode;
  listExperts(): ExpertNode[];
  setCombiner(config: CombinerConfig): void;
  setSharedPrefix(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  setSharedPrefixSnapshot(snapshot: KVCacheSnapshot): void;
  getSharedPrefixSnapshot(): KVCacheSnapshot;
  executeExpert(expertId: string, prompt: string, options?: GenerateOptions, overrides?: { adapterName?: string; adapter?: LoRAAdapter | null; prefix?: KVCacheSnapshot | null; usePool?: boolean; }): Promise<string>;
  executeChain(expertIds: string[], prompt: string, options?: GenerateOptions): Promise<string[]>;
  executeRing(expertIds: string[], prompt: string, options?: GenerateOptions): Promise<string[]>;
  executeCircularRing(expertIds: string[], prompt: string, options?: GenerateOptions, config?: { maxIterations?: number; convergenceThreshold?: number; }): Promise<{ output: string; iterations: number; converged: boolean; }>;
  executeTemporalRing(expertId: string, task: { description: string; maxTokens?: number; convergenceThreshold?: number; }, config?: { turns?: number; temperatureStart?: number; temperatureDecay?: number; temperatureMin?: number; shortcutInterval?: number; enableShortcuts?: boolean; }): Promise<{ finalOutput: string; history: Array<{ turn: number; output: string; timestamp: number; role: "seed" | "reflect" | "refine"; }>; turnsUsed: number; converged: boolean; }>;
  executeBatch(tasks: ExpertTask[], options?: GenerateOptions): Promise<Record<string, string>>;
  executeParallel(tasks: ExpertTask[], options?: GenerateOptions): Promise<Record<string, string>>;
  selectExpertsByEmbedding(embedding: number[], topK?: number): ExpertNode[];
  combineOutputs(outputs: string[], combinerOverride?: CombinerConfig): Promise<string>;
  executeGenome(genome: NetworkGenome, prompt: string, options?: GenerateOptions, router?: TopologyRouter): Promise<string>;
}
```

#### MultiPipelinePool

```typescript
class MultiPipelinePool {
  constructor(loader: MultiModelLoader, options?: MultiPipelinePoolOptions): void;
  setRecorder(recorder: MultiModelRecorder): void;
  getRecorder(): MultiModelRecorder;
  getPartitionedPool(): PartitionedBufferPool;
  setSharedPrefixSnapshot(snapshot: KVCacheSnapshot): void;
  getSharedPrefixSnapshot(): KVCacheSnapshot;
  getPipeline(id: string, contexts?: PipelineContexts): Promise<InferencePipeline>;
  listPipelines(): string[];
  warmPool(ids: string[], contexts?: PipelineContexts): Promise<void>;
  unloadAll(): Promise<void>;
  execute(id: string, prompt: string, options?: GenerateOptions, adapter?: LoRAAdapter, prefix?: KVCacheSnapshot): Promise<string>;
}
```

#### SpeculativeDecoder

```typescript
class SpeculativeDecoder {
  constructor(config?: SpeculativeConfig): void;
  setDraftModel(model: DraftModel): void;
  setMainModel(model: MainModel): void;
  generateDraftTokens(inputIds: number[], kvCache?: KVCache, numTokens?: number): Promise<DraftResult>;
  sampleToken(logits: Float32Array<ArrayBufferLike>, temperature?: number): SampleResult;
  verifyDraftTokens(inputIds: number[], draftTokens: number[], draftLogprobs: Float32Array<ArrayBufferLike>[], kvCache?: KVCache): Promise<VerificationResult>;
  step(inputIds: number[], mainKVCache?: KVCache, draftKVCache?: KVCache): Promise<StepResult>;
  getStats(): StatsWithSpeedup;
  resetStats(): void;
}
```

#### Tokenizer

```typescript
class Tokenizer {
  initialize(manifest: ModelManifest, options?: { baseUrl?: string; }): Promise<void>;
  encode(text: string): number[];
  decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string;
  getSpecialTokens(): SpecialTokens;
  getVocabSize(): number;
}
```

### Functions

#### computeLoRAScale

```typescript
function computeLoRAScale(rank: number, alpha: number): number
```

#### createDopplerLoader

```typescript
function createDopplerLoader(): DopplerLoader
```

#### createManifest

```typescript
function createManifest(options: MinimalAdapterManifest & Partial<AdapterManifest>): AdapterManifest
```

#### createMemoryRegistry

```typescript
function createMemoryRegistry(): AdapterRegistry
```

#### createPipeline

```typescript
function createPipeline(manifest: Manifest, contexts?: PipelineContexts): Promise<InferencePipeline>
```

#### evolveNetwork

```typescript
function evolveNetwork(config: EvolutionConfig): Promise<NetworkGenome>
```

#### getAdapterManager

```typescript
function getAdapterManager(): AdapterManager
```

#### getAdapterRegistry

```typescript
function getAdapterRegistry(): AdapterRegistry
```

#### getDopplerLoader

```typescript
function getDopplerLoader(): DopplerLoader
```

#### loadLoRAFromManifest

```typescript
function loadLoRAFromManifest(manifest: LoRAManifest, options?: LoRALoadOptions): Promise<LoRAAdapter>
```

#### loadLoRAFromSafetensors

```typescript
function loadLoRAFromSafetensors(data: ArrayBuffer, manifest: AdapterManifest): Promise<LoRAAdapter>
```

#### loadLoRAFromUrl

```typescript
function loadLoRAFromUrl(url: string, options?: LoRALoadOptions): Promise<LoRAAdapter>
```

#### loadLoRAWeights

```typescript
function loadLoRAWeights(path: string, options?: LoRALoadOptions): Promise<LoRAWeightsResult>
```

#### parseManifest

```typescript
function parseManifest(json: string): AdapterManifest
```

#### resetAdapterManager

```typescript
function resetAdapterManager(): void
```

#### resetAdapterRegistry

```typescript
function resetAdapterRegistry(): void
```

#### serializeManifest

```typescript
function serializeManifest(manifest: AdapterManifest, pretty?: boolean): string
```

#### validateManifest

```typescript
function validateManifest(manifest: unknown): ManifestValidationResult
```

### Constants

#### ADAPTER_MANIFEST_SCHEMA

```typescript
const ADAPTER_MANIFEST_SCHEMA: { readonly $schema: "http://json-schema.org/draft-07/schema#"; readonly $id: "https://doppler.dev/schemas/adapter-manifest.json"; readonly title: "Adapter Manifest"; readonly description: "Schema for LoRA adapter manifests in Doppler"; readonly type: "object"; readonly required: readonly ["id", "name", "baseModel", "rank", "alpha", "targetModules"]; readonly properties: { readonly id: { readonly type: "string"; readonly description: "Unique identifier for the adapter (UUID or slug)"; readonly pattern: "^[a-zA-Z0-9_-]+$"; }; readonly name: { readonly type: "string"; readonly description: "Human-readable name for the adapter"; readonly minLength: 1; readonly maxLength: 256; }; readonly version: { readonly type: "string"; readonly description: "Semantic version of the adapter"; readonly pattern: "^\\d+\\.\\d+\\.\\d+(-[a-zA-Z0-9.]+)?$"; readonly default: "1.0.0"; }; readonly description: { readonly type: "string"; readonly description: "Detailed description of the adapter purpose"; readonly maxLength: 4096; }; readonly baseModel: { readonly type: "string"; readonly description: "Identifier of the base model this adapter is trained for"; readonly examples: readonly ["gemma-3-1b", "llama-3-8b"]; }; readonly rank: { readonly type: "integer"; readonly description: "LoRA rank (dimensionality of the low-rank matrices)"; readonly minimum: 1; readonly maximum: 1024; }; readonly alpha: { readonly type: "number"; readonly description: "LoRA alpha scaling factor"; readonly minimum: 0.1; }; readonly targetModules: { readonly type: "array"; readonly description: "List of modules this adapter modifies"; readonly items: { readonly type: "string"; readonly enum: readonly ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "gate_up_proj"]; }; readonly minItems: 1; readonly uniqueItems: true; }; readonly checksum: { readonly type: "string"; readonly description: "SHA-256 or BLAKE3 hash of the weight file for integrity verification"; readonly pattern: "^[a-fA-F0-9]{64}$"; }; readonly checksumAlgorithm: { readonly type: "string"; readonly description: "Algorithm used for checksum"; readonly enum: readonly ["sha256", "blake3"]; readonly default: "sha256"; }; readonly weightsFormat: { readonly type: "string"; readonly description: "Format of the weight tensors"; readonly enum: readonly ["safetensors", "npz", "json", "binary"]; readonly default: "safetensors"; }; readonly weightsPath: { readonly type: "string"; readonly description: "Path or URL to the weights file (relative to manifest)"; }; readonly weightsSize: { readonly type: "integer"; readonly description: "Size of the weights file in bytes"; readonly minimum: 0; }; readonly tensors: { readonly type: "array"; readonly description: "Inline tensor specifications (for small adapters)"; readonly items: { readonly type: "object"; readonly required: readonly ["name", "shape"]; readonly properties: { readonly name: { readonly type: "string"; }; readonly shape: { readonly type: "array"; readonly items: { readonly type: "integer"; }; readonly minItems: 2; readonly maxItems: 2; }; readonly dtype: { readonly type: "string"; readonly enum: readonly ["f32", "f16", "bf16"]; readonly default: "f32"; }; readonly data: { readonly type: "array"; readonly items: { readonly type: "number"; }; }; readonly base64: { readonly type: "string"; }; readonly opfsPath: { readonly type: "string"; }; readonly url: { readonly type: "string"; }; }; }; }; readonly metadata: { readonly type: "object"; readonly description: "Additional metadata about the adapter"; readonly properties: { readonly author: { readonly type: "string"; }; readonly license: { readonly type: "string"; }; readonly tags: { readonly type: "array"; readonly items: { readonly type: "string"; }; }; readonly trainedOn: { readonly type: "string"; }; readonly epochs: { readonly type: "number"; }; readonly learningRate: { readonly type: "number"; }; readonly createdAt: { readonly type: "string"; readonly format: "date-time"; }; readonly updatedAt: { readonly type: "string"; readonly format: "date-time"; }; }; readonly additionalProperties: true; }; }; readonly additionalProperties: false; }
```

#### crossoverGenome

```typescript
const crossoverGenome: (a: NetworkGenome, b: NetworkGenome) => NetworkGenome
```

#### DOPPLER_VERSION

```typescript
const DOPPLER_VERSION: "0.1.0"
```

#### mutateGenome

```typescript
const mutateGenome: (genome: NetworkGenome, mutationRate?: number) => NetworkGenome
```

### Types

#### AdapterManagerEvents

```typescript
/**
 * Adapter manager events.
 */
export interface AdapterManagerEvents {
    onAdapterLoaded?: (id: string, adapter: LoRAAdapter) => void;
    onAdapterEnabled?: (id: string) => void;
    onAdapterDisabled?: (id: string) => void;
    onAdapterUnloaded?: (id: string) => void;
    onActiveAdaptersChanged?: (activeIds: string[]) => void;
}
```

#### AdapterManifest

```typescript
/**
 * Full adapter manifest structure.
 * This is the primary type for adapter definitions.
 */
export interface AdapterManifest {
    /** Unique identifier (UUID or slug) */
    id: string;
    /** Human-readable name */
    name: string;
    /** Semantic version (default: '1.0.0') */
    version?: string;
    /** Detailed description */
    description?: string;
    /** Base model identifier this adapter is compatible with */
    baseModel: string;
    /** LoRA rank (dimensionality) */
    rank: number;
    /** LoRA alpha scaling factor */
    alpha: number;
    /** List of modules this adapter modifies */
    targetModules: LoRAModuleName[];
    /** Content checksum for integrity verification */
    checksum?: string;
    /** Algorithm used for checksum */
    checksumAlgorithm?: 'sha256' | 'blake3';
    /** Format of weight tensors */
    weightsFormat?: 'safetensors' | 'npz' | 'json' | 'binary';
    /** Path or URL to weights file */
    weightsPath?: string;
    /** Size of weights file in bytes */
    weightsSize?: number;
    /** Inline tensor specifications */
    tensors?: AdapterTensorSpec[];
    /** Additional metadata */
    metadata?: AdapterMetadata;
}
```

#### AdapterMetadata

```typescript
/**
 * Adapter metadata for tracking provenance.
 */
export interface AdapterMetadata {
    /** Author or organization */
    author?: string;
    /** License identifier (e.g., 'MIT', 'Apache-2.0') */
    license?: string;
    /** Tags for categorization */
    tags?: string[];
    /** Description of training data */
    trainedOn?: string;
    /** Number of training epochs */
    epochs?: number;
    /** Training learning rate */
    learningRate?: number;
    /** Creation timestamp (ISO 8601) */
    createdAt?: string;
    /** Last update timestamp (ISO 8601) */
    updatedAt?: string;
    /** Additional custom metadata */
    [key: string]: unknown;
}
```

#### AdapterQueryOptions

```typescript
/**
 * Query options for listing adapters.
 */
export interface AdapterQueryOptions {
    /** Filter by base model */
    baseModel?: string;
    /** Filter by target modules (adapter must include all) */
    targetModules?: LoRAModuleName[];
    /** Filter by tags (adapter must include at least one) */
    tags?: string[];
    /** Sort field */
    sortBy?: 'name' | 'registeredAt' | 'lastAccessedAt';
    /** Sort direction */
    sortOrder?: 'asc' | 'desc';
    /** Maximum number of results */
    limit?: number;
    /** Offset for pagination */
    offset?: number;
}
```

#### AdapterRegistryEntry

```typescript
// ============================================================================
// Types
// ============================================================================
/**
 * Registry entry for a stored adapter.
 */
export interface AdapterRegistryEntry {
    /** Unique adapter ID */
    id: string;
    /** Human-readable name */
    name: string;
    /** Version string */
    version: string;
    /** Base model this adapter is for */
    baseModel: string;
    /** LoRA rank */
    rank: number;
    /** LoRA alpha */
    alpha: number;
    /** Target modules */
    targetModules: LoRAModuleName[];
    /** Storage location type */
    storageType: 'opfs' | 'indexeddb' | 'url';
    /** Path to manifest */
    manifestPath: string;
    /** Path to weights (if separate from manifest) */
    weightsPath?: string;
    /** Size of weights in bytes */
    weightsSize?: number;
    /** SHA-256 checksum */
    checksum?: string;
    /** Additional metadata */
    metadata?: AdapterMetadata;
    /** Registration timestamp */
    registeredAt: number;
    /** Last access timestamp */
    lastAccessedAt: number;
}
```

#### AdapterSource

```typescript
export type AdapterSource = LoRAAdapter | LoRAManifest | RDRRManifest | string;
```

#### AdapterStackOptions

```typescript
/**
 * Options for adapter stacking/merging.
 */
export interface AdapterStackOptions {
    /** How to combine multiple adapters */
    strategy: 'sum' | 'weighted_sum' | 'sequential';
    /** Normalize weights to sum to 1.0 */
    normalizeWeights?: boolean;
}
```

#### AdapterState

```typescript
// ============================================================================
// Types
// ============================================================================
/**
 * State of a loaded adapter.
 */
export interface AdapterState {
    /** Unique adapter identifier */
    id: string;
    /** The loaded adapter data */
    adapter: LoRAAdapter;
    /** Original manifest */
    manifest: AdapterManifest;
    /** Whether adapter is currently active */
    enabled: boolean;
    /** Weight multiplier for this adapter (default: 1.0) */
    weight: number;
    /** Load timestamp */
    loadedAt: number;
    /** Last enabled/disabled timestamp */
    lastToggled: number;
}
```

#### AdapterTensorSpec

```typescript
// ============================================================================
// TypeScript Type Definitions
// ============================================================================
/**
 * Tensor specification for inline weight data.
 */
export interface AdapterTensorSpec {
    /** Tensor name following pattern: layer.{N}.{module}.lora_{a|b} */
    name: string;
    /** Shape as [rows, cols] */
    shape: [
        number,
        number
    ];
    /** Data type (default: f32) */
    dtype?: 'f32' | 'f16' | 'bf16';
    /** Inline data as number array */
    data?: number[];
    /** Base64-encoded binary data */
    base64?: string;
    /** Path in OPFS storage */
    opfsPath?: string;
    /** URL to fetch tensor data from */
    url?: string;
}
```

#### EnableAdapterOptions

```typescript
/**
 * Options for enabling an adapter.
 */
export interface EnableAdapterOptions {
    /** Weight multiplier (0.0 - 2.0, default: 1.0) */
    weight?: number;
    /** Whether to validate base model compatibility */
    validateBaseModel?: boolean;
    /** Expected base model ID */
    expectedBaseModel?: string;
}
```

#### EvolutionConfig

```typescript
export interface EvolutionConfig {
    populationSize?: number;
    generations?: number;
    eliteCount?: number;
    mutationRate?: number;
    evaluate: (genome: NetworkGenome) => Promise<number>;
    randomGenome: () => NetworkGenome;
}
```

#### ExpertNode

```typescript
export interface ExpertNode extends ExpertProfile {
    adapterName?: string;
    adapter?: LoRAAdapter | null;
}
```

#### ExpertTask

```typescript
export interface ExpertTask {
    id: string;
    expertId: string;
    prompt: string;
}
```

#### ExpertWeights

```typescript
/**
 * Weights for a single MoE expert.
 */
export interface ExpertWeights {
    gate: GPUBuffer | Float32Array;
    up: GPUBuffer | Float32Array;
    down: GPUBuffer | Float32Array;
}
```

#### GenerateOptions

```typescript
// ============================================================================
// TypeScript Interfaces
// ============================================================================
export interface GenerateOptions {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
    repetitionPenalty?: number;
    stopSequences?: string[];
    useSpeculative?: boolean;
    onToken?: ((tokenId: number, text: string) => void) | null;
    useChatTemplate?: boolean;
    decode?: (tokens: number[]) => string;
    debug?: boolean;
    /** Specific layers to debug (enables batching with selective checkpoints).
     *  If set, CommandRecorder stays enabled but flushes at these layers.
     *  Example: [0, 12, 25] debugs first, middle, and last layers only. */
    debugLayers?: number[];
    signal?: AbortSignal;
    /** Enable GPU timestamp profiling for kernel-level timing.
     *  Requires 'timestamp-query' WebGPU feature. Results logged to console. */
    profile?: boolean;
    /** Log benchmark stats (TTFT, prefill time, decode speed) after generation.
     *  Default: false */
    benchmark?: boolean;
    /** Explicitly disable GPU command batching for debugging.
     *  When true, each GPU operation is submitted individually.
     *  Default: false */
    disableBatching?: boolean;
    // Batch generation options
    /** Number of tokens to generate per GPU submission batch.
     *  Default: 1 (single-token mode for backward compatibility)
     *  Higher values reduce GPU sync overhead but delay token streaming. */
    batchSize?: number;
    /** Callback invoked after each batch completes.
     *  Receives array of {id, text} pairs for the batch. */
    onBatch?: ((tokens: Array<{
        id: number;
        text: string;
    }>) => void) | null;
    /** Stop condition checking mode for batched generation.
     *  - 'batch': Check stop conditions after entire batch (faster, may overshoot by up to batchSize-1)
     *  - 'per-token': Check after each token using GPU kernel (accurate, default)
     *  Default: 'per-token' */
    stopCheckMode?: 'batch' | 'per-token';
}
```

#### GenerationResult

```typescript
/**
 * Result of text generation.
 */
export interface GenerationResult {
    /** All token IDs (prompt + generated) */
    tokens: number[];
    /** Generated text (excluding prompt) */
    text: string;
    /** Why generation stopped */
    finishReason: 'stop' | 'length' | 'eos';
    /** Performance statistics */
    stats: {
        prefillTimeMs: number;
        decodeTimeMs: number;
        totalTimeMs: number;
        tokensGenerated: number;
    };
}
```

#### KVCacheSnapshot

```typescript
export interface KVCacheSnapshot {
    cache: KVCache;
    seqLen: number;
    tokens: number[];
}
```

#### LayerWeights

```typescript
/**
 * Weights for a single transformer layer.
 */
export interface LayerWeights {
    // Attention
    inputNorm: GPUBuffer | Float32Array;
    qProj: GPUBuffer | Float32Array;
    kProj: GPUBuffer | Float32Array;
    vProj: GPUBuffer | Float32Array;
    oProj: GPUBuffer | Float32Array;
    /** Fused Q/K/V projection (runtime-generated for 3→1 matmul optimization) */
    qkvProj?: GPUBuffer | null;
    /** Sizes for splitting fused QKV output: [qSize, kSize, vSize] in elements */
    qkvSizes?: [
        number,
        number,
        number
    ];
    /** Data type of fused QKV weights (f16 or f32) */
    qkvDtype?: 'f16' | 'f32';
    // FFN (dense layers)
    postAttentionNorm?: GPUBuffer | Float32Array;
    postAttnNorm?: GPUBuffer | Float32Array; // LLaMA-style pre-FFN norm
    gate?: GPUBuffer | Float32Array;
    up?: GPUBuffer | Float32Array;
    down?: GPUBuffer | Float32Array;
    gateUp?: GPUBuffer | Float32Array; // Fused gate+up for 2-pass FFN
    // Sandwich norms (Gemma 3)
    preFeedforwardNorm?: GPUBuffer | Float32Array;
    postFeedforwardNorm?: GPUBuffer | Float32Array;
    // MoE
    routerWeight?: GPUBuffer | Float32Array;
    routerBias?: GPUBuffer | Float32Array | null;
    qNorm?: GPUBuffer | Float32Array;
    kNorm?: GPUBuffer | Float32Array;
    experts?: ExpertWeights[];
}
```

#### LoaderStats

```typescript
/**
 * Loader statistics
 */
export interface LoaderStats {
    modelId: string | null;
    isLoaded: boolean;
    isMoE: boolean;
    isUnifiedMemory: boolean;
    layersLoaded: number;
    expertsLoaded: number;
    gpuBuffers: number;
}
```

#### LoadOptions

```typescript
/**
 * Loading options
 */
export interface LoadOptions {
    onProgress?: (progress: LoadProgress) => void;
    verifyHashes?: boolean;
}
```

#### LoadProgress

```typescript
/**
 * Loading progress information
 */
export interface LoadProgress {
    stage: 'manifest' | 'shards' | 'layers' | 'gpu_transfer' | 'complete';
    progress: number;
    /** Current layer index */
    layer?: number;
    /** Total layers */
    total?: number;
    /** Current shard index */
    shard?: number;
    /** Total shards */
    totalShards?: number;
    /** Bytes loaded so far */
    bytesLoaded?: number;
    /** Total bytes to load */
    totalBytes?: number;
    /** Loading speed in bytes per second */
    bytesPerSecond?: number;
    /** Human-readable message */
    message?: string;
}
```

#### LoRAAdapter

```typescript
export interface LoRAAdapter {
    name: string;
    version?: string;
    baseModel?: string;
    rank: number;
    alpha: number;
    targetModules?: LoRAModuleName[];
    layers: Map<number, LoRALayerMap>;
}
```

#### LoRALoadOptions

```typescript
/**
 * Options for loading LoRA weights.
 */
export interface LoRALoadOptions {
    /** Function to read from OPFS storage */
    readOPFS?: (path: string) => Promise<ArrayBuffer>;
    /** Function to write to OPFS storage */
    writeOPFS?: (path: string, data: ArrayBuffer) => Promise<void>;
    /** Function to fetch from URL */
    fetchUrl?: (url: string) => Promise<ArrayBuffer>;
    /** Skip checksum verification */
    skipVerify?: boolean;
    /** Progress callback */
    onProgress?: (loaded: number, total: number) => void;
}
```

#### LoRAModuleName

```typescript
export type LoRAModuleName = 'q_proj' | 'k_proj' | 'v_proj' | 'o_proj' | 'gate_proj' | 'up_proj' | 'down_proj' | 'gate_up_proj';
```

#### LoRAWeightsResult

```typescript
/**
 * Result of loading LoRA weights.
 */
export interface LoRAWeightsResult {
    adapter: LoRAAdapter;
    manifest: AdapterManifest;
    loadedFromCache: boolean;
    checksumValid?: boolean;
}
```

#### NetworkGenome

```typescript
export interface NetworkGenome {
    topology: {
        type: 'chain' | 'ring' | 'tree' | 'mesh' | 'dag';
        depth?: number;
        branchingFactor?: number;
        maxIterations?: number; // For circular ring
    };
    nodes: NetworkNodeGene[];
    edges: NetworkEdgeGene[];
    combiner: {
        type: 'weighted' | 'voting' | 'llm-merge';
        weights?: number[];
        combinerExpertId?: string;
    };
}
```

#### ParsedModelConfig

```typescript
export interface ParsedModelConfig {
    numLayers: number;
    hiddenSize: number;
    intermediateSize: number;
    numHeads: number;
    numKVHeads: number;
    headDim: number;
    vocabSize: number;
    maxSeqLen: number;
    useMoE: boolean;
    numExperts: number;
    moeTopK: number;
    slidingWindow: number | null;
    ropeTheta: number;
    ropeLocalTheta: number | null; // For local/sliding attention layers (Gemma 3: 10K vs 1M global)
    ropeScale: number;
    ropeScalingType: string | null;
    ropeScaling: RopeScalingConfig | null;
    quantization: string;
    quantMethod: string | null;
    rmsNormEps: number;
    rmsNormWeightOffset: boolean;
    scaleEmbeddings: boolean;
    hiddenActivation: ActivationType;
    isGemma3: boolean;
    isGemma2: boolean;
    isLlama3Instruct: boolean;
    isQwen3: boolean;
    isGptOss: boolean;
    stopTokenIds: number[];
    layerTypes: string[] | null;
    attentionBias: boolean;
    embeddingScale?: number;
    // Gemma 2 softcapping
    finalLogitSoftcapping: number | null; // Gemma 2: 30.0
    attnLogitSoftcapping: number | null; // Gemma 2: 50.0
    // Gemma 2 attention scaling: uses head_dim (256) instead of sqrt(head_dim) (16)
    queryPreAttnScalar: number; // Gemma 2: 256, standard: sqrt(head_dim)
}
```

#### RDRRManifest

```typescript
export interface RDRRManifest {
    version: number | string;
    modelId: string;
    modelType: ModelType;
    quantization: string;
    hashAlgorithm?: HashAlgorithm;
    architecture: LayerConfig | string;
    groups?: Record<string, ComponentGroup>;
    shards: ShardInfo[];
    totalSize: number;
    tensorsFile?: string;
    tensorCount?: number;
    tokenizer?: {
        type: string;
        file: string;
        vocabSize: number;
    };
    moeConfig?: MoEConfig;
    optimizations?: RuntimeOptimizations;
    config?: Record<string, unknown>;
    conversion?: ConversionInfo;
    blake3Full?: string;
    defaultWeightLayout?: WeightLayout;
    metadata?: Record<string, unknown>;
    adapterType?: 'lora';
    baseModel?: string;
    loraConfig?: LoRAConfig;
    /** @deprecated Use tensorsFile */
    tensors?: Record<string, TensorLocation>;
    /** @deprecated Use modelId */
    name?: string;
}
```

#### RouterWeights

```typescript
/**
 * Router weights for MoE layers.
 */
export interface RouterWeights {
    weight: GPUBuffer | Float32Array;
    bias?: GPUBuffer | Float32Array | null;
}
```

#### SamplingOptions

```typescript
export interface SamplingOptions {
    temperature: number;
    topP: number;
    topK: number;
    decode?: (tokens: number[]) => string;
    debug?: boolean;
}
```

#### ShardInfo

```typescript
// =============================================================================
// Manifest Types
// =============================================================================
export interface ShardInfo extends Omit<ShardSchema, 'hashAlgorithm'> {
    blake3?: string;
    hashAlgorithm?: HashAlgorithm;
}
```

#### TensorLocation

```typescript
/**
 * Loader Types
 *
 * Type definitions for the DopplerLoader.
 *
 * @module loader/loader-types
 */
/**
 * Tensor location in loaded model
 */
export interface TensorLocation {
    shardIndex: number;
    offset: number;
    size: number;
    shape: number[];
    dtype: string;
    spans?: Array<{
        shardIndex: number;
        offset: number;
        size: number;
    }>;
    /** Weight storage layout: 'column' means pre-transposed for faster matmul */
    layout?: 'row' | 'column';
    /** Original shape before transpose (if layout is 'column') */
    originalShape?: number[];
}
```
