import type { FP8Policy } from './fp8.schema.js';
import type { W8A8Policy } from './w8a8.schema.js';
import type { ManifestRoPESchema } from './manifest-rope.schema.js';
import type { KernelPathRef } from "../execution/kernel-path.schema.js";
import type { LayerPipelineSchema, LayerType, LinearNormGateActivation, LinearNormMode } from "../execution/inference.schema.js";
import type { EnergyModelConfigSchema } from "../execution/energy.schema.js";
import type { HyperConnectionsSchema, IndexedAttentionSchema } from "../execution/hyper-connection-decoder.schema.js";
import type { NgramEmbeddingSchema } from "../execution/ngram-embedding.schema.js";
import type {
  ExecutionV1GraphSchema,
  ExecutionV1SessionSchema,
} from "../execution/execution-v1.schema.js";


/**
 * Attention configuration for inference.
 * All fields required - converter must populate everything.
 * Use `null` to indicate "not applicable" (e.g., no softcapping).
 */
export interface ManifestAttentionSchema {
  /** Query pre-attention scalar: attnScale = 1/sqrt(scalar). Standard = headDim. */
  queryPreAttnScalar: number;
  /** Multiplier applied to Q after optional Q/K normalization and before RoPE. */
  queryScale: number;
  /** Attention logit softcapping (Gemma 2: 50, null = disabled) */
  attnLogitSoftcapping: number | null;
  /** Sliding window size for local attention (null = full attention) */
  slidingWindow: number | null;
  /** Query-key normalization */
  queryKeyNorm: boolean;
  /** Numerical family used for Q/K normalization. */
  queryKeyNormType: 'rmsnorm' | 'layernorm';
  /** Normalize each head independently or the complete projection before head reshaping. */
  queryKeyNormAxis: 'head' | 'projection';
  /** Layers that apply query-key normalization; null means all layers when queryKeyNorm=true */
  queryKeyNormLayers: number[] | null;
  /** Layers that have explicit Q/K RMSNorm scale tensors; null means every normalized layer must have weights */
  queryKeyNormWeightLayers: number[] | null;
  /** Value RMSNorm with implicit unit scale (Gemma 4 text attention) */
  valueNorm: boolean;
  /** Whether attention mask is causal (false = bidirectional attention) */
  causal: boolean;
  /** Attention bias mask enabled */
  attentionBias: boolean;
  /** Apply sigmoid gate from q_proj second half before o_proj */
  attentionOutputGate: boolean;
}

/**
 * Normalization configuration for inference.
 * Controls transformer normalization and sandwich norm architecture.
 */
export interface ManifestNormalizationSchema {
  /** Normalization family used by transformer and final norms. */
  type: 'rmsnorm' | 'layernorm';
  /** RMSNorm epsilon for numerical stability (default: 1e-5) */
  rmsNormEps: number;
  /** Use (1 + weight) pattern for RMSNorm (Gemma models) */
  rmsNormWeightOffset: boolean;
  /** Post-sublayer RMSNorm epsilon; null inherits rmsNormEps. */
  postNormEps: number | null;
  /** Post-sublayer centered-weight offset; null inherits rmsNormWeightOffset. */
  postNormWeightOffset: boolean | null;
  /** Has post-attention normalization (sandwich norm) */
  postAttentionNorm: boolean;
  /** Has pre-feedforward normalization (sandwich norm) */
  preFeedforwardNorm: boolean;
  /** Has post-feedforward normalization (sandwich norm) */
  postFeedforwardNorm: boolean;
  /** Exact final LayerNorm bias tensor; null for RMSNorm or bias-free LayerNorm. */
  finalNormBiasTensor: string | null;
  /**
   * Activation applied to z by the linear-attention gated RMSNorm. Source-owned:
   * runtime overrides are rejected. Manifests converted before this field
   * existed omit it; merge resolves the omission to the schema default `silu`
   * (see resolveLinearNormGateActivation), never to another value.
   */
  linearNormGateActivation?: LinearNormGateActivation;
}

/**
 * FFN configuration for inference.
 */
export interface ManifestFFNSchema {
  /** Activation function type */
  activation: 'silu' | 'gelu' | 'geglu' | 'swiglu' | 'relu';
  /** Whether activation is gated (e.g., SwiGLU, GeGLU) */
  gatedActivation: boolean;
  /** FFN branch composition. `auto` preserves legacy dense-or-MoE routing. */
  branchMode: 'auto' | 'dense' | 'moe' | 'dense-plus-moe';
  /** Double the FFN intermediate width on KV-shared layers. */
  useDoubleWideMlp: boolean;
  /** Clamp SwiGLU output (null = disabled) */
  swigluLimit: number | null;
}

/**
 * Output configuration for inference.
 * All fields required - converter must populate everything.
 */
export interface ManifestOutputSchema {
  /** Optional opt-in model-owned reasoning/final output boundary. */
  reasoningOutput?: import("../../reasoning-output.js").ReasoningOutputContract | null;
  /** Final logit softcapping (Gemma 2: 30, null = disabled) */
  finalLogitSoftcapping: number | null;
  /** Whether embeddings and LM head share weights */
  tieWordEmbeddings: boolean;
  /** Scale embeddings by sqrt(hiddenSize) (Gemma models: true) */
  scaleEmbeddings: boolean;
  /** Explicit embedding multiplier, null to use scaleEmbeddings semantics. */
  embeddingScale: number | null;
  /** Optional weightless normalization applied to gathered embeddings. */
  embeddingNormalization: ManifestEmbeddingNormalizationSchema | null;
  /** Multiplier applied after final norm before LM head projection. */
  logitInputScale: number;
  /** Multiplier applied to LM-head logits before final softcapping. */
  logitOutputScale: number;
  /** Whether embedding weights are stored as [hidden, vocab] (transpose on gather) */
  embeddingTranspose: boolean;
  /** Embedding vocab size from weight tensor (null = use architecture.vocabSize) */
  embeddingVocabSize: number | null;
  /** Optional embedding-only postprocessor stack applied after pooled hidden states. */
  embeddingPostprocessor: ManifestEmbeddingPostprocessorSchema | null;
  /** Optional exact tensor name for an additive decoder/LM-head bias. */
  lmHeadBiasTensor: string | null;
}

export interface ManifestEmbeddingNormalizationSchema {
  type: 'rmsnorm';
  withScale: false;
  eps: number;
  position: 'after-scale';
}

export interface ManifestSequencePoolingSchema {
  mode: 'mean' | 'last';
  excludeTokenIds: number[];
}

export interface ManifestSequenceSchema {
  alphabet: 'amino-acid' | 'nucleotide';
  tokenEmbeddings: boolean;
  pooledEmbedding: ManifestSequencePoolingSchema | null;
  logits: boolean;
}

export interface ManifestEmbeddingProjectionSchema {
  weightTensor: string;
  biasTensor: string | null;
  inputSize: number;
  outputSize: number;
  activation: 'identity';
}

export interface ManifestEmbeddingPostprocessorSchema {
  poolingMode: 'mean' | 'last';
  includePrompt: boolean;
  projections: ManifestEmbeddingProjectionSchema[];
  normalize: 'l2' | null;
}

/**
 * Layer pattern for hybrid attention models.
 * Defines which layers use global vs sliding window attention.
 */
export interface ManifestLayerPatternSchema {
  /** Pattern type */
  type: 'uniform' | 'alternating' | 'every-n' | 'custom';
  /** For alternating: which layers are global ('odd' or 'even'), null if not applicable */
  globalPattern: 'odd' | 'even' | null;
  /** For every-n: period of global layers, null if not applicable */
  period: number | null;
  /** For every-n: first global layer index modulo period, null if not applicable */
  offset: number | null;
  /** For custom: explicit per-layer architecture tags, null if not applicable */
  layerTypes: LayerType[] | null;
  /** Multiplier applied to attention and FFN branches before residual add. */
  residualBranchScale: number;
}

/**
 * Chat template configuration.
 */
export interface ManifestChatTemplateSchema {
  /** Chat template type (null = no chat template) */
  type: 'gemma' | 'gemma4' | 'llama3' | 'gpt-oss' | 'chatml' | 'qwen' | 'glmocr' | 'translategemma' | null;
  /** Whether chat template is enabled */
  enabled: boolean;
  /** Thinking control passed to templates that declare it; null leaves it unspecified. */ thinking?: boolean | null;
}
export interface ManifestUnsupportedSchema {
  code?: string | null;
  message?: string | null;
  recommendation?: string | null;
}

export interface ManifestDiffusionGemmaSchema {
  /** Fixed decoder canvas length for block diffusion. */
  canvasLength: number;
  /** Maximum denoising iterations per canvas. */
  maxDenoisingSteps: number;
  /** Default generation cap in new tokens. */
  maxNewTokens: number;
  /** Lower bound for the linear denoising temperature schedule. */
  tMin: number;
  /** Upper bound for the linear denoising temperature schedule. */
  tMax: number;
  /** Entropy budget used by the entropy-bound sampler. */
  entropyBound: number;
  /** Mean-entropy threshold for adaptive canvas stopping. */
  confidenceThreshold: number;
  /** Number of stable argmax steps required before adaptive stopping can finish. */
  stabilityThreshold: number;
  /** Token used for padding completed canvases. */
  padTokenId: number;
  /** End-of-sequence token set used to terminate generated canvases. */
  eosTokenIds: number[];
  /** Beginning-of-image token id for multimodal inputs, null when unsupported by the artifact. */
  boiTokenId: number | null;
  /** End-of-image token id for multimodal inputs, null when unsupported by the artifact. */
  eoiTokenId: number | null;
  /** Image placeholder token id for multimodal inputs, null when unsupported by the artifact. */
  imageTokenId: number | null;
  /** Whether decoder self-conditioning logits are part of the runtime contract. */
  selfConditioning: boolean;
  /** Decoder attention must read encoder KV plus canvas K/V without committing the canvas to encoder cache. */
  decoderCacheMode: 'encoder-kv-readonly-canvas-concat';
  /** Router contract for DiffusionGemma/Gemma 4 MoE layers. */
  router: {
    scaleHiddenStates: boolean;
    normalizeTopK: boolean;
    perExpertScale: boolean;
  };
}

export interface ManifestRerankSchema {
  format: 'qwen3-yes-no-logit' | string;
  instruction: string;
  inputTemplate: string;
  prefix: string;
  suffix: string;
  trueToken: string;
  trueTokenId: number;
  falseToken: string;
  falseTokenId: number;
  score: 'logit-difference' | 'true-logit' | string;
  probability: 'sigmoid' | string;
}

/**
 * Complete inference configuration embedded in manifest.
 * All fields are required - converter must populate everything.
 * Use `null` values to indicate "not applicable" or "disabled".
 */
export interface ManifestInferenceSchema {
  /** Optional fail-fast marker for manifests with known unverified runtime contracts. */
  unsupported?: ManifestUnsupportedSchema | null;
  /** Execution contract discriminator (null = legacy inference contract). */
  schema: string | null;
  /** Attention configuration */
  attention: ManifestAttentionSchema;
  /** Normalization configuration */
  normalization: ManifestNormalizationSchema;
  /** FFN configuration */
  ffn: ManifestFFNSchema;
  /** RoPE configuration */
  rope: ManifestRoPESchema;
  /** Output configuration */
  output: ManifestOutputSchema;
  /** Layer pattern for hybrid attention */
  layerPattern: ManifestLayerPatternSchema;
  /** Chat template configuration */
  chatTemplate: ManifestChatTemplateSchema;
  /** Whether this artifact exposes embedding workload support through pipeline.embed(). */
  supportsEmbedding: boolean;
  /** Whether this artifact exposes typed biological sequence outputs. */
  supportsSequence: boolean;
  /** Whether this artifact exposes rerank workload support through prefillWithLogits(). */
  supportsRerank: boolean;
  /** DiffusionGemma block-diffusion runtime contract, null for non-DiffusionGemma models. */
  diffusionGemma: ManifestDiffusionGemmaSchema | null;
  /** Source-owned multi-stream gated residual (qwen4_exp); null for single-stream residual decoders. */
  hyperConnections: HyperConnectionsSchema | null;
  /** Source-owned QSA sparse-attention indexer (qwen4_exp); null for dense full attention. */
  indexedAttention: IndexedAttentionSchema | null;
  /** Source-owned hashed n-gram per-layer embedding (qwen4_exp PLE); null when absent. */
  ngramEmbedding: NgramEmbeddingSchema | null;
  /** Manifest-owned rerank scoring contract, null when rerank is unsupported. */
  rerank: ManifestRerankSchema | null;
  /** Biological sequence input/output contract, null when unsupported. */
  sequence: ManifestSequenceSchema | null;
  /** Layer pipeline override (null = use optimized hardcoded path) */
  pipeline: LayerPipelineSchema | null;
  /** Explicit session policy for execution v1 manifests */
  session: ExecutionV1SessionSchema | null;
  /** Explicit execution graph (v1 compact tuple format) */
  execution: ExecutionV1GraphSchema | null;
}
