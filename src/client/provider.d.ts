/**
 * Doppler Provider — governed local-first inference with policy-routed fallback.
 *
 * @module client/provider
 */

/** Canonical four routing modes (see ouroboros/docs/strategy/capabilities/30-hybrid-routing-sdk.md). */
export type PolicyMode = 'local-only' | 'prefer-local' | 'prefer-cloud' | 'cloud-only';

export type InferenceSource = 'local' | 'fallback';

/**
 * Fine-grained failure class vocabulary. Single taxonomy shared between
 * the failure classifier, receipt payloads, and the `fallbackOn` policy list.
 * See ouroboros/docs/strategy/diagnose-router.md for the canonical enum.
 */
export type FailureClass =
  | 'gpu_device_lost'
  | 'gpu_oom'
  | 'gpu_timeout'
  | 'gpu_unsupported'
  | 'gpu_unavailable'
  | 'model_load_failed'
  | 'policy_denied'
  | 'runtime_internal'
  | 'fallback_failed'
  | 'unknown';

export interface ModelHandle {
  loaded: boolean;
  modelId: string;
  manifest: object | null;
  deviceInfo: object | null;
  generateText(prompt: unknown, opts?: object): Promise<string>;
  unload(): Promise<void>;
}

export interface ProviderLocalConfig {
  model?: string;
  handle?: ModelHandle;
  runtimeConfig?: object | null;
  onProgress?: ((progress: { phase: string; percent: number; message: string }) => void) | null;
}

export interface ProviderFallbackConfig {
  provider: string;
  model: string;
  apiKey?: string | null;
  baseUrl?: string | null;
}

export interface ProviderPolicyConfig {
  mode?: PolicyMode;
  id?: string | null;
  fallbackOn?: FailureClass[] | null;
}

export interface FaultInjectionConfig {
  enabled: boolean;
  failureCode?: string;
  stage?: string;
  probability?: number;
}

export interface ProviderDiagnosticsConfig {
  receipts?: boolean;
  faultInjection?: FaultInjectionConfig;
}

export interface ProviderConfig {
  local?: ProviderLocalConfig;
  fallback?: ProviderFallbackConfig;
  policy?: ProviderPolicyConfig;
  diagnostics?: ProviderDiagnosticsConfig;
}

export interface ReceiptModel {
  id: string;
  hash: string | null;
  fallbackId: string | null;
}

export interface ReceiptDevice {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  hasF16: boolean;
  hasSubgroups: boolean;
  maxBufferSize: number;
  submitProbeMs: number | null;
  deviceEpoch: number;
}

export interface ReceiptFailure {
  failureClass: FailureClass;
  failureCode: string;
  stage: string;
  surface: string;
  device: string | null;
  modelId: string | null;
  runtimeProfile: string | null;
  kernelPathId: string | null;
  isSimulated: boolean;
  message: string;
}

export interface ReceiptFallbackDecision {
  reason: string;
  eligible: boolean;
  executed: boolean;
  deniedReason: string | null;
}

export interface ProviderReceiptV1 {
  receiptVersion: 'doppler_provider_receipt_v1';
  receiptId: string;
  source: InferenceSource;
  policyMode: string;
  policyId: string | null;
  model: ReceiptModel;
  device: ReceiptDevice | null;
  failure: ReceiptFailure | null;
  fallbackDecision: ReceiptFallbackDecision | null;
  localDurationMs: number | null;
  fallbackDurationMs: number | null;
  totalDurationMs: number;
  timestamp: string;
  diagnoseArtifactRef: string | null;
}

export interface ProviderResult {
  text: string;
  inferenceSource: InferenceSource;
  receipt: ProviderReceiptV1 | null;
}

export interface ProviderGenerateOptions {
  prompt?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface DopplerProvider {
  generate(promptOrOptions: string | ProviderGenerateOptions): Promise<ProviderResult>;
  unload(): Promise<void>;
}

export function createDopplerProvider(config: ProviderConfig): DopplerProvider;

export { wrapPipelineAsHandle } from './wrap-pipeline-handle.js';
