import type { ProgramBundle } from '../config/schema/program-bundle.schema.js';

import type { NodeWebGPUProviderOptions } from './provider-v1-contract.js';

export declare const PROGRAM_BUNDLE_PARITY_SCHEMA_ID: 'doppler.program-bundle-parity/v2';

export interface ProgramBundleParityOptions {
  bundle?: ProgramBundle;
  bundlePath?: string;
  repoRoot?: string;
  providers: Array<'browser-webgpu' | 'node:webgpu' | 'node:doe-gpu'>;
  mode: 'contract' | 'execute';
  nodeOptions?: Record<string, unknown>;
  nodeWebGPUContractModule?: string;
  nodeWebGPUProviderOptions?: NodeWebGPUProviderOptions;
  doeProviderOptions?: NodeWebGPUProviderOptions;
  doeExecution?: Record<string, unknown>;
}

export interface ProgramBundleParityResult {
  schema: typeof PROGRAM_BUNDLE_PARITY_SCHEMA_ID;
  authority: 'portability-diagnostic-only';
  modelPromotionAuthority: false;
  ok: boolean;
  mode: 'contract' | 'execute';
  schemaValid: true;
  bundleId: string;
  modelId: string;
  executionGraphHash: string;
  reference: Record<string, unknown>;
  tokenEvidence: Record<string, unknown>;
  providers: Array<Record<string, unknown>>;
  parityHash: string;
}

export declare function checkProgramBundleParity(
  options: ProgramBundleParityOptions
): Promise<ProgramBundleParityResult>;
