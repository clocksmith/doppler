import type { ProgramBundle } from '../config/schema/program-bundle.schema.js';

export declare const PROGRAM_BUNDLE_PARITY_SCHEMA_ID: 'doppler.program-bundle-parity/v1';

export interface ProgramBundleParityOptions {
  bundle?: ProgramBundle;
  bundlePath?: string;
  repoRoot?: string;
  providers?: string[];
  mode?: 'contract' | 'execute';
  nodeOptions?: Record<string, unknown>;
}

export interface ProgramBundleParityResult {
  schema: typeof PROGRAM_BUNDLE_PARITY_SCHEMA_ID;
  ok: boolean;
  mode: 'contract' | 'execute';
  bundleId: string;
  modelId: string;
  executionGraphHash: string;
  reference: Record<string, unknown>;
  providers: Array<Record<string, unknown>>;
  parityHash: string;
}

export declare function checkProgramBundleParity(
  options?: ProgramBundleParityOptions
): Promise<ProgramBundleParityResult>;
