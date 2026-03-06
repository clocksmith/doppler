import type { ExecutionContractFacts } from '../config/execution-contract-check.js';

export interface RenderLeanExecutionContractOptions {
  moduleName?: string;
}

export declare function sanitizeLeanModuleName(value: unknown): string;

export declare function extractExecutionContractFacts(
  manifest: Record<string, unknown>
): ExecutionContractFacts;

export declare function renderExecutionContractLeanModule(
  facts: LeanExecutionContractFacts,
  options?: RenderLeanExecutionContractOptions
): string;
