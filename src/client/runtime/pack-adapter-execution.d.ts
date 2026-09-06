import type { PackOperationRequest } from '../../config/pack-operation.js';
import type { PackExecutionAdapter } from '../../config/pack-adapters.js';
import type { TargetPlan } from '../../config/target-plan.js';
export interface PackAdapterArtifactStore { readArtifact(artifact: PackExecutionAdapter['artifact']): Promise<Uint8Array> }
export interface PreparedPackAdapterExecution {
  receiptFields: Record<string, unknown>;
  check(): Promise<void>;
  close(): Promise<void>;
}
export interface PackAdapterProgram {
  getActiveAdapterIdentity?(): Readonly<Record<string, unknown>> | null;
  loadAdapter?(manifest: PackExecutionAdapter['manifest'], control: { bytes: Uint8Array; signal: AbortSignal }): Promise<void>;
  unloadAdapter?(): Promise<void>;
  reset?(): void | Promise<void>;
}
export function createPackAdapterExecution(context: { program: PackAdapterProgram; pack: PackExecutionAdapter['baseModel']; targetPlan: TargetPlan }):
  (request: PackOperationRequest, control: { adapterArtifactStore?: PackAdapterArtifactStore | null; signal: AbortSignal }) => Promise<PreparedPackAdapterExecution | null>;
