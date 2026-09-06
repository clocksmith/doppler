import type { CapsuleOperationRequest } from '../../config/capsule-operation.js';
import type { CapsuleExecutionAdapter } from '../../config/capsule-adapters.js';
import type { TargetPlan } from '../../config/target-plan.js';
export interface CapsuleAdapterArtifactStore { readArtifact(artifact: CapsuleExecutionAdapter['artifact']): Promise<Uint8Array> }
export interface PreparedCapsuleAdapterExecution {
  receiptFields: Record<string, unknown>;
  check(): Promise<void>;
  close(): Promise<void>;
}
export interface CapsuleAdapterProgram {
  getActiveAdapterIdentity?(): Readonly<Record<string, unknown>> | null;
  loadAdapter?(manifest: CapsuleExecutionAdapter['manifest'], control: { bytes: Uint8Array; signal: AbortSignal }): Promise<void>;
  unloadAdapter?(): Promise<void>;
  reset?(): void | Promise<void>;
}
export function createCapsuleAdapterExecution(context: { program: CapsuleAdapterProgram; capsule: CapsuleExecutionAdapter['baseModel']; targetPlan: TargetPlan }):
  (request: CapsuleOperationRequest, control: { adapterArtifactStore?: CapsuleAdapterArtifactStore | null; signal: AbortSignal }) => Promise<PreparedCapsuleAdapterExecution | null>;
