export interface ExecutionBackendDescriptor {
  schema: 'doppler.execution-backend/v1'; id: string; version: 1;
  kind: 'plaintext' | 'attested' | 'cryptographic'; modelIdentities: readonly string[]; operations: readonly ['generate'];
  protocol: string | null; inputRepresentation: 'messages' | 'opaque'; outputRepresentation: 'text' | 'opaque';
}
/** Future implementations must supply all operations; no cryptographic or attested implementation is qualified. */
export interface ExecutionBackendExtension<ClientState, SealedInput, Result> {
  describe(): ExecutionBackendDescriptor;
  prepareClient(binding: Readonly<ExecutionStateIdentity>, signal: AbortSignal): Promise<ClientState>;
  verifyEvidence(state: ClientState, evidence: unknown, signal: AbortSignal): Promise<void>;
  sealInput(state: ClientState, messages: readonly {role: string; content: string}[], signal: AbortSignal): Promise<SealedInput>;
  execute(input: SealedInput, binding: Readonly<ExecutionStateIdentity>, signal: AbortSignal): Promise<Result>;
  finalizeClient(state: ClientState, result: Result, signal: AbortSignal): Promise<string>;
  dispose(state: ClientState): Promise<void>;
}
export interface ExecutionStateIdentity {
  schema: 'doppler.execution-state/v1'; modelIdentity: string; adapterIdentities: readonly string[]; backendId: string; policyId: string; routeId: string;
  tenantId: string; sessionId: string; keyContext: string | null;
}
export function validateExecutionBackend(input: ExecutionBackendDescriptor): Readonly<ExecutionBackendDescriptor>;
export function assertExecutionBackendAvailable(input: ExecutionBackendDescriptor, modelIdentity: string): Readonly<ExecutionBackendDescriptor>;
export function executionStateNamespace(input: ExecutionStateIdentity): string;
