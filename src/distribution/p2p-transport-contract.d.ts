export declare const P2P_TRANSPORT_CONTRACT_VERSION: 1;
export declare const P2P_TRANSPORT_RESULT_SCHEMA_VERSION: 1;

export declare const P2P_TRANSPORT_ERROR_CODES: Readonly<{
  unconfigured: string;
  unavailable: string;
  timeout: string;
  aborted: string;
  integrityMismatch: string;
  policyDenied: string;
  internal: string;
  payloadInvalid: string;
  contractUnsupported: string;
}>;

export interface P2PTransportError extends Error {
  code?: string;
  retryable?: boolean;
  details?: Record<string, unknown>;
}

export declare function createP2PTransportError(
  code: string,
  message: string,
  details?: Record<string, unknown> | null,
  retryable?: boolean
): P2PTransportError;

export declare function assertSupportedP2PTransportContract(
  version: number | null | undefined
): number;

export interface P2PTransportResultEnvelope {
  schemaVersion: number;
  data: ArrayBuffer;
  manifestVersionSet: string | null;
  manifestHash: string | null;
}

export declare function normalizeP2PTransportResult(
  value: unknown,
  label?: string
): P2PTransportResultEnvelope | null;

export declare function normalizeP2PTransportPayload(
  value: unknown,
  label?: string
): ArrayBuffer | null;

export declare function normalizeP2PTransportError(
  error: unknown,
  context?: Record<string, unknown>
): P2PTransportError;

export declare function isP2PTransportRetryable(error: unknown): boolean;
