export declare const P2P_CONTROL_PLANE_CONTRACT_VERSION: 1;

export interface P2PControlPlaneSessionUpdate {
  hasSessionToken: boolean;
  hasTokenExpiresAtMs: boolean;
  sessionToken: string | null;
  tokenExpiresAtMs: number | null;
  metadata: Record<string, unknown> | null;
}

export interface P2PPolicyDecision {
  allow: boolean;
  reason: string | null;
  sessionUpdate: P2PControlPlaneSessionUpdate | null;
  metadata: Record<string, unknown> | null;
}

export interface P2PControlPlaneConfig {
  enabled: boolean;
  contractVersion: number;
  tokenRefreshSkewMs: number;
  tokenProvider: ((context: Record<string, unknown>) => Promise<unknown> | unknown) | null;
  policyEvaluator: ((context: Record<string, unknown>) => Promise<unknown> | unknown) | null;
}

export declare function assertSupportedP2PControlPlaneContract(
  version: number | null | undefined
): number;

export declare function normalizeControlPlaneSessionUpdate(
  value: unknown,
  label?: string
): P2PControlPlaneSessionUpdate | null;

export declare function normalizeP2PPolicyDecision(
  value: unknown,
  label?: string
): P2PPolicyDecision;

export declare function normalizeP2PControlPlaneConfig(
  config?: Record<string, unknown>
): P2PControlPlaneConfig;

export declare function resolveP2PSessionToken(
  controlPlaneConfig: P2PControlPlaneConfig,
  context?: Record<string, unknown>
): Promise<P2PControlPlaneSessionUpdate | null>;

export declare function evaluateP2PPolicyDecision(
  controlPlaneConfig: P2PControlPlaneConfig,
  context?: Record<string, unknown>
): Promise<P2PPolicyDecision>;
