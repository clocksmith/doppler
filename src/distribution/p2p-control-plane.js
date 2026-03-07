import {
  P2P_TRANSPORT_ERROR_CODES,
  createP2PTransportError,
  normalizeP2PTransportError,
} from './p2p-transport-contract.js';

export const P2P_CONTROL_PLANE_CONTRACT_VERSION = 1;

const DEFAULT_TOKEN_REFRESH_SKEW_MS = 5000;

function asOptionalString(value, label) {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a string when provided.`,
      { label }
    );
  }
  const normalized = value.trim();
  return normalized || null;
}

function asOptionalTimestamp(value, label) {
  if (value === undefined || value === null) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a positive timestamp when provided.`,
      { label }
    );
  }
  return Math.floor(parsed);
}

function asOptionalNonNegativeInteger(value, label) {
  if (value === undefined || value === null) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a non-negative integer when provided.`,
      { label }
    );
  }
  return parsed;
}

export function assertSupportedP2PControlPlaneContract(version) {
  const parsed = Number(version ?? P2P_CONTROL_PLANE_CONTRACT_VERSION);
  if (!Number.isInteger(parsed) || parsed !== P2P_CONTROL_PLANE_CONTRACT_VERSION) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.contractUnsupported,
      `Unsupported p2p.controlPlane contractVersion "${version}". Supported: ${P2P_CONTROL_PLANE_CONTRACT_VERSION}.`,
      { contractVersion: version }
    );
  }
  return parsed;
}

export function normalizeControlPlaneSessionUpdate(value, label = 'p2p control-plane session update') {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === 'string') {
    const sessionToken = asOptionalString(value, `${label}.sessionToken`);
    return {
      hasSessionToken: true,
      hasTokenExpiresAtMs: false,
      sessionToken,
      tokenExpiresAtMs: null,
      metadata: null,
    };
  }
  if (typeof value !== 'object') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a string or object when provided.`,
      { label }
    );
  }

  const sessionToken = asOptionalString(
    value.sessionToken ?? value.token,
    `${label}.sessionToken`
  );
  const tokenExpiresAtMs = asOptionalTimestamp(
    value.tokenExpiresAtMs,
    `${label}.tokenExpiresAtMs`
  );
  const metadata = value.metadata && typeof value.metadata === 'object'
    ? value.metadata
    : null;

  return {
    hasSessionToken: Object.prototype.hasOwnProperty.call(value, 'sessionToken')
      || Object.prototype.hasOwnProperty.call(value, 'token'),
    hasTokenExpiresAtMs: Object.prototype.hasOwnProperty.call(value, 'tokenExpiresAtMs'),
    sessionToken,
    tokenExpiresAtMs,
    metadata,
  };
}

export function normalizeP2PPolicyDecision(value, label = 'p2p control-plane policy decision') {
  if (value === undefined || value === null) {
    return {
      allow: true,
      reason: null,
      sessionUpdate: null,
      metadata: null,
    };
  }

  if (typeof value === 'boolean') {
    return {
      allow: value,
      reason: value ? null : 'policy_denied',
      sessionUpdate: null,
      metadata: null,
    };
  }

  if (typeof value !== 'object') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a boolean or object when provided.`,
      { label }
    );
  }

  const allow = value.allow === false || value.deny === true
    ? false
    : true;
  const reason = asOptionalString(value.reason, `${label}.reason`);
  const sessionUpdate = normalizeControlPlaneSessionUpdate(
    {
      sessionToken: value.sessionToken,
      token: value.token,
      tokenExpiresAtMs: value.tokenExpiresAtMs,
      metadata: value.sessionMetadata,
    },
    `${label}.sessionUpdate`
  );
  const metadata = value.metadata && typeof value.metadata === 'object'
    ? value.metadata
    : null;

  return {
    allow,
    reason: reason ?? (allow ? null : 'policy_denied'),
    sessionUpdate,
    metadata,
  };
}

export function normalizeP2PControlPlaneConfig(config = {}) {
  const raw = config && typeof config === 'object'
    ? config
    : {};

  const tokenProvider = typeof raw.tokenProvider === 'function'
    ? raw.tokenProvider
    : null;
  const policyEvaluator = typeof raw.policyEvaluator === 'function'
    ? raw.policyEvaluator
    : null;
  const hasWiring = tokenProvider || policyEvaluator;
  const enabled = raw.enabled === true || !!hasWiring;

  if (raw.enabled === true && !hasWiring) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.unconfigured,
      'p2p.controlPlane.enabled=true requires tokenProvider and/or policyEvaluator callbacks.'
    );
  }

  return {
    enabled,
    contractVersion: assertSupportedP2PControlPlaneContract(
      raw.contractVersion ?? P2P_CONTROL_PLANE_CONTRACT_VERSION
    ),
    tokenRefreshSkewMs: asOptionalNonNegativeInteger(
      raw.tokenRefreshSkewMs,
      'p2p.controlPlane.tokenRefreshSkewMs'
    ) ?? DEFAULT_TOKEN_REFRESH_SKEW_MS,
    tokenProvider,
    policyEvaluator,
  };
}

export async function resolveP2PSessionToken(controlPlaneConfig, context = {}) {
  if (!controlPlaneConfig?.enabled || typeof controlPlaneConfig.tokenProvider !== 'function') {
    return null;
  }

  try {
    const response = await controlPlaneConfig.tokenProvider({
      contractVersion: controlPlaneConfig.contractVersion,
      ...context,
    });
    return normalizeControlPlaneSessionUpdate(response, 'p2p control-plane token provider response');
  } catch (error) {
    throw normalizeP2PTransportError(error, {
      stage: 'control_plane_token_provider',
      shardIndex: context?.shardIndex ?? null,
      attempt: context?.attempt ?? null,
    });
  }
}

export async function evaluateP2PPolicyDecision(controlPlaneConfig, context = {}) {
  if (!controlPlaneConfig?.enabled || typeof controlPlaneConfig.policyEvaluator !== 'function') {
    return {
      allow: true,
      reason: null,
      sessionUpdate: null,
      metadata: null,
    };
  }

  try {
    const response = await controlPlaneConfig.policyEvaluator({
      contractVersion: controlPlaneConfig.contractVersion,
      ...context,
    });
    return normalizeP2PPolicyDecision(response, 'p2p control-plane policy evaluator response');
  } catch (error) {
    throw normalizeP2PTransportError(error, {
      stage: 'control_plane_policy_evaluator',
      shardIndex: context?.shardIndex ?? null,
      attempt: context?.attempt ?? null,
    });
  }
}
