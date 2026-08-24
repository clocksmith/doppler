import {
  hashProductionReleaseEvidence,
  validateReleaseDecision,
} from '../../config/production-release-evidence.js';

export const ELECTRON_RELEASE_STATE_SCHEMA = 'doppler.electron-release-state/v1';
export const ELECTRON_REVOCATION_SNAPSHOT_SCHEMA = 'doppler.electron-revocation-snapshot/v1';

const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/u;

function object(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function exact(value, fields, label) {
  object(value, label);
  const allowed = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!allowed.has(field)) throw new Error(`${label}.${field} is not supported.`);
  }
  for (const field of fields) {
    if (!Object.hasOwn(value, field)) throw new Error(`${label}.${field} is required.`);
  }
}

function exactWithOptional(value, required, optional, label) {
  object(value, label);
  const allowed = new Set([...required, ...optional]);
  for (const field of Object.keys(value)) {
    if (!allowed.has(field)) throw new Error(`${label}.${field} is not supported.`);
  }
  for (const field of required) {
    if (!Object.hasOwn(value, field)) throw new Error(`${label}.${field} is required.`);
  }
}

function text(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be non-empty.`);
  return value;
}

function digest(value, label) {
  if (!DIGEST_PATTERN.test(value || '')) throw new Error(`${label} must be a SHA-256 digest.`);
  return value;
}

function instant(value, label) {
  text(value, label);
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp) || new Date(timestamp).toISOString() !== value) {
    throw new Error(`${label} must be an ISO instant.`);
  }
  return value;
}

function packRef(value, label) {
  exact(value, ['packId', 'semanticRoot', 'path'], label);
  return {
    packId: text(value.packId, `${label}.packId`),
    semanticRoot: digest(value.semanticRoot, `${label}.semanticRoot`),
    path: text(value.path, `${label}.path`),
  };
}

function revocationPolicy(value, label) {
  exact(
    value,
    ['authorityId', 'policyDigest', 'offlineExpirySeconds', 'failClosedAfterExpiry'],
    label
  );
  if (!Number.isSafeInteger(value.offlineExpirySeconds) || value.offlineExpirySeconds < 1) {
    throw new Error(`${label}.offlineExpirySeconds must be a positive integer.`);
  }
  if (value.failClosedAfterExpiry !== true) {
    throw new Error(`${label}.failClosedAfterExpiry must be true.`);
  }
  return {
    authorityId: text(value.authorityId, `${label}.authorityId`),
    policyDigest: digest(value.policyDigest, `${label}.policyDigest`),
    offlineExpirySeconds: value.offlineExpirySeconds,
    failClosedAfterExpiry: true,
  };
}

function releaseSlot(value, label) {
  if (value === null) return null;
  exactWithOptional(
    value,
    ['pack', 'decisionDigest', 'changedAtUtc', 'customerAuthorizationDigest'],
    ['revocationPolicy'],
    label
  );
  return {
    pack: packRef(value.pack, `${label}.pack`),
    decisionDigest: digest(value.decisionDigest, `${label}.decisionDigest`),
    changedAtUtc: instant(value.changedAtUtc, `${label}.changedAtUtc`),
    customerAuthorizationDigest: value.customerAuthorizationDigest === null
      ? null
      : digest(value.customerAuthorizationDigest, `${label}.customerAuthorizationDigest`),
    revocationPolicy: value.revocationPolicy === undefined
      ? null
      : revocationPolicy(value.revocationPolicy, `${label}.revocationPolicy`),
  };
}

function candidateSlot(value) {
  if (value === null) return null;
  exact(value, ['pack', 'decisionDigest', 'installedAtUtc'], 'electron release state.candidate');
  return {
    pack: packRef(value.pack, 'electron release state.candidate.pack'),
    decisionDigest: digest(value.decisionDigest, 'electron release state.candidate.decisionDigest'),
    installedAtUtc: instant(value.installedAtUtc, 'electron release state.candidate.installedAtUtc'),
  };
}

function revocationSnapshot(value) {
  if (value === null) return null;
  exact(value, [
    'schema', 'authorityId', 'policyDigest', 'sequence', 'issuedAtUtc',
    'expiresAtUtc', 'revokedSemanticRoots', 'digest', 'signature',
  ], 'electron release state.revocation');
  if (value.schema !== ELECTRON_REVOCATION_SNAPSHOT_SCHEMA) {
    throw new Error(`electron release state.revocation.schema must be ${ELECTRON_REVOCATION_SNAPSHOT_SCHEMA}.`);
  }
  if (!Number.isSafeInteger(value.sequence) || value.sequence < 1) {
    throw new Error('electron release state.revocation.sequence must be a positive integer.');
  }
  const issuedAtUtc = instant(value.issuedAtUtc, 'electron release state.revocation.issuedAtUtc');
  const expiresAtUtc = instant(value.expiresAtUtc, 'electron release state.revocation.expiresAtUtc');
  if (new Date(expiresAtUtc).getTime() <= new Date(issuedAtUtc).getTime()) {
    throw new Error('electron release state.revocation expiry must follow issuance.');
  }
  if (!Array.isArray(value.revokedSemanticRoots)) {
    throw new Error('electron release state.revocation.revokedSemanticRoots must be an array.');
  }
  const revokedSemanticRoots = value.revokedSemanticRoots.map((entry, index) => (
    digest(entry, `electron release state.revocation.revokedSemanticRoots[${index}]`)
  ));
  if (new Set(revokedSemanticRoots).size !== revokedSemanticRoots.length) {
    throw new Error('electron release state.revocation.revokedSemanticRoots must be unique.');
  }
  exact(
    value.signature,
    ['authority', 'algorithm', 'publicKeyDigest', 'signedDigest', 'signatureHex'],
    'electron release state.revocation.signature'
  );
  if (value.signature.algorithm !== 'Ed25519') {
    throw new Error('electron release state.revocation.signature.algorithm must be Ed25519.');
  }
  const signatureAuthority = text(
    value.signature.authority,
    'electron release state.revocation.signature.authority'
  );
  if (signatureAuthority !== value.authorityId) {
    throw new Error('electron release state.revocation signature authority must match authorityId.');
  }
  if (!/^[0-9a-f]{128}$/u.test(value.signature.signatureHex || '')) {
    throw new Error('electron release state.revocation.signature.signatureHex must be a 64-byte hexadecimal signature.');
  }
  const normalizedDigest = digest(value.digest, 'electron release state.revocation.digest');
  if (value.signature.signedDigest !== normalizedDigest) {
    throw new Error('electron release state.revocation.signature.signedDigest must equal digest.');
  }
  digest(value.signature.publicKeyDigest, 'electron release state.revocation.signature.publicKeyDigest');
  const expectedDigest = hashProductionReleaseEvidence(value);
  if (normalizedDigest !== expectedDigest) {
    throw new Error('electron release state.revocation.digest does not match its semantic payload.');
  }
  return {
    schema: ELECTRON_REVOCATION_SNAPSHOT_SCHEMA,
    authorityId: text(value.authorityId, 'electron release state.revocation.authorityId'),
    policyDigest: digest(value.policyDigest, 'electron release state.revocation.policyDigest'),
    sequence: value.sequence,
    issuedAtUtc,
    expiresAtUtc,
    revokedSemanticRoots,
    digest: normalizedDigest,
    signature: structuredClone(value.signature),
  };
}

function failure(value, index) {
  const label = `electron release state.failures[${index}]`;
  exact(value, ['candidateSemanticRoot', 'failureBundleDigest', 'rejectedAtUtc'], label);
  return {
    candidateSemanticRoot: digest(value.candidateSemanticRoot, `${label}.candidateSemanticRoot`),
    failureBundleDigest: digest(value.failureBundleDigest, `${label}.failureBundleDigest`),
    rejectedAtUtc: instant(value.rejectedAtUtc, `${label}.rejectedAtUtc`),
  };
}

export function validateElectronReleaseState(value) {
  exact(
    value,
    ['schema', 'sequence', 'current', 'previous', 'candidate', 'failures', 'revocation'],
    'electron release state'
  );
  if (value.schema !== ELECTRON_RELEASE_STATE_SCHEMA) {
    throw new Error(`electron release state.schema must be ${ELECTRON_RELEASE_STATE_SCHEMA}.`);
  }
  if (!Number.isSafeInteger(value.sequence) || value.sequence < 0) {
    throw new Error('electron release state.sequence must be a non-negative integer.');
  }
  if (!Array.isArray(value.failures)) throw new Error('electron release state.failures must be an array.');
  return {
    schema: ELECTRON_RELEASE_STATE_SCHEMA,
    sequence: value.sequence,
    current: releaseSlot(value.current, 'electron release state.current'),
    previous: releaseSlot(value.previous, 'electron release state.previous'),
    candidate: candidateSlot(value.candidate),
    failures: value.failures.map(failure),
    revocation: revocationSnapshot(value.revocation),
  };
}

function initialState() {
  return {
    schema: ELECTRON_RELEASE_STATE_SCHEMA,
    sequence: 0,
    current: null,
    previous: null,
    candidate: null,
    failures: [],
    revocation: null,
  };
}

export function createElectronReleaseStateCoordinator(options) {
  const stateStore = object(options?.stateStore, 'electron release stateStore');
  if (typeof stateStore.load !== 'function' || typeof stateStore.compareAndSwap !== 'function') {
    throw new Error('electron release stateStore requires load() and compareAndSwap().');
  }
  if (typeof options.verifyReleaseDecision !== 'function') {
    throw new Error('electron release coordinator requires verifyReleaseDecision().');
  }
  if (typeof options.verifyRevocationSnapshot !== 'function') {
    throw new Error('electron release coordinator requires verifyRevocationSnapshot().');
  }
  const now = options.now ?? (() => new Date().toISOString());
  if (typeof now !== 'function') throw new Error('electron release coordinator now must be a function.');

  async function load() {
    const stored = await stateStore.load();
    return stored === null ? initialState() : validateElectronReleaseState(stored);
  }

  async function commit(current, next) {
    const value = validateElectronReleaseState({ ...next, sequence: current.sequence + 1 });
    if (await stateStore.compareAndSwap(current.sequence, value) !== true) {
      throw new Error('Electron release state changed concurrently; reload before retrying.');
    }
    return value;
  }

  async function installCandidate(pack, decisionDigest) {
    const current = await load();
    return commit(current, {
      ...current,
      candidate: {
        pack: packRef(pack, 'candidate Pack'),
        decisionDigest: digest(decisionDigest, 'candidate decisionDigest'),
        installedAtUtc: instant(now(), 'candidate installedAtUtc'),
      },
    });
  }

  async function activateCandidate(decision, customerAuthorizationDigest) {
    const validation = validateReleaseDecision(decision);
    if (!validation.ok) throw new Error(`Electron activation rejected invalid decision: ${validation.errors.join('; ')}`);
    if (await options.verifyReleaseDecision(decision) !== true) {
      throw new Error('Electron activation requires a verified release-decision signature.');
    }
    if (decision.eligibility !== 'eligible' || decision.selfPromotionAllowed !== false
      || decision.activationAuthority !== 'customer') {
      throw new Error('Electron activation requires an eligible customer-authorized decision.');
    }
    const current = await load();
    if (!current.candidate || current.candidate.pack.semanticRoot !== decision.pack.semanticRoot
      || current.candidate.pack.packId !== decision.pack.packId
      || current.candidate.decisionDigest !== decision.digest) {
      throw new Error('Electron activation decision does not bind the installed candidate.');
    }
    const changedAtUtc = instant(now(), 'activation changedAtUtc');
    return commit(current, {
      ...current,
      previous: current.current,
      current: {
        pack: current.candidate.pack,
        decisionDigest: decision.digest,
        changedAtUtc,
        customerAuthorizationDigest: digest(
          customerAuthorizationDigest,
          'customerAuthorizationDigest'
        ),
        revocationPolicy: revocationPolicy(
          decision.revocation,
          'release decision.revocation'
        ),
      },
      candidate: null,
    });
  }

  async function rejectCandidate(failureBundleDigest) {
    const current = await load();
    if (!current.candidate) throw new Error('Electron release state has no candidate to reject.');
    return commit(current, {
      ...current,
      candidate: null,
      failures: [
        ...current.failures,
        {
          candidateSemanticRoot: current.candidate.pack.semanticRoot,
          failureBundleDigest: digest(failureBundleDigest, 'failureBundleDigest'),
          rejectedAtUtc: instant(now(), 'candidate rejectedAtUtc'),
        },
      ],
    });
  }

  async function rollback(customerAuthorizationDigest) {
    const current = await load();
    if (!current.current || !current.previous) {
      throw new Error('Electron rollback requires current and previous releases.');
    }
    const changedAtUtc = instant(now(), 'rollback changedAtUtc');
    const restored = {
      ...current.previous,
      changedAtUtc,
      customerAuthorizationDigest: digest(
        customerAuthorizationDigest,
        'customerAuthorizationDigest'
      ),
    };
    return commit(current, {
      ...current,
      current: restored,
      previous: current.current,
      candidate: null,
    });
  }

  async function applyRevocationSnapshot(snapshot) {
    const normalized = revocationSnapshot(snapshot);
    if (await options.verifyRevocationSnapshot(normalized) !== true) {
      throw new Error('Electron revocation update requires a verified snapshot signature.');
    }
    const current = await load();
    if (current.revocation && normalized.sequence === current.revocation.sequence
      && normalized.digest === current.revocation.digest) {
      return current;
    }
    if (current.revocation && normalized.sequence <= current.revocation.sequence) {
      throw new Error('Electron revocation snapshot must advance monotonically.');
    }
    const activePolicy = current.current?.revocationPolicy;
    if (activePolicy) {
      if (normalized.authorityId !== activePolicy.authorityId
        || normalized.policyDigest !== activePolicy.policyDigest) {
        throw new Error('Electron revocation snapshot does not bind the active release policy.');
      }
      const lifetimeMs = new Date(normalized.expiresAtUtc).getTime()
        - new Date(normalized.issuedAtUtc).getTime();
      if (lifetimeMs > activePolicy.offlineExpirySeconds * 1000) {
        throw new Error('Electron revocation snapshot exceeds the active release offline-expiry policy.');
      }
    }
    if (new Date(normalized.expiresAtUtc).getTime() <= new Date(now()).getTime()) {
      throw new Error('Electron revocation snapshot is already expired.');
    }
    return commit(current, { ...current, revocation: normalized });
  }

  async function resolveCurrent() {
    const current = await load();
    if (!current.current) throw new Error('Electron release state has no active Pack.');
    if (!current.revocation) throw new Error('Electron release state has no verified revocation snapshot.');
    if (!current.current.revocationPolicy) {
      throw new Error('Electron release state current Pack lacks a bound revocation policy.');
    }
    if (current.revocation.authorityId !== current.current.revocationPolicy.authorityId
      || current.revocation.policyDigest !== current.current.revocationPolicy.policyDigest) {
      throw new Error('Electron release revocation state does not bind the current release policy.');
    }
    const lifetimeMs = new Date(current.revocation.expiresAtUtc).getTime()
      - new Date(current.revocation.issuedAtUtc).getTime();
    if (lifetimeMs > current.current.revocationPolicy.offlineExpirySeconds * 1000) {
      throw new Error('Electron release revocation state exceeds the current offline-expiry policy.');
    }
    if (new Date(current.revocation.expiresAtUtc).getTime() <= new Date(now()).getTime()) {
      throw new Error('Electron release revocation state is expired; execution fails closed.');
    }
    if (current.revocation.revokedSemanticRoots.includes(current.current.pack.semanticRoot)) {
      throw new Error('Electron release current Pack is revoked.');
    }
    return structuredClone(current.current.pack);
  }

  return Object.freeze({
    load,
    installCandidate,
    activateCandidate,
    rejectCandidate,
    rollback,
    applyRevocationSnapshot,
    resolveCurrent,
  });
}
