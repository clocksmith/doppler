import { validateReleaseDecision } from '../../config/production-release-evidence.js';

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

function releaseSlot(value, label) {
  if (value === null) return null;
  exact(
    value,
    ['pack', 'decisionDigest', 'changedAtUtc', 'customerAuthorizationDigest'],
    label
  );
  return {
    pack: packRef(value.pack, `${label}.pack`),
    decisionDigest: digest(value.decisionDigest, `${label}.decisionDigest`),
    changedAtUtc: instant(value.changedAtUtc, `${label}.changedAtUtc`),
    customerAuthorizationDigest: value.customerAuthorizationDigest === null
      ? null
      : digest(value.customerAuthorizationDigest, `${label}.customerAuthorizationDigest`),
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
    'schema', 'authorityId', 'sequence', 'expiresAtUtc', 'revokedSemanticRoots',
    'digest', 'signatureVerified',
  ], 'electron release state.revocation');
  if (value.schema !== ELECTRON_REVOCATION_SNAPSHOT_SCHEMA) {
    throw new Error(`electron release state.revocation.schema must be ${ELECTRON_REVOCATION_SNAPSHOT_SCHEMA}.`);
  }
  if (!Number.isSafeInteger(value.sequence) || value.sequence < 1) {
    throw new Error('electron release state.revocation.sequence must be a positive integer.');
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
  if (value.signatureVerified !== true) {
    throw new Error('electron release state.revocation.signatureVerified must be true.');
  }
  return {
    schema: ELECTRON_REVOCATION_SNAPSHOT_SCHEMA,
    authorityId: text(value.authorityId, 'electron release state.revocation.authorityId'),
    sequence: value.sequence,
    expiresAtUtc: instant(value.expiresAtUtc, 'electron release state.revocation.expiresAtUtc'),
    revokedSemanticRoots,
    digest: digest(value.digest, 'electron release state.revocation.digest'),
    signatureVerified: true,
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
    const current = await load();
    if (current.revocation && normalized.sequence <= current.revocation.sequence) {
      throw new Error('Electron revocation snapshot must advance monotonically.');
    }
    return commit(current, { ...current, revocation: normalized });
  }

  async function resolveCurrent() {
    const current = await load();
    if (!current.current) throw new Error('Electron release state has no active Pack.');
    if (!current.revocation) throw new Error('Electron release state has no verified revocation snapshot.');
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
