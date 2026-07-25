export const RESOURCE_OWNERSHIP = Object.freeze([
  'borrowed',
  'scopeOwned',
  'submitOwned',
  'transferred',
  'retained',
]);

function assertOwnership(value) {
  if (!RESOURCE_OWNERSHIP.includes(value)) {
    throw new Error(`[ResourceScope] unsupported ownership "${String(value)}".`);
  }
  return value;
}

function normalizeLabel(label) {
  const value = String(label ?? '').trim();
  return value || 'resource';
}

class ResourceScope {
  constructor(mode, dispose) {
    this.mode = mode;
    this.dispose = dispose;
    this.entries = new Map();
    this.events = [];
    this.closed = false;
  }

  emit(action, entry, detail = null) {
    this.events.push(Object.freeze({
      sequence: this.events.length,
      action,
      label: entry.label,
      ownership: entry.ownership,
      detail,
    }));
  }

  assertOpen() {
    if (this.closed) {
      throw new Error('[ResourceScope] scope is already closed.');
    }
  }

  register(resource, label, ownership = 'scopeOwned') {
    this.assertOpen();
    if (!resource) {
      return resource;
    }
    const normalizedOwnership = assertOwnership(ownership);
    const existing = this.entries.get(resource);
    if (existing) {
      if (existing.disposed) {
        const entry = {
          resource,
          label: normalizeLabel(label),
          ownership: normalizedOwnership,
          disposed: false,
        };
        this.entries.set(resource, entry);
        this.emit('reacquire', entry, existing.label);
        return resource;
      }
      if (existing.ownership !== normalizedOwnership && existing.ownership !== 'transferred') {
        throw new Error(
          `[ResourceScope] "${existing.label}" is already registered as ${existing.ownership}; ` +
          `cannot register it as ${normalizedOwnership}.`
        );
      }
      this.emit('alias', existing, normalizeLabel(label));
      return resource;
    }
    const entry = {
      resource,
      label: normalizeLabel(label),
      ownership: normalizedOwnership,
      disposed: false,
    };
    this.entries.set(resource, entry);
    this.emit('acquire', entry);
    return resource;
  }

  transfer(resource, ownership, detail = null) {
    this.assertOpen();
    const entry = this.entries.get(resource);
    if (!entry) {
      throw new Error('[ResourceScope] cannot transfer an unregistered resource.');
    }
    if (entry.disposed) {
      throw new Error(`[ResourceScope] cannot transfer disposed resource "${entry.label}".`);
    }
    entry.ownership = assertOwnership(ownership);
    this.emit('transfer', entry, detail);
    return resource;
  }

  release(resource, label = null) {
    this.assertOpen();
    if (!resource) {
      return false;
    }
    let entry = this.entries.get(resource);
    if (!entry) {
      this.register(resource, label, this.mode === 'recorded' ? 'submitOwned' : 'scopeOwned');
      entry = this.entries.get(resource);
    }
    if (entry.disposed || entry.ownership === 'borrowed' || entry.ownership === 'transferred' || entry.ownership === 'retained') {
      this.emit('release-skip', entry);
      return false;
    }
    this.dispose(resource);
    entry.disposed = true;
    this.emit(this.mode === 'recorded' ? 'submit-retain' : 'release', entry);
    return true;
  }

  retain(resource, label = null, detail = null) {
    this.assertOpen();
    if (!resource) {
      return resource;
    }
    if (!this.entries.has(resource)) {
      this.register(resource, label, 'scopeOwned');
    }
    return this.transfer(resource, 'retained', detail);
  }

  close(outcome = 'success') {
    this.assertOpen();
    for (const entry of this.entries.values()) {
      if (
        !entry.disposed
        && (entry.ownership === 'scopeOwned' || entry.ownership === 'submitOwned')
      ) {
        this.release(entry.resource);
      }
    }
    this.closed = true;
    this.events.push(Object.freeze({
      sequence: this.events.length,
      action: 'close',
      label: this.mode,
      ownership: null,
      detail: outcome,
    }));
    return this.getEvents();
  }

  getEvents() {
    return Object.freeze([...this.events]);
  }
}

export function createImmediateResourceScope(options = {}) {
  if (typeof options.release !== 'function') {
    throw new Error('[ResourceScope] immediate scope requires a release(resource) function.');
  }
  return new ResourceScope('immediate', options.release);
}

export function createRecordedResourceScope(recorder) {
  if (!recorder || typeof recorder.trackTemporaryBuffer !== 'function') {
    throw new Error('[ResourceScope] recorded scope requires recorder.trackTemporaryBuffer().');
  }
  return new ResourceScope('recorded', (resource) => recorder.trackTemporaryBuffer(resource));
}
