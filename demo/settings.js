import { applyRuntimeProfile, getRuntimeConfig, setRuntimeConfig } from 'doppler-gpu/tooling';
import { state } from './ui/state.js';

function $(id) { return document.getElementById(id); }

// ---------------------------------------------------------------------------
// Every field maps to a path in the runtime config. When a profile is loaded
// via applyRuntimeProfile(), it calls setRuntimeConfig() which merges the
// profile values into the config. We then read the resolved config back into
// the UI fields. When the user edits a field, we write it back to the config.
//
// The ONLY settings not in the runtime config are the demo-only UI toggles:
// Token Press and Live tok/s.
// ---------------------------------------------------------------------------

const FIELDS = [
  // Sampling
  { id: 'set-temperature', path: ['inference', 'sampling', 'temperature'], parse: parseFloat, fallback: 1.0 },
  { id: 'set-top-k', path: ['inference', 'sampling', 'topK'], parse: (v) => parseInt(v, 10), fallback: 50 },
  { id: 'set-top-p', path: ['inference', 'sampling', 'topP'], parse: parseFloat, fallback: 0.95 },
  // Generation
  { id: 'set-max-tokens', path: ['inference', 'generation', 'maxTokens'], parse: (v) => parseInt(v, 10), fallback: 256 },
  // Batching
  { id: 'set-batch-max-tokens', path: ['inference', 'batching', 'maxTokens'], parse: (v) => parseInt(v, 10), fallback: 64 },
  { id: 'set-readback', path: ['inference', 'batching', 'readbackInterval'], parse: (v) => { const n = parseInt(v, 10); return n > 0 ? n : null; }, fallback: null },
  // KV Cache
  { id: 'set-kv-dtype', path: ['inference', 'kvcache', 'kvDtype'], parse: String, fallback: '' },
  { id: 'set-kv-max-seq', path: ['inference', 'kvcache', 'maxSeqLen'], parse: (v) => { const n = parseInt(v, 10); return n > 0 ? n : null; }, fallback: null },
  // Debug
  { id: 'set-trace', path: ['shared', 'debug', 'trace', 'enabled'], parse: (v) => v === 'true' || v === true, fallback: false },
  { id: 'set-log-level', path: ['shared', 'debug', 'logLevel', 'defaultLogLevel'], parse: String, fallback: 'info' },
];

// ---------------------------------------------------------------------------
// Nested object helpers
// ---------------------------------------------------------------------------

function getNestedValue(obj, path) {
  let cur = obj;
  for (const key of path) {
    if (cur == null || typeof cur !== 'object') return undefined;
    cur = cur[key];
  }
  return cur;
}

function setNestedValue(obj, path, value) {
  let cur = obj;
  for (let i = 0; i < path.length - 1; i++) {
    if (cur[path[i]] == null || typeof cur[path[i]] !== 'object') cur[path[i]] = {};
    cur = cur[path[i]];
  }
  cur[path[path.length - 1]] = value;
}

// ---------------------------------------------------------------------------
// Read / write UI ↔ runtime config
// ---------------------------------------------------------------------------

function readField(el, parse) {
  if (!el) return undefined;
  if (el.type === 'checkbox') return el.checked;
  const v = parse(el.value);
  if (v === null) return null;
  if (typeof v === 'string') return v || undefined;
  return Number.isFinite(v) ? v : undefined;
}

function writeField(el, value, fallback) {
  if (!el) return;
  if (el.type === 'checkbox') {
    el.checked = !!value;
  } else if (value != null && value !== undefined) {
    el.value = value;
  } else {
    el.value = fallback != null ? fallback : '';
  }
}

/** Read resolved runtime config → populate all UI fields. */
function populateUIFromConfig() {
  const config = getRuntimeConfig() || {};
  for (const field of FIELDS) {
    const value = getNestedValue(config, field.path);
    writeField($(field.id), value, field.fallback);
  }
}

/** Read all UI fields → write to runtime config via setRuntimeConfig(). */
function syncUIToConfig() {
  const config = getRuntimeConfig() || {};
  for (const field of FIELDS) {
    const v = readField($(field.id), field.parse);
    if (v !== undefined) {
      setNestedValue(config, field.path, v);
    }
  }
  setRuntimeConfig(config);
}

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------

async function applyProfile(profileId, { required = false } = {}) {
  const targetProfile = typeof profileId === 'string'
    ? profileId.trim()
    : '';
  if (!targetProfile) {
    if (required) {
      throw new Error('Runtime profile id is required.');
    }
    return false;
  }

  const previousProfile = state.settings.runtimeProfile || 'profiles/default';
  try {
    await applyRuntimeProfile(targetProfile);
  } catch (error) {
    if (required) {
      throw new Error(
        `Failed to load default runtime profile "${targetProfile}": ${error?.message ?? String(error)}`
      );
    }
    const profileSelect = $('set-profile');
    if (profileSelect) {
      profileSelect.value = previousProfile;
    }
    state.settings.runtimeProfile = previousProfile;
    console.warn('[DemoSettings] Failed to load runtime profile:', error?.message || error);
    return false;
  }

  state.settings.runtimeProfile = targetProfile;
  // Read the resolved (profile-merged) config back into UI
  populateUIFromConfig();
  return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Called by core.js before generation. Syncs UI → engine, returns settings. */
export function getSettings() {
  syncUIToConfig();

  // Read demo-only toggles
  state.tokenPressEnabled = $('set-token-press')?.checked ?? false;
  state.liveTokSec = $('set-live-toks')?.checked ?? true;

  // Return a snapshot for core.js to use in the decode loop.
  // All values come from the runtime config (now synced).
  const config = getRuntimeConfig() || {};
  return {
    temperature: getNestedValue(config, ['inference', 'sampling', 'temperature']) ?? 1.0,
    topK: getNestedValue(config, ['inference', 'sampling', 'topK']) ?? 50,
    topP: getNestedValue(config, ['inference', 'sampling', 'topP']) ?? 0.95,
    maxTokens: getNestedValue(config, ['inference', 'generation', 'maxTokens']) ?? 256,
    runtimeProfile: state.settings.runtimeProfile || 'profiles/default',
  };
}

export async function initSettings({ requireDefaultProfile = false } = {}) {
  // Toggle panel visibility
  const toggle = $('settings-toggle');
  const panel = $('settings-panel');
  if (toggle && panel) {
    toggle.addEventListener('click', () => panel.classList.toggle('is-open'));
  }

  // Profile select → load profile, merge into engine, repopulate UI
  $('set-profile')?.addEventListener('change', (e) => {
    void applyProfile(e.target.value);
  });

  // Any field change → sync to engine
  if (panel) {
    const syncHandler = (e) => {
      if (e.target?.id === 'set-profile') return;
      syncUIToConfig();
    };
    panel.addEventListener('change', syncHandler);
    panel.addEventListener('input', syncHandler);
  }

  // Apply default profile and populate UI from schema defaults.
  // Default profile load is required in normal demo mode so GPU capability policy
  // is initialized deterministically before any model can be compiled.
  const defaultProfile = $('set-profile')?.value || 'profiles/default';
  return applyProfile(defaultProfile, { required: requireDefaultProfile });
}
