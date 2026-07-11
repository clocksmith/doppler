import { applyRuntimeProfile, getRuntimeConfig } from 'doppler-gpu/tooling';
import { log } from '../src/debug/index.js';
import { state } from './ui/state.js';

function $(id) { return document.getElementById(id); }

const DEMO_DEFAULT_MAX_TOKENS = 1024;

const GENERATION_FIELDS = [
  { key: 'temperature', id: 'set-temperature', path: ['inference', 'sampling', 'temperature'], parse: parseFiniteNumber },
  { key: 'topK', id: 'set-top-k', path: ['inference', 'sampling', 'topK'], parse: parsePositiveInteger },
  { key: 'topP', id: 'set-top-p', path: ['inference', 'sampling', 'topP'], parse: parseFiniteNumber },
  { key: 'maxTokens', id: 'set-max-tokens', path: ['inference', 'generation', 'maxTokens'], parse: parsePositiveInteger },
];

const PROFILE_OWNED_FIELDS = [
  { id: 'set-batch-max-tokens', path: ['inference', 'batching', 'maxTokens'] },
  { id: 'set-readback', path: ['inference', 'batching', 'readbackInterval'] },
  { id: 'set-kv-dtype', path: ['inference', 'kvcache', 'kvDtype'] },
  { id: 'set-kv-max-seq', path: ['inference', 'kvcache', 'maxSeqLen'] },
  { id: 'set-trace', path: ['shared', 'debug', 'trace', 'enabled'] },
  { id: 'set-log-level', path: ['shared', 'debug', 'logLevel', 'defaultLogLevel'] },
];

const ALL_PROFILE_DISPLAY_FIELDS = [...GENERATION_FIELDS, ...PROFILE_OWNED_FIELDS];

function parseFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parsePositiveInteger(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || Math.floor(parsed) !== parsed || parsed < 1) {
    return undefined;
  }
  return parsed;
}

function getNestedValue(obj, path) {
  let cur = obj;
  for (const key of path) {
    if (cur == null || typeof cur !== 'object') return undefined;
    cur = cur[key];
  }
  return cur;
}

function readField(el, parse) {
  if (!el) return undefined;
  if (el.type === 'checkbox') {
    return el.checked;
  }
  return parse(el.value);
}

function writeField(el, value) {
  if (!el) return;
  if (el.type === 'checkbox') {
    el.checked = value === true;
  } else if (value != null && value !== undefined) {
    el.value = String(value);
  } else {
    el.value = '';
  }
}

function populateUIFromConfig() {
  const config = getRuntimeConfig() || {};
  for (const field of ALL_PROFILE_DISPLAY_FIELDS) {
    const value = getNestedValue(config, field.path);
    writeField($(field.id), value);
  }
}

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
    log.warn('DemoSettings', `Failed to load runtime profile: ${error?.message || error}`);
    return false;
  }

  state.settings.runtimeProfile = targetProfile;
  populateUIFromConfig();
  return true;
}

function readGenerationSettings() {
  const settings = {};
  for (const field of GENERATION_FIELDS) {
    const value = readField($(field.id), field.parse);
    if (value !== undefined) {
      settings[field.key] = value;
    }
  }
  return settings;
}

export function getSettings() {
  state.tokenPressEnabled = $('set-token-press')?.checked ?? false;
  state.liveTokSec = $('set-live-toks')?.checked ?? true;

  const generationSettings = readGenerationSettings();
  const nextSettings = {
    ...generationSettings,
    runtimeProfile: state.settings.runtimeProfile || null,
  };
  state.settings = {
    ...state.settings,
    ...nextSettings,
  };
  return nextSettings;
}

export async function initSettings({ requireDefaultProfile = false } = {}) {
  // Toggle panel visibility
  const toggle = $('settings-toggle');
  const panel = $('settings-panel');
  if (toggle && panel) {
    toggle.addEventListener('click', () => {
      const isOpen = panel.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(isOpen));
    });
  }

  $('set-profile')?.addEventListener('change', (e) => {
    void applyProfile(e.target.value);
  });

  for (const field of PROFILE_OWNED_FIELDS) {
    const element = $(field.id);
    if (element) {
      element.disabled = true;
    }
  }

  const defaultProfile = $('set-profile')?.value || 'profiles/default';
  const applied = await applyProfile(defaultProfile, { required: requireDefaultProfile });
  if (applied) {
    writeField($('set-max-tokens'), DEMO_DEFAULT_MAX_TOKENS);
    state.settings.maxTokens = DEMO_DEFAULT_MAX_TOKENS;
  }
  return applied;
}
