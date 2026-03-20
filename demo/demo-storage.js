import { runQuickModelAction as runQuickModelActionFromCore } from './demo-core.js';

function resolveModeForTask(task, surface, preferredMode = null) {
  const normalizedSurface = normalizeSurface(surface, 'demo');
  const resolvedTask = resolveTaskForSurface(task, normalizedSurface, preferredMode);
  const allowedModes = getAllowedModesForTask(resolvedTask, normalizedSurface);
  if (allowedModes.length === 0) {
    return resolveModeForSurface(preferredMode || state.uiMode || 'run', normalizedSurface);
  }
  const normalizedPreferred = normalizeDeepLinkMode(preferredMode, null);
  if (normalizedPreferred && allowedModes.includes(normalizedPreferred)) {
    return normalizedPreferred;
  }
  const rememberedMode = normalizeDeepLinkMode(state.lastTaskMode?.[resolvedTask], null);
  if (rememberedMode && allowedModes.includes(rememberedMode)) {
    return rememberedMode;
  }
  const defaultTaskMode = normalizeDeepLinkMode(DEFAULT_TASK_MODE[resolvedTask], null);
  if (defaultTaskMode && allowedModes.includes(defaultTaskMode)) {
    return defaultTaskMode;
  }
  return allowedModes[0];
}

function parseAllowedTasks(rawTasks) {
  return resolveText(rawTasks, '')
    .split(/\s+/)
    .map((value) => normalizeTask(value, null))
    .filter(Boolean);
}

function syncSurfaceUI(surface) {
  const normalizedSurface = normalizeSurface(surface, 'demo');
  const normalizedTask = resolveTaskForSurface(
    getTaskForMode(state.uiMode, state.uiTask || 'run'),
    normalizedSurface,
    state.uiMode
  );
  const app = $('app');
  if (app) {
    app.dataset.surface = normalizedSurface;
  }
  document.querySelectorAll('.surface-tab').forEach((button) => {
    const isActive = button.dataset.surface === normalizedSurface;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
  document.querySelectorAll('.task-tab').forEach((button) => {
    const task = normalizeTask(button.dataset.task, null);
    const taskAllowedForSurface = task != null && isTaskAllowedForSurface(task, normalizedSurface);
    const isVisible = true;
    if (button instanceof HTMLButtonElement) {
      button.hidden = !isVisible;
    }
    button.classList.toggle('is-unavailable', isVisible && !taskAllowedForSurface);
    button.setAttribute('aria-disabled', (isVisible && !taskAllowedForSurface) ? 'true' : 'false');
    button.setAttribute('aria-hidden', isVisible ? 'false' : 'true');
  });
  document.querySelectorAll('.mode-subtab').forEach((button) => {
    const mode = normalizeDeepLinkMode(button.dataset.mode, null);
    const modeAllowed = mode != null && isModeAllowedForSurface(mode, normalizedSurface);
    const isVisible = true;
    const isUnavailable = !modeAllowed;
    button.hidden = false;
    button.classList.toggle('is-unavailable', isUnavailable);
    button.setAttribute('aria-disabled', isUnavailable ? 'true' : 'false');
    button.setAttribute('aria-hidden', 'false');
  });
  if (typeof document !== 'undefined') {
    document.title = 'Doppler';
  }
}

function readDeepLinkValue(hashParams, queryParams, keys) {
  for (const key of keys) {
    const hashValue = hashParams.get(key);
    if (hashValue != null && hashValue !== '') return hashValue;
    const queryValue = queryParams.get(key);
    if (queryValue != null && queryValue !== '') return queryValue;
  }
  return null;
}

function decodeDeepLinkText(rawText) {
  const text = resolveText(rawText);
  if (!text) return '';
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

function readDeepLinkStateFromLocation() {
  if (typeof window === 'undefined') {
    return {
      surface: 'demo',
      task: null,
      mode: null,
      sourceCode: DEFAULT_TRANSLATE_SOURCE,
      targetCode: DEFAULT_TRANSLATE_TARGET,
      text: null,
      compareEnabled: false,
      compareLayoutId: 'proof',
      lanes: null,
    };
  }

  const queryParams = new URLSearchParams(window.location.search);
  const hashRaw = resolveText(window.location.hash).replace(/^#/, '').replace(/^\?/, '');
  const hashParams = new URLSearchParams(hashRaw);

  const sourceRaw = readDeepLinkValue(hashParams, queryParams, ['sl', 'source', 'source_lang_code']);
  const targetRaw = readDeepLinkValue(hashParams, queryParams, ['tl', 'target', 'target_lang_code']);
  const textRaw = readDeepLinkValue(hashParams, queryParams, ['text', 'prompt', 'q']);
  const taskRaw = readDeepLinkValue(hashParams, queryParams, ['task', 't']);
  const modeRaw = readDeepLinkValue(hashParams, queryParams, ['mode', 'm']);
  const surfaceRaw = readDeepLinkValue(hashParams, queryParams, ['surface', 's']);
  const compareRaw = readDeepLinkValue(hashParams, queryParams, ['compare', 'cv']);
  const compareLayoutRaw = readDeepLinkValue(hashParams, queryParams, ['compare_layout', 'cp']);
  const leftEngineRaw = readDeepLinkValue(hashParams, queryParams, ['left_engine', 'le']);
  const rightEngineRaw = readDeepLinkValue(hashParams, queryParams, ['right_engine', 're']);
  const leftModelRaw = readDeepLinkValue(hashParams, queryParams, ['left_model', 'lm']);
  const rightModelRaw = readDeepLinkValue(hashParams, queryParams, ['right_model', 'rm']);
  const surface = normalizeSurface(surfaceRaw, 'demo');

  let task = normalizeTask(taskRaw, null);
  let mode = normalizeDeepLinkMode(modeRaw, null);
  if (!mode && (sourceRaw != null || targetRaw != null || textRaw != null)) {
    mode = 'translate';
  }
  if (!mode && task) {
    mode = resolveModeForTask(task, surface, null);
  }
  if (mode && !isModeAllowedForSurface(mode, surface)) {
    mode = resolveModeForSurface(mode, surface);
  }
  if (mode) {
    task = resolveTaskForSurface(getTaskForMode(mode, task), surface, mode);
  } else if (task && !isTaskAllowedForSurface(task, surface)) {
    task = resolveTaskForSurface(task, surface, null);
  }

  const sourceCode = normalizeTranslateLanguageCode(sourceRaw, DEFAULT_TRANSLATE_SOURCE);
  let targetCode = normalizeTranslateLanguageCode(targetRaw, DEFAULT_TRANSLATE_TARGET);
  if (targetCode === sourceCode) {
    targetCode = sourceCode === DEFAULT_TRANSLATE_TARGET
      ? DEFAULT_TRANSLATE_SOURCE
      : DEFAULT_TRANSLATE_TARGET;
  }

  return {
    surface,
    task,
    mode,
    sourceCode,
    targetCode,
    text: textRaw == null ? null : decodeDeepLinkText(textRaw),
    compareEnabled: compareRaw === '1' || compareRaw === 'true' || compareRaw === 'compare',
    compareLayoutId: resolveText(compareLayoutRaw, 'proof'),
    lanes: {
      left: {
        engine: resolveText(leftEngineRaw, ''),
        modelId: resolveText(leftModelRaw, ''),
      },
      right: {
        engine: resolveText(rightEngineRaw, ''),
        modelId: resolveText(rightModelRaw, ''),
      },
    },
  };
}

function applyDeepLinkStateToUI(deepLinkState) {
  const sourceSelect = $('translate-source-language');
  const targetSelect = $('translate-target-language');
  if (sourceSelect instanceof HTMLSelectElement) {
    sourceSelect.value = normalizeTranslateLanguageCode(deepLinkState?.sourceCode, DEFAULT_TRANSLATE_SOURCE);
  }
  if (targetSelect instanceof HTMLSelectElement) {
    targetSelect.value = normalizeTranslateLanguageCode(deepLinkState?.targetCode, DEFAULT_TRANSLATE_TARGET);
  }
  if (sourceSelect instanceof HTMLSelectElement && targetSelect instanceof HTMLSelectElement) {
    const selected = getTranslateLanguageSelection();
    sourceSelect.value = selected.sourceCode;
    targetSelect.value = selected.targetCode;
  }

  if (typeof deepLinkState?.text === 'string') {
    const promptEl = $('run-prompt');
    if (promptEl instanceof HTMLTextAreaElement) {
      promptEl.value = deepLinkState.text;
      setStarterExampleInput(promptEl, false);
    }
  }
  ensureTranslateCompareRuntimeState();
  state.compareEnabled = deepLinkState?.compareEnabled === true;
  state.compareLayoutId = resolveText(deepLinkState?.compareLayoutId, state.compareLayoutId || 'proof');
  for (const laneId of getCompareLaneIds()) {
    const laneState = deepLinkState?.lanes?.[laneId] || {};
    const lane = getCompareLane(laneId);
    if (!lane) continue;
    if (laneState.engine === 'doppler' || laneState.engine === 'transformersjs') {
      lane.engine = laneState.engine;
    }
    if (laneState.modelId) {
      lane.modelId = laneState.modelId;
    }
  }
}

function buildDeepLinkHash(modeOverride = null, taskOverride = null) {
  const surface = normalizeSurface(state.surface, 'demo');
  const mode = resolveModeForSurface(resolveText(modeOverride, state.uiMode || 'run'), surface);
  const modeTask = getTaskForMode(mode, 'run');
  const task = resolveTaskForSurface(
    resolveText(taskOverride, modeTask),
    surface,
    mode
  );
  const params = new URLSearchParams();

  if (task !== 'run') {
    params.set('task', task);
  }

  if (mode !== 'run') {
    params.set('mode', mode);
  }

  if (mode === 'translate') {
    const promptEl = $('run-prompt');
    const prompt = resolveText(promptEl?.value, '');
    const { sourceCode, targetCode } = getTranslateLanguageSelection();
    params.set('sl', sourceCode);
    params.set('tl', targetCode);
    const shouldIncludePrompt = prompt.length > 0 && !isStarterExampleInput(promptEl);
    if (shouldIncludePrompt) {
      params.set('text', prompt);
    }
    if (state.compareEnabled) {
      params.set('compare', '1');
      params.set('compare_layout', state.compareLayoutId || 'proof');
      for (const laneId of getCompareLaneIds()) {
        const lane = getCompareLane(laneId);
        if (!lane) continue;
        const engineKey = laneId === 'left' ? 'le' : 're';
        const modelKey = laneId === 'left' ? 'lm' : 'rm';
        params.set(engineKey, resolveText(lane.engine, 'doppler'));
        if (lane.modelId) {
          params.set(modelKey, lane.modelId);
        }
      }
    }
  }

  return params.toString();
}

function syncDeepLinkFromUI() {
  if (typeof window === 'undefined' || typeof window.history?.replaceState !== 'function') {
    return;
  }
  const next = new URL(window.location.href);
  next.hash = buildDeepLinkHash();
  const nextPath = `${next.pathname}${next.search}${next.hash}`;
  const currentPath = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextPath === currentPath) return;
  window.history.replaceState(null, '', nextPath);
}

function buildTranslateDeepLinkUrl() {
  const next = new URL(window.location.href);
  next.hash = buildDeepLinkHash('translate');
  return next.toString();
}

function setTranslateCompareEnabled(enabled) {
  state.compareEnabled = enabled === true;
  syncTranslateCompareUI();
  syncDeepLinkFromUI();
}

async function copyTranslateCompareShareLink() {
  const url = buildTranslateDeepLinkUrl();
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(url);
    updateRunStatus('Compare link copied');
    return;
  }
  updateRunStatus(url);
}

function getRunStarterPromptPool() {
  if (state.uiMode === 'translate') {
    return TRANSLATE_STARTER_PROMPTS;
  }
  return RUN_STARTER_PROMPTS;
}

function readGlobalString(key) {
  if (!key || typeof globalThis !== 'object' || !globalThis) return '';
  const value = globalThis[key];
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeUrlPathname(pathname) {
  return typeof pathname === 'string' ? pathname.replace(/\/+/g, '/') : '';
}

function isHuggingFaceHost(hostname) {
  if (typeof hostname !== 'string' || !hostname) return false;
  const lowered = hostname.toLowerCase();
  return lowered === QUICK_MODEL_HF_HOST || lowered.endsWith(`.${QUICK_MODEL_HF_HOST}`);
}

function buildHfResolveUrl(repoId, revision, path) {
  const normalizedRepoId = typeof repoId === 'string' ? repoId.trim().replace(/^\/+|\/+$/g, '') : '';
  const normalizedRevision = typeof revision === 'string' ? revision.trim() : '';
  const normalizedPath = typeof path === 'string' ? path.trim().replace(/^\/+/, '') : '';
  if (!normalizedRepoId || !normalizedRevision) return '';
  const pathSuffix = normalizedPath ? `/${normalizedPath}` : '';
  return `https://huggingface.co/${normalizedRepoId}/resolve/${encodeURIComponent(normalizedRevision)}${pathSuffix}`;
}

function extractHfResolveRevisionFromUrl(inputUrl) {
  try {
    const parsed = new URL(inputUrl);
    if (!isHuggingFaceHost(parsed.hostname)) return null;
    const parts = normalizeUrlPathname(parsed.pathname).split('/').filter(Boolean);
    const resolveIndex = parts.indexOf('resolve');
    if (resolveIndex < 0 || resolveIndex + 1 >= parts.length) return null;
    return decodeURIComponent(parts[resolveIndex + 1]);
  } catch {
    return null;
  }
}

function isImmutableHfResolveUrl(inputUrl) {
  const revision = extractHfResolveRevisionFromUrl(inputUrl);
  return !!(revision && QUICK_MODEL_HF_COMMIT_PATTERN.test(revision));
}

function resolveRemoteCacheMode(inputUrl) {
  return isImmutableHfResolveUrl(inputUrl) ? 'force-cache' : 'default';
}

function buildQuickCatalogCandidateUrls() {
  const candidates = [];
  if (QUICK_MODEL_CATALOG_OVERRIDE_URL) {
    candidates.push(QUICK_MODEL_CATALOG_OVERRIDE_URL);
  }
  const hfCatalogUrl = buildHfResolveUrl(
    QUICK_MODEL_CATALOG_HF_REPO_ID,
    QUICK_MODEL_CATALOG_HF_REVISION,
    QUICK_MODEL_CATALOG_HF_PATH
  );
  if (hfCatalogUrl) {
    candidates.push(hfCatalogUrl);
  }
  candidates.push(QUICK_MODEL_CATALOG_LOCAL_URL);
  const deduped = [];
  const seen = new Set();
  for (const candidate of candidates) {
    if (typeof candidate !== 'string') continue;
    const trimmed = candidate.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    deduped.push(trimmed);
  }
  return deduped;
}

function normalizeQuickLookupToken(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

function normalizeQuickCatalogAliases(rawAliases, modelId) {
  const aliases = [];
  if (Array.isArray(rawAliases)) {
    for (const alias of rawAliases) {
      if (typeof alias !== 'string') continue;
      const trimmed = alias.trim();
      if (trimmed) aliases.push(trimmed);
    }
  } else if (typeof rawAliases === 'string') {
    const trimmed = rawAliases.trim();
    if (trimmed) aliases.push(trimmed);
  }
  aliases.push(modelId);
  const deduped = [];
  const seen = new Set();
  for (const alias of aliases) {
    const token = normalizeQuickLookupToken(alias);
    if (!token || seen.has(token)) continue;
    seen.add(token);
    deduped.push(alias.trim());
  }
  return deduped;
}

function normalizeQuickCatalogHfSpec(rawHf) {
  if (!rawHf || typeof rawHf !== 'object' || Array.isArray(rawHf)) return null;
  const repoId = typeof rawHf.repoId === 'string' ? rawHf.repoId.trim() : '';
  const revision = typeof rawHf.revision === 'string' ? rawHf.revision.trim() : '';
  const path = typeof rawHf.path === 'string' ? rawHf.path.trim() : '';
  if (!repoId || !revision) return null;
  return {
    repoId,
    revision,
    path,
  };
}

function isQuickCatalogHfSourceUrl(catalogSourceUrl) {
  try {
    return isHuggingFaceHost(new URL(catalogSourceUrl).hostname);
  } catch {
    return false;
  }
}

function hasQuickCatalogExplicitBaseUrl(baseUrl) {
  return typeof baseUrl === 'string' && baseUrl.trim().length > 0;
}

function extractHfRepoIdFromInput(value) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) return '';
  if (raw.startsWith('hf://')) {
    const sliced = raw.slice(5).replace(/^\/+/, '');
    const [owner, repo] = sliced.split('/');
    if (owner && repo) {
      return `${owner}/${repo}`.toLowerCase();
    }
  }
  try {
    const parsed = new URL(raw);
    if (!isHuggingFaceHost(parsed.hostname)) return '';
    const [owner, repo] = normalizeUrlPathname(parsed.pathname).split('/').filter(Boolean);
    if (owner && repo) {
      return `${owner}/${repo}`.toLowerCase();
    }
    return '';
  } catch {
    const match = raw.match(/^([A-Za-z0-9._-]+)\/([A-Za-z0-9._-]+)$/);
    if (!match) return '';
    return `${match[1]}/${match[2]}`.toLowerCase();
  }
}

function collectQuickCatalogLookupTokens(values) {
  const tokens = new Set();
  for (const value of values || []) {
    const raw = typeof value === 'string' ? value.trim() : '';
    if (!raw) continue;
    tokens.add(normalizeQuickLookupToken(raw));
    const hfRepoId = extractHfRepoIdFromInput(raw);
    if (hfRepoId) {
      tokens.add(hfRepoId);
    }
  }
  return tokens;
}

function findQuickCatalogEntryForRegistryInput(values) {
  const lookup = collectQuickCatalogLookupTokens(values);
  if (lookup.size === 0) return null;
  for (const entry of getQuickCatalogEntries()) {
    const modelToken = normalizeQuickLookupToken(entry?.modelId);
    if (modelToken && lookup.has(modelToken)) return entry;
    const hfRepoToken = normalizeQuickLookupToken(entry?.hfRepoId);
    if (hfRepoToken && lookup.has(hfRepoToken)) return entry;
    const aliases = Array.isArray(entry?.aliases) ? entry.aliases : [];
    for (const alias of aliases) {
      const aliasToken = normalizeQuickLookupToken(alias);
      if (aliasToken && lookup.has(aliasToken)) {
        return entry;
      }
    }
  }
  return null;
}

function resolveDirectRdrrBaseUrlFromInput(value) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) return '';
  try {
    const parsed = new URL(raw);
    const normalizedPath = normalizeUrlPathname(parsed.pathname);
    if (!normalizedPath.endsWith('/manifest.json')) return '';
    parsed.pathname = normalizedPath.replace(/\/manifest\.json$/, '/');
    parsed.search = '';
    parsed.hash = '';
    return parsed.toString().replace(/\/+$/, '');
  } catch {
    return '';
  }
}

function normalizeQuickModeToken(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'run' || normalized === 'text') return 'run';
  if (normalized === 'translate' || normalized === 'translation') return 'translate';
  if (normalized === 'embedding' || normalized === 'embed') return 'embedding';
  if (normalized === 'diffusion' || normalized === 'image') return 'diffusion';
  if (normalized === 'energy') return 'energy';
  return null;
}

function normalizeQuickModes(rawMode, rawModes) {
  const values = [];
  if (Array.isArray(rawModes)) values.push(...rawModes);
  if (rawMode !== undefined) values.push(rawMode);
  const tokens = new Set();
  for (const value of values) {
    if (typeof value === 'string') {
      const lowered = value.trim().toLowerCase();
      if (lowered === 'both' || lowered === 'all' || lowered === 'text+embedding') {
        tokens.add('run');
        tokens.add('translate');
        tokens.add('embedding');
        continue;
      }
      const splitValues = lowered.split(/[,\s+/]+/).filter(Boolean);
      for (const token of splitValues) {
        const normalized = normalizeQuickModeToken(token);
        if (normalized) tokens.add(normalized);
      }
      continue;
    }
    const normalized = normalizeQuickModeToken(value);
    if (normalized) tokens.add(normalized);
  }
  if (tokens.size === 0) {
    tokens.add('run');
  }
  return [...tokens];
}

function resolveQuickModelBaseUrl(baseUrl, modelId, catalogSourceUrl, hfSpec = null) {
  if (hfSpec?.repoId && hfSpec?.revision) {
    const hfPath = hfSpec.path || `models/${encodeURIComponent(modelId)}`;
    const resolvedHfUrl = buildHfResolveUrl(hfSpec.repoId, hfSpec.revision, hfPath).replace(/\/+$/, '');
    return isQuickModelAllowedUrl(resolvedHfUrl) ? resolvedHfUrl : null;
  }

  if (!hasQuickCatalogExplicitBaseUrl(baseUrl)) {
    return null;
  }

  const resolved = new URL(baseUrl.trim(), catalogSourceUrl || QUICK_MODEL_CATALOG_LOCAL_BASE_URL).toString();
  return isQuickModelAllowedUrl(resolved) ? resolved : null;
}

function isQuickModelLocalUrl(resolvedUrl) {
  try {
    const resolved = new URL(resolvedUrl);
    const catalogUrl = new URL(QUICK_MODEL_CATALOG_LOCAL_BASE_URL);
    if (resolved.origin !== catalogUrl.origin) return false;
    const normalizedPath = normalizeUrlPathname(resolved.pathname);
    return normalizedPath.startsWith('/models/local/');
  } catch {
    return false;
  }
}

function isQuickModelHfResolveUrl(resolvedUrl) {
  try {
    const resolved = new URL(resolvedUrl);
    if (!isHuggingFaceHost(resolved.hostname)) return false;
    const normalizedPath = normalizeUrlPathname(resolved.pathname);
    return normalizedPath.includes('/resolve/');
  } catch {
    return false;
  }
}

function isQuickModelAllowedUrl(resolvedUrl) {
  return isQuickModelLocalUrl(resolvedUrl) || isQuickModelHfResolveUrl(resolvedUrl);
}

function normalizeQuickCatalogEntry(raw, index, catalogSourceUrl) {
  if (!raw || typeof raw !== 'object') return null;
  const modelId = typeof raw.modelId === 'string' ? raw.modelId.trim() : '';
  if (!modelId) return null;
  const hfSpec = normalizeQuickCatalogHfSpec(
    (raw.hf && typeof raw.hf === 'object' && !Array.isArray(raw.hf))
      ? raw.hf
      : {
        repoId: raw.hfRepoId,
        revision: raw.hfRevision,
        path: raw.hfPath,
      }
  );
  if (isQuickCatalogHfSourceUrl(catalogSourceUrl) && !hfSpec) {
    return null;
  }
  const resolvedBaseUrl = resolveQuickModelBaseUrl(raw.baseUrl, modelId, catalogSourceUrl, hfSpec);
  if (!resolvedBaseUrl) return null;
  const modes = normalizeQuickModes(raw.mode, raw.modes);
  const sizeBytes = Number(raw.sizeBytes);
  const aliases = normalizeQuickCatalogAliases(raw.aliases, modelId);
  return {
    id: modelId,
    modelId,
    aliases,
    label: typeof raw.label === 'string' && raw.label.trim() ? raw.label.trim() : modelId,
    description: typeof raw.description === 'string' ? raw.description.trim() : '',
    baseUrl: resolvedBaseUrl,
    hfRepoId: hfSpec?.repoId || null,
    hfRevision: hfSpec?.revision || null,
    modes,
    sizeBytes: Number.isFinite(sizeBytes) && sizeBytes > 0 ? Math.floor(sizeBytes) : null,
    recommended: raw.recommended === true,
    sortOrder: Number.isFinite(Number(raw.sortOrder)) ? Number(raw.sortOrder) : index,
  };
}

function parseQuickCatalogPayload(payload, catalogSourceUrl) {
  if (!payload || typeof payload !== 'object') {
    return [];
  }
  const entries = Array.isArray(payload.models) ? payload.models : [];
  const normalized = [];
  for (let i = 0; i < entries.length; i += 1) {
    const entry = normalizeQuickCatalogEntry(entries[i], i, catalogSourceUrl);
    if (!entry) continue;
    normalized.push(entry);
  }
  normalized.sort((a, b) => {
    if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
    if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
    return a.label.localeCompare(b.label);
  });
  return normalized;
}

function getQuickCatalogEntries() {
  return Array.isArray(state.quickModelCatalog) ? state.quickModelCatalog : [];
}

function getQuickCatalogEntriesForSurface(surface = state.surface) {
  const allowedModes = getAllowedModesForSurface(surface);
  return getQuickCatalogEntries().filter((entry) => (
    Array.isArray(entry?.modes) && entry.modes.some((modeToken) => allowedModes.has(modeToken))
  ));
}

function formatQuickModelBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'size unknown';
  return formatBytes(bytes);
}

function setDistillStatus(message, isError = false) {
  const statusEl = $('distill-status');
  if (!statusEl) return;
  statusEl.textContent = message;
  statusEl.dataset.state = isError ? 'error' : 'ready';
}

function setDistillOutput(payload) {
  const outputEl = $('distill-output');
  if (!outputEl) return;
  if (!payload || typeof payload !== 'object') {
    outputEl.textContent = 'No distill output yet.';
    return;
  }
  outputEl.textContent = JSON.stringify(payload, null, 2);
}

function getDistillWorkloads() {
  return Array.isArray(state.distillWorkloads) ? state.distillWorkloads : [];
}

function populateDistillWorkloadSelect() {
  const selectEl = $('distill-workload-select');
  if (!(selectEl instanceof HTMLSelectElement)) return;
  const previous = selectEl.value || '';
  selectEl.innerHTML = '';

  const noneOption = document.createElement('option');
  noneOption.value = '';
  noneOption.textContent = 'None';
  selectEl.appendChild(noneOption);

  for (const workload of getDistillWorkloads()) {
    const option = document.createElement('option');
    option.value = workload.id;
    const suffix = workload.workloadKind ? ` (${workload.workloadKind})` : '';
    option.textContent = `${workload.id}${suffix}`;
    selectEl.appendChild(option);
  }
  selectEl.value = Array.from(selectEl.options).some((option) => option.value === previous)
    ? previous
    : '';
}

function findDistillWorkloadById(workloadId) {
  if (!workloadId) return null;
  return getDistillWorkloads().find((entry) => entry.id === workloadId) || null;
}

async function loadDistillWorkloadRegistry() {
  state.distillWorkloadsLoading = true;
  state.distillWorkloadsError = null;
  try {
    const response = await fetch(DISTILL_WORKLOAD_REGISTRY_URL, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const workloads = Array.isArray(payload?.workloads) ? payload.workloads : [];
    state.distillWorkloads = workloads
      .filter((entry) => entry && typeof entry === 'object' && typeof entry.id === 'string' && entry.id.trim())
      .map((entry) => ({
        id: entry.id.trim(),
        path: typeof entry.path === 'string' ? entry.path : null,
        sha256: typeof entry.sha256 === 'string' ? entry.sha256 : null,
        workloadKind: typeof entry.workloadKind === 'string' ? entry.workloadKind : null,
      }));
    populateDistillWorkloadSelect();
    if (state.distillWorkloads.length > 0) {
      setDistillStatus(`Loaded ${state.distillWorkloads.length} workload pack entries.`);
    } else {
      setDistillStatus('No workload packs found in registry.');
    }
  } catch (error) {
    state.distillWorkloads = [];
    state.distillWorkloadsError = error instanceof Error ? error.message : String(error);
    populateDistillWorkloadSelect();
    setDistillStatus(`Workload registry unavailable: ${state.distillWorkloadsError}`, true);
  } finally {
    state.distillWorkloadsLoading = false;
  }
}

async function readFileAsText(file) {
  if (!file) return '';
  return file.text();
}

async function handleDistillReplay() {
  const teacherJsonEl = $('distill-teacher-json');
  const workloadSelect = $('distill-workload-select');
  const teacherJsonText = teacherJsonEl?.value || '';
  const selectedWorkloadId = workloadSelect?.value || '';
  const workloadEntry = findDistillWorkloadById(selectedWorkloadId);

  setDistillStatus('Running replay...');
  const result = await runDistillReplay({
    teacherJsonText,
    workloadEntry,
  });
  state.distillLastReplay = result;
  setDistillOutput(result);
  const steps = Array.isArray(result.timeline) ? result.timeline.length : 0;
  const reportId = result.traceability?.teacherReportId || 'unknown';
  setDistillStatus(`Replay complete. Steps: ${steps}. teacherReportId: ${reportId}`);
}

function exportDistillReplay() {
  if (!state.distillLastReplay) {
    setDistillStatus('No replay result available to export.', true);
    return;
  }
  const payload = state.distillLastReplay;
  const timestamp = new Date().toISOString().replace(/[:]/g, '-');
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `distill-replay-${timestamp}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}


function resolveDownloadProgressForModel(modelId) {
  const progress = state.downloadProgress;
  if (!progress || typeof progress !== 'object') return null;
  const progressModelId = typeof progress.modelId === 'string' ? progress.modelId : '';
  if (modelId && progressModelId && progressModelId !== modelId) return null;

  const percent = Number(progress.percent);
  const downloadedBytes = Number(progress.downloadedBytes);
  const totalBytes = Number(progress.totalBytes);
  const totalShards = Number(progress.totalShards);
  const completedShards = Number(progress.completedShards);
  const currentShard = Number(progress.currentShard);
  const speed = Number(progress.speed);
  return {
    modelId: progressModelId || modelId || '',
    percent: Number.isFinite(percent) ? clampPercent(percent) : null,
    downloadedBytes: Number.isFinite(downloadedBytes) && downloadedBytes > 0 ? downloadedBytes : 0,
    totalBytes: Number.isFinite(totalBytes) && totalBytes > 0 ? totalBytes : 0,
    totalShards: Number.isFinite(totalShards) && totalShards > 0 ? totalShards : 0,
    completedShards: Number.isFinite(completedShards) && completedShards > 0 ? completedShards : 0,
    currentShard: Number.isFinite(currentShard) && currentShard > 0 ? currentShard : null,
    speed: Number.isFinite(speed) && speed > 0 ? speed : 0,
  };
}

function findQuickModelEntry(modelId) {
  return getQuickCatalogEntries().find((entry) => entry.modelId === modelId) || null;
}

function formatQuickModelModeBadge(modes = []) {
  if (!Array.isArray(modes) || modes.length === 0) return 'text';
  const labels = [];
  if (modes.includes('run')) {
    labels.push('text');
  } else if (modes.includes('translate')) {
    labels.push('translate');
  }
  if (modes.includes('embedding')) labels.push('embedding');
  if (modes.includes('diffusion')) labels.push('diffusion');
  if (modes.includes('energy')) labels.push('energy');
  return labels.length > 0 ? labels.join('+') : 'text';
}

function getComparableQuickModelSize(entry) {
  const size = Number(entry?.sizeBytes);
  return Number.isFinite(size) && size > 0 ? size : Number.POSITIVE_INFINITY;
}

function getSmallestQuickModelForMode(modeToken) {
  if (!modeToken) return null;
  const candidates = getQuickCatalogEntries().filter((entry) => entry.modes.includes(modeToken));
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => {
    const sizeDiff = getComparableQuickModelSize(a) - getComparableQuickModelSize(b);
    if (sizeDiff !== 0) return sizeDiff;
    if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
    return a.label.localeCompare(b.label);
  });
  return candidates[0] || null;
}

function getPreferredQuickModelForMode(modeToken) {
  if (!modeToken) return null;
  if (modeToken === 'translate') {
    const candidates = getQuickCatalogEntries()
      .filter((entry) => Array.isArray(entry?.modes) && entry.modes.includes('translate'));
    if (candidates.length === 0) return null;
    candidates.sort((a, b) => {
      const sizeDiff = getComparableQuickModelSize(a) - getComparableQuickModelSize(b);
      if (sizeDiff !== 0) return sizeDiff;
      if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
      return a.label.localeCompare(b.label);
    });
    const preferredModelId = TRANSLATE_COMPARE_DEFAULT_STUDENT_MODEL_ID;
    const preferredEntry = candidates.find((entry) => entry?.modelId === preferredModelId);
    return preferredEntry || candidates[0] || null;
  }
  if (modeToken !== 'run') {
    return getSmallestQuickModelForMode(modeToken);
  }
  const candidates = getQuickCatalogEntries().filter((entry) => (
    Array.isArray(entry?.modes) && entry.modes.includes(modeToken)
  ));
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => {
    const scoreDiff = getModelSelectionScore(modeToken, b.modelId) - getModelSelectionScore(modeToken, a.modelId);
    if (scoreDiff !== 0) return scoreDiff;
    const sizeDiff = getComparableQuickModelSize(a) - getComparableQuickModelSize(b);
    if (sizeDiff !== 0) return sizeDiff;
    if (a.sortOrder !== b.sortOrder) return a.sortOrder - b.sortOrder;
    return a.label.localeCompare(b.label);
  });
  return candidates[0] || null;
}

function getDiagnosticsRequiredQuickMode() {
  const selection = state.diagnosticsSelections?.diagnostics || {};
  const selectedProfile = decodeDiagnosticsProfileId(selection.profile || '');
  const suite = selectedProfile?.suite || selection.suite || getDiagnosticsDefaultSuite('diagnostics');
  if (suite === 'kernels') return null;
  if (suite === 'diffusion') return 'diffusion';
  if (suite === 'energy') return 'energy';
  const runtimeProfile = String(selectedProfile?.runtimeProfile || selection.runtimeProfile || '').toLowerCase();
  if (runtimeProfile.includes('embedding')) return 'embedding';
  return 'run';
}

function updateNavState(mode, task = null) {
  const normalizedMode = normalizeDeepLinkMode(mode, 'run');
  const normalizedTask = resolveTaskForSurface(
    getTaskForMode(normalizedMode, task || state.uiTask || 'run'),
    state.surface,
    normalizedMode
  );

  document.querySelectorAll('.task-tab').forEach((button) => {
    const buttonTask = normalizeTask(button.dataset.task, null);
    const isActive = buttonTask === normalizedTask && !button.hidden;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });

  document.querySelectorAll('.mode-subtab').forEach((button) => {
    const buttonMode = normalizeDeepLinkMode(button.dataset.mode, null);
    const isActive = buttonMode === normalizedMode && !button.hidden;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
}


function cloneRuntimeConfig(config) {
  try {
    return structuredClone(config);
  } catch {
    return JSON.parse(JSON.stringify(config));
  }
}

function applyModeVisibility(mode) {
  const panels = document.querySelectorAll('[data-modes]');
  panels.forEach((panel) => {
    const modes = panel.dataset.modes?.split(/\s+/).filter(Boolean) || [];
    const modeVisible = modes.length === 0 || modes.includes(mode);
    panel.hidden = !modeVisible;
  });
}

function ensurePrimaryModeControlStack() {
  const panelGrid = $('panel-grid');
  if (!panelGrid) return;

  const primaryStack = panelGrid.querySelector('.panel-stack-primary');
  const railStack = panelGrid.querySelector('.panel-stack-rail');
  const insertionPoint = primaryStack || railStack;
  if (!insertionPoint) return;

  let controlsStack = panelGrid.querySelector('.panel-stack-controls');
  if (!controlsStack) {
    controlsStack = document.createElement('div');
    controlsStack.className = 'panel-stack panel-stack-controls';
    controlsStack.dataset.modes = 'run translate embedding diffusion energy';
  }
  panelGrid.insertBefore(controlsStack, insertionPoint);

  const controlSectionSelectors = [
    '.run-controls-panel',
    '.diffusion-controls-panel',
    '.energy-controls-panel',
  ];
  for (const selector of controlSectionSelectors) {
    const section = panelGrid.querySelector(selector);
    if (!section || section.parentElement === controlsStack) continue;
    controlsStack.appendChild(section);
  }
}

function syncRunModeUI(mode) {
  const isEmbeddingMode = mode === 'embedding';
  const isTranslateMode = mode === 'translate';
  setText(
    $('run-panel-title'),
    isEmbeddingMode ? 'Embeddings' : (isTranslateMode ? 'Translation' : 'Text Generation')
  );
  setText(
    $('run-controls-title'),
    isEmbeddingMode ? 'Embedding Controls' : (isTranslateMode ? 'Translation Controls' : 'Run Controls')
  );
  setText($('run-prompt-label'), isEmbeddingMode ? 'Input text' : (isTranslateMode ? 'Text to translate' : 'Prompt'));
  setText($('run-generate-btn'), isEmbeddingMode ? 'Embed' : (isTranslateMode ? 'Translate' : 'Generate'));
  const prompt = $('run-prompt');
  if (prompt) {
    prompt.placeholder = isEmbeddingMode
      ? 'Enter text to embed...'
      : (isTranslateMode
        ? 'Enter text to translate...'
        : 'Ask a question or provide a prompt...');
    if (isTranslateMode && isStarterExampleInput(prompt)) {
      applyStarterPrompt(prompt, TRANSLATE_STARTER_PROMPTS, { force: true });
    }
  }
  setHidden($('run-sampling-controls'), isEmbeddingMode);
  setHidden($('run-embedding-docs'), !isEmbeddingMode);
  setHidden($('translate-controls'), !isTranslateMode);
  if (isEmbeddingMode) {
    refreshEmbeddingDemoDocuments();
  }
  renderEmbeddingDocumentSet();
  updateRunAutoLabels();
  syncTranslateCompareUI();
}

async function setUiTask(task, modeHint = null) {
  const surface = normalizeSurface(state.surface, 'demo');
  const resolvedTask = resolveTaskForSurface(task, surface, modeHint || state.uiMode);
  const targetMode = resolveModeForTask(
    resolvedTask,
    surface,
    modeHint || state.lastTaskMode?.[resolvedTask] || state.uiMode || DEFAULT_TASK_MODE[resolvedTask]
  );
  await setUiMode(targetMode, { task: resolvedTask });
}

async function setUiMode(mode, options = {}) {
  const app = $('app');
  if (!app) return;
  const surface = normalizeSurface(state.surface, 'demo');
  const resolvedMode = resolveModeForSurface(mode, surface);
  const modeTask = getTaskForMode(resolvedMode, options?.task || state.uiTask || 'run');
  const resolvedTask = resolveTaskForSurface(
    modeTask,
    surface,
    resolvedMode
  );
  state.uiMode = resolvedMode;
  state.uiTask = resolvedTask;
  state.lastTaskMode[resolvedTask] = resolvedMode;
  if (PRIMARY_MODES.has(resolvedMode)) {
    state.lastPrimaryMode = resolvedMode;
  }
  app.dataset.mode = resolvedMode;
  app.dataset.task = resolvedTask;
  syncSurfaceUI(surface);
  updateNavState(resolvedMode, resolvedTask);
  applyModeVisibility(resolvedMode);
  syncRunModeUI(resolvedMode);
  syncDiagnosticsModeUI(resolvedMode);
  if (resolvedMode === 'models') {
    refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onTryModel: handleStorageTryModel,
      onUnloadActiveModel: unloadActivePipeline,
      onStorageInventoryRefreshed: renderQuickModelPanels,
      onModelsUpdated: refreshModelList,
    }).catch((error) => {
      log.warn('DopplerDemo', `Storage inspector refresh failed: ${error.message}`);
    });
  }
  try {
    await refreshModelList();
  } catch (error) {
    log.warn('DopplerDemo', `Model list refresh failed: ${error.message}`);
  }
  updatePerformancePanel();
  renderRunLog();
  syncDiagnosticsDefaultsForMode(resolvedMode).catch((error) => {
    updateDiagnosticsStatus(`Diagnostics config error: ${error.message}`, true);
  });
  if (resolvedMode === 'energy') {
    syncEnergyDemoSelection();
  }
  updateModelEmptyStates();
  syncDeepLinkFromUI();
}

function getModelAvailability() {
  const availability = state.modelAvailability;
  if (!availability || typeof availability !== 'object') {
    return { ...DEFAULT_MODEL_AVAILABILITY };
  }
  return {
    total: Number.isFinite(availability.total) ? availability.total : 0,
    run: Number.isFinite(availability.run) ? availability.run : 0,
    translate: Number.isFinite(availability.translate) ? availability.translate : 0,
    embedding: Number.isFinite(availability.embedding) ? availability.embedding : 0,
    diffusion: Number.isFinite(availability.diffusion) ? availability.diffusion : 0,
    energy: Number.isFinite(availability.energy) ? availability.energy : 0,
  };
}

function setEmptyNotice(scope, message) {
  const notice = $(`${scope}-empty-notice`);
  const kicker = $(`${scope}-empty-notice-kicker`);
  const text = $(`${scope}-empty-notice-text`);
  const detail = $(`${scope}-empty-notice-detail`);
  const normalized = message && typeof message === 'object'
    ? message
    : null;
  const title = typeof normalized?.title === 'string' ? normalized.title.trim() : '';
  const support = typeof normalized?.detail === 'string' ? normalized.detail.trim() : '';
  const kickerText = typeof normalized?.kicker === 'string' ? normalized.kicker.trim() : 'Setup required';
  setHidden(notice, title.length === 0);
  setText(kicker, title ? kickerText : '');
  setText(text, title);
  setText(detail, support);
}

function setEmptyNoticeAction(scope, quickModelEntry) {
  const button = $(`${scope}-empty-notice-btn`);
  if (!button) return;
  const busyModelId = state.quickModelActionModelId;
  const hasBusyImport = typeof busyModelId === 'string' && busyModelId.length > 0;
  const isDownloadLocked = state.downloadActive;

  if (quickModelEntry?.modelId) {
    const isBusy = busyModelId === quickModelEntry.modelId;
    button.dataset.noticeAction = 'download';
    button.dataset.quickModelId = quickModelEntry.modelId;
    if (isBusy) {
      const progress = resolveDownloadProgressForModel(quickModelEntry.modelId);
      const pct = progress?.percent;
      button.textContent = Number.isFinite(pct) ? `Importing ${Math.round(pct)}%` : 'Importing...';
    } else {
      const size = quickModelEntry.sizeBytes ? formatBytes(quickModelEntry.sizeBytes) : '';
      button.textContent = size
        ? `Import ${quickModelEntry.label} (${size})`
        : `Import ${quickModelEntry.label}`;
    }
    button.title = `Import ${quickModelEntry.label}`;
    button.disabled = isBusy || (hasBusyImport && !isBusy) || isDownloadLocked;
    return;
  }

  button.dataset.noticeAction = 'models';
  delete button.dataset.quickModelId;
  button.textContent = 'Browse models';
  button.title = 'Browse imported and available models';
  button.disabled = hasBusyImport || isDownloadLocked;
}

function createMissingModelNotice(title, detail, kicker = 'Setup required') {
  return {
    kicker,
    title,
    detail,
  };
}

function getMissingModelMessage(mode, availability, quickModelEntry) {
  if (mode === 'energy') {
    return null;
  }
  const total = Number.isFinite(availability?.total) ? availability.total : 0;
  const hasQuickSuggestion = !!(quickModelEntry && typeof quickModelEntry.modelId === 'string' && quickModelEntry.modelId.length > 0);
  if (total <= 0) {
    if (mode === 'embedding') {
      return hasQuickSuggestion
        ? createMissingModelNotice(
          'Import an embedding model to get started.',
          'Or open Models to choose a different one.'
        )
        : createMissingModelNotice(
          'Import an embedding model to get started.',
          'Open Models to choose one that supports similarity and retrieval.'
        );
    }
    if (mode === 'translate') {
      return hasQuickSuggestion
        ? createMissingModelNotice(
          'Import a translation model to get started.',
          'Or open Models to choose a different one.'
        )
        : createMissingModelNotice(
          'Import a translation model to get started.',
          'Open Models to choose one that supports translation.'
        );
    }
    if (mode === 'diffusion') {
      return hasQuickSuggestion
        ? createMissingModelNotice(
          'Import an image model to get started.',
          'Or open Models to choose a different one.'
        )
        : createMissingModelNotice(
          'Import an image model to get started.',
          'Open Models to choose one that supports diffusion.'
        );
    }
    return hasQuickSuggestion
      ? createMissingModelNotice(
        'Import a text model to get started.',
        'Or open Models to choose a different one.'
      )
      : createMissingModelNotice(
        'Import a text model to get started.',
        'Open Models to choose one that supports text generation.'
      );
  }
  const compatible = Number.isFinite(availability?.[mode]) ? availability[mode] : 0;
  if (compatible > 0) return null;
  if (mode === 'embedding') {
    return createMissingModelNotice(
      'This mode needs an embedding model.',
      'Your imported models do not support embedding yet. Import a compatible model to continue.'
    );
  }
  if (mode === 'translate') {
    return createMissingModelNotice(
      'This mode needs a translation model.',
      'Your imported models do not support translation yet. Import a compatible model to continue.'
    );
  }
  if (mode === 'diffusion') {
    return createMissingModelNotice(
      'This mode needs an image model.',
      'Your imported models do not support diffusion yet. Import a compatible model to continue.'
    );
  }
  return createMissingModelNotice(
    'This mode needs a text model.',
    'Your imported models do not support text generation yet. Import a compatible model to continue.'
  );
}

function setQuickModelStatus(message) {
  const statusEl = $('models-quick-models-status');
  if (!statusEl) return;
  setText(statusEl, message || '');
}

function createQuickModelBadge(text) {
  const badge = document.createElement('span');
  badge.className = 'quick-model-badge';
  badge.textContent = text;
  return badge;
}

function createQuickModelActionButton({ label, action, modelId, disabled, title = '' }) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn btn-small';
  button.textContent = label;
  button.dataset.quickAction = action;
  button.dataset.quickModelId = modelId;
  if (title) {
    button.title = title;
  }
  button.disabled = disabled;
  return button;
}

function formatQuickModelDownloadLabel(progress) {
  const percent = Number(progress?.percent);
  const percentLabel = Number.isFinite(percent) ? `${Math.round(clampPercent(percent))}%` : '';
  const totalShards = Number(progress?.totalShards);
  const completedShards = Number(progress?.completedShards);
  const shardLabel = Number.isFinite(totalShards) && totalShards > 0 && Number.isFinite(completedShards)
    ? `Shard ${Math.max(0, completedShards)}/${Math.max(0, totalShards)}`
    : '';
  if (percentLabel && shardLabel) {
    return `${percentLabel} · ${shardLabel}`;
  }
  if (percentLabel) return percentLabel;
  if (shardLabel) return shardLabel;
  return '';
}

function renderQuickModelList(listEl, catalogEntries) {
  if (!listEl) return;
  listEl.textContent = '';

  const busyId = state.quickModelActionModelId;
  const hasBusyAction = typeof busyId === 'string' && busyId.length > 0;
  const isDownloadActive = state.downloadActive;
  const storageEntries = Array.isArray(state.storageEntriesData) ? state.storageEntriesData : [];
  const storageByModelId = new Map(storageEntries.map((e) => [e.modelId, e]));
  const catalogIds = new Set(catalogEntries.map((e) => e.modelId));

  const storageDeleteCallbacks = {
    onUnloadActiveModel: unloadActivePipeline,
    onModelsUpdated: async () => {
      await refreshModelList();
      await refreshStorageInspector({
        onTryModel: handleStorageTryModel,
        onUnloadActiveModel: unloadActivePipeline,
        onStorageInventoryRefreshed: renderQuickModelPanels,
        onModelsUpdated: refreshModelList,
      });
    },
  };

  // OPFS-only orphans first (in storage but not in catalog)
  const orphans = storageEntries.filter((e) => !e.missingStorage && !catalogIds.has(e.modelId));
  // Catalog entries: OPFS models first, then not-in-OPFS
  const catalogSorted = [
    ...catalogEntries.filter((e) => storageByModelId.has(e.modelId)),
    ...catalogEntries.filter((e) => !storageByModelId.has(e.modelId)),
  ];

  function appendCard(card) {
    listEl.appendChild(card);
  }

  // Render orphan OPFS cards
  for (const storageEntry of orphans) {
    const card = document.createElement('article');
    card.className = 'quick-model-card';

    const row = document.createElement('div');
    row.className = 'quick-model-row';

    const main = document.createElement('div');
    main.className = 'quick-model-main';

    const title = document.createElement('div');
    title.className = 'quick-model-title';
    title.textContent = storageEntry.modelId;
    main.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'quick-model-meta';
    meta.appendChild(createQuickModelBadge(storageEntry.backend === 'indexeddb' ? 'idb' : (storageEntry.backend || 'opfs')));
    if (Number.isFinite(storageEntry.totalBytes) && storageEntry.totalBytes > 0) {
      meta.appendChild(createQuickModelBadge(formatQuickModelBytes(storageEntry.totalBytes)));
    }
    if (storageEntry.modelId === state.activeModelId) {
      meta.appendChild(createQuickModelBadge('active'));
    }
    main.appendChild(meta);

    row.appendChild(main);

    const actions = document.createElement('div');
    actions.className = 'quick-model-actions';
    if (isRunnableStorageEntry(storageEntry)) {
      const tryBtn = document.createElement('button');
      tryBtn.type = 'button';
      tryBtn.className = 'btn btn-small btn-primary';
      tryBtn.textContent = 'Try It';
      tryBtn.disabled = isDownloadActive;
      tryBtn.addEventListener('click', () => handleStorageTryModel(storageEntry.modelId));
      actions.appendChild(tryBtn);
    }
    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'btn btn-small';
    deleteBtn.textContent = 'Delete';
    deleteBtn.disabled = isDownloadActive;
    deleteBtn.addEventListener('click', () => deleteStorageModel(storageEntry, storageDeleteCallbacks));
    actions.appendChild(deleteBtn);
    row.appendChild(actions);

    card.appendChild(row);
    appendCard(card);
  }

  // Render catalog cards (OPFS first, then available)
  for (const entry of catalogSorted) {
    const isBusy = hasBusyAction && busyId === entry.modelId;
    const storageEntry = storageByModelId.get(entry.modelId);
    const isInOpfs = isRunnableStorageEntry(storageEntry);

    const card = document.createElement('article');
    card.className = entry.recommended ? 'quick-model-card is-recommended' : 'quick-model-card';

    const row = document.createElement('div');
    row.className = 'quick-model-row';

    const main = document.createElement('div');
    main.className = 'quick-model-main';

    const title = document.createElement('div');
    title.className = 'quick-model-title';
    title.textContent = entry.label;
    main.appendChild(title);

    const modelId = document.createElement('div');
    modelId.className = 'quick-model-id type-caption';
    modelId.textContent = entry.modelId;
    main.appendChild(modelId);

    const meta = document.createElement('div');
    meta.className = 'quick-model-meta';
    if (entry.recommended) {
      meta.appendChild(createQuickModelBadge('recommended'));
    }
    meta.appendChild(createQuickModelBadge(formatQuickModelModeBadge(entry.modes)));
    meta.appendChild(createQuickModelBadge(formatQuickModelBytes(entry.sizeBytes)));
    if (isInOpfs && storageEntry.modelId === state.activeModelId) {
      meta.appendChild(createQuickModelBadge('active'));
    }
    main.appendChild(meta);

    if (isBusy) {
      const dlProgress = resolveDownloadProgressForModel(entry.modelId);
      const pct = dlProgress?.percent ?? 0;
      const bar = document.createElement('div');
      bar.className = 'quick-model-progress';
      const fill = document.createElement('div');
      fill.className = 'quick-model-progress-fill';
      fill.style.width = `${pct}%`;
      bar.appendChild(fill);
      main.appendChild(bar);
    }

    row.appendChild(main);

    const actions = document.createElement('div');
    actions.className = 'quick-model-actions';
    if (isInOpfs) {
      const tryBtn = document.createElement('button');
      tryBtn.type = 'button';
      tryBtn.className = 'btn btn-small btn-primary';
      tryBtn.textContent = 'Try It';
      tryBtn.disabled = isDownloadActive;
      tryBtn.addEventListener('click', () => handleStorageTryModel(entry.modelId));
      actions.appendChild(tryBtn);
      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.className = 'btn btn-small';
      deleteBtn.textContent = 'Delete';
      deleteBtn.disabled = isDownloadActive;
      deleteBtn.addEventListener('click', () => deleteStorageModel(storageEntry, storageDeleteCallbacks));
      actions.appendChild(deleteBtn);
    } else {
      actions.appendChild(createQuickModelActionButton({
        label: isBusy
          ? formatQuickModelDownloadLabel(resolveDownloadProgressForModel(entry.modelId)) || 'Fetching...'
          : 'Fetch',
        action: 'download',
        modelId: entry.modelId,
        disabled: isBusy || hasBusyAction || isDownloadActive,
      }));
    }

    row.appendChild(actions);
    card.appendChild(row);
    appendCard(card);
  }

  if (orphans.length === 0 && catalogEntries.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'type-caption';
    empty.textContent = 'No models configured yet.';
    listEl.appendChild(empty);
  }
}

function renderQuickModelPanels() {
  const catalog = getQuickCatalogEntriesForSurface();
  const rawCatalog = getQuickCatalogEntries();

  if (state.quickModelActionModelId) {
    const modelId = state.quickModelActionModelId;
    const progress = resolveDownloadProgressForModel(modelId);
    const progressLabel = formatQuickModelDownloadLabel(progress);
    setQuickModelStatus(progressLabel ? `Fetching ${modelId}: ${progressLabel}` : `Fetching ${modelId}...`);
  } else if (state.quickModelCatalogLoading) {
    setQuickModelStatus('Loading quick models...');
  } else if (state.quickModelCatalogError) {
    const message = `Quick model catalog unavailable: ${state.quickModelCatalogError}`;
    setQuickModelStatus(message);
  } else if (rawCatalog.length > 0 && catalog.length === 0) {
    setQuickModelStatus('No quick models are tagged for currently supported demo modes.');
  } else {
    setQuickModelStatus(
      catalog.length > 0
        ? ''
        : 'No quick models configured in catalog.json yet.'
    );
  }

  renderQuickModelList($('models-list'), catalog);
}

async function loadQuickModelCatalog() {
  state.quickModelCatalogLoading = true;
  state.quickModelCatalogError = null;
  renderQuickModelPanels();
  try {
    let lastError = null;
    const loadedCatalogs = [];
    for (const catalogUrl of QUICK_MODEL_CATALOG_URLS) {
      try {
        const response = await fetch(catalogUrl, { cache: resolveRemoteCacheMode(catalogUrl) });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} (${catalogUrl})`);
        }
        const payload = await response.json();
        loadedCatalogs.push(parseQuickCatalogPayload(payload, catalogUrl));
      } catch (error) {
        lastError = error;
      }
    }
    if (loadedCatalogs.length === 0) {
      throw lastError || new Error('Quick model catalog fetch failed.');
    }
    state.quickModelCatalog = mergeQuickCatalogEntryLists(loadedCatalogs);
  } catch (error) {
    state.quickModelCatalog = [];
    state.quickModelCatalogError = error instanceof Error ? error.message : String(error);
  } finally {
    state.quickModelCatalogLoading = false;
    renderQuickModelPanels();
  }
}

async function applyImportedModelToCurrentMode(modelId) {
  if (!modelId) return;
  const mode = state.uiMode;
  if (mode === 'models') return;

  if (mode === 'diagnostics') {
    selectDiagnosticsModel(modelId);
    return;
  }

  if (!isModeModelSelectable(mode)) return;
  if (mode === 'translate' && !isTranslateCompatibleModelId(modelId)) return;
  const modelType = await getModelTypeForId(modelId);
  if (!isCompatibleModelType(modelType, mode)) return;

  selectDiagnosticsModel(modelId);
  state.modeModelId[mode] = modelId;
}

async function handleEmptyNoticeAction(scope) {
  const button = $(`${scope}-empty-notice-btn`);
  if (!button) return;
  if (state.downloadActive) return;
  const action = button.dataset.noticeAction || 'models';
  if (action !== 'download') {
    setUiMode('models');
    return;
  }
  const modelId = button.dataset.quickModelId || '';
  if (!modelId) {
    setUiMode('models');
    return;
  }
  await runQuickModelActionFromCore('download', modelId);
}

function handleDownloadProgressEvent(progress) {
  const modelId = typeof progress?.modelId === 'string' && progress.modelId.trim()
    ? progress.modelId.trim()
    : (typeof state.activeDownloadId === 'string' ? state.activeDownloadId : '');
  const percent = Number(progress?.percent);
  const downloadedBytes = Number(progress?.downloadedBytes);
  const totalBytes = Number(progress?.totalBytes);
  const totalShards = Number(progress?.totalShards);
  const completedShards = Number(progress?.completedShards);
  const currentShard = Number(progress?.currentShard);
  const speed = Number(progress?.speed);

  state.downloadProgress = {
    modelId,
    percent: Number.isFinite(percent) ? clampPercent(percent) : null,
    downloadedBytes: Number.isFinite(downloadedBytes) && downloadedBytes > 0 ? downloadedBytes : 0,
    totalBytes: Number.isFinite(totalBytes) && totalBytes > 0 ? totalBytes : 0,
    totalShards: Number.isFinite(totalShards) && totalShards > 0 ? totalShards : 0,
    completedShards: Number.isFinite(completedShards) && completedShards > 0 ? completedShards : 0,
    currentShard: Number.isFinite(currentShard) && currentShard > 0 ? currentShard : null,
    speed: Number.isFinite(speed) && speed > 0 ? speed : 0,
    status: typeof progress?.status === 'string' ? progress.status : '',
  };
  if (modelId) {
    state.activeDownloadId = modelId;
  }
  state.downloadActive = true;
  updateStatusIndicator();
  if (state.quickModelActionModelId && modelId && modelId === state.quickModelActionModelId) {
    updateModelEmptyStates();
  } else {
    renderQuickModelPanels();
  }
}

function handleDownloadStateChangeEvent(update) {
  if (!update || typeof update !== 'object') return;
  const modelId = typeof update.modelId === 'string' && update.modelId.trim() ? update.modelId.trim() : '';
  if (modelId) {
    state.activeDownloadId = modelId;
  }
  if (update.active === true) {
    state.downloadActive = true;
  } else if (update.active === false) {
    state.downloadActive = false;
    if (!modelId || state.downloadProgress?.modelId === modelId) {
      state.downloadProgress = null;
    }
  }
  updateStatusIndicator();
  if (state.quickModelActionModelId && (!modelId || modelId === state.quickModelActionModelId)) {
    updateModelEmptyStates();
  } else {
    renderQuickModelPanels();
  }
}

async function importRdrrFromBaseUrl(baseUrl, modelIdOverride = '') {
  const imported = await startDownloadFromBaseUrl(baseUrl, modelIdOverride);
  if (!imported) {
    throw new Error(`Could not import model ${modelIdOverride || baseUrl}.`);
  }
  await updateStorageInfo();
  await refreshModelList();
  if (state.uiMode === 'models') {
    await refreshStorageInspector({
      onSelectModel: selectDiagnosticsModel,
      onTryModel: handleStorageTryModel,
      onUnloadActiveModel: unloadActivePipeline,
      onStorageInventoryRefreshed: renderQuickModelPanels,
      onModelsUpdated: refreshModelList,
    });
  }
}


function updateModelEmptyStates() {
  if (state.modelAvailabilityLoading) {
    const emptyMessage = '';
    setEmptyNotice('run', emptyMessage);
    setEmptyNotice('diffusion', emptyMessage);
    setEmptyNotice('energy', emptyMessage);
    setEmptyNotice('diagnostics', emptyMessage);
    setEmptyNoticeAction('run', null);
    setEmptyNoticeAction('diffusion', null);
    setEmptyNoticeAction('energy', null);
    setEmptyNoticeAction('diagnostics', null);
    renderQuickModelPanels();
    return;
  }

  const availability = getModelAvailability();
  const runTargetMode = state.uiMode === 'embedding'
    ? 'embedding'
    : (state.uiMode === 'translate' ? 'translate' : 'run');
  const runQuickSuggestion = getPreferredQuickModelForMode(runTargetMode);
  const diffusionQuickSuggestion = getPreferredQuickModelForMode('diffusion');
  const energyQuickSuggestion = getPreferredQuickModelForMode('energy');
  const diagnosticsQuickSuggestion = getPreferredQuickModelForMode(getDiagnosticsRequiredQuickMode());
  const runMessage = getMissingModelMessage(runTargetMode, availability, runQuickSuggestion);
  const diffusionMessage = getMissingModelMessage('diffusion', availability, diffusionQuickSuggestion);
  const energyMessage = getMissingModelMessage('energy', availability, energyQuickSuggestion);
  const diagnosticsTargetMode = getDiagnosticsRequiredQuickMode();
  const diagnosticsMessage = (
    state.uiMode === 'diagnostics'
      ? (diagnosticsTargetMode ? getMissingModelMessage(diagnosticsTargetMode, availability, diagnosticsQuickSuggestion) : '')
      : ''
  );

  setEmptyNotice('run', runMessage);
  setEmptyNotice('diffusion', diffusionMessage);
  setEmptyNotice('energy', energyMessage);
  setEmptyNotice('diagnostics', diagnosticsMessage);
  setEmptyNoticeAction('run', runMessage ? runQuickSuggestion : null);
  setEmptyNoticeAction('diffusion', diffusionMessage ? diffusionQuickSuggestion : null);
  setEmptyNoticeAction('energy', energyMessage ? energyQuickSuggestion : null);
  setEmptyNoticeAction('diagnostics', diagnosticsMessage ? diagnosticsQuickSuggestion : null);
  renderQuickModelPanels();

  const diffusionRun = $('diffusion-run-btn');
  if (diffusionRun) {
    diffusionRun.disabled = state.diffusionGenerating || state.diffusionLoading || state.downloadActive || !!diffusionMessage;
  }
  const energyRun = $('energy-run-btn');
  if (energyRun) {
    energyRun.disabled = state.energyGenerating || state.energyLoading || state.downloadActive || !!energyMessage;
  }
  syncRunControls();
}

function updateConvertStatus(message, percent) {
  const status = $('convert-status');
  const progress = $('convert-progress');
  const label = $('convert-message');
  if (!status || !progress || !label) return;
  setHidden(status, false);
  setText(label, message || '');
  if (Number.isFinite(percent)) {
    progress.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  }
}

function resetConvertStatus() {
  const status = $('convert-status');
  const progress = $('convert-progress');
  const label = $('convert-message');
  if (!status || !progress || !label) return;
  setHidden(status, false);
  progress.style.width = '0%';
  setText(label, 'Ready');
}

function updateRunStatus(message) {
  const status = $('run-output-status');
  if (!status) return;
  setText(status, message || 'Idle');
}

function updateDiffusionStatus(message) {
  const status = $('diffusion-output-status');
  if (!status) return;
  setText(status, message || 'Idle');
}

const AUX_IMPORT_FILENAMES = [
  'config.json',
  'generation_config.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'added_tokens.json',
  'preprocessor_config.json',
  'vocab.txt',
  'merges.txt',
];

function getPickedFilePath(file) {
  if (!file) return '';
  if (typeof file.relativePath === 'string' && file.relativePath.length > 0) {
    return file.relativePath;
  }
  if (typeof file.webkitRelativePath === 'string' && file.webkitRelativePath.length > 0) {
    return file.webkitRelativePath;
  }
  if (typeof file.name === 'string') return file.name;
  return '';
}

function normalizePickedPath(path) {
  return String(path || '')
    .replace(/\\/g, '/')
    .replace(/^\.?\//, '')
    .trim();
}

function getPathBaseName(path) {
  const normalized = normalizePickedPath(path);
  if (!normalized) return '';
  const parts = normalized.split('/');
  return parts[parts.length - 1] || '';
}

function findPickedFileByPath(files, path) {
  const targetPath = normalizePickedPath(path);
  if (!targetPath) return null;

  const exact = files.find((file) => normalizePickedPath(getPickedFilePath(file)) === targetPath);
  if (exact) return exact;

  const targetBase = getPathBaseName(targetPath);
  if (!targetBase) return null;
  const baseMatches = files.filter((file) => getPathBaseName(getPickedFilePath(file)) === targetBase);
  if (baseMatches.length === 1) return baseMatches[0];
  return null;
}

function findPickedFileByBaseName(files, name) {
  const target = String(name || '').trim();
  if (!target) return null;
  const matches = files.filter((file) => getPathBaseName(getPickedFilePath(file)) === target);
  if (matches.length === 0) return null;
  return matches[0];
}

const MODEL_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$/;
const MODEL_ID_LABEL_MAX = 56;

function normalizeModelIdInput(value) {
  return String(value || '').trim();
}

function isValidModelId(value) {
  return MODEL_ID_PATTERN.test(normalizeModelIdInput(value));
}

function assertValidModelId(value, sourceLabel = 'modelId') {
  const normalized = normalizeModelIdInput(value);
  if (!normalized) {
    throw new Error(`${sourceLabel} is required.`);
  }
  if (!isValidModelId(normalized)) {
    throw new Error(
      `${sourceLabel} must match ${MODEL_ID_PATTERN.source} (2-128 chars, alnum, dot, underscore, hyphen).`
    );
  }
  return normalized;
}

function formatModelIdLabel(modelId, maxLength = MODEL_ID_LABEL_MAX) {
  const normalized = normalizeModelIdInput(modelId).replace(/\s+/g, ' ');
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, Math.max(0, maxLength - 3))}...`;
}

function getRegisteredModelId(entry) {
  const candidate = typeof entry?.modelId === 'string' && entry.modelId
    ? entry.modelId
    : (typeof entry?.id === 'string' ? entry.id : '');
  const normalized = normalizeModelIdInput(candidate);
  if (!normalized) return '';
  if (!isValidModelId(normalized)) {
    log.warn('DopplerDemo', `Skipping invalid modelId from registry: ${formatModelIdLabel(normalized, 96)}`);
    return '';
  }
  return normalized;
}

const TRANSLATE_MODEL_HINTS = Object.freeze([
  'translate',
  'translation',
  'nllb',
  'm2m',
  'marian',
  'madlad',
  'seamless',
  'opus',
  'mt',
]);
const DEMO_DEFAULT_TEXT_MODEL_ID = 'qwen-3-5-0-8b-q4k-ehaf16';

function getModelSelectionScore(mode, modelId) {
  const normalizedMode = normalizeDeepLinkMode(mode, 'run');
  const id = String(modelId || '').toLowerCase();
  let score = 0;

  if (normalizedMode === 'run') {
    if (id === DEMO_DEFAULT_TEXT_MODEL_ID) score += 200;
    else if (id.includes('qwen-3-5-0-8b')) score += 120;
    else if (id.includes('qwen-3-5')) score += 90;
    else if (id.includes('gemma-3')) score += 50;
    else if (id.includes('gemma')) score += 25;
  }

  if (normalizedMode === 'translate') {
    let hasTranslateHint = false;
    for (const hint of TRANSLATE_MODEL_HINTS) {
      if (id.includes(hint)) {
        score += 24;
        hasTranslateHint = true;
      }
    }
    if (id.includes('gemma-3')) score -= 60;
    else if (id.includes('gemma')) score -= 30;
    if (!hasTranslateHint) score -= 10;
  }

  if (id.includes('embedding') || id.includes('diffusion') || id.includes('energy')) {
    score -= 80;
  }

  return score;
}

async function deriveModelIdFromFiles(files, fallbackLabel) {
  const fallback = normalizeModelIdInput(fallbackLabel);
  if (isValidModelId(fallback)) return fallback;

  const configFile = files.find((file) => file.name === 'config.json');
  if (configFile) {
    try {
      const text = await configFile.text();
      const json = JSON.parse(text);
      const rawName = json?._name_or_path || json?.model_id || json?.modelId || json?.name;
      if (typeof rawName === 'string' && rawName.trim()) {
        const parts = rawName.trim().split('/');
        const name = parts[parts.length - 1];
        if (isValidModelId(name)) return name;
      }
    } catch {
      // Ignore config parsing errors here; converter will handle validation.
    }
  }

  const weightFile = files.find((file) => {
    const name = file.name.toLowerCase();
    return name.endsWith('.safetensors') || name.endsWith('.gguf');
  });
  if (weightFile) {
    const base = weightFile.name.replace(/\.(safetensors|gguf)$/i, '');
    if (isValidModelId(base)) return base;
  }

  return null;
}

async function filterModelsForMode(models, mode) {
  if (!isModeModelSelectable(mode)) return models;
  const filtered = [];
  for (const model of models) {
    const modelId = getRegisteredModelId(model);
    if (!modelId) continue;
    if (mode === 'translate' && !isTranslateCompatibleModelId(modelId)) continue;
    const modelType = await getModelTypeForId(modelId);
    if (isCompatibleModelType(modelType, mode)) {
      filtered.push(model);
    }
  }
  return filtered;
}

async function registerDownloadedModel(modelId) {
  const normalizedModelId = assertValidModelId(modelId, 'Downloaded modelId');
  await openModelStore(normalizedModelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) return null;
  const manifest = parseManifest(manifestText);
  const entry = {
    modelId: normalizedModelId,
    totalSize: manifest.totalSize,
    quantization: manifest.quantization,
    hashAlgorithm: manifest.hashAlgorithm,
    modelType: manifest.modelType,
  };
  if (manifest.modelId && manifest.modelId !== normalizedModelId) {
    entry.sourceModelId = manifest.modelId;
  }
  return registerModel(entry);
}

async function resolveCompatibleModelId(mode) {
  if (!isModeModelSelectable(mode)) return null;
  const normalizedMode = normalizeDeepLinkMode(mode, 'run');
  let models = [];
  try {
    models = await listRegisteredModels();
  } catch (error) {
    log.warn('DopplerDemo', `Model registry unavailable: ${error.message}`);
  }
  const modelIds = models
    .map((entry) => getRegisteredModelId(entry))
    .filter(Boolean);
  if (!modelIds.length) return null;

  const preferred = state.modeModelId?.[normalizedMode] || null;
  if (preferred && modelIds.includes(preferred)) {
    const preferredType = await getModelTypeForId(preferred);
    if (
      (normalizedMode !== 'translate' || isTranslateCompatibleModelId(preferred))
      && isCompatibleModelType(preferredType, normalizedMode)
    ) {
      return preferred;
    }
  }

  if (normalizedMode !== 'translate') {
    const pipelineId = state.activePipelineModelId;
    if (pipelineId && modelIds.includes(pipelineId)) {
      const pipelineType = normalizeModelType(state.activePipeline?.manifest?.modelType)
        || await getModelTypeForId(pipelineId);
      if (isCompatibleModelType(pipelineType, normalizedMode)) {
        return pipelineId;
      }
    }

    const current = state.activeModelId;
    if (current && modelIds.includes(current)) {
      const currentType = await getModelTypeForId(current);
      if (isCompatibleModelType(currentType, normalizedMode)) {
        return current;
      }
    }
  }

  let bestModelId = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const modelId of modelIds) {
    if (normalizedMode === 'translate' && !isTranslateCompatibleModelId(modelId)) continue;
    const modelType = await getModelTypeForId(modelId);
    if (!isCompatibleModelType(modelType, normalizedMode)) {
      continue;
    }
    const score = getModelSelectionScore(normalizedMode, modelId);
    if (bestModelId == null || score > bestScore) {
      bestModelId = modelId;
      bestScore = score;
    }
  }
  return bestModelId;
}

async function syncModelForMode(mode) {
  if (!isModeModelSelectable(mode)) return;
  const compatibleId = await resolveCompatibleModelId(mode);
  if (!compatibleId) {
    state.modeModelId[mode] = null;
    if (state.uiMode === mode) {
      state.activeModelId = null;
      const modelSelect = $('diagnostics-model');
      if (modelSelect) modelSelect.value = '';
    }
    return;
  }
  if (state.activeModelId !== compatibleId) {
    if (state.activePipeline && state.activePipelineModelId && state.activePipelineModelId !== compatibleId) {
      await unloadActivePipeline();
    }
    selectDiagnosticsModel(compatibleId);
  }
  state.modeModelId[mode] = compatibleId;
}

function getUiModeForModelType(modelType) {
  const normalizedType = normalizeModelType(modelType);
  if (normalizedType === 'embedding') return 'embedding';
  if (normalizedType === 'diffusion') return 'diffusion';
  if (normalizedType === 'energy') return 'energy';
  return 'run';
}

async function handleStorageTryModel(modelId) {
  if (!modelId) return;
  const modelType = await getModelTypeForId(modelId);
  const targetMode = getUiModeForModelType(modelType);
  await setUiMode(targetMode);
  selectDiagnosticsModel(modelId);
}

function updateSidebarLayout(models) {
  const panelGrid = $('panel-grid');
  if (!panelGrid) return;
  const hasModels = Array.isArray(models) && models.length > 0;
  panelGrid.dataset.layout = hasModels ? 'ready' : 'empty';
}

async function computeModelAvailability(models) {
  const availability = { ...DEFAULT_MODEL_AVAILABILITY };
  if (!Array.isArray(models)) return availability;
  const seenModelIds = new Set();
  for (const model of models) {
    const modelId = getRegisteredModelId(model);
    if (!modelId || seenModelIds.has(modelId)) continue;
    seenModelIds.add(modelId);
    availability.total += 1;

    let modelType = normalizeModelType(model?.modelType);
    if (!modelType) {
      modelType = normalizeModelType(await getModelTypeForId(modelId));
    }
    if (isCompatibleModelType(modelType, 'run')) availability.run += 1;
    if (isTranslateCompatibleModelId(modelId) && isCompatibleModelType(modelType, 'translate')) {
      availability.translate += 1;
    }
    if (isCompatibleModelType(modelType, 'embedding')) availability.embedding += 1;
    if (isCompatibleModelType(modelType, 'diffusion')) availability.diffusion += 1;
    if (isCompatibleModelType(modelType, 'energy')) availability.energy += 1;
  }
  return availability;
}

async function refreshModelList() {
  const modelSelect = $('diagnostics-model');
  if (!modelSelect) return;
  const refreshVersion = ++modelListRefreshVersion;
  state.modelAvailabilityLoading = true;
  state.modelAvailability = { ...DEFAULT_MODEL_AVAILABILITY };
  updateStatusIndicator();
  let models = [];
  try {
    models = await listRegisteredModels();
  } catch (error) {
    log.warn('DopplerDemo', `Model registry unavailable: ${error.message}`);
  }
  try {
    state.registeredModelIds = [...new Set(models
      .map((entry) => getRegisteredModelId(entry))
      .filter(Boolean))];
    const filteredModels = await filterModelsForMode(models, state.uiMode);
    if (refreshVersion !== modelListRefreshVersion) return;
    modelSelect.innerHTML = '';
    const modelIds = [];
    const seenModelIds = new Set();
    for (const model of filteredModels) {
      const modelId = getRegisteredModelId(model);
      if (!modelId || seenModelIds.has(modelId)) continue;
      const entryModelType = normalizeModelType(model?.modelType);
      if (entryModelType) {
        state.modelTypeCache[modelId] = entryModelType;
      }
      seenModelIds.add(modelId);
      modelIds.push(modelId);
    }
    if (!modelIds.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = `No ${getModeModelLabel(state.uiMode)} models`;
      modelSelect.appendChild(opt);
    } else {
      for (const modelId of modelIds) {
        const opt = document.createElement('option');
        opt.value = modelId;
        opt.textContent = formatModelIdLabel(modelId);
        opt.title = modelId;
        modelSelect.appendChild(opt);
      }
    }
    updateSidebarLayout(models);
    state.modelAvailability = await computeModelAvailability(models);
    await updateStorageInfo();
    await syncModelForMode(state.uiMode);
    renderTranslateCompareSelectors();
    syncTranslateCompareUI();
    if (state.uiMode === 'energy') {
      await preloadEnergyPipelineIfNeeded();
    }
    if (state.uiMode === 'models') {
      await refreshStorageInspector({
        onSelectModel: selectDiagnosticsModel,
        onTryModel: handleStorageTryModel,
        onUnloadActiveModel: unloadActivePipeline,
        onStorageInventoryRefreshed: renderQuickModelPanels,
        onModelsUpdated: refreshModelList,
      });
      renderQuickModelPanels();
    }
    updateDiagnosticsGuidance();
    updateModelEmptyStates();
  } finally {
    if (refreshVersion === modelListRefreshVersion) {
      state.modelAvailabilityLoading = false;
      if (!state.appInitializing) {
        updateStatusIndicator();
      }
    }
  }
}

async function refreshGpuInfo() {
  const deviceEl = $('gpu-device');
  const ramRow = $('gpu-ram-row');
  const ramEl = $('gpu-ram');
  const vramEl = $('gpu-vram');
  const featuresEl = $('gpu-features');
  const vramLabel = $('gpu-vram-label');
  const unifiedNote = $('gpu-unified-note');

  if (!isWebGPUAvailable()) {
    setText(deviceEl, 'WebGPU unavailable');
    setText(vramEl, '--');
    setText(featuresEl, 'none');
    setHidden(ramRow, true);
    setHidden(unifiedNote, true);
    return;
  }

  try {
    await initDevice();
    const caps = getKernelCapabilities();
    const adapter = caps.adapterInfo || {};
    const deviceLabel = [adapter.vendor, adapter.architecture || adapter.device, adapter.description]
      .filter(Boolean)
      .join(' ');
    setText(deviceEl, deviceLabel || 'Unknown GPU');

    if (Number.isFinite(navigator.deviceMemory)) {
      state.systemMemoryBytes = navigator.deviceMemory * 1024 * 1024 * 1024;
      setText(ramEl, `${navigator.deviceMemory} GB`);
      setHidden(ramRow, false);
    } else {
      setHidden(ramRow, true);
    }

    state.gpuMaxBytes = caps.maxBufferSize || 0;
    if (vramLabel) vramLabel.textContent = 'Buffer Limit';
    setText(vramEl, caps.maxBufferSize ? formatBytes(caps.maxBufferSize) : '--');

    const features = [
      caps.hasF16 && 'f16',
      caps.hasSubgroups && 'subgroups',
      caps.hasSubgroupsF16 && 'subgroups-f16',
      caps.hasTimestampQuery && 'timestamp',
    ].filter(Boolean);
    setText(featuresEl, features.length ? features.join(', ') : 'basic');

    let preferUnified = false;
    try {
      const platformConfig = getPlatformConfig();
      preferUnified = Boolean(platformConfig?.platform?.memoryHints?.preferUnifiedMemory);
    } catch {
      preferUnified = false;
    }
    setHidden(unifiedNote, !preferUnified);
  } catch (error) {
    setText(deviceEl, `GPU init failed`);
    setText(vramEl, '--');
    setText(featuresEl, 'none');
    setHidden(ramRow, true);
    setHidden(unifiedNote, true);
    log.warn('DopplerDemo', `GPU init failed: ${error.message}`);
  }
}

function getSelectedModelId() {
  if (state.activeModelId) return state.activeModelId;
  const modelSelect = $('diagnostics-model');
  const selected = modelSelect?.value || '';
  if (selected) {
    state.activeModelId = selected;
    return selected;
  }
  if (modelSelect?.options?.length) {
    const fallback = modelSelect.options[0].value;
    state.activeModelId = fallback || null;
    return fallback || null;
  }
  return null;
}

function pickRandomStarter(pool) {
  if (!Array.isArray(pool) || pool.length === 0) return '';
  const index = Math.floor(Math.random() * pool.length);
  return String(pool[index] || '').trim();
}

function isStarterExampleInput(inputEl) {
  return inputEl?.dataset?.starterExample === '1';
}

function setStarterExampleInput(inputEl, isExample) {
  if (!inputEl) return;
  inputEl.dataset.starterExample = isExample ? '1' : '0';
}

function pickRandomStarterDifferent(pool, currentValue) {
  if (!Array.isArray(pool) || pool.length === 0) return '';
  const current = String(currentValue || '').trim();
  if (pool.length === 1) return String(pool[0] || '').trim();
  for (let attempt = 0; attempt < pool.length * 2; attempt += 1) {
    const next = pickRandomStarter(pool);
    if (next && next !== current) {
      return next;
    }
  }
  return pickRandomStarter(pool);
}

function pickRandomSubset(pool, count) {
  if (!Array.isArray(pool) || pool.length === 0) return [];
  const targetCount = Math.max(1, Math.min(Number(count) || 1, pool.length));
  const copy = pool.slice();
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, targetCount);
}

function refreshEmbeddingDemoDocuments(options = {}) {
  const { force = false } = options;
  const current = Array.isArray(state.embeddingDemoDocuments) ? state.embeddingDemoDocuments : [];
  if (!force && current.length === EMBEDDING_DEMO_DOCUMENT_COUNT) {
    return current;
  }
  state.embeddingDemoDocuments = pickRandomSubset(
    EMBEDDING_DEMO_DOCUMENT_CATALOG,
    EMBEDDING_DEMO_DOCUMENT_COUNT
  );
  renderEmbeddingDocumentSet();
  return state.embeddingDemoDocuments;
}

function renderEmbeddingDocumentSet() {
  const wrap = $('run-embedding-docs');
  const list = $('run-embedding-docs-list');
  if (!wrap || !list) return;
  if (state.uiMode !== 'embedding') {
    setHidden(wrap, true);
    return;
  }
  setHidden(wrap, false);
  const docs = Array.isArray(state.embeddingDemoDocuments) ? state.embeddingDemoDocuments : [];
  if (docs.length === 0) {
    list.innerHTML = '<div class="type-caption">No documents configured.</div>';
    return;
  }
  const rows = docs
    .map((doc, index) => {
      const text = String(doc?.text || '').trim();
      const snippet = text.length > 140 ? `${text.slice(0, 140)}...` : text;
      return `<div class="embedding-doc-item"><div class="type-caption"><strong>${index + 1}. ${doc.title}</strong></div><div class="type-caption">${snippet}</div></div>`;
    })
    .join('');
  list.innerHTML = rows;
}

function applyStarterPrompt(inputEl, pool, options = {}) {
  if (!inputEl) return;
  const { force = false } = options;
  const current = String(inputEl.value || '').trim();
  if (!force && current.length > 0) return;
  const next = force ? pickRandomStarterDifferent(pool, current) : pickRandomStarter(pool);
  if (!next) return;
  inputEl.value = next;
  setStarterExampleInput(inputEl, true);
}

function prefillDemoTextInputs() {
  applyStarterPrompt($('run-prompt'), RUN_STARTER_PROMPTS);
  applyStarterPrompt($('diffusion-prompt'), DIFFUSION_STARTER_PROMPTS);
  applyStarterPrompt($('diffusion-negative'), DIFFUSION_NEGATIVE_STARTER_PROMPTS);
}

function bindStarterPromptInput(inputEl) {
  if (!inputEl) return;
  inputEl.addEventListener('focus', () => {
    if (isStarterExampleInput(inputEl)) {
      inputEl.select();
    }
  });
  inputEl.addEventListener('input', () => {
    setStarterExampleInput(inputEl, false);
  });
}

function syncRunControls() {
  const runPrompt = $('run-prompt');
  const runGenerate = $('run-generate-btn');
  const runStop = $('run-stop-btn');
  const runClear = $('run-clear-btn');
  const runResetKvToggle = $('run-reset-kv-toggle');
  const translateSourceSelect = $('translate-source-language');
  const translateTargetSelect = $('translate-target-language');
  const translateSwapBtn = $('translate-swap-btn');
  const temperatureInput = $('temperature-input');
  const topPInput = $('top-p-input');
  const topKInput = $('top-k-input');
  const maxTokensInput = $('max-tokens-input');
  const availability = getModelAvailability();
  const modeKey = state.uiMode === 'embedding'
    ? 'embedding'
    : (state.uiMode === 'translate' ? 'translate' : 'run');
  const hasCompatibleModel = Number.isFinite(availability[modeKey]) && availability[modeKey] > 0;
  const isRunning = state.runGenerating;
  const disabled = isRunning || state.runLoading || state.compareGenerating || state.compareLoading || state.downloadActive;
  if (runPrompt) runPrompt.disabled = disabled;
  if (runGenerate) runGenerate.disabled = disabled || !hasCompatibleModel;
  if (runClear) runClear.disabled = disabled;
  if (runResetKvToggle) runResetKvToggle.disabled = disabled;
  if (translateSourceSelect) translateSourceSelect.disabled = disabled;
  if (translateTargetSelect) translateTargetSelect.disabled = disabled;
  if (translateSwapBtn) translateSwapBtn.disabled = disabled;
  if (temperatureInput) temperatureInput.disabled = disabled;
  if (topPInput) topPInput.disabled = disabled;
  if (topKInput) topKInput.disabled = disabled;
  if (maxTokensInput) maxTokensInput.disabled = disabled;
  if (runGenerate) setHidden(runGenerate, isRunning);
  if (runStop) setHidden(runStop, !isRunning);
}

function setRunGenerating(isGenerating) {
  state.runGenerating = Boolean(isGenerating);
  if (!state.runGenerating) {
    state.runPrefilling = false;
  }
  syncRunControls();
  updateStatusIndicator();
}

function setRunLoading(isLoading) {
  state.runLoading = Boolean(isLoading);
  syncRunControls();
  updateStatusIndicator();
}

function setRunAutoLabel(inputId, labelId, value, options) {
  const input = $(inputId);
  const label = $(labelId);
  if (!label) return;
  const hasOverride = input?.value != null && input.value !== '';
  const prefix = hasOverride ? 'default' : 'auto';
  label.textContent = `${prefix}: ${formatAutoValue(value, options)}`;
}

function updateRunAutoLabels() {
  const runtime = getRuntimeConfig();
  const sampling = runtime?.inference?.sampling ?? {};
  const batching = runtime?.inference?.batching ?? {};
  const useTranslateDefaults = state.uiMode === 'translate';
  const defaultTemperature = useTranslateDefaults ? DEFAULT_TRANSLATE_TEMPERATURE : sampling.temperature;
  const defaultTopP = useTranslateDefaults ? DEFAULT_TRANSLATE_TOP_P : sampling.topP;
  const defaultTopK = useTranslateDefaults ? DEFAULT_TRANSLATE_TOP_K : sampling.topK;
  const defaultMaxTokens = useTranslateDefaults
    ? (state.compareEnabled ? TRANSLATE_COMPARE_DEFAULT_MAX_TOKENS : DEFAULT_TRANSLATE_MAX_TOKENS)
    : batching.maxTokens;
  setRunAutoLabel('temperature-input', 'temperature-auto', defaultTemperature);
  setRunAutoLabel('top-p-input', 'top-p-auto', defaultTopP);
  setRunAutoLabel('top-k-input', 'top-k-auto', defaultTopK, { integer: true });
  setRunAutoLabel('max-tokens-input', 'max-tokens-auto', defaultMaxTokens, { integer: true });
}

function formatCharCounter(value, maxLength) {
  const length = String(value || '').length;
  if (Number.isFinite(maxLength) && maxLength > 0) {
    return `${length}/${maxLength}`;
  }
  return String(length);
}

function updateDiffusionCharCounters() {
  const promptEl = $('diffusion-prompt');
  const negativeEl = $('diffusion-negative');
  const promptCountEl = $('diffusion-prompt-count');
  const negativeCountEl = $('diffusion-negative-count');

  if (promptCountEl) {
    const maxLength = promptEl?.maxLength ?? null;
    promptCountEl.textContent = formatCharCounter(promptEl?.value, maxLength);
  }
  if (negativeCountEl) {
    const maxLength = negativeEl?.maxLength ?? null;
    negativeCountEl.textContent = formatCharCounter(negativeEl?.value, maxLength);
  }
}
// --- Exports ---
export {
  resolveModeForTask,
  parseAllowedTasks,
  syncSurfaceUI,
  readDeepLinkValue,
  decodeDeepLinkText,
  readDeepLinkStateFromLocation,
  applyDeepLinkStateToUI,
  buildDeepLinkHash,
  syncDeepLinkFromUI,
  buildTranslateDeepLinkUrl,
  setTranslateCompareEnabled,
  getRunStarterPromptPool,
  readGlobalString,
  normalizeUrlPathname,
  isHuggingFaceHost,
  buildHfResolveUrl,
  extractHfResolveRevisionFromUrl,
  isImmutableHfResolveUrl,
  resolveRemoteCacheMode,
  buildQuickCatalogCandidateUrls,
  normalizeQuickLookupToken,
  normalizeQuickCatalogAliases,
  normalizeQuickCatalogHfSpec,
  isQuickCatalogHfSourceUrl,
  hasQuickCatalogExplicitBaseUrl,
  extractHfRepoIdFromInput,
  collectQuickCatalogLookupTokens,
  findQuickCatalogEntryForRegistryInput,
  resolveDirectRdrrBaseUrlFromInput,
  normalizeQuickModeToken,
  normalizeQuickModes,
  resolveQuickModelBaseUrl,
  isQuickModelLocalUrl,
  isQuickModelHfResolveUrl,
  isQuickModelAllowedUrl,
  normalizeQuickCatalogEntry,
  parseQuickCatalogPayload,
  getQuickCatalogEntries,
  getQuickCatalogEntriesForSurface,
  formatQuickModelBytes,
  setDistillStatus,
  setDistillOutput,
  getDistillWorkloads,
  populateDistillWorkloadSelect,
  findDistillWorkloadById,
  exportDistillReplay,
  resolveDownloadProgressForModel,
  findQuickModelEntry,
  formatQuickModelModeBadge,
  getComparableQuickModelSize,
  getSmallestQuickModelForMode,
  getPreferredQuickModelForMode,
  getDiagnosticsRequiredQuickMode,
  updateNavState,
  cloneRuntimeConfig,
  applyModeVisibility,
  ensurePrimaryModeControlStack,
  syncRunModeUI,
  getModelAvailability,
  setEmptyNotice,
  setEmptyNoticeAction,
  createMissingModelNotice,
  getMissingModelMessage,
  setQuickModelStatus,
  createQuickModelBadge,
  createQuickModelActionButton,
  renderQuickModelList,
  renderQuickModelPanels,
  handleDownloadProgressEvent,
  handleDownloadStateChangeEvent,
  updateModelEmptyStates,
  updateConvertStatus,
  resetConvertStatus,
  updateRunStatus,
  updateDiffusionStatus,
  getPickedFilePath,
  normalizePickedPath,
  getPathBaseName,
  findPickedFileByPath,
  findPickedFileByBaseName,
  normalizeModelIdInput,
  isValidModelId,
  assertValidModelId,
  formatModelIdLabel,
  getRegisteredModelId,
  getModelSelectionScore,
  getUiModeForModelType,
  updateSidebarLayout,
  getSelectedModelId,
  pickRandomStarter,
  isStarterExampleInput,
  setStarterExampleInput,
  pickRandomStarterDifferent,
  pickRandomSubset,
  refreshEmbeddingDemoDocuments,
  renderEmbeddingDocumentSet,
  applyStarterPrompt,
  prefillDemoTextInputs,
  bindStarterPromptInput,
  syncRunControls,
  setRunGenerating,
  setRunLoading,
  setRunAutoLabel,
  updateRunAutoLabels,
  formatCharCounter,
  updateDiffusionCharCounters,
};
