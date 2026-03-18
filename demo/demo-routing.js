function normalizeDeepLinkMode(mode, defaultMode = null) {
  const normalized = resolveText(mode, '').toLowerCase();
  if (normalized === 'text') return 'run';
  if (normalized === 'translation') return 'translate';
  if (normalized === 'embed') return 'embedding';
  if (normalized === 'image') return 'diffusion';
  if (DEEP_LINK_MODES.has(normalized)) return normalized;
  return defaultMode;
}

function normalizeTask(value, fallback = null) {
  const normalized = resolveText(value, '').toLowerCase();
  if (TASK_SET.has(normalized)) return normalized;
  return fallback;
}

function getTaskModes(task) {
  const normalizedTask = normalizeTask(task, 'run');
  return TASK_MODE_ALLOWLIST[normalizedTask] || TASK_MODE_ALLOWLIST.run;
}

function getTaskForMode(mode, fallback = null) {
  const normalizedMode = normalizeDeepLinkMode(mode, null);
  if (normalizedMode && MODE_TASK_MAP[normalizedMode]) {
    return MODE_TASK_MAP[normalizedMode];
  }
  return normalizeTask(fallback, null);
}

function normalizeSurface(value, fallback = 'demo') {
  const normalized = resolveText(value, fallback).toLowerCase();
  if (SURFACE_SET.has(normalized)) return normalized;
  return fallback;
}

function getAllowedModesForSurface(surface) {
  return SURFACE_MODE_ALLOWLIST[normalizeSurface(surface, 'demo')] || SURFACE_MODE_ALLOWLIST.demo;
}

function getAllowedModesForTask(task, surface) {
  const modesForTask = getTaskModes(task);
  return modesForTask.filter((mode) => isModeAllowedForSurface(mode, surface));
}

function isModeAllowedForSurface(mode, surface) {
  return getAllowedModesForSurface(surface).has(mode);
}

function isTaskAllowedForSurface(task, surface) {
  return getAllowedModesForTask(task, surface).length > 0;
}

function resolveModeForSurface(mode, surface) {
  const normalizedMode = normalizeDeepLinkMode(mode, 'run');
  if (isModeAllowedForSurface(normalizedMode, surface)) {
    return normalizedMode;
  }
  const fallbackPrimary = normalizeDeepLinkMode(state.lastPrimaryMode, 'run');
  if (isModeAllowedForSurface(fallbackPrimary, surface)) {
    return fallbackPrimary;
  }
  return 'run';
}

function resolveTaskForSurface(task, surface, modeHint = null) {
  const normalizedSurface = normalizeSurface(surface, 'demo');
  const requestedTask = normalizeTask(task, null);
  if (requestedTask && isTaskAllowedForSurface(requestedTask, normalizedSurface)) {
    return requestedTask;
  }
  const modeTask = getTaskForMode(modeHint, null);
  if (modeTask && isTaskAllowedForSurface(modeTask, normalizedSurface)) {
    return modeTask;
  }
  const rememberedTask = normalizeTask(state.uiTask, null);
  if (rememberedTask && isTaskAllowedForSurface(rememberedTask, normalizedSurface)) {
    return rememberedTask;
  }
  for (const fallbackTask of ['run', 'evaluate']) {
    if (isTaskAllowedForSurface(fallbackTask, normalizedSurface)) {
      return fallbackTask;
    }
  }
  return 'run';
}
// --- Exports ---
export {
  normalizeDeepLinkMode,
  normalizeTask,
  getTaskModes,
  getTaskForMode,
  normalizeSurface,
  getAllowedModesForSurface,
  getAllowedModesForTask,
  isModeAllowedForSurface,
  isTaskAllowedForSurface,
  resolveModeForSurface,
  resolveTaskForSurface,
};
