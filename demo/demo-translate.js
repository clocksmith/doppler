function getCompareLaneIds() {
  return ['left', 'right'];
}

function getCompareLane(laneId) {
  if (!laneId || !state.compareLanes || typeof state.compareLanes !== 'object') {
    return null;
  }
  return state.compareLanes[laneId] || null;
}

function createCompareRuntimeLane(label) {
  return {
    engine: 'doppler',
    modelId: null,
    tjsModelId: '',
    tjsDtype: TRANSLATE_COMPARE_DEFAULT_TJS_DTYPE,
    label,
    status: 'Idle',
    statusTone: 'info',
    output: '',
    metrics: null,
    error: null,
    pipeline: null,
    pipelineModelId: null,
    tjsGenerator: null,
    tjsGeneratorKey: null,
  };
}

function ensureTranslateCompareRuntimeState() {
  if (!state.compareLanes || typeof state.compareLanes !== 'object') {
    state.compareLanes = {
      left: createCompareRuntimeLane('Left Lane'),
      right: createCompareRuntimeLane('Right Lane'),
    };
  }
  if (!state.compareLanes.left) {
    state.compareLanes.left = createCompareRuntimeLane('Left Lane');
  }
  if (!state.compareLanes.right) {
    state.compareLanes.right = createCompareRuntimeLane('Right Lane');
  }
  if (!Array.isArray(state.compareHistory)) {
    state.compareHistory = [];
  }
  if (!resolveText(state.compareHistoryFilter, '')) {
    state.compareHistoryFilter = 'all';
  }
  if (!Array.isArray(state.compareProfiles)) {
    state.compareProfiles = [];
  }
  if (!state.compareEvidence || typeof state.compareEvidence !== 'object') {
    state.compareEvidence = cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK);
  }
  if (state.activeCompareSmokeSampleId != null && !resolveText(state.activeCompareSmokeSampleId, '')) {
    state.activeCompareSmokeSampleId = null;
  }
  if (state.lastCompareArtifact != null && typeof state.lastCompareArtifact !== 'object') {
    state.lastCompareArtifact = null;
  }
}

function setCompareLaneStatus(laneId, message, tone = 'info') {
  const lane = getCompareLane(laneId);
  if (!lane) return;
  lane.status = resolveText(message, 'Idle');
  lane.statusTone = tone || 'info';
}

function clearCompareLaneResult(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) return;
  lane.output = '';
  lane.metrics = null;
  lane.error = null;
  setCompareLaneStatus(laneId, 'Idle', 'info');
}

function isTranslateCompareEnabled() {
  return state.uiMode === 'translate' && state.compareEnabled === true;
}

function isTranslateCompatibleModelId(modelId) {
  const entry = findQuickModelEntry(normalizeModelIdInput(modelId));
  return Array.isArray(entry?.modes) && entry.modes.includes('translate');
}

function getTranslateCompareStudentModelId() {
  const explicit = readGlobalString('__DOPPLER_TRANSLATE_COMPARE_STUDENT_MODEL_ID');
  if (explicit && isTranslateCompatibleModelId(explicit)) {
    return explicit;
  }
  if (explicit) {
    log.warn('DopplerDemo', `__DOPPLER_TRANSLATE_COMPARE_STUDENT_MODEL_ID "${explicit}" is not a translate-compatible catalog entry`);
  }
  const evidenceModelId = resolveText(state.compareEvidence?.student?.modelId, '');
  if (evidenceModelId && isTranslateCompatibleModelId(evidenceModelId)) {
    return evidenceModelId;
  }
  const activeTranslateModelId = resolveText(state.modeModelId?.translate, '');
  if (
    activeTranslateModelId
    && isTranslateCompatibleModelId(activeTranslateModelId)
    && activeTranslateModelId !== TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID
  ) {
    return activeTranslateModelId;
  }
  for (const modelId of state.registeredModelIds || []) {
    if (!modelId || modelId === TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID) continue;
    const normalizedType = normalizeModelType(state.modelTypeCache?.[modelId]);
    if (
      isTranslateCompatibleModelId(modelId)
      && isCompatibleModelType(normalizedType, 'translate')
    ) {
      return modelId;
    }
  }
  return null;
}

function getMappedCompareBaselineProfile() {
  const profiles = Array.isArray(state.compareProfiles) ? state.compareProfiles : [];
  return profiles.find((entry) => (
    entry?.dopplerModelId === TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID
    && resolveText(entry?.defaultTjsModelId, '')
  )) || null;
}

function findCompareProfileByDopplerModelId(modelId) {
  const normalizedModelId = resolveText(modelId, '');
  if (!normalizedModelId) return null;
  return (state.compareProfiles || []).find((entry) => entry?.dopplerModelId === normalizedModelId) || null;
}

function resolveTranslateCompareRole(role) {
  const mappedBaseline = getMappedCompareBaselineProfile();
  const studentModelId = getTranslateCompareStudentModelId();
  if (role === 'baseline') {
    return {
      modelId: TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID,
      tjsModelId: resolveText(findCompareProfileByDopplerModelId(TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID)?.defaultTjsModelId, ''),
    };
  }
  if (role === 'student') {
    return {
      modelId: studentModelId,
      tjsModelId: resolveText(findCompareProfileByDopplerModelId(studentModelId)?.defaultTjsModelId, ''),
    };
  }
  if (role === 'mapped-baseline') {
    return {
      modelId: mappedBaseline?.dopplerModelId || TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID,
      tjsModelId: resolveText(mappedBaseline?.defaultTjsModelId, ''),
    };
  }
  if (role === 'student-mapped') {
    return {
      modelId: studentModelId,
      tjsModelId: resolveText(findCompareProfileByDopplerModelId(studentModelId)?.defaultTjsModelId, ''),
    };
  }
  return {
    modelId: null,
    tjsModelId: '',
  };
}

function serializeTranslateCompareArtifactPayload(artifact) {
  if (!artifact || typeof artifact !== 'object') {
    return null;
  }
  const lanes = {};
  for (const laneId of getCompareLaneIds()) {
    const lane = artifact?.lanes?.[laneId] || {};
    lanes[laneId] = {
      engine: resolveText(lane.engine, ''),
      modelId: resolveText(lane.modelId, ''),
      modelLabel: resolveText(lane.modelLabel, laneId),
      tjsModelId: resolveText(lane.tjsModelId, ''),
      roleLabel: resolveText(lane.roleLabel, 'custom'),
      status: resolveText(lane.status, ''),
      output: String(lane.output || ''),
      metrics: lane.metrics || null,
      error: lane.error || null,
    };
  }
  return {
    schemaVersion: Number.isFinite(Number(artifact.schemaVersion))
      ? Number(artifact.schemaVersion)
      : TRANSLATE_COMPARE_CONFIG_VERSION,
    kind: resolveText(artifact.kind, TRANSLATE_COMPARE_ARTIFACT_KIND),
    artifactId: resolveText(artifact.artifactId, ''),
    createdAt: resolveText(artifact.createdAt, ''),
    shareUrl: resolveText(artifact.shareUrl, '') || null,
    request: artifact.request && typeof artifact.request === 'object'
      ? {
        prompt: String(artifact.request.prompt || ''),
        sourceCode: resolveText(artifact.request.sourceCode, DEFAULT_TRANSLATE_SOURCE),
        sourceName: resolveText(artifact.request.sourceName, DEFAULT_TRANSLATE_SOURCE),
        targetCode: resolveText(artifact.request.targetCode, DEFAULT_TRANSLATE_TARGET),
        targetName: resolveText(artifact.request.targetName, DEFAULT_TRANSLATE_TARGET),
        options: artifact.request.options && typeof artifact.request.options === 'object'
          ? artifact.request.options
          : {},
        presetId: resolveText(artifact.request.presetId, 'custom'),
      }
      : {
        prompt: '',
        sourceCode: DEFAULT_TRANSLATE_SOURCE,
        sourceName: DEFAULT_TRANSLATE_SOURCE,
        targetCode: DEFAULT_TRANSLATE_TARGET,
        targetName: DEFAULT_TRANSLATE_TARGET,
        options: {},
        presetId: 'custom',
      },
    environment: artifact.environment && typeof artifact.environment === 'object'
      ? artifact.environment
      : {},
    evidence: artifact.evidence && typeof artifact.evidence === 'object'
      ? {
        updatedAt: resolveText(artifact.evidence.updatedAt, '') || null,
        summary: resolveText(artifact.evidence.summary, ''),
        receipts: Array.isArray(artifact.evidence.receipts) ? artifact.evidence.receipts : [],
      }
      : {
        updatedAt: null,
        summary: '',
        receipts: [],
      },
    summary: artifact.summary && typeof artifact.summary === 'object' ? artifact.summary : {},
    lanes,
  };
}

function serializeTranslateCompareHistoryEntry(entry) {
  const lanes = {};
  for (const laneId of getCompareLaneIds()) {
    const lane = entry?.lanes?.[laneId] || {};
    lanes[laneId] = {
      engine: resolveText(lane.engine, ''),
      modelId: resolveText(lane.modelId, ''),
      tjsModelId: resolveText(lane.tjsModelId, ''),
      label: resolveText(lane.label, laneId),
      status: resolveText(lane.status, ''),
      output: String(lane.output || ''),
      metrics: lane.metrics || null,
      error: lane.error || null,
    };
  }
  return {
    id: resolveText(entry?.id, ''),
    createdAt: resolveText(entry?.createdAt, ''),
    sourceCode: resolveText(entry?.sourceCode, DEFAULT_TRANSLATE_SOURCE),
    targetCode: resolveText(entry?.targetCode, DEFAULT_TRANSLATE_TARGET),
    prompt: String(entry?.prompt || ''),
    presetId: resolveText(entry?.presetId, 'custom'),
    artifact: serializeTranslateCompareArtifactPayload(entry?.artifact),
    lanes,
  };
}

function persistTranslateCompareHistory() {
  if (typeof localStorage === 'undefined') return;
  try {
    const payload = {
      schemaVersion: TRANSLATE_COMPARE_CONFIG_VERSION,
      history: (state.compareHistory || []).slice(0, TRANSLATE_COMPARE_MAX_HISTORY).map(serializeTranslateCompareHistoryEntry),
    };
    localStorage.setItem(TRANSLATE_COMPARE_HISTORY_STORAGE_KEY, JSON.stringify(payload));
  } catch (error) {
    log.warn('DopplerDemo', `Compare history persistence skipped: ${error.message}`);
  }
}

function hydrateTranslateCompareHistory() {
  if (typeof localStorage === 'undefined') return;
  try {
    const raw = localStorage.getItem(TRANSLATE_COMPARE_HISTORY_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    const rows = Array.isArray(parsed?.history) ? parsed.history : [];
    state.compareHistory = rows.map(serializeTranslateCompareHistoryEntry).slice(0, TRANSLATE_COMPARE_MAX_HISTORY);
    state.lastCompareArtifact = state.compareHistory[0]?.artifact || null;
  } catch (error) {
    log.warn('DopplerDemo', `Compare history restore skipped: ${error.message}`);
  }
}

async function loadTranslateCompareProfiles() {
  try {
    const response = await fetch(TRANSLATE_COMPARE_ENGINES_CONFIG_URL, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const rows = Array.isArray(payload?.modelProfiles) ? payload.modelProfiles : [];
    state.compareProfiles = rows.map((entry) => ({
      dopplerModelId: resolveText(entry?.dopplerModelId, ''),
      defaultTjsModelId: resolveText(entry?.defaultTjsModelId, ''),
      defaultKernelPath: resolveText(entry?.defaultKernelPath, ''),
      modelBaseDir: resolveText(entry?.modelBaseDir, ''),
      defaultDopplerSurface: resolveText(entry?.defaultDopplerSurface, 'auto'),
    }));
  } catch (error) {
    state.compareProfiles = [];
    log.warn('DopplerDemo', `Compare profiles unavailable: ${error.message}`);
  }
}

function normalizeTranslateCompareEvidence(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK);
  }
  return {
    ...cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK),
    ...payload,
    teacher: {
      ...cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK.teacher),
      ...(payload.teacher && typeof payload.teacher === 'object' ? payload.teacher : {}),
    },
    student: {
      ...cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK.student),
      ...(payload.student && typeof payload.student === 'object' ? payload.student : {}),
    },
    receipts: Array.isArray(payload.receipts) ? payload.receipts : cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK.receipts),
  };
}

async function loadTranslateCompareEvidence() {
  const embedded = globalThis?.__DOPPLER_TRANSLATE_COMPARE_EVIDENCE__;
  if (embedded && typeof embedded === 'object' && !Array.isArray(embedded)) {
    state.compareEvidence = normalizeTranslateCompareEvidence(embedded);
    return;
  }

  const evidenceUrl = readGlobalString('__DOPPLER_TRANSLATE_COMPARE_EVIDENCE_URL');
  if (!evidenceUrl) {
    state.compareEvidence = cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK);
    return;
  }

  try {
    const response = await fetch(evidenceUrl, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    state.compareEvidence = normalizeTranslateCompareEvidence(payload);
  } catch (error) {
    state.compareEvidence = cloneRuntimeConfig(TRANSLATE_COMPARE_EVIDENCE_FALLBACK);
    log.warn('DopplerDemo', `Compare evidence unavailable: ${error.message}`);
  }
}

function buildTranslateInstructionPrompt(prompt, sourceCode, targetCode) {
  return `Translate the following from ${sourceCode} to ${targetCode}. Output only the translation, no explanation.\n\n${prompt}`;
}

function buildTransformersTranslatePrompt(prompt, sourceCode, targetCode, modelHint = '') {
  const normalizedHint = resolveText(modelHint, '').toLowerCase();
  if (normalizedHint.includes('translategemma')) {
    return formatChatMessages(
      createTranslateTextRequest(prompt, sourceCode, targetCode).messages,
      'translategemma'
    );
  }
  return buildTranslateInstructionPrompt(prompt, sourceCode, targetCode);
}

function getTranslateCompatibleRegisteredModelIds() {
  const ids = [];
  for (const modelId of state.registeredModelIds || []) {
    const normalizedType = normalizeModelType(state.modelTypeCache?.[modelId]);
    if (
      isTranslateCompatibleModelId(modelId)
      && isCompatibleModelType(normalizedType, 'translate')
    ) {
      ids.push(modelId);
    }
  }
  return ids;
}

function getTranslateComparePreset(presetId) {
  const normalizedId = resolveText(presetId, 'proof');
  return TRANSLATE_COMPARE_PRESETS.find((entry) => entry.id === normalizedId)
    || TRANSLATE_COMPARE_PRESETS[0];
}

const TRANSLATE_COMPARE_TJS_BASELINE_NOTE = 'Baseline parity is currently unsupported in public TJS ONNX exports.';

function getTranslateComparePresetNote(presetId) {
  const preset = getTranslateComparePreset(presetId);
  if (preset.id === 'engine-parity' && !getMappedCompareBaselineProfile()) {
    return `${preset.description} The UI will fail closed until a baseline mapping is configured.`;
  }
  const usesTjsMappedBaseline = ['left', 'right'].some((laneId) => {
    const lane = preset?.lanes?.[laneId];
    return lane?.engine === 'transformersjs' && lane?.role === 'mapped-baseline';
  });
  if (usesTjsMappedBaseline) {
    return `${preset.description} ${TRANSLATE_COMPARE_TJS_BASELINE_NOTE}`;
  }
  const usesStudentRole = ['left', 'right'].some((laneId) => {
    const role = resolveText(preset?.lanes?.[laneId]?.role, '');
    return role === 'student' || role === 'student-mapped';
  });
  if (usesStudentRole && !resolveText(getTranslateCompareStudentModelId(), '')) {
    return `${preset.description} Waiting for the student artifact and any TJS mapping.`;
  }
  return preset.description;
}

function getTranslateCompareModelLabel(modelId) {
  const normalizedModelId = resolveText(modelId, '');
  if (!normalizedModelId) return 'Select model';
  const quickEntry = (state.quickModelCatalog || []).find((entry) => entry?.modelId === normalizedModelId);
  return resolveText(quickEntry?.label, formatModelIdLabel(normalizedModelId));
}

function resolveCompareModelSizeBytes(modelId) {
  const normalizedModelId = resolveText(modelId, '');
  if (!normalizedModelId) return null;
  const quickEntry = (state.quickModelCatalog || []).find((entry) => entry?.modelId === normalizedModelId);
  if (Number.isFinite(quickEntry?.sizeBytes)) {
    return quickEntry.sizeBytes;
  }
  const evidenceTeacherModelId = resolveText(state.compareEvidence?.teacher?.modelId, '');
  if (normalizedModelId === evidenceTeacherModelId && Number.isFinite(state.compareEvidence?.teacher?.sizeBytes)) {
    return state.compareEvidence.teacher.sizeBytes;
  }
  const evidenceStudentModelId = resolveText(state.compareEvidence?.student?.modelId, '');
  if (normalizedModelId === evidenceStudentModelId && Number.isFinite(state.compareEvidence?.student?.sizeBytes)) {
    return state.compareEvidence.student.sizeBytes;
  }
  return null;
}

function resolveTransformersModelIdForLane(lane) {
  if (!lane) return '';
  const explicit = resolveText(lane.tjsModelId, '');
  if (explicit) return explicit;
  return resolveText(findCompareProfileByDopplerModelId(lane.modelId)?.defaultTjsModelId, '');
}

function formatCompareMetricMs(value) {
  return Number.isFinite(value) ? `${Math.round(value)} ms` : '--';
}

function formatCompareMetricRate(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '--';
}

function formatCompareMetricBytes(value) {
  return Number.isFinite(value) ? formatBytes(value) : '--';
}

function formatCompareTimestamp(isoString) {
  if (!isoString) return '--';
  const timestamp = Date.parse(isoString);
  if (!Number.isFinite(timestamp)) return '--';
  try {
    return new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      month: 'short',
      day: 'numeric',
    }).format(new Date(timestamp));
  } catch {
    return new Date(timestamp).toISOString();
  }
}

let compareAdapterLabelPromise = null;

async function resolveWebGpuDeviceLabel(prefix = 'WebGPU') {
  if (compareAdapterLabelPromise) {
    const resolved = await compareAdapterLabelPromise;
    return resolved || prefix;
  }
  compareAdapterLabelPromise = (async () => {
    try {
      const adapter = await globalThis?.navigator?.gpu?.requestAdapter?.();
      if (!adapter) {
        return prefix;
      }
      const info = adapter.info || (await adapter.requestAdapterInfo?.()) || {};
      const deviceName = resolveText(info.description || info.device, '');
      const vendorName = resolveText(info.vendor, '');
      const suffix = [vendorName, deviceName].filter(Boolean).join(' ');
      return suffix ? `${prefix} · ${suffix}` : prefix;
    } catch {
      return prefix;
    }
  })();
  const resolved = await compareAdapterLabelPromise;
  return resolved || prefix;
}

function getCompareLaneSelectId(laneId) {
  return laneId === 'left' ? 'translate-left-model' : 'translate-right-model';
}

function getCompareLaneEngineSelectId(laneId) {
  return laneId === 'left' ? 'translate-left-engine' : 'translate-right-engine';
}

function populateCompareLaneModelSelect(laneId) {
  const lane = getCompareLane(laneId);
  const select = $(getCompareLaneSelectId(laneId));
  if (!(select instanceof HTMLSelectElement) || !lane) return;
  const previousValue = resolveText(lane.modelId, '');
  select.innerHTML = '';
  const options = [];

  if (lane.engine === 'transformersjs') {
    for (const profile of state.compareProfiles || []) {
      const dopplerModelId = resolveText(profile?.dopplerModelId, '');
      const tjsModelId = resolveText(profile?.defaultTjsModelId, '');
      if (!dopplerModelId || !tjsModelId) continue;
      options.push({
        value: dopplerModelId,
        label: `${getTranslateCompareModelLabel(dopplerModelId)} · ${tjsModelId}`,
      });
    }
  } else {
    for (const modelId of getTranslateCompatibleRegisteredModelIds()) {
      options.push({
        value: modelId,
        label: getTranslateCompareModelLabel(modelId),
      });
    }
  }

  if (options.length === 0) {
    const emptyOption = document.createElement('option');
    emptyOption.value = '';
    emptyOption.textContent = lane.engine === 'transformersjs'
      ? 'No mapped Transformers.js profiles'
      : 'No translate models imported';
    select.appendChild(emptyOption);
    lane.modelId = '';
    return;
  }

  for (const optionInfo of options) {
    const option = document.createElement('option');
    option.value = optionInfo.value;
    option.textContent = optionInfo.label;
    option.title = optionInfo.label;
    select.appendChild(option);
  }
  const nextValue = options.some((entry) => entry.value === previousValue)
    ? previousValue
    : options[0].value;
  select.value = nextValue;
  lane.modelId = nextValue;
  if (lane.engine === 'transformersjs') {
    lane.tjsModelId = resolveTransformersModelIdForLane(lane);
  }
}

function renderTranslateCompareSelectors() {
  for (const laneId of getCompareLaneIds()) {
    const lane = getCompareLane(laneId);
    const engineSelect = $(getCompareLaneEngineSelectId(laneId));
    if (engineSelect instanceof HTMLSelectElement && lane) {
      engineSelect.value = lane.engine;
    }
    populateCompareLaneModelSelect(laneId);
  }
}

function renderTranslateCompareEvidence() {
  const evidence = state.compareEvidence || TRANSLATE_COMPARE_EVIDENCE_FALLBACK;
  setText($('translate-proof-bundle'), evidence.updatedAt ? `Frozen ${evidence.updatedAt}` : 'Awaiting frozen scoreboard');
  setText(
    $('translate-proof-delta'),
    Number.isFinite(evidence?.teacher?.bleu) && Number.isFinite(evidence?.student?.bleu)
      ? `${(evidence.student.bleu - evidence.teacher.bleu).toFixed(2)} BLEU`
      : 'Delta pending'
  );
  setText($('translate-proof-claim'), evidence.summary || TRANSLATE_COMPARE_EVIDENCE_FALLBACK.summary);
  setText($('translate-proof-source'), evidence.updatedAt ? `Updated ${evidence.updatedAt}` : 'No receipt loaded');
  setText($('translate-evidence-teacher-bleu'), Number.isFinite(evidence?.teacher?.bleu) ? evidence.teacher.bleu.toFixed(2) : '--');
  setText($('translate-evidence-student-bleu'), Number.isFinite(evidence?.student?.bleu) ? evidence.student.bleu.toFixed(2) : '--');
  setText($('translate-evidence-teacher-chrf'), Number.isFinite(evidence?.teacher?.chrf) ? evidence.teacher.chrf.toFixed(2) : '--');
  setText($('translate-evidence-student-chrf'), Number.isFinite(evidence?.student?.chrf) ? evidence.student.chrf.toFixed(2) : '--');
  setText(
    $('translate-evidence-size-delta'),
    Number.isFinite(evidence?.teacher?.sizeBytes) && Number.isFinite(evidence?.student?.sizeBytes)
      ? `${formatBytes(evidence.teacher.sizeBytes - evidence.student.sizeBytes)} saved`
      : '--'
  );
  setText(
    $('translate-evidence-artifact'),
    Array.isArray(evidence?.receipts) && evidence.receipts.length > 0
      ? String(evidence.receipts[0].label || evidence.receipts[0].href || 'Open receipt')
      : 'Pending'
  );
}

function normalizeTranslateCompareHistoryFilter(filterId) {
  const normalized = resolveText(filterId, 'all');
  return TRANSLATE_COMPARE_HISTORY_FILTERS.some((entry) => entry.id === normalized)
    ? normalized
    : 'all';
}

function getTranslateCompareLaneRoleLabel({ presetId, laneId, modelId }) {
  const presetRole = resolveText(getTranslateComparePreset(presetId)?.lanes?.[laneId]?.role, '');
  if (presetRole === 'baseline' || presetRole === 'mapped-baseline') {
    return 'baseline';
  }
  if (presetRole === 'student' || presetRole === 'student-mapped') {
    return resolveText(modelId, '') ? 'student' : 'student slot';
  }
  if (resolveText(modelId, '') === TRANSLATE_COMPARE_DEFAULT_BASELINE_MODEL_ID) {
    return 'baseline';
  }
  const studentModelId = resolveText(getTranslateCompareStudentModelId(), '');
  if (studentModelId && resolveText(modelId, '') === studentModelId) {
    return 'student';
  }
  return 'custom';
}

function getTranslateCompareEntryLaneRoleLabel(entry, laneId) {
  const artifactRole = resolveText(entry?.artifact?.lanes?.[laneId]?.roleLabel, '');
  if (artifactRole) {
    return artifactRole;
  }
  return getTranslateCompareLaneRoleLabel({
    presetId: entry?.presetId,
    laneId,
    modelId: entry?.lanes?.[laneId]?.modelId,
  });
}

function summarizeTranslateCompareHistoryEntry(entry) {
  const left = entry?.lanes?.left || {};
  const right = entry?.lanes?.right || {};
  const leftTotalMs = Number(left?.metrics?.totalMs);
  const rightTotalMs = Number(right?.metrics?.totalMs);
  const leftSizeBytes = Number(left?.metrics?.sizeBytes);
  const rightSizeBytes = Number(right?.metrics?.sizeBytes);
  const errorLaneIds = getCompareLaneIds().filter((laneId) => {
    const lane = entry?.lanes?.[laneId];
    return resolveText(lane?.status, '').toLowerCase() === 'error' || lane?.error != null;
  });
  let fasterLaneId = null;
  if (Number.isFinite(leftTotalMs) && Number.isFinite(rightTotalMs) && leftTotalMs !== rightTotalMs) {
    fasterLaneId = leftTotalMs < rightTotalMs ? 'left' : 'right';
  }
  let smallerLaneId = null;
  if (Number.isFinite(leftSizeBytes) && Number.isFinite(rightSizeBytes) && leftSizeBytes !== rightSizeBytes) {
    smallerLaneId = leftSizeBytes < rightSizeBytes ? 'left' : 'right';
  }
  const sameModel = resolveText(left.modelId, '') !== '' && resolveText(left.modelId, '') === resolveText(right.modelId, '');
  const sameEngine = resolveText(left.engine, '') !== '' && resolveText(left.engine, '') === resolveText(right.engine, '');
  const roleLabels = getCompareLaneIds().map((laneId) => getTranslateCompareEntryLaneRoleLabel(entry, laneId));
  return {
    fasterLaneId,
    smallerLaneId,
    sameModel,
    sameEngine,
    hasBaseline: roleLabels.includes('baseline'),
    hasStudent: roleLabels.includes('student') || roleLabels.includes('student slot'),
    errorLaneIds,
  };
}

function matchesTranslateCompareHistoryFilter(entry, filterId) {
  const normalizedFilter = normalizeTranslateCompareHistoryFilter(filterId);
  const summary = summarizeTranslateCompareHistoryEntry(entry);
  if (normalizedFilter === 'same-model') {
    return summary.sameModel;
  }
  if (normalizedFilter === 'same-engine') {
    return summary.sameEngine;
  }
  if (normalizedFilter === 'proof') {
    return resolveText(entry?.presetId, '') === 'proof';
  }
  return true;
}

function getLatestTranslateCompareArtifact() {
  return state.lastCompareArtifact || state.compareHistory?.[0]?.artifact || null;
}

function buildTranslateCompareBrowserLabel() {
  const nav = typeof navigator === 'object' && navigator ? navigator : null;
  const brands = Array.isArray(nav?.userAgentData?.brands)
    ? nav.userAgentData.brands.map((entry) => resolveText(entry?.brand, '')).filter(Boolean)
    : [];
  if (brands.length > 0) {
    return brands.join(' / ');
  }
  const ua = resolveText(nav?.userAgent, '');
  if (!ua) return 'Browser';
  if (ua.includes('Chrome/')) return 'Chrome';
  if (ua.includes('Firefox/')) return 'Firefox';
  if (ua.includes('Safari/')) return 'Safari';
  return ua.split(' ').slice(0, 2).join(' ') || 'Browser';
}

function buildTranslateCompareEnvironmentMetadata() {
  const nav = typeof navigator === 'object' && navigator ? navigator : null;
  return {
    browserLabel: buildTranslateCompareBrowserLabel(),
    userAgent: resolveText(nav?.userAgent, ''),
    language: resolveText(nav?.language, ''),
    languages: Array.isArray(nav?.languages) ? nav.languages.slice() : [],
    platform: resolveText(nav?.platform, ''),
    hardwareConcurrency: Number.isFinite(Number(nav?.hardwareConcurrency))
      ? Number(nav.hardwareConcurrency)
      : null,
    deviceMemoryGb: Number.isFinite(Number(nav?.deviceMemory))
      ? Number(nav.deviceMemory)
      : null,
    devicePixelRatio: Number.isFinite(Number(globalThis?.devicePixelRatio))
      ? Number(globalThis.devicePixelRatio)
      : null,
    webgpuDeviceLabel: resolveText(state.compareDeviceLabel, ''),
    url: typeof window === 'object' ? window.location.href : '',
  };
}

function downloadJsonFile(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportTranslateCompareArtifactPayload(artifact) {
  if (!artifact || typeof artifact !== 'object') {
    updateRunStatus('No compare artifact available to export.');
    return;
  }
  const artifactId = resolveText(artifact.artifactId, new Date().toISOString().replace(/[:]/g, '-'));
  downloadJsonFile(`translate-compare-${artifactId}.json`, artifact);
  updateRunStatus(`Exported compare artifact ${artifactId}`);
}

function buildTranslateCompareArtifact(prompt, sourceCode, targetCode, options) {
  const createdAt = new Date().toISOString();
  const lanes = {};
  for (const laneId of getCompareLaneIds()) {
    const lane = getCompareLane(laneId) || {};
    lanes[laneId] = {
      engine: resolveText(lane.engine, ''),
      modelId: resolveText(lane.modelId, ''),
      modelLabel: getTranslateCompareModelLabel(lane.modelId),
      tjsModelId: resolveText(resolveTransformersModelIdForLane(lane), ''),
      roleLabel: getTranslateCompareLaneRoleLabel({
        presetId: state.comparePresetId,
        laneId,
        modelId: lane.modelId,
      }),
      status: resolveText(lane.status, ''),
      output: String(lane.output || ''),
      metrics: lane.metrics || null,
      error: lane.error || null,
    };
  }
  const snapshot = {
    presetId: state.comparePresetId,
    lanes,
  };
  const summary = summarizeTranslateCompareHistoryEntry(snapshot);
  const evidence = state.compareEvidence || TRANSLATE_COMPARE_EVIDENCE_FALLBACK;
  return {
    schemaVersion: TRANSLATE_COMPARE_CONFIG_VERSION,
    kind: TRANSLATE_COMPARE_ARTIFACT_KIND,
    artifactId: crypto?.randomUUID?.() || `${Date.now()}`,
    createdAt,
    shareUrl: buildTranslateDeepLinkUrl(),
    request: {
      prompt,
      sourceCode,
      sourceName: getTranslateLanguageName(sourceCode),
      targetCode,
      targetName: getTranslateLanguageName(targetCode),
      options: {
        temperature: options.temperature,
        topP: options.topP,
        topK: options.topK,
        maxTokens: options.maxTokens,
      },
      presetId: state.comparePresetId || 'custom',
    },
    environment: buildTranslateCompareEnvironmentMetadata(),
    evidence: {
      updatedAt: evidence.updatedAt,
      summary: evidence.summary,
      receipts: Array.isArray(evidence.receipts) ? evidence.receipts : [],
    },
    summary,
    lanes,
  };
}

function renderTranslateCompareReceipts() {
  const artifact = getLatestTranslateCompareArtifact();
  const exportButton = $('translate-compare-export-latest-btn');
  const linksWrap = $('translate-compare-receipts-links');
  setText($('translate-compare-artifact-id'), resolveText(artifact?.artifactId, '') || 'Pending');
  setText(
    $('translate-compare-artifact-env'),
    artifact?.environment?.browserLabel
      ? `${artifact.environment.browserLabel} · ${resolveText(artifact.environment.webgpuDeviceLabel, 'WebGPU')}`
      : 'Awaiting run'
  );
  const receiptCount = Array.isArray(artifact?.evidence?.receipts) ? artifact.evidence.receipts.length : 0;
  setText($('translate-compare-receipt-count'), String(receiptCount));
  if (exportButton instanceof HTMLButtonElement) {
    exportButton.disabled = !artifact;
  }
  if (!linksWrap) return;
  linksWrap.textContent = '';
  if (!artifact || receiptCount === 0) {
    const empty = document.createElement('span');
    empty.className = 'type-caption';
    empty.textContent = 'No benchmark receipts linked yet.';
    linksWrap.appendChild(empty);
    return;
  }
  let appended = 0;
  for (const receipt of artifact.evidence.receipts) {
    const href = resolveText(receipt?.href, '');
    if (!href) {
      const label = resolveText(receipt?.label, '');
      if (!label) continue;
      const span = document.createElement('span');
      span.className = 'type-caption';
      span.textContent = label;
      linksWrap.appendChild(span);
      appended += 1;
      continue;
    }
    const link = document.createElement('a');
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = resolveText(receipt?.label, href);
    linksWrap.appendChild(link);
    appended += 1;
  }
  if (appended === 0) {
    const empty = document.createElement('span');
    empty.className = 'type-caption';
    empty.textContent = 'No benchmark receipts linked yet.';
    linksWrap.appendChild(empty);
  }
}

function formatTranslateSmokeBucketLabel(bucket) {
  if (bucket === 'easy') return 'easy';
  if (bucket === 'nuanced') return 'nuanced';
  if (bucket === 'domain') return 'domain';
  if (bucket === 'edge') return 'edge';
  return 'sample';
}

function renderTranslateCompareSmokePanel() {
  const grid = $('translate-smoke-grid');
  if (!grid) return;
  grid.textContent = '';
  for (const sample of TRANSLATE_COMPARE_SMOKE_SAMPLES) {
    const card = document.createElement('article');
    card.className = state.activeCompareSmokeSampleId === sample.id
      ? 'translate-smoke-card is-active'
      : 'translate-smoke-card';
    card.innerHTML = `
      <div class="translate-smoke-card-top">
        <span class="translate-smoke-chip">${escapeHtml(formatTranslateSmokeBucketLabel(sample.bucket))}</span>
        <span class="type-caption">${escapeHtml(sample.label)}</span>
      </div>
      <p class="translate-history-snippet">${escapeHtml(sample.text)}</p>
      <div class="translate-smoke-meta">
        <span class="type-caption">${escapeHtml(getTranslateLanguageName(sample.sourceCode))} -> ${escapeHtml(getTranslateLanguageName(sample.targetCode))}</span>
        <span class="type-caption">${escapeHtml(sample.note)}</span>
      </div>
      <div class="translate-smoke-card-actions">
        <button class="btn btn-small" type="button" data-compare-smoke-sample="${escapeHtml(sample.id)}">Load sample</button>
      </div>
    `;
    grid.appendChild(card);
  }
}

function renderTranslateCompareLane(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) return;
  const prefix = laneId === 'left' ? 'translate-left' : 'translate-right';
  const metrics = lane.metrics || {};
  const statusEl = $(`${prefix}-status`);
  const badgeEl = $(`${prefix}-badge`);
  const sizeBytes = Number.isFinite(metrics.sizeBytes)
    ? metrics.sizeBytes
    : resolveCompareModelSizeBytes(lane.modelId);
  setText(statusEl, lane.status || 'Idle');
  setText($(`${prefix}-size`), formatCompareMetricBytes(sizeBytes));
  setText($(`${prefix}-load-ms`), formatCompareMetricMs(metrics.modelLoadMs));
  setText($(`${prefix}-ttft`), formatCompareMetricMs(metrics.ttftMs));
  setText($(`${prefix}-decode-rate`), formatCompareMetricRate(metrics.decodeTokensPerSec));
  setText($(`${prefix}-total-ms`), formatCompareMetricMs(metrics.totalMs));
  setText(
    $(`${prefix}-device`),
    resolveText(metrics.deviceLabel, state.compareDeviceLabel || (lane.engine === 'transformersjs' ? 'WebGPU / ORT' : 'WebGPU'))
  );
  setText($(`${prefix}-meta`), resolveText(metrics.metaLabel, lane.error ? 'Run failed' : 'No run yet'));
  setText($(`${prefix}-output`), lane.output || (lane.error ? String(lane.error) : 'Awaiting compare run.'));
  if (statusEl) {
    statusEl.dataset.tone = resolveText(lane.statusTone, 'info');
  }
  setText(
    badgeEl,
    getTranslateCompareLaneRoleLabel({
      presetId: state.comparePresetId,
      laneId,
      modelId: lane.modelId,
    })
  );
}

function renderTranslateCompareHistory() {
  const list = $('translate-history-list');
  if (!list) return;
  const rows = (Array.isArray(state.compareHistory) ? state.compareHistory : [])
    .filter((entry) => matchesTranslateCompareHistoryFilter(entry, state.compareHistoryFilter));
  list.innerHTML = '';

  if (rows.length === 0) {
    const filterLabel = TRANSLATE_COMPARE_HISTORY_FILTERS.find((entry) => entry.id === normalizeTranslateCompareHistoryFilter(state.compareHistoryFilter))?.label || 'All';
    list.innerHTML = `
      <article class="translate-history-card is-placeholder">
        <div class="translate-history-card-top">
          <span class="translate-history-time type-caption">Pending</span>
          <span class="translate-history-badge type-caption">${escapeHtml(filterLabel)}</span>
        </div>
        <p class="translate-history-snippet">Compare receipts will stack here with engine/model labels, timing, and expandable outputs.</p>
        <div class="translate-history-empty type-caption">No history entries match the current filter yet.</div>
      </article>
    `;
    return;
  }

  for (const entry of rows) {
    const summaryState = summarizeTranslateCompareHistoryEntry(entry);
    const card = document.createElement('details');
    card.className = 'translate-history-card';
    const left = entry?.lanes?.left || {};
    const right = entry?.lanes?.right || {};
    const badges = [];
    if (summaryState.fasterLaneId) {
      badges.push({ label: `${summaryState.fasterLaneId} faster`, tone: 'success' });
    }
    if (summaryState.smallerLaneId) {
      badges.push({ label: `${summaryState.smallerLaneId} smaller`, tone: 'success' });
    }
    if (summaryState.sameModel) {
      badges.push({ label: 'same model', tone: 'neutral' });
    }
    if (summaryState.sameEngine) {
      badges.push({ label: 'same engine', tone: 'neutral' });
    }
    if (summaryState.hasBaseline) {
      badges.push({ label: 'baseline', tone: 'neutral' });
    }
    if (summaryState.hasStudent) {
      badges.push({ label: 'student', tone: 'warning' });
    }
    if (summaryState.errorLaneIds.length > 0) {
      badges.push({ label: `error:${summaryState.errorLaneIds.join(',')}`, tone: 'warning' });
    }
    const badgeMarkup = badges.map((badge) => (
      `<span class="translate-history-badge type-caption is-${escapeHtml(badge.tone)}">${escapeHtml(badge.label)}</span>`
    )).join('');
    const receiptLinks = Array.isArray(entry?.artifact?.evidence?.receipts)
      ? entry.artifact.evidence.receipts.filter((receipt) => resolveText(receipt?.href, ''))
      : [];
    const receiptMarkup = receiptLinks.length > 0
      ? receiptLinks.map((receipt) => (
        `<a class="btn btn-small" href="${escapeHtml(receipt.href)}" target="_blank" rel="noopener">${escapeHtml(resolveText(receipt.label, 'Receipt'))}</a>`
      )).join('')
      : '<span class="type-caption">No linked receipts</span>';
    const rawTiming = JSON.stringify({
      left: left.metrics || null,
      right: right.metrics || null,
    }, null, 2);
    const summary = document.createElement('summary');
    summary.innerHTML = `
      <div class="translate-history-card-top">
        <span class="translate-history-time type-caption">${formatCompareTimestamp(entry.createdAt)}</span>
        <span class="translate-history-badge type-caption">${entry.presetId || 'custom'}</span>
      </div>
      <div class="translate-history-badges">${badgeMarkup}</div>
      <p class="translate-history-snippet">${escapeHtml(String(entry.prompt || '').slice(0, 180) || 'No prompt captured.')}</p>
      <div class="translate-history-meta">
        <span class="type-caption">${escapeHtml(getTranslateLanguageName(entry.sourceCode))} -> ${escapeHtml(getTranslateLanguageName(entry.targetCode))}</span>
        <span class="type-caption">${escapeHtml(left.engine || 'left')} vs ${escapeHtml(right.engine || 'right')}</span>
      </div>
    `;
    card.appendChild(summary);

    const body = document.createElement('div');
    body.className = 'translate-history-body';
    body.innerHTML = `
      <div class="translate-history-lane-block">
        <div class="translate-history-meta">
          <span class="type-caption">${escapeHtml(getTranslateCompareModelLabel(left.modelId))} · ${escapeHtml(left.status || 'idle')} · ${escapeHtml(getTranslateCompareEntryLaneRoleLabel(entry, 'left'))}</span>
          <span class="type-caption">load ${formatCompareMetricMs(left.metrics?.modelLoadMs)} · ttft ${formatCompareMetricMs(left.metrics?.ttftMs)} · total ${formatCompareMetricMs(left.metrics?.totalMs)} · ${formatCompareMetricRate(left.metrics?.decodeTokensPerSec)}</span>
        </div>
        <pre class="playground-output-box translate-lane-output-box">${escapeHtml(String(left.output || ''))}</pre>
      </div>
      <div class="translate-history-lane-block">
        <div class="translate-history-meta">
          <span class="type-caption">${escapeHtml(getTranslateCompareModelLabel(right.modelId))} · ${escapeHtml(right.status || 'idle')} · ${escapeHtml(getTranslateCompareEntryLaneRoleLabel(entry, 'right'))}</span>
          <span class="type-caption">load ${formatCompareMetricMs(right.metrics?.modelLoadMs)} · ttft ${formatCompareMetricMs(right.metrics?.ttftMs)} · total ${formatCompareMetricMs(right.metrics?.totalMs)} · ${formatCompareMetricRate(right.metrics?.decodeTokensPerSec)}</span>
        </div>
        <pre class="playground-output-box translate-lane-output-box">${escapeHtml(String(right.output || ''))}</pre>
      </div>
      <div class="translate-history-actions">
        <button class="btn btn-small" type="button" data-compare-history-export="${escapeHtml(entry.id)}">Export JSON</button>
        ${receiptMarkup}
      </div>
      <details class="diagnostics-output-json">
        <summary class="type-caption">Raw timing breakdown</summary>
        <pre class="playground-output-box translate-history-raw">${escapeHtml(rawTiming)}</pre>
      </details>
    `;
    card.appendChild(body);
    list.appendChild(card);
  }
}

function syncTranslateCompareToggleButtons() {
  const enabled = isTranslateCompareEnabled();
  document.querySelectorAll('[data-translate-view], [data-translate-layout]').forEach((button) => {
    const target = button?.dataset?.translateView || button?.dataset?.translateLayout;
    const isCompareTarget = target === 'compare';
    button.classList.toggle('is-active', enabled === isCompareTarget);
  });
}

function syncTranslateCompareUI() {
  const compareShell = $('translate-compare-shell');
  const singleOutputBox = $('run-output')?.closest('.playground-output');
  const presetSelect = $('translate-compare-preset');
  const presetNote = $('translate-compare-preset-note');
  const runButton = $('translate-compare-run-btn');
  const exportButton = $('translate-compare-export-btn');
  const shareButton = $('translate-compare-share-btn');
  const enabled = isTranslateCompareEnabled();
  setHidden(compareShell, !enabled);
  setHidden(singleOutputBox, enabled);
  if (presetSelect instanceof HTMLSelectElement) {
    presetSelect.value = state.comparePresetId || 'proof';
  }
  setText(presetNote, getTranslateComparePresetNote(state.comparePresetId || 'proof'));
  if (runButton instanceof HTMLButtonElement) {
    runButton.disabled = state.compareGenerating || state.compareLoading;
  }
  if (exportButton instanceof HTMLButtonElement) {
    exportButton.disabled = state.compareGenerating || state.compareLoading || !getLatestTranslateCompareArtifact();
  }
  if (shareButton instanceof HTMLButtonElement) {
    shareButton.disabled = state.compareGenerating || state.compareLoading;
  }
  document.querySelectorAll('[data-compare-history-filter]').forEach((button) => {
    const filterId = normalizeTranslateCompareHistoryFilter(button?.dataset?.compareHistoryFilter);
    button.classList.toggle('is-active', filterId === normalizeTranslateCompareHistoryFilter(state.compareHistoryFilter));
  });
  syncTranslateCompareToggleButtons();
  renderTranslateCompareEvidence();
  renderTranslateCompareReceipts();
  renderTranslateCompareSmokePanel();
  renderTranslateCompareSelectors();
  renderTranslateCompareHistory();
  for (const laneId of getCompareLaneIds()) {
    renderTranslateCompareLane(laneId);
  }
}

async function applyTranslateComparePreset(presetId, options = {}) {
  ensureTranslateCompareRuntimeState();
  const preset = getTranslateComparePreset(presetId);
  state.comparePresetId = preset.id;
  const { preserveExisting = false } = options;

  for (const laneId of getCompareLaneIds()) {
    const lane = getCompareLane(laneId);
    const lanePreset = preset.lanes?.[laneId] || {};
    const resolved = resolveTranslateCompareRole(lanePreset.role);
    if (!preserveExisting || !resolveText(lane.modelId, '')) {
      lane.engine = resolveText(lanePreset.engine, lane.engine || 'doppler');
      lane.modelId = resolveText(resolved.modelId, lane.modelId || '');
      lane.tjsModelId = resolveText(resolved.tjsModelId, lane.tjsModelId || '');
    }
    clearCompareLaneResult(laneId);
  }
  syncTranslateCompareUI();
}

let transformersRuntimePromise = null;

async function loadTransformersJsRuntime() {
  if (transformersRuntimePromise) {
    return transformersRuntimePromise;
  }
  transformersRuntimePromise = (async () => {
    let lastError = null;
    for (const candidate of TRANSFORMERSJS_IMPORT_CANDIDATES) {
      try {
        const runtime = await import(candidate);
        if (!runtime?.pipeline) {
          throw new Error('module did not expose pipeline()');
        }
        if (runtime.env && typeof runtime.env === 'object') {
          runtime.env.allowLocalModels = false;
          runtime.env.allowRemoteModels = true;
        }
        return runtime;
      } catch (error) {
        lastError = error;
      }
    }
    throw new Error(`Transformers.js runtime unavailable: ${lastError?.message || 'unknown import failure'}`);
  })();
  return transformersRuntimePromise;
}

async function unloadCompareLaneRuntime(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) return;
  if (lane.pipeline) {
    try {
      await lane.pipeline.unload?.();
    } catch (error) {
      log.warn('DopplerDemo', `Compare lane unload failed (${laneId}): ${error.message}`);
    }
  }
  lane.pipeline = null;
  lane.pipelineModelId = null;
  lane.tjsGenerator = null;
  lane.tjsGeneratorKey = null;
}

async function unloadAllCompareLaneRuntimes() {
  for (const laneId of getCompareLaneIds()) {
    await unloadCompareLaneRuntime(laneId);
  }
}

async function ensureCompareDopplerPipeline(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) {
    throw new Error(`Unknown compare lane "${laneId}".`);
  }
  const modelId = resolveText(lane.modelId, '');
  if (!modelId) {
    throw new Error(`Compare ${laneId}: select a Doppler model first.`);
  }
  if (lane.pipeline && lane.pipelineModelId === modelId) {
    return {
      pipeline: lane.pipeline,
      modelLoadMs: 0,
    };
  }

  await unloadCompareLaneRuntime(laneId);
  setCompareLaneStatus(laneId, 'Loading', 'info');
  renderTranslateCompareLane(laneId);
  const startedAt = performance.now();
  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error(`Compare ${laneId}: manifest missing for "${modelId}".`);
  }
  const manifest = parseManifest(manifestText);
  await initDevice();
  const device = getDevice();
  const pipeline = await createPipeline(manifest, {
    gpu: { device },
    runtimeConfig: cloneRuntimeConfig(getRuntimeConfig()),
  });
  lane.pipeline = pipeline;
  lane.pipelineModelId = modelId;
  return {
    pipeline,
    modelLoadMs: Math.max(0, performance.now() - startedAt),
  };
}

async function ensureTransformersGeneratorForLane(laneId) {
  const lane = getCompareLane(laneId);
  if (!lane) {
    throw new Error(`Unknown compare lane "${laneId}".`);
  }
  const modelId = resolveTransformersModelIdForLane(lane);
  if (!modelId) {
    throw new Error(`Compare ${laneId}: no Transformers.js profile is configured for "${lane.modelId || 'this lane'}".`);
  }
  const generatorKey = `${modelId}::${lane.tjsDtype || TRANSLATE_COMPARE_DEFAULT_TJS_DTYPE}`;
  if (lane.tjsGenerator && lane.tjsGeneratorKey === generatorKey) {
    return {
      generator: lane.tjsGenerator,
      runtime: await loadTransformersJsRuntime(),
      modelLoadMs: 0,
    };
  }
  const runtime = await loadTransformersJsRuntime();
  setCompareLaneStatus(laneId, 'Loading', 'info');
  renderTranslateCompareLane(laneId);
  const startedAt = performance.now();
  const generator = await runtime.pipeline('text-generation', modelId, {
    device: 'webgpu',
    dtype: lane.tjsDtype || TRANSLATE_COMPARE_DEFAULT_TJS_DTYPE,
  });
  lane.tjsGenerator = generator;
  lane.tjsGeneratorKey = generatorKey;
  lane.tjsModelId = modelId;
  return {
    generator,
    runtime,
    modelLoadMs: Math.max(0, performance.now() - startedAt),
  };
}

function buildTranslateCompareOptions() {
  const temperature = readOptionalNumber($('temperature-input'));
  const topP = readOptionalNumber($('top-p-input'));
  const topK = readOptionalNumber($('top-k-input'), { integer: true });
  const maxTokens = readOptionalNumber($('max-tokens-input'), { integer: true });
  return {
    temperature: temperature != null ? Math.max(0, temperature) : 0,
    topP: topP != null ? Math.max(0, Math.min(1, topP)) : 1,
    topK: topK != null ? Math.max(1, topK) : 1,
    maxTokens: maxTokens != null && maxTokens > 0 ? maxTokens : TRANSLATE_COMPARE_DEFAULT_MAX_TOKENS,
  };
}

async function runDopplerCompareLane(laneId, context) {
  const lane = getCompareLane(laneId);
  const { prompt, sourceCode, targetCode, options } = context;
  const { pipeline, modelLoadMs } = await ensureCompareDopplerPipeline(laneId);
  const manifestModelId = resolveText(pipeline?.manifest?.modelId, lane.modelId);
  const modelType = normalizeModelType(pipeline?.manifest?.modelType);
  const translateRequest = createTranslateTextRequest(prompt, sourceCode, targetCode);
  let generationInput = translateRequest;
  if (pipeline?.manifest?.inference?.chatTemplate?.type !== 'translategemma') {
    generationInput = buildTranslateInstructionPrompt(prompt, sourceCode, targetCode);
  }
  if (modelType !== 'transformer' && modelType !== null) {
    throw new Error(`Compare ${laneId}: selected model "${manifestModelId}" is not a text model.`);
  }

  lane.output = '';
  lane.error = null;
  setCompareLaneStatus(laneId, modelLoadMs > 0 ? 'Warm' : 'Warm', 'info');
  renderTranslateCompareLane(laneId);
  pipeline.reset?.();
  setCompareLaneStatus(laneId, 'Translating', 'info');
  renderTranslateCompareLane(laneId);

  for await (const token of pipeline.generate(generationInput, {
    ...options,
    useChatTemplate: true,
  })) {
    lane.output += token;
    renderTranslateCompareLane(laneId);
  }

  const stats = pipeline.getStats?.() || {};
  const totalMs = Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : null;
  const decodeTokens = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
  const decodeMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
  lane.metrics = {
    modelLoadMs,
    ttftMs: Number.isFinite(stats.ttftMs) ? stats.ttftMs : stats.prefillTimeMs,
    totalMs,
    decodeTokensPerSec: (decodeTokens != null && decodeMs && decodeMs > 0) ? decodeTokens / (decodeMs / 1000) : null,
    sizeBytes: resolveCompareModelSizeBytes(lane.modelId),
    deviceLabel: state.compareDeviceLabel || 'WebGPU',
    metaLabel: `${getTranslateCompareModelLabel(lane.modelId)} · ${lane.output.length} chars`,
  };
  setCompareLaneStatus(laneId, 'Complete', 'success');
  renderTranslateCompareLane(laneId);
}

async function runTransformersCompareLane(laneId, context) {
  const lane = getCompareLane(laneId);
  const { prompt, sourceCode, targetCode, options } = context;
  const { generator, runtime, modelLoadMs } = await ensureTransformersGeneratorForLane(laneId);
  const modelRef = resolveTransformersModelIdForLane(lane);
  const generationPrompt = buildTransformersTranslatePrompt(prompt, sourceCode, targetCode, modelRef);
  const doSample = Number(options.temperature) > 0 && Number(options.topK) > 1;

  lane.output = '';
  lane.error = null;
  setCompareLaneStatus(laneId, 'Warm', 'info');
  renderTranslateCompareLane(laneId);

  let firstTokenAt = null;
  let decodeTokens = 0;
  const tokenTimestamps = [];
  const chunks = [];
  const startedAt = performance.now();
  const TextStreamer = runtime.TextStreamer;
  const streamer = typeof TextStreamer === 'function'
    ? new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      callback_function: (text) => {
        chunks.push(text);
        lane.output = chunks.join('');
        renderTranslateCompareLane(laneId);
      },
      token_callback_function: (tokens) => {
        const now = performance.now();
        const count = Array.isArray(tokens) ? tokens.length : 1;
        decodeTokens += count;
        if (firstTokenAt === null) {
          firstTokenAt = now;
        }
        for (let index = 0; index < count; index += 1) {
          tokenTimestamps.push(now);
        }
      },
    })
    : null;

  setCompareLaneStatus(laneId, 'Translating', 'info');
  renderTranslateCompareLane(laneId);
  await generator(generationPrompt, {
    max_new_tokens: options.maxTokens,
    do_sample: doSample,
    temperature: doSample ? options.temperature : 1,
    top_k: doSample ? options.topK : 1,
    top_p: doSample ? options.topP : 1,
    num_beams: 1,
    num_beam_groups: 1,
    ...(streamer ? { streamer } : {}),
  });
  const endedAt = performance.now();
  const totalMs = Math.max(1, endedAt - startedAt);
  const ttftMs = firstTokenAt != null ? Math.max(1, firstTokenAt - startedAt) : totalMs;
  const decodeMs = Math.max(totalMs - ttftMs, 1);
  const effectiveDecodeTokens = Math.max(decodeTokens - 1, 0);
  lane.metrics = {
    modelLoadMs,
    ttftMs,
    totalMs,
    decodeTokensPerSec: effectiveDecodeTokens > 0 ? effectiveDecodeTokens / (decodeMs / 1000) : null,
    sizeBytes: resolveCompareModelSizeBytes(lane.modelId),
    deviceLabel: await resolveWebGpuDeviceLabel('WebGPU / ORT'),
    metaLabel: `${resolveText(modelRef, lane.modelId)} · ${lane.output.length} chars`,
  };
  setCompareLaneStatus(laneId, 'Complete', 'success');
  renderTranslateCompareLane(laneId);
}

function captureTranslateCompareHistoryEntry(prompt, sourceCode, targetCode, artifact) {
  return {
    id: crypto?.randomUUID?.() || `${Date.now()}`,
    createdAt: new Date().toISOString(),
    sourceCode,
    targetCode,
    prompt,
    presetId: state.comparePresetId,
    artifact: serializeTranslateCompareArtifactPayload(artifact),
    lanes: {
      left: serializeTranslateCompareHistoryEntry({ lanes: { left: state.compareLanes.left } }).lanes.left,
      right: serializeTranslateCompareHistoryEntry({ lanes: { right: state.compareLanes.right } }).lanes.right,
    },
  };
}

function applyTranslateCompareSmokeSample(sampleId) {
  const sample = TRANSLATE_COMPARE_SMOKE_SAMPLES.find((entry) => entry.id === resolveText(sampleId, ''));
  if (!sample) {
    updateRunStatus('Unknown smoke sample.');
    return;
  }
  const promptEl = $('run-prompt');
  const sourceEl = $('translate-source-language');
  const targetEl = $('translate-target-language');
  if (promptEl instanceof HTMLTextAreaElement || promptEl instanceof HTMLInputElement) {
    promptEl.value = sample.text;
    setStarterExampleInput(promptEl, false);
  }
  if (sourceEl instanceof HTMLSelectElement) {
    sourceEl.value = normalizeTranslateLanguageCode(sample.sourceCode, DEFAULT_TRANSLATE_SOURCE);
  }
  if (targetEl instanceof HTMLSelectElement) {
    targetEl.value = normalizeTranslateLanguageCode(sample.targetCode, DEFAULT_TRANSLATE_TARGET);
  }
  state.activeCompareSmokeSampleId = sample.id;
  renderTranslateCompareSmokePanel();
  syncDeepLinkFromUI();
  updateRunStatus(`Loaded smoke sample: ${sample.label}`);
}

function exportTranslateCompareHistoryArtifactById(entryId) {
  const entry = (state.compareHistory || []).find((row) => row?.id === entryId);
  if (!entry?.artifact) {
    updateRunStatus('Saved compare artifact not found.');
    return;
  }
  exportTranslateCompareArtifactPayload(entry.artifact);
}

async function handleTranslateCompareRun() {
  if (state.compareGenerating || state.compareLoading) return;
  ensureTranslateCompareRuntimeState();
  const prompt = $('run-prompt')?.value?.trim() || '';
  if (!prompt) {
    updateRunStatus('Enter text to translate.');
    return;
  }
  const { sourceCode, targetCode } = getTranslateLanguageSelection();
  const options = buildTranslateCompareOptions();

  state.compareGenerating = true;
  updateStatusIndicator();
  updateRunStatus('Running compare...');
  syncRunControls();
  syncTranslateCompareUI();
  for (const laneId of getCompareLaneIds()) {
    clearCompareLaneResult(laneId);
    renderTranslateCompareLane(laneId);
  }

  try {
    const laneErrors = [];
    for (const laneId of getCompareLaneIds()) {
      const lane = getCompareLane(laneId);
      if (!lane) continue;
      try {
        if (lane.engine === 'transformersjs') {
          await runTransformersCompareLane(laneId, { prompt, sourceCode, targetCode, options });
        } else {
          await runDopplerCompareLane(laneId, { prompt, sourceCode, targetCode, options });
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        lane.error = message;
        lane.output = '';
        lane.metrics = {
          modelLoadMs: lane.metrics?.modelLoadMs ?? null,
          ttftMs: lane.metrics?.ttftMs ?? null,
          totalMs: lane.metrics?.totalMs ?? null,
          decodeTokensPerSec: lane.metrics?.decodeTokensPerSec ?? null,
          sizeBytes: resolveCompareModelSizeBytes(lane.modelId),
          deviceLabel: resolveText(state.compareDeviceLabel, lane.engine === 'transformersjs' ? 'WebGPU / ORT' : 'WebGPU'),
          metaLabel: 'Run failed',
        };
        setCompareLaneStatus(laneId, 'Error', 'error');
        renderTranslateCompareLane(laneId);
        laneErrors.push({ laneId, message });
        log.error('DopplerDemo', `Translate compare lane failed (${laneId}): ${message}`);
      }
    }
    const artifact = buildTranslateCompareArtifact(prompt, sourceCode, targetCode, options);
    state.lastCompareArtifact = artifact;
    const entry = captureTranslateCompareHistoryEntry(prompt, sourceCode, targetCode, artifact);
    state.compareHistory.unshift(entry);
    state.compareHistory = state.compareHistory.slice(0, TRANSLATE_COMPARE_MAX_HISTORY);
    persistTranslateCompareHistory();
    renderTranslateCompareHistory();
    renderTranslateCompareReceipts();
    if (laneErrors.length > 0) {
      updateRunStatus(`Compare completed with ${laneErrors.length} lane error${laneErrors.length === 1 ? '' : 's'}.`);
    } else {
      updateRunStatus('Compare complete');
    }
  } finally {
    state.compareGenerating = false;
    updateStatusIndicator();
    syncRunControls();
    syncTranslateCompareUI();
  }
}

// --- Exports ---
export {
  getCompareLaneIds,
  getCompareLane,
  createCompareRuntimeLane,
  ensureTranslateCompareRuntimeState,
  setCompareLaneStatus,
  clearCompareLaneResult,
  isTranslateCompareEnabled,
  isTranslateCompatibleModelId,
  getTranslateCompareStudentModelId,
  getMappedCompareBaselineProfile,
  findCompareProfileByDopplerModelId,
  resolveTranslateCompareRole,
  serializeTranslateCompareArtifactPayload,
  serializeTranslateCompareHistoryEntry,
  persistTranslateCompareHistory,
  hydrateTranslateCompareHistory,
  normalizeTranslateCompareEvidence,
  buildTranslateInstructionPrompt,
  buildTransformersTranslatePrompt,
  getTranslateCompatibleRegisteredModelIds,
  getTranslateComparePreset,
  getTranslateComparePresetNote,
  getTranslateCompareModelLabel,
  resolveCompareModelSizeBytes,
  resolveTransformersModelIdForLane,
  formatCompareMetricMs,
  formatCompareMetricRate,
  formatCompareMetricBytes,
  formatCompareTimestamp,
  getCompareLaneSelectId,
  getCompareLaneEngineSelectId,
  populateCompareLaneModelSelect,
  renderTranslateCompareSelectors,
  renderTranslateCompareEvidence,
  normalizeTranslateCompareHistoryFilter,
  getTranslateCompareLaneRoleLabel,
  getTranslateCompareEntryLaneRoleLabel,
  summarizeTranslateCompareHistoryEntry,
  matchesTranslateCompareHistoryFilter,
  getLatestTranslateCompareArtifact,
  buildTranslateCompareBrowserLabel,
  buildTranslateCompareEnvironmentMetadata,
  downloadJsonFile,
  exportTranslateCompareArtifactPayload,
  buildTranslateCompareArtifact,
  renderTranslateCompareReceipts,
  formatTranslateSmokeBucketLabel,
  renderTranslateCompareSmokePanel,
  renderTranslateCompareLane,
  renderTranslateCompareHistory,
  syncTranslateCompareToggleButtons,
  syncTranslateCompareUI,
  buildTranslateCompareOptions,
  captureTranslateCompareHistoryEntry,
  applyTranslateCompareSmokeSample,
  exportTranslateCompareHistoryArtifactById,
};
