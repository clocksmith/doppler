function resolveText(value, defaultValue = '') {
  if (value == null) return defaultValue;
  return String(value).trim();
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function normalizeTranslateLanguageCode(code, defaultLanguageCode = DEFAULT_TRANSLATE_SOURCE) {
  const requested = resolveText(code, '');
  if (TRANSLATE_LANGUAGE_OPTIONS.some((entry) => entry.code === requested)) {
    return requested;
  }
  return defaultLanguageCode;
}

function populateTranslateLanguageSelect(selectEl, selectedCode) {
  if (!(selectEl instanceof HTMLSelectElement)) return;
  const previousCode = resolveText(selectedCode, selectEl.value || DEFAULT_TRANSLATE_SOURCE);
  selectEl.innerHTML = '';
  for (const entry of TRANSLATE_LANGUAGE_OPTIONS) {
    const option = document.createElement('option');
    option.value = entry.code;
    option.textContent = `${entry.name} (${entry.code})`;
    selectEl.appendChild(option);
  }
  selectEl.value = normalizeTranslateLanguageCode(previousCode, DEFAULT_TRANSLATE_SOURCE);
}

function populateTranslateLanguageControls() {
  const sourceSelect = $('translate-source-language');
  const targetSelect = $('translate-target-language');
  populateTranslateLanguageSelect(sourceSelect, DEFAULT_TRANSLATE_SOURCE);
  populateTranslateLanguageSelect(targetSelect, DEFAULT_TRANSLATE_TARGET);
  if (sourceSelect && targetSelect && sourceSelect.value === targetSelect.value) {
    targetSelect.value = DEFAULT_TRANSLATE_TARGET;
  }
}

function swapTranslateLanguages() {
  const sourceSelect = $('translate-source-language');
  const targetSelect = $('translate-target-language');
  if (!(sourceSelect instanceof HTMLSelectElement) || !(targetSelect instanceof HTMLSelectElement)) {
    return;
  }
  const sourceCode = normalizeTranslateLanguageCode(sourceSelect.value, DEFAULT_TRANSLATE_SOURCE);
  const targetCode = normalizeTranslateLanguageCode(targetSelect.value, DEFAULT_TRANSLATE_TARGET);
  sourceSelect.value = targetCode;
  targetSelect.value = sourceCode;
}

function getTranslateLanguageSelection() {
  const sourceSelect = $('translate-source-language');
  const targetSelect = $('translate-target-language');
  const sourceCode = normalizeTranslateLanguageCode(sourceSelect?.value, DEFAULT_TRANSLATE_SOURCE);
  let targetCode = normalizeTranslateLanguageCode(targetSelect?.value, DEFAULT_TRANSLATE_TARGET);
  if (targetCode === sourceCode) {
    targetCode = sourceCode === DEFAULT_TRANSLATE_TARGET
      ? DEFAULT_TRANSLATE_SOURCE
      : DEFAULT_TRANSLATE_TARGET;
    if (targetSelect instanceof HTMLSelectElement) {
      targetSelect.value = targetCode;
    }
  }
  return { sourceCode, targetCode };
}

function getTranslateLanguageName(code) {
  const normalized = normalizeTranslateLanguageCode(code, DEFAULT_TRANSLATE_SOURCE);
  const match = TRANSLATE_LANGUAGE_OPTIONS.find((entry) => entry.code === normalized);
  return match?.name || normalized;
}

// --- Exports ---
export {
  resolveText,
  escapeHtml,
  normalizeTranslateLanguageCode,
  populateTranslateLanguageSelect,
  populateTranslateLanguageControls,
  swapTranslateLanguages,
  getTranslateLanguageSelection,
  getTranslateLanguageName,
};
