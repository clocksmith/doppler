export function normalizeOptionalString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

export function normalizeDistillDatasetPath(value) {
  return normalizeOptionalString(value);
}

function normalizeLangCode(value) {
  const normalized = normalizeOptionalString(value);
  if (!normalized) return null;
  const compact = normalized.toLowerCase().replace(/_/g, '-');
  if (compact.startsWith('en')) return 'en';
  if (compact.startsWith('es')) return 'es';
  return compact;
}

function normalizePairDirection(value) {
  const pair = normalizeOptionalString(value);
  if (!pair) return null;
  const normalized = pair.toLowerCase().replace(/_/g, '-').replace(/\s+/g, '');
  const parts = normalized.includes('->')
    ? normalized.split('->').filter(Boolean)
    : normalized.split('-').filter(Boolean);
  if (parts.length !== 2) return null;
  return `${normalizeLangCode(parts[0]) || parts[0]}->${normalizeLangCode(parts[1]) || parts[1]}`;
}

function normalizeOptionalStringArray(value) {
  if (value === undefined || value === null) return null;
  const list = Array.isArray(value)
    ? value
    : (typeof value === 'string' ? value.split(',') : null);
  if (!Array.isArray(list)) return null;
  const normalized = list
    .map((entry) => normalizeOptionalString(entry))
    .filter(Boolean);
  return normalized.length > 0 ? normalized : null;
}

function normalizeDistillLanguageAllowlist(value) {
  const list = normalizeOptionalStringArray(value);
  if (!list) return null;
  const normalized = list
    .map((entry) => normalizeLangCode(entry))
    .filter(Boolean);
  if (normalized.length === 0) return null;
  return [...new Set(normalized)];
}

function normalizeDistillPairAllowlist(value) {
  const list = normalizeOptionalStringArray(value);
  if (!list) return null;
  const normalized = list
    .map((entry) => normalizePairDirection(entry))
    .filter(Boolean);
  if (normalized.length === 0) return null;
  return [...new Set(normalized)];
}

export function resolveDistillDataScope(options = {}, trainingConfig = null) {
  const distillConfig = trainingConfig?.distill || {};
  const sourceLangs = normalizeDistillLanguageAllowlist(
    options.distillSourceLangs ?? distillConfig.sourceLangs ?? null
  );
  const targetLangs = normalizeDistillLanguageAllowlist(
    options.distillTargetLangs ?? distillConfig.targetLangs ?? null
  );
  const pairAllowlist = normalizeDistillPairAllowlist(
    options.distillPairAllowlist ?? distillConfig.pairAllowlist ?? null
  );
  const strictPairContract = (
    options.strictPairContract === true
    || distillConfig.strictPairContract === true
  );
  return {
    sourceLangs,
    targetLangs,
    pairAllowlist,
    sourceLangSet: sourceLangs ? new Set(sourceLangs) : null,
    targetLangSet: targetLangs ? new Set(targetLangs) : null,
    pairAllowlistSet: pairAllowlist ? new Set(pairAllowlist) : null,
    strictPairContract,
  };
}

function resolveDistillDirection(record) {
  const pairDirection = normalizePairDirection(record?.pair);
  if (pairDirection) return pairDirection;
  const srcLang = normalizeLangCode(record?.src_lang);
  const tgtLang = normalizeLangCode(record?.tgt_lang || record?.lang);
  if (srcLang && tgtLang) {
    return `${srcLang}->${tgtLang}`;
  }
  return null;
}

function resolveStringCandidate(record, keys) {
  for (const key of keys) {
    const value = normalizeOptionalString(record?.[key]);
    if (value) return value;
  }
  return null;
}

export function encodeDistillRow(record, index, scope = null) {
  if (!record || typeof record !== 'object') return null;
  const source = resolveStringCandidate(record, ['source', 'query']);
  const targetPos = resolveStringCandidate(record, ['target_pos', 'target', 'pos']);
  const targetNeg = resolveStringCandidate(record, ['target_neg', 'neg']);
  if (!source || !targetPos) return null;
  const sourceLangRaw = normalizeLangCode(record?.src_lang);
  const targetLangRaw = normalizeLangCode(record?.tgt_lang || record?.lang);
  const pairDirection = normalizePairDirection(record?.pair);
  const sourceTargetDirection = (
    sourceLangRaw && targetLangRaw
      ? `${sourceLangRaw}->${targetLangRaw}`
      : null
  );
  if (scope?.strictPairContract === true) {
    if (!sourceLangRaw || !targetLangRaw) {
      throw new Error('strictPairContract requires src_lang and tgt_lang/lang on each row.');
    }
    if (!pairDirection) {
      throw new Error('strictPairContract requires pair on each row.');
    }
    if (pairDirection !== sourceTargetDirection) {
      throw new Error(`pair "${record?.pair}" does not match src/tgt "${sourceLangRaw}-${targetLangRaw}".`);
    }
  }
  const direction = pairDirection || sourceTargetDirection || resolveDistillDirection(record) || 'unknown';
  const [directionSourceLang, directionTargetLang] = String(direction).split('->');
  const sourceLang = sourceLangRaw || normalizeLangCode(directionSourceLang);
  const targetLang = targetLangRaw || normalizeLangCode(directionTargetLang);
  if (scope?.sourceLangSet && (!sourceLang || !scope.sourceLangSet.has(sourceLang))) {
    return null;
  }
  if (scope?.targetLangSet && (!targetLang || !scope.targetLangSet.has(targetLang))) {
    return null;
  }
  if (scope?.pairAllowlistSet && !scope.pairAllowlistSet.has(direction)) {
    return null;
  }

  return {
    index,
    direction,
    sourceLang: sourceLang || null,
    targetLang: targetLang || null,
    source,
    targetPos,
    targetNeg: targetNeg || null,
  };
}

export function summarizeDirectionCounts(samples) {
  const counts = {};
  for (const sample of samples) {
    const key = sample?.direction || 'unknown';
    counts[key] = (counts[key] || 0) + 1;
  }
  return counts;
}

function resolveLanguageName(langCode) {
  const normalized = normalizeLangCode(langCode);
  if (normalized === 'en') return 'English';
  if (normalized === 'es') return 'Spanish';
  return normalized || 'target';
}

export function buildDistillPrompt(sample) {
  const direction = String(sample?.direction || '').trim();
  const [srcCodeRaw, tgtCodeRaw] = direction.split('->');
  const srcCode = normalizeLangCode(srcCodeRaw) || srcCodeRaw || 'source';
  const tgtCode = normalizeLangCode(tgtCodeRaw) || tgtCodeRaw || 'target';
  const srcName = resolveLanguageName(srcCode);
  const tgtName = resolveLanguageName(tgtCode);
  const source = String(sample?.source || '').trim();
  return `Translate from ${srcName} to ${tgtName}:\n${source}\nTranslation:`;
}

export function buildDistillCandidatePrompt(sample, candidate) {
  const base = buildDistillPrompt(sample);
  const text = String(candidate || '').trim();
  return text ? `${base} ${text}` : base;
}
