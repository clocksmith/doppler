function normalizeSpecialTokenLabel(value) {
  if (typeof value !== 'string') {
    return '';
  }
  return value.trim().toLowerCase();
}

function isBosLikeLabel(value) {
  const normalized = normalizeSpecialTokenLabel(value);
  return normalized === '<bos>' || normalized === '<s>' || normalized.includes('bos');
}

function isEosLikeLabel(value) {
  const normalized = normalizeSpecialTokenLabel(value);
  return normalized === '<eos>' || normalized === '</s>' || normalized.includes('eos');
}

function classifySpecialTokenReference(ref, specialTokens) {
  if (typeof ref === 'number' && Number.isFinite(ref)) {
    if (specialTokens?.bos === ref) return 'bos';
    if (specialTokens?.eos === ref) return 'eos';
    return null;
  }

  if (Array.isArray(ref)) {
    for (const item of ref) {
      const kind = classifySpecialTokenReference(item, specialTokens);
      if (kind) {
        return kind;
      }
    }
    return null;
  }

  if (ref && typeof ref === 'object') {
    if (typeof ref.id === 'number' && Number.isFinite(ref.id)) {
      return classifySpecialTokenReference(ref.id, specialTokens);
    }
    if (Array.isArray(ref.ids)) {
      return classifySpecialTokenReference(ref.ids, specialTokens);
    }
    if (Array.isArray(ref.tokens)) {
      return classifySpecialTokenReference(ref.tokens, specialTokens);
    }
  }

  if (typeof ref === 'string') {
    if (isBosLikeLabel(ref)) return 'bos';
    if (isEosLikeLabel(ref)) return 'eos';
  }

  return null;
}

function getTemplateBoundaryKind(template, boundary, specialTokens) {
  if (!Array.isArray(template) || template.length === 0) {
    return null;
  }

  const entries = boundary === 'last'
    ? [...template].reverse()
    : template;

  for (const entry of entries) {
    const special = entry?.SpecialToken;
    if (!special) {
      continue;
    }

    const kind = classifySpecialTokenReference(
      special.tokens ?? special.ids ?? special.id ?? null,
      specialTokens
    );
    if (kind) {
      return kind;
    }
  }

  return null;
}

export function inferBundledTokenizerBehaviorFlags(tokenizerJson, specialTokens = null) {
  if (!tokenizerJson || typeof tokenizerJson !== 'object' || Array.isArray(tokenizerJson)) {
    return { addBosToken: null, addEosToken: null };
  }

  const postProcessor = tokenizerJson.post_processor;
  if (!postProcessor || postProcessor.type !== 'TemplateProcessing') {
    return { addBosToken: null, addEosToken: null };
  }

  const single = Array.isArray(postProcessor.single) ? postProcessor.single : null;
  const firstKind = getTemplateBoundaryKind(single, 'first', specialTokens);
  const lastKind = getTemplateBoundaryKind(single, 'last', specialTokens);

  return {
    addBosToken: firstKind === 'bos' ? true : null,
    addEosToken: lastKind === 'eos' ? true : null,
  };
}
