import { selectByRules } from '../gpu/kernels/rule-matcher.js';

const SUPPORTED_DTYPES = new Set(['f16', 'f32']);
const DTYPE_RANK = Object.freeze({
  f16: 1,
  f32: 2,
});
const DTYPE_BYTES = Object.freeze({
  f16: 2,
  f32: 4,
});

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeKernelPathContractDtype(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (!SUPPORTED_DTYPES.has(normalized)) {
    throw new Error(
      `kernel-path contract: ${label} must be one of ${[...SUPPORTED_DTYPES].join(', ')}.`
    );
  }
  return normalized;
}

function normalizeRegistryEntry(entry, index) {
  if (!isPlainObject(entry)) {
    throw new Error(`kernel-path contract: entries[${index}] must be an object.`);
  }
  const id = String(entry.id ?? '').trim();
  if (!id) {
    throw new Error(`kernel-path contract: entries[${index}].id is required.`);
  }
  const aliasOf = typeof entry.aliasOf === 'string' && entry.aliasOf.trim() !== ''
    ? entry.aliasOf.trim()
    : null;
  const hasFile = typeof entry.file === 'string' && entry.file.trim() !== '';
  if (aliasOf && hasFile) {
    throw new Error(
      `kernel-path contract: entries[${index}] must not include both file and aliasOf.`
    );
  }
  if (!aliasOf && !hasFile) {
    throw new Error(
      `kernel-path contract: entries[${index}] must include file or aliasOf.`
    );
  }
  return {
    id,
    aliasOf,
    hasFile,
  };
}

function normalizeFallbackMapping(mapping, index) {
  if (!isPlainObject(mapping)) {
    throw new Error(`kernel-path contract: fallbackMappings[${index}] must be an object.`);
  }
  const primaryKernelPathId = String(mapping.primaryKernelPathId ?? '').trim();
  const fallbackKernelPathId = String(mapping.fallbackKernelPathId ?? '').trim();
  if (!primaryKernelPathId) {
    throw new Error(`kernel-path contract: fallbackMappings[${index}].primaryKernelPathId is required.`);
  }
  if (!fallbackKernelPathId) {
    throw new Error(`kernel-path contract: fallbackMappings[${index}].fallbackKernelPathId is required.`);
  }
  return {
    primaryKernelPathId,
    fallbackKernelPathId,
    primaryActivationDtype: normalizeKernelPathContractDtype(
      mapping.primaryActivationDtype,
      `fallbackMappings[${index}].primaryActivationDtype`
    ),
    fallbackActivationDtype: mapping.fallbackActivationDtype == null
      ? null
      : normalizeKernelPathContractDtype(
        mapping.fallbackActivationDtype,
        `fallbackMappings[${index}].fallbackActivationDtype`
      ),
  };
}

function normalizeFallbackRule(rule, index) {
  if (!isPlainObject(rule)) {
    throw new Error(`kernel-path contract: fallbackRules[${index}] must be an object.`);
  }
  const match = isPlainObject(rule.match) ? rule.match : {};
  const rawKernelPathId = match.kernelPathId;
  const matchKernelPathId = typeof rawKernelPathId === 'string' && rawKernelPathId.trim() !== ''
    ? rawKernelPathId.trim()
    : null;
  const value = rule.value == null ? null : String(rule.value).trim();
  if (value === '') {
    throw new Error(`kernel-path contract: fallbackRules[${index}].value must be null or a non-empty string.`);
  }
  return {
    matchKernelPathId,
    value,
    isDefault: Object.keys(match).length === 0,
  };
}

function normalizeAutoSelectRule(rule, index) {
  if (!isPlainObject(rule)) {
    throw new Error(`kernel-path contract: autoSelectRules[${index}] must be an object.`);
  }
  const match = isPlainObject(rule.match) ? rule.match : {};
  const rawKernelPathRef = match.kernelPathRef;
  const matchKernelPathRef = typeof rawKernelPathRef === 'string' && rawKernelPathRef.trim() !== ''
    ? rawKernelPathRef.trim()
    : null;
  const allowCapabilityAutoSelection = typeof match.allowCapabilityAutoSelection === 'boolean'
    ? match.allowCapabilityAutoSelection
    : null;
  const hasSubgroups = typeof match.hasSubgroups === 'boolean'
    ? match.hasSubgroups
    : null;
  const value = rule.value;
  if (typeof value === 'string' && value.trim() !== '') {
    return {
      matchKernelPathRef,
      allowCapabilityAutoSelection,
      hasSubgroups,
      valueKind: 'string',
      value: value.trim(),
      isDefault: Object.keys(match).length === 0,
    };
  }
  if (isPlainObject(value) && Object.keys(value).length === 1 && typeof value.context === 'string') {
    return {
      matchKernelPathRef,
      allowCapabilityAutoSelection,
      hasSubgroups,
      valueKind: 'context',
      value: value.context.trim(),
      isDefault: Object.keys(match).length === 0,
    };
  }
  throw new Error(
    `kernel-path contract: autoSelectRules[${index}].value must be a non-empty string ` +
    'or a { context: ... } directive.'
  );
}

function findAliasCycles(entriesById) {
  const visited = new Set();
  const visiting = new Set();
  const stack = [];
  const cycles = [];

  function walk(id) {
    if (visited.has(id)) {
      return;
    }
    if (visiting.has(id)) {
      const cycleStart = stack.indexOf(id);
      const cycle = cycleStart >= 0 ? [...stack.slice(cycleStart), id] : [id, id];
      cycles.push(cycle);
      return;
    }
    visiting.add(id);
    stack.push(id);
    const nextId = entriesById.get(id)?.aliasOf ?? null;
    if (nextId) {
      walk(nextId);
    }
    stack.pop();
    visiting.delete(id);
    visited.add(id);
  }

  for (const id of entriesById.keys()) {
    walk(id);
  }

  return cycles;
}

export function extractKernelPathContractFacts(input, options = {}) {
  const registryId = String(options.registryId ?? input?.registryId ?? 'kernel-path-registry').trim()
    || 'kernel-path-registry';
  const rawEntries = Array.isArray(input)
    ? input
    : Array.isArray(input?.entries)
      ? input.entries
      : null;
  if (!rawEntries) {
    throw new Error('kernel-path contract: entries must be an array or an object with entries.');
  }

  const entries = rawEntries.map(normalizeRegistryEntry);
  const seenIds = new Set();
  for (const entry of entries) {
    if (seenIds.has(entry.id)) {
      throw new Error(`kernel-path contract: duplicate registry entry id "${entry.id}".`);
    }
    seenIds.add(entry.id);
  }

  const rawFallbackMappings = Array.isArray(input?.fallbackMappings) ? input.fallbackMappings : [];
  const fallbackMappings = rawFallbackMappings.map(normalizeFallbackMapping);
  const rawFallbackRules = Array.isArray(input?.fallbackRules) ? input.fallbackRules : [];
  const fallbackRules = rawFallbackRules.map(normalizeFallbackRule);
  const rawAutoSelectRules = Array.isArray(input?.autoSelectRules) ? input.autoSelectRules : [];
  const autoSelectRules = rawAutoSelectRules.map(normalizeAutoSelectRule);

  return {
    registryId,
    entries,
    fallbackMappings,
    fallbackRules,
    autoSelectRules,
  };
}

export function validateKernelPathContractFacts(facts) {
  const errors = [];
  const checks = [];
  const registryId = String(facts?.registryId ?? 'kernel-path-registry');
  const entries = Array.isArray(facts?.entries) ? facts.entries : [];
  const fallbackMappings = Array.isArray(facts?.fallbackMappings) ? facts.fallbackMappings : [];
  const fallbackRules = Array.isArray(facts?.fallbackRules) ? facts.fallbackRules : [];
  const autoSelectRules = Array.isArray(facts?.autoSelectRules) ? facts.autoSelectRules : [];
  const entriesById = new Map(entries.map((entry) => [entry.id, entry]));

  const missingAliasTargets = entries.filter((entry) => entry.aliasOf && !entriesById.has(entry.aliasOf));
  for (const entry of missingAliasTargets) {
    errors.push(
      `[KernelPathContract] registry entry "${entry.id}" aliases missing target "${entry.aliasOf}".`
    );
  }
  checks.push({
    id: `${registryId}.aliasTargets`,
    ok: missingAliasTargets.length === 0,
  });

  const aliasCycles = findAliasCycles(entriesById);
  for (const cycle of aliasCycles) {
    errors.push(`[KernelPathContract] alias cycle detected: ${cycle.join(' -> ')}.`);
  }
  checks.push({
    id: `${registryId}.aliasCycles`,
    ok: aliasCycles.length === 0,
  });

  let fallbackTargetErrors = 0;
  let fallbackDtypeErrors = 0;
  for (const mapping of fallbackMappings) {
    if (!entriesById.has(mapping.primaryKernelPathId)) {
      fallbackTargetErrors += 1;
      errors.push(
        `[KernelPathContract] finiteness fallback mapping references unknown primary kernel path ` +
        `"${mapping.primaryKernelPathId}".`
      );
      continue;
    }
    if (!entriesById.has(mapping.fallbackKernelPathId) || mapping.fallbackActivationDtype == null) {
      fallbackTargetErrors += 1;
      errors.push(
        `[KernelPathContract] finiteness fallback mapping references unknown fallback kernel path ` +
        `"${mapping.fallbackKernelPathId}" for "${mapping.primaryKernelPathId}".`
      );
      continue;
    }
    if (DTYPE_RANK[mapping.fallbackActivationDtype] < DTYPE_RANK[mapping.primaryActivationDtype]) {
      fallbackDtypeErrors += 1;
      errors.push(
        `[KernelPathContract] finiteness fallback "${mapping.primaryKernelPathId}" -> ` +
        `"${mapping.fallbackKernelPathId}" narrows activation dtype ` +
        `${mapping.primaryActivationDtype} -> ${mapping.fallbackActivationDtype}.`
      );
      continue;
    }
    if (DTYPE_BYTES[mapping.fallbackActivationDtype] < DTYPE_BYTES[mapping.primaryActivationDtype]) {
      fallbackDtypeErrors += 1;
      errors.push(
        `[KernelPathContract] finiteness fallback "${mapping.primaryKernelPathId}" -> ` +
        `"${mapping.fallbackKernelPathId}" reduces bytes per element ` +
        `${mapping.primaryActivationDtype} -> ${mapping.fallbackActivationDtype}.`
      );
    }
  }

  checks.push({
    id: `${registryId}.finitenessFallbackTargets`,
    ok: fallbackTargetErrors === 0,
  });
  checks.push({
    id: `${registryId}.finitenessFallbackDtypes`,
    ok: fallbackDtypeErrors === 0,
  });

  let fallbackRuleShapeErrors = 0;
  let fallbackRuleCoverageErrors = 0;
  let fallbackRuleTargetErrors = 0;
  if (fallbackRules.length > 0) {
    const defaultRules = fallbackRules.filter((rule) => rule.isDefault);
    if (defaultRules.length !== 1 || defaultRules[0].value !== null || fallbackRules[fallbackRules.length - 1] !== defaultRules[0]) {
      fallbackRuleShapeErrors += 1;
      errors.push(
        '[KernelPathContract] finiteness fallback rules must end with exactly one default `{ match: {}, value: null }` rule.'
      );
    }
    const seenRuleIds = new Set();
    for (const rule of fallbackRules) {
      if (rule.isDefault) continue;
      if (!rule.matchKernelPathId) {
        fallbackRuleShapeErrors += 1;
        errors.push('[KernelPathContract] non-default finiteness fallback rules must match on kernelPathId.');
        continue;
      }
      if (seenRuleIds.has(rule.matchKernelPathId)) {
        fallbackRuleShapeErrors += 1;
        errors.push(
          `[KernelPathContract] duplicate finiteness fallback rule for "${rule.matchKernelPathId}".`
        );
      }
      seenRuleIds.add(rule.matchKernelPathId);
      if (!entriesById.has(rule.matchKernelPathId)) {
        fallbackRuleTargetErrors += 1;
        errors.push(
          `[KernelPathContract] finiteness fallback rule references unknown primary kernel path "${rule.matchKernelPathId}".`
        );
      }
      if (rule.value != null && !entriesById.has(rule.value)) {
        fallbackRuleTargetErrors += 1;
        errors.push(
          `[KernelPathContract] finiteness fallback rule references unknown fallback kernel path "${rule.value}" for "${rule.matchKernelPathId}".`
        );
      }
    }
    for (const entry of entries) {
      const selected = selectByRules(
        fallbackRules.map((rule) => ({
          match: rule.isDefault ? {} : { kernelPathId: rule.matchKernelPathId },
          value: rule.value,
        })),
        { kernelPathId: entry.id }
      );
      if (!(selected === null || (typeof selected === 'string' && selected.length > 0))) {
        fallbackRuleCoverageErrors += 1;
        errors.push(
          `[KernelPathContract] finiteness fallback rules did not yield a valid result for "${entry.id}".`
        );
      }
    }
  }
  checks.push({
    id: `${registryId}.finitenessFallbackRuleShape`,
    ok: fallbackRuleShapeErrors === 0,
  });
  checks.push({
    id: `${registryId}.finitenessFallbackRuleTargets`,
    ok: fallbackRuleTargetErrors === 0,
  });
  checks.push({
    id: `${registryId}.finitenessFallbackRuleCoverage`,
    ok: fallbackRuleCoverageErrors === 0,
  });

  let autoSelectShapeErrors = 0;
  let autoSelectTargetErrors = 0;
  let autoSelectCoverageErrors = 0;
  if (autoSelectRules.length > 0) {
    const defaultRules = autoSelectRules.filter((rule) => rule.isDefault);
    if (
      defaultRules.length !== 1
      || defaultRules[0].valueKind !== 'context'
      || defaultRules[0].value !== 'kernelPathRef'
      || autoSelectRules[autoSelectRules.length - 1] !== defaultRules[0]
    ) {
      autoSelectShapeErrors += 1;
      errors.push(
        '[KernelPathContract] autoSelect rules must end with exactly one default `{ match: {}, value: { context: "kernelPathRef" } }` rule.'
      );
    }
    for (const rule of autoSelectRules) {
      if (rule.isDefault) continue;
      if (rule.allowCapabilityAutoSelection !== true) {
        autoSelectShapeErrors += 1;
        errors.push('[KernelPathContract] non-default autoSelect rules must require allowCapabilityAutoSelection=true.');
      }
      if (!rule.matchKernelPathRef) {
        autoSelectShapeErrors += 1;
        errors.push('[KernelPathContract] non-default autoSelect rules must match on kernelPathRef.');
      }
      if (rule.hasSubgroups == null) {
        autoSelectShapeErrors += 1;
        errors.push('[KernelPathContract] non-default autoSelect rules must match on hasSubgroups.');
      }
      if (rule.valueKind === 'context') {
        autoSelectShapeErrors += 1;
        errors.push('[KernelPathContract] only the default autoSelect rule may use a context directive.');
      }
      if (rule.matchKernelPathRef && !entriesById.has(rule.matchKernelPathRef)) {
        autoSelectTargetErrors += 1;
        errors.push(
          `[KernelPathContract] autoSelect rule references unknown kernelPathRef "${rule.matchKernelPathRef}".`
        );
      }
      if (rule.valueKind === 'string' && !entriesById.has(rule.value)) {
        autoSelectTargetErrors += 1;
        errors.push(
          `[KernelPathContract] autoSelect rule remaps to unknown kernel path "${rule.value}".`
        );
      }
    }
    const resolvedAutoSelectRules = autoSelectRules.map((rule) => ({
      match: rule.isDefault
        ? {}
        : {
            allowCapabilityAutoSelection: rule.allowCapabilityAutoSelection,
            hasSubgroups: rule.hasSubgroups,
            kernelPathRef: rule.matchKernelPathRef,
          },
      value: rule.valueKind === 'context'
        ? { context: rule.value }
        : rule.value,
    }));
    for (const entry of entries) {
      for (const allowCapabilityAutoSelection of [true, false]) {
        for (const hasSubgroups of [true, false]) {
          const selected = selectByRules(resolvedAutoSelectRules, {
            kernelPathRef: entry.id,
            allowCapabilityAutoSelection,
            hasSubgroups,
          });
          const resolved = isPlainObject(selected) && selected.context === 'kernelPathRef'
            ? entry.id
            : selected;
          if (typeof resolved !== 'string' || !resolved.length || !entriesById.has(resolved)) {
            autoSelectCoverageErrors += 1;
            errors.push(
              `[KernelPathContract] autoSelect rules did not yield a valid kernel path for ` +
              `"${entry.id}" (allowCapabilityAutoSelection=${allowCapabilityAutoSelection}, hasSubgroups=${hasSubgroups}).`
            );
            break;
          }
        }
      }
    }
  }
  checks.push({
    id: `${registryId}.autoSelectRuleShape`,
    ok: autoSelectShapeErrors === 0,
  });
  checks.push({
    id: `${registryId}.autoSelectRuleTargets`,
    ok: autoSelectTargetErrors === 0,
  });
  checks.push({
    id: `${registryId}.autoSelectRuleCoverage`,
    ok: autoSelectCoverageErrors === 0,
  });

  return {
    ok: errors.length === 0,
    errors,
    checks,
  };
}

export function buildKernelPathContractArtifact(input, options = {}) {
  const facts = extractKernelPathContractFacts(input, options);
  const evaluation = validateKernelPathContractFacts(facts);
  const aliasEntries = facts.entries.filter((entry) => entry.aliasOf != null).length;

  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: evaluation.ok,
    checks: evaluation.checks,
    errors: evaluation.errors,
    stats: {
      totalEntries: facts.entries.length,
      aliasEntries,
      canonicalEntries: facts.entries.length - aliasEntries,
      fallbackMappings: facts.fallbackMappings.length,
      fallbackRules: facts.fallbackRules?.length ?? 0,
      autoSelectRules: facts.autoSelectRules?.length ?? 0,
    },
  };
}
