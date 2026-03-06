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

  return {
    registryId,
    entries,
    fallbackMappings,
  };
}

export function validateKernelPathContractFacts(facts) {
  const errors = [];
  const checks = [];
  const registryId = String(facts?.registryId ?? 'kernel-path-registry');
  const entries = Array.isArray(facts?.entries) ? facts.entries : [];
  const fallbackMappings = Array.isArray(facts?.fallbackMappings) ? facts.fallbackMappings : [];
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
    },
  };
}
