const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/u;

function isAuthorized(review, authorizationDigest) {
  return (
    SHA256_PATTERN.test(authorizationDigest ?? '')
    && review?.debtAuthorization === authorizationDigest
  );
}

export function validateSoftLimitReviewDelta(currentPolicy, baselinePolicy, authorizationDigest = null) {
  const errors = [];
  const currentReviews = currentPolicy?.softLimitReviews ?? {};
  const baselineReviews = baselinePolicy?.softLimitReviews ?? {};
  for (const [relativePath, currentReview] of Object.entries(currentReviews)) {
    const baselineReview = baselineReviews[relativePath];
    const added = baselineReview == null;
    const raised = (
      baselineReview != null
      && Number(currentReview?.reviewedLines) > Number(baselineReview?.reviewedLines)
    );
    if (!added && !raised) continue;
    if (isAuthorized(currentReview, authorizationDigest)) continue;
    const action = added
      ? 'adds a soft-limit review'
      : `raises reviewedLines from ${String(baselineReview.reviewedLines)} to ${String(currentReview.reviewedLines)}`;
    errors.push(
      `${relativePath}: ${action} without a matching DOPPLER_ARCHITECTURE_DEBT_AUTHORIZATION digest`
    );
  }
  return errors;
}

function addedValues(currentValues, baselineValues) {
  const baseline = new Set(baselineValues ?? []);
  return (currentValues ?? []).filter((value) => !baseline.has(value));
}

function removedValues(currentValues, baselineValues) {
  const current = new Set(currentValues ?? []);
  return (baselineValues ?? []).filter((value) => !current.has(value));
}

function hasFreshPolicyAuthorization(currentPolicy, baselinePolicy, authorizationDigest) {
  return (
    SHA256_PATTERN.test(authorizationDigest ?? '')
    && currentPolicy?.debtAuthorization === authorizationDigest
    && currentPolicy.debtAuthorization !== baselinePolicy?.debtAuthorization
  );
}

function constitutionalRenameMaps(currentPolicy, baselinePolicy, errors) {
  const domains = currentPolicy.constitutionalRenames?.domains ?? {};
  const files = currentPolicy.constitutionalRenames?.files ?? {};
  for (const [kind, map] of Object.entries({ domains, files })) {
    const entries = Object.entries(map);
    const targets = entries.map(([, target]) => target);
    const malformed = entries.some(([source, target]) => (
      typeof target !== 'string' || !target || source === target
      || (kind === 'files' && source.slice(0, source.lastIndexOf('/')) !== target.slice(0, target.lastIndexOf('/')))
    ));
    if (malformed || new Set(targets).size !== targets.length) {
      errors.push(`constitutional ${kind} renames must be one-to-one names within the same owner directory`);
      return { domains: {}, files: {} };
    }
  }
  const oldDomains = Object.keys(baselinePolicy.constitutionalDomains ?? {});
  const renamedDomains = oldDomains.map(domain => domains[domain] ?? domain);
  const oldFiles = [...new Set(Object.values(baselinePolicy.constitutionalDomains ?? {}).flat())];
  const renamedFiles = oldFiles.map(file => files[file] ?? file);
  if (new Set(renamedDomains).size !== oldDomains.length || new Set(renamedFiles).size !== oldFiles.length) {
    errors.push('constitutional renames cannot merge existing owners or files');
    return { domains: {}, files: {} };
  }
  return { domains, files };
}

// removedForwarders maps a baseline pure re-export module that no longer
// exists to the modules it forwarded to. Inlining such a module into a facade
// moves edges without widening what the facade reaches.
function baselineFacadeReach(baselinePolicy, facade, removedForwarders = {}) {
  const edges = baselinePolicy.facadeDependencies ?? {};
  const reach = new Set();
  const pending = [...(edges[facade] ?? [])];
  while (pending.length > 0) {
    const dependency = pending.pop();
    if (reach.has(dependency)) continue;
    reach.add(dependency);
    const next = edges[dependency] ?? removedForwarders[dependency];
    if (next) pending.push(...next);
  }
  return [...reach];
}

export function findArchitecturePolicyRelaxations(currentPolicy, baselinePolicy, removedForwarders = {}) {
  const relaxations = [];
  for (const key of ['softLineLimit', 'lineLimit']) {
    if (Number(currentPolicy?.[key]) > Number(baselinePolicy?.[key])) {
      relaxations.push(`${key} increased from ${baselinePolicy[key]} to ${currentPolicy[key]}`);
    }
  }

  const currentLegacy = currentPolicy?.legacyOversize ?? {};
  const baselineLegacy = baselinePolicy?.legacyOversize ?? {};
  for (const [relativePath, review] of Object.entries(currentLegacy)) {
    const baselineReview = baselineLegacy[relativePath];
    if (!baselineReview) relaxations.push(`legacyOversize added ${relativePath}`);
    else if (Number(review?.maxLines) > Number(baselineReview?.maxLines)) {
      relaxations.push(
        `legacyOversize raised ${relativePath} from ${baselineReview.maxLines} to ${review.maxLines}`
      );
    }
  }

  const currentAllowed = currentPolicy?.allowedDependencies ?? {};
  const baselineAllowed = baselinePolicy?.allowedDependencies ?? {};
  for (const [owner, dependencies] of Object.entries(currentAllowed)) {
    for (const dependency of addedValues(dependencies, baselineAllowed[owner])) {
      relaxations.push(`allowed dependency added ${owner}->${dependency}`);
    }
  }

  const exceptionKey = (entry) => `${entry?.from ?? ''}->${entry?.toOwner ?? ''}`;
  for (const key of addedValues(
    (currentPolicy?.dependencyExceptions ?? []).map(exceptionKey),
    (baselinePolicy?.dependencyExceptions ?? []).map(exceptionKey)
  )) {
    relaxations.push(`dependency exception added ${key}`);
  }
  for (const relativePath of addedValues(
    Object.keys(currentPolicy?.experimentalBridges ?? {}),
    Object.keys(baselinePolicy?.experimentalBridges ?? {})
  )) {
    relaxations.push(`experimental bridge added ${relativePath}`);
  }
  for (const relativePath of addedValues(
    Object.keys(currentPolicy?.reachability?.standaloneModules ?? {}),
    Object.keys(baselinePolicy?.reachability?.standaloneModules ?? {})
  )) {
    relaxations.push(`standalone reachability exception added ${relativePath}`);
  }
  for (const relativePath of addedValues(currentPolicy?.facades, baselinePolicy?.facades)) {
    relaxations.push(`facade exception added ${relativePath}`);
  }
  for (const [facade, dependencies] of Object.entries(currentPolicy?.facadeDependencies ?? {})) {
    // A previously unrestricted facade becomes an exact edge inventory. Once
    // inventoried, adding an edge is a policy relaxation like any other import.
    if (!(baselinePolicy?.facades ?? []).includes(facade)) continue;
    if (!baselinePolicy?.facadeDependencies?.[facade]) continue;
    // Inlining a baseline facade that this facade already re-exported moves
    // edges without widening what the facade reaches.
    const reachable = baselineFacadeReach(baselinePolicy, facade, removedForwarders);
    for (const dependency of addedValues(dependencies, reachable)) {
      relaxations.push(`facade dependency added ${facade}->${dependency}`);
    }
  }
  for (const owner of removedValues(
    currentPolicy?.compatibilityOnlyOwners,
    baselinePolicy?.compatibilityOnlyOwners
  )) {
    // Deleting the owner itself (its source directory and top-level entry) is a
    // removal, not a relaxation: nothing remains that could import through it.
    if (currentPolicy?.topLevelOwners && !Object.hasOwn(currentPolicy.topLevelOwners, owner)) continue;
    relaxations.push(`compatibility-only restriction removed ${owner}`);
  }

  const currentDomains = currentPolicy?.constitutionalDomains ?? {};
  // Normalize only names. Dependencies, entry points, and forbidden prefixes
  // are still compared in full; a rename cannot authorize dropping a boundary.
  const renames = constitutionalRenameMaps(currentPolicy, baselinePolicy, relaxations);
  for (const [domain, files] of Object.entries(baselinePolicy?.constitutionalDomains ?? {})) {
    const currentDomain = renames.domains[domain] ?? domain;
    for (const relativePath of removedValues(currentDomains[currentDomain], files.map(file => renames.files[file] ?? file))) {
      relaxations.push(`constitutional ${domain} owner removed ${relativePath}`);
    }
  }
  const currentGraphs = new Map(
    (currentPolicy?.constitutionalImportGraphs ?? []).map((rule) => [rule.domain, rule])
  );
  for (const baselineRule of baselinePolicy?.constitutionalImportGraphs ?? []) {
    const currentRule = currentGraphs.get(renames.domains[baselineRule.domain] ?? baselineRule.domain);
    if (!currentRule) {
      relaxations.push(`constitutional import graph removed ${baselineRule.domain}`);
      continue;
    }
    for (const relativePath of removedValues(currentRule.entryPoints, baselineRule.entryPoints.map(file => renames.files[file] ?? file))) {
      relaxations.push(`constitutional ${baselineRule.domain} entry point removed ${relativePath}`);
    }
    for (const prefix of removedValues(
      currentRule.forbiddenPathPrefixes,
      baselineRule.forbiddenPathPrefixes
    )) {
      relaxations.push(`constitutional ${baselineRule.domain} forbidden prefix removed ${prefix}`);
    }
  }
  return relaxations;
}

// movedPaths maps a baseline source path to the path that module now lives at,
// possibly in another owner directory. The baseline policy is rewritten through
// the map before comparison, so an existing exception, review, facade or edge
// re-registered at its new path is compared as the same entry instead of new
// debt. A move cannot loosen anything: the rewritten entry is still ratcheted
// (reviewedLines, maxLines, facade edges, exceptions), and generic-module
// classifications must match. fileState(path) -> { current, baseline } reports
// whether a path exists in the working tree and at the base revision; entries
// whose move is already in the baseline are inert and may be pruned.
const MOVED_PATH_KEYED_SECTIONS = [
  ['softLimitReviews'],
  ['legacyOversize'],
  ['genericModuleReviews'],
  ['experimentalBridges'],
  ['facadeDependencies'],
  ['reachability', 'standaloneModules'],
];

function readSection(policy, sectionPath) {
  return sectionPath.reduce((value, key) => value?.[key], policy);
}

function baselinePathReferences(baselinePolicy) {
  const references = [];
  for (const sectionPath of MOVED_PATH_KEYED_SECTIONS) {
    references.push(...Object.keys(readSection(baselinePolicy, sectionPath) ?? {}));
  }
  references.push(...Object.values(baselinePolicy?.facadeDependencies ?? {}).flat());
  references.push(...(baselinePolicy?.facades ?? []));
  references.push(...(baselinePolicy?.dependencyExceptions ?? []).map((entry) => entry?.from));
  references.push(...Object.values(baselinePolicy?.constitutionalDomains ?? {}).flat());
  references.push(...(baselinePolicy?.constitutionalImportGraphs ?? []).flatMap((rule) => rule?.entryPoints ?? []));
  return new Set(references.filter((reference) => typeof reference === 'string'));
}

export function resolveMovedPaths(currentPolicy, baselinePolicy, fileState = null) {
  const errors = [];
  const movedPaths = currentPolicy?.movedPaths ?? {};
  if (movedPaths === null || typeof movedPaths !== 'object' || Array.isArray(movedPaths)) {
    return { moves: {}, errors: ['movedPaths must map old source paths to new source paths'] };
  }
  const entries = Object.entries(movedPaths);
  const sources = new Set(entries.map(([source]) => source));
  const targets = entries.map(([, target]) => target);
  const malformed = entries.some(([source, target]) => (
    typeof target !== 'string' || !target.endsWith('.js') || !source.endsWith('.js')
    || source === target || sources.has(target)
  ));
  if (malformed || new Set(targets).size !== targets.length) {
    return { moves: {}, errors: ['movedPaths must be one-to-one .js paths without chains'] };
  }
  const baselineReferences = baselinePathReferences(baselinePolicy);
  const moves = {};
  for (const [source, target] of entries) {
    if (baselineReferences.has(target) && baselineReferences.has(source)) {
      errors.push(`movedPaths ${source} -> ${target} merges into an existing policy entry`);
      continue;
    }
    if (fileState) {
      const sourceState = fileState(source);
      const targetState = fileState(target);
      if (sourceState.current) errors.push(`movedPaths ${source} -> ${target}: ${source} still exists`);
      if (!targetState.current) errors.push(`movedPaths ${source} -> ${target}: ${target} does not exist`);
      if (sourceState.baseline && targetState.baseline) {
        errors.push(`movedPaths ${source} -> ${target}: ${target} already existed at the base revision`);
      }
      if (!sourceState.baseline && !targetState.baseline) {
        errors.push(`movedPaths ${source} -> ${target}: neither path exists at the base revision`);
      }
      if (sourceState.current || !targetState.current || !sourceState.baseline || targetState.baseline) continue;
    }
    moves[source] = target;
  }
  return { moves, errors };
}

function renameKeys(record, moves) {
  if (record == null) return record;
  return Object.fromEntries(Object.entries(record).map(([key, value]) => [moves[key] ?? key, value]));
}

export function applyMovedPaths(baselinePolicy, moves) {
  if (Object.keys(moves).length === 0) return baselinePolicy;
  const movePath = (value) => moves[value] ?? value;
  const policy = structuredClone(baselinePolicy);
  for (const sectionPath of MOVED_PATH_KEYED_SECTIONS) {
    const parent = sectionPath.slice(0, -1).reduce((value, key) => value?.[key], policy);
    const key = sectionPath[sectionPath.length - 1];
    if (parent?.[key]) parent[key] = renameKeys(parent[key], moves);
  }
  if (policy.facadeDependencies) {
    for (const [facade, dependencies] of Object.entries(policy.facadeDependencies)) {
      policy.facadeDependencies[facade] = dependencies.map(movePath);
    }
  }
  if (policy.facades) policy.facades = policy.facades.map(movePath);
  for (const entry of policy.dependencyExceptions ?? []) entry.from = movePath(entry.from);
  for (const [domain, files] of Object.entries(policy.constitutionalDomains ?? {})) {
    policy.constitutionalDomains[domain] = files.map(movePath);
  }
  for (const rule of policy.constitutionalImportGraphs ?? []) {
    rule.entryPoints = (rule.entryPoints ?? []).map(movePath);
  }
  return policy;
}

function movedForwarders(removedForwarders, moves) {
  return Object.fromEntries(Object.entries(removedForwarders)
    .filter(([relativePath]) => !Object.hasOwn(moves, relativePath))
    .map(([relativePath, targets]) => [relativePath, targets.map((target) => moves[target] ?? target)]));
}

function findMovedGenericReviewChanges(currentPolicy, baselinePolicy, moves) {
  const changes = [];
  for (const [source, target] of Object.entries(moves)) {
    const baselineReview = baselinePolicy?.genericModuleReviews?.[source];
    if (!baselineReview) continue;
    const currentReview = currentPolicy?.genericModuleReviews?.[target];
    if (currentReview?.classification !== baselineReview.classification) {
      changes.push(
        `moved generic module review ${source} -> ${target} changes classification from `
        + `${String(baselineReview.classification)} to ${String(currentReview?.classification)}`
      );
    }
  }
  return changes;
}

export function validateArchitecturePolicyDelta(
  currentPolicy,
  rawBaselinePolicy,
  authorizationDigest = null,
  rawRemovedForwarders = {},
  fileState = null
) {
  const moved = resolveMovedPaths(currentPolicy, rawBaselinePolicy, fileState);
  const baselinePolicy = applyMovedPaths(rawBaselinePolicy, moved.moves);
  const removedForwarders = movedForwarders(rawRemovedForwarders, moved.moves);
  const errors = [
    ...moved.errors,
    ...validateSoftLimitReviewDelta(currentPolicy, baselinePolicy, authorizationDigest),
  ];
  const relaxations = [
    ...findMovedGenericReviewChanges(currentPolicy, rawBaselinePolicy, moved.moves),
    ...findArchitecturePolicyRelaxations(currentPolicy, baselinePolicy, removedForwarders),
  ];
  if (relaxations.length === 0) return errors;
  if (hasFreshPolicyAuthorization(currentPolicy, baselinePolicy, authorizationDigest)) return errors;
  for (const relaxation of relaxations) {
    errors.push(
      `${relaxation} without a fresh policy debtAuthorization matching `
      + 'DOPPLER_ARCHITECTURE_DEBT_AUTHORIZATION'
    );
  }
  return errors;
}
