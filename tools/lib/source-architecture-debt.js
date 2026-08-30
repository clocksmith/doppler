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

export function findArchitecturePolicyRelaxations(currentPolicy, baselinePolicy) {
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
  for (const owner of removedValues(
    currentPolicy?.compatibilityOnlyOwners,
    baselinePolicy?.compatibilityOnlyOwners
  )) {
    relaxations.push(`compatibility-only restriction removed ${owner}`);
  }

  const currentDomains = currentPolicy?.constitutionalDomains ?? {};
  for (const [domain, files] of Object.entries(baselinePolicy?.constitutionalDomains ?? {})) {
    for (const relativePath of removedValues(currentDomains[domain], files)) {
      relaxations.push(`constitutional ${domain} owner removed ${relativePath}`);
    }
  }
  const currentGraphs = new Map(
    (currentPolicy?.constitutionalImportGraphs ?? []).map((rule) => [rule.domain, rule])
  );
  for (const baselineRule of baselinePolicy?.constitutionalImportGraphs ?? []) {
    const currentRule = currentGraphs.get(baselineRule.domain);
    if (!currentRule) {
      relaxations.push(`constitutional import graph removed ${baselineRule.domain}`);
      continue;
    }
    for (const relativePath of removedValues(currentRule.entryPoints, baselineRule.entryPoints)) {
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

export function validateArchitecturePolicyDelta(
  currentPolicy,
  baselinePolicy,
  authorizationDigest = null
) {
  const errors = validateSoftLimitReviewDelta(
    currentPolicy,
    baselinePolicy,
    authorizationDigest
  );
  const relaxations = findArchitecturePolicyRelaxations(currentPolicy, baselinePolicy);
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
