import { sha256Hex } from '../../utils/sha256.js';

export const WGSL_REPAIR_TASK_CONTRACT = 'replacement_only_wgsl_span_v1';

export const WGSL_REPAIR_MUTATION_OPERATORS = Object.freeze([
  'address_space',
  'attribute_name',
  'builtin_name',
  'declared_type',
  'function_name',
  'identifier_reference',
  'missing_semicolon',
  'numeric_literal_suffix',
  'storage_access',
]);

export const VERIFIER_GUIDED_ARTIFACT_TYPES = Object.freeze([
  'training_rollout_group',
  'training_reward_vector',
  'training_policy_update',
  'training_verifier_report',
  'training_policy_checkpoint',
  'training_promotion_decision',
]);

const BLOCKING_REWARD_IDS = Object.freeze([
  'contract_pass',
  'policy_pass',
]);

function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function requireObject(value, label) {
  if (!isObject(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function requireString(value, label) {
  const normalized = typeof value === 'string' ? value.trim() : '';
  if (!normalized) {
    throw new Error(`${label} is required.`);
  }
  return normalized;
}

function requireInteger(value, label, minimum = 0) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum) {
    throw new Error(`${label} must be an integer >= ${minimum}.`);
  }
  return parsed;
}

function requireFinite(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be finite.`);
  }
  return parsed;
}

function requireArray(value, label, minimumLength = 0) {
  if (!Array.isArray(value) || value.length < minimumLength) {
    throw new Error(`${label} must be an array with at least ${minimumLength} entries.`);
  }
  return value;
}

function requireHash(value, label) {
  const normalized = requireString(value, label);
  if (!/^[a-f0-9]{64}$/.test(normalized)) {
    throw new Error(`${label} must be a SHA-256 hex digest.`);
  }
  return normalized;
}

function stableJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableJson(entry)).join(',')}]`;
  }
  if (isObject(value)) {
    return `{${Object.keys(value).sort().map((key) => (
      `${JSON.stringify(key)}:${stableJson(value[key])}`
    )).join(',')}}`;
  }
  return JSON.stringify(value);
}

function replaceAt(source, start, end, replacement) {
  return `${source.slice(0, start)}${replacement}${source.slice(end)}`;
}

function lineBounds(source, offset) {
  const start = source.lastIndexOf('\n', Math.max(0, offset - 1)) + 1;
  const next = source.indexOf('\n', offset);
  const end = next < 0 ? source.length : next;
  return { start, end };
}

function mutationFromMatch(source, operator, match, replacement, detail) {
  const start = match.index;
  const end = start + match[0].length;
  const span = lineBounds(source, start);
  const originalSpan = source.slice(span.start, span.end);
  const localStart = start - span.start;
  const localEnd = end - span.start;
  const mutatedSpan = replaceAt(originalSpan, localStart, localEnd, replacement);
  return {
    operator,
    detail,
    spanStart: span.start,
    spanEnd: span.end,
    originalSpan,
    mutatedSpan,
    mutatedSource: replaceAt(source, span.start, span.end, mutatedSpan),
  };
}

function firstMatch(source, pattern, predicate = null) {
  for (const match of source.matchAll(pattern)) {
    if (!predicate || predicate(match)) return match;
  }
  return null;
}

function buildAttributeMutation(source) {
  const match = firstMatch(source, /@(binding|group|compute|workgroup_size)\b/g);
  if (!match) return null;
  const replacement = `@${match[1]}_broken`;
  return mutationFromMatch(source, 'attribute_name', match, replacement, match[1]);
}

function buildAddressSpaceMutation(source) {
  const match = firstMatch(source, /\b(uniform|storage|workgroup|private)\b/g);
  if (!match) return null;
  return mutationFromMatch(source, 'address_space', match, `${match[1]}_broken`, match[1]);
}

function buildBuiltinMutation(source) {
  const match = firstMatch(source, /@builtin\(([^)]+)\)/g);
  if (!match) return null;
  return mutationFromMatch(
    source,
    'builtin_name',
    match,
    `@builtin(${match[1]}_broken)`,
    match[1]
  );
}

function buildTypeMutation(source) {
  const match = firstMatch(source, /\b(u32|i32|f32|f16)\b/g);
  if (!match) return null;
  return mutationFromMatch(source, 'declared_type', match, `${match[1]}_broken`, match[1]);
}

function buildFunctionMutation(source) {
  const match = firstMatch(
    source,
    /\b(workgroupBarrier|storageBarrier|arrayLength|select|clamp|min|max|dot)\b/g
  );
  if (!match) return null;
  return mutationFromMatch(source, 'function_name', match, `${match[1]}_broken`, match[1]);
}

function buildIdentifierMutation(source) {
  const declarationPattern = /\b(?:let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\b/g;
  for (const declaration of source.matchAll(declarationPattern)) {
    const name = declaration[1];
    if (name.startsWith('_')) continue;
    const afterDeclaration = declaration.index + declaration[0].length;
    const usePattern = new RegExp(`\\b${name}\\b`, 'g');
    usePattern.lastIndex = afterDeclaration;
    const use = usePattern.exec(source);
    if (!use) continue;
    return mutationFromMatch(
      source,
      'identifier_reference',
      use,
      `${name}_broken`,
      name
    );
  }
  return null;
}

function buildSemicolonMutation(source) {
  const match = firstMatch(
    source,
    /;(?=[ \t]*(?:\/\/[^\n]*)?$)/gm,
    (candidate) => {
      const span = lineBounds(source, candidate.index);
      const line = source.slice(span.start, span.end).trim();
      return line.length > 1 && !line.startsWith('//');
    }
  );
  if (!match) return null;
  return mutationFromMatch(source, 'missing_semicolon', match, '', 'semicolon');
}

function buildNumericLiteralMutation(source) {
  const match = firstMatch(source, /\b\d+u\b/g);
  if (!match) return null;
  return mutationFromMatch(
    source,
    'numeric_literal_suffix',
    match,
    `${match[0].slice(0, -1)}z`,
    match[0]
  );
}

function buildStorageAccessMutation(source) {
  const match = firstMatch(source, /(?<=var<storage,\s*)(read_write|read)(?=>)/g);
  if (!match) return null;
  return mutationFromMatch(source, 'storage_access', match, `${match[1]}_broken`, match[1]);
}

const MUTATION_BUILDERS = Object.freeze({
  address_space: buildAddressSpaceMutation,
  attribute_name: buildAttributeMutation,
  builtin_name: buildBuiltinMutation,
  declared_type: buildTypeMutation,
  function_name: buildFunctionMutation,
  identifier_reference: buildIdentifierMutation,
  missing_semicolon: buildSemicolonMutation,
  numeric_literal_suffix: buildNumericLiteralMutation,
  storage_access: buildStorageAccessMutation,
});

export function deriveKernelFamily(sourceId, sourcePath) {
  const basename = String(sourcePath || '').split('/').pop()?.replace(/\.wgsl$/i, '') || 'unknown';
  const normalized = basename
    .replace(/^\d+_/, '')
    .replace(/_kernel(?:_\d+)?$/, '')
    .replace(/_(?:f16|f32|f16kv|vec\d+|tiled|dynamic|batched|subgroup|shared|prod)$/g, '')
    .replace(/_(?:f16|f32|f16kv|vec\d+|tiled|dynamic|batched|subgroup|shared|prod)$/g, '')
    .replace(/[^a-z0-9]+/gi, '_')
    .replace(/^_+|_+$/g, '')
    .toLowerCase();
  return `${String(sourceId || 'unknown').toLowerCase()}:${normalized || 'unknown'}`;
}

export function createWgslRepairMutations(source, operators = WGSL_REPAIR_MUTATION_OPERATORS) {
  const code = String(source || '');
  if (!code.trim()) {
    throw new Error('WGSL source must not be empty.');
  }
  const mutations = [];
  for (const operator of operators) {
    const builder = MUTATION_BUILDERS[operator];
    if (!builder) {
      throw new Error(`Unknown WGSL mutation operator: ${operator}`);
    }
    const mutation = builder(code);
    if (mutation && mutation.mutatedSource !== code) {
      mutations.push(mutation);
    }
  }
  return mutations;
}

export function buildWgslRepairTask(sourceRecord, mutation) {
  requireObject(sourceRecord, 'sourceRecord');
  requireObject(mutation, 'mutation');
  const sourceId = requireString(sourceRecord.sourceId, 'sourceRecord.sourceId');
  const sourcePath = requireString(sourceRecord.sourcePath, 'sourceRecord.sourcePath');
  const revision = requireString(sourceRecord.revision, 'sourceRecord.revision');
  const license = requireString(sourceRecord.license, 'sourceRecord.license');
  const source = requireString(sourceRecord.source, 'sourceRecord.source');
  const operator = requireString(mutation.operator, 'mutation.operator');
  const kernelFamilyId = sourceRecord.kernelFamilyId
    || deriveKernelFamily(sourceId, sourcePath);
  const taskIdentity = {
    sourceId,
    sourcePath,
    revision,
    sourceSha256: sha256Hex(source),
    operator,
    spanStart: mutation.spanStart,
    spanEnd: mutation.spanEnd,
    mutatedSpan: mutation.mutatedSpan,
  };
  const taskId = `wgsl-${sha256Hex(stableJson(taskIdentity)).slice(0, 20)}`;
  const contextStart = Math.max(0, mutation.spanStart - 600);
  const contextEnd = Math.min(source.length, mutation.spanEnd + 600);
  const contextPrefix = source.slice(contextStart, mutation.spanStart);
  const contextSuffix = source.slice(mutation.spanEnd, contextEnd);
  const prompt = [
    'Repair one harness-owned WGSL span.',
    'Return only the replacement WGSL for <broken_span>; no Markdown fence, diff, or explanation.',
    `Source: ${sourceId}/${sourcePath}@${revision}`,
    `Mutation class: ${operator}`,
    '<context_before>',
    contextPrefix,
    '</context_before>',
    '<broken_span>',
    mutation.mutatedSpan,
    '</broken_span>',
    '<context_after>',
    contextSuffix,
    '</context_after>',
  ].join('\n');
  return {
    schemaVersion: 1,
    taskContract: WGSL_REPAIR_TASK_CONTRACT,
    id: taskId,
    rowId: taskId,
    taskId,
    kernelFamilyId,
    sourceId,
    sourcePath,
    sourceRevision: revision,
    sourceLicense: license,
    sourceSha256: taskIdentity.sourceSha256,
    mutation: {
      operator,
      detail: mutation.detail || null,
      mutatedSourceSha256: sha256Hex(mutation.mutatedSource),
    },
    span: {
      start: mutation.spanStart,
      end: mutation.spanEnd,
      broken: mutation.mutatedSpan,
      reference: mutation.originalSpan,
    },
    prompt,
    completion: mutation.originalSpan,
    source,
    mutatedSource: mutation.mutatedSource,
  };
}

export function parseReplacementOnlyResponse(response) {
  if (typeof response !== 'string') {
    return { ok: false, replacement: '', violations: ['response_not_string'] };
  }
  const violations = [];
  if (!response.trim()) violations.push('empty_response');
  if (/```/.test(response)) violations.push('markdown_fence');
  if (/^\s*(?:here(?:'s| is)|explanation:|patch:|diff --git)/i.test(response)) {
    violations.push('prose_or_diff_wrapper');
  }
  return {
    ok: violations.length === 0,
    replacement: response,
    violations,
  };
}

export function applyWgslRepairResponse(task, response) {
  requireObject(task, 'task');
  const parsed = parseReplacementOnlyResponse(response);
  const source = requireString(task.source, 'task.source');
  const start = requireInteger(task.span?.start, 'task.span.start');
  const end = requireInteger(task.span?.end, 'task.span.end');
  if (end < start || end > source.length) {
    throw new Error('task.span must identify a valid source range.');
  }
  return {
    ...parsed,
    candidateSource: replaceAt(source, start, end, parsed.replacement),
    candidateSha256: sha256Hex(replaceAt(source, start, end, parsed.replacement)),
  };
}

function rewardComponent(id, role, rawValue, normalizedValue, evidence) {
  return {
    id,
    schemaVersion: 1,
    type: 'deterministic',
    role,
    rawValue,
    normalizedValue,
    weight: 1,
    reduction: role === 'blocking' ? 'lexicographic_block' : 'sum',
    evidence,
  };
}

export function buildWgslRewardVector(input) {
  requireObject(input, 'reward input');
  const taskId = requireString(input.taskId, 'reward input.taskId');
  const contractPass = input.contractPass === true;
  const policyPass = input.policyPass === true;
  const compilePass = input.compilePass === true;
  const regressionPass = input.regressionPass === true;
  const exactReferenceMatch = input.exactReferenceMatch === true;
  const components = [
    rewardComponent('contract_pass', 'blocking', contractPass, contractPass ? 1 : 0, input.contractEvidence || null),
    rewardComponent('policy_pass', 'blocking', policyPass, policyPass ? 1 : 0, input.policyEvidence || null),
    rewardComponent('compile_pass', 'promotion', compilePass, compilePass ? 1 : 0, input.compileEvidence || null),
    rewardComponent('regression_pass', 'promotion', regressionPass, regressionPass ? 1 : 0, input.regressionEvidence || null),
    rewardComponent(
      'exact_reference_match',
      'supporting',
      exactReferenceMatch,
      exactReferenceMatch ? 1 : 0,
      input.referenceEvidence || null
    ),
  ];
  const blocked = components.some((component) => (
    component.role === 'blocking' && component.normalizedValue !== 1
  ));
  const scalarReward = blocked
    ? -1
    : compilePass && regressionPass
      ? 1 + (exactReferenceMatch ? 0.05 : 0)
      : 0;
  return {
    artifactType: 'training_reward_vector',
    schemaVersion: 1,
    taskId,
    sampleId: requireString(input.sampleId, 'reward input.sampleId'),
    verifierBundleHash: requireHash(input.verifierBundleHash, 'reward input.verifierBundleHash'),
    components,
    reduction: {
      id: 'wgsl_lexicographic_compile_regression_v1',
      blockingComponentIds: BLOCKING_REWARD_IDS,
      formula: 'blocked ? -1 : (compile_pass && regression_pass ? 1 + 0.05 * exact_reference_match : 0)',
      scalarReward,
      blocked,
    },
    claimBoundary: requireString(input.claimBoundary, 'reward input.claimBoundary'),
  };
}

export function computeGroupRelativeAdvantages(rewards, epsilon = 1e-6) {
  const values = requireArray(rewards, 'rewards', 2).map((value, index) => (
    requireFinite(value, `rewards[${index}]`)
  ));
  const denominatorFloor = requireFinite(epsilon, 'epsilon');
  if (denominatorFloor <= 0) throw new Error('epsilon must be > 0.');
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / values.length;
  const standardDeviation = Math.sqrt(variance);
  const zeroVariance = standardDeviation < denominatorFloor;
  const divisor = Math.max(standardDeviation, denominatorFloor);
  return {
    mean,
    variance,
    standardDeviation,
    epsilon: denominatorFloor,
    zeroVariance,
    zeroVariancePolicy: 'zero_advantages',
    advantages: zeroVariance
      ? values.map(() => 0)
      : values.map((value) => (value - mean) / divisor),
  };
}

export function buildTrainingRolloutGroup(input) {
  requireObject(input, 'rollout group input');
  const samples = requireArray(input.samples, 'rollout group input.samples', 2);
  const rewards = samples.map((sample, index) => {
    requireObject(sample, `rollout group input.samples[${index}]`);
    validateRewardVector(sample.rewardVector);
    return requireFinite(
      sample.rewardVector.reduction.scalarReward,
      `rollout group input.samples[${index}].rewardVector.reduction.scalarReward`
    );
  });
  const statistics = computeGroupRelativeAdvantages(
    rewards,
    input.advantageEpsilon ?? 1e-6
  );
  const artifact = {
    artifactType: 'training_rollout_group',
    schemaVersion: 1,
    workloadId: requireString(input.workloadId, 'rollout group input.workloadId'),
    groupId: requireString(input.groupId, 'rollout group input.groupId'),
    taskId: requireString(input.taskId, 'rollout group input.taskId'),
    datasetHash: requireHash(input.datasetHash, 'rollout group input.datasetHash'),
    policyHash: requireHash(input.policyHash, 'rollout group input.policyHash'),
    referencePolicyHash: requireHash(
      input.referencePolicyHash,
      'rollout group input.referencePolicyHash'
    ),
    verifierBundleHash: requireHash(
      input.verifierBundleHash,
      'rollout group input.verifierBundleHash'
    ),
    sampling: requireObject(input.sampling, 'rollout group input.sampling'),
    samples: samples.map((sample, index) => ({
      ...sample,
      advantage: statistics.advantages[index],
    })),
    groupStatistics: {
      mean: statistics.mean,
      variance: statistics.variance,
      standardDeviation: statistics.standardDeviation,
      advantageEpsilon: statistics.epsilon,
      zeroVariance: statistics.zeroVariance,
      zeroVariancePolicy: statistics.zeroVariancePolicy,
    },
    claimBoundary: requireString(input.claimBoundary, 'rollout group input.claimBoundary'),
  };
  return validateVerifierGuidedArtifact(artifact);
}

export function selectRejectionSamples(groups) {
  const selected = [];
  for (const [groupIndex, group] of requireArray(groups, 'groups', 1).entries()) {
    validateRolloutGroup(group);
    const ranked = [...group.samples].sort((left, right) => {
      const rewardDelta = right.rewardVector.reduction.scalarReward
        - left.rewardVector.reduction.scalarReward;
      return rewardDelta || left.sampleId.localeCompare(right.sampleId);
    });
    const winner = ranked.find((sample) => (
      sample.rewardVector.reduction.blocked !== true
      && sample.rewardVector.reduction.scalarReward > 0
    ));
    if (!winner) continue;
    selected.push({
      rowId: `${group.taskId}-rejection-${groupIndex + 1}`,
      taskId: group.taskId,
      groupId: group.groupId,
      prompt: winner.prompt,
      completion: winner.completion,
      sampleId: winner.sampleId,
      scalarReward: winner.rewardVector.reduction.scalarReward,
      rolloutGroupHash: hashVerifierGuidedArtifact(group),
    });
  }
  return selected;
}

export function deriveDpoPreferencePairs(groups, options = {}) {
  const minimumRewardGap = requireFinite(options.minimumRewardGap ?? 0, 'minimumRewardGap');
  const pairs = [];
  for (const group of requireArray(groups, 'groups', 1)) {
    validateRolloutGroup(group);
    const ranked = [...group.samples].sort((left, right) => {
      const rewardDelta = right.rewardVector.reduction.scalarReward
        - left.rewardVector.reduction.scalarReward;
      return rewardDelta || left.sampleId.localeCompare(right.sampleId);
    });
    const chosen = ranked[0];
    const rejected = ranked[ranked.length - 1];
    const rewardGap = chosen.rewardVector.reduction.scalarReward
      - rejected.rewardVector.reduction.scalarReward;
    if (rewardGap < minimumRewardGap) continue;
    pairs.push({
      pairId: `${group.groupId}-dpo`,
      taskId: group.taskId,
      prompt: chosen.prompt,
      chosen: chosen.completion,
      rejected: rejected.completion,
      chosenSampleId: chosen.sampleId,
      rejectedSampleId: rejected.sampleId,
      chosenReward: chosen.rewardVector.reduction.scalarReward,
      rejectedReward: rejected.rewardVector.reduction.scalarReward,
      rewardGap,
      referencePolicyHash: group.referencePolicyHash,
      rolloutGroupHash: hashVerifierGuidedArtifact(group),
    });
  }
  return pairs;
}

export function buildTrainingPromotionDecision(input) {
  requireObject(input, 'promotion input');
  const gates = requireArray(input.gates, 'promotion input.gates', 1).map((gate, index) => ({
    id: requireString(gate?.id, `promotion input.gates[${index}].id`),
    passed: gate?.passed === true,
    evidence: gate?.evidence || null,
  }));
  const missingEvidence = gates.some((gate) => !gate.evidence);
  const allPassed = gates.every((gate) => gate.passed);
  const decision = missingEvidence ? 'blocked' : (allPassed ? 'promote' : 'reject');
  const artifact = {
    artifactType: 'training_promotion_decision',
    schemaVersion: 1,
    workloadId: requireString(input.workloadId, 'promotion input.workloadId'),
    decisionId: requireString(input.decisionId, 'promotion input.decisionId'),
    decision,
    candidatePolicyHash: requireHash(input.candidatePolicyHash, 'promotion input.candidatePolicyHash'),
    promotionVerifierSplitHash: requireHash(
      input.promotionVerifierSplitHash,
      'promotion input.promotionVerifierSplitHash'
    ),
    gates,
    claimBoundary: requireString(input.claimBoundary, 'promotion input.claimBoundary'),
  };
  return validateVerifierGuidedArtifact(artifact);
}

function validateCommonArtifact(artifact, expectedType) {
  requireObject(artifact, expectedType);
  if (artifact.artifactType !== expectedType) {
    throw new Error(`artifactType must be ${expectedType}.`);
  }
  if (artifact.schemaVersion !== 1) {
    throw new Error(`${expectedType}.schemaVersion must be 1.`);
  }
  requireString(artifact.workloadId, `${expectedType}.workloadId`);
  requireString(artifact.claimBoundary, `${expectedType}.claimBoundary`);
}

function validateRolloutGroup(artifact) {
  validateCommonArtifact(artifact, 'training_rollout_group');
  requireString(artifact.groupId, 'training_rollout_group.groupId');
  requireString(artifact.taskId, 'training_rollout_group.taskId');
  requireHash(artifact.datasetHash, 'training_rollout_group.datasetHash');
  requireHash(artifact.policyHash, 'training_rollout_group.policyHash');
  requireHash(artifact.referencePolicyHash, 'training_rollout_group.referencePolicyHash');
  requireHash(artifact.verifierBundleHash, 'training_rollout_group.verifierBundleHash');
  requireObject(artifact.sampling, 'training_rollout_group.sampling');
  requireInteger(artifact.sampling.seed, 'training_rollout_group.sampling.seed');
  const samples = requireArray(artifact.samples, 'training_rollout_group.samples', 2);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = requireObject(samples[index], `training_rollout_group.samples[${index}]`);
    requireString(sample.sampleId, `training_rollout_group.samples[${index}].sampleId`);
    requireString(sample.prompt, `training_rollout_group.samples[${index}].prompt`);
    requireString(sample.completion, `training_rollout_group.samples[${index}].completion`);
    requireArray(sample.tokenIds, `training_rollout_group.samples[${index}].tokenIds`, 1);
    requireArray(sample.completionMask, `training_rollout_group.samples[${index}].completionMask`, 1);
    requireArray(sample.policyTokenLogprobs, `training_rollout_group.samples[${index}].policyTokenLogprobs`, 1);
    requireArray(sample.referenceTokenLogprobs, `training_rollout_group.samples[${index}].referenceTokenLogprobs`, 1);
    requireObject(sample.rewardVector, `training_rollout_group.samples[${index}].rewardVector`);
    requireFinite(sample.advantage, `training_rollout_group.samples[${index}].advantage`);
  }
  requireObject(artifact.groupStatistics, 'training_rollout_group.groupStatistics');
}

function validateRewardVector(artifact) {
  if (artifact.artifactType !== 'training_reward_vector' || artifact.schemaVersion !== 1) {
    throw new Error('training_reward_vector artifact header is invalid.');
  }
  requireString(artifact.taskId, 'training_reward_vector.taskId');
  requireString(artifact.sampleId, 'training_reward_vector.sampleId');
  requireHash(artifact.verifierBundleHash, 'training_reward_vector.verifierBundleHash');
  const components = requireArray(artifact.components, 'training_reward_vector.components', 1);
  const ids = new Set(components.map((component) => requireString(component.id, 'reward component id')));
  for (const id of BLOCKING_REWARD_IDS) {
    if (!ids.has(id)) throw new Error(`training_reward_vector is missing ${id}.`);
  }
  requireObject(artifact.reduction, 'training_reward_vector.reduction');
  requireFinite(artifact.reduction.scalarReward, 'training_reward_vector.reduction.scalarReward');
  requireString(artifact.claimBoundary, 'training_reward_vector.claimBoundary');
}

function validatePolicyUpdate(artifact) {
  validateCommonArtifact(artifact, 'training_policy_update');
  requireString(artifact.updateId, 'training_policy_update.updateId');
  requireHash(artifact.inputPolicyHash, 'training_policy_update.inputPolicyHash');
  requireHash(artifact.outputPolicyHash, 'training_policy_update.outputPolicyHash');
  requireArray(artifact.parentRolloutHashes, 'training_policy_update.parentRolloutHashes', 1)
    .forEach((hash, index) => requireHash(hash, `training_policy_update.parentRolloutHashes[${index}]`));
  requireObject(artifact.objective, 'training_policy_update.objective');
  requireString(artifact.objective.id, 'training_policy_update.objective.id');
  requireFinite(artifact.metrics?.loss, 'training_policy_update.metrics.loss');
}

function validateVerifierReport(artifact) {
  validateCommonArtifact(artifact, 'training_verifier_report');
  requireString(artifact.reportId, 'training_verifier_report.reportId');
  requireString(artifact.taskId, 'training_verifier_report.taskId');
  requireString(artifact.sampleId, 'training_verifier_report.sampleId');
  requireHash(artifact.verifierBundleHash, 'training_verifier_report.verifierBundleHash');
  requireArray(artifact.rawChecks, 'training_verifier_report.rawChecks', 1);
  validateRewardVector(artifact.rewardVector);
}

function validatePolicyCheckpoint(artifact) {
  validateCommonArtifact(artifact, 'training_policy_checkpoint');
  requireString(artifact.checkpointId, 'training_policy_checkpoint.checkpointId');
  requireHash(artifact.policyHash, 'training_policy_checkpoint.policyHash');
  requireHash(artifact.datasetHash, 'training_policy_checkpoint.datasetHash');
  requireArray(artifact.parentArtifactHashes, 'training_policy_checkpoint.parentArtifactHashes', 1)
    .forEach((hash, index) => requireHash(hash, `training_policy_checkpoint.parentArtifactHashes[${index}]`));
}

function validatePromotionDecision(artifact) {
  validateCommonArtifact(artifact, 'training_promotion_decision');
  requireString(artifact.decisionId, 'training_promotion_decision.decisionId');
  if (!['promote', 'reject', 'blocked'].includes(artifact.decision)) {
    throw new Error('training_promotion_decision.decision must be promote, reject, or blocked.');
  }
  requireHash(artifact.candidatePolicyHash, 'training_promotion_decision.candidatePolicyHash');
  requireHash(artifact.promotionVerifierSplitHash, 'training_promotion_decision.promotionVerifierSplitHash');
  requireArray(artifact.gates, 'training_promotion_decision.gates', 1);
  if (artifact.decision === 'promote' && artifact.gates.some((gate) => gate?.passed !== true)) {
    throw new Error('A promote decision requires every gate to pass.');
  }
}

const ARTIFACT_VALIDATORS = Object.freeze({
  training_rollout_group: validateRolloutGroup,
  training_reward_vector: validateRewardVector,
  training_policy_update: validatePolicyUpdate,
  training_verifier_report: validateVerifierReport,
  training_policy_checkpoint: validatePolicyCheckpoint,
  training_promotion_decision: validatePromotionDecision,
});

export function validateVerifierGuidedArtifact(artifact) {
  const type = requireString(artifact?.artifactType, 'artifact.artifactType');
  const validator = ARTIFACT_VALIDATORS[type];
  if (!validator) {
    throw new Error(`Unsupported verifier-guided artifact type: ${type}`);
  }
  validator(artifact);
  return artifact;
}

export function hashVerifierGuidedArtifact(artifact) {
  validateVerifierGuidedArtifact(artifact);
  return sha256Hex(stableJson(artifact));
}
