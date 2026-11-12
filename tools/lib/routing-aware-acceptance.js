// Routing-aware acceptance gate for routed-MoE models (policy
// doppler.routing-aware-acceptance/v1, tools/policies/routing-aware-acceptance.json).
//
// The gate never loosens doppler.boundary-tolerance/quantized-v1. It accepts an
// unforced run whose logits differ from the CPU reference only when:
// - every routing divergence is either one near-tie root or that root's causal cascade;
// - the forced-indices control and same-input operator checks pass the unchanged tolerance;
// - pre-registered task-quality limits pass on the pinned prompt set;
// - every artifact names the same qualification tuple, the approved policy hash and a
//   run start after approval.
// Pure functions only; tools/evaluate-routing-aware-acceptance.js does file IO.
import { computeCanonicalJsonSha256 } from './canonical-json.js';
import { isPlainObject } from '../../src/formats/plain-object.js';

export const POLICY_SCHEMA = 'doppler.routing-aware-acceptance/v1';
export const EVIDENCE_SCHEMA = 'doppler.routing-aware-acceptance-evidence/v1';
export const TASK_QUALITY_SCHEMA = 'doppler.routing-aware-task-quality/v1';
export const PROMPT_SET_SCHEMA = 'doppler.routing-aware-prompt-set/v1';
export const VERDICT_SCHEMA = 'doppler.routing-aware-acceptance-verdict/v1';

export function hashPolicy(policy) {
  return computeCanonicalJsonSha256(policy);
}

function reason(code, message, caseId = null) {
  return caseId === null ? { code, message } : { code, caseId, message };
}

function sortedSet(values, offset, topK) {
  return Array.from(values.subarray(offset, offset + topK)).sort((a, b) => a - b);
}

function sameJson(a, b) {
  return computeCanonicalJsonSha256(a ?? null) === computeCanonicalJsonSha256(b ?? null);
}

// The policy may only qualify the tuples it lists. Evidence also records the exact
// adapter string, which must agree across all evidence but is not pinned in advance.
export function matchQualificationTuple(policy, tuple) {
  if (!isPlainObject(tuple) || !isPlainObject(tuple.device)) return null;
  const { adapter, ...device } = tuple.device;
  if (typeof adapter !== 'string' || !adapter.trim()) return null;
  return policy.qualificationTuples.find(({ id, ...pinned }) => sameJson(pinned, { ...tuple, device })) ?? null;
}

// Per-layer near-tie bound from the forced-indices control envelope.
export function nearTieBounds(nearTie, forcedLayers, numLayers) {
  const bounds = new Array(numLayers).fill(null);
  for (const row of forcedLayers ?? []) {
    const envelope = row?.routerLogitsMaxAbsDiffVsReference;
    if (Number.isInteger(row?.layer) && row.layer >= 0 && row.layer < numLayers
      && Number.isFinite(envelope) && envelope >= 0) {
      bounds[row.layer] = Math.min(nearTie.absoluteCeiling, nearTie.envelopeMultiplier * envelope);
    }
  }
  return bounds;
}

// CPU-reference margins for one token-layer row of router logits.
export function routingMargins(cpuLogitsRow, cpuExperts, executedExperts, topK) {
  const cpu = new Set(cpuExperts);
  const executed = new Set(executedExperts);
  const dropped = [...cpu].filter(expert => !executed.has(expert)).sort((a, b) => a - b);
  const added = [...executed].filter(expert => !cpu.has(expert)).sort((a, b) => a - b);
  const ordered = Array.from(cpuLogitsRow).sort((a, b) => b - a);
  const rankMargin = ordered[topK - 1] - ordered[topK];
  if (!dropped.length || !added.length) return { dropped, added, flipDisplacement: null, rankMargin };
  const maxDropped = Math.max(...dropped.map(expert => cpuLogitsRow[expert]));
  const minAdded = Math.min(...added.map(expert => cpuLogitsRow[expert]));
  return { dropped, added, flipDisplacement: maxDropped - minAdded, rankMargin };
}

export function findDivergences({ numLayers, numTokens, topK, cpuExperts, executedExperts }) {
  const divergences = [];
  for (let layer = 0; layer < numLayers; layer++) {
    for (let token = 0; token < numTokens; token++) {
      const base = token * topK;
      const cpu = sortedSet(cpuExperts[layer], base, topK);
      const gpu = sortedSet(executedExperts[layer], base, topK);
      if (cpu.some((expert, slot) => expert !== gpu[slot])) divergences.push({ layer, token });
    }
  }
  return divergences;
}

// Split divergences into roots and cascades under the causal cone
// {(L, t): L > L0 and t >= t0}. Divergences must be sorted by layer then token.
export function attributeDivergences(divergences) {
  const roots = [];
  let cascades = 0;
  let minRootTokenBelow = Infinity;
  let pendingLayer = null;
  let pendingMin = Infinity;
  for (const { layer, token } of divergences) {
    if (layer !== pendingLayer) {
      minRootTokenBelow = Math.min(minRootTokenBelow, pendingMin);
      pendingLayer = layer;
      pendingMin = Infinity;
    }
    if (token >= minRootTokenBelow) {
      cascades++;
    } else {
      roots.push({ layer, token });
      pendingMin = Math.min(pendingMin, token);
    }
  }
  return { roots, cascades };
}

function evaluateRouting(policy, item, bounds, reasons) {
  const { caseId, numLayers, numTokens, topK, numExperts, cpu, unforced } = item;
  const divergences = findDivergences({ numLayers, numTokens, topK,
    cpuExperts: cpu.experts, executedExperts: unforced.experts });
  const { roots, cascades } = attributeDivergences(divergences);
  const classified = roots.map(({ layer, token }) => {
    const row = cpu.logits[layer].subarray(token * numExperts, (token + 1) * numExperts);
    const margins = routingMargins(row, cpu.experts[layer].subarray(token * topK, (token + 1) * topK),
      unforced.experts[layer].subarray(token * topK, (token + 1) * topK), topK);
    const bound = bounds[layer];
    const nearTie = bound !== null && margins.flipDisplacement !== null && margins.flipDisplacement <= bound;
    return { layer, token, ...margins, bound, nearTie };
  });
  const limit = policy.attribution.maxIndependentRootsPerCase;
  if (classified.length > limit) {
    reasons.push(reason('INDEPENDENT_ROOTS_EXCEEDED', `${classified.length} independent first divergences `
      + `(limit ${limit}) at ${classified.map(r => `L${r.layer}/t${r.token}`).join(', ')}`, caseId));
  }
  for (const root of classified) {
    if (!root.nearTie && policy.attribution.rootsMustBeNearTie) {
      reasons.push(reason('ROOT_NOT_NEAR_TIE', `first divergence L${root.layer}/t${root.token} has CPU flip `
        + `displacement ${root.flipDisplacement} above bound ${root.bound}`, caseId));
    }
  }
  return { divergentTokenLayers: divergences.length, roots: classified, cascadeTokenLayers: cascades };
}

function evaluateForced(policy, item, bounds, reasons) {
  const { caseId, numLayers, numExperts, topK, cpu, forcedIndices: forced } = item;
  const rules = policy.requiredControls.forcedIndices;
  if (forced.mode !== rules.mode || forced.completed !== true) {
    reasons.push(reason('FORCED_CONTROL_INCOMPLETE', `forced control must be a completed '${rules.mode}' run`, caseId));
  }
  if (forced.executedSelectionDifferingTokenLayers !== rules.maxExecutedSelectionDifferences) {
    reasons.push(reason('FORCED_CONTROL_SELECTION', `forced control executed `
      + `${forced.executedSelectionDifferingTokenLayers} differing token-layers`, caseId));
  }
  const logits = forced.logits;
  if (!isPlainObject(logits) || logits.policyId !== policy.numericalTolerance.policyId
    || !Number.isInteger(logits.count) || logits.count <= 0 || !Number.isInteger(logits.outsidePolicy)) {
    reasons.push(reason('FORCED_CONTROL_LOGITS_MISSING', 'forced control lacks a quantized-v1 logit comparison', caseId));
  } else if (logits.outsidePolicy > rules.maxLogitViolations) {
    reasons.push(reason('FORCED_CONTROL_VIOLATIONS', `forced control has ${logits.outsidePolicy} quantized-v1 violations`, caseId));
  }
  const layers = new Set((forced.layers ?? []).map(row => row?.layer));
  if (layers.size !== numLayers || bounds.some(bound => bound === null)) {
    reasons.push(reason('FORCED_CONTROL_LAYERS', 'forced control must report a router-logit envelope for every layer', caseId));
  }
  let ownDifferences = 0;
  for (const row of forced.layers ?? []) {
    if (!Array.isArray(row?.ownSelectionDifferences)) {
      reasons.push(reason('FORCED_CONTROL_OWN_SELECTIONS_MISSING', `layer ${row?.layer}: per-token own selections missing`, caseId));
      continue;
    }
    for (const own of row.ownSelectionDifferences) {
      ownDifferences++;
      if (!rules.ownSelectionDifferencesMustBeNearTie || !cpu) continue;
      const token = own.token;
      const valid = Number.isInteger(token) && token >= 0 && token < item.numTokens
        && Array.isArray(own.experts) && own.experts.length === topK;
      const margins = valid ? routingMargins(cpu.logits[row.layer].subarray(token * numExperts, (token + 1) * numExperts),
        cpu.experts[row.layer].subarray(token * topK, (token + 1) * topK), own.experts, topK) : null;
      const bound = bounds[row.layer];
      if (!margins || margins.flipDisplacement === null || bound === null || margins.flipDisplacement > bound) {
        reasons.push(reason('FORCED_CONTROL_OWN_FLIP_NOT_NEAR_TIE', `GPU own selection at L${row.layer}/t${token} `
          + `differs from the CPU set by ${margins?.flipDisplacement ?? 'an unmeasurable'} displacement (bound ${bound})`, caseId));
      }
    }
  }
  return { ownSelectionDifferences: ownDifferences, logitViolations: logits?.outsidePolicy ?? null };
}

function evaluateSameInput(policy, sameInput, numLayers, caseIds, reasons) {
  const rules = policy.requiredControls.sameInputOperators;
  if (sameInput.policyId !== policy.numericalTolerance.policyId) {
    reasons.push(reason('SAME_INPUT_POLICY', `same-input checks use ${sameInput.policyId}`));
  }
  const checks = Array.isArray(sameInput.checks) ? sameInput.checks : [];
  let violations = 0;
  const coverage = new Map();
  for (const check of checks) {
    if (!caseIds.has(check?.caseId) || !Number.isInteger(check?.count) || check.count <= 0
      || !Number.isInteger(check?.outsidePolicy)) {
      reasons.push(reason('SAME_INPUT_CHECK_INVALID', `invalid same-input check ${JSON.stringify(check?.family)}`));
      continue;
    }
    violations += check.outsidePolicy;
    const layers = coverage.get(check.family) ?? new Set();
    for (const layer of check.layers ?? []) layers.add(layer);
    coverage.set(check.family, layers);
  }
  if (violations > rules.maxViolations) {
    reasons.push(reason('SAME_INPUT_VIOLATIONS', `${violations} same-input quantized-v1 violations`));
  }
  const mixer = new Set();
  for (const { family, coverage: kind } of rules.families) {
    const layers = coverage.get(family);
    if (!layers) {
      reasons.push(reason('SAME_INPUT_FAMILY_MISSING', `no same-input check for ${family}`));
      continue;
    }
    if (kind === 'all-layers') {
      const missing = Array.from({ length: numLayers }, (_, i) => i).filter(layer => !layers.has(layer));
      if (missing.length) {
        reasons.push(reason('SAME_INPUT_LAYERS_MISSING', `${family} lacks ${missing.length} of ${numLayers} layers`));
      }
    }
    if (kind === 'sequence-mixer') for (const layer of layers) mixer.add(layer);
  }
  const mixerMissing = Array.from({ length: numLayers }, (_, i) => i).filter(layer => !mixer.has(layer));
  if (rules.families.some(entry => entry.coverage === 'sequence-mixer') && mixerMissing.length) {
    reasons.push(reason('SAME_INPUT_LAYERS_MISSING', `sequence mixers lack ${mixerMissing.length} of ${numLayers} layers`));
  }
  return { checks: checks.length, violations };
}

export function samplePositions(rule, numTokens) {
  const positions = [];
  for (let position = rule.offset; position < numTokens; position += rule.stride) positions.push(position);
  if (rule.includeLast && positions.at(-1) !== numTokens - 1) positions.push(numTokens - 1);
  return positions;
}

export function extractAnswer(text) {
  const first = String(text ?? '').trim().split(/\s+/u)[0] ?? '';
  return first.replace(/[.,;:!?]+$/u, '');
}

function evaluateTaskQuality(policy, promptSet, quality, reasons) {
  const limits = policy.taskQuality;
  const cases = new Map(promptSet.cases.map(entry => [entry.id, entry]));
  const counts = quality.caseTokenCounts ?? {};
  for (const [id, entry] of cases) {
    const n = counts[id];
    if (!Number.isInteger(n) || n < entry.minimumTokens || n > entry.maximumTokens) {
      reasons.push(reason('TASK_QUALITY_CASE_TOKENS', `token count ${n} outside the declared bounds`, id));
    }
  }
  const rows = new Map();
  for (const row of quality.distributionRows ?? []) {
    const key = `${row?.caseId}@${row?.position}`;
    const valid = cases.has(row?.caseId) && Number.isInteger(row.cpuTop1) && Number.isInteger(row.gpuTop1)
      && Number.isFinite(row.cpuTop1MarginLogits) && row.cpuTop1MarginLogits >= 0
      && Number.isFinite(row.klCpuGpuNats) && row.klCpuGpuNats >= -1e-9;
    if (!valid || rows.has(key)) {
      reasons.push(reason('TASK_QUALITY_ROW_INVALID', `invalid or duplicate distribution row ${key}`));
      continue;
    }
    rows.set(key, row);
  }
  let expectedRows = 0;
  for (const [id] of cases) {
    if (!Number.isInteger(counts[id])) continue;
    const missing = samplePositions(promptSet.samplePositions, counts[id]).filter(position => {
      expectedRows++;
      return !rows.has(`${id}@${position}`);
    });
    if (missing.length) reasons.push(reason('TASK_QUALITY_ROWS_MISSING', `${missing.length} sampled positions missing`, id));
  }
  if (rows.size !== expectedRows) {
    reasons.push(reason('TASK_QUALITY_ROWS_UNDECLARED', `${rows.size} rows for ${expectedRows} declared positions`));
  }
  const all = [...rows.values()];
  const decisive = all.filter(row => row.cpuTop1MarginLogits >= limits.distribution.decisiveMarginLogits);
  const rate = list => (list.length ? list.filter(row => row.cpuTop1 === row.gpuTop1).length / list.length : null);
  const kls = all.map(row => Math.max(0, row.klCpuGpuNats)).sort((a, b) => a - b);
  const distribution = {
    positions: all.length, decisivePositions: decisive.length,
    top1AgreementOverall: rate(all), top1AgreementDecisive: rate(decisive),
    meanKlNats: kls.length ? kls.reduce((sum, value) => sum + value, 0) / kls.length : null,
    p99KlNats: kls.length ? kls[Math.min(kls.length - 1, Math.ceil(0.99 * kls.length) - 1)] : null,
  };
  const d = limits.distribution;
  const checks = [
    ['top1AgreementOverall', distribution.top1AgreementOverall, v => v >= d.minTop1AgreementOverall],
    ['top1AgreementDecisive', distribution.top1AgreementDecisive, v => v >= d.minTop1AgreementDecisive],
    ['meanKlNats', distribution.meanKlNats, v => v <= d.maxMeanKlNats],
    ['p99KlNats', distribution.p99KlNats, v => v <= d.maxP99KlNats],
  ];
  for (const [name, value, ok] of checks) {
    // An empty decisive subset is vacuous only when rows exist; missing rows fail above.
    if (value === null ? name !== 'top1AgreementDecisive' || !all.length : !ok(value)) {
      reasons.push(reason('TASK_QUALITY_DISTRIBUTION', `${name} = ${value} violates the policy limit`));
    }
  }
  const answers = new Map();
  for (const answer of quality.answers ?? []) {
    const valid = cases.has(answer?.caseId) && Array.isArray(answer.cpuTokens) && Array.isArray(answer.gpuTokens)
      && typeof answer.cpuText === 'string' && typeof answer.gpuText === 'string';
    if (!valid || answers.has(answer.caseId)) {
      reasons.push(reason('TASK_QUALITY_ANSWER_INVALID', `invalid or duplicate answer ${answer?.caseId}`));
      continue;
    }
    answers.set(answer.caseId, answer);
  }
  let cpuCorrect = 0, gpuCorrect = 0, cpuCorrectGpuWrong = 0, identical = 0;
  for (const [id, entry] of cases) {
    const answer = answers.get(id);
    if (!answer) {
      reasons.push(reason('TASK_QUALITY_ANSWER_MISSING', 'no greedy answer pair', id));
      continue;
    }
    const cpuOk = extractAnswer(answer.cpuText) === entry.expected;
    const gpuOk = extractAnswer(answer.gpuText) === entry.expected;
    cpuCorrect += cpuOk; gpuCorrect += gpuOk; cpuCorrectGpuWrong += cpuOk && !gpuOk;
    identical += sameJson(answer.cpuTokens, answer.gpuTokens);
  }
  const e = limits.exactAnswer;
  const exactAnswer = { items: cases.size, cpuCorrect, gpuCorrect, cpuCorrectGpuWrong,
    greedyTokenAgreement: cases.size ? identical / cases.size : null };
  if (cpuCorrectGpuWrong > e.maxCpuCorrectGpuWrong) {
    reasons.push(reason('TASK_QUALITY_EXACT_ANSWER', `${cpuCorrectGpuWrong} items lost against the CPU reference`));
  }
  if (cpuCorrect - gpuCorrect > e.maxNetItemLoss) {
    reasons.push(reason('TASK_QUALITY_EXACT_ANSWER', `net item loss ${cpuCorrect - gpuCorrect}`));
  }
  if (!(exactAnswer.greedyTokenAgreement >= e.minGreedyTokenAgreement)) {
    reasons.push(reason('TASK_QUALITY_EXACT_ANSWER', `greedy token agreement ${exactAnswer.greedyTokenAgreement}`));
  }
  return { distribution, exactAnswer };
}

function checkBinding(label, artifact, context, reasons, caseId = null) {
  if (!isPlainObject(artifact)) return;
  if (!sameJson(artifact.tuple, context.tuple)) {
    reasons.push(reason('TUPLE_MISMATCH', `${label} names a different model/precision/backend/device tuple`, caseId));
  }
}

function checkFreshness(label, startedAtUtc, approvedAtUtc, reasons) {
  const started = Date.parse(startedAtUtc ?? '');
  const approved = Date.parse(approvedAtUtc ?? '');
  if (!Number.isFinite(started)) {
    reasons.push(reason('RUN_TIME_MISSING', `${label} lacks runStartedAtUtc`));
  } else if (Number.isFinite(approved) && started <= approved) {
    reasons.push(reason('RUN_NOT_FRESH', `${label} started before the policy approval`));
  }
}

// policy, promptSet and tolerancePolicies are the checked-in files; evidence is the
// loaded bundle (binary routing arrays already verified against their hashes).
export function evaluateRoutingAwareAcceptance({ policy, promptSet, tolerancePolicies, evidence }) {
  const reasons = [];
  const policySha256 = hashPolicy(policy);
  const out = { schema: VERDICT_SCHEMA, verdict: 'fail', policyId: POLICY_SCHEMA, policySha256,
    policyStatus: policy?.approval?.status ?? null, qualifiedTuple: null, reasons, cases: [],
    sameInputOperators: null, taskQuality: null };
  const sections = ['approval', 'qualificationTuples', 'numericalTolerance', 'cpuReference', 'nearTie',
    'attribution', 'requiredControls', 'taskQuality'];
  if (policy?.schema !== POLICY_SCHEMA || sections.some(key => !policy[key])) {
    reasons.push(reason('POLICY_SCHEMA', `policy must be a complete ${POLICY_SCHEMA} document`));
    return out;
  }
  const approved = policy.approval.status === 'approved' && policy.taskQuality.status === 'approved'
    && policy.nearTie.status === 'approved' && Number.isFinite(Date.parse(policy.approval.approvedAtUtc ?? ''));
  if (!approved) {
    reasons.push(reason('POLICY_NOT_APPROVED', 'policy limits are proposed, not owner-approved; no run can qualify'));
  }
  const tolerance = tolerancePolicies?.[policy.numericalTolerance.policyId];
  if (!sameJson(tolerance, policy.numericalTolerance.expected)) {
    reasons.push(reason('TOLERANCE_CHANGED', `${policy.numericalTolerance.policyId} differs from the pinned values`));
  }
  const promptSetSha256 = promptSet ? computeCanonicalJsonSha256(promptSet) : null;
  if (promptSet?.schema !== PROMPT_SET_SCHEMA || promptSetSha256 !== policy.taskQuality.promptSet.sha256) {
    reasons.push(reason('PROMPT_SET_CHANGED', 'pinned prompt set is missing or its hash differs from the policy'));
  }
  if (!isPlainObject(evidence) || evidence.schema !== EVIDENCE_SCHEMA) {
    reasons.push(reason('EVIDENCE_MISSING', `evidence bundle (${EVIDENCE_SCHEMA}) is missing`));
    return out;
  }
  if (evidence.policySha256 !== policySha256) {
    reasons.push(reason('POLICY_HASH_MISMATCH', 'evidence was not recorded against this exact policy'));
  }
  if (evidence.promptSetSha256 !== policy.taskQuality.promptSet.sha256) {
    reasons.push(reason('PROMPT_SET_HASH_MISMATCH', 'evidence does not bind the pinned prompt set'));
  }
  checkFreshness('evidence', evidence.runStartedAtUtc, policy.approval.approvedAtUtc, reasons);
  const tupleMatch = matchQualificationTuple(policy, evidence.tuple);
  if (!tupleMatch) {
    reasons.push(reason('TUPLE_NOT_QUALIFIABLE', 'evidence tuple is not one of the policy qualification tuples'));
  }
  const context = { tuple: evidence.tuple };
  const promptIds = new Set((promptSet?.cases ?? []).map(entry => entry.id));
  const caseIds = new Set();
  let numLayers = null;
  for (const item of Array.isArray(evidence.cases) ? evidence.cases : []) {
    const caseId = item?.caseId;
    if (!promptIds.has(caseId) || caseIds.has(caseId)) {
      reasons.push(reason('CASE_NOT_PINNED', 'case is not in the pinned prompt set or is duplicated', caseId ?? null));
      continue;
    }
    caseIds.add(caseId);
    numLayers ??= item.numLayers;
    const summary = { caseId, routing: null, forcedIndices: null, observationNeutral: null };
    out.cases.push(summary);
    const missing = ['cpu', 'unforced', 'forcedIndices'].filter(key => !isPlainObject(item[key]));
    for (const key of missing) reasons.push(reason('INPUT_MISSING', `${key} evidence missing`, caseId));
    checkBinding('unforced run', item.unforced, context, reasons, caseId);
    checkBinding('forced-indices control', item.forcedIndices, context, reasons, caseId);
    if (isPlainObject(item.cpu)) {
      if (item.cpu.producerSha256 !== policy.cpuReference.producerSha256
        || item.cpu.artifactManifestSha256 !== evidence.tuple?.artifactManifestSha256) {
        reasons.push(reason('CPU_REFERENCE_IDENTITY', 'CPU reference producer or artifact identity differs', caseId));
      }
    }
    const bounds = nearTieBounds(policy.nearTie, item.forcedIndices?.layers, item.numLayers);
    if (isPlainObject(item.forcedIndices)) {
      summary.forcedIndices = evaluateForced(policy, { ...item, cpu: isPlainObject(item.cpu) ? item.cpu : null }, bounds, reasons);
    }
    if (isPlainObject(item.unforced)) {
      if (item.unforced.mode !== 'observe' || item.unforced.selectionSource !== policy.attribution.selectionSource) {
        reasons.push(reason('ROUTING_NOT_OBSERVED', 'unforced selections must be executed GPU readbacks', caseId));
      }
      summary.observationNeutral = typeof item.unforced.prefillLogitsSha256 === 'string'
        && item.unforced.prefillLogitsSha256 === item.unforced.uninstrumentedPrefillLogitsSha256;
      if (policy.requiredControls.observationNeutrality.requireByteIdenticalPrefillLogits && !summary.observationNeutral) {
        reasons.push(reason('OBSERVATION_NOT_NEUTRAL', 'instrumented and uninstrumented prefill logits differ', caseId));
      }
    }
    if (missing.length === 0 && item.unforced.experts?.length === item.numLayers
      && item.cpu.experts?.length === item.numLayers && item.cpu.logits?.length === item.numLayers) {
      summary.routing = evaluateRouting(policy, item, bounds, reasons);
    } else if (!missing.length) {
      reasons.push(reason('ROUTING_LAYERS_MISSING', 'routing arrays do not cover every layer', caseId));
    }
  }
  for (const id of promptIds) {
    if (!caseIds.has(id)) reasons.push(reason('CASE_MISSING', 'pinned case has no routing evidence', id));
  }
  if (!isPlainObject(evidence.sameInputOperators)) {
    reasons.push(reason('INPUT_MISSING', 'same-input operator checks missing'));
  } else {
    checkBinding('same-input operator checks', evidence.sameInputOperators, context, reasons);
    out.sameInputOperators = evaluateSameInput(policy, evidence.sameInputOperators, numLayers ?? 0, caseIds, reasons);
  }
  const quality = evidence.taskQuality;
  if (!isPlainObject(quality) || quality.schema !== TASK_QUALITY_SCHEMA) {
    reasons.push(reason('TASK_QUALITY_MISSING', `task-quality results (${TASK_QUALITY_SCHEMA}) missing`));
  } else {
    checkBinding('task-quality results', quality, context, reasons);
    if (quality.policySha256 !== policySha256 || quality.promptSetSha256 !== policy.taskQuality.promptSet.sha256) {
      reasons.push(reason('TASK_QUALITY_BINDING', 'task-quality results bind a different policy or prompt set'));
    }
    checkFreshness('task-quality results', quality.runStartedAtUtc, policy.approval.approvedAtUtc, reasons);
    for (const item of evidence.cases ?? []) {
      if (caseIds.has(item?.caseId) && quality.caseTokenCounts?.[item.caseId] !== item.numTokens) {
        reasons.push(reason('TASK_QUALITY_CASE_TOKENS', 'routing and task-quality token counts differ', item.caseId));
      }
    }
    if (promptSet) out.taskQuality = evaluateTaskQuality(policy, promptSet, quality, reasons);
  }
  if (reasons.length === 0) {
    out.verdict = 'pass';
    out.qualifiedTuple = { ...evidence.tuple, qualificationTupleId: tupleMatch.id };
  }
  return out;
}
