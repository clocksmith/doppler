// Judging a device benchmark run. Pure functions: the rules come from the
// versioned protocol file, never from feature flags.

const SCHEMA = 'doppler.observatory.protocol/v1';

export function validateProtocol(protocol) {
  if (protocol?.schema !== SCHEMA) throw new Error(`Protocol must use schema ${SCHEMA}.`);
  const w = protocol.workload;
  if (!protocol.id || !/^\d+\.\d+\.\d+$/.test(protocol.version ?? '')) throw new Error('Protocol needs an id and a semantic version.');
  if (!w?.modelId || !w.prompt || !w.observationPolicy || !Number.isInteger(w.warmRuns) || w.warmRuns < 1) {
    throw new Error('Protocol workload needs modelId, prompt, observationPolicy and warmRuns.');
  }
  const trim = protocol.eligibility?.repeatSpreadTrim ?? 0;
  const checksSpread = Number.isFinite(protocol.eligibility?.maxRepeatSpread);
  if (!Number.isInteger(trim) || trim < 0 || (checksSpread && w.warmRuns < 2 * trim + 2)) {
    throw new Error('Protocol eligibility.repeatSpreadTrim leaves fewer than two repeat runs to compare.');
  }
  const reference = protocol.reference?.generatedTokenIds;
  if (reference !== null && !(Array.isArray(reference) && reference.every(Number.isInteger))) {
    throw new Error('Protocol reference.generatedTokenIds must be null or an array of token IDs.');
  }
  return protocol;
}

const sameTokens = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((id, i) => id === b[i]);

export function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

// Names the first execution setting on which two runs differ, or null.
export function executionDifference(a, b) {
  if (!a || !b) return 'not reported';
  for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
    if (a[key] !== b[key]) return `${key}: ${JSON.stringify(a[key])} and ${JSON.stringify(b[key])}`;
  }
  return null;
}

// How far apart the slowest and fastest values are, as a fraction of the
// fastest. `trim` drops that many values from each end first, so one stray run
// does not decide the answer; the median reported beside it ignores them too.
export function spread(values, trim = 0) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const kept = sorted.slice(trim, sorted.length - trim);
  return kept.length > 1 && kept[0] > 0 ? kept.at(-1) / kept[0] - 1 : null;
}

// Where two token sequences first differ, or -1 when they are the same.
export function firstDifference(a, b) {
  const length = Math.max(a?.length ?? 0, b?.length ?? 0);
  for (let i = 0; i < length; i++) if (a?.[i] !== b?.[i]) return i;
  return -1;
}

const percent = (fraction) => `${(fraction * 100).toFixed(1)}%`;

// evidence: { failure, policy, cold, warm: [...], plainPath: [...] | null }.
// Each run carries generatedTokenIds, wallMs and the runtime's own timing.
export function judgeRun(protocol, evidence) {
  validateProtocol(protocol);
  const rules = protocol.eligibility ?? {};
  const reasons = [];
  const notes = [];
  if (evidence.failure) {
    return { eligible: false, state: 'failed', reasons: [`The run stopped at "${evidence.failure.stage}": ${evidence.failure.message}`], notes, summary: null };
  }
  const warm = evidence.warm ?? [];
  const runs = [evidence.cold, ...warm].filter(Boolean);
  if (evidence.runtime?.preview === true) {
    notes.push('This result was made on a preview runtime, a build that is not qualified. Read it as a test of that preview, not as a result for a released Doppler.');
  }
  if (!evidence.cold || warm.length < protocol.workload.warmRuns) {
    reasons.push(`The protocol needs one cold run and ${protocol.workload.warmRuns} warm runs; ${runs.length} runs completed.`);
  }
  if (rules.requirePolicyPerformanceRepresentative && evidence.policy?.performanceRepresentative !== true) {
    reasons.push('The observation tier used does not declare itself performance-representative.');
  }
  if (rules.requirePolicyDoesNotModifyExecution && evidence.policy?.modifiesExecution !== false) {
    reasons.push('The observation tier used changes how the model executes, so its timing is not a benchmark.');
  }
  if (rules.requireVisibleTab && evidence.tabHiddenDuringRun !== false) {
    reasons.push('The tab was hidden for part of the run. Browsers slow hidden tabs down, so the timing is not valid.');
  }
  const artifact = evidence.fingerprint?.identity?.artifact ?? null;
  if (rules.requireProtocolArtifact && (artifact?.modelId !== protocol.workload.modelId || (protocol.workload.manifestHash && artifact?.manifestHash !== protocol.workload.manifestHash))) {
    reasons.push('The model files that ran are not the ones this protocol names.');
  }
  // The runtime reports the sampling it actually used. Every setting the protocol
  // names must have been used as named, on every run, whatever the runtime defaults are.
  if (rules.requireProtocolSampling) {
    const wanted = Object.entries(protocol.workload.generation ?? {});
    const off = [...runs, ...(evidence.plainPath ?? [])].flatMap((run) => wanted.filter(([name, value]) => run.sampling?.[name] !== value).map(([name]) => name));
    if (off.length) reasons.push(`The runtime did not use the protocol's sampling settings (${[...new Set(off)].join(', ')}).`);
  }
  const first = runs[0]?.generatedTokenIds ?? null;
  if (rules.requireIdenticalTokensAcrossRuns && runs.some((run) => !sameTokens(run.generatedTokenIds, first))) {
    reasons.push('Runs of the same fixed workload produced different tokens.');
  }
  const reference = protocol.reference?.generatedTokenIds ?? null;
  let referenceMatch = null;
  if (reference === null) {
    notes.push('No reference answer is pinned for this protocol version, so correctness was judged only by the runs agreeing with each other.');
  } else if (first) {
    const at = firstDifference(first, reference);
    referenceMatch = at === -1;
    if (!referenceMatch && rules.requireReferenceMatchWhenPinned) {
      reasons.push(`The answer differs from the pinned reference at token ${at + 1}. The reference was recorded on one device, so this can be numeric divergence rather than a fault; either way the result is not comparable.`);
    }
  }

  const trim = rules.repeatSpreadTrim ?? 0;
  const warmSpread = spread(warm.map((run) => run.wallMs), trim);
  if (Number.isFinite(rules.maxRepeatSpread)) {
    if (warmSpread === null) {
      if (warm.length >= protocol.workload.warmRuns) reasons.push('There are too few repeat runs to tell whether the timing is stable.');
    } else if (warmSpread > rules.maxRepeatSpread) {
      reasons.push(`Repeat runs varied by ${percent(warmSpread)}${trim ? ' even without the fastest and slowest' : ''}, more than the protocol's ${percent(rules.maxRepeatSpread)}. The timing is not stable enough to report: something else was using this device, or it was slowing down as it ran.`);
    }
  }

  // The benchmark tier must cost the same as generating with no inspection at all.
  const warmWallMs = median(warm.map((run) => run.wallMs));
  const plainRuns = evidence.plainPath ?? null;
  const plainWallMs = plainRuns ? median(plainRuns.map((run) => run.wallMs)) : null;
  let plainPathDeviation = null;
  if (plainRuns === null) {
    const text = 'Generation without inspection was not timed, so the benchmark tier\'s timing was not cross-checked.';
    (rules.requirePlainPathCheck ? reasons : notes).push(text);
  } else if (warmWallMs === null || plainWallMs === null || plainWallMs <= 0 || plainRuns.length < warm.length) {
    reasons.push('Generation without inspection could not be timed for every warm run, so the benchmark tier\'s timing is unverified.');
  } else {
    plainPathDeviation = warmWallMs / plainWallMs - 1;
    if (plainRuns.some((run) => !sameTokens(run.generatedTokenIds, first))) {
      reasons.push('Generation without inspection produced different tokens from the benchmark tier.');
    }
    if (rules.requireSameExecutionAsPlainPath) {
      const differs = [...warm, ...plainRuns].map((run) => executionDifference(evidence.cold?.execution, run.execution)).find(Boolean);
      if (differs) reasons.push(`The benchmark tier and generation without inspection did not run with the same execution settings (${differs}).`);
    }
    if (Number.isFinite(rules.maxPlainPathDeviation) && Math.abs(plainPathDeviation) > rules.maxPlainPathDeviation) {
      reasons.push(`The benchmark tier took ${percent(Math.abs(plainPathDeviation))} ${plainPathDeviation > 0 ? 'longer' : 'less time'} than generation without inspection, outside the protocol's ${percent(rules.maxPlainPathDeviation)} tolerance.`);
    }
  }
  return {
    eligible: reasons.length === 0,
    state: reasons.length === 0 ? 'passed' : 'not-eligible',
    reasons,
    notes,
    summary: {
      generatedTokens: first?.length ?? null,
      referenceMatch,
      coldWallMs: evidence.cold?.wallMs ?? null,
      coldTimeToFirstTokenMs: evidence.cold?.timeToFirstTokenMs ?? null,
      coldDecodeTokensPerSecond: evidence.cold?.decodeTokensPerSecond ?? null,
      warmWallMs,
      warmSpread,
      warmFastestMs: warm.length ? Math.min(...warm.map((run) => run.wallMs)) : null,
      warmSlowestMs: warm.length ? Math.max(...warm.map((run) => run.wallMs)) : null,
      warmTimeToFirstTokenMs: median(warm.map((run) => run.timeToFirstTokenMs)),
      warmDecodeTokensPerSecond: median(warm.map((run) => run.decodeTokensPerSecond)),
      plainWallMs,
      plainDecodeTokensPerSecond: plainRuns ? median(plainRuns.map((run) => run.decodeTokensPerSecond)) : null,
      plainPathDeviation,
      streamingWallMs: evidence.streaming?.wallMs ?? null,
    },
  };
}
