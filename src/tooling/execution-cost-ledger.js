import { computeCanonicalSha256 } from '../utils/canonical-hash.js';

export const TOKEN_COST_LEDGER_SCHEMA = 'doppler.token-cost-ledger/v1';

export function isExecutionObservationRequested(runtimeConfig) {
  return runtimeConfig?.shared?.benchmark?.run?.executionObserver?.enabled === true
    || runtimeConfig?.shared?.debug?.profiler?.enabled === true;
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function aggregateOperations(profileSteps) {
  const operations = new Map();
  for (const step of profileSteps) {
    const dispatches = step.recorderStats?.dispatches ?? [];
    const dispatchesByLabel = new Map();
    for (const dispatch of dispatches) {
      const aggregate = dispatchesByLabel.get(dispatch.label) ?? {
        count: 0,
        knownWorkgroups: 0,
        workgroups: 0,
      };
      aggregate.count += 1;
      if (Array.isArray(dispatch.workgroups)) {
        aggregate.knownWorkgroups += 1;
        aggregate.workgroups += dispatch.workgroups.reduce(
          (product, dimension) => product * dimension,
          1
        );
      }
      dispatchesByLabel.set(dispatch.label, aggregate);
    }
    for (const [label, dispatch] of dispatchesByLabel) {
      const current = operations.get(label) ?? {
        label,
        gpuMs: 0,
        hasGpuTiming: false,
        dispatches: 0,
        knownDispatchGeometry: 0,
        workgroups: 0,
      };
      current.dispatches += dispatch.count;
      current.knownDispatchGeometry += dispatch.knownWorkgroups;
      current.workgroups += dispatch.workgroups;
      operations.set(label, current);
    }
    for (const [label, gpuMs] of Object.entries(step.timings ?? {})) {
      const current = operations.get(label) ?? {
        label,
        gpuMs: 0,
        hasGpuTiming: false,
        dispatches: 0,
        knownDispatchGeometry: 0,
        workgroups: 0,
      };
      current.gpuMs += Number(gpuMs);
      current.hasGpuTiming = true;
      const dispatch = dispatchesByLabel.get(label);
      if (!dispatch) {
        current.dispatches += step.recorderStats?.opLabelCounts?.[label] ?? 0;
      }
      operations.set(label, current);
    }
  }
  return Array.from(operations.values())
    .map(({ hasGpuTiming, ...operation }) => ({
      ...operation,
      gpuMs: hasGpuTiming ? operation.gpuMs : null,
    }))
    .sort((left, right) => (
      (right.gpuMs ?? 0) - (left.gpuMs ?? 0) || left.label.localeCompare(right.label)
    ));
}

function buildPhase(name, wallMs, profileSteps) {
  const steps = Array.isArray(profileSteps) ? profileSteps : [];
  const operations = aggregateOperations(steps);
  const attributedGpuMs = operations.reduce(
    (sum, operation) => sum + (operation.gpuMs ?? 0),
    0
  );
  const hasTimestampEvidence = operations.some((operation) => operation.gpuMs !== null);
  const dispatches = operations.reduce((sum, operation) => sum + operation.dispatches, 0);
  const knownDispatchGeometry = operations.reduce(
    (sum, operation) => sum + operation.knownDispatchGeometry,
    0
  );
  const phaseWallMs = finiteOrNull(wallMs);
  return {
    phase: name,
    measurementSource: hasTimestampEvidence
      ? 'gpu-timestamp-query'
      : 'cpu-wall-estimate',
    wallMs: phaseWallMs,
    attributedGpuMs: hasTimestampEvidence ? attributedGpuMs : null,
    unattributedWallMs: hasTimestampEvidence && phaseWallMs !== null
      ? Math.max(0, phaseWallMs - attributedGpuMs)
      : null,
    timestampCoverageRatio: hasTimestampEvidence && phaseWallMs > 0
      ? Math.min(1, attributedGpuMs / phaseWallMs)
      : null,
    overlapSemantics:
      'Operation GPU timestamps are additive pass durations. They are not asserted to equal wall time.',
    operations,
    dispatches,
    dispatchGeometryCoverage: dispatches > 0 ? knownDispatchGeometry / dispatches : null,
    commandBufferSubmissions: steps.length,
    observerReadbacks: hasTimestampEvidence ? steps.length : 0,
    executionReadbacks: {
      value: null,
      status: 'not-attributed-by-observer',
    },
    estimatedBytesMoved: {
      value: null,
      unit: 'bytes',
      status: 'unavailable',
      semantics: 'estimated-not-measured',
    },
  };
}

export function buildTokenCostLedger({
  metrics,
  identity,
  device,
  browser,
}) {
  const phases = [
    buildPhase('prefill', metrics?.prefillMs, metrics?.prefillProfileSteps),
    buildPhase('decode', metrics?.decodeMs, metrics?.decodeProfileSteps),
  ];
  const selectedVariants = {
    kernelPathId: metrics?.kernelPathId ?? null,
    kernelPathSource: metrics?.kernelPathSource ?? null,
    executionPlan: metrics?.executionPlan ?? null,
  };
  const dominantOperation = phases
    .flatMap((phase) => phase.operations.map((operation) => ({
      phase: phase.phase,
      label: operation.label,
      gpuMs: operation.gpuMs,
    })))
    .filter((operation) => operation.gpuMs !== null)
    .sort((left, right) => right.gpuMs - left.gpuMs)[0] ?? null;
  const core = {
    schema: TOKEN_COST_LEDGER_SCHEMA,
    identity: {
      artifactDigest: identity?.artifactDigest ?? null,
      manifestDigest: identity?.manifestDigest ?? null,
      executionGraphDigest: identity?.executionGraphDigest ?? null,
      runtimeConfigDigest: identity?.runtimeConfigDigest ?? null,
      kernelSetDigest: identity?.kernelSetDigest ?? null,
      wrapperDigest: identity?.wrapperDigest ?? null,
      browserDigest: identity?.browserDigest ?? (
        browser ? computeCanonicalSha256(browser) : null
      ),
      adapterDigest: identity?.adapterDigest ?? (
        device ? computeCanonicalSha256(device) : null
      ),
    },
    device: device ?? null,
    browser: browser ?? null,
    phases,
    selectedVariants,
    rejectedOrFallbackVariants: metrics?.rejectedOrFallbackVariants ?? [],
    dominantOperation,
  };
  return { ...core, digest: computeCanonicalSha256(core) };
}

export function classifyTokenCostLedger(ledger, policy) {
  if (policy?.schema !== 'doppler.token-cost-classifier-policy/v1') {
    throw new Error('token cost ledger: classifier policy v1 is required');
  }
  const totals = Object.fromEntries(policy.walls.map((wall) => [wall.id, 0]));
  for (const phase of ledger.phases ?? []) {
    for (const operation of phase.operations ?? []) {
      const wall = policy.walls.find((candidate) => (
        candidate.patterns.some((pattern) => new RegExp(pattern, 'i').test(operation.label))
      ));
      if (wall) totals[wall.id] += operation.gpuMs;
    }
  }
  const classified = Object.entries(totals)
    .map(([wall, gpuMs]) => ({ wall, gpuMs }))
    .sort((left, right) => right.gpuMs - left.gpuMs);
  const dominant = classified[0]?.gpuMs > 0 ? classified[0] : {
    wall: 'unclassified',
    gpuMs: 0,
  };
  return {
    dominantWall: dominant.wall,
    classifiedGpuMs: classified,
    prescribedExperiments:
      policy.walls.find((wall) => wall.id === dominant.wall)?.experiments ?? [],
  };
}
