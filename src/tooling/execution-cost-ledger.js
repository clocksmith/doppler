import { computeCanonicalSha256 } from '../utils/canonical-hash.js';

export const TOKEN_COST_LEDGER_SCHEMA = 'doppler.token-cost-ledger/v1';

export function isExecutionObservationRequested(runtimeConfig) {
  return runtimeConfig?.shared?.benchmark?.run?.executionObserver?.enabled === true
    || runtimeConfig?.shared?.debug?.profiler?.enabled === true;
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function sampleMetric(metrics, key) {
  const direct = metrics?.[key];
  if (Number.isFinite(direct)) {
    return Number(direct);
  }
  const nested = metrics?.gpu?.[key];
  if (Number.isFinite(nested)) {
    return Number(nested);
  }
  if (Number.isFinite(nested?.median)) {
    return Number(nested.median);
  }
  if (Number.isFinite(nested?.mean)) {
    return Number(nested.mean);
  }
  return null;
}

function stableAdapterIdentity(device) {
  if (!device || typeof device !== 'object' || Array.isArray(device)) {
    return null;
  }
  const adapterInfo = device.adapterInfo && typeof device.adapterInfo === 'object'
    ? {
      vendor: device.adapterInfo.vendor ?? null,
      architecture: device.adapterInfo.architecture ?? null,
      device: device.adapterInfo.device ?? null,
      description: device.adapterInfo.description ?? null,
    }
    : null;
  return {
    hasSubgroups: device.hasSubgroups ?? null,
    hasSubgroupsF16: device.hasSubgroupsF16 ?? null,
    hasF16: device.hasF16 ?? null,
    hasTimestampQuery: device.hasTimestampQuery ?? null,
    maxBufferSize: device.maxBufferSize ?? null,
    maxWorkgroupSize: device.maxWorkgroupSize ?? null,
    maxWorkgroupStorageSize: device.maxWorkgroupStorageSize ?? null,
    adapterInfo,
  };
}

function buildHostCosts(phase, metrics) {
  const recordMs = sampleMetric(metrics, `${phase}RecordMs`);
  const submitWaitMs = sampleMetric(metrics, `${phase}SubmitWaitMs`);
  const readbackWaitMs = sampleMetric(metrics, `${phase}ReadbackWaitMs`);
  const mapWaitMs = sampleMetric(metrics, `${phase}ReadbackMapWaitMs`);
  const cleanupMs = sampleMetric(metrics, `${phase}ReadbackCleanupMs`);
  const copyMs = sampleMetric(metrics, `${phase}ReadbackCopyMs`);
  const orchestrationMs = sampleMetric(metrics, `${phase}OrchestrationMs`);
  const fenceWaitMs = submitWaitMs === null && readbackWaitMs === null
    ? null
    : Math.max(submitWaitMs ?? 0, readbackWaitMs ?? 0);
  const observedSerialMs = recordMs === null && fenceWaitMs === null && orchestrationMs === null
    ? null
    : (recordMs ?? 0) + (fenceWaitMs ?? 0) + (orchestrationMs ?? 0);
  const candidates = [
    ['command-recording', recordMs],
    ['submit-readback-fence', fenceWaitMs],
    ['host-orchestration', orchestrationMs],
  ].filter(([, value]) => value !== null);
  candidates.sort((left, right) => right[1] - left[1]);
  return {
    recordMs,
    submitWaitMs,
    readbackWaitMs,
    fenceWaitMs,
    orchestrationMs,
    readbackBreakdown: {
      mapWaitMs,
      cleanupMs,
      copyMs,
    },
    observedSerialMs,
    dominantWall: candidates[0]?.[0] ?? null,
    overlapSemantics:
      'submitWaitMs and readbackWaitMs may observe the same submitted GPU work; fenceWaitMs is their maximum. Readback breakdown fields are nested observations and are not added again.',
  };
}

function aggregateOperations(profileSteps) {
  const operations = new Map();
  for (const step of profileSteps) {
    const dispatches = step.recorderStats?.dispatches ?? [];
    const dispatchesByLabel = new Map();
    for (const dispatch of dispatches) {
      const count = Number.isFinite(dispatch.count) && dispatch.count > 0
        ? Math.floor(dispatch.count)
        : 1;
      const aggregate = dispatchesByLabel.get(dispatch.label) ?? {
        count: 0,
        knownWorkgroups: 0,
        workgroups: 0,
      };
      aggregate.count += count;
      if (Array.isArray(dispatch.workgroups)) {
        aggregate.knownWorkgroups += count;
        aggregate.workgroups += count * dispatch.workgroups.reduce(
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

function buildFallbackOperations(topOps) {
  if (!Array.isArray(topOps)) {
    return [];
  }
  return topOps
    .filter((entry) => (
      typeof entry?.label === 'string'
      && entry.label.length > 0
      && Number.isFinite(entry.count)
      && entry.count > 0
    ))
    .map((entry) => ({
      label: entry.label,
      gpuMs: null,
      dispatches: entry.count,
      knownDispatchGeometry: 0,
      workgroups: 0,
    }));
}

function buildPhase(name, wallMs, profileSteps, metrics, fallbackOps = null) {
  const steps = Array.isArray(profileSteps) ? profileSteps : [];
  const observedOperations = aggregateOperations(steps);
  const operations = observedOperations.length > 0
    ? observedOperations
    : buildFallbackOperations(fallbackOps);
  const attributedGpuMs = operations.reduce(
    (sum, operation) => sum + (operation.gpuMs ?? 0),
    0
  );
  const hasTimestampEvidence = operations.some((operation) => operation.gpuMs !== null);
  const attributedDispatches = operations.reduce(
    (sum, operation) => sum + operation.dispatches,
    0
  );
  const observedDispatches = sampleMetric(metrics, `${name}RecordOps`);
  const dispatches = observedOperations.length > 0
    ? attributedDispatches
    : Math.max(attributedDispatches, observedDispatches ?? 0);
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
    hostCosts: buildHostCosts(name, metrics),
    operations,
    dispatches,
    unattributedDispatches: Math.max(0, dispatches - attributedDispatches),
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
    buildPhase(
      'prefill',
      metrics?.prefillMs,
      metrics?.prefillProfileSteps,
      metrics,
      metrics?.gpu?.prefillRecordTopOps
    ),
    buildPhase(
      'decode',
      metrics?.decodeMs,
      metrics?.decodeProfileSteps,
      metrics,
      metrics?.gpu?.decodeRecordTopOps
    ),
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
  const dominantObservedWall = phases
    .flatMap((phase) => [
      {
        phase: phase.phase,
        wall: 'gpu-operation',
        ms: phase.attributedGpuMs,
      },
      {
        phase: phase.phase,
        wall: phase.hostCosts.dominantWall,
        ms: phase.hostCosts.dominantWall === 'command-recording'
          ? phase.hostCosts.recordMs
          : phase.hostCosts.dominantWall === 'submit-readback-fence'
            ? phase.hostCosts.fenceWaitMs
            : phase.hostCosts.orchestrationMs,
      },
    ])
    .filter((entry) => entry.wall !== null && entry.ms !== null)
    .sort((left, right) => right.ms - left.ms)[0] ?? null;
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
        device ? computeCanonicalSha256(stableAdapterIdentity(device)) : null
      ),
    },
    device: device ?? null,
    browser: browser ?? null,
    phases,
    selectedVariants,
    rejectedOrFallbackVariants: metrics?.rejectedOrFallbackVariants ?? [],
    dominantOperation,
    dominantObservedWall,
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
  const hostTotals = {
    'command-recording': 0,
    'submit-readback-fence': 0,
    'host-orchestration': 0,
  };
  for (const phase of ledger.phases ?? []) {
    hostTotals['command-recording'] += phase.hostCosts?.recordMs ?? 0;
    hostTotals['submit-readback-fence'] += phase.hostCosts?.fenceWaitMs ?? 0;
    hostTotals['host-orchestration'] += phase.hostCosts?.orchestrationMs ?? 0;
  }
  const classifiedHostMs = Object.entries(hostTotals)
    .map(([wall, ms]) => ({ wall, ms }))
    .sort((left, right) => right.ms - left.ms);
  const gpuDominant = classified[0]?.gpuMs > 0
    ? { wall: classified[0].wall, ms: classified[0].gpuMs }
    : null;
  const hostDominant = classifiedHostMs[0]?.ms > 0 ? classifiedHostMs[0] : null;
  const dominant = !gpuDominant || (hostDominant?.ms ?? 0) > gpuDominant.ms
    ? hostDominant
    : gpuDominant;
  const builtInExperiments = {
    'command-recording': [
      'reuse-resolved-execution-steps',
      'remove-unrequested-recorder-metadata',
      'reduce-command-pass-count',
    ],
    'submit-readback-fence': [
      'gpu-resident-sampling-feedback',
      'reduce-submissions',
      'deepen-readback-ring',
    ],
    'host-orchestration': [
      'remove-per-token-graph-traversal',
      'reuse-stable-descriptors',
    ],
  };
  return {
    dominantWall: dominant?.wall ?? 'unclassified',
    classifiedGpuMs: classified,
    classifiedHostMs,
    prescribedExperiments:
      policy.walls.find((wall) => wall.id === dominant?.wall)?.experiments
      ?? builtInExperiments[dominant?.wall]
      ?? [],
  };
}
