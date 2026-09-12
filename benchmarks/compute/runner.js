import { RECEIPT_SCHEMA, requireCondition, sha256, validateSuite, validateSample } from './contract.js';
import { createLinearInputs, referenceLinear, compareOutput, runOracleControls, createLinearPlan, linearResourceData } from './linear.js';
import { preparePlan } from './executor.js';
import { analyzeComparison, runStatisticsControls, runEvidenceControls } from './analysis.js';

const SHADERS = ['benchmarks/compute/linear.wgsl', 'benchmarks/compute/override_probe.wgsl', 'src/gpu/kernels/bias_add.wgsl', 'src/gpu/kernels/silu.wgsl'];

export async function runComputeEvidence(suite, ports) {
  validateSuite(suite);
  const receipt = { schema: RECEIPT_SCHEMA, schemaVersion: 1, timestamp: new Date().toISOString(), suite: suite.id, runType: 'paired-operator-ablation',
    env: { surface: 'browser-webgpu', userAgent: navigator.userAgent },
    model: { kind: 'none', scope: 'operator fixture; not a model inference benchmark' }, config: suite,
    workload: { operation: 'silu(matmul(x,weights)+bias)', cases: suite.cases, experiments: suite.experiments },
    metrics: { comparisons: [] }, quality: { passed: false, productionPromotionAllowed: false, controls: {} },
    timing: { primary: 'performance.now: encode through queue completion; divided by repetitions', gpu: 'timestamp-query in separate diagnostic runs only', excluded: ['preparation', 'poison/reset upload', 'output readback', 'correctness comparison', 'hashing', 'artifact writes'], commandCountersEnabled: true, primaryTimestampInstrumentation: false, driverCacheState: 'uncontrolled' },
    sources: {}, artifacts: [], failures: [] };
  let device;
  let intentionalClose = false;
  let deviceLoss = null;
  const gpuErrors = [];
  try {
    requireCondition(navigator.gpu, 'This browser does not expose WebGPU.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    requireCondition(adapter, 'No WebGPU adapter is available.');
    const info = adapter.info ?? await adapter.requestAdapterInfo?.() ?? {};
    const identity = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize'].filter(key => info[key] !== undefined).map(key => [key, info[key]]));
    receipt.env.adapter = identity;
    receipt.env.adapterIsFallback = adapter.isFallbackAdapter ?? null;
    receipt.env.adapterFeatures = Array.from(adapter.features).sort();
    requireCondition(adapter.isFallbackAdapter !== true && !/swiftshader|llvmpipe|software/i.test(JSON.stringify(identity)), 'A software/fallback adapter cannot establish hardware compute evidence.');
    const timestamps = suite.engine.timestampDiagnostics && adapter.features.has('timestamp-query');
    device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] });
    receipt.env.requestedFeatures = timestamps ? ['timestamp-query'] : [];
    receipt.env.deviceFeatures = Array.from(device.features).sort();
    receipt.env.limits = Object.fromEntries(['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupsPerDimension'].map(key => [key, device.limits[key]]));
    device.lost.then(info => { if (!intentionalClose) deviceLoss = { reason: info.reason, message: info.message }; });
    device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
    const assertHealthy = () => requireCondition(!deviceLoss && gpuErrors.length === 0, `GPU execution failed: ${JSON.stringify({ deviceLoss, gpuErrors })}`);
    const sources = {};
    for (const path of SHADERS) {
      const response = await fetch(`/${path}`);
      requireCondition(response.ok, `Missing source snapshot: ${path}`);
      sources[path] = await response.text();
      receipt.sources[path] = await sha256(sources[path]);
    }
    receipt.quality.controls.statistics = runStatisticsControls();
    receipt.quality.controls.workgroupOverrides = [];
    for (const size of suite.engine.overrideProbeSizes) {
      requireCondition(size <= device.limits.maxComputeInvocationsPerWorkgroup && size <= device.limits.maxComputeWorkgroupSizeX && size * 4 <= device.limits.maxComputeWorkgroupStorageSize, 'Override probe exceeds device limits.');
      const bytes = (1 + suite.shared.guardElements) * 4;
      const plan = { id: `workgroup-override-${size}`, resources: { output: { bytes, kind: 'storage' } }, output: 'output', outputBytes: bytes, outputElements: 1, resetResources: ['output'], submission: 'batch',
        steps: [{ id: 'reduce-workgroup', source: 'benchmarks/compute/override_probe.wgsl', entryPoint: 'main', constants: { ELEMENTS: size }, bindings: [{ resource: 'output', type: 'storage' }], dispatch: [1, 1, 1] }] };
      const prepared = await preparePlan(device, plan, {}, sources, suite.shared);
      try {
        const result = await prepared.run({ repetitions: 1, trace: true });
        const correctness = compareOutput(result.output, new Float32Array([size]), suite.shared);
        requireCondition(correctness.passed, `Workgroup override execution failed for ${size}.`);
        const { output, ...measurement } = result;
        receipt.quality.controls.workgroupOverrides.push({ size, plan, correctness, output: Array.from(output), measurement });
      } finally { prepared.close(); }
    }
    for (let caseIndex = 0; caseIndex < suite.cases.length; caseIndex += 1) {
      const shape = suite.cases[caseIndex];
      await ports.progress({ phase: 'case-start', caseId: shape.id });
      const inputs = createLinearInputs(shape, suite.shared, caseIndex);
      const expected = referenceLinear(shape, inputs, suite.shared);
      const oracleControls = runOracleControls(expected, suite.shared);
      const inputArtifacts = {};
      for (const [id, array] of Object.entries({ ...inputs, expected })) {
        const artifact = await ports.artifact(`data/${shape.id}/${id}.bin`, array);
        requireCondition(artifact.sha256 === await sha256(array), 'Artifact transport changed input bytes.');
        inputArtifacts[id] = artifact;
        receipt.artifacts.push(artifact);
      }
      const inputIdentity = await sha256(JSON.stringify(Object.fromEntries(Object.entries(inputArtifacts).map(([id, value]) => [id, value.sha256]))));
      for (const experiment of suite.experiments) {
        const index = receipt.metrics.comparisons.length;
        const row = { caseId: shape.id, experimentId: experiment.id, hypothesis: experiment.hypothesis, shape, inputIdentity, inputArtifacts, oracleControls, plans: {}, preparation: {}, outputs: {}, warmup: [], pairs: [], diagnostics: {}, status: 'running' };
        receipt.metrics.comparisons.push(row);
        const prepared = {};
        try {
          for (const role of index % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate']) {
            const plan = createLinearPlan(shape, experiment[role], suite);
            prepared[role] = await preparePlan(device, plan, linearResourceData(shape, inputs), sources, suite.shared);
            row.plans[role] = plan;
            row.preparation[role] = prepared[role].preparation;
          }
          requireCondition(JSON.stringify(row.plans.baseline.semanticContract) === JSON.stringify(row.plans.candidate.semanticContract), 'Compared lanes changed the semantic workload.');
          if (experiment.id === 'submission') requireCondition(JSON.stringify(row.plans.baseline.steps) === JSON.stringify(row.plans.candidate.steps), 'Submission treatment changed the shader graph.');
          async function measure(role, phase, iteration, profile = false) {
            const raw = await prepared[role].run({ repetitions: suite.shared.repetitions, profile, trace: phase === 'diagnostic' });
            assertHealthy();
            const correctness = compareOutput(raw.output, expected, suite.shared);
            const outputHash = await sha256(raw.output);
            if (!row.outputs[role] || !correctness.passed || row.outputs[role].sha256 !== outputHash) {
              const suffix = row.outputs[role] ? `-${phase}-${iteration}` : '';
              const artifact = await ports.artifact(`data/${shape.id}/${experiment.id}-${role}${suffix}.bin`, raw.output);
              receipt.artifacts.push(artifact);
              if (!row.outputs[role]) row.outputs[role] = artifact;
            }
            const { output, ...sample } = raw;
            Object.assign(sample, { phase, iteration, outputHash, correctness });
            if (!correctness.passed) {
              receipt.failures.push({ caseId: shape.id, experimentId: experiment.id, role, sample });
              throw new Error(`Numerical oracle rejected ${shape.id}/${experiment.id}/${role}/${phase}/${iteration}.`);
            }
            requireCondition(outputHash === row.outputs[role].sha256, 'Repeated execution changed output identity within the same lane.');
            if (phase !== 'diagnostic') validateSample(sample, row.plans[role], suite.shared.repetitions);
            return sample;
          }
          for (const phase of ['warmup', 'timed']) {
            const count = phase === 'warmup' ? suite.shared.warmupPairs : suite.shared.timedPairs;
            for (let i = 0; i < count; i += 1) {
              const order = i % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate'];
              const pair = { index: i, order };
              for (const role of order) pair[role] = await measure(role, phase, i);
              (phase === 'warmup' ? row.warmup : row.pairs).push(pair);
            }
          }
          row.summary = analyzeComparison(row, suite, index);
          row.evidenceControls = runEvidenceControls(row, suite, index);
          for (const role of ['baseline', 'candidate']) row.diagnostics[role] = await measure(role, 'diagnostic', 0, timestamps);
          row.status = 'complete';
          await ports.progress({ phase: 'comparison-complete', caseId: shape.id, experimentId: experiment.id, summary: row.summary });
        } finally { for (const value of Object.values(prepared)) value.close(); }
      }
    }
    await device.queue.onSubmittedWorkDone();
    assertHealthy();
    receipt.quality.passed = true;
  } catch (error) {
    receipt.failures.push({ name: error.name, message: error.message, stack: error.stack, deviceLoss, gpuErrors });
  } finally {
    intentionalClose = true;
    device?.destroy();
  }
  return receipt;
}
