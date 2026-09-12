export const RECEIPT_SCHEMA = 'doppler.compute-evidence/v1';

export function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function exactKeys(value, keys, label) {
  requireCondition(value && typeof value === 'object' && !Array.isArray(value), `${label} must be an object.`);
  requireCondition(Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key)), `${label} has missing or unknown fields.`);
}

export function validateSuite(suite) {
  exactKeys(suite, ['schema', 'id', 'shared', 'engine', 'cases', 'experiments'], 'suite');
  requireCondition(suite.schema === 'doppler.compute-suite/v1', 'Unsupported compute suite schema.');
  requireCondition(/^[a-z0-9-]+$/.test(suite.id), 'Suite ID must be a path-safe slug.');
  const s = suite.shared;
  exactKeys(s, ['seed', 'warmupPairs', 'timedPairs', 'repetitions', 'absoluteTolerance', 'relativeTolerance', 'sigmoidClamp', 'guardElements', 'guardValue', 'bootstrapResamples', 'familyAlpha', 'minimumImprovementPercent', 'maxMultiplyAddsPerCase', 'timeoutMs'], 'shared');
  for (const key of ['seed', 'warmupPairs', 'timedPairs', 'repetitions', 'guardElements', 'bootstrapResamples', 'maxMultiplyAddsPerCase', 'timeoutMs']) {
    requireCondition(Number.isSafeInteger(s[key]) && s[key] > 0, `shared.${key} must be a positive safe integer.`);
  }
  requireCondition(s.timedPairs >= 20 && s.timedPairs <= 1000 && s.repetitions <= 64, 'Require 20..1000 pairs and 1..64 repetitions.');
  requireCondition(s.bootstrapResamples >= 1000 && s.bootstrapResamples <= 100000, 'Require 1000..100000 bootstrap resamples.');
  requireCondition(s.guardElements <= 256 && s.timeoutMs <= 600000, 'Guard or timeout exceeds the harness envelope.');
  for (const key of ['absoluteTolerance', 'relativeTolerance', 'sigmoidClamp', 'guardValue', 'familyAlpha', 'minimumImprovementPercent']) {
    requireCondition(Number.isFinite(s[key]) && s[key] > 0, `shared.${key} must be positive and finite.`);
  }
  requireCondition(s.familyAlpha < 1 && s.minimumImprovementPercent < 100, 'Invalid statistical policy.');
  requireCondition(s.sigmoidClamp === 15, 'The production SiLU fixture fixes the sigmoid clamp at 15.');
  exactKeys(suite.engine, ['surface', 'channel', 'headless', 'workgroupSize', 'timestampDiagnostics', 'overrideProbeSizes'], 'engine');
  const e = suite.engine;
  requireCondition(e.surface === 'browser-webgpu' && typeof e.channel === 'string' && e.channel.length > 0, 'An explicit browser WebGPU channel is required.');
  requireCondition(typeof e.headless === 'boolean' && typeof e.timestampDiagnostics === 'boolean', 'Engine switches must be boolean.');
  requireCondition(Number.isInteger(e.workgroupSize) && e.workgroupSize > 0 && e.workgroupSize <= 256, 'Invalid workgroup size.');
  requireCondition(Array.isArray(e.overrideProbeSizes) && e.overrideProbeSizes.length > 0 && e.overrideProbeSizes.every(n => Number.isInteger(n) && n > 0 && n <= 256), 'Invalid workgroup override probes.');
  requireCondition(Array.isArray(suite.cases) && suite.cases.length > 0 && suite.cases.length <= 16, 'Require 1..16 cases.');
  const ids = new Set();
  for (const c of suite.cases) {
    exactKeys(c, ['id', 'm', 'n', 'k', 'inputScale', 'biasScale'], 'case');
    requireCondition(/^[a-z0-9-]+$/.test(c.id) && !ids.has(c.id), 'Case IDs must be unique path-safe slugs.');
    ids.add(c.id);
    requireCondition(['m', 'n', 'k'].every(key => Number.isSafeInteger(c[key]) && c[key] > 0 && c[key] <= 4096), 'Invalid matrix dimensions.');
    requireCondition(c.m * c.n * c.k <= s.maxMultiplyAddsPerCase && c.m * c.n * c.k <= 100000000, 'Case exceeds its declared work envelope.');
    requireCondition(Number.isFinite(c.inputScale) && c.inputScale > 0 && Number.isFinite(c.biasScale) && c.biasScale > 0, 'Invalid input scales.');
  }
  requireCondition(Array.isArray(suite.experiments) && suite.experiments.length > 0 && suite.experiments.length <= 2, 'Require the declared fusion and/or submission experiments.');
  const experiments = new Set();
  for (const x of suite.experiments) {
    exactKeys(x, ['id', 'hypothesis', 'baseline', 'candidate'], 'experiment');
    requireCondition(!experiments.has(x.id) && ['fusion', 'submission'].includes(x.id), 'Unknown or duplicate experiment.');
    experiments.add(x.id);
    requireCondition(typeof x.hypothesis === 'string' && x.hypothesis.length > 0, 'A predeclared hypothesis is required.');
    for (const lane of [x.baseline, x.candidate]) {
      exactKeys(lane, ['id', 'fused', 'submission'], 'lane');
      requireCondition(/^[a-z0-9-]+$/.test(lane.id) && typeof lane.fused === 'boolean' && ['batch', 'dispatch'].includes(lane.submission), 'Invalid lane contract.');
    }
    requireCondition(x.baseline.id !== x.candidate.id, 'Compared lane IDs must differ.');
    requireCondition(x.id === 'fusion'
      ? !x.baseline.fused && x.candidate.fused && x.baseline.submission === 'batch' && x.candidate.submission === 'batch'
      : !x.baseline.fused && !x.candidate.fused && x.baseline.submission === 'dispatch' && x.candidate.submission === 'batch', 'Experiment changes an undeclared axis.');
  }
  return suite;
}

export async function sha256(value) {
  const bytes = typeof value === 'string' ? new TextEncoder().encode(value)
    : ArrayBuffer.isView(value) ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength) : new Uint8Array(value);
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes));
  return `sha256:${Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('')}`;
}

export function validateSample(sample, plan, repetitions) {
  requireCondition(sample.correctness?.passed === true, 'A sample failed its numerical or complete-write oracle.');
  for (const key of ['wallMs', 'encodeMs', 'submitCpuMs', 'waitAfterLastSubmitMs', 'readbackMs', 'resetMs']) {
    requireCondition(Number.isFinite(sample[key]) && sample[key] >= 0, `Invalid timing: ${key}`);
  }
  requireCondition(sample.wallMs > 0, 'A zero wall-time sample is below the measurement boundary.');
  requireCondition(sample.gpu === null, 'Timestamp-instrumented samples cannot enter the primary timing population.');
  const dispatches = plan.steps.length * repetitions;
  requireCondition(sample.counters.dispatches === dispatches, 'Observed dispatch count differs from the declared work.');
  requireCondition(sample.counters.submissions === (plan.submission === 'batch' ? 1 : dispatches), 'Observed submission count differs from the treatment.');
  requireCondition(sample.counters.readbackSubmissions === 1 && sample.counters.outputBytesRead === plan.outputBytes, 'Output coverage differs from the contract.');
  requireCondition(sample.counters.workgroups === plan.steps.reduce((total, step) => total + step.dispatch.reduce((a, b) => a * b, 1), 0) * repetitions, 'Observed workgroup count differs from the plan.');
}
