import {
  sanitizeLeanModuleName,
} from '../config/execution-contract-check.js';
import { DEFAULT_BATCHING_DEFAULTS, DEFAULT_GENERATION_CONFIG } from '../config/schema/inference-defaults.schema.js';
import { DEFAULT_KVCACHE_CONFIG } from '../config/schema/kvcache.schema.js';

export {
  sanitizeLeanModuleName,
};

const KV_LAYOUTS = new Set(['contiguous', 'paged', 'tiered', 'bdpa', 'bdpa_paged']);
const PHASES = new Set(['prefill', 'decode', 'both']);
const COLD_QUANT_MODES = new Set(['none', 'int8', 'int4']);
const ATTENTION_OPS = new Set(['attention']);
const EMBED_OPS = new Set(['embed', 'gather']);
const SAMPLE_OPS = new Set(['sample']);

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assertManifestObject(manifest) {
  if (!isPlainObject(manifest)) {
    throw new Error('lean execution contract: manifest must be an object.');
  }
  return manifest;
}

function assertPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`lean execution contract: ${label} must be a non-negative integer.`);
  }
  return value;
}

function assertPositiveIntegerOrDefault(value, fallback, label) {
  if (value == null) {
    return assertPositiveInteger(fallback, `${label} fallback`);
  }
  return assertPositiveInteger(value, label);
}

function normalizeKVLayout(value) {
  const normalized = String(value ?? DEFAULT_KVCACHE_CONFIG.layout).trim().toLowerCase();
  if (!KV_LAYOUTS.has(normalized)) {
    throw new Error(
      `lean execution contract: unsupported KV layout "${value}". ` +
      `Expected one of ${[...KV_LAYOUTS].join(', ')}.`
    );
  }
  return normalized;
}

function normalizePhase(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (!PHASES.has(normalized)) {
    throw new Error(
      `lean execution contract: ${label} must be one of ${[...PHASES].join(', ')}.`
    );
  }
  return normalized;
}

function normalizeColdQuantMode(kvcache) {
  const tieringMode = String(
    kvcache?.tiering?.mode
      ?? DEFAULT_KVCACHE_CONFIG.tiering.mode
  ).trim().toLowerCase();
  if (tieringMode === 'off' || tieringMode === 'fp16') {
    return 'none';
  }
  if (!COLD_QUANT_MODES.has(tieringMode)) {
    throw new Error(
      `lean execution contract: unsupported tiered cold quant mode "${tieringMode}".`
    );
  }
  return tieringMode;
}

function classifyOp(op) {
  const normalized = String(op ?? '').trim().toLowerCase();
  if (!normalized) {
    throw new Error('lean execution contract: execution step op is required.');
  }
  if (ATTENTION_OPS.has(normalized)) return 'attention';
  if (EMBED_OPS.has(normalized)) return 'embed';
  if (SAMPLE_OPS.has(normalized)) return 'sample';
  if (normalized.includes('norm')) return 'norm';
  if (normalized.includes('residual')) return 'residual';
  if (normalized.endsWith('_proj') || normalized.startsWith('rope_') || normalized === 'activation') {
    return 'projection';
  }
  return 'other';
}

export function extractExecutionContractFacts(manifest) {
  const normalizedManifest = assertManifestObject(manifest);
  const modelId = String(normalizedManifest.modelId ?? 'model').trim() || 'model';
  const architecture = isPlainObject(normalizedManifest.architecture)
    ? normalizedManifest.architecture
    : {};
  const inference = isPlainObject(normalizedManifest.inference)
    ? normalizedManifest.inference
    : {};
  const sessionDefaults = isPlainObject(inference.sessionDefaults)
    ? inference.sessionDefaults
    : {};
  const execution = isPlainObject(inference.execution)
    ? inference.execution
    : {};
  const kvcache = isPlainObject(sessionDefaults.kvcache)
    ? sessionDefaults.kvcache
    : {};
  const decodeLoop = isPlainObject(sessionDefaults.decodeLoop)
    ? sessionDefaults.decodeLoop
    : {};

  const steps = Array.isArray(execution.steps)
    ? execution.steps.map((step, index) => {
      if (!isPlainObject(step)) {
        throw new Error(`lean execution contract: execution.steps[${index}] must be an object.`);
      }
      const id = String(step.id ?? '').trim();
      if (!id) {
        throw new Error(`lean execution contract: execution.steps[${index}].id is required.`);
      }
      return {
        id,
        phase: normalizePhase(step.phase, `execution.steps[${index}].phase`),
        opClass: classifyOp(step.op),
      };
    })
    : [];

  return {
    modelId,
    session: {
      layout: normalizeKVLayout(kvcache.layout),
      disableCommandBatching: decodeLoop.disableCommandBatching
        ?? DEFAULT_GENERATION_CONFIG.disableCommandBatching,
      decodeBatchSize: assertPositiveIntegerOrDefault(
        decodeLoop.batchSize,
        DEFAULT_BATCHING_DEFAULTS.batchSize,
        'sessionDefaults.decodeLoop.batchSize'
      ),
      headDim: assertPositiveInteger(architecture.headDim, 'architecture.headDim'),
      kvLen: assertPositiveIntegerOrDefault(
        architecture.maxSeqLen ?? kvcache.maxSeqLen,
        DEFAULT_KVCACHE_CONFIG.maxSeqLen,
        'architecture.maxSeqLen'
      ),
      coldQuantMode: normalizeColdQuantMode(kvcache),
    },
    steps,
  };
}

function asLeanString(value) {
  return JSON.stringify(String(value));
}

function renderExecutionStep(step) {
  return [
    '  {',
    `    id := ${asLeanString(step.id)},`,
    `    phase := .${step.phase},`,
    `    opClass := .${step.opClass},`,
    '  }',
  ].join('\n');
}

function renderCheckName(modelId, suffix) {
  return `${modelId}.${suffix}`;
}

export function renderExecutionContractLeanModule(facts, options = {}) {
  const moduleName = sanitizeLeanModuleName(options.moduleName ?? facts?.modelId ?? 'GeneratedExecutionContractCheck');
  const modelId = String(facts?.modelId ?? 'model').trim() || 'model';
  const session = facts?.session;
  const steps = Array.isArray(facts?.steps) ? facts.steps : [];
  if (!session || typeof session !== 'object') {
    throw new Error('lean execution contract: facts.session is required.');
  }

  const renderedSteps = steps.length > 0
    ? steps.map(renderExecutionStep).join(',\n')
    : '';
  const stepsLiteral = steps.length > 0
    ? `[\n${renderedSteps}\n]`
    : '[]';

  return [
    'import Doppler.ExecutionContract',
    '',
    `def extractedModelId : String := ${asLeanString(modelId)}`,
    '',
    'def extractedSession : SessionConfig := {',
    `  layout := .${session.layout},`,
    `  disableCommandBatching := ${session.disableCommandBatching ? 'true' : 'false'},`,
    `  decodeBatchSize := ${session.decodeBatchSize},`,
    `  headDim := ${session.headDim},`,
    `  kvLen := ${session.kvLen},`,
    `  coldQuantMode := .${session.coldQuantMode},`,
    '}',
    '',
    `def extractedSteps : List ExecutionStep := ${stepsLiteral}`,
    '',
    'def executionContractChecks : List (String × Bool) := [',
    `  (${asLeanString(renderCheckName(modelId, 'steps'))}, allStepsCompatible extractedSteps extractedSession),`,
    `  (${asLeanString(renderCheckName(modelId, 'session'))}, sessionConsistent extractedSession)`,
    ']',
    '',
    'def renderCheck (entry : String × Bool) : String :=',
    '  let status := if entry.2 then "pass" else "fail"',
    '  s!"{entry.1}: {status}"',
    '',
    'def renderedChecks : List String :=',
    '  executionContractChecks.map renderCheck',
    '',
    'def executionContractOverall : Bool :=',
    '  executionContractChecks.all (fun entry => entry.2)',
    '',
    '#eval s!"executionContractModule:' + moduleName + '"',
    '#eval s!"executionContractOverall:{if executionContractOverall then "pass" else "fail"}"',
    '#eval renderedChecks',
    '',
  ].join('\n');
}
