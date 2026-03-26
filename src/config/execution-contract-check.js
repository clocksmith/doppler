const KV_LAYOUTS = new Set(['contiguous', 'paged', 'tiered', 'bdpa']);
const PHASES = new Set(['prefill', 'decode', 'both']);
const COLD_QUANT_MODES = new Set(['none', 'int8', 'int4']);
const ATTENTION_OPS = new Set(['attention']);
const EMBED_OPS = new Set(['embed', 'gather']);
const SAMPLE_OPS = new Set(['sample']);
const BDPA_MAX_HEAD_DIM = 256;
const BDPA_MAX_KV_LEN = 2048;
const TIERED_MAX_QUANT_HEAD_DIM = 256;

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assertManifestObject(manifest) {
  if (!isPlainObject(manifest)) {
    throw new Error('execution contract: manifest must be an object.');
  }
  return manifest;
}

function assertPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`execution contract: ${label} must be a non-negative integer.`);
  }
  return value;
}

function assertRequiredValue(value, label) {
  if (value == null) {
    throw new Error(`execution contract: ${label} is required.`);
  }
  return value;
}

function assertBoolean(value, label) {
  if (value !== true && value !== false) {
    throw new Error(`execution contract: ${label} must be boolean.`);
  }
  return value;
}

function normalizeKVLayout(value) {
  const normalized = String(assertRequiredValue(value, 'session.kvcache.layout')).trim().toLowerCase();
  if (!KV_LAYOUTS.has(normalized)) {
    throw new Error(
      `execution contract: unsupported KV layout "${value}". ` +
      `Expected one of ${[...KV_LAYOUTS].join(', ')}.`
    );
  }
  return normalized;
}

function normalizePhase(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (!PHASES.has(normalized)) {
    throw new Error(
      `execution contract: ${label} must be one of ${[...PHASES].join(', ')}.`
    );
  }
  return normalized;
}

function normalizeColdQuantMode(kvcache) {
  const tieringMode = String(
    assertRequiredValue(kvcache?.tiering?.mode, 'session.kvcache.tiering.mode')
  ).trim().toLowerCase();
  if (tieringMode === 'off' || tieringMode === 'fp16') {
    return 'none';
  }
  if (!COLD_QUANT_MODES.has(tieringMode)) {
    throw new Error(
      `execution contract: unsupported tiered cold quant mode "${tieringMode}".`
    );
  }
  return tieringMode;
}

function classifyOp(op) {
  const normalized = String(op ?? '').trim().toLowerCase();
  if (!normalized) {
    throw new Error('execution contract: execution step op is required.');
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

export function sanitizeLeanModuleName(value) {
  const raw = String(value ?? 'GeneratedExecutionContractCheck').trim();
  const alnum = raw.replace(/[^A-Za-z0-9_]/g, '_');
  if (!alnum) {
    return 'GeneratedExecutionContractCheck';
  }
  if (/^[A-Za-z_]/.test(alnum)) {
    return alnum;
  }
  return `Generated_${alnum}`;
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
  const session = isPlainObject(inference.session) ? inference.session : {};
  const execution = isPlainObject(inference.execution)
    ? inference.execution
    : {};
  const kvcache = isPlainObject(session.kvcache)
    ? session.kvcache
    : {};
  const decodeLoop = isPlainObject(session.decodeLoop)
    ? session.decodeLoop
    : {};

  const steps = Array.isArray(execution.steps)
    ? execution.steps.map((step, index) => {
      if (!isPlainObject(step)) {
        throw new Error(`execution contract: execution.steps[${index}] must be an object.`);
      }
      const id = String(step.id ?? '').trim();
      if (!id) {
        throw new Error(`execution contract: execution.steps[${index}].id is required.`);
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
      disableCommandBatching: assertBoolean(
        decodeLoop.disableCommandBatching,
        'session.decodeLoop.disableCommandBatching'
      ),
      decodeBatchSize: assertPositiveInteger(
        assertRequiredValue(decodeLoop.batchSize, 'session.decodeLoop.batchSize'),
        'session.decodeLoop.batchSize'
      ),
      headDim: assertPositiveInteger(architecture.headDim, 'architecture.headDim'),
      kvLen: assertPositiveInteger(
        assertRequiredValue(
          architecture.maxSeqLen ?? kvcache.maxSeqLen,
          'architecture.maxSeqLen'
        ),
        'architecture.maxSeqLen'
      ),
      coldQuantMode: normalizeColdQuantMode(kvcache),
    },
    steps,
  };
}

export function validateExecutionContractFacts(facts) {
  const errors = [];
  const checks = [];
  const modelId = String(facts?.modelId ?? 'model');
  const session = facts?.session ?? {};
  const steps = Array.isArray(facts?.steps) ? facts.steps : [];

  const incompatibleStep = session.layout === 'bdpa'
    ? steps.find((step) => step.opClass === 'attention' && (step.phase === 'prefill' || step.phase === 'both'))
    : null;
  if (incompatibleStep) {
    errors.push(
      `[ExecutionContract] session.kvcache.layout="bdpa" is decode-only, ` +
      `but step "${incompatibleStep.id}" declares ${incompatibleStep.phase} attention.`
    );
  }
  checks.push({
    id: `${modelId}.steps`,
    ok: incompatibleStep == null,
  });

  const sessionErrorCount = errors.length;
  if (session.layout === 'bdpa') {
    if (session.disableCommandBatching !== true) {
      errors.push(
        '[ExecutionContract] session.kvcache.layout="bdpa" requires ' +
        'session.decodeLoop.disableCommandBatching=true.'
      );
    }
    if (session.decodeBatchSize > 1) {
      errors.push(
        `[ExecutionContract] session.kvcache.layout="bdpa" requires ` +
        `session.decodeLoop.batchSize <= 1; got ${session.decodeBatchSize}.`
      );
    }
    if (session.headDim > BDPA_MAX_HEAD_DIM) {
      errors.push(
        `[ExecutionContract] session.kvcache.layout="bdpa" requires architecture.headDim <= ${BDPA_MAX_HEAD_DIM}; ` +
        `got ${session.headDim}.`
      );
    }
    if (session.kvLen > BDPA_MAX_KV_LEN) {
      errors.push(
        `[ExecutionContract] session.kvcache.layout="bdpa" requires architecture.maxSeqLen <= ${BDPA_MAX_KV_LEN}; ` +
        `got ${session.kvLen}.`
      );
    }
  }

  if (
    session.layout === 'tiered'
    && session.coldQuantMode !== 'none'
    && session.headDim > TIERED_MAX_QUANT_HEAD_DIM
  ) {
    errors.push(
      `[ExecutionContract] session.kvcache.layout="tiered" with cold quantization requires ` +
      `architecture.headDim <= ${TIERED_MAX_QUANT_HEAD_DIM}; got ${session.headDim}.`
    );
  }

  checks.push({
    id: `${modelId}.session`,
    ok: errors.length === sessionErrorCount,
  });

  return {
    ok: errors.length === 0,
    errors,
    checks,
  };
}

export function validateManifestExecutionContract(manifest) {
  const facts = extractExecutionContractFacts(manifest);
  const evaluation = validateExecutionContractFacts(facts);
  return {
    ...evaluation,
    facts,
  };
}

export function buildExecutionContractArtifact(manifest) {
  if (!manifest || typeof manifest !== 'object') {
    return null;
  }
  if (
    manifest.modelType === 'diffusion'
    || manifest.modelType === 'energy'
    || manifest.modelType === 'embedding'
  ) {
    return null;
  }
  if (!manifest.architecture || !manifest.inference || typeof manifest.inference !== 'object') {
    return null;
  }
  try {
    const evaluation = validateManifestExecutionContract(manifest);
    const attentionPhaseCounts = { prefill: 0, decode: 0, both: 0 };
    for (const step of evaluation.facts.steps) {
      if (step.opClass !== 'attention') continue;
      if (Object.prototype.hasOwnProperty.call(attentionPhaseCounts, step.phase)) {
        attentionPhaseCounts[step.phase] += 1;
      }
    }
    return {
      schemaVersion: 1,
      source: 'doppler',
      ok: evaluation.ok,
      checks: [...evaluation.checks],
      errors: [...evaluation.errors],
      session: evaluation.facts.session,
      steps: {
        total: evaluation.facts.steps.length,
        attention: attentionPhaseCounts.prefill + attentionPhaseCounts.decode + attentionPhaseCounts.both,
        attentionPhases: attentionPhaseCounts,
      },
    };
  } catch (error) {
    return {
      schemaVersion: 1,
      source: 'doppler',
      ok: false,
      checks: [],
      errors: [error instanceof Error ? error.message : String(error)],
      session: null,
      steps: null,
    };
  }
}
