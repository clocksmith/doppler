import { DEFAULT_ENTRY } from './schema/kernel-path.schema.js';
import { KERNEL_CONFIGS } from '../gpu/kernels/utils.js';
import { selectByRules } from '../gpu/kernels/rule-matcher.js';
import { loadJson } from '../utils/load-json.js';
import { buildKernelPathContractArtifact } from './kernel-path-contract-check.js';

// =============================================================================
// Built-in Kernel Paths (imported at build time)
// =============================================================================

function parseKernelPathRegistry(raw) {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Kernel path registry must be a JSON object');
  }

  const entries = Array.isArray(raw.entries) ? raw.entries : [];
  if (entries.length === 0) {
    throw new Error('Kernel path registry has no entries');
  }

  const byId = new Map();
  const normalized = [];

  for (const entry of entries) {
    if (!entry || typeof entry !== 'object') {
      throw new Error('Kernel path registry entry must be an object');
    }
    if (typeof entry.id !== 'string' || entry.id.trim() === '') {
      throw new Error('Kernel path registry entry is missing required string id');
    }

    const id = entry.id.trim();
    if (byId.has(id)) {
      throw new Error(`Duplicate kernel path registry id: ${id}`);
    }

    const trimmedAliasOf = typeof entry.aliasOf === 'string' && entry.aliasOf.trim() !== ''
      ? entry.aliasOf.trim()
      : null;
    const trimmedFile = typeof entry.file === 'string' && entry.file.trim() !== ''
      ? entry.file.trim()
      : null;

    if (!trimmedAliasOf && !trimmedFile) {
      throw new Error(`Kernel path registry entry "${id}" must include file or aliasOf`);
    }

    normalized.push({
      ...entry,
      id,
      aliasOf: trimmedAliasOf,
      file: trimmedFile,
      status: typeof entry.status === 'string' ? entry.status : 'canonical',
    });
    byId.set(id, normalized[normalized.length - 1]);
  }

  return normalized;
}

const KERNEL_PATH_REGISTRY_ENTRIES = parseKernelPathRegistry(
  await loadJson('./presets/kernel-paths/registry.json', import.meta.url, 'Failed to load kernel path')
);

const KERNEL_PATH_REGISTRY_BY_FILE = new Map(
  await Promise.all(
    [...new Set(KERNEL_PATH_REGISTRY_ENTRIES
      .map((entry) => entry.file)
      .filter((fileName) => typeof fileName === 'string'))
    ].map(async (fileName) => [
      fileName,
      await loadJson(`./presets/kernel-paths/${fileName}`, import.meta.url, 'Failed to load kernel path')
    ])
  )
);

const KERNEL_PATH_REGISTRY_INDEX = new Map(
  KERNEL_PATH_REGISTRY_ENTRIES.map((entry) => [entry.id, entry])
);

const KERNEL_PATH_REGISTRY = Object.create(null);
const KERNEL_PATH_RULES = await loadJson(
  '../rules/inference/kernel-path.rules.json',
  import.meta.url,
  'Failed to load kernel path rules'
);

const resolveKernelPathConfig = (id, chain = new Set()) => {
  if (KERNEL_PATH_REGISTRY[id] !== undefined) {
    return KERNEL_PATH_REGISTRY[id];
  }

  const entry = KERNEL_PATH_REGISTRY_INDEX.get(id);
  if (!entry) {
    throw new Error(`Unknown kernel path in registry: ${id}`);
  }

  if (chain.has(id)) {
    throw new Error(`Kernel path alias cycle detected: ${[...chain, id].join(' -> ')}`);
  }

  const nextChain = new Set(chain);
  nextChain.add(id);

  if (entry.file) {
    const resolved = KERNEL_PATH_REGISTRY_BY_FILE.get(entry.file);
    if (!resolved) {
      throw new Error(`Kernel path registry entry ${id} references missing file: ${entry.file}`);
    }
    KERNEL_PATH_REGISTRY[id] = resolved;
    return resolved;
  }

  if (!entry.aliasOf) {
    throw new Error(`Kernel path registry entry ${id} is missing aliasOf and file`);
  }

  const resolved = resolveKernelPathConfig(entry.aliasOf, nextChain);
  KERNEL_PATH_REGISTRY[id] = resolved;
  return resolved;
};

for (const entry of KERNEL_PATH_REGISTRY_ENTRIES) {
  resolveKernelPathConfig(entry.id);
}

const KERNEL_PATH_FINITENESS_FALLBACK_MAPPINGS = KERNEL_PATH_REGISTRY_ENTRIES
  .map((entry) => {
    const fallbackKernelPathId = selectByRules(
      Array.isArray(KERNEL_PATH_RULES?.finitenessFallback) ? KERNEL_PATH_RULES.finitenessFallback : [],
      { kernelPathId: entry.id }
    );
    if (typeof fallbackKernelPathId !== 'string' || fallbackKernelPathId.length === 0) {
      return null;
    }
    return {
      primaryKernelPathId: entry.id,
      fallbackKernelPathId,
      primaryActivationDtype: KERNEL_PATH_REGISTRY[entry.id]?.activationDtype ?? null,
      fallbackActivationDtype: KERNEL_PATH_REGISTRY[fallbackKernelPathId]?.activationDtype ?? null,
    };
  })
  .filter(Boolean);

const KERNEL_PATH_CONTRACT_ARTIFACT = buildKernelPathContractArtifact(
  {
    registryId: 'builtin-kernel-paths',
    entries: KERNEL_PATH_REGISTRY_ENTRIES,
    fallbackMappings: KERNEL_PATH_FINITENESS_FALLBACK_MAPPINGS,
  }
);

if (!KERNEL_PATH_CONTRACT_ARTIFACT.ok) {
  throw new Error(KERNEL_PATH_CONTRACT_ARTIFACT.errors[0]);
}

// =============================================================================
// Public API
// =============================================================================

export function getKernelPath(id) {
  return KERNEL_PATH_REGISTRY[id] ?? null;
}

export function listKernelPaths() {
  return Object.keys(KERNEL_PATH_REGISTRY);
}

export function getKernelPathContractArtifact() {
  return {
    schemaVersion: KERNEL_PATH_CONTRACT_ARTIFACT.schemaVersion,
    source: KERNEL_PATH_CONTRACT_ARTIFACT.source,
    ok: KERNEL_PATH_CONTRACT_ARTIFACT.ok,
    checks: KERNEL_PATH_CONTRACT_ARTIFACT.checks.map((entry) => ({ ...entry })),
    errors: [...KERNEL_PATH_CONTRACT_ARTIFACT.errors],
    stats: { ...KERNEL_PATH_CONTRACT_ARTIFACT.stats },
  };
}

export function resolveKernelPath(ref) {
  if (typeof ref === 'string') {
    const path = getKernelPath(ref);
    if (!path) {
      throw new Error(`Unknown kernel path: ${ref}. Available: ${listKernelPaths().join(', ')}`);
    }
    return path;
  }
  return ref;
}

export function getKernelPathActivationDtype(path) {
  if (!path?.activationDtype) return null;
  return path.activationDtype;
}

export function getKernelPathOutputDtype(path) {
  if (!path?.outputDtype) return null;
  return path.outputDtype;
}

export function getKernelPathKVDtype(path) {
  if (!path) return null;
  if (path.kvDtype) return path.kvDtype;
  if (path.activationDtype) return path.activationDtype;
  return null;
}

// =============================================================================
// Step Resolution
// =============================================================================

export function resolveWeightRef(template, layerIndex) {
  return template.replace(/\{L\}/g, String(layerIndex));
}

export function getLayerSteps(
  path,
  layerIndex,
  phase
) {
  // Check for layer-specific overrides
  if (path.layerOverrides) {
    for (const override of path.layerOverrides) {
      if (override.layers.includes(layerIndex)) {
        return override.steps;
      }
    }
  }

  // Use phase-specific or decode as fallback
  const layerPath = phase === 'prefill' && path.prefill ? path.prefill : path.decode;
  return layerPath.steps;
}

export function validateKernelPath(path) {
  const errors = [];

  if (!path.id) errors.push('Missing path id');
  if (!path.name) errors.push('Missing path name');
  if (!path.activationDtype) errors.push('Missing activationDtype');
  if (!path.decode?.steps?.length) errors.push('Missing decode steps');

  const validateSteps = (steps, context) => {
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      if (!step.op) errors.push(`${context}[${i}]: missing op`);
      if (!step.kernel) errors.push(`${context}[${i}]: missing kernel`);
    }
  };

  if (path.decode?.steps) validateSteps(path.decode.steps, 'decode');
  if (path.prefill?.steps) validateSteps(path.prefill.steps, 'prefill');
  if (path.preLayer) validateSteps(path.preLayer, 'preLayer');
  if (path.postLayer) validateSteps(path.postLayer, 'postLayer');
  if (path.sampling) validateSteps(path.sampling, 'sampling');

  return errors;
}

// =============================================================================
// Kernel Path Variant Resolution
// =============================================================================

const MATMUL_ROLE_ALIASES = {
  q_proj: { section: 'layer', ops: ['q_proj'] },
  k_proj: { section: 'layer', ops: ['k_proj'] },
  v_proj: { section: 'layer', ops: ['v_proj'] },
  qkv_proj: { section: 'layer', ops: ['qkv_proj', 'q_proj'] },
  o_proj: { section: 'layer', ops: ['o_proj'] },
  ffn_gate: { section: 'layer', ops: ['ffn_gate', 'gate_proj'] },
  ffn_up: { section: 'layer', ops: ['ffn_up', 'up_proj'] },
  ffn_down: { section: 'layer', ops: ['ffn_down', 'down_proj'] },
  ffn_gate_up: { section: 'layer', ops: ['ffn_gate_up'] },
  lm_head: { section: 'postLayer', ops: ['lm_head'] },
};

function normalizeKernelFile(kernel) {
  const trimmed = kernel.trim();
  if (!trimmed) return trimmed;
  const parts = trimmed.split('/');
  return parts[parts.length - 1] ?? trimmed;
}

function getKernelPathStepsForSection(
  path,
  section,
  phase,
  layerIndex
) {
  switch (section) {
    case 'preLayer':
      return path.preLayer ?? [];
    case 'postLayer':
      return path.postLayer ?? [];
    case 'sampling':
      return path.sampling ?? [];
    case 'layer':
    default:
      return getLayerSteps(path, layerIndex, phase);
  }
}

function findStepByOp(steps, op) {
  return steps.find((step) => step.op === op) ?? null;
}

function pickOverrideConstants(constants, overrideKeys) {
  if (!constants || overrideKeys.size === 0) return {};
  const selected = {};
  for (const key of overrideKeys) {
    if (constants[key] !== undefined) {
      selected[key] = constants[key];
    }
  }
  return selected;
}

function overridesEqual(a, b) {
  const aKeys = Object.keys(a);
  const bKeys = Object.keys(b);
  if (aKeys.length !== bKeys.length) return false;
  for (const key of aKeys) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}

function findKernelVariant(
  operation,
  kernel,
  entry,
  phase,
  constants
) {
  const variants = KERNEL_CONFIGS[operation];
  if (!variants) return null;
  const normalizedKernel = normalizeKernelFile(kernel);
  const normalizedEntry = entry ?? DEFAULT_ENTRY;

  const entryMatches = [];
  let fallbackVariant = null;
  let fallbackCount = 0;

  for (const [variant, config] of Object.entries(variants)) {
    if (config.shaderFile !== normalizedKernel) continue;
    fallbackVariant = variant;
    fallbackCount += 1;
    if (config.entryPoint === normalizedEntry) {
      entryMatches.push({ variant, config });
    }
  }

  if (entryMatches.length === 1) {
    return entryMatches[0].variant;
  }
  if (entryMatches.length > 1) {
    const overrideKeys = new Set();
    for (const { config } of entryMatches) {
      const keys = Object.keys(config.wgslOverrides ?? {});
      for (const key of keys) overrideKeys.add(key);
    }
    if (overrideKeys.size > 0) {
      const requestedOverrides = pickOverrideConstants(constants, overrideKeys);
      const overrideMatches = entryMatches.filter(({ config }) =>
        overridesEqual(config.wgslOverrides ?? {}, requestedOverrides)
      );
      if (overrideMatches.length === 1) {
        return overrideMatches[0].variant;
      }
    }
  }
  if (entryMatches.length > 1 && phase) {
    const phasePrefix = `${phase}_`;
    const phaseMatch = entryMatches.find(({ variant }) => variant.startsWith(phasePrefix));
    if (phaseMatch) {
      return phaseMatch.variant;
    }
  }

  if (fallbackCount === 1) {
    return fallbackVariant;
  }
  return null;
}

export function getKernelPathMatmulVariant(
  role,
  phase,
  layerIndex,
  path = undefined
) {
  const step = getKernelPathMatmulStep(role, phase, layerIndex, path);
  if (!step) return null;
  return findKernelVariant('matmul', step.kernel, step.entry, phase, step.constants);
}

export function getKernelPathMatmulConstants(
  role,
  phase,
  layerIndex,
  path = undefined
) {
  const step = getKernelPathMatmulStep(role, phase, layerIndex, path);
  return step?.constants ?? null;
}

function getKernelPathMatmulStep(
  role,
  phase,
  layerIndex,
  path = undefined
) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath || !role) return null;
  const alias = MATMUL_ROLE_ALIASES[role] ?? { section: 'layer', ops: [role] };
  const steps = getKernelPathStepsForSection(lookupPath, alias.section, phase, layerIndex ?? 0);
  if (role === 'lm_head' && phase === 'prefill') {
    const prefillStep = findStepByOp(steps, 'lm_head_prefill');
    if (prefillStep) {
      return prefillStep;
    }
  }
  for (const op of alias.ops) {
    const step = findStepByOp(steps, op);
    if (step) {
      return step;
    }
  }
  return null;
}

export function getKernelPathAttentionVariant(
  phase,
  layerIndex,
  path = undefined
) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath) return null;
  const steps = getKernelPathStepsForSection(lookupPath, 'layer', phase, layerIndex ?? 0);
  const step = findStepByOp(steps, 'attention');
  if (!step) return null;
  return findKernelVariant('attention', step.kernel, step.entry, phase, step.constants);
}

// =============================================================================
// Active Kernel Path Registry
// =============================================================================

let activeKernelPath = null;
let activeKernelPathSource = 'none';
let activeKernelPathPolicy = {
  mode: 'locked',
  sourceScope: ['model', 'manifest'],
  onIncompatible: 'error',
};

function normalizeKernelPathSource(source) {
  const normalized = String(source ?? '').trim().toLowerCase();
  if (normalized === 'runtime') return 'config';
  if (normalized === 'execution_v0') return 'execution-v0';
  return normalized;
}

function normalizeKernelPathPolicy(policy) {
  if (!policy || typeof policy !== 'object' || Array.isArray(policy)) {
    return {
      mode: 'locked',
      sourceScope: ['model', 'manifest'],
      onIncompatible: 'error',
    };
  }
  const mode = String(policy.mode ?? '').trim().toLowerCase() === 'capability-aware'
    ? 'capability-aware'
    : 'locked';
  const sourceScope = Array.isArray(policy.sourceScope ?? policy.allowSources)
    ? (policy.sourceScope ?? policy.allowSources)
      .map((source) => normalizeKernelPathSource(source))
      .filter((source) => source.length > 0)
    : ['model', 'manifest'];
  const onIncompatible = String(policy.onIncompatible ?? '').trim().toLowerCase() === 'remap'
    ? 'remap'
    : 'error';
  return {
    mode,
    sourceScope: sourceScope.length > 0 ? [...new Set(sourceScope)] : ['model', 'manifest'],
    onIncompatible,
  };
}

export function setActiveKernelPath(path, source = 'none', policy = null) {
  activeKernelPath = path;
  activeKernelPathSource = path ? source : 'none';
  activeKernelPathPolicy = normalizeKernelPathPolicy(policy);
}

export function getActiveKernelPath() {
  return activeKernelPath;
}

export function getActiveKernelPathSource() {
  return activeKernelPathSource;
}

export function getActiveKernelPathPolicy() {
  return {
    mode: activeKernelPathPolicy.mode,
    sourceScope: [...activeKernelPathPolicy.sourceScope],
    allowSources: [...activeKernelPathPolicy.sourceScope],
    onIncompatible: activeKernelPathPolicy.onIncompatible,
  };
}

export function getKernelPathStrict() {
  // Kernel-path overrides stay strict; capability-aware policy is handled at path-selection time.
  return true;
}

export function isKernelPathFusedQ4K(path = undefined) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath) return false;
  const kernelSteps = [
    ...(lookupPath.decode?.steps ?? []),
    ...(lookupPath.prefill?.steps ?? []),
    ...(lookupPath.preLayer ?? []),
    ...(lookupPath.postLayer ?? []),
    ...(lookupPath.layerOverrides?.flatMap((override) => override.steps) ?? []),
  ];
  return kernelSteps.some((step) => step.kernel.includes('fused_matmul_q4'));
}

export function isActiveKernelPathFusedQ4K() {
  return isKernelPathFusedQ4K(activeKernelPath);
}

export function isKernelPathDequant(path = undefined) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath) return false;
  const kernelSteps = [
    ...(lookupPath.decode?.steps ?? []),
    ...(lookupPath.prefill?.steps ?? []),
    ...(lookupPath.preLayer ?? []),
    ...(lookupPath.postLayer ?? []),
    ...(lookupPath.layerOverrides?.flatMap((override) => override.steps) ?? []),
  ];
  return kernelSteps.some((step) => step.kernel.startsWith('matmul_'));
}

export function isActiveKernelPathDequant() {
  return isKernelPathDequant(activeKernelPath);
}

// =============================================================================
// Debug/Logging
// =============================================================================

export function formatKernelPath(path) {
  const decodeOps = path.decode.steps.map(s => s.op).join(' -> ');
  return `${path.id}: ${decodeOps}`;
}

export function getKernelPathStats(path) {
  const allKernels = new Set();

  const collectKernels = (steps) => {
    for (const step of steps) {
      allKernels.add(step.kernel);
    }
  };

  collectKernels(path.decode.steps);
  if (path.prefill) collectKernels(path.prefill.steps);
  if (path.preLayer) collectKernels(path.preLayer);
  if (path.postLayer) collectKernels(path.postLayer);
  if (path.sampling) collectKernels(path.sampling);

  return {
    decodeSteps: path.decode.steps.length,
    prefillSteps: path.prefill?.steps.length ?? path.decode.steps.length,
    uniqueKernels: allKernels.size,
    hasLayerOverrides: !!path.layerOverrides?.length,
  };
}
