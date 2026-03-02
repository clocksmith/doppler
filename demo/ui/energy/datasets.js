import { computeHash, buildVliwDatasetFromSpec } from '@doppler/core';
import { VLIW_DATASETS } from '../constants.js';
import { energyDatasetCache, energySpecCache } from './cache.js';
import { stableStringify } from './utils.js';

export function applyWorkloadSpec(specInput, workloadSpec) {
  if (!workloadSpec) return specInput;
  const out = { ...(specInput || {}) };
  const fields = ['rounds', 'vectors', 'vlen', 'total_cycles'];
  fields.forEach((field) => {
    if (Number.isFinite(workloadSpec[field])) {
      out[field] = workloadSpec[field];
    }
  });
  return out;
}

export async function loadVliwDataset(datasetId, options = {}) {
  const entry = VLIW_DATASETS[datasetId];
  if (!entry) {
    throw new Error(`Unknown VLIW dataset "${datasetId}".`);
  }
  if (energyDatasetCache.has(datasetId)) {
    return energyDatasetCache.get(datasetId);
  }
  if (entry.spec) {
    const specKey = stableStringify(entry.spec);
    const dataset = await buildVliwDatasetFromSpecInput(entry.spec, specKey, {
      mode: entry.mode,
      capsMode: entry.capsMode,
      dependencyModel: entry.dependencyModel,
      workloadSpec: entry.spec,
      includeOps: options.includeOps === true,
    });
    dataset.label = entry.label || dataset.label;
    dataset.source = entry.source || dataset.source;
    energyDatasetCache.set(datasetId, dataset);
    return dataset;
  }
  const response = await fetch(entry.path);
  if (!response.ok) {
    throw new Error(`Failed to load VLIW dataset: ${response.status}`);
  }
  const payload = await response.json();
  energyDatasetCache.set(datasetId, payload);
  return payload;
}

export async function computeDagHash(dataset) {
  const capsOrdered = {};
  Object.keys(dataset?.caps || {}).sort().forEach((key) => {
    capsOrdered[key] = dataset.caps[key];
  });
  const tasks = (dataset?.tasks || []).map((task) => {
    const reads = Array.isArray(task.reads) ? task.reads.slice().sort((a, b) => a - b) : [];
    const writes = Array.isArray(task.writes) ? task.writes.slice().sort((a, b) => a - b) : [];
    const deps = Array.isArray(task.deps) ? task.deps.slice().sort((a, b) => a - b) : [];
    return {
      engine: task.engine,
      reads,
      writes,
      deps,
    };
  });
  const payload = {
    caps: capsOrdered,
    tasks,
    dependencyModel: dataset?.dependencyModel ?? null,
    bundleCount: Number.isFinite(dataset?.bundleCount) ? dataset.bundleCount : null,
  };
  const encoded = new TextEncoder().encode(stableStringify(payload));
  return computeHash(encoded, 'sha256');
}

export async function buildVliwDatasetFromSpecInput(specInput, cacheKey, options = {}) {
  const resolvedSpecInput = applyWorkloadSpec(specInput, options.workloadSpec);
  const specKey = cacheKey && !options.workloadSpec
    ? cacheKey
    : stableStringify(resolvedSpecInput);
  const includeOps = options.includeOps === true;
  const key = stableStringify({
    spec: specKey,
    options: {
      mode: options.mode ?? null,
      capsMode: options.capsMode ?? null,
      dependencyModel: options.dependencyModel ?? null,
      includeOps,
    },
  });
  if (energySpecCache.has(key)) {
    return energySpecCache.get(key);
  }
  const dataset = buildVliwDatasetFromSpec(resolvedSpecInput, { ...options, includeOps });
  const dagHash = await computeDagHash(dataset);
  dataset.dag = {
    taskCount: dataset.taskCount ?? dataset.tasks.length,
    caps: dataset.caps,
    hash: dagHash,
  };
  energySpecCache.set(key, dataset);
  return dataset;
}

export function sliceVliwDataset(dataset, bundleLimit) {
  if (!dataset || !Array.isArray(dataset.tasks)) {
    return { tasks: [], caps: {} };
  }
  const rawLimit = Number.isFinite(bundleLimit) ? Math.floor(bundleLimit) : null;
  const limit = rawLimit && rawLimit > 0 ? Math.max(1, rawLimit) : null;
  const tasks = limit == null
    ? dataset.tasks
    : dataset.tasks.filter((task) => (task.bundle ?? 0) < limit);
  const idMap = new Map();
  const remapped = [];
  let maxBundle = -1;
  tasks.forEach((task, index) => {
    idMap.set(task.id, index);
    const bundle = Number.isFinite(task.bundle) ? task.bundle : 0;
    if (bundle > maxBundle) maxBundle = bundle;
    remapped.push({ ...task, id: index });
  });
  remapped.forEach((task) => {
    const deps = Array.isArray(task.deps) ? task.deps : [];
    task.deps = deps.map((dep) => idMap.get(dep)).filter((dep) => dep != null);
  });
  const bundleCount = limit == null && Number.isFinite(dataset.bundleCount)
    ? dataset.bundleCount
    : maxBundle + 1;
  return {
    tasks: remapped,
    caps: dataset.caps ?? {},
    bundleCount,
    taskCount: remapped.length,
  };
}
