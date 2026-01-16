import { readFile } from 'fs/promises';
import { resolve } from 'path';

function normalizeString(value) {
  return (value ?? '').toString().toLowerCase();
}

function matchesRule(value, rule) {
  if (!rule) return true;
  const target = normalizeString(value);
  if (typeof rule === 'string') {
    return target.includes(normalizeString(rule));
  }
  const mode = rule.matchType || 'includes';
  const pattern = rule.match ?? '';
  if (!pattern) return false;
  if (mode === 'exact') {
    return target === normalizeString(pattern);
  }
  if (mode === 'regex') {
    try {
      const regex = new RegExp(pattern, 'i');
      return regex.test(value ?? '');
    } catch {
      return false;
    }
  }
  return target.includes(normalizeString(pattern));
}

function matchesModel(result, baseline) {
  const modelId = result.model?.modelId ?? '';
  const modelName = result.model?.modelName ?? '';
  const modelRule = baseline.model ?? baseline.modelMatch ?? null;
  if (!modelRule) return true;
  return matchesRule(modelId, modelRule) || matchesRule(modelName, modelRule);
}

function matchesGPU(result, baseline) {
  const gpu = result.env?.gpu?.description ?? result.env?.gpu?.device ?? '';
  return matchesRule(gpu, baseline.gpu ?? baseline.gpuMatch ?? null);
}

function matchesBrowser(result, baseline) {
  const browser = result.env?.browser?.name ?? '';
  return matchesRule(browser, baseline.browser ?? baseline.browserMatch ?? null);
}

export async function loadBaselineRegistry(path) {
  const resolved = resolve(path);
  try {
    const raw = await readFile(resolved, 'utf-8');
    return JSON.parse(raw);
  } catch (err) {
    if (err?.code === 'ENOENT') {
      return { baselines: [], _path: resolved };
    }
    throw err;
  }
}

export function findBaselineForResult(result, registry) {
  const baselines = registry?.baselines ?? [];
  for (const baseline of baselines) {
    if (baseline.enabled === false) continue;
    if (!matchesModel(result, baseline)) continue;
    if (!matchesGPU(result, baseline)) continue;
    if (!matchesBrowser(result, baseline)) continue;
    return baseline;
  }
  return null;
}

export function evaluateBaseline(result, baseline) {
  const metrics = baseline.metrics ?? {};
  const violations = [];

  for (const [key, range] of Object.entries(metrics)) {
    if (!range || typeof range !== 'object') continue;
    const value = result.metrics?.[key];
    if (typeof value !== 'number') continue;
    const min = typeof range.min === 'number' ? range.min : null;
    const max = typeof range.max === 'number' ? range.max : null;
    if (min !== null && value < min) {
      violations.push({ metric: key, value, min, max });
      continue;
    }
    if (max !== null && value > max) {
      violations.push({ metric: key, value, min, max });
    }
  }

  return {
    ok: violations.length === 0,
    violations,
  };
}
