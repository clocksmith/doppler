import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isPlainObject } from '../utils/plain-object.js';

const DEFAULT_RUNTIME_CONFIG_ROOT = fileURLToPath(new URL('../config/runtime', import.meta.url));
const PROFILE_SCHEMA_VERSION = 1;

function normalizeProfileId(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim().replace(/\.json$/u, '');
  return trimmed || null;
}

function normalizeExtends(value) {
  if (Array.isArray(value)) {
    const refs = value
      .map((entry) => normalizeProfileId(entry))
      .filter(Boolean);
    return refs.length > 0 ? refs : null;
  }
  return normalizeProfileId(value);
}

function hasRuntimeSignal(runtime, pathSegments) {
  let node = runtime;
  for (const segment of pathSegments) {
    if (!isPlainObject(node) || node[segment] === undefined) {
      return false;
    }
    node = node[segment];
  }
  return node === true;
}

function summarizeSignals(runtime) {
  const shared = isPlainObject(runtime?.shared) ? runtime.shared : {};
  const debug = isPlainObject(shared.debug) ? shared.debug : {};
  const benchmark = isPlainObject(shared.benchmark) ? shared.benchmark : {};
  const benchmarkRun = isPlainObject(benchmark.run) ? benchmark.run : {};
  const inference = isPlainObject(runtime?.inference) ? runtime.inference : {};
  return {
    trace: hasRuntimeSignal(runtime, ['shared', 'debug', 'trace', 'enabled']),
    profiler: (
      hasRuntimeSignal(runtime, ['shared', 'debug', 'profiler', 'enabled'])
      || benchmarkRun.profile === true
    ),
    probes: Array.isArray(debug.probes) && debug.probes.length > 0,
    debugTokens: inference.debugTokens === true,
    benchmark: Object.keys(benchmarkRun).length > 0,
  };
}

function signalLabels(signals) {
  return Object.entries(signals)
    .filter(([, enabled]) => enabled === true)
    .map(([name]) => name);
}

async function walkJsonFiles(rootDir, currentDir = rootDir, output = []) {
  const entries = await fs.readdir(currentDir, { withFileTypes: true });
  for (const entry of entries) {
    const entryPath = path.join(currentDir, entry.name);
    if (entry.isDirectory()) {
      await walkJsonFiles(rootDir, entryPath, output);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      output.push(entryPath);
    }
  }
  return output;
}

async function readProfileCandidate(filePath) {
  let parsed;
  try {
    parsed = JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch {
    return null;
  }
  if (!isPlainObject(parsed) || !isPlainObject(parsed.runtime)) {
    return null;
  }
  const id = normalizeProfileId(parsed.id);
  if (!id) {
    return null;
  }
  return parsed;
}

function compareProfiles(a, b) {
  return a.id.localeCompare(b.id);
}

function toProfileSummary(profile, filePath, rootDir) {
  const runtime = profile.runtime;
  const shared = isPlainObject(runtime?.shared) ? runtime.shared : {};
  const tooling = isPlainObject(shared.tooling) ? shared.tooling : {};
  const relativePath = path.relative(process.cwd(), filePath).split(path.sep).join('/');
  const rootRelativePath = path.relative(rootDir, filePath).split(path.sep).join('/');
  const signals = summarizeSignals(runtime);
  return {
    id: normalizeProfileId(profile.id),
    name: typeof profile.name === 'string' && profile.name.trim() ? profile.name.trim() : null,
    description: (
      typeof profile.description === 'string' && profile.description.trim()
        ? profile.description.trim()
        : null
    ),
    intent: typeof profile.intent === 'string' && profile.intent.trim() ? profile.intent.trim() : null,
    toolingIntent: (
      typeof tooling.intent === 'string' && tooling.intent.trim()
        ? tooling.intent.trim()
        : null
    ),
    stability: (
      typeof profile.stability === 'string' && profile.stability.trim()
        ? profile.stability.trim()
        : null
    ),
    owner: typeof profile.owner === 'string' && profile.owner.trim() ? profile.owner.trim() : null,
    createdAtUtc: (
      typeof profile.createdAtUtc === 'string' && profile.createdAtUtc.trim()
        ? profile.createdAtUtc.trim()
        : null
    ),
    supersedes: normalizeProfileId(profile.supersedes),
    replacementId: normalizeProfileId(profile.replacementId),
    deprecatedAtUtc: (
      typeof profile.deprecatedAtUtc === 'string' && profile.deprecatedAtUtc.trim()
        ? profile.deprecatedAtUtc.trim()
        : null
    ),
    extends: normalizeExtends(profile.extends),
    modelId: (
      typeof profile.modelId === 'string' && profile.modelId.trim()
        ? profile.modelId.trim()
        : null
    ),
    model: typeof profile.model === 'string' && profile.model.trim() ? profile.model.trim() : null,
    path: relativePath,
    runtimePath: rootRelativePath,
    signals,
  };
}

export async function listRuntimeProfiles(options = {}) {
  const rootDir = path.resolve(options.rootDir || DEFAULT_RUNTIME_CONFIG_ROOT);
  const files = await walkJsonFiles(rootDir);
  const profiles = [];
  for (const filePath of files) {
    const profile = await readProfileCandidate(filePath);
    if (!profile) {
      continue;
    }
    profiles.push(toProfileSummary(profile, filePath, rootDir));
  }
  profiles.sort(compareProfiles);
  return {
    ok: true,
    schemaVersion: PROFILE_SCHEMA_VERSION,
    profileRoot: path.relative(process.cwd(), rootDir).split(path.sep).join('/'),
    profiles,
  };
}

function pad(value, width) {
  const text = String(value ?? '-');
  if (text.length >= width) return text;
  return `${text}${' '.repeat(width - text.length)}`;
}

export function formatRuntimeProfiles(result) {
  const profiles = Array.isArray(result?.profiles) ? result.profiles : [];
  if (profiles.length === 0) {
    return 'Runtime profiles: none';
  }

  const rows = profiles.map((profile) => {
    const signals = signalLabels(profile.signals).join(',') || '-';
    const extendsValue = Array.isArray(profile.extends)
      ? profile.extends.join(',')
      : (profile.extends || '-');
    return {
      id: profile.id,
      intent: profile.intent || '-',
      stability: profile.stability || '-',
      extends: extendsValue,
      signals,
    };
  });
  const widths = {
    id: Math.max('id'.length, ...rows.map((row) => row.id.length)),
    intent: Math.max('intent'.length, ...rows.map((row) => row.intent.length)),
    stability: Math.max('stability'.length, ...rows.map((row) => row.stability.length)),
    extends: Math.max('extends'.length, ...rows.map((row) => row.extends.length)),
    signals: Math.max('signals'.length, ...rows.map((row) => row.signals.length)),
  };

  return [
    `Runtime profiles (${profiles.length})`,
    [
      pad('id', widths.id),
      pad('intent', widths.intent),
      pad('stability', widths.stability),
      pad('extends', widths.extends),
      pad('signals', widths.signals),
    ].join('  '),
    [
      '-'.repeat(widths.id),
      '-'.repeat(widths.intent),
      '-'.repeat(widths.stability),
      '-'.repeat(widths.extends),
      '-'.repeat(widths.signals),
    ].join('  '),
    ...rows.map((row) => [
      pad(row.id, widths.id),
      pad(row.intent, widths.intent),
      pad(row.stability, widths.stability),
      pad(row.extends, widths.extends),
      pad(row.signals, widths.signals),
    ].join('  ')),
  ].join('\n');
}
