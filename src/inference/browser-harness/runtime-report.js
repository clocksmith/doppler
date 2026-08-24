function parseReportTimestamp(rawTimestamp, label = 'timestamp') {
  if (rawTimestamp == null) {
    return null;
  }
  if (rawTimestamp instanceof Date) {
    const timestamp = rawTimestamp.getTime();
    if (!Number.isFinite(timestamp)) {
      throw new Error(`Invalid ${label}: not a valid Date.`);
    }
    return rawTimestamp.toISOString();
  }
  if (typeof rawTimestamp === 'number') {
    if (!Number.isFinite(rawTimestamp)) {
      throw new Error(`Invalid ${label}: must be a finite epoch timestamp.`);
    }
    return new Date(rawTimestamp).toISOString();
  }
  if (typeof rawTimestamp === 'string') {
    const trimmed = rawTimestamp.trim();
    if (trimmed.length === 0) {
      return null;
    }
    const numericCandidate = Number(trimmed);
    if (Number.isFinite(numericCandidate)) {
      return new Date(numericCandidate).toISOString();
    }
    const parsed = new Date(trimmed);
    if (Number.isNaN(parsed.getTime())) {
      throw new Error(`Invalid ${label}: expected ISO-8601 string or epoch milliseconds.`);
    }
    return parsed.toISOString();
  }
  throw new Error(`Invalid ${label}: expected Date, ISO-8601 string, epoch milliseconds, or nullish.`);
}

export function resolveReportTimestamp(rawTimestamp, label, fallbackTimestamp = null) {
  const parsed = parseReportTimestamp(rawTimestamp, label);
  return parsed ?? (fallbackTimestamp == null ? new Date().toISOString() : String(fallbackTimestamp));
}

export function sanitizeReportOutput(output) {
  if (output == null) return null;
  if (typeof output !== 'object') return output;
  if (ArrayBuffer.isView(output)) {
    return {
      type: output.constructor?.name || 'TypedArray',
      length: Number.isFinite(output.length) ? output.length : null,
    };
  }
  if (
    Number.isFinite(output?.width)
    && Number.isFinite(output?.height)
    && ArrayBuffer.isView(output?.pixels)
  ) {
    const { pixels, ...rest } = output;
    return {
      ...rest,
      width: output.width,
      height: output.height,
      pixels: {
        type: pixels.constructor?.name || 'TypedArray',
        length: Number.isFinite(pixels.length) ? pixels.length : null,
      },
    };
  }
  return output;
}

export function normalizeManifest(manifest) {
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('Harness manifest must be an object.');
  }
  const runs = Array.isArray(manifest.runs) ? manifest.runs : [];
  if (!runs.length) {
    throw new Error('Harness manifest must include at least one run.');
  }
  return {
    defaults: manifest.defaults ?? {},
    runs,
    reportModelId: manifest.reportModelId ?? manifest.id ?? 'manifest',
    report: manifest.report ?? null,
  };
}

export function mergeRunDefaults(defaults, run) {
  return {
    ...defaults,
    ...run,
    configChain: run.configChain ?? defaults.configChain ?? null,
    runtimeProfile: run.runtimeProfile ?? defaults.runtimeProfile ?? null,
    runtimeConfigUrl: run.runtimeConfigUrl ?? defaults.runtimeConfigUrl ?? null,
    runtimeConfig: run.runtimeConfig ?? defaults.runtimeConfig ?? null,
    mode: run.mode ?? defaults.mode ?? run.command ?? defaults.command ?? null,
    workload: run.workload ?? run.suite ?? defaults.workload ?? defaults.suite ?? 'inference',
  };
}

export function summarizeManifestRuns(results) {
  let passedRuns = 0;
  let failedRuns = 0;
  let durationMs = 0;
  for (const result of results) {
    const failures = (result.results || []).filter((entry) => !entry.passed && !entry.skipped);
    if (failures.length > 0) {
      failedRuns += 1;
    } else {
      passedRuns += 1;
    }
    durationMs += result.duration || 0;
  }
  return {
    totalRuns: results.length,
    passedRuns,
    failedRuns,
    durationMs,
  };
}
