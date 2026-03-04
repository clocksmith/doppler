function normalizeString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function resolveTrainingMetrics(report) {
  const direct = Array.isArray(report?.metrics?.trainingMetricsReport)
    ? report.metrics.trainingMetricsReport
    : null;
  if (direct) return direct;
  const fromResults = Array.isArray(report?.results)
    ? report.results.flatMap((entry) => (Array.isArray(entry?.metrics?.trainingMetricsReport)
      ? entry.metrics.trainingMetricsReport
      : []))
    : [];
  return fromResults;
}

function bytesToHex(bytes) {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

async function sha256Hex(text) {
  if (!globalThis.crypto?.subtle) {
    throw new Error('crypto.subtle is unavailable in this browser.');
  }
  const encoded = new TextEncoder().encode(String(text));
  const digest = await globalThis.crypto.subtle.digest('SHA-256', encoded);
  return bytesToHex(new Uint8Array(digest));
}

async function computeReportId(reportObj, rawJson) {
  const explicit = normalizeString(reportObj?.reportId || reportObj?.id || reportObj?.report?.id);
  if (explicit) return explicit;
  return sha256Hex(rawJson);
}

function resolveWorkloadPackTraceability(workloadEntry) {
  if (!workloadEntry || typeof workloadEntry !== 'object') return null;
  const id = normalizeString(workloadEntry.id);
  if (!id) {
    throw new Error('Workload entry is missing id.');
  }
  return {
    id,
    path: normalizeString(workloadEntry.path) || null,
    sha256: normalizeString(workloadEntry.sha256) || null,
  };
}

function parseTeacherReport(teacherJsonText) {
  try {
    return JSON.parse(String(teacherJsonText));
  } catch (error) {
    throw new Error(`Teacher report is not valid JSON: ${error.message}`);
  }
}

export async function runDistillReplay({
  teacherJsonText,
  workloadEntry = null,
}) {
  const teacherRaw = String(teacherJsonText || '');
  if (!teacherRaw.trim()) {
    throw new Error('Teacher report JSON is required.');
  }
  const teacherReport = parseTeacherReport(teacherRaw);
  const traceability = {
    teacherReportId: await computeReportId(teacherReport, teacherRaw),
    studentReportId: null,
    workloadPack: resolveWorkloadPackTraceability(workloadEntry),
  };
  const metrics = resolveTrainingMetrics(teacherReport);
  return {
    schemaVersion: 1,
    mode: 'replay-teacher',
    generatedAt: new Date().toISOString(),
    traceability,
    timeline: metrics.map((entry) => ({
      step: entry.step ?? null,
      objective: entry.objective ?? null,
      total_loss: entry.total_loss ?? null,
      step_time_ms: entry.step_time_ms ?? null,
      telemetry_alerts: entry.telemetry_alerts ?? [],
    })),
  };
}
