const FILE_REF_PATTERN = /(?:^|[\s`"'([])([A-Za-z0-9_./-]+\.(?:d\.ts|js|json|jsonl|wgsl|md|html|css|py|zig|toml|yaml|yml|sh))(?:$|[\s`"',):;\]])/g;

const KNOWN_TOOL_NAMES = Object.freeze([
  'apply_patch',
  'exec_command',
  'write_stdin',
  'request_user_input',
  'read_mcp_resource',
  'list_mcp_resources',
  'list_mcp_resource_templates',
  'web.run',
  'rg',
  'git',
  'node',
  'npm',
  'python3',
  'curl',
]);

function isObjectRecord(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

function normalizeString(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeLower(value) {
  return normalizeString(value).toLowerCase();
}

function normalizeStringArray(value) {
  return Array.isArray(value)
    ? value.map((entry) => normalizeString(entry)).filter(Boolean)
    : [];
}

function normalizePathReference(value) {
  let text = normalizeString(value)
    .replace(/^[`"'(]+/, '')
    .replace(/[)`"',:;]+$/, '');
  while (text.startsWith('./')) {
    text = text.slice(2);
  }
  if (text.startsWith('a/') || text.startsWith('b/')) {
    text = text.slice(2);
  }
  return text;
}

function resolveRowId(row, index) {
  const id = normalizeString(row?.id) || normalizeString(row?.rowId);
  return id || `row-${index + 1}`;
}

function readCompletion(row) {
  for (const key of ['completion', 'output', 'response', 'candidate', 'assistant']) {
    const value = row?.[key];
    if (typeof value === 'string') {
      return value;
    }
    if (isObjectRecord(value) && typeof value.content === 'string') {
      return value.content;
    }
  }
  return '';
}

function buildCandidateMap(candidates) {
  const byId = new Map();
  const rows = Array.isArray(candidates) ? candidates : [];
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    if (!isObjectRecord(row)) continue;
    byId.set(resolveRowId(row, index), row);
  }
  return byId;
}

export function extractFileReferences(text) {
  const refs = new Set();
  const source = String(text || '');
  for (const match of source.matchAll(FILE_REF_PATTERN)) {
    const ref = normalizePathReference(match[1]);
    if (ref) refs.add(ref);
  }
  return [...refs].sort((left, right) => left.localeCompare(right));
}

export function extractToolReferences(text, knownTools = KNOWN_TOOL_NAMES) {
  const source = String(text || '');
  const refs = [];
  for (const toolName of knownTools) {
    const escaped = toolName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`(^|[^A-Za-z0-9_-])${escaped}([^A-Za-z0-9_-]|$)`, 'i');
    if (pattern.test(source)) {
      refs.push(toolName);
    }
  }
  return refs.sort((left, right) => left.localeCompare(right));
}

function mergePolicyArray(row, policy, field) {
  return [
    ...normalizeStringArray(policy?.[field]),
    ...normalizeStringArray(row?.sourceFiles),
    ...normalizeStringArray(row?.agentEval?.[field]),
  ];
}

function resolveRowPolicy(row, policy = {}) {
  const rowPolicy = isObjectRecord(row?.agentEval) ? row.agentEval : {};
  return {
    categories: normalizeStringArray(rowPolicy.categories).length
      ? normalizeStringArray(rowPolicy.categories)
      : normalizeStringArray(policy.categories),
    requiredTerms: normalizeStringArray(rowPolicy.requiredTerms),
    forbiddenTerms: normalizeStringArray(rowPolicy.forbiddenTerms),
    requirePatchApplies: rowPolicy.requirePatchApplies === true || policy.requirePatchApplies === true,
    requireNoHallucinatedFiles: rowPolicy.requireNoHallucinatedFiles === true
      || policy.requireNoHallucinatedFiles === true,
    requireNoHallucinatedTools: rowPolicy.requireNoHallucinatedTools === true
      || policy.requireNoHallucinatedTools === true,
    allowedFiles: mergePolicyArray(row, policy, 'allowedFiles'),
    allowedTools: [
      ...normalizeStringArray(policy.allowedTools),
      ...normalizeStringArray(rowPolicy.allowedTools),
    ],
  };
}

function evaluateTermChecks(text, rowPolicy) {
  const checks = [];
  const normalized = normalizeLower(text);
  for (const term of rowPolicy.requiredTerms) {
    const ok = normalized.includes(term.toLowerCase());
    checks.push({
      name: 'required_term',
      ok,
      term,
      error: ok ? null : `Missing required term "${term}".`,
    });
  }
  for (const term of rowPolicy.forbiddenTerms) {
    const ok = !normalized.includes(term.toLowerCase());
    checks.push({
      name: 'forbidden_term',
      ok,
      term,
      error: ok ? null : `Found forbidden term "${term}".`,
    });
  }
  return checks;
}

function evaluateFileChecks(text, rowPolicy) {
  if (rowPolicy.requireNoHallucinatedFiles !== true) return [];
  const refs = extractFileReferences(text);
  const allowed = new Set(rowPolicy.allowedFiles.map((entry) => normalizePathReference(entry)).filter(Boolean));
  const unexpected = refs.filter((ref) => !allowed.has(ref));
  return [{
    name: 'no_hallucinated_files',
    ok: unexpected.length === 0,
    references: refs,
    allowedFiles: [...allowed].sort((left, right) => left.localeCompare(right)),
    unexpected,
    error: unexpected.length === 0 ? null : `Unexpected file references: ${unexpected.join(', ')}`,
  }];
}

function evaluateToolChecks(text, rowPolicy) {
  if (rowPolicy.requireNoHallucinatedTools !== true) return [];
  const refs = extractToolReferences(text);
  const allowed = new Set(rowPolicy.allowedTools);
  const unexpected = refs.filter((ref) => !allowed.has(ref));
  return [{
    name: 'no_hallucinated_tools',
    ok: unexpected.length === 0,
    references: refs,
    allowedTools: [...allowed].sort((left, right) => left.localeCompare(right)),
    unexpected,
    error: unexpected.length === 0 ? null : `Unexpected tool references: ${unexpected.join(', ')}`,
  }];
}

function evaluatePatchCheck(rowId, rowPolicy, patchStatuses) {
  if (rowPolicy.requirePatchApplies !== true) return [];
  const status = isObjectRecord(patchStatuses?.[rowId]) ? patchStatuses[rowId] : null;
  const ok = status?.applies === true;
  return [{
    name: 'patch_applies',
    ok,
    status,
    error: ok ? null : (status?.error || 'Patch apply evidence is required.'),
  }];
}

function buildCategorySummary(rows, requiredCategories) {
  const summary = {};
  for (const category of requiredCategories) {
    summary[category] = {
      total: 0,
      passed: 0,
      passRate: 0,
    };
  }
  for (const row of rows) {
    for (const category of row.categories) {
      if (!summary[category]) {
        summary[category] = { total: 0, passed: 0, passRate: 0 };
      }
      summary[category].total += 1;
      if (row.passed) {
        summary[category].passed += 1;
      }
    }
  }
  for (const entry of Object.values(summary)) {
    entry.passRate = entry.total === 0 ? 0 : entry.passed / entry.total;
  }
  return summary;
}

export function evaluateAgentHeldoutRows(datasetRows, candidateRows, options = {}) {
  const policy = isObjectRecord(options.policy) ? options.policy : {};
  const candidatesById = buildCandidateMap(candidateRows);
  const patchStatuses = isObjectRecord(options.patchStatuses) ? options.patchStatuses : {};
  const minPassRate = Number.isFinite(Number(policy.minPassRate)) ? Number(policy.minPassRate) : 1;
  const rows = [];
  const sourceRows = Array.isArray(datasetRows) ? datasetRows : [];

  for (let index = 0; index < sourceRows.length; index += 1) {
    const row = sourceRows[index];
    const rowId = resolveRowId(row, index);
    const candidate = candidatesById.get(rowId) || null;
    const completion = readCompletion(candidate);
    const rowPolicy = resolveRowPolicy(row, policy);
    const checks = [];
    if (!candidate) {
      checks.push({
        name: 'candidate_present',
        ok: false,
        error: `No candidate completion found for row "${rowId}".`,
      });
    } else {
      checks.push({
        name: 'candidate_present',
        ok: true,
      });
      checks.push(...evaluateTermChecks(completion, rowPolicy));
      checks.push(...evaluateFileChecks(completion, rowPolicy));
      checks.push(...evaluateToolChecks(completion, rowPolicy));
      checks.push(...evaluatePatchCheck(rowId, rowPolicy, patchStatuses));
    }
    rows.push({
      id: rowId,
      categories: rowPolicy.categories,
      passed: checks.every((check) => check.ok === true),
      checks,
    });
  }

  const passedCount = rows.filter((row) => row.passed).length;
  const passRate = sourceRows.length === 0 ? 0 : passedCount / sourceRows.length;
  const requiredCategories = normalizeStringArray(policy.categories);
  const categorySummary = buildCategorySummary(rows, requiredCategories);
  const missingCategories = requiredCategories.filter((category) => !categorySummary[category]?.total);
  return {
    schemaVersion: 1,
    suiteId: normalizeString(policy.suiteId) || null,
    totalRows: sourceRows.length,
    passedRows: passedCount,
    failedRows: sourceRows.length - passedCount,
    passRate,
    minPassRate,
    passed: passRate >= minPassRate && missingCategories.length === 0,
    requiredCategories,
    missingCategories,
    categorySummary,
    rows,
  };
}

function readAgentEvalReceipt(report) {
  if (!isObjectRecord(report)) return null;
  if (isObjectRecord(report.agentEval)) return report.agentEval;
  if (isObjectRecord(report.heldoutGate)) return report.heldoutGate;
  const metric = report.metrics?.agent_heldout_gate;
  return isObjectRecord(metric) ? metric : null;
}

function reportPassesAgentEval(report, config) {
  const receipt = readAgentEvalReceipt(report);
  if (!receipt) return false;
  const score = Number(receipt.passRate ?? receipt.score);
  const minPassRate = Number(config?.minPassRate);
  return receipt.passed === true
    && Number.isFinite(score)
    && Number.isFinite(minPassRate)
    && score >= minPassRate;
}

export function summarizeAgentEvalReportRequirements(workload, reports) {
  const evalDatasets = Array.isArray(workload?.evalDatasets) ? workload.evalDatasets : [];
  const required = evalDatasets.filter((entry) => isObjectRecord(entry?.agentEval));
  const reportRows = Array.isArray(reports) ? reports : [];
  const requirements = required.map((evalDataset) => {
    const matchingReports = reportRows.filter((report) => report?.evalDatasetId === evalDataset.id);
    const passingReports = matchingReports.filter((report) => reportPassesAgentEval(report, evalDataset.agentEval));
    return {
      evalDatasetId: evalDataset.id,
      suiteId: evalDataset.agentEval.suiteId,
      minPassRate: evalDataset.agentEval.minPassRate,
      requiredCategories: evalDataset.agentEval.categories,
      reportCount: matchingReports.length,
      passed: passingReports.length > 0,
      passingReportPaths: passingReports.map((report) => report.reportPath || null).filter(Boolean),
    };
  });
  return {
    requiredCount: requirements.length,
    passedCount: requirements.filter((entry) => entry.passed).length,
    failedCount: requirements.filter((entry) => !entry.passed).length,
    requirements,
  };
}
