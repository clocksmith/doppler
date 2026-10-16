/** @type {import('./report-ranking.js').resolveComparableReportMetric} */
export function resolveComparableReportMetric(report, metric) {
  if (!report || typeof report !== 'object') return null;
  const direct = report[metric];
  if (typeof direct === 'number' && Number.isFinite(direct)) {
    return direct;
  }
  const nested = report.metrics?.[metric];
  if (typeof nested === 'number' && Number.isFinite(nested)) {
    return nested;
  }
  if (nested && typeof nested === 'object' && typeof nested.score === 'number' && Number.isFinite(nested.score)) {
    return nested.score;
  }
  return null;
}

/** @type {import('./report-ranking.js').sortTrainingReports} */
export function sortTrainingReports(reports, workload = null) {
  const metric = String(workload?.selectionMetric || reports[0]?.primaryMetric || 'primaryScore').trim();
  const goal = String(workload?.selectionGoal || (metric === 'loss' ? 'min' : 'max')).trim();
  return reports
    .slice()
    .sort((left, right) => {
      const missingScore = goal === 'min' ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
      const leftScore = resolveComparableReportMetric(left, metric) ?? missingScore;
      const rightScore = resolveComparableReportMetric(right, metric) ?? missingScore;
      if (goal === 'min') return leftScore - rightScore;
      return rightScore - leftScore;
    });
}

/** @type {import('./report-ranking.js').collectRequiredImprovementEvalIds} */
export function collectRequiredImprovementEvalIds(workload) {
  const evalDatasets = Array.isArray(workload?.evalDatasets) ? workload.evalDatasets : [];
  return evalDatasets
    .filter((entry) => entry?.quality?.requireImprovement === true)
    .map((entry) => entry.id);
}

/** @type {import('./report-ranking.js').summarizeQualityClaims} */
export function summarizeQualityClaims(reports) {
  const claims = reports
    .map((report) => report.qualityClaim)
    .filter((claim) => claim != null && typeof claim === 'object');
  return {
    count: claims.length,
    improvedCount: claims.filter((claim) => claim.improved === true).length,
    requiredCount: claims.filter((claim) => claim.requireImprovement === true).length,
    failedRequiredCount: claims.filter((claim) => claim.requireImprovement === true && claim.improved !== true).length,
  };
}
