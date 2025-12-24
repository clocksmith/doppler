/**
 * Comparison Utilities - Baseline comparison and statistical analysis
 */

import type { ComparisonResult, TTestResult } from './types.js';

// ============================================================================
// Comparison Utilities
// ============================================================================

export function compareResults(baseline: any, current: any): ComparisonResult[] {
  const results: ComparisonResult[] = [];
  const bm = baseline.metrics;
  const cm = current.metrics;

  const metrics: Array<{ key: string; name: string; lowerIsBetter: boolean }> = [
    { key: 'ttft_ms', name: 'TTFT', lowerIsBetter: true },
    { key: 'prefill_ms', name: 'Prefill', lowerIsBetter: true },
    { key: 'decode_ms_total', name: 'Decode Total', lowerIsBetter: true },
    { key: 'prefill_tokens_per_sec', name: 'Prefill tok/s', lowerIsBetter: false },
    { key: 'decode_tokens_per_sec', name: 'Decode tok/s', lowerIsBetter: false },
    { key: 'decode_ms_per_token_p50', name: 'Decode P50', lowerIsBetter: true },
    { key: 'decode_ms_per_token_p90', name: 'Decode P90', lowerIsBetter: true },
    { key: 'decode_ms_per_token_p99', name: 'Decode P99', lowerIsBetter: true },
    { key: 'gpu_submit_count_prefill', name: 'GPU Submits (prefill)', lowerIsBetter: true },
    { key: 'gpu_submit_count_decode', name: 'GPU Submits (decode)', lowerIsBetter: true },
  ];

  for (const { key, name, lowerIsBetter } of metrics) {
    const baseVal = bm[key];
    const currVal = cm[key];
    if (baseVal !== undefined && currVal !== undefined && baseVal !== 0) {
      const delta = currVal - baseVal;
      const deltaPercent = (delta / baseVal) * 100;
      const improved = lowerIsBetter ? delta < 0 : delta > 0;
      results.push({
        metric: name,
        baseline: baseVal,
        current: currVal,
        delta,
        deltaPercent,
        improved,
      });
    }
  }

  return results;
}

export function formatComparison(comparisons: ComparisonResult[]): string {
  const lines: string[] = [
    '',
    '='.repeat(60),
    'COMPARISON VS BASELINE',
    '='.repeat(60),
    '',
  ];

  for (const c of comparisons) {
    const sign = c.delta >= 0 ? '+' : '';
    const arrow = c.improved ? '\u2193' : c.delta === 0 ? '=' : '\u2191';
    const status = c.improved ? 'BETTER' : c.delta === 0 ? 'SAME' : 'WORSE';
    lines.push(
      `${c.metric.padEnd(20)} ${c.baseline.toFixed(1).padStart(10)} -> ${c.current.toFixed(1).padStart(10)}  ${sign}${c.deltaPercent.toFixed(1)}% ${arrow} ${status}`
    );
  }

  const improved = comparisons.filter((c) => c.improved).length;
  const regressed = comparisons.filter((c) => !c.improved && c.delta !== 0).length;
  lines.push('');
  lines.push(`Summary: ${improved} improved, ${regressed} regressed, ${comparisons.length - improved - regressed} unchanged`);

  return lines.join('\n');
}

// ============================================================================
// Statistical Significance (Welch's t-test)
// ============================================================================

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function variance(values: number[], m: number): number {
  return values.reduce((sum, v) => sum + (v - m) ** 2, 0) / (values.length - 1);
}

export function welchTTest(a: number[], b: number[]): TTestResult {
  const n1 = a.length;
  const n2 = b.length;
  const m1 = mean(a);
  const m2 = mean(b);
  const v1 = variance(a, m1);
  const v2 = variance(b, m2);

  const se1 = v1 / n1;
  const se2 = v2 / n2;
  const se = Math.sqrt(se1 + se2);

  const t = (m1 - m2) / se;

  const num = (se1 + se2) ** 2;
  const denom = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1);
  const df = num / denom;

  const pValue = tDistPValue(Math.abs(t), df);

  return {
    tStatistic: t,
    degreesOfFreedom: df,
    pValue,
    significant: pValue < 0.05,
    meanA: m1,
    meanB: m2,
    stdA: Math.sqrt(v1),
    stdB: Math.sqrt(v2),
  };
}

function tDistPValue(t: number, df: number): number {
  const x = t * Math.sqrt(df / (df - 2 + t * t));
  const p = 2 * (1 - normalCDF(x));
  return Math.max(0, Math.min(1, p));
}

function normalCDF(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);

  const t = 1 / (1 + p * x);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return 0.5 * (1 + sign * y);
}

export function formatTTestResult(metric: string, result: TTestResult): string {
  const sig = result.significant ? 'SIGNIFICANT' : 'not significant';
  return `${metric}: t=${result.tStatistic.toFixed(2)}, df=${result.degreesOfFreedom.toFixed(1)}, p=${result.pValue.toFixed(4)} (${sig})`;
}
