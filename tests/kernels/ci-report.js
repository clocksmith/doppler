#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const reportPath = process.argv[2] || 'results/report.json';
const summaryPath = process.env.GITHUB_STEP_SUMMARY;

const readJson = (filePath) => {
  const raw = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(raw);
};

if (!fs.existsSync(reportPath)) {
  console.error(`[ci-report] Missing Playwright report: ${reportPath}`);
  process.exit(1);
}

const data = readJson(reportPath);

const stats = {
  total: 0,
  passed: 0,
  failed: 0,
  skipped: 0,
  flaky: 0,
  timedOut: 0,
  interrupted: 0,
  durationMs: 0
};
const failures = [];

const formatTitle = (parts) => parts.filter(Boolean).join(' › ');

const summarizeTest = (spec, test, result) => {
  const title = formatTitle([spec?.title, test?.title]);
  const location = spec?.file || test?.location?.file || '';
  const line = spec?.line || test?.location?.line || null;
  const status = result?.status || test?.status || 'unknown';
  const expected = test?.expectedStatus || 'passed';
  const duration = result?.duration || 0;
  const errors = result?.errors?.length ? result.errors : [];

  return {
    title,
    location,
    line,
    status,
    expected,
    duration,
    errorMessage: errors[0]?.message || errors[0]?.value || null
  };
};

const classify = (spec, test) => {
  const results = test?.results || [];
  const finalResult = results[results.length - 1] || null;
  const summary = summarizeTest(spec, test, finalResult);
  const isFlaky = summary.status === 'passed' && results.some((r) => r.status !== 'passed');

  stats.total += 1;
  stats.durationMs += summary.duration;

  if (summary.status === 'passed') stats.passed += 1;
  if (summary.status === 'failed') stats.failed += 1;
  if (summary.status === 'skipped') stats.skipped += 1;
  if (summary.status === 'timedOut') stats.timedOut += 1;
  if (summary.status === 'interrupted') stats.interrupted += 1;
  if (isFlaky) stats.flaky += 1;

  if (summary.status !== 'passed' && summary.status !== 'skipped') {
    failures.push(summary);
  }
};

const walkSuite = (suite) => {
  if (!suite) return;
  if (Array.isArray(suite.specs)) {
    for (const spec of suite.specs) {
      for (const test of spec.tests || []) {
        classify(spec, test);
      }
    }
  }
  for (const child of suite.suites || []) {
    walkSuite(child);
  }
};

for (const suite of data.suites || []) {
  walkSuite(suite);
}

const formatDuration = (ms) => {
  if (!Number.isFinite(ms) || ms <= 0) return '0s';
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder}s`;
};

const summaryLines = [
  '## Kernel Test Report',
  '',
  `Total: ${stats.total}`,
  `Passed: ${stats.passed}`,
  `Failed: ${stats.failed}`,
  `Skipped: ${stats.skipped}`,
  `Flaky: ${stats.flaky}`,
  `Timed out: ${stats.timedOut}`,
  `Interrupted: ${stats.interrupted}`,
  `Duration: ${formatDuration(stats.durationMs)}`
];

if (failures.length > 0) {
  summaryLines.push('', '### Failures');
  for (const failure of failures.slice(0, 10)) {
    const location = failure.location
      ? `${path.basename(failure.location)}${failure.line ? `:${failure.line}` : ''}`
      : 'unknown';
    summaryLines.push(`- ${failure.title || '(untitled)'} (${location})`);
    if (failure.errorMessage) {
      summaryLines.push(`  ${failure.errorMessage.split('\n')[0]}`);
    }
  }
  if (failures.length > 10) {
    summaryLines.push(`- ...and ${failures.length - 10} more`);
  }
}

const output = summaryLines.join('\n');
console.log(output);

if (summaryPath) {
  try {
    fs.appendFileSync(summaryPath, `${output}\n`);
  } catch (err) {
    console.error(`[ci-report] Failed to write summary: ${err.message}`);
  }
}
