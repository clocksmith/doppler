#!/usr/bin/env node

import process from 'node:process';
import { pathToFileURL } from 'node:url';
import { buildGoalCompletionReport } from './check-goal-completion.js';

function parseArgs(argv) {
  const args = {
    json: false,
  };
  for (const token of argv) {
    if (token === '--json') {
      args.json = true;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return args;
}

function formatGoal(goal) {
  const claim = goal.claimAllowed ? 'claimable' : 'not claimable';
  const blockers = goal.blockers.length > 0 ? goal.blockers.join(', ') : 'none';
  return [
    `## ${goal.label}`,
    '',
    `- id: ${goal.id}`,
    `- status: ${goal.status}`,
    `- completion: ${goal.completionPercent}% (${goal.claimableRows}/${goal.rows} rows claimable)`,
    `- public claim: ${claim}`,
    `- blockers: ${blockers}`,
    '',
  ].join('\n');
}

function formatMarkdown(report) {
  const lines = [
    '# Doppler Goal Completion Report',
    '',
    `- matrix: ${report.matrixPath}`,
    `- status: ${report.ok ? 'ok' : 'invalid'}`,
    '',
  ];
  if (!report.ok) {
    lines.push('## Validation errors', '');
    for (const error of report.errors) {
      lines.push(`- ${error}`);
    }
    lines.push('');
  }
  for (const goal of report.goals) {
    lines.push(formatGoal(goal));
  }
  return lines.join('\n').trimEnd();
}

export async function renderGoalCompletionSummary(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const report = await buildGoalCompletionReport();
  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(formatMarkdown(report));
  }
  if (!report.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  renderGoalCompletionSummary().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
