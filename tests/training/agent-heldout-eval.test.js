import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { parseJsonl } from '../../src/experimental/training/datasets/jsonl.js';
import {
  evaluateAgentHeldoutRows,
  extractFileReferences,
  summarizeAgentEvalReportRequirements,
} from '../../src/experimental/training/operator-agent-eval.js';
import { loadTrainingWorkloadPack } from '../../src/experimental/training/workloads.js';

const workload = await loadTrainingWorkloadPack(
  'src/experimental/training/workload-packs/lora-doppler-code-agent-tiny.json'
);
const evalDataset = workload.workload.evalDatasets.find(
  (entry) => entry.id === 'doppler-agent-heldout-suite-tiny-eval'
);
assert.ok(evalDataset?.agentEval);
assert.deepEqual(evalDataset.agentEval.categories, [
  'js_patching',
  'wgsl_review',
  'manifest_config_review',
  'reploid_vfs_status_tool_loop',
  'patch_applies',
  'no_hallucinated_files_tools',
]);

const datasetText = await readFile(evalDataset.datasetPath, 'utf8');
const datasetRows = parseJsonl(datasetText);
const candidates = datasetRows.map((row) => ({
  id: row.id,
  completion: row.completion,
}));

const directEval = evaluateAgentHeldoutRows(datasetRows, candidates, {
  policy: evalDataset.agentEval,
  patchStatuses: {
    'heldout-patch-applies-1': {
      applies: true,
    },
  },
});
assert.equal(directEval.passed, true);
assert.equal(directEval.passRate, 1);
assert.equal(directEval.categorySummary.patch_applies.passRate, 1);
assert.deepEqual(
  extractFileReferences(candidates[4].completion),
  ['tests/fixtures/training/agent-patch-targets/sample-tool-loop.js']
);

const hallucinated = evaluateAgentHeldoutRows(datasetRows.slice(0, 1), [{
  id: datasetRows[0].id,
  completion: 'Patch src/imaginary.js with a fallback and call curl.',
}], {
  policy: evalDataset.agentEval,
});
assert.equal(hallucinated.passed, false);
assert.match(JSON.stringify(hallucinated.rows[0].checks), /Unexpected file references/);
assert.match(JSON.stringify(hallucinated.rows[0].checks), /Unexpected tool references/);

const tmpRoot = await mkdtemp(join(tmpdir(), 'doppler-agent-heldout-eval-test-'));
try {
  const candidatesPath = join(tmpRoot, 'candidates.jsonl');
  const outPath = join(tmpRoot, 'agent-eval-report.json');
  await writeFile(
    candidatesPath,
    `${candidates.map((row) => JSON.stringify(row)).join('\n')}\n`,
    'utf8'
  );
  const result = spawnSync(process.execPath, [
    'tools/run-agent-heldout-eval.js',
    '--workload',
    'src/experimental/training/workload-packs/lora-doppler-code-agent-tiny.json',
    '--candidates',
    candidatesPath,
    '--patch-root',
    '.',
    '--out',
    outPath,
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(await readFile(outPath, 'utf8'));
  assert.equal(report.artifactType, 'training_eval_report');
  assert.equal(report.evalDatasetId, 'doppler-agent-heldout-suite-tiny-eval');
  assert.equal(report.primaryMetric, 'agent_heldout_pass_rate');
  assert.equal(report.agentEval.passed, true);
  assert.equal(report.metrics.agent_heldout_gate.score, 1);

  const summary = summarizeAgentEvalReportRequirements(workload.workload, [{
    ...report,
    reportPath: outPath,
  }]);
  assert.equal(summary.requiredCount, 1);
  assert.equal(summary.failedCount, 0);
  assert.deepEqual(summary.requirements[0].passingReportPaths, [outPath]);
} finally {
  await rm(tmpRoot, { recursive: true, force: true });
}

console.log('agent-heldout-eval.test: ok');
