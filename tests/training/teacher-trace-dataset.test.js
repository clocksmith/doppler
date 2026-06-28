import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  loadTeacherTraceDataset,
  parseTeacherTraceDataset,
  writeTeacherTraceTextPairs,
} from '../../src/experimental/training/datasets/teacher-traces.js';
import { loadTextPairsDataset } from '../../src/experimental/training/datasets/text-pairs.js';

const fixturePath = 'tests/fixtures/training/doppler-code-agent-teacher-traces-tiny.jsonl';
const dataset = await loadTeacherTraceDataset(fixturePath);
assert.equal(dataset.rowCount, 3);
assert.deepEqual(dataset.lineage.teacherModelIds, ['zai-org/GLM-5.2']);
assert.deepEqual(dataset.lineage.gepaCandidateIds, ['gepa-pareto-0001', 'gepa-pareto-0002']);
assert.deepEqual(dataset.lineage.policyIds, ['doppler-agent-policy-v1', 'doppler-review-policy-v1']);
assert.deepEqual(dataset.lineage.taskKinds, ['cross_cutting_review', 'js_json_write', 'wgsl_write']);
assert.deepEqual(dataset.rows[0].sourceFiles, [
  'src/experimental/training/workloads.js',
  'tests/config/training-workload-loader-v2.test.js',
]);
assert.equal(dataset.rows[0].generationParams.maxTokens, 512);
assert.equal(dataset.rows[0].license, 'Apache-2.0');
assert.equal(dataset.rows[1].promptField, 'messages');
assert.match(dataset.rows[1].prompt, /system: You are a Doppler WGSL patching assistant/);
assert.equal(dataset.textPairs[0].teacherModelId, 'zai-org/GLM-5.2');
assert.equal(dataset.textPairs[0].policyId, 'doppler-agent-policy-v1');

assert.throws(
  () => parseTeacherTraceDataset('{"prompt":"a","completion":"b"}\n'),
  /requires teacherModelId/
);

const tmpRoot = await mkdtemp(join(tmpdir(), 'doppler-teacher-traces-'));
const outPath = join(tmpRoot, 'text-pairs.jsonl');
const written = await writeTeacherTraceTextPairs(fixturePath, outPath);
assert.equal(written.rowCount, 3);
const outputText = await readFile(outPath, 'utf8');
assert.match(outputText, /"taskKind":"wgsl_write"/);
const textPairs = await loadTextPairsDataset(outPath);
assert.equal(textPairs.rowCount, 3);
assert.equal(textPairs.rows[2].id, 'trace-review-1');
assert.match(textPairs.rows[2].completion, /schema compatibility/);

console.log('teacher-trace-dataset.test: ok');
