import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  buildTeacherTracesFromGepaFrontier,
  loadGepaFrontier,
  writeGepaTeacherTraces,
} from '../../src/experimental/training/datasets/gepa-frontier.js';
import { loadTeacherTraceDataset } from '../../src/experimental/training/datasets/teacher-traces.js';
import { loadTextPairsDataset } from '../../src/experimental/training/datasets/text-pairs.js';

const fixturePath = 'tests/fixtures/training/gepa-frontier-doppler-tiny.json';
const frontier = await loadGepaFrontier(fixturePath);
assert.equal(frontier.candidates.length, 2);
assert.deepEqual(frontier.lineage.candidateIds, ['gepa-pareto-0001', 'gepa-pareto-0002']);
assert.deepEqual(frontier.lineage.objectiveNames, ['accuracy', 'cost', 'efficiency', 'robustness']);

const rows = buildTeacherTracesFromGepaFrontier(frontier.candidates, {
  teacherModelId: 'zai-org/GLM-5.2',
  studentBaseModelId: 'gemma-3-270m-it-f16-af32',
  domain: 'doppler',
});
assert.equal(rows.length, 3);
assert.equal(rows[0].sourcePolicyId, 'gepa:gepa-pareto-0001');
assert.equal(rows[0].gepaCandidateId, 'gepa-pareto-0001');
assert.match(rows[0].prompt, /Task:/);

const tmpRoot = await mkdtemp(join(tmpdir(), 'doppler-gepa-frontier-'));
const outPath = join(tmpRoot, 'teacher-traces.jsonl');
const written = await writeGepaTeacherTraces(fixturePath, outPath, {
  teacherModelId: 'zai-org/GLM-5.2',
  studentBaseModelId: 'gemma-3-270m-it-f16-af32',
  domain: 'doppler',
});
assert.equal(written.candidateCount, 2);
assert.equal(written.rowCount, 3);
const raw = await readFile(outPath, 'utf8');
assert.match(raw, /"gepaCandidateId":"gepa-pareto-0002"/);

const teacherTraces = await loadTeacherTraceDataset(outPath);
assert.equal(teacherTraces.rowCount, 3);
assert.deepEqual(teacherTraces.lineage.gepaCandidateIds, ['gepa-pareto-0001', 'gepa-pareto-0002']);

const textPairs = await loadTextPairsDataset(outPath);
assert.equal(textPairs.rowCount, 3);
assert.match(textPairs.rows[1].completion, /storage read/);

console.log('gepa-frontier-dataset.test: ok');
