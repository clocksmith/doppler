#!/usr/bin/env node

import { writeGepaTeacherTraces } from '../src/experimental/training/datasets/gepa-frontier.js';

function parseArgs(argv) {
  const options = {
    input: null,
    out: null,
    teacherModelId: null,
    studentBaseModelId: null,
    domain: null,
    taskKind: null,
    sourcePolicyId: null,
    sourceFiles: null,
    license: null,
    includeFailures: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const readValue = () => {
      const value = argv[index + 1];
      if (!value) {
        throw new Error(`${arg} requires a value.`);
      }
      index += 1;
      return value;
    };
    if (arg === '--input') {
      options.input = readValue();
      continue;
    }
    if (arg === '--out') {
      options.out = readValue();
      continue;
    }
    if (arg === '--teacher-model-id') {
      options.teacherModelId = readValue();
      continue;
    }
    if (arg === '--student-base-model-id') {
      options.studentBaseModelId = readValue();
      continue;
    }
    if (arg === '--domain') {
      options.domain = readValue();
      continue;
    }
    if (arg === '--task-kind') {
      options.taskKind = readValue();
      continue;
    }
    if (arg === '--source-policy-id') {
      options.sourcePolicyId = readValue();
      continue;
    }
    if (arg === '--source-files') {
      options.sourceFiles = readValue().split(',').map((entry) => entry.trim()).filter(Boolean);
      continue;
    }
    if (arg === '--license') {
      options.license = readValue();
      continue;
    }
    if (arg === '--include-failures') {
      options.includeFailures = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!options.input) {
    throw new Error('--input is required.');
  }
  if (!options.out) {
    throw new Error('--out is required.');
  }
  if (!options.teacherModelId) {
    throw new Error('--teacher-model-id is required.');
  }
  return options;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const result = await writeGepaTeacherTraces(options.input, options.out, options);
  console.log(JSON.stringify({
    ok: true,
    inputPath: result.inputPath,
    outputPath: result.outputPath,
    candidateCount: result.candidateCount,
    rowCount: result.rowCount,
    lineage: result.lineage,
  }, null, 2));
}

main().catch((error) => {
  console.error(`[gepa-frontier] ${error.message}`);
  process.exit(1);
});
