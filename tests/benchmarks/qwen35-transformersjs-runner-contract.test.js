import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const runnerPath = new URL('../../benchmarks/runners/transformersjs-runner.html', import.meta.url);
const runnerSource = await fs.readFile(runnerPath, 'utf8');

assert.match(runnerSource, /Qwen3_5ForConditionalGeneration/);
assert.match(runnerSource, /shouldUseQwen35ConditionalGeneration/);
assert.match(runnerSource, /runnerKind:\s*'qwen3_5_conditional_generation'/);
assert.match(runnerSource, /modelArchitecture:\s*'qwen3_5'/);

console.log('qwen35-transformersjs-runner-contract.test: ok');
