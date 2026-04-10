import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const runnerPath = new URL('../../benchmarks/runners/transformersjs-runner.html', import.meta.url);
const runnerSource = await fs.readFile(runnerPath, 'utf8');

assert.match(runnerSource, /window\.__runGreedyPromptPack\s*=\s*async/);
assert.match(runnerSource, /generatedTokenIds\.push\(\.\.\.normalizeGeneratedTokenIds\(tokens\)\)/);
assert.match(runnerSource, /firstTokenId:\s*generatedTokenIds\[0\]\s*\?\?\s*null/);
assert.match(runnerSource, /strictDeterministicGreedy:\s*generationSettings\.strictDeterministicGreedy/);

console.log('transformersjs-runner-greedy-pack-contract.test: ok');
