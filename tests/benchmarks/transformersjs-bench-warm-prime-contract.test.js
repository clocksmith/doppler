import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const benchPath = new URL('../../benchmarks/runners/transformersjs-bench.js', import.meta.url);
const benchSource = await fs.readFile(benchPath, 'utf8');

assert.match(benchSource, /page\.on\('requestfailed'/);
assert.match(benchSource, /prompt,\s*\n\s*useChatTemplate,\s*\n\s*maxNewTokens:\s*1,/);
assert.match(benchSource, /const LOCAL_MODEL_RELAY_PATH = '\/__tjs_local_models';/);
assert.match(benchSource, /if \(browserBaseUrl && localModelPath\)/);
assert.match(benchSource, /runnerParams\.set\('localModelPath', LOCAL_MODEL_RELAY_PATH\)/);

console.log('transformersjs-bench-warm-prime-contract.test: ok');
