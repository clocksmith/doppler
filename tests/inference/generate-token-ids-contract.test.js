import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Verify that generateTokenIds handles the same contract semantics as the
// _runDecodeLoop path: abort signal, EOS, stop sequences, finiteness fallback,
// and stats/cleanup. This is a structural parity guard — if someone rewrites
// generateTokenIds and drops one of these checks, this test fails.

const generatorSource = readFileSync(
  new URL('../../src/inference/pipelines/text/generator.js', import.meta.url),
  'utf8'
);

// Extract the generateTokenIds method body. It starts at "async generateTokenIds"
// and ends at the next top-level method or closing brace pattern.
const startMarker = 'async generateTokenIds(prompt';
const startIdx = generatorSource.indexOf(startMarker);
assert.ok(startIdx !== -1, 'generateTokenIds method not found in generator.js');

// Find a reasonable chunk — the method is ~90 lines so 3000 chars is generous
const methodBody = generatorSource.slice(startIdx, startIdx + 4000);

// 1. Abort signal must be checked inside the decode loop
assert.ok(
  methodBody.includes('signal?.aborted'),
  'generateTokenIds must check options.signal?.aborted in the decode loop'
);

// 2. EOS / stop token check
assert.ok(
  methodBody.includes('isStopToken('),
  'generateTokenIds must check isStopToken for EOS handling'
);

// 3. Stop sequences must be checked
assert.ok(
  methodBody.includes('stopSequences'),
  'generateTokenIds must check opts.stopSequences for string-based stop conditions'
);

// 4. Finiteness fallback: retry + consume + close
assert.ok(
  methodBody.includes('shouldRetryWithFinitenessFallback'),
  'generateTokenIds must handle finiteness fallback retry'
);
assert.ok(
  methodBody.includes('_consumeFinitenessFallbackToken'),
  'generateTokenIds must call _consumeFinitenessFallbackToken on each iteration'
);
assert.ok(
  methodBody.includes('_closeFinitenessFallbackWindow'),
  'generateTokenIds must close finiteness fallback window in finally'
);

// 5. Execution plan reset in finally
assert.ok(
  methodBody.includes('resetActiveExecutionPlan'),
  'generateTokenIds must reset execution plan in finally'
);

// 6. isGenerating flag cleared in finally
assert.ok(
  methodBody.includes('isGenerating = false'),
  'generateTokenIds must clear isGenerating in finally'
);

// 7. Stats are tracked
assert.ok(
  methodBody.includes('stats.decodeTimeMs'),
  'generateTokenIds must track decode time'
);
assert.ok(
  methodBody.includes('stats.tokensGenerated'),
  'generateTokenIds must track tokens generated count'
);
assert.ok(
  methodBody.includes('stats.prefillTimeMs'),
  'generateTokenIds must track prefill time'
);

// 8. markKernelCacheWarmed is called
assert.ok(
  methodBody.includes('markKernelCacheWarmed'),
  'generateTokenIds must call markKernelCacheWarmed for consistency with _runDecodeLoop'
);

console.log('generate-token-ids-contract.test: ok');
