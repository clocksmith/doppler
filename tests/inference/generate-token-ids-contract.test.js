import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// =============================================================================
// generateTokenIds / _runDecodeLoop semantic parity contract test
//
// generateTokenIds() is an intentional lean fast path that bypasses
// _runDecodeLoop(). This test proves both paths honor the same contract
// obligations. If a future change drops a contract check from either path,
// this test fails and forces explicit acknowledgment.
//
// What this does NOT test (requires a real GPU pipeline):
//   - Numeric output parity (use bench-text-decode-paths.js --teacher-forced)
//   - GPU buffer lifecycle and cleanup
//   - Actual WebGPU dispatch behavior
//
// What this DOES test:
//   - Both paths check the same contract points (structural parity)
//   - Contract points are in the correct structural positions
//   - Stats fields are tracked equivalently
//   - Cleanup is in finally blocks
// =============================================================================

const generatorSource = readFileSync(
  new URL('../../src/inference/pipelines/text/generator.js', import.meta.url),
  'utf8'
);

// === Extract method bodies ===

function extractMethodBody(source, marker) {
  const startIdx = source.indexOf(marker);
  assert.ok(startIdx !== -1, `Method "${marker}" not found in generator.js`);

  // Find the opening brace of the method body (skip any braces in
  // default parameter values like `options = {}`).
  // Strategy: find the `) {` pattern that opens the method body.
  const sigEnd = source.indexOf(') {', startIdx);
  assert.ok(sigEnd !== -1, `Could not find method body opening for "${marker}"`);
  const bodyOpenBrace = sigEnd + 2; // index of the `{`

  let depth = 0;
  for (let i = bodyOpenBrace; i < source.length; i++) {
    if (source[i] === '{') {
      depth++;
    } else if (source[i] === '}') {
      depth--;
      if (depth === 0) {
        return source.slice(startIdx, i + 1);
      }
    }
  }
  throw new Error(`Could not find balanced body for "${marker}"`);
}

const generateTokenIdsBody = extractMethodBody(generatorSource, 'async generateTokenIds(prompt');
const runDecodeLoopBody = extractMethodBody(generatorSource, 'async *_runDecodeLoop(generatedIds');
const generateTokensInternalBody = extractMethodBody(generatorSource, 'async *_generateTokensInternal(prompt');

// Sanity: bodies should be substantial
assert.ok(generateTokenIdsBody.length > 500, `generateTokenIds body too short (${generateTokenIdsBody.length})`);
assert.ok(runDecodeLoopBody.length > 500, `_runDecodeLoop body too short (${runDecodeLoopBody.length})`);
assert.ok(generateTokensInternalBody.length > 500, `_generateTokensInternal body too short (${generateTokensInternalBody.length})`);

// === Contract obligations that both decode surfaces must honor ===

const DECODE_CONTRACT = [
  {
    id: 'abort-signal',
    description: 'Abort signal checked in decode loop',
    generateTokenIdsPattern: 'signal?.aborted',
    runDecodeLoopPattern: 'signal?.aborted',
  },
  {
    id: 'eos-stop-token',
    description: 'EOS / stop token ID check',
    generateTokenIdsPattern: 'stopTokenIds',
    runDecodeLoopPattern: 'isStopToken(',
  },
  {
    id: 'stop-sequences',
    description: 'String-based stop sequence check',
    generateTokenIdsPattern: '_shouldStopAfterAppendedToken(',
    runDecodeLoopPattern: 'stopSequences',
  },
  {
    id: 'finiteness-retry',
    description: 'Finiteness fallback retry on decode error',
    generateTokenIdsPattern: '_shouldUseFinitenessFallback',
    runDecodeLoopPattern: '_shouldUseFinitenessFallback',
  },
  {
    id: 'finiteness-consume',
    description: 'Finiteness fallback token consumed each iteration',
    generateTokenIdsPattern: '_consumeFinitenessFallbackToken',
    runDecodeLoopPattern: '_consumeFinitenessFallbackToken',
  },
  {
    id: 'kernel-cache-warmed',
    description: 'Kernel cache marked warmed at decode entry',
    generateTokenIdsPattern: 'markKernelCacheWarmed',
    runDecodeLoopPattern: 'markKernelCacheWarmed',
  },
];

// Stats tracked equivalently in both surfaces
const STATS_CONTRACT = [
  'stats.decodeTimeMs',
  'stats.tokensGenerated',
  'stats.decodeTokens',
];

// Cleanup that must be in finally blocks
const CLEANUP_CONTRACT = [
  'resetActiveExecutionPlan',
  'isGenerating = false',
  '_closeFinitenessFallbackWindow',
];

// === 1. generateTokenIds has all decode contract points ===

for (const { id, description, generateTokenIdsPattern } of DECODE_CONTRACT) {
  assert.ok(
    generateTokenIdsBody.includes(generateTokenIdsPattern),
    `generateTokenIds missing contract: ${id} (${description})`
  );
}

// === 2. _runDecodeLoop has all decode contract points ===

for (const { id, description, runDecodeLoopPattern } of DECODE_CONTRACT) {
  assert.ok(
    runDecodeLoopBody.includes(runDecodeLoopPattern),
    `_runDecodeLoop missing contract: ${id} (${description})`
  );
}

// === 3. Stats fields tracked in generateTokenIds ===

for (const stat of STATS_CONTRACT) {
  assert.ok(
    generateTokenIdsBody.includes(stat),
    `generateTokenIds must track ${stat}`
  );
}

// === 4. Stats fields tracked in _runDecodeLoop ===

for (const stat of STATS_CONTRACT) {
  assert.ok(
    runDecodeLoopBody.includes(stat),
    `_runDecodeLoop must track ${stat}`
  );
}

// === 5. Cleanup in generateTokenIds finally block ===

const tokenIdsFinallyIdx = generateTokenIdsBody.lastIndexOf('finally');
assert.ok(tokenIdsFinallyIdx !== -1, 'generateTokenIds must have a finally block');
const tokenIdsFinallyBlock = generateTokenIdsBody.slice(tokenIdsFinallyIdx);

for (const cleanup of CLEANUP_CONTRACT) {
  assert.ok(
    tokenIdsFinallyBlock.includes(cleanup),
    `generateTokenIds finally block must include ${cleanup}`
  );
}

// === 6. Cleanup in _generateTokensInternal finally block ===

const internalFinallyIdx = generateTokensInternalBody.lastIndexOf('finally');
assert.ok(internalFinallyIdx !== -1, '_generateTokensInternal must have a finally block');
const internalFinallyBlock = generateTokensInternalBody.slice(internalFinallyIdx);

for (const cleanup of CLEANUP_CONTRACT) {
  assert.ok(
    internalFinallyBlock.includes(cleanup),
    `_generateTokensInternal finally block must include ${cleanup}`
  );
}

// === 7. generateTokenIds prefill stats ===

assert.ok(
  generateTokenIdsBody.includes('stats.prefillTimeMs'),
  'generateTokenIds must track prefill time'
);
assert.ok(
  generateTokenIdsBody.includes('stats.ttftMs'),
  'generateTokenIds must track time to first token'
);
assert.ok(
  generateTokenIdsBody.includes('stats.totalTimeMs'),
  'generateTokenIds must track total time'
);
assert.ok(
  generateTokenIdsBody.includes('stats.prefillTokens'),
  'generateTokenIds must track prefill token count'
);

// === 8. _generateTokensInternal prefill stats ===

assert.ok(
  generateTokensInternalBody.includes('stats.prefillTimeMs'),
  '_generateTokensInternal must track prefill time'
);
assert.ok(
  generateTokensInternalBody.includes('stats.ttftMs'),
  '_generateTokensInternal must track time to first token'
);
assert.ok(
  generateTokensInternalBody.includes('stats.totalTimeMs'),
  '_generateTokensInternal must track total time'
);

// === 9. Abort signal is inside the while loop, not before it ===

const tokenIdsWhileIdx = generateTokenIdsBody.indexOf('while (tokenIds.length');
assert.ok(tokenIdsWhileIdx !== -1, 'generateTokenIds must have a while loop');
const tokenIdsAbortIdx = generateTokenIdsBody.indexOf('signal?.aborted');
assert.ok(
  tokenIdsAbortIdx > tokenIdsWhileIdx,
  'Abort signal check must be inside the while loop, not before it'
);

// === 10. _resetDecodeRuntimeState called at entry in both surfaces ===

assert.ok(
  generateTokenIdsBody.includes('_resetDecodeRuntimeState'),
  'generateTokenIds must call _resetDecodeRuntimeState at entry'
);
assert.ok(
  generateTokensInternalBody.includes('_resetDecodeRuntimeState'),
  '_generateTokensInternal must call _resetDecodeRuntimeState at entry'
);

// === 11. validateCallTimeOptions at entry ===

assert.ok(
  generateTokenIdsBody.includes('validateCallTimeOptions'),
  'generateTokenIds must validate call-time options'
);
assert.ok(
  generateTokensInternalBody.includes('validateCallTimeOptions'),
  '_generateTokensInternal must validate call-time options'
);

// === 12. generate() and generateTokens() delegate to _generateTokensInternal ===

const generateBody = extractMethodBody(generatorSource, 'async *generate(prompt');
const generateTokensBody = extractMethodBody(generatorSource, 'async *generateTokens(prompt');

assert.ok(
  generateBody.includes('_generateTokensInternal'),
  'generate() must delegate to _generateTokensInternal'
);
assert.ok(
  generateTokensBody.includes('_generateTokensInternal'),
  'generateTokens() must delegate to _generateTokensInternal'
);

console.log('generate-token-ids-contract.test: ok');
