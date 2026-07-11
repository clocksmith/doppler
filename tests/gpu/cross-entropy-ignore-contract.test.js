import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const [forwardSource, backwardSource] = await Promise.all([
  readFile(new URL('../../src/gpu/kernels/cross_entropy_loss.wgsl', import.meta.url), 'utf8'),
  readFile(
    new URL('../../src/gpu/kernels/backward/cross_entropy_backward.wgsl', import.meta.url),
    'utf8'
  ),
]);

for (const [label, source] of [
  ['forward', forwardSource],
  ['backward', backwardSource],
]) {
  assert.match(
    source,
    /if\s*\(target_idx\s*>=\s*u\.vocab_size\)\s*\{\s*output\[[^\]]+\]\s*=\s*0\.0;\s*return;/s,
    `cross-entropy ${label} must emit zero for ignored targets`
  );
}

console.log('cross-entropy-ignore-contract.test: ok');
