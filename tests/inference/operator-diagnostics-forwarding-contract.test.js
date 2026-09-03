import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function readSource(relativePath) {
  return readFileSync(new URL(`../../${relativePath}`, import.meta.url), 'utf8');
}

function findLine(source, offset) {
  return source.slice(0, offset).split('\n').length;
}

function findCallEnd(source, start, marker) {
  let depth = 1;
  let quote = null;
  let escaped = false;
  for (let index = start + marker.length; index < source.length; index += 1) {
    const char = source[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === quote) quote = null;
      continue;
    }
    if (char === "'" || char === '"' || char === '`') {
      quote = char;
      continue;
    }
    if (char === '(') depth += 1;
    else if (char === ')') {
      depth -= 1;
      if (depth === 0) return index + 1;
    }
  }
  throw new Error(`Unterminated call at line ${findLine(source, start)}`);
}

function assertCallBlocksContain(source, marker, needle, label) {
  let offset = 0;
  let found = 0;
  while (true) {
    const start = source.indexOf(marker, offset);
    if (start === -1) break;
    found += 1;

    const call = source.slice(start, findCallEnd(source, start, marker));
    assert.match(
      call,
      new RegExp(needle),
      `${label}: "${marker}" at line ${findLine(source, start)} must include ${needle}`
    );
    offset = start + marker.length;
  }

  assert.ok(found > 0, `${label}: expected to find at least one "${marker}" call`);
}

const probeForwardingFiles = [
  'src/inference/pipelines/text/attention/interpreter.js',
  'src/inference/pipelines/text/layer.js',
  'src/inference/pipelines/text/ffn/standard.js',
  'src/inference/pipelines/text/ffn/sandwich.js',
  'src/inference/pipelines/text/linear-attention.js',
  'src/inference/pipelines/text/attention/rope-observation.js',
  'src/inference/pipelines/text/embedding-normalization.js',
  'src/inference/pipelines/text/logits/index.js',
  'src/experimental/logits/cpu-output.js',
  'src/inference/pipelines/text/logits/gpu.js',
];

const ropeObservation = readSource(
  'src/inference/pipelines/text/attention/rope-observation.js'
);
assert.match(ropeObservation, /runProbes\('q_rope'/u);
assert.match(ropeObservation, /runProbes\('k_rope'/u);

for (const relativePath of probeForwardingFiles) {
  const source = readSource(relativePath);
  assertCallBlocksContain(source, 'runProbes(', 'operatorDiagnostics', relativePath);
}

{
  const source = [
    'src/inference/pipelines/text/generator.js',
    'src/inference/pipelines/text/generator/decode.js',
    'src/inference/pipelines/text/generator/prefill-runtime.js',
  ].map(readSource).join('\n');
  assertCallBlocksContain(source, 'await embed(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator.js');
  assertCallBlocksContain(source, 'recordLogitsGPU(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator.js');
  assertCallBlocksContain(source, 'computeLogits(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator.js');
}

{
  const source = [
    'src/inference/pipelines/text/generator-steps.js',
    'src/inference/pipelines/text/generator/decode.js',
    'src/inference/pipelines/text/generator/diffusion.js',
  ].map(readSource).join('\n');
  assertCallBlocksContain(source, 'await embed(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
  assertCallBlocksContain(source, 'recordLogitsGPU(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
  assertCallBlocksContain(source, 'computeLogitsGPU(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
  assertCallBlocksContain(source, 'computeLogits(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
}

console.log('operator-diagnostics-forwarding-contract.test: ok');
