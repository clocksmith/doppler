import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function readSource(relativePath) {
  return readFileSync(new URL(`../../${relativePath}`, import.meta.url), 'utf8');
}

function findLine(source, offset) {
  return source.slice(0, offset).split('\n').length;
}

function assertCallBlocksContain(source, marker, needle, label) {
  let offset = 0;
  let found = 0;
  while (true) {
    const start = source.indexOf(marker, offset);
    if (start === -1) break;
    found += 1;

    const window = source.slice(start, start + 600);
    assert.match(
      window,
      new RegExp(needle),
      `${label}: "${marker}" at line ${findLine(source, start)} must include ${needle}`
    );
    offset = start + marker.length;
  }

  assert.ok(found > 0, `${label}: expected to find at least one "${marker}" call`);
}

const probeForwardingFiles = [
  'src/inference/pipelines/text/attention/run.js',
  'src/inference/pipelines/text/layer.js',
  'src/inference/pipelines/text/ffn/standard.js',
  'src/inference/pipelines/text/ffn/sandwich.js',
  'src/inference/pipelines/text/linear-attention.js',
  'src/inference/pipelines/text/embed.js',
  'src/inference/pipelines/text/logits/index.js',
  'src/inference/pipelines/text/logits/utils.js',
  'src/inference/pipelines/text/logits/gpu.js',
];

for (const relativePath of probeForwardingFiles) {
  const source = readSource(relativePath);
  assertCallBlocksContain(source, 'runProbes(', 'operatorDiagnostics', relativePath);
}

{
  const source = readSource('src/inference/pipelines/text/generator.js');
  assertCallBlocksContain(source, 'embed(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator.js');
  assertCallBlocksContain(source, 'recordLogitsGPU(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator.js');
  assertCallBlocksContain(source, 'computeLogits(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator.js');
}

{
  const source = readSource('src/inference/pipelines/text/generator-steps.js');
  assertCallBlocksContain(source, 'embed(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
  assertCallBlocksContain(source, 'recordLogitsGPU(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
  assertCallBlocksContain(source, 'computeLogitsGPU(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
  assertCallBlocksContain(source, 'computeLogits(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
  assertCallBlocksContain(source, 'finalizeLogits(', 'operatorDiagnostics', 'src/inference/pipelines/text/generator-steps.js');
}

console.log('operator-diagnostics-forwarding-contract.test: ok');
