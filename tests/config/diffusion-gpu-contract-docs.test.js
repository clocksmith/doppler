import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

function readLocal(pathValue) {
  return readFileSync(resolve(process.cwd(), pathValue), 'utf8');
}

{
  const schemaDts = readLocal('src/config/schema/diffusion.schema.d.ts');
  assert.match(schemaDts, /export type DiffusionBackendPipeline = 'gpu';/);
}

{
  const schemaJs = readLocal('src/config/schema/diffusion.schema.js');
  assert.match(schemaJs, /pipeline:\s*'gpu'/);
}

{
  const commandGuide = readLocal('docs/style/command-interface-design-guide.md');
  assert.match(commandGuide, /workloadType="diffusion"/);
  assert.match(commandGuide, /suite="diffusion"/);
}

{
  const harnessGuide = readLocal('docs/style/harness-style-guide.md');
  assert.match(harnessGuide, /workloadType="diffusion"/);
  assert.match(harnessGuide, /suite="diffusion"/);
}

{
  const staleReferencePaths = [
    'docs/style/command-interface-design-guide.md',
    'docs/style/harness-style-guide.md',
    'src/config/schema/diffusion.schema.d.ts',
    'src/inference/pipelines/diffusion/init.js',
    'src/inference/pipelines/diffusion/pipeline.js',
    'tests/integration/diffusion-runtime-contract.test.js',
  ];
  for (const pathValue of staleReferencePaths) {
    const content = readLocal(pathValue);
    assert.equal(
      content.includes('gpu_scaffold'),
      false,
      `${pathValue} should not include stale "gpu_scaffold" references`
    );
  }
}

console.log('diffusion-gpu-contract-docs.test: ok');
