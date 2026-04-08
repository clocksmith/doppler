import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const conversionRoot = path.join(repoRoot, 'src/config/conversion');

function collectJsonFiles(dir) {
  const entries = readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const entryPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectJsonFiles(entryPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(entryPath);
    }
  }
  return files;
}

for (const filePath of collectJsonFiles(conversionRoot)) {
  const config = JSON.parse(readFileSync(filePath, 'utf8'));
  const activationDtype = config.session?.compute?.defaults?.activationDtype;
  const kvDtype = config.session?.kvcache?.kvDtype;
  const dtypeTransition = config.execution?.policies?.dtypeTransition;
  if (activationDtype !== 'f32' || kvDtype !== 'f16' || dtypeTransition !== 'require_cast_step') {
    continue;
  }

  const attentionKernels = Object.entries(config.execution?.kernels ?? {}).filter(([, decl]) =>
    typeof decl?.kernel === 'string'
    && decl.kernel.startsWith('attention_')
    && decl.kernel.includes('f16kv')
  );
  if (attentionKernels.length === 0) {
    continue;
  }

  for (const [kernelKey, decl] of attentionKernels) {
    assert.equal(
      decl.precision?.kvDtype,
      'f16',
      `${path.relative(repoRoot, filePath)} execution.kernels.${kernelKey} must declare precision.kvDtype="f16"`
    );
  }
}

console.log('attention-f16kv-precision-contract.test: ok');
