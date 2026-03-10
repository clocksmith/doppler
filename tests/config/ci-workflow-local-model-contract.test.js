import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const workflowDir = path.join(process.cwd(), '.github/workflows');
const workflowFiles = fs.readdirSync(workflowDir)
  .filter((entry) => entry.endsWith('.yml'))
  .sort();

for (const fileName of workflowFiles) {
  const workflowPath = path.join(workflowDir, fileName);
  const source = fs.readFileSync(workflowPath, 'utf8');
  assert.equal(
    source.includes('models/local/'),
    false,
    `${fileName}: CI workflows must not depend on gitignored models/local artifacts`
  );
}

console.log('ci-workflow-local-model-contract.test: ok');
