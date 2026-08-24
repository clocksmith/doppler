import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const action = await fs.readFile('.github/actions/doppler-release/action.yml', 'utf8');
const workflow = await fs.readFile('.github/workflows/doppler-release.yml', 'utf8');

assert.match(action, /doppler-gpu@\$DOPPLER_VERSION/u);
assert.match(action, /activationPerformed !== false/u);
assert.match(action, /--action "\$RELEASE_ACTION"/u);
assert.match(workflow, /contents: read/u);
assert.match(workflow, /pull-requests: read/u);
assert.match(workflow, /checks: write/u);
assert.match(workflow, /actions\/download-artifact/u);
assert.match(workflow, /if: always\(\)/u);
assert.doesNotMatch(workflow, /contents: write/u);
assert.doesNotMatch(workflow, /git push|deploy|self-promote/iu);

console.log('github-release-action.test: ok');
