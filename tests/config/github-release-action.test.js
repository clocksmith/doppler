import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const action = await fs.readFile('.github/actions/doppler-release/action.yml', 'utf8');
const workflow = await fs.readFile('.github/workflows/doppler-release.yml', 'utf8');

assert.match(action, /activationPerformed !== false/u);
assert.match(action, /set \+e/u);
assert.match(action, /"\$RELEASE_ACTION" == "qualify" && "\$STATUS" != "passed"/u);
assert.match(action, /"\$RELEASE_ACTION" == "decide" && "\$ELIGIBILITY" != "eligible"/u);
assert.match(workflow, /contents: read/u);
assert.match(workflow, /pull-requests: read/u);
assert.match(workflow, /checks: write/u);
assert.match(workflow, /if: always\(\)/u);
assert.doesNotMatch(workflow, /contents: write/u);
assert.doesNotMatch(workflow, /git push|deploy|self-promote/iu);

console.log('github-release-action.test: ok');
