import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { DOPPLER_VERSION, DOPPLER_PROVIDER_VERSION } from '../../src/version.js';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(TEST_DIR, '..', '..');
const PACKAGE_JSON_PATH = path.join(REPO_ROOT, 'package.json');

const packageJson = JSON.parse(await fs.readFile(PACKAGE_JSON_PATH, 'utf8'));

assert.equal(DOPPLER_VERSION, packageJson.version, 'DOPPLER_VERSION must match package.json version');
assert.equal(DOPPLER_PROVIDER_VERSION, packageJson.version, 'DOPPLER_PROVIDER_VERSION must match package.json version');

console.log('public-version-exports.test: ok');
