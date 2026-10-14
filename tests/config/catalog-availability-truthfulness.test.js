import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const REPO_ROOT = process.cwd();
const CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');
const LOCAL_MODELS_DIR = path.join(REPO_ROOT, 'models', 'local');

const catalog = JSON.parse(fs.readFileSync(CATALOG_PATH, 'utf8'));
const models = Array.isArray(catalog.models) ? catalog.models : [];

// =============================================================================
// Invariant: availability.local=true iff models/local/<modelId>/manifest.json exists
// =============================================================================

{
  for (const entry of models) {
    const local = entry?.lifecycle?.availability?.local;
    const manifestPath = path.join(LOCAL_MODELS_DIR, entry.modelId, 'manifest.json');
    const manifestExists = fs.existsSync(manifestPath);

    if (local === true) {
      assert.ok(
        manifestExists,
        `${entry.modelId}: availability.local=true but models/local/${entry.modelId}/manifest.json does not exist`
      );
    }
    if (manifestExists) {
      assert.equal(
        local,
        true,
        `${entry.modelId}: models/local/${entry.modelId}/manifest.json exists but availability.local is not true`
      );
    }
  }
}

// =============================================================================
// Invariant: availability.hf=true requires hf coordinates
// =============================================================================

{
  for (const entry of models) {
    const hfAvail = entry?.lifecycle?.availability?.hf;
    const hfSpec = entry?.hf;
    if (hfAvail === true) {
      assert.ok(
        hfSpec?.repoId && hfSpec?.path,
        `${entry.modelId}: availability.hf=true requires hf.repoId and hf.path`
      );
    }
  }
}

// =============================================================================
// Invariant: baseUrl is either null or a non-empty string
// =============================================================================

{
  for (const entry of models) {
    const baseUrl = entry?.baseUrl;
    if (baseUrl !== null && baseUrl !== undefined) {
      assert.equal(
        typeof baseUrl,
        'string',
        `${entry.modelId}: baseUrl must be null or a string`
      );
      assert.ok(
        baseUrl.trim().length > 0,
        `${entry.modelId}: baseUrl must not be an empty string`
      );
    }
  }
}

// =============================================================================
// Invariant: quickstart entries carry pinned hosted coordinates
// =============================================================================

{
  for (const entry of models) {
    if (entry.quickstart !== true) {
      continue;
    }
    assert.ok(
      entry?.hf?.repoId && entry?.hf?.revision && entry?.hf?.path,
      `Quickstart entry ${entry.modelId} requires hf.repoId, hf.revision, and hf.path`
    );
    assert.notEqual(
      entry.hf.revision,
      'unknown-local-snapshot',
      `Quickstart entry ${entry.modelId} must not use placeholder hf.revision`
    );
  }
}

console.log('catalog-availability-truthfulness.test: ok');
