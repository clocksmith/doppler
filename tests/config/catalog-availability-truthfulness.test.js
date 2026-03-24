import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const REPO_ROOT = process.cwd();
const CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');

const catalog = JSON.parse(fs.readFileSync(CATALOG_PATH, 'utf8'));
const models = Array.isArray(catalog.models) ? catalog.models : [];

// =============================================================================
// Invariant: availability.local=true requires a repo-local baseUrl
// =============================================================================

{
  for (const entry of models) {
    const local = entry?.lifecycle?.availability?.local;
    const baseUrl = typeof entry?.baseUrl === 'string' ? entry.baseUrl : null;
    if (local === true) {
      assert.ok(
        baseUrl !== null && baseUrl.startsWith('./local/'),
        `${entry.modelId}: availability.local=true requires a repo-local baseUrl starting with ./local/`
      );
    }
  }
}

// =============================================================================
// Invariant: baseUrl=null implies availability.local must not be true
// =============================================================================

{
  for (const entry of models) {
    const baseUrl = entry?.baseUrl;
    if (baseUrl === null || baseUrl === undefined) {
      assert.notEqual(
        entry?.lifecycle?.availability?.local,
        true,
        `${entry.modelId}: baseUrl is null/missing so availability.local must not be true`
      );
    }
  }
}

// =============================================================================
// TranslateGemma: no local artifact — baseUrl must be null, local must be false
// =============================================================================

{
  const tg = models.find((entry) => entry.modelId === 'translategemma-4b-it-q4k-ehf16-af32');
  assert.ok(tg, 'translategemma-4b-it-q4k-ehf16-af32 must exist in catalog');
  assert.equal(
    tg.baseUrl,
    null,
    'translategemma-4b-it-q4k-ehf16-af32: baseUrl must be null (no local artifact)'
  );
  assert.notEqual(
    tg?.lifecycle?.availability?.local,
    true,
    'translategemma-4b-it-q4k-ehf16-af32: availability.local must not be true when baseUrl is null'
  );
  assert.equal(
    tg?.lifecycle?.availability?.hf,
    true,
    'translategemma-4b-it-q4k-ehf16-af32: availability.hf must be true'
  );
  assert.equal(
    tg?.lifecycle?.status?.tested,
    'verified',
    'translategemma-4b-it-q4k-ehf16-af32: tested status must reflect the verified browser path'
  );
  assert.equal(
    tg?.lifecycle?.status?.demo,
    'compare',
    'translategemma-4b-it-q4k-ehf16-af32: demo status must reflect compare-surface availability'
  );
  assert.equal(
    tg?.lifecycle?.tested?.result,
    'pass',
    'translategemma-4b-it-q4k-ehf16-af32: tested result must be pass after browser verification'
  );
  assert.deepEqual(
    tg?.lifecycle?.tested?.surface,
    ['browser', 'node'],
    'translategemma-4b-it-q4k-ehf16-af32: tested surface must reflect the verified surfaces'
  );
}

// =============================================================================
// Qwen 3.5 0.8B: no local artifact — baseUrl must be null, local must be false
// =============================================================================

{
  const qwen08b = models.find((entry) => entry.modelId === 'qwen-3-5-0-8b-q4k-ehaf16');
  assert.ok(qwen08b, 'qwen-3-5-0-8b-q4k-ehaf16 must exist in catalog');
  assert.equal(
    qwen08b.baseUrl,
    null,
    'qwen-3-5-0-8b-q4k-ehaf16: baseUrl must be null (no local artifact)'
  );
  assert.notEqual(
    qwen08b?.lifecycle?.availability?.local,
    true,
    'qwen-3-5-0-8b-q4k-ehaf16: availability.local must not be true when baseUrl is null'
  );
}

// =============================================================================
// Qwen 3.5 2B: no local artifact — baseUrl must be null, local must be false
// =============================================================================

{
  const qwen2b = models.find((entry) => entry.modelId === 'qwen-3-5-2b-q4k-ehaf16');
  assert.ok(qwen2b, 'qwen-3-5-2b-q4k-ehaf16 must exist in catalog');
  assert.equal(
    qwen2b.baseUrl,
    null,
    'qwen-3-5-2b-q4k-ehaf16: baseUrl must be null (no local artifact)'
  );
  assert.notEqual(
    qwen2b?.lifecycle?.availability?.local,
    true,
    'qwen-3-5-2b-q4k-ehaf16: availability.local must not be true when baseUrl is null'
  );
}

// =============================================================================
// Models with repo-local baseUrls must have availability.local=true
// =============================================================================

{
  for (const entry of models) {
    const baseUrl = typeof entry?.baseUrl === 'string' ? entry.baseUrl : null;
    if (baseUrl !== null && baseUrl.startsWith('./local/')) {
      assert.equal(
        entry?.lifecycle?.availability?.local,
        true,
        `${entry.modelId}: baseUrl starts with ./local/ so availability.local must be true`
      );
    }
  }
}

console.log('catalog-availability-truthfulness.test: ok');
