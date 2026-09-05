import assert from 'node:assert/strict';
import { checkShaderCacheScopes, inspectShaderCacheOwner } from '../../tools/check-shader-cache-scopes.js';

const consumer = "import { getShaderModule as compile } from './shader-cache.js';";
const guard = "import { getShaderScopeCacheKey as scopeKey } from './shader-source-scope.js';";
assert.equal(inspectShaderCacheOwner(consumer, 'fixture.js').guarded, false);
assert.equal(inspectShaderCacheOwner(consumer + guard, 'fixture.js').guarded, false);
assert.equal(inspectShaderCacheOwner(consumer + guard + 'scopeKey();', 'fixture.js').guarded, true);
assert.equal(inspectShaderCacheOwner('// getShaderModule()', 'fixture.js'), null);
const result = await checkShaderCacheScopes();
assert.ok(result.owners.some((owner) => owner.path === 'src/inference/moe-router.js'));
assert.deepEqual(result.failures, []);
console.log(`shader-cache-scopes.test: ok (${result.owners.length} source owners)`);
