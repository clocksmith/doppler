import assert from 'node:assert/strict';

import { mergeQuickCatalogEntryLists } from '../../demo/ui/models/quick-catalog.js';

{
  const merged = mergeQuickCatalogEntryLists([
    [
      { modelId: 'translategemma-4b-it-q4k-ehf16-af32', source: 'remote' },
      { modelId: 'gemma-3-270m-it-q4k-ehf16-af32', source: 'remote' },
    ],
    [
      { modelId: 'translategemma-4b-it-q4k-ehf16-af32', source: 'local' },
      { modelId: 'translategemma-4b-1b-enes-q4k-ehf16-af32', source: 'local' },
    ],
  ]);

  assert.deepEqual(merged, [
    { modelId: 'translategemma-4b-it-q4k-ehf16-af32', source: 'remote' },
    { modelId: 'gemma-3-270m-it-q4k-ehf16-af32', source: 'remote' },
    { modelId: 'translategemma-4b-1b-enes-q4k-ehf16-af32', source: 'local' },
  ]);
}

{
  const merged = mergeQuickCatalogEntryLists([
    null,
    undefined,
    [
      { modelId: 'translategemma-4b-1b-enes-q4k-ehf16-af32' },
      { modelId: '  ' },
      {},
    ],
  ]);

  assert.deepEqual(merged, [
    { modelId: 'translategemma-4b-1b-enes-q4k-ehf16-af32' },
  ]);
}

console.log('quick-catalog-merge.test: ok');
