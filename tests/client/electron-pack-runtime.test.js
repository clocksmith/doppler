import { createDocumentSearchRenderer } from '../../examples/electron-document-search/renderer.js';
import { runElectronPackContract } from '../helpers/electron-pack-contract.js';
import { createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';

await runElectronPackContract({
  fixture: await createSignedPackFixture(),
  trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  createRenderer: createDocumentSearchRenderer,
});
console.log('electron-pack-runtime.test: ok (signed fixture; synthetic execution)');
