import { createDocumentSearchRenderer } from '../../examples/electron-document-search/renderer.js';
import { runElectronCapsuleContract } from '../helpers/electron-capsule-contract.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

await runElectronCapsuleContract({
  fixture: await createSignedCapsuleFixture({ operation: 'rerank' }),
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
  createRenderer: createDocumentSearchRenderer,
});
console.log('electron-capsule-runtime.test: ok (signed fixture; synthetic execution)');
