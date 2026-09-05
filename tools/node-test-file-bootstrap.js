import './node-test-runtime-setup.js';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

if (process.env.NODE_TEST_CONTEXT === 'child-v8') {
  const { after, test } = await import('node:test');
  // Finish plain scripts' top-level assertions before enabling native test cleanup.
  // The worker's subsequent entrypoint import reuses this module evaluation.
  const initialization = import(pathToFileURL(resolve(process.argv[1])).href);
  // Registered tests may finish while their module still awaits top-level work.
  after(() => initialization);
  await initialization;
  test('test module initialization completed', () => {});
}
