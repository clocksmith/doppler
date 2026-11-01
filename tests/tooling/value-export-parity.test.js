import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createValueExportResolver } from '../../tools/lib/module-dependencies.js';

const root = await mkdtemp(path.join(tmpdir(), 'doppler-value-exports-'));
try {
  const files = {
    'leaf.js': 'export const value = 1; export default function namedDefault() {}',
    'leaf.d.ts': 'export declare const value: number; export default function namedDefault(): void; export interface Shape {}',
    'other.js': 'export const value = 2;',
    'other.d.ts': 'export declare const value: number;',
    'facade.js': "export * from './leaf.js'; export const missing = 1; export * as namespace from './leaf.js';",
    'facade.d.ts': "export * from './leaf.js'; export * as namespace from './leaf.js'; export type * as Types from './leaf.js';",
    'diamond.js': "export * from './leaf.js'; export * from './forward.js';",
    'forward.js': "export * from './leaf.js';",
    'ambiguous.js': "export * from './leaf.js'; export * from './other.js';",
    'override.js': "export * from './ambiguous.js'; export { value } from './leaf.js';",
    'default.js': "export { default } from './leaf.js';",
    'anonymous.js': 'export default class {}',
    'expression.js': 'export default 123;',
    'types.d.ts': `
      interface LocalType {}
      declare const privateBrand: unique symbol;
      export { LocalType };
      export namespace TypeNamespace { interface Shape {} }
      export namespace ValueNamespace { const value: number; }
      export type { value as TypeValue } from './leaf.js';
    `,
    'cycle-a.js': "export * from './cycle-b.js'; export const a = 1;",
    'cycle-b.js': "export * from './cycle-a.js'; export const b = 2;",
    'typed-forward.d.ts': "export type { value } from './leaf.js';",
    'typed-consumer.d.ts': "export { value } from './typed-forward.js';",
    'broken-named.js': "export { absent } from './leaf.js';",
    'broken.js': "export * from './missing.js';",
  };
  await Promise.all(Object.entries(files).map(([file, source]) => writeFile(path.join(root, file), source)));
  const resolve = createValueExportResolver(Object.keys(files).filter(file => file.endsWith('.js')).map(file => path.join(root, file)));
  const declarations = createValueExportResolver(Object.keys(files).filter(file => file.endsWith('.d.ts')).map(file => path.join(root, file)), true);
  const names = (resolver, file) => [...resolver(path.join(root, file))].sort();
  assert.deepEqual(names(resolve, 'leaf.js'), ['default', 'value']);
  assert.deepEqual(names(resolve, 'facade.js'), ['missing', 'namespace', 'value']);
  assert.deepEqual(names(declarations, 'facade.d.ts'), ['namespace', 'value']);
  assert.deepEqual(names(resolve, 'diamond.js'), ['value']);
  assert.deepEqual(names(resolve, 'ambiguous.js'), []);
  assert.deepEqual(names(resolve, 'override.js'), ['value']);
  for (const file of ['default.js', 'anonymous.js', 'expression.js']) assert.deepEqual(names(resolve, file), ['default']);
  assert.deepEqual(names(declarations, 'types.d.ts'), ['ValueNamespace']);
  assert.deepEqual(names(resolve, 'cycle-a.js'), ['a', 'b']);
  assert.deepEqual(names(resolve, 'cycle-b.js'), ['a', 'b']);
  assert.deepEqual(names(declarations, 'typed-consumer.d.ts'), []);
  assert.throws(() => resolve(path.join(root, 'broken-named.js')), /Cannot resolve exported binding/);
  assert.throws(() => resolve(path.join(root, 'broken.js')), /Cannot resolve export/);
} finally {
  await rm(root, { recursive: true, force: true });
}
console.log('value-export-parity.test: wildcard drift, types, namespaces, defaults, conflicts and cycles passed');
