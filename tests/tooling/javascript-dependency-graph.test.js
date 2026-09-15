import assert from 'node:assert/strict';
import { parseJavaScriptDependencies, moduleSpecifiers } from '../../tools/lib/javascript-dependency-graph.js';

const source = `
// import './comment.js';
/* export * from './comment-too.js'; */
const prose = "import './string.js'";
import './side-effect.js';
import { value } from './static.js';
export * from './export.js';
await import(/* split */ './dynamic.js', { with: { type: 'json' } });
await import(\`./template.js\`);
await import(provider.module);
const shader = new URL('./kernel.wgsl?version=1', import.meta.url);
const config = './config.json';
const template = \`literal import('./not-code.js')\`;
const evaluated = \`value: \${await import('./evaluated.js')}\`;
`;
assert.deepEqual([...moduleSpecifiers(source)], [
  './side-effect.js', './static.js', './export.js', './dynamic.js', './template.js', './evaluated.js',
]);
const node = parseJavaScriptDependencies(source);
assert.deepEqual(node.edges.filter(edge => edge.kind === 'resource').map(edge => edge.specifier),
  ['./kernel.wgsl?version=1', './config.json']);
assert.deepEqual(node.unresolved.map(edge => edge.expression), ['provider.module']);
assert.deepEqual([...moduleSpecifiers("export type T = import('./types.js').T; import type { U } from './u.js';", 'types.d.ts')],
  ['./types.js', './u.js']);
assert.throws(() => parseJavaScriptDependencies('import {', 'broken.js'), /broken.js/);
console.log('javascript-dependency-graph.test: comments, strings, dynamic imports, types and resources passed');
