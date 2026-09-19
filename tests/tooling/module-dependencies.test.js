import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { parseModuleDependencies, buildDependencyGraph, collectStronglyConnectedComponents, collectReachable } from '../../tools/lib/module-dependencies.js';

const parsed = parseModuleDependencies(`
// import './comment.js';
const text = "import './string.js'";
import { value } from './static.js';
export * from './forward.js';
const lazy = () => import('./hidden.js');
const template = () => import(\`./template.js\`);
new Worker(new URL('./worker.js', import.meta.url));
const asset = new URL('./shader.wgsl', import.meta.url);
const computed = name => import(name);
`);
assert.deepEqual(parsed.edges.map(({ kind, specifier }) => [kind, specifier]), [
  ['static', './static.js'], ['forward', './forward.js'], ['dynamic', './hidden.js'],
  ['dynamic', './template.js'], ['worker', './worker.js'], ['asset', './shader.wgsl'],
]);
assert.equal(parsed.diagnostics.length, 1);
assert.equal(parsed.diagnostics[0].expression, 'import(name)');
assert.deepEqual(parseModuleDependencies('export const { one, two: renamed, nested: { three } } = source;').valueExports.sort(), ['one', 'renamed', 'three']);
assert.deepEqual(parseModuleDependencies('export interface NotValue {}; export type AlsoNot = string; export const actual: number;', 'exports.d.ts').valueExports, ['actual']);
const graph = new Map([['A', ['forward']], ['forward', ['B']], ['B', ['A']], ['test', ['onlyTest']], ['onlyTest', []]]);
assert.deepEqual(collectStronglyConnectedComponents(graph).map(group => group.sort()), [['A', 'B', 'forward']]);
assert.deepEqual([...collectReachable(graph, ['A'])].sort(), ['A', 'B', 'forward']);
assert.deepEqual([...collectReachable(graph, ['test'])].sort(), ['onlyTest', 'test']);

const root = await mkdtemp(path.join(tmpdir(), 'doppler-dependency-fixture-'));
try {
  const files = {
    'a.js': "export * from './forward.js'; export const lazy = () => import('./hidden.js'); export const computed = name => import(name);",
    'forward.js': "export * from './b.js';",
    'b.js': "import './a.js';",
    'hidden.js': "new Worker(new URL('./worker.js', import.meta.url)); new URL('./kernel.wgsl', import.meta.url);",
    'worker.js': '', 'kernel.wgsl': '', 'declared.js': '',
  };
  await Promise.all(Object.entries(files).map(([name, contents]) => writeFile(path.join(root, name), contents)));
  const { graph: actual, diagnostics } = await buildDependencyGraph(root, ['a.js']);
  assert.equal(diagnostics[0].expression, 'import(name)');
  assert(collectReachable(actual, [path.join(root, 'a.js')]).has(path.join(root, 'worker.js')));
  const eager = collectReachable(actual, [path.join(root, 'a.js')], edge => edge.kind !== 'dynamic');
  assert(!eager.has(path.join(root, 'hidden.js')));
  const cycleGraph = new Map([...actual].map(([file, edges]) => [file, edges.map(edge => edge.target).filter(Boolean)]));
  assert.deepEqual(collectStronglyConnectedComponents(cycleGraph)[0].map(file => path.basename(file)).sort(), ['a.js', 'b.js', 'forward.js']);
  const declared = await buildDependencyGraph(root, ['a.js'], { 'a.js': { 'import(name)': ['./declared.js'] } });
  assert.equal(declared.diagnostics.length, 0);
  assert(declared.graph.has(path.join(root, 'declared.js')));
  const types = parseModuleDependencies("export type Result = import('./a.js').Result;", 'boundary.d.ts');
  assert.equal(types.edges[0].kind, 'type');
  assert(parseModuleDependencies('import { broken').diagnostics.some(item => item.kind === 'syntax'));
} finally {
  await rm(root, { recursive: true, force: true });
}
