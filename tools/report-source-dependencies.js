#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildDependencyGraph, collectReachable } from './lib/module-dependencies.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'docs/architecture-dependencies.json');
const relative = file => path.relative(root, file).split(path.sep).join('/');
async function filesUnder(directory) {
  const files = [];
  for (const entry of await fs.readdir(path.join(root, directory), { withFileTypes: true })) {
    const name = `${directory}/${entry.name}`;
    if (entry.isDirectory()) files.push(...await filesUnder(name));
    else if (entry.isFile() && entry.name.endsWith('.js')) files.push(name);
  }
  return files;
}
const packageJson = JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf8'));
function entryPoints(value) {
  if (typeof value === 'string') return value.endsWith('.js') ? [value.replace(/^\.\//, '')] : [];
  return value && typeof value === 'object' ? Object.values(value).flatMap(entryPoints) : [];
}
const files = (await Promise.all(['src', 'tools', 'tests', 'demo', 'benchmarks'].map(filesUnder))).flat();
const declarations = JSON.parse(await fs.readFile(path.join(root, 'tools/policies/module-dependencies.json'), 'utf8'));
const { graph, diagnostics } = await buildDependencyGraph(root, files, declarations);
const roots = {
  injectedRuntimeCore: ['src/capsule-runtime.js'],
  hostLoadedExecution: entryPoints(packageJson.exports['./host']),
  compilerAndTooling: [...files.filter(file => file.startsWith('tools/')), ...entryPoints(packageJson.exports['./tooling'])],
  experimental: files.filter(file => file.startsWith('src/experimental/')),
  publicPackage: [...entryPoints(packageJson.exports), ...entryPoints(packageJson.bin)],
  applications: files.filter(file => file.startsWith('demo/') || file.startsWith('benchmarks/')),
  tests: files.filter(file => file.startsWith('tests/')),
};
const views = {};
const product = new Set();
for (const [name, entries] of Object.entries(roots)) {
  const absolute = entries.map(file => path.join(root, file));
  const all = collectReachable(graph, absolute);
  const eager = collectReachable(graph, absolute, edge => edge.kind !== 'dynamic' && edge.kind !== 'worker');
  const sourceFiles = [...all].map(relative).filter(file => file.startsWith('src/')).sort();
  views[name] = { roots: entries.sort(), eager: sourceFiles.filter(file => eager.has(path.join(root, file))),
    lazyOnly: sourceFiles.filter(file => !eager.has(path.join(root, file))) };
  if (name !== 'tests') for (const file of sourceFiles) product.add(file);
}
const testFiles = [...views.tests.eager, ...views.tests.lazyOnly];
const report = {
  schemaVersion: 1,
  description: 'Parsed potential dependencies, not observed execution. Injected core excludes executors supplied by the host. Computed dependencies without declarations remain diagnostics.',
  views,
  testOnly: testFiles.filter(file => !product.has(file)).sort(),
  diagnostics: diagnostics.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line),
};
const changed = process.argv.filter(arg => arg.startsWith('--changed=')).map(arg => path.resolve(root, arg.slice(10)));
if (changed.length) {
  const reverse = new Map();
  for (const [file, edges] of graph) for (const edge of edges) {
    if (!edge.target) continue;
    if (!reverse.has(edge.target)) reverse.set(edge.target, []);
    reverse.get(edge.target).push(file);
  }
  console.log(JSON.stringify({ changed: changed.map(relative), affected: [...collectReachable(reverse, changed)].map(relative).sort() }, null, 2));
} else {
  const text = `${JSON.stringify(report, null, 2)}\n`;
  if (process.argv.includes('--write')) await fs.writeFile(output, text);
  else if (await fs.readFile(output, 'utf8').catch(() => '') !== text) {
    throw new Error('Dependency views are stale. Run node tools/report-source-dependencies.js --write.');
  }
  console.log(`[source-dependencies] ${graph.size} files; ${report.testOnly.length} test-only; ${diagnostics.length} computed/unresolved diagnostics`);
}
