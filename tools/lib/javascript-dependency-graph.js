import fs from 'node:fs/promises';
import path from 'node:path';
import { parseModuleDependencies } from './module-dependencies.js';

// Compatibility view for existing browser/closure consumers of the shared model.
export function parseJavaScriptDependencies(source, filename = 'module.js') {
  const parsed = parseModuleDependencies(source, filename);
  const syntax = parsed.diagnostics.find(item => item.kind === 'syntax');
  if (syntax) throw new Error(`${filename}: ${syntax.expression}`);
  const edges = parsed.edges.filter(edge => !['asset', 'worker'].includes(edge.kind))
    .map(edge => ({ ...edge, kind: edge.kind === 'forward' ? 'static' : edge.kind }));
  const modules = new Set(edges.map(edge => edge.specifier));
  for (const specifier of parsed.assetReferences) {
    if (!modules.has(specifier)) edges.push({ kind: 'resource', specifier,
      line: parsed.edges.find(edge => edge.specifier === specifier)?.line ?? 1 });
  }
  const unresolved = parsed.diagnostics.filter(item => item.kind === 'dynamic').map(item => ({
    line: item.line, expression: item.expression.replace(/^import\(([\s\S]*)\)$/, '$1'),
  }));
  return { edges, unresolved };
}

export function moduleSpecifiers(source, filename) {
  return new Set(parseJavaScriptDependencies(source, filename).edges
    .filter(edge => edge.kind !== 'resource').map(edge => edge.specifier));
}

export function createJavaScriptDependencyGraph() {
  const nodes = new Map();
  return {
    async read(file) {
      const filename = path.resolve(file);
      if (!nodes.has(filename)) nodes.set(filename, fs.readFile(filename, 'utf8').then(source => ({
        path: filename, source, ...(/\.(?:js|cjs|ts)$/.test(filename)
          ? parseJavaScriptDependencies(source, filename) : { edges: [], unresolved: [] }),
      })));
      return nodes.get(filename);
    },
  };
}
