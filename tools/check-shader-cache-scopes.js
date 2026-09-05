import { readFile, readdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

// Inventory direct model shader consumers across all source owners, not just
// gpu/kernels. This is a drift guard; behavioral cache tests prove isolation.
// Device bootstrap probes dynamically import shader-cache outside model runs.
export function inspectShaderCacheOwner(source, path) {
  const ast = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  let consumesShaders = false;
  const guards = new Set();
  for (const node of ast.statements) {
    if (!ts.isImportDeclaration(node) || !ts.isStringLiteral(node.moduleSpecifier)) continue;
    const module = node.moduleSpecifier.text;
    const bindings = node.importClause?.namedBindings;
    if (module.endsWith('/shader-cache.js') && bindings &&
      (ts.isNamespaceImport(bindings) || bindings.elements.some((binding) => (binding.propertyName ?? binding.name).text === 'getShaderModule'))) consumesShaders = true;
    if (!module.endsWith('/shader-source-scope.js')) continue;
    if (!bindings || !ts.isNamedImports(bindings)) continue;
    for (const binding of bindings.elements) {
      if (['getShaderScopeCacheKey', 'getScopedShaderSource'].includes((binding.propertyName ?? binding.name).text)) {
        guards.add(binding.name.text);
      }
    }
  }
  if (!consumesShaders) return null;
  let guarded = false;
  function visit(node) {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && guards.has(node.expression.text)) guarded = true;
    ts.forEachChild(node, visit);
  }
  visit(ast);
  return { path, guarded };
}

export async function checkShaderCacheScopes(root = ROOT) {
  const paths = (await readdir(resolve(root, 'src'), { recursive: true }))
    .filter((path) => path.endsWith('.js')).sort();
  const owners = [];
  for (const path of paths) {
    const owner = inspectShaderCacheOwner(await readFile(resolve(root, 'src', path), 'utf8'), `src/${path}`);
    if (owner) owners.push(owner);
  }
  return { owners, failures: owners.filter((owner) => !owner.guarded) };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3 || process.argv[2] !== '--check') throw new Error('Usage: node tools/check-shader-cache-scopes.js --check');
  const result = await checkShaderCacheScopes();
  console.log(JSON.stringify(result, null, 2));
  if (result.failures.length) process.exitCode = 1;
}
