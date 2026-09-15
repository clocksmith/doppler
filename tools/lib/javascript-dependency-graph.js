import fs from 'node:fs/promises';
import path from 'node:path';
import ts from 'typescript';

// Shared syntax inventory for the different graph views (runtime, package,
// browser and ownership). Keep node metadata; do not retain compiler ASTs.
export function parseJavaScriptDependencies(source, filename = 'module.js') {
  const ast = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  if (ast.parseDiagnostics.length) {
    const diagnostic = ast.parseDiagnostics[0];
    throw new Error(`${filename}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`);
  }
  const edges = [], unresolved = [];
  const moduleLiterals = new Set();
  function add(kind, node) {
    moduleLiterals.add(node);
    edges.push({ kind, specifier: node.text, line: ast.getLineAndCharacterOfPosition(node.getStart(ast)).line + 1 });
  }
  function visit(node) {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier) {
      add(node.isTypeOnly || node.importClause?.isTypeOnly ? 'type' : 'static', node.moduleSpecifier);
    } else if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument) && ts.isStringLiteralLike(node.argument.literal)) {
      add('type', node.argument.literal);
    } else if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      const argument = node.arguments[0];
      if (argument && ts.isStringLiteralLike(argument)) add('dynamic', argument);
      else unresolved.push({ expression: argument?.getText(ast) ?? '',
        line: ast.getLineAndCharacterOfPosition(node.getStart(ast)).line + 1 });
    }
    if (ts.isStringLiteralLike(node) && !moduleLiterals.has(node)
      && /^(?:\.\.\/|\.\/).+\.(?:d\.ts|js|json|html|wgsl)(?:[?#].*)?$/.test(node.text)) {
      add('resource', node);
    }
    ts.forEachChild(node, visit);
  }
  visit(ast);
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
