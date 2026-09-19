import ts from 'typescript';
import fs from 'node:fs/promises';
import path from 'node:path';

// One syntax model for boundary checks, inventories, and impact analysis. The
// parser never executes source and cannot mistake strings/comments for imports.
export function parseModuleDependencies(source, fileName = 'module.js') {
  const ast = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true,
    fileName.endsWith('.ts') ? ts.ScriptKind.TS : ts.ScriptKind.JS);
  const edges = [];
  const diagnostics = [];
  const assetReferences = new Set();
  const valueExports = new Set();
  let hasWildcardExport = false;
  function bindingNames(name) {
    if (ts.isIdentifier(name)) valueExports.add(name.text);
    else for (const element of name.elements) if (ts.isBindingElement(element)) bindingNames(element.name);
  }
  for (const statement of ast.statements) {
    if (ts.isExportDeclaration(statement) && !statement.isTypeOnly) {
      if (!statement.exportClause) hasWildcardExport = true;
      else if (ts.isNamedExports(statement.exportClause)) {
        for (const item of statement.exportClause.elements) if (!item.isTypeOnly) valueExports.add(item.name.text);
      } else valueExports.add(statement.exportClause.name.text);
    }
    if (!statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword)) continue;
    if (ts.isVariableStatement(statement)) for (const item of statement.declarationList.declarations) bindingNames(item.name);
    else if ((ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement) || ts.isEnumDeclaration(statement)) && statement.name) {
      valueExports.add(statement.name.text);
    }
  }
  const literal = node => node && (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node))
    ? node.text : null;
  function add(kind, node, value) {
    const position = ast.getLineAndCharacterOfPosition(node.getStart(ast));
    if (value === null) {
      diagnostics.push({ kind, line: position.line + 1, expression: node.getText(ast) });
    } else {
      edges.push({ kind, specifier: value, line: position.line + 1 });
    }
  }
  function visit(node) {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      if (/^(?:\.\.\/|\.\/|src\/).*\.(?:d\.ts|js|json|html|wgsl)(?:[?#].*)?$/.test(node.text)) assetReferences.add(node.text);
    }
    if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) {
      add('type', node, literal(node.argument.literal));
    }
    if (ts.isImportDeclaration(node)) add(node.importClause?.isTypeOnly ? 'type' : 'static', node, literal(node.moduleSpecifier));
    if (ts.isExportDeclaration(node) && node.moduleSpecifier) add(node.isTypeOnly ? 'type' : 'forward', node, literal(node.moduleSpecifier));
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      add('dynamic', node, literal(node.arguments[0]));
    }
    if (ts.isNewExpression(node) && node.expression.getText(ast) === 'URL'
      && node.arguments?.[1]?.getText(ast) === 'import.meta.url') {
      const parent = node.parent;
      const worker = ts.isNewExpression(parent) && ['Worker', 'SharedWorker'].includes(parent.expression.getText(ast));
      add(worker ? 'worker' : 'asset', node, literal(node.arguments[0]));
    }
    ts.forEachChild(node, visit);
  }
  visit(ast);
  for (const diagnostic of ast.parseDiagnostics) {
    diagnostics.push({ kind: 'syntax', line: ast.getLineAndCharacterOfPosition(diagnostic.start ?? 0).line + 1,
      expression: ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n') });
  }
  return { edges, diagnostics, assetReferences: [...assetReferences], valueExports: [...valueExports], hasWildcardExport };
}

export function collectRelativeAssetSpecifiers(source, fileName) {
  return parseModuleDependencies(source, fileName).assetReferences;
}

// The graph keeps forwarding and lazy edges. Policies may select views, but
// cannot erase an intermediate module when checking cycles or ownership.
export async function buildDependencyGraph(root, files, declarations = {}) {
  const graph = new Map();
  const diagnostics = [];
  const pending = [...files].map(file => path.resolve(root, file));
  async function resolveTarget(importer, specifier, kind) {
    if (!specifier.startsWith('.')) return null;
    const target = path.resolve(path.dirname(importer), specifier.split(/[?#]/, 1)[0]);
    const candidates = kind === 'asset' || path.extname(target)
      ? [target] : [target, `${target}.js`, path.join(target, 'index.js')];
    for (const candidate of candidates) {
      const stat = await fs.stat(candidate).catch(() => null);
      if (stat?.isFile()) return candidate;
    }
    return null;
  }
  while (pending.length) {
    const file = pending.pop();
    if (graph.has(file)) continue;
    const relative = path.relative(root, file).split(path.sep).join('/');
    const parsed = file.endsWith('.js') || file.endsWith('.ts')
      ? parseModuleDependencies(await fs.readFile(file, 'utf8'), file) : { edges: [], diagnostics: [] };
    for (const diagnostic of parsed.diagnostics) {
      const declared = declarations[relative]?.[diagnostic.expression];
      if (declared) {
        for (const specifier of declared) parsed.edges.push({ kind: diagnostic.kind, specifier, line: diagnostic.line, declared: true });
      } else diagnostics.push({ file: relative, ...diagnostic });
    }
    const edges = [];
    for (const edge of parsed.edges) {
      const target = await resolveTarget(file, edge.specifier, edge.kind);
      edges.push({ ...edge, target });
      if (target) pending.push(target);
      else if (edge.specifier.startsWith('.') && edge.kind !== 'asset') {
        diagnostics.push({ file: relative, kind: 'unresolved', line: edge.line, expression: edge.specifier });
      }
    }
    graph.set(file, edges);
  }
  return { graph, diagnostics };
}

export function collectModuleSpecifiers(source, fileName) {
  return [...new Set(parseModuleDependencies(source, fileName).edges
    .filter(edge => edge.kind !== 'asset' || edge.specifier.endsWith('.js'))
    .map(edge => edge.specifier))];
}

export function collectReachable(graph, roots, includeEdge = () => true) {
  const reached = new Set();
  const pending = [...roots];
  while (pending.length) {
    const file = pending.pop();
    if (reached.has(file)) continue;
    reached.add(file);
    for (const edge of graph.get(file) ?? []) {
      const target = typeof edge === 'string' ? edge : edge.target;
      if (target && includeEdge(edge)) pending.push(target);
    }
  }
  return reached;
}

export function collectStronglyConnectedComponents(graph) {
  let nextIndex = 0;
  const indices = new Map();
  const lowLinks = new Map();
  const stack = [];
  const onStack = new Set();
  const components = [];
  function visit(node) {
    indices.set(node, nextIndex);
    lowLinks.set(node, nextIndex++);
    stack.push(node);
    onStack.add(node);
    for (const dependency of graph.get(node) ?? []) {
      if (!graph.has(dependency)) continue;
      if (!indices.has(dependency)) {
        visit(dependency);
        lowLinks.set(node, Math.min(lowLinks.get(node), lowLinks.get(dependency)));
      } else if (onStack.has(dependency)) {
        lowLinks.set(node, Math.min(lowLinks.get(node), indices.get(dependency)));
      }
    }
    if (lowLinks.get(node) !== indices.get(node)) return;
    const component = [];
    for (;;) {
      const current = stack.pop();
      onStack.delete(current);
      component.push(current);
      if (current === node) break;
    }
    if (component.length > 1 || (graph.get(node) ?? []).includes(node)) components.push(component);
  }
  for (const node of graph.keys()) if (!indices.has(node)) visit(node);
  return components;
}
