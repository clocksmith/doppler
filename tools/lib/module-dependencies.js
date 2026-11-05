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
    if (ts.isExportAssignment(statement) || statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.DefaultKeyword)) {
      valueExports.add('default');
      continue;
    }
    if (ts.isExportDeclaration(statement) && !statement.isTypeOnly) {
      if (!statement.exportClause) hasWildcardExport = true;
      else if (ts.isNamedExports(statement.exportClause)) {
        for (const item of statement.exportClause.elements) if (!item.isTypeOnly) valueExports.add(item.name.text);
      } else valueExports.add(statement.exportClause.name.text);
    }
    if (!statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword)) continue;
    if (ts.isVariableStatement(statement)) for (const item of statement.declarationList.declarations) bindingNames(item.name);
    else if ((ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement) || ts.isEnumDeclaration(statement) || ts.isModuleDeclaration(statement)) && statement.name) {
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
      if (!['dynamic', 'worker', 'asset'].includes(diagnostic.kind)) {
        diagnostics.push({ file: relative, ...diagnostic });
        continue;
      }
      const declared = declarations[relative]?.[diagnostic.expression];
      if (Array.isArray(declared) && declared.length && declared.every(value => typeof value === 'string' && value.trim())) {
        for (const specifier of declared) parsed.edges.push({ kind: diagnostic.kind, specifier, line: diagnostic.line, declared: true });
      } else if (declared && typeof declared.externalContract === 'string' && declared.externalContract.trim()
        && typeof declared.validatedBy === 'string' && !path.isAbsolute(declared.validatedBy)
        && !declared.validatedBy.split('/').includes('..')
        && (await fs.stat(path.join(root, declared.validatedBy)).catch(() => null))?.isFile()) {
        // Runtime extension modules cannot be represented as invented local graph
        // edges. Keep their interface and validation owner visible in the report.
        diagnostics.push({ file: relative, ...diagnostic, kind: 'external-contract',
          dependencyKind: diagnostic.kind, contract: declared.externalContract, validatedBy: declared.validatedBy });
      } else diagnostics.push({ file: relative, ...diagnostic,
        ...(declared === undefined ? {} : { kind: 'invalid-declaration' }) });
    }
    const edges = [];
    for (const edge of parsed.edges) {
      // A sibling checkout may or may not exist (local workspace vs CI), so an
      // import that leaves the repository is never followed; the views stay
      // identical in both environments.
      if (edge.specifier.startsWith('.')
        && path.relative(root, path.resolve(path.dirname(file), edge.specifier)).startsWith('..')) {
        edges.push({ ...edge, target: null });
        diagnostics.push({ file: relative, kind: 'outside-repository', line: edge.line, expression: edge.specifier });
        continue;
      }
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

// Resolve value-export names, not general type correctness. Keep runtime and
// declaration programs separate: TypeScript normally substitutes .d.ts for .js,
// which would hide precisely the implementation/declaration drift being checked.
export function createValueExportResolver(files, declarations = false) {
  const options = { allowJs: true, noLib: true, skipLibCheck: true,
    module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler };
  const host = ts.createCompilerHost(options);
  host.resolveModuleNames = (names, importer) => names.map(name => {
    if (name.startsWith('.') && name.endsWith('.js')) {
      const target = path.resolve(path.dirname(importer), name);
      const declarationTarget = declarations || importer.endsWith('.d.ts');
      const file = declarationTarget ? target.replace(/\.js$/, '.d.ts') : target;
      if (host.fileExists(file)) return { resolvedFileName: file,
        extension: declarationTarget ? ts.Extension.Dts : ts.Extension.Js };
      return undefined;
    }
    return ts.resolveModuleName(name, importer, options, host).resolvedModule;
  });
  const program = ts.createProgram(files, options, host);
  const checker = program.getTypeChecker();
  function resolve(module, visiting = new Set()) {
    if (!module || visiting.has(module)) return new Map();
    const next = new Set(visiting).add(module);
    const values = new Map();
    // Only explicitly advertised declarations are API: private ambient helper
    // classes and unique-symbol brands are not runtime exports.
    const explicit = new Set((module.declarations ?? []).filter(ts.isSourceFile)
      .flatMap(source => parseModuleDependencies(source.text, source.fileName).valueExports));
    for (const symbol of module.exports?.values() ?? []) {
      const name = symbol.name;
      if (!explicit.has(name)) continue;
      if (symbol.flags & ts.SymbolFlags.Alias && checker.getTypeOnlyAliasDeclaration(symbol)) continue;
      const target = symbol.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(symbol) : symbol;
      if (symbol.flags & ts.SymbolFlags.Alias && !target.declarations?.length) {
        throw new Error(`Cannot resolve exported binding: ${name}`);
      }
      if (target.flags & ts.SymbolFlags.Value) values.set(name, target);
    }
    for (const source of module.declarations ?? []) {
      if (!ts.isSourceFile(source)) continue;
      for (const statement of source.statements) {
        if (!ts.isExportDeclaration(statement) || statement.isTypeOnly
          || statement.exportClause || !statement.moduleSpecifier) continue;
        const target = checker.getSymbolAtLocation(statement.moduleSpecifier);
        if (!target) throw new Error(`Cannot resolve export in ${source.fileName}: ${statement.getText(source)}`);
        for (const [name, symbol] of resolve(target, next)) {
          if (name === 'default' || explicit.has(name)) continue;
          // Conflicting star bindings are not exported. A diamond forwarding the
          // same original binding is not a conflict. Explicit exports win.
          if (values.has(name) && values.get(name) !== symbol) values.set(name, null);
          else values.set(name, symbol);
        }
      }
    }
    return values;
  }
  return file => {
    const source = program.getSourceFile(file);
    if (!source) throw new Error(`Missing export source: ${file}`);
    if (source.parseDiagnostics.length) throw new Error(`Invalid export source: ${file}`);
    return new Set([...resolve(checker.getSymbolAtLocation(source))]
      .filter(([, symbol]) => symbol !== null).map(([name]) => name));
  };
}

export function dependencyPolicyErrors(diagnostics) {
  return diagnostics.filter(item => item.kind !== 'external-contract');
}
