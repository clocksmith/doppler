#!/usr/bin/env node
/* eslint-disable no-console */
const fs = require("fs");
const path = require("path");
const ts = require("typescript");

const projectRoot = path.resolve(__dirname, "..");
const configPath = path.join(projectRoot, "tsconfig.json");
const outputRoot = projectRoot;
const entryPoints = [
  path.join(projectRoot, "index.ts"),
  path.join(projectRoot, "doppler-provider.ts"),
];

const excludedDirs = new Set([
  "node_modules",
  "dist",
  "build",
  "tests",
  "test-results",
  "kernel-tests",
  "app",
  "tools",
  "browser",
  "formats",
  "extension",
  "native",
]);
const excludedFilePatterns = [
  /\.spec\.tsx?$/i,
  /\.test\.tsx?$/i,
  /\.bench\.tsx?$/i,
  /(^|\/)test_.*\.tsx?$/i,
  /(^|\/)serve\.tsx?$/i,
  /(^|\/)vitest\.config\.tsx?$/i,
  /(^|\/)storage\/quickstart-downloader\.tsx?$/i,
  /(^|\/)storage\/preflight\.tsx?$/i,
];

const configFile = ts.readConfigFile(configPath, ts.sys.readFile);
if (configFile.error) {
  throw new Error(
    ts.flattenDiagnosticMessageText(configFile.error.messageText, "\n"),
  );
}
const parsedConfig = ts.parseJsonConfigFileContent(
  configFile.config,
  ts.sys,
  projectRoot,
);
const compilerOptions = parsedConfig.options;

function isExcludedPath(filePath) {
  const parts = filePath.split(path.sep);
  for (const part of parts) {
    if (excludedDirs.has(part)) {
      return true;
    }
  }
  return false;
}

function isSourceFile(filePath) {
  const ext = path.extname(filePath);
  if (ext !== ".ts" && ext !== ".tsx") {
    return false;
  }
  if (filePath.endsWith(".d.ts")) {
    return false;
  }
  if (isExcludedPath(filePath)) {
    return false;
  }
  for (const pattern of excludedFilePatterns) {
    if (pattern.test(filePath)) {
      return false;
    }
  }
  return true;
}

function collectSourceFiles(dir) {
  const files = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (isExcludedPath(fullPath)) {
        continue;
      }
      files.push(...collectSourceFiles(fullPath));
    } else if (entry.isFile()) {
      if (isSourceFile(fullPath)) {
        files.push(fullPath);
      }
    }
  }
  return files;
}

function toPosix(filePath) {
  return filePath.split(path.sep).join("/");
}

function toRel(filePath) {
  return toPosix(path.relative(projectRoot, filePath));
}

function resolveModule(moduleName, containingFile) {
  const resolved = ts.resolveModuleName(
    moduleName,
    containingFile,
    compilerOptions,
    ts.sys,
  );
  if (!resolved.resolvedModule) {
    return null;
  }
  const resolvedFile = resolved.resolvedModule.resolvedFileName;
  if (!resolvedFile) {
    return null;
  }
  const normalized = path.normalize(resolvedFile);
  if (!normalized.startsWith(projectRoot)) {
    return null;
  }
  if (!isSourceFile(normalized)) {
    return null;
  }
  return normalized;
}

function getModuleSymbolForFile(checker, sourceFile) {
  return checker.getSymbolAtLocation(sourceFile);
}

function getExportNames(checker, sourceFile) {
  const moduleSymbol = getModuleSymbolForFile(checker, sourceFile);
  if (!moduleSymbol) {
    return [];
  }
  return checker.getExportsOfModule(moduleSymbol).map((sym) => sym.getName());
}

function extractJSDocSummary(text) {
  const match = text.match(/^\s*\/\*\*([\s\S]*?)\*\//);
  if (!match) {
    return null;
  }
  const body = match[1];
  const lines = body
    .split("\n")
    .map((line) => line.replace(/^\s*\*\s?/, "").trim());
  const summary = lines.find((line) => line.length > 0);
  if (!summary) {
    return null;
  }
  return summary.replace(/\s+/g, " ");
}

function addToMapSet(map, key, value) {
  if (!map.has(key)) {
    map.set(key, new Set());
  }
  map.get(key).add(value);
}

function addToNestedMapSet(map, key, innerKey, value) {
  if (!map.has(key)) {
    map.set(key, new Map());
  }
  const inner = map.get(key);
  if (!inner.has(innerKey)) {
    inner.set(innerKey, new Set());
  }
  inner.get(innerKey).add(value);
}

const sourceFiles = collectSourceFiles(projectRoot);
const program = ts.createProgram({
  rootNames: sourceFiles,
  options: compilerOptions,
});
const checker = program.getTypeChecker();

const fileInfo = new Map();
for (const filePath of sourceFiles) {
  const sourceFile = program.getSourceFile(filePath);
  if (!sourceFile) {
    continue;
  }
  const exportNames = new Set(getExportNames(checker, sourceFile));
  fileInfo.set(filePath, { sourceFile, exportNames });
}

const dependencyGraph = new Map();
const importUsage = new Map();
const importAllUsage = new Map();
const importSpecs = [];

for (const [filePath, info] of fileInfo.entries()) {
  dependencyGraph.set(filePath, new Set());
  for (const statement of info.sourceFile.statements) {
    if (ts.isImportDeclaration(statement)) {
      const moduleName = statement.moduleSpecifier.text;
      const resolved = resolveModule(moduleName, filePath);
      if (!resolved) {
        continue;
      }
      dependencyGraph.get(filePath).add(resolved);
      const importClause = statement.importClause;
      if (!importClause) {
        continue;
      }
      if (importClause.name) {
        importSpecs.push({
          importer: filePath,
          target: resolved,
          name: "default",
          kind: "default",
        });
        addToNestedMapSet(importUsage, resolved, "default", filePath);
      }
      const namedBindings = importClause.namedBindings;
      if (namedBindings) {
        if (ts.isNamedImports(namedBindings)) {
          for (const element of namedBindings.elements) {
            const importedName = element.propertyName
              ? element.propertyName.text
              : element.name.text;
            importSpecs.push({
              importer: filePath,
              target: resolved,
              name: importedName,
              kind: "named",
            });
            addToNestedMapSet(importUsage, resolved, importedName, filePath);
          }
        } else if (ts.isNamespaceImport(namedBindings)) {
          addToMapSet(importAllUsage, resolved, filePath);
        }
      }
    } else if (
      ts.isExportDeclaration(statement) &&
      statement.moduleSpecifier
    ) {
      const moduleName = statement.moduleSpecifier.text;
      const resolved = resolveModule(moduleName, filePath);
      if (!resolved) {
        continue;
      }
      dependencyGraph.get(filePath).add(resolved);
      if (!statement.exportClause) {
        addToMapSet(importAllUsage, resolved, filePath);
        continue;
      }
      if (ts.isNamedExports(statement.exportClause)) {
        for (const element of statement.exportClause.elements) {
          const exportedName = element.propertyName
            ? element.propertyName.text
            : element.name.text;
          importSpecs.push({
            importer: filePath,
            target: resolved,
            name: exportedName,
            kind: "named",
          });
          addToNestedMapSet(importUsage, resolved, exportedName, filePath);
        }
      }
    } else if (ts.isImportEqualsDeclaration(statement)) {
      if (!ts.isExternalModuleReference(statement.moduleReference)) {
        continue;
      }
      const moduleName = statement.moduleReference.expression.text;
      const resolved = resolveModule(moduleName, filePath);
      if (!resolved) {
        continue;
      }
      dependencyGraph.get(filePath).add(resolved);
      addToMapSet(importAllUsage, resolved, filePath);
    }
  }

  const visitDynamicImports = (node) => {
    if (
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword
    ) {
      const [arg] = node.arguments;
      if (arg && ts.isStringLiteral(arg)) {
        const resolved = resolveModule(arg.text, filePath);
        if (resolved) {
          dependencyGraph.get(filePath).add(resolved);
          addToMapSet(importAllUsage, resolved, filePath);
        }
      }
    }
    ts.forEachChild(node, visitDynamicImports);
  };
  visitDynamicImports(info.sourceFile);
}

const auditResults = [];

function addAudit(result) {
  auditResults.push(result);
}

const exportNameToFiles = new Map();
for (const [filePath, info] of fileInfo.entries()) {
  for (const exportName of info.exportNames) {
    if (exportName === "default") {
      continue;
    }
    addToMapSet(exportNameToFiles, exportName, filePath);
  }
}

const entryPointSet = new Set(entryPoints.map((p) => path.normalize(p)));

for (const [filePath, info] of fileInfo.entries()) {
  if (entryPointSet.has(filePath)) {
    continue;
  }
  for (const exportName of info.exportNames) {
    if (exportName === "default") {
      continue;
    }
    const importedByAll = importAllUsage.get(filePath);
    const importedByName = importUsage.get(filePath)?.get(exportName);
    if (
      (importedByAll && importedByAll.size > 0) ||
      (importedByName && importedByName.size > 0)
    ) {
      continue;
    }
    addAudit({
      type: "dead_export",
      file: toRel(filePath),
      symbol: exportName,
      severity: "medium",
    });
  }
}

for (const spec of importSpecs) {
  const targetInfo = fileInfo.get(spec.target);
  if (!targetInfo) {
    continue;
  }
  if (!targetInfo.exportNames.has(spec.name)) {
    addAudit({
      type: "missing_export",
      file: toRel(spec.importer),
      symbol: spec.name,
      from: toRel(spec.target),
      severity: "high",
    });
  }
}

for (const [symbol, files] of exportNameToFiles.entries()) {
  if (files.size < 2) {
    continue;
  }
  addAudit({
    type: "shadow_export",
    symbol,
    files: Array.from(files).map(toRel).sort(),
    severity: "low",
  });
}

const fileList = Array.from(fileInfo.keys());
const indexByFile = new Map(fileList.map((file, idx) => [file, idx]));
const indices = new Array(fileList.length).fill(-1);
const lowLinks = new Array(fileList.length).fill(-1);
const indexStack = [];
const onStack = new Array(fileList.length).fill(false);
let indexCounter = 0;
const sccs = [];

function tarjan(filePath) {
  const idx = indexByFile.get(filePath);
  indices[idx] = indexCounter;
  lowLinks[idx] = indexCounter;
  const lowLinkIndex = indexCounter;
  indexCounter += 1;
  indexStack.push(filePath);
  onStack[idx] = true;

  const deps = dependencyGraph.get(filePath) || new Set();
  for (const dep of deps) {
    const depIdx = indexByFile.get(dep);
    if (depIdx === undefined) {
      continue;
    }
    if (indices[depIdx] === -1) {
      tarjan(dep);
      lowLinks[idx] = Math.min(lowLinks[idx], lowLinks[depIdx]);
    } else if (onStack[depIdx]) {
      lowLinks[idx] = Math.min(lowLinks[idx], indices[depIdx]);
    }
  }

  if (lowLinks[idx] === lowLinkIndex) {
    const component = [];
    while (true) {
      const w = indexStack.pop();
      const wIdx = indexByFile.get(w);
      onStack[wIdx] = false;
      component.push(w);
      if (w === filePath) {
        break;
      }
    }
    sccs.push(component);
  }
}
indexCounter = 0;
for (const filePath of fileList) {
  const idx = indexByFile.get(filePath);
  if (indices[idx] === -1) {
    tarjan(filePath);
  }
}

for (const component of sccs) {
  if (component.length < 2) {
    continue;
  }
  const cycle = component.map(toRel).sort();
  cycle.push(cycle[0]);
  addAudit({
    type: "circular_dep",
    cycle,
    severity: "high",
  });
}

const reachable = new Set();
const stack = entryPoints.filter((file) => fileInfo.has(file));
while (stack.length > 0) {
  const current = stack.pop();
  if (reachable.has(current)) {
    continue;
  }
  reachable.add(current);
  const deps = dependencyGraph.get(current) || new Set();
  for (const dep of deps) {
    if (!reachable.has(dep)) {
      stack.push(dep);
    }
  }
}

for (const filePath of fileInfo.keys()) {
  if (!reachable.has(filePath)) {
    addAudit({
      type: "orphan_file",
      file: toRel(filePath),
      severity: "medium",
    });
  }
}

function isDeclarationName(node) {
  const parent = node.parent;
  if (!parent) {
    return false;
  }
  if (
    (ts.isFunctionDeclaration(parent) ||
      ts.isVariableDeclaration(parent) ||
      ts.isClassDeclaration(parent)) &&
    parent.name === node
  ) {
    return true;
  }
  if (
    (ts.isParameter(parent) || ts.isPropertyDeclaration(parent)) &&
    parent.name === node
  ) {
    return true;
  }
  if (ts.isMethodDeclaration(parent) && parent.name === node) {
    return true;
  }
  return false;
}

for (const [filePath, info] of fileInfo.entries()) {
  const sourceFile = info.sourceFile;
  const localCandidates = [];
  for (const statement of sourceFile.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name) {
      const symbol = checker.getSymbolAtLocation(statement.name);
      if (!symbol) {
        continue;
      }
      if (info.exportNames.has(statement.name.text)) {
        continue;
      }
      localCandidates.push({ symbol, name: statement.name.text });
    } else if (ts.isVariableStatement(statement)) {
      for (const decl of statement.declarationList.declarations) {
        if (!ts.isIdentifier(decl.name) || !decl.initializer) {
          continue;
        }
        if (
          !ts.isArrowFunction(decl.initializer) &&
          !ts.isFunctionExpression(decl.initializer)
        ) {
          continue;
        }
        const symbol = checker.getSymbolAtLocation(decl.name);
        if (!symbol) {
          continue;
        }
        if (info.exportNames.has(decl.name.text)) {
          continue;
        }
        localCandidates.push({ symbol, name: decl.name.text });
      }
    }
  }

  if (localCandidates.length === 0) {
    continue;
  }

  const usageMap = new Map(
    localCandidates.map((candidate) => [candidate.symbol, 0]),
  );

  function visit(node) {
    if (ts.isIdentifier(node) && !isDeclarationName(node)) {
      const symbol = checker.getSymbolAtLocation(node);
      if (usageMap.has(symbol)) {
        usageMap.set(symbol, usageMap.get(symbol) + 1);
      }
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);

  for (const candidate of localCandidates) {
    const count = usageMap.get(candidate.symbol) || 0;
    if (count === 0) {
      addAudit({
        type: "dead_internal",
        file: toRel(filePath),
        symbol: candidate.name,
        severity: "low",
      });
    }
  }
}

auditResults.sort((a, b) => {
  const typeCmp = a.type.localeCompare(b.type);
  if (typeCmp !== 0) {
    return typeCmp;
  }
  const fileA = a.file || (a.cycle ? a.cycle[0] : "");
  const fileB = b.file || (b.cycle ? b.cycle[0] : "");
  const fileCmp = fileA.localeCompare(fileB);
  if (fileCmp !== 0) {
    return fileCmp;
  }
  const symA = a.symbol || "";
  const symB = b.symbol || "";
  return symA.localeCompare(symB);
});

const auditPath = path.join(outputRoot, "AUDIT_RESULTS.jsonl");
fs.writeFileSync(
  auditPath,
  auditResults.map((item) => JSON.stringify(item)).join("\n") + "\n",
);

const moduleIndexByGroup = new Map();
for (const [filePath, info] of fileInfo.entries()) {
  const rel = toRel(filePath);
  const group = rel.includes("/") ? rel.split("/")[0] : "Root";
  if (!moduleIndexByGroup.has(group)) {
    moduleIndexByGroup.set(group, []);
  }
  const text = info.sourceFile.getFullText();
  const summary =
    (extractJSDocSummary(text) || "No description.").replace(/\|/g, "\\|");
  const exportsList = Array.from(info.exportNames).sort();
  moduleIndexByGroup.get(group).push({
    module: rel,
    summary,
    exports: exportsList,
  });
}

const moduleGroups = Array.from(moduleIndexByGroup.keys()).sort((a, b) => {
  if (a === "Root") return -1;
  if (b === "Root") return 1;
  return a.localeCompare(b);
});

let moduleIndexMd = "# Doppler Module Index\n\n";
for (const group of moduleGroups) {
  moduleIndexMd += `## ${group}\n\n`;
  moduleIndexMd += "| Module | Description | Exports |\n";
  moduleIndexMd += "|--------|-------------|---------|\n";
  const entries = moduleIndexByGroup.get(group);
  entries.sort((a, b) => a.module.localeCompare(b.module));
  for (const entry of entries) {
    const exportsCell = entry.exports.length
      ? entry.exports.join(", ").replace(/\|/g, "\\|")
      : "—";
    moduleIndexMd += `| \`${entry.module}\` | ${entry.summary} | ${exportsCell} |\n`;
  }
  moduleIndexMd += "\n";
}

fs.writeFileSync(path.join(outputRoot, "MODULE_INDEX.md"), moduleIndexMd);

const nodeIdMap = new Map();
function nodeIdFor(filePath) {
  if (nodeIdMap.has(filePath)) {
    return nodeIdMap.get(filePath);
  }
  const base = toRel(filePath).replace(/[^a-zA-Z0-9_]/g, "_");
  const id = `node_${base}`;
  nodeIdMap.set(filePath, id);
  return id;
}

let dependencyMd = "# Doppler Dependency Graph\n\n";
dependencyMd += "## Entry Points\n";
for (const entry of entryPoints) {
  if (fileInfo.has(entry)) {
    dependencyMd += `- \`${toRel(entry)}\`\n`;
  }
}
dependencyMd += "\n## Graph (Mermaid)\n";
dependencyMd += "```mermaid\ngraph TD\n";

const groupsByDir = new Map();
for (const filePath of fileInfo.keys()) {
  const rel = toRel(filePath);
  const group = rel.includes("/") ? rel.split("/")[0] : "Root";
  if (!groupsByDir.has(group)) {
    groupsByDir.set(group, []);
  }
  groupsByDir.get(group).push(filePath);
}

for (const [group, files] of Array.from(groupsByDir.entries()).sort(
  (a, b) => a[0].localeCompare(b[0]),
)) {
  dependencyMd += `  subgraph ${group}\n`;
  const sortedFiles = files.slice().sort();
  for (const filePath of sortedFiles) {
    const nodeId = nodeIdFor(filePath);
    dependencyMd += `    ${nodeId}["${toRel(filePath)}"]\n`;
  }
  dependencyMd += "  end\n";
}

const edgeLines = [];
for (const [from, deps] of dependencyGraph.entries()) {
  for (const dep of deps) {
    if (!fileInfo.has(dep)) {
      continue;
    }
    edgeLines.push(
      `  ${nodeIdFor(from)} --> ${nodeIdFor(dep)}`,
    );
  }
}
edgeLines.sort();
dependencyMd += `${edgeLines.join("\n")}\n`;
dependencyMd += "```\n";

fs.writeFileSync(path.join(outputRoot, "DEPENDENCY_GRAPH.md"), dependencyMd);

function formatParameters(signature, fallbackNode) {
  return signature.parameters
    .map((param) => {
      const decl =
        param.valueDeclaration ||
        (param.declarations && param.declarations[0]) ||
        fallbackNode;
      const name = decl && decl.name ? decl.name.getText() : param.getName();
      const typeNode = decl || fallbackNode;
      const type = checker.typeToString(
        checker.getTypeOfSymbolAtLocation(param, typeNode),
        typeNode,
        ts.TypeFormatFlags.NoTruncation,
      );
      const optional =
        decl && (decl.questionToken || decl.initializer) ? "?" : "";
      const rest = decl && decl.dotDotDotToken ? "..." : "";
      return `${rest}${name}${optional}: ${type}`;
    })
    .join(", ");
}

function formatFunctionSignature(name, declaration) {
  const signature = checker.getSignatureFromDeclaration(declaration);
  if (!signature) {
    return `${name}()`;
  }
  const params = formatParameters(signature, declaration);
  const returnType = checker.typeToString(
    signature.getReturnType(),
    declaration,
    ts.TypeFormatFlags.NoTruncation,
  );
  return `${name}(${params}): ${returnType}`;
}

function formatClassSignature(classDecl) {
  const lines = [];
  const className = classDecl.name ? classDecl.name.text : "AnonymousClass";
  lines.push(`class ${className} {`);
  for (const member of classDecl.members) {
    if (ts.isConstructorDeclaration(member)) {
      const signature = checker.getSignatureFromDeclaration(member);
      if (!signature) {
        continue;
      }
      const params = formatParameters(signature, member);
      lines.push(`  constructor(${params}): void;`);
    } else if (ts.isMethodDeclaration(member) && member.name) {
      if (
        member.modifiers &&
        member.modifiers.some((mod) =>
          [ts.SyntaxKind.PrivateKeyword, ts.SyntaxKind.ProtectedKeyword].includes(
            mod.kind,
          ),
        )
      ) {
        continue;
      }
      if (ts.isPrivateIdentifier(member.name)) {
        continue;
      }
      const name = member.name.getText();
      const signature = checker.getSignatureFromDeclaration(member);
      if (!signature) {
        continue;
      }
      const params = formatParameters(signature, member);
      const returnType = checker.typeToString(
        signature.getReturnType(),
        member,
        ts.TypeFormatFlags.NoTruncation,
      );
      lines.push(`  ${name}(${params}): ${returnType};`);
    } else if (ts.isPropertyDeclaration(member) && member.name) {
      if (
        member.modifiers &&
        member.modifiers.some((mod) =>
          [ts.SyntaxKind.PrivateKeyword, ts.SyntaxKind.ProtectedKeyword].includes(
            mod.kind,
          ),
        )
      ) {
        continue;
      }
      if (ts.isPrivateIdentifier(member.name)) {
        continue;
      }
      const name = member.name.getText();
      const type = checker.typeToString(
        checker.getTypeAtLocation(member),
        member,
        ts.TypeFormatFlags.NoTruncation,
      );
      lines.push(`  ${name}: ${type};`);
    }
  }
  lines.push("}");
  return lines.join("\n");
}

function formatTypeDeclaration(node) {
  const printer = ts.createPrinter({ newLine: ts.NewLineKind.LineFeed });
  return printer.printNode(ts.EmitHint.Unspecified, node, node.getSourceFile());
}

function collectApiForModule(modulePath) {
  const sourceFile = program.getSourceFile(modulePath);
  if (!sourceFile) {
    return [];
  }
  const moduleSymbol = getModuleSymbolForFile(checker, sourceFile);
  if (!moduleSymbol) {
    return [];
  }
  const exports = checker.getExportsOfModule(moduleSymbol);
  const entries = [];
  for (const sym of exports) {
    const name = sym.getName();
    if (name === "default") {
      continue;
    }
    const resolvedSymbol =
      sym.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(sym) : sym;
    const decls = resolvedSymbol.declarations || sym.declarations || [];
    if (decls.length === 0) {
      continue;
    }
    const decl = decls[0];
    if (ts.isClassDeclaration(decl)) {
      entries.push({
        kind: "class",
        name,
        text: formatClassSignature(decl),
      });
    } else if (ts.isFunctionDeclaration(decl)) {
      entries.push({
        kind: "function",
        name,
        text: `function ${formatFunctionSignature(name, decl)}`,
      });
    } else if (ts.isVariableDeclaration(decl) && decl.name) {
      const type = checker.typeToString(
        checker.getTypeAtLocation(decl),
        decl,
        ts.TypeFormatFlags.NoTruncation,
      );
      entries.push({
        kind: "variable",
        name,
        text: `const ${decl.name.getText()}: ${type}`,
      });
    } else if (ts.isTypeAliasDeclaration(decl) || ts.isInterfaceDeclaration(decl)) {
      entries.push({
        kind: "type",
        name,
        text: formatTypeDeclaration(decl),
      });
    } else if (ts.isEnumDeclaration(decl)) {
      entries.push({
        kind: "enum",
        name,
        text: formatTypeDeclaration(decl),
      });
    } else {
      const type = checker.typeToString(
        checker.getTypeOfSymbolAtLocation(sym, decl),
        decl,
        ts.TypeFormatFlags.NoTruncation,
      );
      entries.push({
        kind: "value",
        name,
        text: `const ${name}: ${type}`,
      });
    }
  }
  return entries.sort((a, b) => a.name.localeCompare(b.name));
}

let apiMd = "# Doppler API Surface\n\n";
for (const entry of entryPoints) {
  if (!fileInfo.has(entry)) {
    continue;
  }
  apiMd += `## ${toRel(entry)}\n\n`;
  const entries = collectApiForModule(entry);
  const byKind = new Map();
  for (const entryItem of entries) {
    if (!byKind.has(entryItem.kind)) {
      byKind.set(entryItem.kind, []);
    }
    byKind.get(entryItem.kind).push(entryItem);
  }
  const kindOrder = ["class", "function", "variable", "type", "enum", "value"];
  for (const kind of kindOrder) {
    const list = byKind.get(kind);
    if (!list || list.length === 0) {
      continue;
    }
    const heading =
      kind === "class"
        ? "Classes"
        : kind === "function"
          ? "Functions"
          : kind === "variable"
            ? "Constants"
            : kind === "type"
              ? "Types"
              : kind === "enum"
                ? "Enums"
                : "Values";
    apiMd += `### ${heading}\n\n`;
    for (const item of list) {
      apiMd += `#### ${item.name}\n\n`;
      apiMd += "```typescript\n";
      apiMd += `${item.text}\n`;
      apiMd += "```\n\n";
    }
  }
}

fs.writeFileSync(path.join(outputRoot, "API_SURFACE.md"), apiMd);

console.log("Wrote:", {
  audit: auditPath,
  moduleIndex: path.join(outputRoot, "MODULE_INDEX.md"),
  dependencyGraph: path.join(outputRoot, "DEPENDENCY_GRAPH.md"),
  apiSurface: path.join(outputRoot, "API_SURFACE.md"),
});
