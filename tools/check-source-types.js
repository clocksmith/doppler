#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';

const root = process.cwd();
const configPath = path.join(root, 'tsconfig.source-strict.json');
const config = ts.readConfigFile(configPath, ts.sys.readFile);
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root);
if (!parsed.options.noImplicitAny || !parsed.options.strictNullChecks || !parsed.options.checkJs) {
  throw new Error('Strict source checks require checkJs, noImplicitAny and strictNullChecks.');
}
const program = ts.createProgram(parsed.fileNames, parsed.options);
const diagnostics = [...parsed.errors, ...ts.getPreEmitDiagnostics(program)];
for (const name of parsed.fileNames.filter(file => file.endsWith('.js'))) {
  if (!program.getSourceFile(name) || program.getSourceFile(name).isDeclarationFile) {
    throw new Error(`Implementation check was shadowed by a declaration: ${name}`);
  }
}
const relative = file => path.relative(root, file).split(path.sep).join('/');
const checked = new Set(parsed.fileNames.map(relative));
const implementations = ts.sys.readDirectory(root, ['.js'], ['node_modules', 'dist', 'tmp'], ['src/**/*.js', 'demo/**/*.js']).map(relative).sort();
const declarationAnyCounts = {};
for (const file of ts.sys.readDirectory(root, ['.ts'], ['node_modules', 'dist', 'tmp'], ['src/**/*.d.ts', 'demo/**/*.d.ts']).sort()) {
  const source = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
  let count = 0;
  function visit(node) { if (node.kind === ts.SyntaxKind.AnyKeyword) count++; ts.forEachChild(node, visit); }
  visit(source);
  if (count) declarationAnyCounts[relative(file)] = count;
}
const current = { schemaVersion: 1, uncheckedImplementations: implementations.filter(file => !checked.has(file)), declarationAnyCounts };
const policyPath = path.join(root, 'tools/policies/source-type-debt.json');
const previous = fs.existsSync(policyPath) ? JSON.parse(fs.readFileSync(policyPath, 'utf8')) : null;
const failures = [];
if (previous) {
  const allowed = new Set(previous.uncheckedImplementations);
  for (const file of current.uncheckedImplementations) if (!allowed.has(file)) failures.push(`New unchecked implementation: ${file}. Add it to tsconfig.source-strict.json.`);
  for (const [file, count] of Object.entries(declarationAnyCounts)) {
    if (count > (previous.declarationAnyCounts[file] ?? 0)) failures.push(`Declaration any debt grew: ${file} (${count}).`);
  }
}
if (diagnostics.length) console.error(ts.formatDiagnosticsWithColorAndContext(diagnostics, {
  getCanonicalFileName: name => name, getCurrentDirectory: () => root, getNewLine: () => '\n',
}));
if (failures.length) console.error(failures.join('\n'));
if (diagnostics.length || failures.length) process.exitCode = 1;
else if (process.argv.includes('--write')) fs.writeFileSync(policyPath, `${JSON.stringify(current, null, 2)}\n`);
else if (!previous || JSON.stringify(previous) !== JSON.stringify(current)) {
  throw new Error('Source type debt inventory is stale; run node tools/check-source-types.js --write to ratchet improvements.');
}
if (!process.exitCode) console.log(`[source-types] ${checked.size} strict roots; ${current.uncheckedImplementations.length} explicitly inventoried unchecked implementations; no debt growth`);
