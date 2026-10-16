#!/usr/bin/env node

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const configPath = path.join(root, 'tsconfig.source-unused.json');
const config = ts.readConfigFile(configPath, ts.sys.readFile);
const formatHost = {
  getCanonicalFileName: name => name,
  getCurrentDirectory: () => root,
  getNewLine: () => '\n',
};
if (config.error) {
  throw new Error(ts.formatDiagnostics([config.error], formatHost));
}
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root);
if (!parsed.options.checkJs || !parsed.options.noUnusedLocals || !parsed.options.noEmit) {
  throw new Error('Unused-local checks require checkJs, noUnusedLocals and noEmit.');
}
if (parsed.errors.length || parsed.fileNames.length === 0) {
  throw new Error(ts.formatDiagnostics(parsed.errors, formatHost) || 'No unused-local roots configured.');
}
const program = ts.createProgram(parsed.fileNames, parsed.options);
// This gradual gate checks unused declarations, not general type correctness.
// The strict source/declaration checks remain separate and unchanged.
const unusedCodes = new Set([6133, 6192, 6196, 6198, 6199]);
const diagnostics = [...program.getOptionsDiagnostics()];
for (const file of parsed.fileNames) {
  const source = program.getSourceFile(file);
  if (!source || source.isDeclarationFile || !file.endsWith('.js')) {
    throw new Error(`Unused-local root must resolve to its JavaScript implementation: ${file}`);
  }
  diagnostics.push(...program.getSyntacticDiagnostics(source));
  diagnostics.push(...program.getSemanticDiagnostics(source).filter(d => unusedCodes.has(d.code)));
}
if (diagnostics.length) {
  console.error(ts.formatDiagnosticsWithColorAndContext(diagnostics, formatHost));
  process.exitCode = 1;
} else {
  console.log(`[unused-locals] ${parsed.fileNames.length} JavaScript roots; no unused declarations`);
}
