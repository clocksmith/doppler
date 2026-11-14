import ts from 'typescript';

export const RULE_IDS = ["JS-001", "JS-002"];

// Parser-only baseline: semantic/type checking remains in its owning source gates.
export function checkJavaScriptSource(source, file, legacyMjs = []) {
  const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const options = { allowJs: true, noEmit: true, noLib: true, noResolve: true };
  const host = ts.createCompilerHost(options);
  host.getSourceFile = name => name === file ? ast : undefined;
  const program = ts.createProgram([file], options, host);
  const diagnostics = program.getSyntacticDiagnostics(ast).map(diagnostic => ({
    ruleId: 'JS-001',
    line: ast.getLineAndCharacterOfPosition(diagnostic.start ?? 0).line + 1,
    message: ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'),
  }));
  if (file.endsWith('.mjs') && !legacyMjs.some(entry => entry.path === file)) {
    diagnostics.push({ ruleId: 'JS-002', line: 1, message: 'Use .js for new modules; legacy .mjs paths are explicitly inventoried.' });
  }
  return diagnostics;
}
