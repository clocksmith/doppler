import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

// Installed-package tests omit Node typings. This complementary consumer has them
// and must be able to use the handler directly with Node's real createServer type.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const filename = path.join(root, 'pack-http-consumer.typecheck.ts');
const source = `
import http from 'node:http';
import { createPackServeHandler } from './src/cli/serve/pack-handler.js';
declare const options: Parameters<typeof createPackServeHandler>[0];
const handler = createPackServeHandler(options);
http.createServer(handler);
const drained: Promise<void> = handler.close();
// @ts-expect-error Explicit session authority cannot be omitted.
createPackServeHandler({ policy: options.policy, token: options.token });
`;
const options = { strict: true, noEmit: true, types: ['node'],
  module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext,
  target: ts.ScriptTarget.ES2022, skipLibCheck: false };
const host = ts.createCompilerHost(options);
const getSourceFile = host.getSourceFile.bind(host);
host.getSourceFile = (name, ...args) => name === filename
  ? ts.createSourceFile(name, source, ts.ScriptTarget.ES2022, true)
  : getSourceFile(name, ...args);
const diagnostics = ts.getPreEmitDiagnostics(ts.createProgram([filename], options, host));
assert.equal(diagnostics.length, 0, ts.formatDiagnosticsWithColorAndContext(diagnostics, {
  getCurrentDirectory: () => root, getCanonicalFileName: name => name, getNewLine: () => '\n',
}));
console.log('pack-http-declarations.test: ok');
