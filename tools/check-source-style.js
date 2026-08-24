#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const GOVERNED_ROOTS = ['src', 'demo'];
const WRITE = process.argv.includes('--write');
const LIST_COMPUTE_CANDIDATES = process.argv.includes('--list-compute-candidates');
const RAW_CONSOLE_ALLOWED_PREFIXES = ['src/cli/', 'src/debug/', 'demo/'];
const RAW_CONSOLE_ALLOWED_FILES = new Set(['src/gpu/device.js']);
const RAW_CONSOLE_PATTERN = /\bconsole\.(?:debug|error|info|log|warn)\s*\(/g;
const INLINE_WGSL_PATTERNS = [
  /\/\*\s*wgsl\s*\*\/\s*`/gu,
  /\b(?:const|let|var)\s+[A-Z0-9_]*SHADER[A-Z0-9_]*\s*=\s*`/gu,
  /createShaderModule\s*\(\s*\{\s*code\s*:\s*['"`]/gu,
  /@compute\b/gu,
  /@group\s*\(/gu,
];
const WGSL_CODEGEN_PREFIX = 'src/gpu/kernels/codegen/';
const SOURCE_COMPUTE_POLICY_PATH = path.join(
  ROOT,
  'tools/policies/source-compute-policy.json'
);
const SOURCE_COMPUTE_POLICY = JSON.parse(fs.readFileSync(SOURCE_COMPUTE_POLICY_PATH, 'utf8'));
const TYPED_ARRAY_PATTERN = /\b(?:Float(?:16|32|64)Array|Uint(?:8|16|32)Array|Int(?:8|16|32)Array)\b/u;
const TENSOR_SEMANTIC_PATTERN = /(?:tensor|logit|hidden|embedding|expert|attention|softmax|norm|projection|latent|feature|weight|activation|gradient|pool|matrix|matmul|router|modelOutput|sample|codebook|rotation|qjl|quantiz|dequantiz|rope|scheduler|cache|f16|bf16)\w*/iu;
const NUMERIC_ASSIGNMENT_OPERATORS = new Set(['=', '+=', '-=', '*=', '/=']);
const NUMERIC_MATH_METHODS = new Set(['cos', 'exp', 'pow', 'sin', 'sqrt', 'tanh']);
const INFERRED_GEOMETRY_PATTERNS = [
  {
    pattern: /Math\.sqrt\s*\(\s*(?:elementCount|numElements|numPatches|patchCount)\s*\)/g,
    invariant: 'INV-GEOMETRY-009',
  },
  {
    pattern: /Math\.sqrt\s*\(\s*(?:features|patches|tokens)\.length\s*\)/g,
    invariant: 'INV-GEOMETRY-009',
  },
];

function walkJavaScript(directory, files = []) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      walkJavaScript(entryPath, files);
    } else if (entry.name.endsWith('.js')) {
      files.push(entryPath);
    }
  }
  return files;
}

function removeJSDoc(source) {
  return source
    .replace(/^[ \t]*\/\*\*[\s\S]*?\*\/[ \t]*(?:\r?\n)?/gm, '')
    .replace(/\n{3,}/g, '\n\n');
}

function getFunctionSymbol(node, sourceFile) {
  if (node.name) {
    const owner = node.parent && ts.isClassLike(node.parent) && node.parent.name
      ? `${node.parent.name.getText(sourceFile)}.`
      : '';
    return `${owner}${node.name.getText(sourceFile)}`;
  }
  if (ts.isVariableDeclaration(node.parent) && ts.isIdentifier(node.parent.name)) {
    return node.parent.name.text;
  }
  if (ts.isPropertyAssignment(node.parent)) {
    return node.parent.name.getText(sourceFile);
  }
  const line = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1;
  return `<anonymous@${line}>`;
}

function isFunctionLike(node) {
  return ts.isFunctionDeclaration(node)
    || ts.isFunctionExpression(node)
    || ts.isArrowFunction(node)
    || ts.isMethodDeclaration(node)
    || ts.isGetAccessorDeclaration(node)
    || ts.isSetAccessorDeclaration(node);
}

function inspectNumericFunction(node, sourceFile) {
  const functionSource = node.getText(sourceFile);
  if (!TENSOR_SEMANTIC_PATTERN.test(functionSource)) {
    return [];
  }
  let loopDepth = 0;
  let hasElementWrite = false;
  let hasLoopedElementWrite = false;
  let hasLoopedSet = false;
  let hasNumericMath = false;
  let hasReadback = false;
  let hasUpload = false;

  function visit(current) {
    if (current !== node && isFunctionLike(current)) {
      return;
    }
    const isLoop = ts.isForStatement(current)
      || ts.isForInStatement(current)
      || ts.isForOfStatement(current)
      || ts.isWhileStatement(current)
      || ts.isDoStatement(current);
    if (isLoop) loopDepth += 1;

    if (
      ts.isBinaryExpression(current)
      && ts.isElementAccessExpression(current.left)
      && NUMERIC_ASSIGNMENT_OPERATORS.has(current.operatorToken.getText(sourceFile))
    ) {
      hasElementWrite = true;
      if (loopDepth > 0) hasLoopedElementWrite = true;
    }
    if (ts.isCallExpression(current)) {
      const callee = current.expression;
      const callName = ts.isIdentifier(callee)
        ? callee.text
        : ts.isPropertyAccessExpression(callee)
          ? callee.name.text
          : '';
      if (loopDepth > 0 && callName === 'set') hasLoopedSet = true;
      if (['getMappedRange', 'mapAsync', 'readBuffer', 'readBufferWithCleanup', 'readback'].includes(callName)) {
        hasReadback = true;
      }
      if (['createTensor', 'createUploadedTensor', 'upload', 'writeBuffer'].includes(callName)) {
        hasUpload = true;
      }
      if (
        ts.isPropertyAccessExpression(callee)
        && callee.expression.getText(sourceFile) === 'Math'
        && NUMERIC_MATH_METHODS.has(callee.name.text)
      ) {
        hasNumericMath = true;
      }
    }

    ts.forEachChild(current, visit);
    if (isLoop) loopDepth -= 1;
  }

  if (node.body) visit(node.body);
  const signals = [];
  if (hasLoopedElementWrite) signals.push('looped-element-write');
  if (hasLoopedSet) signals.push('looped-typed-array-set');
  if (hasElementWrite && hasNumericMath) signals.push('math-with-element-write');
  if ((hasLoopedElementWrite || hasLoopedSet) && hasReadback && hasUpload) {
    signals.push('readback-compute-upload');
  }
  return signals;
}

function detectSourceComputeCandidates(file, relative, source) {
  if (!relative.startsWith('src/') || !TYPED_ARRAY_PATTERN.test(source)) {
    return [];
  }
  const sourceFile = ts.createSourceFile(
    file,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.JS
  );
  const candidates = [];

  function visit(node) {
    if (isFunctionLike(node)) {
      const signals = inspectNumericFunction(node, sourceFile);
      if (signals.length > 0) {
        const line = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1;
        candidates.push({
          id: `${relative}#${getFunctionSymbol(node, sourceFile)}`,
          file: relative,
          line,
          signals,
        });
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  return candidates;
}

function validateSourceComputePolicy(candidates) {
  const policyViolations = [];
  const categories = new Map(
    SOURCE_COMPUTE_POLICY.categories.map((category) => [category.id, category])
  );
  const reviews = new Map();
  for (const review of SOURCE_COMPUTE_POLICY.reviews) {
    if (reviews.has(review.id)) {
      policyViolations.push({
        file: SOURCE_COMPUTE_POLICY_PATH,
        invariant: 'INV-COMPUTE-008',
        detail: `duplicate source-compute review "${review.id}"`,
      });
      continue;
    }
    reviews.set(review.id, review);
    if (!categories.has(review.category)) {
      policyViolations.push({
        file: SOURCE_COMPUTE_POLICY_PATH,
        invariant: 'INV-COMPUTE-008',
        detail: `review "${review.id}" has unknown category "${review.category}"`,
      });
    }
    if (
      review.category === 'quarantined-experimental'
      && !review.id.startsWith('src/experimental/')
    ) {
      policyViolations.push({
        file: SOURCE_COMPUTE_POLICY_PATH,
        invariant: 'INV-COMPUTE-008',
        detail: `review "${review.id}" is marked quarantined-experimental outside src/experimental`,
      });
    }
    if (typeof review.reason !== 'string' || review.reason.trim().length < 24) {
      policyViolations.push({
        file: SOURCE_COMPUTE_POLICY_PATH,
        invariant: 'INV-COMPUTE-008',
        detail: `review "${review.id}" requires a specific reason`,
      });
    }
  }

  const candidateIds = new Set(candidates.map((candidate) => candidate.id));
  for (const candidate of candidates) {
    const review = reviews.get(candidate.id);
    if (!review) {
      policyViolations.push({
        file: candidate.file,
        invariant: 'INV-COMPUTE-008',
        detail: `unreviewed numeric candidate ${candidate.id} at line ${candidate.line} (${candidate.signals.join(', ')})`,
      });
      continue;
    }
    const category = categories.get(review.category);
    if (category?.runtimeTensorCompute === true) {
      policyViolations.push({
        file: candidate.file,
        invariant: 'INV-COMPUTE-008',
        detail: `forbidden runtime tensor compute ${candidate.id}: ${review.reason}`,
      });
    }
  }
  for (const review of SOURCE_COMPUTE_POLICY.reviews) {
    if (!candidateIds.has(review.id)) {
      policyViolations.push({
        file: SOURCE_COMPUTE_POLICY_PATH,
        invariant: 'INV-COMPUTE-008',
        detail: `stale source-compute review "${review.id}"`,
      });
    }
  }
  return policyViolations;
}

const files = GOVERNED_ROOTS.flatMap((root) => walkJavaScript(path.join(ROOT, root)));
const violations = [];
const sourceComputeCandidates = [];
let changedFiles = 0;
let removedBlocks = 0;

for (const file of files) {
  const source = fs.readFileSync(file, 'utf8');
  const relative = path.relative(ROOT, file).split(path.sep).join('/');
  sourceComputeCandidates.push(...detectSourceComputeCandidates(file, relative, source));
  if (WRITE) {
    if (source.includes('/**')) {
      changedFiles += 1;
      removedBlocks += source.match(/\/\*\*/g)?.length ?? 0;
      fs.writeFileSync(file, removeJSDoc(source));
    }
    continue;
  }
  const lines = source.split(/\r?\n/);
  const jsDocLines = [];
  for (let index = 0; index < lines.length; index += 1) {
    if (lines[index].includes('/**')) {
      jsDocLines.push(index + 1);
    }
  }
  if (jsDocLines.length > 0) {
    violations.push({
      file: relative,
      invariant: 'declaration-files-own-api-types',
      detail: `implementation JSDoc at lines ${jsDocLines.join(',')}`,
    });
  }

  const declarationPath = file.replace(/\.js$/u, '.d.ts');
  if (!fs.existsSync(declarationPath)) {
    violations.push({
      file: relative,
      invariant: 'declaration-files-own-api-types',
      detail: 'missing sibling declaration file',
    });
  }

  const rawConsoleAllowed = RAW_CONSOLE_ALLOWED_FILES.has(relative)
    || RAW_CONSOLE_ALLOWED_PREFIXES.some((prefix) => relative.startsWith(prefix));
  if (!rawConsoleAllowed) {
    RAW_CONSOLE_PATTERN.lastIndex = 0;
    const consoleLines = [];
    for (;;) {
      const match = RAW_CONSOLE_PATTERN.exec(source);
      if (!match) break;
      consoleLines.push(source.slice(0, match.index).split(/\r?\n/).length);
    }
    if (consoleLines.length > 0) {
      violations.push({
        file: relative,
        invariant: 'no-raw-runtime-console',
        detail: `raw console call at lines ${consoleLines.join(',')}`,
      });
    }
  }

  for (const pattern of INLINE_WGSL_PATTERNS) {
    if (relative.startsWith(WGSL_CODEGEN_PREFIX)) continue;
    pattern.lastIndex = 0;
    const inlineShaderLines = [];
    for (;;) {
      const match = pattern.exec(source);
      if (!match) break;
      inlineShaderLines.push(source.slice(0, match.index).split(/\r?\n/).length);
    }
    if (inlineShaderLines.length > 0) {
      violations.push({
        file: relative,
        invariant: 'INV-COMPUTE-008',
        detail: `inline WGSL at lines ${inlineShaderLines.join(',')}; shader programs require owned .wgsl files`,
      });
    }
  }

  for (const check of INFERRED_GEOMETRY_PATTERNS) {
    check.pattern.lastIndex = 0;
    const geometryLines = [];
    for (;;) {
      const match = check.pattern.exec(source);
      if (!match) break;
      geometryLines.push(source.slice(0, match.index).split(/\r?\n/).length);
    }
    if (geometryLines.length > 0) {
      violations.push({
        file: relative,
        invariant: check.invariant,
        detail: `runtime geometry inferred at lines ${geometryLines.join(',')}`,
      });
    }
  }
}

if (!WRITE) {
  violations.push(...validateSourceComputePolicy(sourceComputeCandidates));
}

if (LIST_COMPUTE_CANDIDATES) {
  console.log(JSON.stringify(sourceComputeCandidates, null, 2));
  process.exit(0);
}

if (WRITE) {
  console.log(
    `[source:style:sync] removed ${removedBlocks} implementation JSDoc block(s) ` +
    `from ${changedFiles} of ${files.length} governed JavaScript modules`
  );
  process.exit(0);
}

if (violations.length === 0) {
  console.log(
    `[source:style:check] ${files.length} governed JavaScript modules have sibling declarations, ` +
    `${sourceComputeCandidates.length} reviewed numeric candidate(s), no runtime tensor compute, ` +
    'no implementation JSDoc, no undeclared raw console calls, and no banned geometry inference'
  );
  process.exit(0);
}

console.error(`[source:style:check] ${violations.length} source style violation(s):`);
for (const violation of violations) {
  console.error(`  ${violation.file}: [${violation.invariant}] ${violation.detail}`);
}
process.exit(1);
