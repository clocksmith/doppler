#!/usr/bin/env node
// Lists the snake_case string values Doppler's packaged source can emit: values
// written into result fields, receipts, stats, statuses and reasons, members of
// exported vocabularies, and the code prefix of error messages. Reads every
// packaged `src/**/*.js` file with the TypeScript parser (no execution).
// Output: a JSON report grouped by value with the field and file of each site.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const SNAKE = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/;
const CODE_PREFIX = /^([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?::|\s|$)/;

function propName(node) {
  if (!node) return null;
  if (ts.isIdentifier(node) || ts.isPrivateIdentifier(node)) return node.text;
  if (ts.isStringLiteral(node) || ts.isNumericLiteral(node)) return node.text;
  if (ts.isPropertyAccessExpression(node)) return node.name.text;
  if (ts.isElementAccessExpression(node)) {
    const arg = node.argumentExpression;
    return arg && ts.isStringLiteral(arg) ? arg.text : null;
  }
  return null;
}

function calleeName(expr) {
  if (ts.isIdentifier(expr)) return expr.text;
  if (ts.isPropertyAccessExpression(expr)) return expr.name.text;
  return null;
}

// Walk up through value-preserving wrappers to the context that receives the value.
function classify(node) {
  let child = node;
  let parent = node.parent;
  while (parent) {
    if (ts.isParenthesizedExpression(parent) || ts.isAsExpression(parent) || ts.isSatisfiesExpression?.(parent)
      || ts.isConditionalExpression(parent) && parent.condition !== child
      || ts.isBinaryExpression(parent) && (parent.operatorToken.kind === ts.SyntaxKind.QuestionQuestionToken
        || parent.operatorToken.kind === ts.SyntaxKind.BarBarToken) && parent.right === child
      // A code that starts a concatenated message ('code: ' + detail) is the value of the whole expression.
      || ts.isBinaryExpression(parent) && parent.operatorToken.kind === ts.SyntaxKind.PlusToken && parent.left === child
      || ts.isArrayLiteralExpression(parent)
      || ts.isSpreadElement(parent)) {
      child = parent;
      parent = parent.parent;
      continue;
    }
    if (ts.isCallExpression(parent) && calleeName(parent.expression) === 'freeze') {
      child = parent; parent = parent.parent; continue;
    }
    if (ts.isNewExpression(parent) && parent.expression && calleeName(parent.expression) === 'Set') {
      child = parent; parent = parent.parent; continue;
    }
    break;
  }
  if (!parent) return null;
  if (ts.isPropertyAssignment(parent) && parent.initializer === child) return { site: 'property', field: propName(parent.name) };
  if (ts.isShorthandPropertyAssignment(parent)) return null;
  if (ts.isBinaryExpression(parent)) {
    const op = parent.operatorToken.kind;
    if (op === ts.SyntaxKind.EqualsToken || op === ts.SyntaxKind.QuestionQuestionEqualsToken || op === ts.SyntaxKind.BarBarEqualsToken) {
      return parent.right === child ? { site: 'assign', field: propName(parent.left) } : null;
    }
    if ([ts.SyntaxKind.EqualsEqualsEqualsToken, ts.SyntaxKind.ExclamationEqualsEqualsToken,
      ts.SyntaxKind.EqualsEqualsToken, ts.SyntaxKind.ExclamationEqualsToken].includes(op)) {
      const other = parent.left === child ? parent.right : parent.left;
      return { site: 'compare', field: propName(other) ?? (ts.isIdentifier(other) ? other.text : null) };
    }
    return null;
  }
  if (ts.isVariableDeclaration(parent) && parent.initializer === child) return { site: 'variable', field: ts.isIdentifier(parent.name) ? parent.name.text : null };
  if (ts.isReturnStatement(parent) || ts.isArrowFunction(parent)) return { site: 'return', field: null };
  if (ts.isCaseClause(parent)) return { site: 'compare', field: '(case)' };
  if (ts.isParameter(parent) || ts.isBindingElement(parent)) return { site: 'default', field: propName(parent.name) };
  if (ts.isCallExpression(parent) || ts.isNewExpression(parent)) {
    const name = calleeName(parent.expression);
    return { site: 'call', field: name };
  }
  if (ts.isElementAccessExpression(parent) || ts.isPropertyName?.(child) || ts.isImportDeclaration(parent)
    || ts.isExportDeclaration(parent) || ts.isLiteralTypeNode(parent)) return null;
  return null;
}

export function listEmittedSnakeValues({ root }) {
  const files = execFileSync('git', ['-C', root, 'ls-files', '--', 'src'], { maxBuffer: 1 << 28 })
    .toString().split('\n').filter((f) => f.endsWith('.js') && !f.endsWith('.d.js'));
  const values = new Map();
  const record = (value, site, file) => {
    if (!values.has(value)) values.set(value, { value, sites: [] });
    values.get(value).sites.push({ file, ...site });
  };
  for (const file of files) {
    const absolute = path.join(root, file);
    if (!fs.existsSync(absolute)) continue;
    const text = fs.readFileSync(absolute, 'utf8');
    const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
    const visit = (node) => {
      if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node) || ts.isTemplateHead(node)) {
        const raw = node.text;
        if (SNAKE.test(raw)) {
          const site = classify(ts.isTemplateHead(node) ? node.parent : node);
          if (site) record(raw, site, file);
        } else {
          const prefix = CODE_PREFIX.exec(raw);
          const host = ts.isTemplateHead(node) ? node.parent : node;
          const call = host.parent;
          if (prefix && call && (ts.isNewExpression(call) || ts.isCallExpression(call)) && /Error$/.test(calleeName(call.expression) ?? '')) {
            record(prefix[1], { site: 'error-message-code', field: calleeName(call.expression) }, file);
          } else if (prefix && raw[prefix[1].length] === ':') {
            // A code with a detail suffix ('missing_check:' + id): classify it like the bare value.
            const site = classify(host);
            if (site) record(prefix[1], site, file);
          }
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(source);
  }
  return [...values.values()].sort((a, b) => a.value.localeCompare(b.value));
}

function siteField(site) {
  if (site.site === 'error-message-code') return '<error-message-code>';
  if (site.site === 'call') return `call:${site.field}`;
  return site.field;
}

// Classifies the discovered values against tools/policies/emitted-enum-values.json.
// retained: values at outcome fields that belong to a retained identifier category.
// unclassified: any other snake_case value at an outcome field; there must be none.
export function classifyEmittedSnakeValues({ root, policy }) {
  const outcome = new Set(policy.outcomeFields);
  const retainedBy = new Map();
  for (const [category, entry] of Object.entries(policy.retainedValues)) {
    for (const value of entry.values) retainedBy.set(value, category);
  }
  const retained = [];
  const unclassified = [];
  for (const entry of listEmittedSnakeValues({ root })) {
    const sites = entry.sites.filter((site) => site.site !== 'compare' && outcome.has(siteField(site)));
    if (sites.length === 0) continue;
    const fields = [...new Set(sites.map(siteField))].sort();
    const files = [...new Set(sites.map((site) => site.file))].sort();
    if (retainedBy.has(entry.value)) retained.push({ value: entry.value, category: retainedBy.get(entry.value), fields, files });
    else unclassified.push({ value: entry.value, kebab: entry.value.replace(/_/g, '-'), fields, files });
  }
  return { retained, unclassified };
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  if (process.argv.includes('--raw')) {
    process.stdout.write(`${JSON.stringify(listEmittedSnakeValues({ root }), null, 1)}\n`);
  } else {
    const policy = JSON.parse(fs.readFileSync(path.join(root, 'tools/policies/emitted-enum-values.json'), 'utf8'));
    process.stdout.write(`${JSON.stringify(classifyEmittedSnakeValues({ root, policy }), null, 1)}\n`);
  }
}
