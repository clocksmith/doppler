import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WGSL_GENERATED_VARIANTS } from './configs/wgsl-variants.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_ROOT = path.resolve(__dirname, '..');

function ensureTrailingNewline(text) {
  return text.endsWith('\n') ? text : `${text}\n`;
}

function countLiteralMatches(text, needle) {
  if (needle.length === 0) {
    return 0;
  }
  let count = 0;
  let index = 0;
  while (true) {
    const found = text.indexOf(needle, index);
    if (found === -1) {
      return count;
    }
    count += 1;
    index = found + needle.length;
  }
}

function replaceLiteral(text, rule, variantId) {
  const from = String(rule.from ?? '');
  const to = String(rule.to ?? '');
  const expectedCount = Number.isInteger(rule.count) ? rule.count : null;
  const matches = countLiteralMatches(text, from);
  if (expectedCount != null && matches !== expectedCount) {
    throw new Error(
      `variant "${variantId}" literal replacement expected ${expectedCount} match(es), found ${matches}: ${JSON.stringify(from.slice(0, 64))}`
    );
  }
  if (matches === 0) {
    throw new Error(
      `variant "${variantId}" literal replacement found no matches: ${JSON.stringify(from.slice(0, 64))}`
    );
  }
  return text.split(from).join(to);
}

function replaceRegex(text, rule, variantId) {
  const pattern = String(rule.pattern ?? '');
  if (!pattern) {
    throw new Error(`variant "${variantId}" regex replacement missing pattern`);
  }
  const flags = String(rule.flags ?? 'g');
  const expectedCount = Number.isInteger(rule.count) ? rule.count : null;
  const expression = new RegExp(pattern, flags);
  const matches = [...text.matchAll(expression)].length;
  if (expectedCount != null && matches !== expectedCount) {
    throw new Error(
      `variant "${variantId}" regex replacement expected ${expectedCount} match(es), found ${matches}: /${pattern}/${flags}`
    );
  }
  if (matches === 0) {
    throw new Error(`variant "${variantId}" regex replacement found no matches: /${pattern}/${flags}`);
  }
  return text.replace(expression, String(rule.to ?? ''));
}

function applyRule(text, rule, variantId) {
  const type = String(rule.type ?? 'literal');
  if (type === 'literal') {
    return replaceLiteral(text, rule, variantId);
  }
  if (type === 'regex') {
    return replaceRegex(text, rule, variantId);
  }
  throw new Error(`variant "${variantId}" has unsupported rule type "${type}"`);
}

function parseUnifiedHunks(patchText, variantId, patchPath) {
  const lines = String(patchText).replace(/\r\n/g, '\n').split('\n');
  const hunks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.startsWith('@@ ')) {
      index += 1;
      continue;
    }
    index += 1;
    const body = [];
    while (index < lines.length) {
      const bodyLine = lines[index];
      if (bodyLine.startsWith('@@ ')) {
        break;
      }
      if (bodyLine === '' && index === lines.length - 1) {
        index += 1;
        continue;
      }
      if (bodyLine.startsWith('\\ No newline at end of file')) {
        index += 1;
        continue;
      }
      const prefix = bodyLine[0];
      if (!(prefix === ' ' || prefix === '+' || prefix === '-')) {
        throw new Error(
          `variant "${variantId}" patch has invalid hunk line in ${patchPath}: ${JSON.stringify(bodyLine)}`
        );
      }
      body.push(bodyLine);
      index += 1;
    }
    hunks.push(body);
  }
  return hunks;
}

function findLineSequence(lines, sequence, startIndex) {
  if (sequence.length === 0) {
    return startIndex;
  }
  for (let index = startIndex; index <= lines.length - sequence.length; index += 1) {
    let matches = true;
    for (let offset = 0; offset < sequence.length; offset += 1) {
      if (lines[index + offset] !== sequence[offset]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return index;
    }
  }
  return -1;
}

function applyUnifiedPatch(sourceText, patchText, variantId, patchPath) {
  const hunks = parseUnifiedHunks(patchText, variantId, patchPath);
  if (hunks.length === 0) {
    throw new Error(`variant "${variantId}" patch has no hunks: ${patchPath}`);
  }

  const sourceLines = String(sourceText).replace(/\r\n/g, '\n').split('\n');
  if (sourceLines[sourceLines.length - 1] === '') {
    sourceLines.pop();
  }
  let cursor = 0;

  for (const body of hunks) {
    const oldSegment = [];
    const newSegment = [];
    for (const line of body) {
      const prefix = line[0];
      const content = line.slice(1);
      if (prefix === ' ' || prefix === '-') {
        oldSegment.push(content);
      }
      if (prefix === ' ' || prefix === '+') {
        newSegment.push(content);
      }
    }

    const start = findLineSequence(sourceLines, oldSegment, cursor);
    if (start === -1) {
      throw new Error(
        `variant "${variantId}" patch hunk not found in source: ${patchPath}`
      );
    }
    sourceLines.splice(start, oldSegment.length, ...newSegment);
    cursor = start + newSegment.length;
  }

  return sourceLines.join('\n');
}

function buildGeneratedText(variant, transformedSource) {
  const source = String(variant.source);
  const header = [
    `// AUTO-GENERATED from ${source}.`,
    '// Edit the source kernel and tools/configs/wgsl-variants.js, then run `npm run kernels:generate`.',
    '',
  ].join('\n');
  return ensureTrailingNewline(`${header}${transformedSource}`);
}

async function readUtf8(filePath) {
  return fs.readFile(filePath, 'utf8');
}

async function buildVariantContent(rootDir, variant) {
  const sourcePath = path.join(rootDir, String(variant.source));
  const sourceText = await readUtf8(sourcePath);
  const variantId = String(variant.id);
  const hasPatch = typeof variant.patch === 'string' && variant.patch.trim().length > 0;
  const hasRules = Array.isArray(variant.rules) && variant.rules.length > 0;
  if (hasPatch && hasRules) {
    throw new Error(`variant "${variantId}" cannot define both patch and rules`);
  }

  let transformed = sourceText;
  if (hasPatch) {
    const patchPath = path.join(rootDir, String(variant.patch));
    const patchText = await readUtf8(patchPath);
    transformed = applyUnifiedPatch(sourceText, patchText, variantId, patchPath);
  } else {
    const rules = Array.isArray(variant.rules) ? variant.rules : [];
    for (const rule of rules) {
      transformed = applyRule(transformed, rule, variantId);
    }
  }
  return buildGeneratedText(variant, transformed);
}

async function readExistingTarget(rootDir, variant) {
  const targetPath = path.join(rootDir, String(variant.target));
  try {
    return await readUtf8(targetPath);
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

export async function generateWgslVariants(options = {}) {
  const rootDir = path.resolve(options.rootDir ?? DEFAULT_ROOT);
  const checkOnly = options.checkOnly === true;
  const pending = [];
  const errors = [];

  for (const variant of WGSL_GENERATED_VARIANTS) {
    const id = String(variant.id ?? '');
    const source = String(variant.source ?? '');
    const target = String(variant.target ?? '');
    if (!id || !source || !target) {
      errors.push(`invalid variant entry: ${JSON.stringify(variant)}`);
      continue;
    }

    try {
      const generated = await buildVariantContent(rootDir, variant);
      const existing = await readExistingTarget(rootDir, variant);
      const changed = existing !== generated;
      const targetPath = path.join(rootDir, target);
      pending.push({ id, source, target, generated, changed, targetPath });
    } catch (error) {
      errors.push(`variant "${id}" failed: ${error.message}`);
    }
  }

  if (errors.length === 0 && !checkOnly) {
    for (const item of pending) {
      if (item.changed) {
        // eslint-disable-next-line no-await-in-loop
        await fs.writeFile(item.targetPath, item.generated, 'utf8');
      }
    }
  }

  const results = pending.map(({ id, source, target, changed }) => ({ id, source, target, changed }));
  return {
    rootDir,
    checkOnly,
    variantCount: WGSL_GENERATED_VARIANTS.length,
    changedCount: results.filter((item) => item.changed).length,
    unchangedCount: results.filter((item) => !item.changed).length,
    changedTargets: results.filter((item) => item.changed).map((item) => item.target),
    results,
    errors,
  };
}
