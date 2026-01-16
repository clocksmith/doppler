import { promises as fs } from 'node:fs';
import path from 'node:path';

const KERNEL_DIR = path.join(process.cwd(), 'src', 'gpu', 'kernels');


const WORKGROUP_ARRAY_OVERRIDE_WHITELIST = new Set([
  // Add legacy exceptions here (relative to src/gpu/kernels)
]);


function stripComments(source) {
  const withoutBlock = source.replace(/\/\*[\s\S]*?\*\//g, '');
  return withoutBlock.replace(/\/\/.*$/gm, '');
}


function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}


function findArrayLengths(statement) {
  
  const lengths = [];
  for (let i = 0; i < statement.length; i += 1) {
    if (!statement.startsWith('array<', i)) {
      continue;
    }
    const start = i + 'array<'.length;
    let depth = 1;
    let j = start;
    for (; j < statement.length; j += 1) {
      const ch = statement[j];
      if (ch === '<') depth += 1;
      if (ch === '>') {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    if (depth !== 0) {
      continue;
    }
    const inner = statement.slice(start, j);
    let commaIndex = -1;
    let innerDepth = 0;
    for (let k = 0; k < inner.length; k += 1) {
      const ch = inner[k];
      if (ch === '<') innerDepth += 1;
      if (ch === '>') innerDepth -= 1;
      if (ch === ',' && innerDepth === 0) {
        commaIndex = k;
      }
    }
    if (commaIndex !== -1) {
      lengths.push(inner.slice(commaIndex + 1).trim());
    }
    i = j;
  }
  return lengths;
}


async function main() {
  const files = await fs.readdir(KERNEL_DIR);
  const wgslFiles = files.filter((file) => file.endsWith('.wgsl'));

  
  const violations = [];

  for (const file of wgslFiles) {
    if (WORKGROUP_ARRAY_OVERRIDE_WHITELIST.has(file)) {
      continue;
    }
    const filePath = path.join(KERNEL_DIR, file);
    const raw = await fs.readFile(filePath, 'utf8');
    const source = stripComments(raw);

    const overrideMatches = Array.from(source.matchAll(/\boverride\s+([A-Za-z_][A-Za-z0-9_]*)\b/g));
    const overrides = overrideMatches.map((match) => match[1]);
    if (overrides.length === 0) {
      continue;
    }

    const overrideRegexes = overrides.map((name) => new RegExp(`\\b${escapeRegExp(name)}\\b`));
    const statements = source.split(';');
    for (const statement of statements) {
      if (!statement.includes('var<workgroup>') || !statement.includes('array<')) {
        continue;
      }
      const lengths = findArrayLengths(statement);
      for (const lengthExpr of lengths) {
        for (const regex of overrideRegexes) {
          if (regex.test(lengthExpr)) {
            violations.push(`${file}: workgroup array length uses override (${lengthExpr})`);
            break;
          }
        }
      }
    }
  }

  if (violations.length > 0) {
    console.error('WGSL override-array lint failed:');
    for (const issue of violations) {
      console.error(`  - ${issue}`);
    }
    process.exit(1);
  }

  console.log('WGSL override-array lint OK.');
}

main().catch((error) => {
  console.error('WGSL override-array lint failed:', error instanceof Error ? error.message : error);
  process.exit(1);
});
