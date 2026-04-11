#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PACKAGE_PATH = path.join(REPO_ROOT, 'package.json');
const OUTPUT_PATH = path.join(REPO_ROOT, 'docs', 'api', 'reference', 'exports.md');

const EXPORT_META = {
  '.': {
    audience: 'app authors',
    stability: 'preferred public',
    docPath: 'docs/api/root.md',
    notes: 'Primary application-facing surface. Prefer this over lower-level exports.',
  },
  './tooling': {
    audience: 'tool builders',
    stability: 'public advanced',
    docPath: 'docs/api/tooling.md',
    notes: 'Tooling and command-runner surface, not the main app-facing API.',
  },
  './loaders': {
    audience: 'advanced loader consumers',
    stability: 'public advanced',
    docPath: 'docs/api/loaders.md',
    notes: 'Explicit loader and manifest/bootstrap helpers.',
  },
  './orchestration': {
    audience: 'advanced runtime consumers',
    stability: 'public advanced',
    docPath: 'docs/api/orchestration.md',
    notes: 'Tokenizer, KV cache, router, adapter, and logit-merge orchestration helpers.',
  },
  './generation': {
    audience: 'advanced runtime consumers',
    stability: 'public advanced',
    docPath: 'docs/api/generation.md',
    notes: 'Lower-level text pipeline construction and pipeline types.',
  },
  './diffusion': {
    audience: 'advanced diffusion consumers',
    stability: 'public advanced',
    docPath: 'docs/api/diffusion.md',
    notes: 'Diffusion/image pipeline surface.',
  },
  './energy': {
    audience: 'advanced energy consumers',
    stability: 'public advanced',
    docPath: 'docs/api/energy.md',
    notes: 'Energy pipeline surface.',
  },
};

function normalizeExportMap(exportsField) {
  return Object.entries(exportsField || {}).map(([exportPath, value]) => {
    if (typeof value === 'string') {
      return {
        exportPath,
        importFile: value,
        typesFile: '',
      };
    }
    return {
      exportPath,
      importFile: typeof value?.import === 'string' ? value.import : '',
      typesFile: typeof value?.types === 'string' ? value.types : '',
    };
  });
}

function relativeFromRepo(filePath) {
  return filePath.replace(/^\.\/+/, '');
}

function toDocLink(targetPath) {
  const absoluteTarget = path.join(REPO_ROOT, relativeFromRepo(targetPath));
  return path.relative(path.dirname(OUTPUT_PATH), absoluteTarget).replace(/\\/g, '/');
}

function extractNamedExports(source) {
  const names = new Set();
  const directPatterns = [
    /export\s+declare\s+function\s+([A-Za-z0-9_$]+)/g,
    /export\s+declare\s+const\s+([A-Za-z0-9_$]+)/g,
    /export\s+declare\s+class\s+([A-Za-z0-9_$]+)/g,
    /export\s+interface\s+([A-Za-z0-9_$]+)/g,
    /export\s+type\s+([A-Za-z0-9_$]+)/g,
    /export\s+declare\s+namespace\s+([A-Za-z0-9_$]+)/g,
  ];
  for (const pattern of directPatterns) {
    for (const match of source.matchAll(pattern)) {
      names.add(match[1]);
    }
  }

  const blockPatterns = [
    /export\s+\{([\s\S]*?)\}\s+from\s+['"][^'"]+['"]/g,
    /export\s+type\s+\{([\s\S]*?)\}\s+from\s+['"][^'"]+['"]/g,
  ];
  for (const pattern of blockPatterns) {
    for (const match of source.matchAll(pattern)) {
      const cleanedBlock = match[1]
        .split('\n')
        .map((line) => line.replace(/\/\/.*$/g, '').trim())
        .filter(Boolean)
        .join(' ');
      const raw = cleanedBlock
        .split(',')
        .map((part) => part.trim())
        .filter(Boolean);
      for (const part of raw) {
        const normalized = part.replace(/^type\s+/, '').trim();
        if (!normalized) continue;
        if (normalized.includes(' as ')) {
          names.add(normalized.split(/\s+as\s+/)[1].trim());
          continue;
        }
        names.add(normalized);
      }
    }
  }

  if (/export\s+\*\s+from\s+['"][^'"]+['"]/.test(source)) {
    names.add('*');
  }

  return [...names].sort((a, b) => a.localeCompare(b));
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

async function buildReferenceMarkdown() {
  const packageJson = await readJson(PACKAGE_PATH);
  const exportsList = normalizeExportMap(packageJson.exports);
  const lines = [];
  lines.push('# Public Export Inventory');
  lines.push('');
  lines.push('Auto-generated from `package.json` exports and shipped `.d.ts` entrypoints.');
  lines.push('This is a reference inventory, not the behavior guide. Manual API guides live one level up in `docs/api/`.');

  for (const entry of exportsList) {
    const meta = EXPORT_META[entry.exportPath] || {
      audience: 'unspecified',
      stability: 'unspecified',
      docPath: '',
      notes: 'No manual classification recorded for this export path.',
    };
    const importPath = entry.exportPath === '.'
      ? 'doppler-gpu'
      : `doppler-gpu/${entry.exportPath.replace(/^\.\//, '')}`;
    let symbols = [];
    if (entry.typesFile) {
      const absTypesPath = path.join(REPO_ROOT, relativeFromRepo(entry.typesFile));
      const source = await fs.readFile(absTypesPath, 'utf8');
      symbols = extractNamedExports(source);
    }

    lines.push('');
    lines.push(`## \`${importPath}\``);
    lines.push('');
    lines.push(`- Audience: ${meta.audience}`);
    lines.push(`- Stability: ${meta.stability}`);
    if (meta.docPath) {
      lines.push(`- Manual guide: [${meta.docPath}](${toDocLink(meta.docPath)})`);
    }
    if (entry.typesFile) {
      lines.push(`- Types: [${relativeFromRepo(entry.typesFile)}](${toDocLink(entry.typesFile)})`);
    }
    if (entry.importFile) {
      lines.push(`- Implementation: [${relativeFromRepo(entry.importFile)}](${toDocLink(entry.importFile)})`);
    }
    lines.push(`- Notes: ${meta.notes}`);

    if (symbols.length > 0) {
      lines.push('- Exported symbols:');
      for (const symbol of symbols) {
        lines.push(`  - \`${symbol}\``);
      }
    }
  }

  return `${lines.join('\n')}\n`;
}

async function readFileIfExists(filePath) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch {}
  return '';
}

async function writeIfChanged(filePath, nextContent) {
  const current = await readFileIfExists(filePath);
  if (current === nextContent) {
    return false;
  }
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, nextContent, 'utf8');
  return true;
}

function parseArgs(argv) {
  return {
    check: argv.includes('--check'),
  };
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const nextContent = await buildReferenceMarkdown();
  const current = await readFileIfExists(OUTPUT_PATH);
  const changed = current !== nextContent;

  if (args.check) {
    if (changed) {
      throw new Error(`API export inventory is out of date: ${path.relative(REPO_ROOT, OUTPUT_PATH)}`);
    }
    console.log(`[api-docs] up to date (${path.relative(REPO_ROOT, OUTPUT_PATH)})`);
    return;
  }

  await writeIfChanged(OUTPUT_PATH, nextContent);
  console.log(`[api-docs] wrote ${path.relative(REPO_ROOT, OUTPUT_PATH)}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`[api-docs] ${error.message}`);
    process.exit(1);
  });
}
