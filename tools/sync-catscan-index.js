#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'catscan-policy.json');
const COMPONENT_ID_PATTERN = /^doppler(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)*$/;
const REQUIRED_METADATA = Object.freeze(['Component', 'Parent']);
const REQUIRED_SECTIONS = Object.freeze([
  'Target',
  'Authority',
  'Scope',
  'Contracts',
  'Invariants',
  'Acceptance',
  'Non-goals',
  'Freedom',
]);

function toRepoPath(value) {
  return value.replaceAll(path.sep, '/');
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isRepoRelativePath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.posix.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
  );
}

function validatePolicy(policy) {
  const errors = [];
  if (!isPlainObject(policy)) return ['CATSCAN policy must be an object'];
  if (policy.schemaVersion !== 1) errors.push('CATSCAN policy schemaVersion must be 1');
  if (policy.source !== 'doppler') errors.push('CATSCAN policy source must be "doppler"');
  if (policy.charterFilename !== 'CATSCAN.md') errors.push('CATSCAN policy charterFilename must be "CATSCAN.md"');
  if (!isRepoRelativePath(policy.indexPath)) errors.push('CATSCAN policy indexPath must be repo-relative');
  if (!Number.isInteger(policy.maxWords) || policy.maxWords < 100 || policy.maxWords > 1000) {
    errors.push('CATSCAN policy maxWords must be an integer from 100 through 1000');
  }
  if (!Array.isArray(policy.requiredMetadata) || REQUIRED_METADATA.some((field) => !policy.requiredMetadata.includes(field))) {
    errors.push(`CATSCAN policy requiredMetadata must include ${REQUIRED_METADATA.join(', ')}`);
  }
  if (!Array.isArray(policy.requiredSections) || REQUIRED_SECTIONS.some((field) => !policy.requiredSections.includes(field))) {
    errors.push(`CATSCAN policy requiredSections must include ${REQUIRED_SECTIONS.join(', ')}`);
  }
  if (!normalizeText(policy.freedomText)) errors.push('CATSCAN policy freedomText is required');
  if (!Array.isArray(policy.ignoredDirectories) || policy.ignoredDirectories.some((entry) => !normalizeText(entry))) {
    errors.push('CATSCAN policy ignoredDirectories must be an array of names');
  }
  if (!Array.isArray(policy.requiredCharterPaths) || policy.requiredCharterPaths.length === 0) {
    errors.push('CATSCAN policy requiredCharterPaths must be a non-empty array');
  } else {
    const seenPaths = new Set();
    for (const charterPath of policy.requiredCharterPaths) {
      if (!isRepoRelativePath(charterPath) || path.posix.basename(charterPath) !== policy.charterFilename) {
        errors.push(`CATSCAN policy has invalid charter path: ${charterPath}`);
      }
      if (seenPaths.has(charterPath)) errors.push(`CATSCAN policy repeats charter path: ${charterPath}`);
      seenPaths.add(charterPath);
    }
  }
  return errors;
}

async function collectCharterPaths(repoRoot, policy) {
  const ignored = new Set(policy.ignoredDirectories);
  const charterPaths = [];

  async function visit(directory, relativeDirectory = '') {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      if (entry.isSymbolicLink()) continue;
      const relativePath = relativeDirectory
        ? path.posix.join(relativeDirectory, entry.name)
        : entry.name;
      if (entry.isDirectory()) {
        if (!ignored.has(entry.name)) {
          await visit(path.join(directory, entry.name), relativePath);
        }
        continue;
      }
      if (entry.isFile() && entry.name === policy.charterFilename) {
        charterPaths.push(relativePath);
      }
    }
  }

  await visit(repoRoot);
  return charterPaths.sort();
}

function parseMetadata(source, field) {
  const match = source.match(new RegExp(`^${field}:\\s*(.+?)\\s*$`, 'm'));
  return match ? match[1].trim() : null;
}

export function parseCatscanCharter(source, charterPath) {
  const errors = [];
  const lines = source.split(/\r?\n/);
  const headingMatch = lines[0]?.match(/^# CATSCAN: (.+)$/);
  if (!headingMatch) errors.push(`${charterPath}: first line must be "# CATSCAN: <Component>"`);

  const firstSectionIndex = lines.findIndex((line) => line.startsWith('## '));
  const metadataSource = lines.slice(0, firstSectionIndex < 0 ? lines.length : firstSectionIndex).join('\n');
  const componentValue = parseMetadata(metadataSource, 'Component');
  const parentValue = parseMetadata(metadataSource, 'Parent');
  const componentMatch = componentValue?.match(/^`([^`]+)`$/);
  const componentId = componentMatch?.[1] ?? null;

  const sections = new Map();
  let activeSection = null;
  let activeLines = [];
  const flushSection = () => {
    if (activeSection == null) return;
    if (sections.has(activeSection)) {
      errors.push(`${charterPath}: duplicate section "${activeSection}"`);
    } else {
      sections.set(activeSection, activeLines.join('\n').trim());
    }
  };
  for (const line of lines.slice(firstSectionIndex < 0 ? lines.length : firstSectionIndex)) {
    const sectionMatch = line.match(/^## (.+)$/);
    if (sectionMatch) {
      flushSection();
      activeSection = sectionMatch[1].trim();
      activeLines = [];
    } else if (activeSection != null) {
      activeLines.push(line);
    }
  }
  flushSection();

  return {
    path: charterPath,
    name: headingMatch?.[1]?.trim() ?? null,
    componentId,
    componentValue,
    parentValue,
    target: sections.get('Target') ?? null,
    sections,
    source,
    errors,
  };
}

function markdownLinks(source) {
  const links = [];
  const pattern = /(?<!!)\[[^\]]+\]\(([^)]+)\)/g;
  for (const match of source.matchAll(pattern)) links.push(match[1].trim());
  return links;
}

function localLinkTarget(charterPath, rawTarget) {
  let target = rawTarget;
  if (target.startsWith('<') && target.endsWith('>')) target = target.slice(1, -1);
  if (/^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith('#')) return null;
  target = target.split('#', 1)[0].split('?', 1)[0];
  if (!target) return charterPath;
  let decoded = target;
  try {
    decoded = decodeURIComponent(target);
  } catch {
    return undefined;
  }
  if (path.posix.isAbsolute(decoded) || decoded.includes('\\')) return undefined;
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(charterPath), decoded));
  if (resolved === '..' || resolved.startsWith('../')) return undefined;
  return resolved;
}

function nearestAncestorCharter(charterPath, charterPathSet, charterFilename) {
  let directory = path.posix.dirname(charterPath);
  if (directory === '.') return null;
  for (;;) {
    directory = path.posix.dirname(directory);
    const candidate = directory === '.'
      ? charterFilename
      : path.posix.join(directory, charterFilename);
    if (charterPathSet.has(candidate)) return candidate;
    if (directory === '.') return null;
  }
}

function parseParentTarget(record) {
  if (record.parentValue === 'none') return null;
  const match = record.parentValue?.match(/^\[[^\]]+\]\(([^)]+)\)$/);
  if (!match) return undefined;
  return localLinkTarget(record.path, match[1]);
}

async function pathExists(repoRoot, repoRelativePath) {
  try {
    await fs.stat(path.join(repoRoot, repoRelativePath));
    return true;
  } catch {
    return false;
  }
}

async function validateCharter(record, context) {
  const { repoRoot, policy, charterPathSet } = context;
  const errors = [...record.errors];
  if (!record.componentValue) errors.push(`${record.path}: missing Component metadata`);
  else if (!record.componentId || !COMPONENT_ID_PATTERN.test(record.componentId)) {
    errors.push(`${record.path}: Component must be a backticked dot-delimited Doppler ID`);
  }
  if (!record.parentValue) errors.push(`${record.path}: missing Parent metadata`);
  if (record.target && record.target.includes('\n')) errors.push(`${record.path}: Target must be one paragraph`);

  for (const sectionName of policy.requiredSections) {
    if (!record.sections.has(sectionName)) {
      errors.push(`${record.path}: missing section "${sectionName}"`);
    } else if (!record.sections.get(sectionName)) {
      errors.push(`${record.path}: section "${sectionName}" must not be empty`);
    }
  }
  const authority = record.sections.get('Authority') ?? '';
  if (!/^- Owns\s+/m.test(authority)) errors.push(`${record.path}: Authority must declare an "Owns" bullet`);
  if (!/^- Does not own\s+/m.test(authority)) errors.push(`${record.path}: Authority must declare a "Does not own" bullet`);
  const acceptance = record.sections.get('Acceptance') ?? '';
  if (!/^- Evidence:\s*\[[^\]]+\]\([^)]+\)/m.test(acceptance)) {
    errors.push(`${record.path}: Acceptance must contain a linked "Evidence" bullet`);
  }
  if ((record.sections.get('Freedom') ?? '') !== policy.freedomText) {
    errors.push(`${record.path}: Freedom must match the policy text exactly`);
  }

  const words = record.source.trim().split(/\s+/).filter(Boolean).length;
  if (words > policy.maxWords) {
    errors.push(`${record.path}: ${words} words exceeds the ${policy.maxWords}-word limit`);
  }

  const expectedParent = nearestAncestorCharter(record.path, charterPathSet, policy.charterFilename);
  const declaredParent = parseParentTarget(record);
  if (record.path === policy.charterFilename) {
    if (record.parentValue !== 'none') errors.push(`${record.path}: repository root Parent must be none`);
  } else if (declaredParent === undefined) {
    errors.push(`${record.path}: Parent must be one relative markdown link`);
  } else if (declaredParent !== expectedParent) {
    errors.push(`${record.path}: Parent must resolve to nearest ancestor ${expectedParent ?? '(missing)'}`);
  }

  for (const rawTarget of markdownLinks(record.source)) {
    const resolvedTarget = localLinkTarget(record.path, rawTarget);
    if (resolvedTarget === null) continue;
    if (resolvedTarget === undefined) {
      errors.push(`${record.path}: invalid repository link target "${rawTarget}"`);
    } else if (!await pathExists(repoRoot, resolvedTarget)) {
      errors.push(`${record.path}: link target does not exist: ${resolvedTarget}`);
    }
  }
  return errors;
}

function escapeTableCell(value) {
  return String(value ?? '').replaceAll('|', '\\|').replaceAll('\n', ' ');
}

export function renderComponentIndex(records) {
  const recordsByPath = new Map(records.map((record) => [record.path, record]));
  const sorted = [...records].sort((left, right) => {
    const leftDepth = left.path.split('/').length;
    const rightDepth = right.path.split('/').length;
    return leftDepth - rightDepth || left.path.localeCompare(right.path);
  });
  const lines = [
    '# Doppler Component Index',
    '',
    'Generated from the repository\'s `CATSCAN.md` files.',
    'Run `npm run catscan:sync` after adding, removing, or changing a component charter.',
    '',
    `Components: ${sorted.length}`,
    '',
    '| Component | Target | Charter | Parent |',
    '| --- | --- | --- | --- |',
  ];
  for (const record of sorted) {
    const parentPath = parseParentTarget(record);
    const parentId = parentPath ? recordsByPath.get(parentPath)?.componentId ?? parentPath : 'none';
    const charterLink = `../${record.path}`;
    lines.push(
      `| \`${record.componentId}\` | ${escapeTableCell(record.target)} | ` +
      `[${record.path}](${charterLink}) | ${parentId === 'none' ? 'none' : `\`${parentId}\``} |`
    );
  }
  lines.push('');
  lines.push('Child charters narrow inherited authority. They do not replace or broaden their parent contracts.');
  lines.push('');
  return lines.join('\n');
}

export async function buildCatscanReport(options = {}) {
  const repoRoot = path.resolve(options.repoRoot ?? REPO_ROOT);
  const policyPath = path.resolve(options.policyPath ?? path.join(repoRoot, 'tools', 'policies', 'catscan-policy.json'));
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const errors = validatePolicy(policy);
  if (errors.length > 0) {
    return { ok: false, errors, records: [], renderedIndex: null, indexCurrent: false };
  }

  const charterPaths = await collectCharterPaths(repoRoot, policy);
  const charterPathSet = new Set(charterPaths);
  const expectedPaths = new Set(policy.requiredCharterPaths);
  for (const expectedPath of [...expectedPaths].sort()) {
    if (!charterPathSet.has(expectedPath)) errors.push(`CATSCAN inventory is missing required charter: ${expectedPath}`);
  }
  for (const actualPath of charterPaths) {
    if (!expectedPaths.has(actualPath)) errors.push(`CATSCAN inventory has undeclared charter: ${actualPath}`);
  }

  const records = [];
  for (const charterPath of charterPaths) {
    const source = await fs.readFile(path.join(repoRoot, charterPath), 'utf8');
    records.push(parseCatscanCharter(source, charterPath));
  }
  const seenIds = new Map();
  for (const record of records) {
    if (record.componentId && seenIds.has(record.componentId)) {
      errors.push(`${record.path}: duplicate Component ID ${record.componentId} also used by ${seenIds.get(record.componentId)}`);
    } else if (record.componentId) {
      seenIds.set(record.componentId, record.path);
    }
    errors.push(...await validateCharter(record, { repoRoot, policy, charterPathSet }));
  }

  const renderedIndex = renderComponentIndex(records);
  const indexPath = path.join(repoRoot, policy.indexPath);
  let indexCurrent = false;
  try {
    indexCurrent = await fs.readFile(indexPath, 'utf8') === renderedIndex;
  } catch {
    indexCurrent = false;
  }
  if (options.checkIndex !== false && !indexCurrent) {
    errors.push(`${policy.indexPath} is stale; run npm run catscan:sync`);
  }
  return {
    ok: errors.length === 0,
    errors,
    records,
    renderedIndex,
    indexCurrent,
    indexPath,
    policy,
  };
}

export function parseArgs(argv) {
  const args = { check: false, json: false };
  for (const token of argv) {
    if (token === '--check') args.check = true;
    else if (token === '--json') args.json = true;
    else throw new Error(`Unknown argument: ${token}`);
  }
  return args;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const report = await buildCatscanReport({
    repoRoot: REPO_ROOT,
    policyPath: DEFAULT_POLICY_PATH,
    checkIndex: args.check,
  });
  if (!report.ok) {
    if (args.json) console.log(JSON.stringify({ ok: false, errors: report.errors }, null, 2));
    else for (const error of report.errors) console.error(`catscan: ${error}`);
    process.exitCode = 1;
    return;
  }
  if (!args.check) {
    await fs.writeFile(report.indexPath, report.renderedIndex, 'utf8');
  }
  if (args.json) {
    console.log(JSON.stringify({
      ok: true,
      components: report.records.length,
      indexPath: toRepoPath(path.relative(REPO_ROOT, report.indexPath)),
      mode: args.check ? 'check' : 'sync',
    }, null, 2));
  } else {
    const action = args.check ? 'current' : 'wrote';
    console.log(`catscan: ${action} ${toRepoPath(path.relative(REPO_ROOT, report.indexPath))} (${report.records.length} components)`);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
