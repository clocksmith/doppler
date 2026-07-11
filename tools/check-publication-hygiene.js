#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, statSync } from 'node:fs';

const PRIVATE_HOME_PATTERNS = Object.freeze([
  {
    label: 'macOS user home',
    pattern: /\/Users\/(?!(?:<user>|runner|Shared)(?=\/|[\s"'`,:}\]]|$))[^/\s"'`]+/g,
  },
  {
    label: 'Linux user home',
    pattern: /\/home\/(?!(?:user|runner|x|web_user|chrome)(?=\/|[\s"'`,:}\]]|$))[^/\s"'`]+/g,
  },
  {
    label: 'Windows user home',
    pattern: /[A-Za-z]:\\Users\\(?!<user>(?:\\|$)|runner(?:\\|$)|Public(?:\\|$))[^\\\s"'`]+/g,
  },
]);

function listTrackedFiles() {
  return execFileSync('git', ['ls-files', '-z'], {
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  }).split('\0').filter(Boolean);
}

function scanFile(filePath) {
  if (!existsSync(filePath) || !statSync(filePath).isFile()) return [];
  const bytes = readFileSync(filePath);
  if (bytes.includes(0)) return [];

  const findings = [];
  const lines = bytes.toString('utf8').split('\n');
  for (let index = 0; index < lines.length; index++) {
    for (const { label, pattern } of PRIVATE_HOME_PATTERNS) {
      pattern.lastIndex = 0;
      if (pattern.test(lines[index])) {
        findings.push({ filePath, line: index + 1, label });
      }
    }
  }
  return findings;
}

function main() {
  const files = listTrackedFiles();
  const findings = files.flatMap(scanFile);
  if (findings.length > 0) {
    for (const finding of findings) {
      console.error(`${finding.filePath}:${finding.line}: ${finding.label}`);
    }
    console.error(`publication hygiene failed: ${findings.length} private path(s)`);
    process.exitCode = 1;
    return;
  }

  console.log(`publication hygiene passed: ${files.length} tracked files scanned`);
}

main();
