#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const REQUIRED_SKILLS = Object.freeze([
  'doppler-debug',
  'doppler-bench',
  'doppler-perf-squeeze',
  'doppler-convert',
  'doppler-kernel-reviewer',
]);

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function verifySymlink(relativePath, expectedTarget, issues) {
  const absolutePath = path.join(ROOT_DIR, relativePath);
  if (!(await pathExists(absolutePath))) {
    issues.push(`missing path: ${relativePath}`);
    return;
  }

  const stat = await fs.lstat(absolutePath);
  if (!stat.isSymbolicLink()) {
    issues.push(`expected symlink: ${relativePath}`);
    return;
  }

  const resolvedTarget = await fs.readlink(absolutePath);
  if (resolvedTarget !== expectedTarget) {
    issues.push(
      `symlink target mismatch: ${relativePath} -> ${resolvedTarget} (expected ${expectedTarget})`
    );
  }
}

async function verifySkillFiles(issues) {
  for (const skillName of REQUIRED_SKILLS) {
    const filePath = path.join(ROOT_DIR, 'skills', skillName, 'SKILL.md');
    if (!(await pathExists(filePath))) {
      issues.push(`missing skill file: skills/${skillName}/SKILL.md`);
    }
  }
}

async function main() {
  const issues = [];

  await verifySymlink('CLAUDE.md', 'AGENTS.md', issues);
  await verifySymlink('GEMINI.md', 'AGENTS.md', issues);
  await verifySymlink('.claude/skills', '../skills', issues);
  await verifySymlink('.gemini/skills', '../skills', issues);
  await verifySkillFiles(issues);

  if (issues.length > 0) {
    console.error('agent parity check failed:');
    for (const issue of issues) {
      console.error(`- ${issue}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log('agent parity check passed');
  console.log('- instruction aliases: CLAUDE.md, GEMINI.md -> AGENTS.md');
  console.log('- skill aliases: .claude/skills, .gemini/skills -> skills/');
  console.log(`- skills verified: ${REQUIRED_SKILLS.join(', ')}`);
}

await main();
