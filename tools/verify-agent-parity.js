#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isObject } from './utils/policy-utils.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const POLICY_PATH = path.join(ROOT_DIR, 'tools', 'policies', 'agent-parity-policy.json');

export const REQUIRED_INSTRUCTION_ALIASES = Object.freeze([
  { path: 'CLAUDE.md', target: 'AGENTS.md' },
  { path: 'GEMINI.md', target: 'AGENTS.md' },
]);

export const REQUIRED_SKILL_ALIASES = Object.freeze([
  { path: '.claude/skills', target: '../skills' },
  { path: '.gemini/skills', target: '../skills' },
  { path: '.codex/skills', target: '../skills' },
]);

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch (error) {
    if (error.code === 'ENOENT') return false;
    throw error;
  }
}

function normalizeRuleArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty array`);
  }
  return value.map((item, index) => {
    if (!isObject(item)) {
      throw new Error(`${label}[${index}] must be an object`);
    }
    const rulePath = typeof item.path === 'string' ? item.path.trim() : '';
    const target = typeof item.target === 'string' ? item.target.trim() : '';
    if (!rulePath || !target) {
      throw new Error(`${label}[${index}] must include non-empty "path" and "target"`);
    }
    return { path: rulePath, target };
  });
}

function normalizeStringArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty array`);
  }
  return value.map((item, index) => {
    if (typeof item !== 'string' || item.trim() === '') {
      throw new Error(`${label}[${index}] must be a non-empty string`);
    }
    return item.trim();
  });
}

async function loadPolicy() {
  const raw = await fs.readFile(POLICY_PATH, 'utf-8');
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`invalid agent parity policy JSON: ${error.message}`);
  }
  if (!isObject(parsed)) {
    throw new Error('agent parity policy root must be an object');
  }
  if (!Number.isInteger(parsed.schemaVersion) || parsed.schemaVersion !== 1) {
    throw new Error('agent parity policy schemaVersion must be 1');
  }
  const instructionCanonicalPath = typeof parsed.instructionCanonicalPath === 'string'
    ? parsed.instructionCanonicalPath.trim()
    : '';
  if (!instructionCanonicalPath) {
    throw new Error('agent parity policy must define instructionCanonicalPath');
  }
  const policy = {
    instructionCanonicalPath,
    instructionAliases: normalizeRuleArray(parsed.instructionAliases, 'instructionAliases'),
    skillAliases: normalizeRuleArray(parsed.skillAliases, 'skillAliases'),
    requiredSkills: normalizeStringArray(parsed.requiredSkills, 'requiredSkills'),
    requiredAgentsMarkers: normalizeStringArray(parsed.requiredAgentsMarkers, 'requiredAgentsMarkers'),
  };
  assertRequiredAliases(policy);
  return policy;
}

function assertRequiredAliases(policy) {
  for (const alias of REQUIRED_INSTRUCTION_ALIASES) {
    const matched = policy.instructionAliases.some((entry) => entry.path === alias.path && entry.target === alias.target);
    if (!matched) {
      throw new Error(`agent parity policy missing required instruction alias ${alias.path} -> ${alias.target}`);
    }
  }
  for (const alias of REQUIRED_SKILL_ALIASES) {
    const matched = policy.skillAliases.some((entry) => entry.path === alias.path && entry.target === alias.target);
    if (!matched) {
      throw new Error(`agent parity policy missing required skill alias ${alias.path} -> ${alias.target}`);
    }
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

async function verifySkillFiles(requiredSkills, issues) {
  for (const skillName of requiredSkills) {
    const filePath = path.join(ROOT_DIR, 'skills', skillName, 'SKILL.md');
    if (!(await pathExists(filePath))) {
      issues.push(`missing skill file: skills/${skillName}/SKILL.md`);
    }
  }
}

async function verifyAgentsMarkers(policy, issues) {
  const agentsPath = path.join(ROOT_DIR, policy.instructionCanonicalPath);
  if (!(await pathExists(agentsPath))) {
    issues.push(`missing canonical instruction file: ${policy.instructionCanonicalPath}`);
    return;
  }
  const agentsText = await fs.readFile(agentsPath, 'utf-8');
  for (const marker of policy.requiredAgentsMarkers) {
    if (!agentsText.includes(marker)) {
      issues.push(
        `AGENTS marker missing: "${marker}" (expected in ${policy.instructionCanonicalPath})`
      );
    }
  }
}

async function main() {
  const policy = await loadPolicy();
  const issues = [];

  for (const alias of policy.instructionAliases) {
    await verifySymlink(alias.path, alias.target, issues);
  }
  for (const alias of policy.skillAliases) {
    await verifySymlink(alias.path, alias.target, issues);
  }
  await verifySkillFiles(policy.requiredSkills, issues);
  await verifyAgentsMarkers(policy, issues);

  if (issues.length > 0) {
    console.error('agent parity check failed:');
    for (const issue of issues) {
      console.error(`- ${issue}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log('agent parity check passed');
  console.log(`- policy: ${path.relative(ROOT_DIR, POLICY_PATH)}`);
  console.log(
    `- instruction aliases: ${policy.instructionAliases.map((rule) => `${rule.path} -> ${rule.target}`).join(', ')}`
  );
  console.log(
    `- skill aliases: ${policy.skillAliases.map((rule) => `${rule.path} -> ${rule.target}`).join(', ')}`
  );
  console.log(`- skills verified: ${policy.requiredSkills.join(', ')}`);
  console.log(`- AGENTS markers verified: ${policy.requiredAgentsMarkers.length}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
