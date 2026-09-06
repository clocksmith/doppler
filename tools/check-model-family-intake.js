#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { listDirectoriesAtGitRef, resolvePolicyBaseRef } from './lib/policy-base.js';
import schema from './policies/model-family-authorization.schema.json' with { type: 'json' };

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CONVERSION_ROOT = 'src/config/conversion';
const AUTHORIZATION_ROOT = 'tools/policies/model-family-authorizations';

function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isRepoRelativePath(value) {
  return typeof value === 'string' && value.length > 0 && !path.isAbsolute(value)
    && !value.includes('\\') && !value.includes('\0')
    && value.split('/').every((part) => part && part !== '.' && part !== '..');
}

export function findNewModelFamilies(currentFamilies, baselineFamilies) {
  const baseline = new Set(baselineFamilies);
  return [...new Set(currentFamilies)].filter((family) => !baseline.has(family)).sort();
}

export function validateModelFamilyAuthorization(authorization, family) {
  const errors = [];
  if (!isObject(authorization)) return ['authorization must be an object'];
  for (const key of Object.keys(authorization)) {
    if (!Object.hasOwn(schema.properties, key)) errors.push(`authorization.${key} is not allowed`);
  }
  for (const key of schema.required) {
    const rule = schema.properties[key];
    const value = authorization[key];
    if (Object.hasOwn(rule, 'const') && value !== rule.const) {
      errors.push(`authorization.${key} must equal ${JSON.stringify(rule.const)}`);
    } else if (rule.type === 'string' && (typeof value !== 'string' || !value.trim()
      || (rule.pattern && !new RegExp(rule.pattern).test(value)))) {
      errors.push(`authorization.${key} must satisfy its declared string contract`);
    }
  }
  if (authorization.family !== family) errors.push(`authorization.family must equal "${family}"`);
  try {
    const source = new URL(authorization.sourceRepository);
    if (source.protocol !== 'https:' || source.username || source.password || source.hash || source.search) throw new Error();
  } catch { errors.push('authorization.sourceRepository must be a credential-free HTTPS repository URL'); }

  const configs = authorization.conversionConfigs;
  if (!Array.isArray(configs) || configs.length === 0) errors.push('authorization.conversionConfigs must be a non-empty array');
  const inputs = [...(Array.isArray(configs) ? configs : []), authorization.referenceTest, authorization.licenseEvidence];
  const paths = new Set();
  for (const input of inputs) {
    if (!isObject(input) || Object.keys(input).some((key) => !Object.hasOwn(schema.$defs.input.properties, key))
      || !isRepoRelativePath(input.path) || typeof input.digest !== 'string'
      || !new RegExp(schema.$defs.input.properties.digest.pattern).test(input.digest)) {
      errors.push('authorization inputs require repository-relative path and SHA-256 digest only');
      continue;
    }
    if (paths.has(input.path)) errors.push(`authorization duplicates input ${input.path}`);
    paths.add(input.path);
  }
  for (const config of Array.isArray(configs) ? configs : []) {
    if (typeof config?.path !== 'string' || !config.path.startsWith(`${CONVERSION_ROOT}/${family}/`) || !config.path.endsWith('.json')) {
      errors.push('authorization conversion config is outside the exact family');
    }
  }
  if (typeof authorization.referenceTest?.path !== 'string'
    || !authorization.referenceTest.path.startsWith('tests/') || !authorization.referenceTest.path.endsWith('.js')) {
    errors.push('authorization.referenceTest must name a repository JavaScript test');
  }
  return errors;
}

async function readContained(root, relativePath) {
  if (!isRepoRelativePath(relativePath)) throw new Error('invalid repository-relative path');
  const resolved = await fs.realpath(path.join(root, relativePath));
  if (!resolved.startsWith(`${root}${path.sep}`)) throw new Error('input symlink escapes repository');
  return fs.readFile(resolved);
}

async function listConfigPaths(root, relativePath) {
  const paths = [];
  for (const entry of await fs.readdir(path.join(root, relativePath), { withFileTypes: true })) {
    const next = `${relativePath}/${entry.name}`;
    if (entry.isSymbolicLink()) throw new Error(`conversion scope may not contain symlinks: ${next}`);
    if (entry.isDirectory()) paths.push(...await listConfigPaths(root, next));
    else if (entry.isFile() && entry.name.endsWith('.json')) paths.push(next);
  }
  return paths.sort();
}

export async function checkModelFamilyIntake(root, baseRef) {
  root = await fs.realpath(root);
  const currentFamilies = (await fs.readdir(path.join(root, CONVERSION_ROOT), { withFileTypes: true }))
    .filter((entry) => entry.isDirectory() || entry.isSymbolicLink()).map((entry) => entry.name).sort();
  const newFamilies = findNewModelFamilies(currentFamilies, listDirectoriesAtGitRef(root, baseRef, CONVERSION_ROOT));
  const errors = [];
  for (const family of newFamilies) {
    try {
      if ((await fs.lstat(path.join(root, CONVERSION_ROOT, family))).isSymbolicLink()) throw new Error('family directory may not be a symlink');
      const authorization = JSON.parse(await readContained(root, `${AUTHORIZATION_ROOT}/${family}.json`));
      const invalid = validateModelFamilyAuthorization(authorization, family);
      if (invalid.length) throw new Error(invalid.join('; '));
      const configs = await listConfigPaths(root, `${CONVERSION_ROOT}/${family}`);
      if (JSON.stringify(configs) !== JSON.stringify(authorization.conversionConfigs.map((input) => input.path).sort())) {
        throw new Error('conversion configs do not match the exact maintainer-approved scope');
      }
      for (const input of [...authorization.conversionConfigs, authorization.referenceTest, authorization.licenseEvidence]) {
        const bytes = await readContained(root, input.path);
        const digest = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
        if (digest !== input.digest) throw new Error(`approved input changed: ${input.path}`);
        if (bytes.length === 0) throw new Error(`approved input is empty: ${input.path}`);
      }
    } catch (error) { errors.push(`${family}: ${error.message}`); }
  }
  return { ok: errors.length === 0, baseRef, families: currentFamilies.length, newFamilies, errors };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const report = await checkModelFamilyIntake(repoRoot, resolvePolicyBaseRef(process.argv.slice(2)));
  if (!report.ok) {
    console.error(`model family intake check failed:\n${report.errors.map((error) => `- ${error}`).join('\n')}`);
    process.exitCode = 1;
  } else {
    console.log(`model family intake check passed: authority=maintainer, base=${report.baseRef}, families=${report.families}, newFamilies=${report.newFamilies.length}`);
  }
}
