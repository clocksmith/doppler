#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { validateProductionRelease } from '../src/config/production-release.js';
import { listDirectoriesAtGitRef, resolvePolicyBaseRef } from './lib/policy-base.js';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const conversionRoot = path.join(repoRoot, 'src/config/conversion');
const goalMatrixPath = path.join(repoRoot, 'src/config/goal-completion-matrix.json');
const authorizationRoot = path.join(repoRoot, 'tools/policies/model-family-authorizations');
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/u;
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/u;
const AUTHORIZATION_KEYS = new Set([
  'schema',
  'family',
  'authority',
  'customerId',
  'applicationId',
  'releaseContractPath',
  'authorizationDigest',
]);

function isObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isRepoRelativePath(value) {
  return (
    typeof value === 'string'
    && value.length > 0
    && !path.isAbsolute(value)
    && !value.includes('\\')
    && !value.split('/').includes('..')
  );
}

export function findNewModelFamilies(currentFamilies, baselineFamilies) {
  const baseline = new Set(baselineFamilies);
  return [...new Set(currentFamilies)].filter((family) => !baseline.has(family)).sort();
}

export function validateModelFamilyAuthorization(
  authorization,
  family,
  release,
  releaseValidation,
  externalAuthorizationDigest = null
) {
  const errors = [];
  if (!isObject(authorization)) return ['authorization must be an object'];
  for (const key of Object.keys(authorization)) {
    if (!AUTHORIZATION_KEYS.has(key)) errors.push(`authorization.${key} is not allowed`);
  }
  if (authorization.schema !== 'doppler.model-family-authorization/v1') {
    errors.push('authorization.schema must be "doppler.model-family-authorization/v1"');
  }
  if (authorization.family !== family) errors.push(`authorization.family must equal "${family}"`);
  if (authorization.authority !== 'customer') errors.push('authorization.authority must be "customer"');
  for (const key of ['customerId', 'applicationId']) {
    if (!ID_PATTERN.test(authorization[key] ?? '')) {
      errors.push(`authorization.${key} must be a kebab-case identifier`);
    }
  }
  if (!isRepoRelativePath(authorization.releaseContractPath)) {
    errors.push('authorization.releaseContractPath must be a repository-relative path');
  }
  if (!SHA256_PATTERN.test(authorization.authorizationDigest ?? '')) {
    errors.push('authorization.authorizationDigest must be a SHA-256 digest');
  } else if (authorization.authorizationDigest !== externalAuthorizationDigest) {
    errors.push(
      'authorization.authorizationDigest must match DOPPLER_MODEL_FAMILY_AUTHORIZATION'
    );
  }
  if (!releaseValidation?.ok) {
    for (const error of releaseValidation?.errors ?? ['release contract is invalid']) {
      errors.push(`release contract: ${error}`);
    }
  }
  if (!isObject(release)) {
    errors.push('release contract must be an object');
  } else {
    if (!['external-candidate', 'external-production'].includes(release.evidenceClass)) {
      errors.push('release contract must declare external-candidate or external-production evidence');
    }
    if (release.claimBoundary?.externalCustomer !== true) {
      errors.push('release contract must bind an external customer');
    }
    if (release.application?.applicationId !== authorization.applicationId) {
      errors.push('release contract applicationId must match authorization.applicationId');
    }
    if (release.rollout?.activationAuthority !== 'customer') {
      errors.push('release contract activation authority must remain customer-controlled');
    }
  }
  return errors;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

async function main() {
  const baseRef = resolvePolicyBaseRef(process.argv.slice(2));
  const matrix = await readJson(goalMatrixPath);
  const goal = matrix.goals?.find((entry) => entry.id === 'local-webgpu-product-surface');
  if (!goal) throw new Error('Goal matrix is missing local-webgpu-product-surface.');
  const currentFamilies = (await fs.readdir(conversionRoot, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  const baselineFamilies = listDirectoriesAtGitRef(
    repoRoot,
    baseRef,
    'src/config/conversion'
  );
  const newFamilies = findNewModelFamilies(currentFamilies, baselineFamilies);
  const errors = [];

  if (goal.status !== 'complete') {
    for (const family of newFamilies) {
      const authorizationPath = path.join(authorizationRoot, `${family}.json`);
      let authorization;
      try {
        authorization = await readJson(authorizationPath);
      } catch (error) {
        errors.push(`${family}: missing readable customer authorization at ${path.relative(repoRoot, authorizationPath)} (${error.message})`);
        continue;
      }
      let release = null;
      let releaseValidation = { ok: false, errors: ['release contract path is invalid'] };
      if (isRepoRelativePath(authorization.releaseContractPath)) {
        try {
          release = await readJson(path.join(repoRoot, authorization.releaseContractPath));
          releaseValidation = validateProductionRelease(release);
        } catch (error) {
          releaseValidation = { ok: false, errors: [error.message] };
        }
      }
      for (const error of validateModelFamilyAuthorization(
        authorization,
        family,
        release,
        releaseValidation,
        process.env.DOPPLER_MODEL_FAMILY_AUTHORIZATION ?? null
      )) {
        errors.push(`${family}: ${error}`);
      }
    }
  }

  if (errors.length > 0) {
    console.error('model family intake check failed:');
    for (const error of errors) console.error(`- ${error}`);
    process.exitCode = 1;
    return;
  }
  console.log(
    `model family intake check passed: goal=${goal.status}, base=${baseRef}, `
    + `families=${currentFamilies.length}, newFamilies=${newFamilies.length}`
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
