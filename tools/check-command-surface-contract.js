#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isPlainObject } from '../src/formats/plain-object.js';
import { TOOLING_COMMANDS, TOOLING_SURFACES, ensureCommandSupportedOnSurface } from '../src/tooling/command-api.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'command-surface-contract.json');
const CLI_PATH = path.join(REPO_ROOT, 'src', 'cli', 'doppler-cli.js');

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function sameSet(left, right) {
  if (left.length !== right.length) return false;
  const rightSet = new Set(right);
  return left.every((value) => rightSet.has(value));
}

function validatePolicy(policy, errors) {
  if (!isPlainObject(policy)) {
    errors.push('command surface policy must be an object');
    return [];
  }
  if (policy.schemaVersion !== 1) {
    errors.push('command surface policy schemaVersion must be 1');
  }
  if (policy.source !== 'doppler') {
    errors.push('command surface policy source must be "doppler"');
  }
  if (!Array.isArray(policy.commands)) {
    errors.push('command surface policy commands must be an array');
    return [];
  }

  const commandIds = [];
  const seen = new Set();
  for (const command of policy.commands) {
    const id = normalizeText(command?.id);
    if (!id) {
      errors.push('command entry is missing id');
      continue;
    }
    if (seen.has(id)) {
      errors.push(`duplicate command surface entry ${id}`);
    }
    seen.add(id);
    commandIds.push(id);

    if (!Array.isArray(command.surfaces) || command.surfaces.length === 0) {
      errors.push(`${id}: surfaces must be a non-empty array`);
    } else {
      const surfaceSet = new Set();
      for (const surface of command.surfaces) {
        const normalizedSurface = normalizeText(surface);
        if (!TOOLING_SURFACES.includes(normalizedSurface)) {
          errors.push(`${id}: unsupported surface ${String(surface)}`);
        }
        if (surfaceSet.has(normalizedSurface)) {
          errors.push(`${id}: duplicate surface ${normalizedSurface}`);
        }
        surfaceSet.add(normalizedSurface);
      }
    }
    if (typeof command.cliUsageRequired !== 'boolean') {
      errors.push(`${id}: cliUsageRequired must be boolean`);
    }
    if (!normalizeText(command.reason)) {
      errors.push(`${id}: reason is required`);
    }
  }
  return commandIds;
}

function validateCommandSet(policyCommandIds, errors) {
  if (!sameSet(policyCommandIds, TOOLING_COMMANDS)) {
    errors.push(
      `command surface policy must match TOOLING_COMMANDS. policy=${policyCommandIds.sort().join(',')} code=${[...TOOLING_COMMANDS].sort().join(',')}`
    );
  }
}

function validateCliUsage(policy, cliSource, errors) {
  if (!Array.isArray(policy?.commands)) return;
  for (const command of policy.commands) {
    if (command.cliUsageRequired !== true) continue;
    const id = normalizeText(command.id);
    if (!cliSource.includes(`doppler ${id}`)) {
      errors.push(`${id}: CLI usage text must include "doppler ${id}"`);
    }
  }
}

const COMMAND_PROBES = {
  convert: { inputDir: '/contract/input', convertPayload: { converterConfig: {} } },
  'refresh-integrity': { modelDir: '/contract/model' },
  release: {
    action: 'decide', manifestPath: '/contract/release.json', outputDirectory: '/contract/output',
    capsuleTrustedSignersPath: '/contract/trust.json', signingPrivateKeyPath: '/contract/private.json',
    signingPublicKeyPath: '/contract/public.json', signingAuthority: 'contract-probe',
    fleetTrustedSignersPath: '/contract/fleet.json',
  },
  lora: { action: 'run', workloadPath: '/contract/workload.json' },
  distill: { action: 'run', workloadPath: '/contract/workload.json' },
  debug: { workload: 'inference', modelId: 'contract-probe' },
  bench: { workload: 'inference', modelId: 'contract-probe' },
  verify: { workload: 'inference', modelId: 'contract-probe' },
  diagnose: { workload: 'inference', modelId: 'contract-probe' },
};

// Probe normalization and admission only: no command is executed and no file is read.
export function validateCommandAdmission(policy, admit = ensureCommandSupportedOnSurface) {
  const errors = [];
  for (const command of policy.commands ?? []) {
    if (!Object.hasOwn(COMMAND_PROBES, command.id)) {
      errors.push(`${command.id}: add an admission probe for this command`);
      continue;
    }
    for (const surface of TOOLING_SURFACES) {
      const allowed = command.surfaces.includes(surface);
      try {
        const result = admit({ command: command.id, ...COMMAND_PROBES[command.id] }, surface);
        if (!allowed) errors.push(`${command.id}: ${surface} admitted a forbidden command`);
        if (result.request.command !== command.id || result.surface !== surface) {
          errors.push(`${command.id}: admission changed the command or surface`);
        }
      } catch (error) {
        if (allowed || !/currently Node-only/u.test(error.message)) {
          errors.push(`${command.id}: unexpected ${surface} admission failure: ${error.message}`);
        }
      }
    }
  }
  return errors;
}

export async function buildCommandSurfaceContractReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const [policy, cliSource] = await Promise.all([
    readJson(policyPath),
    fs.readFile(options.cliPath || CLI_PATH, 'utf8'),
  ]);
  const errors = [];
  const policyCommandIds = validatePolicy(policy, errors);
  validateCommandSet(policyCommandIds, errors);
  validateCliUsage(policy, cliSource, errors);
  errors.push(...validateCommandAdmission(policy));
  const commands = Array.isArray(policy?.commands) ? policy.commands : [];
  return {
    ok: errors.length === 0,
    policyPath: path.relative(options.repoRoot || REPO_ROOT, policyPath),
    errors,
    commands: commands.map((command) => ({
      id: command.id,
      surfaces: command.surfaces,
      cliUsageRequired: command.cliUsageRequired,
    })),
  };
}

export async function main() {
  const report = await buildCommandSurfaceContractReport();
  if (!report.ok) {
    for (const error of report.errors) {
      console.error(`command-surface-contract: ${error}`);
    }
    process.exitCode = 1;
    return;
  }
  console.log(`command-surface-contract: policy ok (${report.commands.length} commands)`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
