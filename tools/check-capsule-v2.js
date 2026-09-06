#!/usr/bin/env node

/**
 * Doppler Capsule v2 Invariant & Schema Validator CLI
 *
 * @module tools/check-capsule-v2
 */

import path from 'node:path';
import process from 'node:process';
import { loadCapsuleV2, validateCapsuleV2 } from '../src/tooling/capsule-v2.js';

async function main(argv = process.argv.slice(2)) {
  if (argv.length === 0 || argv.includes('--help') || argv.includes('-h')) {
    console.log('Usage: node tools/check-capsule-v2.js <capsule-path-1> [capsule-path-2...]');
    return;
  }

  const results = [];
  let allOk = true;

  for (const rawPath of argv) {
    const resolved = path.resolve(rawPath);
    try {
      const capsule = await loadCapsuleV2(resolved);
      const validation = validateCapsuleV2(capsule);
      results.push({
        ok: validation.ok,
        path: resolved,
        modelId: capsule.modelId,
        capsuleId: capsule.capsuleId,
        targetPlanCount: capsule.targetPlans?.length ?? 0,
        wgslModuleCount: capsule.wgslModules?.length ?? 0,
        artifactCount: capsule.artifacts?.length ?? 0,
        errors: validation.errors,
      });
      if (!validation.ok) allOk = false;
    } catch (error) {
      allOk = false;
      results.push({
        ok: false,
        path: resolved,
        error: error.message,
      });
    }
  }

  console.log(JSON.stringify({ ok: allOk, results }, null, 2));
  if (!allOk) process.exit(1);
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  main().catch((err) => {
    console.error(`[check-capsule-v2] ${err.message}`);
    process.exit(1);
  });
}
