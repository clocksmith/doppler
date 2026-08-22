#!/usr/bin/env node

/**
 * Doppler Pack v2 Invariant & Schema Validator CLI
 *
 * @module tools/check-pack-v2
 */

import path from 'node:path';
import process from 'node:process';
import { loadPackV2, validatePackV2 } from '../src/tooling/pack-v2.js';

async function main(argv = process.argv.slice(2)) {
  if (argv.length === 0 || argv.includes('--help') || argv.includes('-h')) {
    console.log('Usage: node tools/check-pack-v2.js <pack-path-1> [pack-path-2...]');
    return;
  }

  const results = [];
  let allOk = true;

  for (const rawPath of argv) {
    const resolved = path.resolve(rawPath);
    try {
      const pack = await loadPackV2(resolved);
      const validation = validatePackV2(pack);
      results.push({
        ok: validation.ok,
        path: resolved,
        modelId: pack.modelId,
        packId: pack.packId,
        targetPlanCount: pack.targetPlans?.length ?? 0,
        wgslModuleCount: pack.wgslModules?.length ?? 0,
        artifactCount: pack.artifacts?.length ?? 0,
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
    console.error(`[check-pack-v2] ${err.message}`);
    process.exit(1);
  });
}
