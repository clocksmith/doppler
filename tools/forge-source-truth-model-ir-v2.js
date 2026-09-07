#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { forgeSourceTruthFromFiles } from '../src/tooling/source-truth-inputs.js';
import { runModelOnboarding } from '../src/tooling/model-onboarding.js';

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!['--spec', '--config', '--execution', '--out'].includes(token)) throw new Error(`Unknown argument "${token}".`);
    const value = argv[++index];
    if (!value || value.startsWith('--') || Object.hasOwn(options, token.slice(2))) {
      throw new Error(`${token} requires one value and may not repeat.`);
    }
    options[token.slice(2)] = value;
  }
  if (!options.out || Boolean(options.spec) === Boolean(options.config) || (options.execution && !options.config)) {
    throw new Error('Usage: --spec <source-spec> --out <receipt.json> OR --config <onboarding.json> [--execution <embedding-plan.json>] --out <directory>');
  }
  return options;
}

const options = parseArgs(process.argv.slice(2));
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
if (options.config) {
  const config = JSON.parse(await fs.readFile(path.resolve(options.config), 'utf8'));
  let execution;
  if (options.execution) {
    const driver = await import('./model-onboarding-embedding-driver.js');
    execution = { config: JSON.parse(await fs.readFile(path.resolve(options.execution), 'utf8')),
      runStage: driver.runEmbeddingOnboardingStage, verifyStage: driver.verifyEmbeddingOnboardingStage };
  }
  const result = await runModelOnboarding(config, { sourceRoot: repoRoot, outputDir: path.resolve(options.out), execution });
  console.log(JSON.stringify(result, null, 2));
  if (result.status === 'blocked') process.exitCode = 1;
} else {
  const spec = JSON.parse(await fs.readFile(path.resolve(options.spec), 'utf8'));
  const receipt = await forgeSourceTruthFromFiles(spec, repoRoot);
  const outputPath = path.resolve(options.out);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' });
  console.log(`${receipt.modelIR.modelId}: ${receipt.intakeDigest} -> ${outputPath}`);
}
