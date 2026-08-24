#!/usr/bin/env node

import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  FORGE_VERSION,
  buildForgeOptions as buildForgeOptionsCore,
  forgeModelPack,
  parseArgs,
  readJsonInput,
  usage,
} from '../src/tooling/model-pack-forge.js';
import {
  PACK_V0_DEVELOPMENT_AUTHORITY,
  PACK_V0_TRUSTED_SIGNERS,
} from '../src/config/pack-v0-trusted-signers.js';

const DEVELOPMENT_PRIVATE_KEY_PATH = fileURLToPath(
  new URL('./fixtures/pack-v0-development-signing-private.jwk.json', import.meta.url)
);
const DEVELOPMENT_PUBLIC_KEY_PATH = fileURLToPath(
  new URL('./fixtures/pack-v0-development-signing-public.jwk.json', import.meta.url)
);

export {
  FORGE_VERSION,
  forgeModelPack,
  parseArgs,
  readJsonInput,
  usage,
};

export async function buildForgeOptions(flags, metaUrl = import.meta.url) {
  const options = await buildForgeOptionsCore(flags, metaUrl);
  const authority = options.signingAuthority ?? PACK_V0_DEVELOPMENT_AUTHORITY;
  return {
    ...options,
    signingPrivateKeyPath: options.signingPrivateKeyPath ?? DEVELOPMENT_PRIVATE_KEY_PATH,
    signingPublicKeyPath: options.signingPublicKeyPath ?? DEVELOPMENT_PUBLIC_KEY_PATH,
    signingAuthority: authority,
    allowDevelopmentSigner: Object.hasOwn(PACK_V0_TRUSTED_SIGNERS, authority),
  };
}

export async function main(argv = process.argv.slice(2)) {
  const flags = parseArgs(argv);
  if (flags.help) {
    console.log(usage());
    return;
  }
  const receipt = await forgeModelPack(await buildForgeOptions(flags));
  if (flags.json) {
    console.log(JSON.stringify(receipt, null, 2));
    return;
  }
  console.log('✔ Doppler Forge: signed Pack v2 compiled');
  console.log(`  Model ID:       ${receipt.modelId}`);
  console.log(`  Pack ID:        ${receipt.packId}`);
  console.log(`  Semantic root:  ${receipt.semanticRoot}`);
  console.log(`  Target plans:   ${receipt.targetPlanDigests.length}`);
  console.log(`  Output Pack:    ${receipt.outputPath}`);
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  return entryPath && path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[doppler-forge] ${error.message}`);
    process.exit(1);
  });
}
