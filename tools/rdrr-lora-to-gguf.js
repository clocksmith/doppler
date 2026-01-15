import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { validateManifest } from '../src/adapters/adapter-manifest.js';

function parseArgs(argv) {
  const args = { manifest: null, out: null };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--manifest') {
      args.manifest = argv[i + 1];
      i += 1;
    } else if (arg === '--out') {
      args.out = argv[i + 1];
      i += 1;
    }
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.manifest) {
    console.error('Usage: node tools/rdrr-lora-to-gguf.js --manifest adapter.json [--out ./out]');
    process.exit(1);
  }

  const manifestPath = resolve(args.manifest);
  const outDir = args.out ? resolve(args.out) : process.cwd();
  const manifestText = await readFile(manifestPath, 'utf-8');
  const manifest = JSON.parse(manifestText);
  const validation = validateManifest(manifest);
  if (!validation.valid) {
    const errors = validation.errors.map((e) => `${e.field}: ${e.message}`).join('; ');
    throw new Error(`Invalid manifest: ${errors}`);
  }

  const summary = {
    id: manifest.id,
    name: manifest.name,
    baseModel: manifest.baseModel,
    rank: manifest.rank,
    alpha: manifest.alpha,
    targetModules: manifest.targetModules,
    tensorCount: manifest.tensors?.length ?? 0,
  };

  const plan = [
    'RDRR-LoRA to GGUF conversion (manual steps):',
    '',
    '1) Convert inline tensors to a LoRA safetensors/npz file.',
    '   - Each tensor name must match the layer/module naming in your conversion tool.',
    '2) Use llama.cpp conversion tooling to produce GGUF:',
    '   - ./llama.cpp/convert-lora-to-gguf --in lora.safetensors --out lora.gguf',
    '3) Load the adapter in your GGUF runtime with the base model.',
    '',
    `Manifest summary: ${JSON.stringify(summary)}`,
  ];

  const planPath = resolve(outDir, `${manifest.id || 'adapter'}-gguf-plan.txt`);
  await writeFile(planPath, plan.join('\n'), 'utf-8');
  console.log(`Wrote conversion plan: ${planPath}`);
}

main().catch((err) => {
  console.error(err.stack || err.message || err);
  process.exit(1);
});
