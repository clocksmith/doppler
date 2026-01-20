import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { loadConfig } from '../cli/config/index.js';
import { validateManifest } from '../src/adapters/adapter-manifest.js';

function parseArgs(argv) {
  const opts = { config: null, help: false };
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      i += 1;
      continue;
    }
    if (arg === '--config' || arg === '-c') {
      opts.config = argv[i + 1] || null;
      i += 2;
      continue;
    }
    if (!arg.startsWith('-') && !opts.config) {
      opts.config = arg;
      i += 1;
      continue;
    }
    console.error(`Unknown argument: ${arg}`);
    opts.help = true;
    break;
  }
  return opts;
}

function printHelp() {
  console.log(`
RDRR-LoRA to GGUF plan helper.

Usage:
  doppler --config <ref>

Config requirements:
  tools.rdrrLoraToGguf.manifest (string, required)
  tools.rdrrLoraToGguf.outputDir (string|null, optional)
`);
}

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertStringOrNull(value, label) {
  if (value === null) return;
  assertString(value, label);
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.help) {
    printHelp();
    process.exit(0);
  }
  if (!parsed.config) {
    console.error('Error: --config is required');
    printHelp();
    process.exit(1);
  }

  const loaded = await loadConfig(parsed.config);
  const raw = loaded.raw ?? {};
  assertObject(raw.tools, 'tools');
  const toolConfig = raw.tools?.rdrrLoraToGguf;
  if (!toolConfig || typeof toolConfig !== 'object') {
    throw new Error('tools.rdrrLoraToGguf is required in config');
  }

  assertString(toolConfig.manifest, 'tools.rdrrLoraToGguf.manifest');
  assertStringOrNull(toolConfig.outputDir ?? null, 'tools.rdrrLoraToGguf.outputDir');

  const manifestPath = resolve(toolConfig.manifest);
  const outDir = toolConfig.outputDir ? resolve(toolConfig.outputDir) : process.cwd();
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
