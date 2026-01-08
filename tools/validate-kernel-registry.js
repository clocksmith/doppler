import { promises as fs } from 'node:fs';
import path from 'node:path';

const REGISTRY_PATH = path.join(process.cwd(), 'src', 'config', 'kernels', 'registry.json');
const KERNEL_DIR = path.join(process.cwd(), 'src', 'gpu', 'kernels');

/**
 * @param {string} filePath
 * @returns {Promise<boolean>}
 */
async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

/**
 * @returns {Promise<void>}
 */
async function main() {
  const raw = await fs.readFile(REGISTRY_PATH, 'utf8');
  /** @type {import('./validate-kernel-registry.js').KernelRegistry} */
  const registry = JSON.parse(raw);

  /** @type {Set<string>} */
  const wgslFiles = new Set();
  for (const operation of Object.values(registry.operations)) {
    for (const variant of Object.values(operation.variants)) {
      wgslFiles.add(variant.wgsl);
    }
  }

  /** @type {string[]} */
  const missing = [];
  for (const wgsl of wgslFiles) {
    const filePath = path.join(KERNEL_DIR, wgsl);
    if (!(await fileExists(filePath))) {
      missing.push(wgsl);
    }
  }

  if (missing.length > 0) {
    console.error('Kernel registry references missing WGSL files:');
    for (const name of missing.sort()) {
      console.error(`  - ${name}`);
    }
    process.exit(1);
  }

  console.log(`Kernel registry OK (${wgslFiles.size} WGSL files referenced).`);
}

main().catch((error) => {
  console.error('Kernel registry validation failed:', error instanceof Error ? error.message : error);
  process.exit(1);
});
