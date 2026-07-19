#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import fs from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? ROOT_DIR,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(' ')} failed:\n${result.stderr || result.stdout || `exit code ${result.status ?? 1}`}`
    );
  }
  return result.stdout;
}

function collectExportSpecifiers(packageJson) {
  return Object.keys(packageJson.exports ?? {}).map((key) => (
    key === '.' ? packageJson.name : `${packageJson.name}${key.slice(1)}`
  ));
}

async function writeImportSmoke(consumerDir, packageJson) {
  const specifiers = collectExportSpecifiers(packageJson);
  const browserToolingPath = pathToFileURL(
    path.join(consumerDir, 'node_modules', packageJson.name, 'src/tooling-exports.browser.js')
  ).href;
  const source = [
    `const specifiers = ${JSON.stringify(specifiers)};`,
    'for (const specifier of specifiers) {',
    '  await import(specifier);',
    '}',
    `await import(${JSON.stringify(browserToolingPath)});`,
    `console.log(\`package import smoke passed (${specifiers.length} exports + browser condition)\`);`,
    '',
  ].join('\n');
  const smokePath = path.join(consumerDir, 'import-smoke.js');
  await fs.writeFile(smokePath, source, 'utf8');
  run(process.execPath, [smokePath], { cwd: consumerDir });
  console.log(`package import smoke passed (${specifiers.length} exports + browser condition)`);
}

async function runTrainingApiSmoke(consumerDir, packageJson) {
  const source = [
    `import { getTrainingCapabilities, TRAINING_BACKENDS } from '${packageJson.name}/training';`,
    "const capabilities = getTrainingCapabilities({ kind: 'lora', baseModelId: 'qwen-3-5-0-8b-q4k-ehaf16', pipeline: { datasetFormat: 'text-pairs', taskType: 'text_generation' } });",
    "if (!capabilities.supported || capabilities.backends.webgpuNative.supported || !capabilities.backends.external.supported) throw new Error('training capability contract mismatch');",
    "if (TRAINING_BACKENDS.join(',') !== 'webgpu_native,external') throw new Error('training backend registry mismatch');",
    "console.log('package training API smoke passed');",
    '',
  ].join('\n');
  const smokePath = path.join(consumerDir, 'training-smoke.js');
  await fs.writeFile(smokePath, source, 'utf8');
  run(process.execPath, [smokePath], { cwd: consumerDir });
  console.log('package training API smoke passed');
}

async function writeTypeSmoke(consumerDir, packageJson) {
  const specifiers = collectExportSpecifiers(packageJson);
  const source = specifiers
    .map((specifier, index) => `type PackageExport${index} = typeof import(${JSON.stringify(specifier)});`)
    .join('\n') + '\n';
  await fs.writeFile(path.join(consumerDir, 'consumer.ts'), source, 'utf8');
  await fs.writeFile(
    path.join(consumerDir, 'tsconfig.json'),
    `${JSON.stringify({
      compilerOptions: {
        module: 'NodeNext',
        moduleResolution: 'NodeNext',
        target: 'ES2022',
        lib: ['ES2022', 'DOM'],
        strict: true,
        noEmit: true,
        skipLibCheck: false,
      },
      include: ['consumer.ts'],
    }, null, 2)}\n`,
    'utf8'
  );
  const tscPath = path.join(ROOT_DIR, 'node_modules/typescript/bin/tsc');
  run(process.execPath, [tscPath, '-p', 'tsconfig.json'], { cwd: consumerDir });
  console.log(`package type smoke passed (${specifiers.length} public export declarations)`);
}

async function assertInstalledFiles(consumerDir, packageJson) {
  const packageDir = path.join(consumerDir, 'node_modules', packageJson.name);
  const required = [
    'models/catalog.json',
    'src/tooling/command-runner.html',
    'tests/kernels/browser/kernel-suite.js',
    'tests/kernels/browser/test-page.js',
  ];
  for (const relativePath of required) {
    await fs.access(path.join(packageDir, relativePath));
  }
  for (const optionalName of Object.keys(packageJson.optionalDependencies ?? {})) {
    try {
      await fs.access(path.join(consumerDir, 'node_modules', optionalName));
    } catch {
      continue;
    }
    throw new Error(`optional dependency should be omitted from package smoke install: ${optionalName}`);
  }
}

async function runCliSmokes(consumerDir, packageJson) {
  const packageDir = path.join(consumerDir, 'node_modules', packageJson.name);
  for (const [name, target] of Object.entries(packageJson.bin ?? {})) {
    run(process.execPath, [path.join(packageDir, target), '--help'], { cwd: consumerDir });
    console.log(`package CLI smoke passed (${name})`);
  }
}

async function main() {
  const packageJson = JSON.parse(await fs.readFile(path.join(ROOT_DIR, 'package.json'), 'utf8'));
  const tempRoot = await fs.mkdtemp(path.join(tmpdir(), 'doppler-package-smoke-'));
  const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  try {
    const packOutput = run(
      npmCommand,
      [
        'pack',
        '--json',
        '--ignore-scripts',
        '--pack-destination',
        tempRoot,
        '--cache',
        path.join(tempRoot, 'npm-cache'),
      ]
    );
    const packed = JSON.parse(packOutput)[0];
    const tarballPath = path.join(tempRoot, packed.filename);
    const consumerDir = path.join(tempRoot, 'consumer');
    await fs.mkdir(consumerDir, { recursive: true });
    await fs.writeFile(
      path.join(consumerDir, 'package.json'),
      `${JSON.stringify({ private: true, type: 'module' }, null, 2)}\n`,
      'utf8'
    );
    run(
      npmCommand,
      [
        'install',
        tarballPath,
        '--ignore-scripts',
        '--omit=optional',
        '--offline',
        '--no-audit',
        '--no-fund',
        '--cache',
        path.join(tempRoot, 'npm-cache'),
      ],
      { cwd: consumerDir }
    );

    await assertInstalledFiles(consumerDir, packageJson);
    await writeImportSmoke(consumerDir, packageJson);
    await runTrainingApiSmoke(consumerDir, packageJson);
    await runCliSmokes(consumerDir, packageJson);
    await writeTypeSmoke(consumerDir, packageJson);
    console.log(
      `packed package smoke passed (${packed.entryCount} files, ${packed.size} bytes packed, `
      + `${packed.unpackedSize} bytes unpacked)`
    );
  } finally {
    await fs.rm(tempRoot, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
