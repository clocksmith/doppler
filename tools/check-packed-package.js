#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../tests/helpers/pack-v2-fixture.js';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const commandLog = [];

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? ROOT_DIR,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  commandLog.push({ command, args, cwd: options.cwd ?? ROOT_DIR, status: result.status,
    signal: result.signal, stdout: result.stdout, stderr: result.stderr, error: result.error?.message ?? null });
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
    .join('\n') + `
import { createDocumentSearchRenderer, createDocumentSearchHostRenderer } from './renderer.js';
import { openPack } from '${packageJson.name}/host';
import type { DopplerPackOpenOptions } from '${packageJson.name}/host';
import { registerDocumentSearchReleaseMain } from './main.js';
import type { RuntimePorts, PackRerankRequest, DopplerRuntimeSession } from '${packageJson.name}';
import type { ElectronReleaseStateCoordinator } from '${packageJson.name}/electron';
declare const ports: RuntimePorts;
declare const releaseState: ElectronReleaseStateCoordinator;
declare const request: PackRerankRequest;
declare const trustOptions: DopplerPackOpenOptions;
const hostRenderer = createDocumentSearchHostRenderer(releaseState, trustOptions);
hostRenderer.rerank(request).then(receipt => receipt.pack.semanticRoot);
const hostSession: Promise<DopplerRuntimeSession> = openPack('https://application.example/pack.json', trustOptions);
const renderer = createDocumentSearchRenderer(releaseState, ports);
renderer.rerank(request).then(receipt => receipt.pack.semanticRoot);
const session: Promise<DopplerRuntimeSession> = renderer.openCurrent();
// @ts-expect-error Positional reranking cannot omit the application binding.
renderer.rerank('query', ['document']);
// @ts-expect-error Host ports must be explicit.
createDocumentSearchRenderer(releaseState);
type MainOptions = Parameters<typeof registerDocumentSearchReleaseMain>[0];
`;
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

async function runElectronPackSmoke(consumerDir) {
  for (const name of ['main', 'preload', 'renderer']) {
    for (const extension of ['js', 'd.ts']) {
      const filename = `${name}.${extension}`;
      await fs.copyFile(
        path.join(ROOT_DIR, 'examples/electron-document-search', filename),
        path.join(consumerDir, filename),
      );
    }
  }
  await fs.copyFile(
    path.join(ROOT_DIR, 'tests/helpers/electron-pack-contract.js'),
    path.join(consumerDir, 'electron-pack-contract.js'),
  );
  await fs.copyFile(
    path.join(ROOT_DIR, 'tests/fixtures/packed-electron-consumer.js'),
    path.join(consumerDir, 'electron-smoke.js'),
  );
  const fixture = await createSignedPackFixture();
  await fs.writeFile(path.join(consumerDir, 'pack-fixture.json'), JSON.stringify({
    pack: fixture.pack,
    trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
    artifacts: [...fixture.artifactBytes].map(([id, bytes]) => [id, [...bytes]]),
  }));
  const output = run(process.execPath, ['electron-smoke.js'], { cwd: consumerDir });
  process.stdout.write(output);
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

export function parsePackageSmokeArgs(argv) {
  if (argv.length === 0) return { retain: null };
  if (argv.length === 2 && argv[0] === '--retain' && argv[1] && !argv[1].startsWith('--')) {
    return { retain: path.resolve(argv[1]) };
  }
  throw new Error('Usage: node tools/check-packed-package.js [--retain <new-bundle-directory>]');
}

async function main() {
  const options = parsePackageSmokeArgs(process.argv.slice(2));
  const packageJson = JSON.parse(await fs.readFile(path.join(ROOT_DIR, 'package.json'), 'utf8'));
  const tempRoot = options.retain ?? await fs.mkdtemp(path.join(tmpdir(), 'doppler-package-smoke-'));
  if (options.retain) {
    await fs.mkdir(path.dirname(tempRoot), { recursive: true });
    await fs.mkdir(tempRoot); // Never overwrite an earlier evidence bundle.
  }
  const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const receipt = {
    evidenceClass: 'installed-package-test', passed: false,
    physicalExecution: false, externalAdoption: false,
    nodeVersion: process.version, platform: process.platform, arch: process.arch,
    startedAtUtc: new Date().toISOString(), package: null, failure: null,
  };
  try {
    if (options.retain) {
      const source = {
        commit: run('git', ['rev-parse', 'HEAD']).trim(),
        status: run('git', ['status', '--porcelain=v1']),
        runnerSha256: createHash('sha256').update(await fs.readFile(fileURLToPath(import.meta.url))).digest('hex'),
      };
      await fs.writeFile(path.join(tempRoot, 'source-state.json'), JSON.stringify(source, null, 2));
    }
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
    if (!packOutput.trim()) {
      throw new Error('npm pack returned no JSON package metadata.');
    }
    const packed = JSON.parse(packOutput)[0];
    const tarballPath = path.join(tempRoot, packed.filename);
    receipt.package = {
      filename: packed.filename, integrity: packed.integrity, sizeBytes: packed.size,
      sha256: createHash('sha256').update(await fs.readFile(tarballPath)).digest('hex'),
    };
    if (options.retain) await fs.writeFile(path.join(tempRoot, 'npm-pack.json'), packOutput);
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
    await runElectronPackSmoke(consumerDir);
    await writeTypeSmoke(consumerDir, packageJson);
    receipt.passed = true;
    console.log(
      `packed package smoke passed (${packed.entryCount} files, ${packed.size} bytes packed, `
      + `${packed.unpackedSize} bytes unpacked)`
    );
  } catch (error) {
    receipt.failure = { message: error.message, stack: error.stack };
    throw error;
  } finally {
    if (options.retain) {
      receipt.completedAtUtc = new Date().toISOString();
      await fs.writeFile(path.join(tempRoot, 'commands.json'), JSON.stringify(commandLog, null, 2));
      await fs.writeFile(path.join(tempRoot, 'receipt.json'), JSON.stringify(receipt, null, 2));
      console.log(`Package evidence retained: ${tempRoot}`);
    } else {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.stack : String(error));
    process.exitCode = 1;
  });
}
