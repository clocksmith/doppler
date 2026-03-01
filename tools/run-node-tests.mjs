#!/usr/bin/env node

import { existsSync, readdirSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installNodeFileFetchShim } from '../src/tooling/node-file-fetch.js';

const ROOT_DIR = process.cwd();

function ensureGpuEnumGlobals() {
  if (typeof globalThis.GPUBufferUsage === 'undefined') {
    globalThis.GPUBufferUsage = {
      MAP_READ: 0x0001,
      MAP_WRITE: 0x0002,
      COPY_SRC: 0x0004,
      COPY_DST: 0x0008,
      INDEX: 0x0010,
      VERTEX: 0x0020,
      UNIFORM: 0x0040,
      STORAGE: 0x0080,
      INDIRECT: 0x0100,
      QUERY_RESOLVE: 0x0200,
    };
  }
  if (typeof globalThis.GPUMapMode === 'undefined') {
    globalThis.GPUMapMode = {
      READ: 0x0001,
      WRITE: 0x0002,
    };
  }
  if (typeof globalThis.GPUShaderStage === 'undefined') {
    globalThis.GPUShaderStage = {
      VERTEX: 0x1,
      FRAGMENT: 0x2,
      COMPUTE: 0x4,
    };
  }
  if (typeof globalThis.GPUTextureUsage === 'undefined') {
    globalThis.GPUTextureUsage = {
      COPY_SRC: 0x01,
      COPY_DST: 0x02,
      TEXTURE_BINDING: 0x04,
      STORAGE_BINDING: 0x08,
      RENDER_ATTACHMENT: 0x10,
    };
  }
}

const suites = {
  unit: [
    'tests/config',
    'tests/converter',
    'tests/integration',
    'tests/inference',
  ],
  gpu: [
    'tests/kernels',
  ],
  all: [
    'tests/config',
    'tests/converter',
    'tests/integration',
    'tests/inference',
    'tests/kernels',
  ],
};

function parseArgs() {
  const args = process.argv.slice(2);
  const directories = [];
  let suite = 'all';

  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--suite') {
      const value = args[i + 1];
      if (value) {
        suite = value;
        i += 1;
      }
      continue;
    }

    if (args[i].startsWith('--suite=')) {
      suite = args[i].split('=', 2)[1];
      continue;
    }

    directories.push(args[i]);
  }

  return { suite, directories };
}

function collectTestFiles(dir, files) {
  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      collectTestFiles(fullPath, files);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.test.js')) {
      files.push(fullPath);
    }
  }
}

function listRootsFromSuite(suiteName, explicitDirs) {
  if (explicitDirs.length > 0) return explicitDirs;
  return suites[suiteName] ? suites[suiteName].map((dir) => resolve(ROOT_DIR, dir)) : suites.all.map((dir) => resolve(ROOT_DIR, dir));
}

async function main() {
  ensureGpuEnumGlobals();
  installNodeFileFetchShim();
  const { suite, directories } = parseArgs();
  const resolvedSuite = Object.hasOwn(suites, suite) ? suite : 'all';
  const selectedRoots = listRootsFromSuite(resolvedSuite, directories.map((dir) => resolve(ROOT_DIR, dir)));
  const testFiles = [];

  for (const root of selectedRoots) {
    if (!existsSync(root)) {
      // keep behavior permissive: skip missing directories
      continue;
    }
    collectTestFiles(root, testFiles);
  }

  if (testFiles.length === 0) {
    console.log('[node-tests] no matching tests found');
    return;
  }

  const failures = [];
  for (const file of testFiles.sort()) {
    const rel = relative(ROOT_DIR, file);
    try {
      // eslint-disable-next-line no-await-in-loop
      await import(pathToFileURL(file).href);
      console.log(`[node-tests] ok: ${rel}`);
    } catch (error) {
      failures.push({ file, error });
      console.error(`[node-tests] fail: ${rel}`);
      console.error(error?.stack || String(error));
    }
  }

  if (failures.length > 0) {
    console.error(`[node-tests] failed ${failures.length}/${testFiles.length}`);
    process.exit(1);
  }

  console.log(`[node-tests] ok: ${testFiles.length} files`);
}

await main();
