import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(fileURLToPath(new URL('../..', import.meta.url)));
const tempDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-supported-manifest-kernels-'));

function runReport(args) {
  return spawnSync(process.execPath, ['tools/list-supported-manifest-kernels.js', ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

async function writeJson(filePath, payload) {
  await writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

try {
  const manifestRoot = path.join(tempDir, 'models');
  const modelDir = path.join(manifestRoot, 'fixture-model');
  const catalogPath = path.join(tempDir, 'catalog.json');
  const registryPath = path.join(tempDir, 'registry.json');
  const capabilityRulesPath = path.join(tempDir, 'capability-transforms.rules.json');
  await mkdir(modelDir, { recursive: true });

  await writeJson(catalogPath, {
    models: [
      {
        modelId: 'fixture-model',
        lifecycle: {
          status: {
            runtime: 'active',
            conversion: 'ready',
            tested: 'verified',
          },
          tested: {
            result: 'pass',
          },
        },
      },
    ],
  });

  await writeJson(path.join(modelDir, 'manifest.json'), {
    modelId: 'fixture-model',
    architecture: {
      headDim: 128,
    },
    inference: {
      session: {
        compute: {
          defaults: {
            activationDtype: 'f16',
            mathDtype: 'f16',
            accumDtype: 'f16',
          },
        },
        kvcache: {
          kvDtype: 'f16',
        },
        retainQ4KMaterialization: false,
      },
      execution: {
        kernels: {
          fixture_kernel: {
            kernel: 'fixture_used.wgsl',
            entry: 'main',
            digest: 'sha256:fixture',
          },
        },
        prefill: [],
        decode: [],
      },
    },
  });

  await writeJson(registryPath, {
    operations: {
      fixture: {
        variants: {
          used: {
            wgsl: 'fixture_used.wgsl',
            entryPoint: 'main',
            reachability: {
              status: 'model-selectable',
            },
          },
        },
      },
      matmul: {
        variants: {
          q4_fused_multicol_f16a_f32acc: {
            wgsl: 'fused_matmul_q4_multicol_f16a_f32acc.wgsl',
            entryPoint: 'main_multicol_f16a_f32acc',
            reachability: {
              status: 'unused',
            },
          },
        },
      },
    },
  });

  await writeJson(capabilityRulesPath, {
    capabilityTransforms: [
      {
        match: {
          modelId: 'fixture-model',
          platformVendor: 'apple',
        },
        transforms: ['failClosedLaneMismatch'],
        reason: 'Fixture rejection lane must not execute during static kernel reachability reporting.',
      },
    ],
  });

  const result = runReport([
    '--catalog', catalogPath,
    '--manifest-root', manifestRoot,
    '--registry', registryPath,
    '--capability-rules', capabilityRulesPath,
    '--no-allowlist',
    '--json',
    '--limit', '0',
  ]);
  assert.equal(result.status, 0, result.stderr || result.stdout);

  const report = JSON.parse(result.stdout);
  assert.equal(report.summary.loadedManifests, 1);
  assert.equal(report.summary.capabilityTransformApplications, 0);
  assert.equal(report.summary.capabilityKernelKeysUsed, 0);
  assert.equal(report.summary.liveKernelKeysUsed, 1);
  assert.ok(
    report.jsDispatchedRegistryVariants.some((entry) => {
      return entry.operation === 'matmul'
        && entry.variant === 'q4_fused_multicol_f16a_f32acc'
        && entry.sources.includes('src/gpu/kernels/matmul.js');
    }),
    'operation-local variant literals in kernel wrappers must count as JS-dispatched reachability'
  );
} finally {
  await rm(tempDir, { recursive: true, force: true });
}

console.log('supported-manifest-kernels-cli.test: ok');
