#!/usr/bin/env node

import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

import { hashWgslSemanticEvidenceValue } from '../src/tooling/wgsl-repair-semantic-gate.js';
import { createWgslAuthorBrowserExecutor } from './lib/wgsl-author-browser-executor.js';
import {
  evaluateWgslAuthorReferenceOracle,
  materializeWgslAuthorReferenceTask,
  validateWgslAuthorReferenceManifest,
} from './lib/wgsl-author-reference.js';

const DEFAULT_POLICY = 'tools/policies/wgsl-writer-v3-campaign-policy.json';

function parseArgs(argv) {
  const args = { policyPath: DEFAULT_POLICY, outputPath: '' };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--policy') args.policyPath = argv[++index] || '';
    else if (token === '--out') args.outputPath = argv[++index] || '';
    else throw new Error(`Unknown argument: ${token}`);
  }
  if (!args.policyPath) throw new Error('--policy requires a path.');
  return args;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(path.resolve(filePath), 'utf8'));
}

async function sha256File(filePath) {
  return createHash('sha256').update(await fs.readFile(path.resolve(filePath))).digest('hex');
}

async function requireBinding(binding, label) {
  const actual = await sha256File(binding.path);
  if (actual !== binding.sha256) {
    throw new Error(`${label} SHA-256 mismatch: expected ${binding.sha256}, got ${actual}.`);
  }
  return actual;
}

export async function runWgslAuthorV3Reference(args) {
  const policy = await readJson(args.policyPath);
  if (policy?.policyId !== 'doppler-wgsl-writer-v3-general-authoring'
    || policy?.status !== 'mechanics_implemented_reference_qualification_blocked'
    || policy?.authority?.training !== false
    || policy?.authority?.promotion !== false) {
    throw new Error('WGSL author v3 reference runner requires the blocked v3 campaign policy.');
  }
  const bindings = [
    ['response schema', policy.mechanics.responseSchema],
    ['package validator', policy.mechanics.packageValidator],
    ['execution planner', policy.mechanics.executionPlanner],
    ['format catalog', policy.mechanics.formatCatalog],
    ['capability catalog', policy.mechanics.capabilityCatalog],
    ['browser executor', policy.mechanics.browserExecutor],
    ['reference library', policy.mechanics.referenceQualification.library],
    ['reference harness', policy.mechanics.referenceQualification.harness],
    ['reference manifest', policy.mechanics.referenceQualification.manifest],
  ];
  await Promise.all(bindings.map(([label, binding]) => requireBinding(binding, label)));
  const manifest = validateWgslAuthorReferenceManifest(
    await readJson(policy.mechanics.referenceQualification.manifest.path)
  );
  if (manifest.experimentId !== policy.experimentId
    || manifest.formatCatalog.path !== policy.mechanics.formatCatalog.path
    || manifest.formatCatalog.sha256 !== policy.mechanics.formatCatalog.sha256) {
    throw new Error('WGSL author reference manifest is not bound to the campaign.');
  }
  const formatCatalog = await readJson(manifest.formatCatalog.path);
  const materialized = [];
  for (const task of manifest.tasks) {
    materialized.push(await materializeWgslAuthorReferenceTask(
      task,
      manifest,
      formatCatalog
    ));
  }
  const executor = await createWgslAuthorBrowserExecutor({
    browserArgs: manifest.runtime.browserArgs,
    requiredFeatures: manifest.runtime.requiredFeatures,
    requiredLimits: manifest.runtime.requiredLimits,
    powerPreference: manifest.runtime.powerPreference,
    executionTimeoutMs: manifest.runtime.executionTimeoutMs,
  });
  try {
    const tasks = [];
    for (const task of materialized) {
      const execution = await executor.execute(task.plan, { id: task.taskId });
      const oracle = evaluateWgslAuthorReferenceOracle(task.oracle, execution);
      tasks.push({
        taskId: task.taskId,
        pipelineKind: task.pipelineKind,
        objective: task.objective,
        sourceBindings: task.sourceBindings,
        packageSha256: task.packageSha256,
        planSha256: task.planSha256,
        execution,
        oracle,
        pass: execution.passed === true && oracle.pass === true,
      });
    }
    const passedTasks = tasks.filter((task) => task.pass).length;
    const allTasksPass = passedTasks === tasks.length;
    const core = {
      schema: 'doppler.wgsl-author-reference-qualification/v1',
      experimentId: policy.experimentId,
      evaluationRole: manifest.role,
      policy: {
        path: args.policyPath,
        sha256: await sha256File(args.policyPath),
      },
      manifest: policy.mechanics.referenceQualification.manifest,
      runtime: {
        backend: 'chromium_webgpu',
        deviceInfo: executor.deviceInfo,
        browserArgs: executor.browserArgs,
        executionTimeoutMs: executor.executionTimeoutMs,
      },
      tasks,
      summary: {
        tasks: tasks.length,
        passedTasks,
        failedTasks: tasks.length - passedTasks,
        computeTasks: tasks.filter((task) => task.pipelineKind === 'compute').length,
        renderTasks: tasks.filter((task) => task.pipelineKind === 'render').length,
        multiPassTasks: tasks.filter((task) => task.pipelineKind === 'multi_pass').length,
      },
      decision: allTasksPass
        ? 'reference_package_mechanics_qualified'
        : 'reference_package_mechanics_failed',
      mechanicsQualificationAuthority: true,
      corpusMaterializationAuthority: false,
      trainingAuthority: false,
      selectionAuthority: false,
      confirmationAuthority: false,
      promotionAuthority: false,
      generalWgslWriterClaim: false,
      productizationAllowed: false,
      claimBoundary: 'A passing receipt qualifies visible compute, procedural-render, indexed-render, multi-pass, readback, and oracle mechanics only. It is not model capability, corpus, training, selection, confirmation, promotion, or product evidence.',
    };
    return { ...core, receiptHash: hashWgslSemanticEvidenceValue(core) };
  } finally {
    await executor.close();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const receipt = await runWgslAuthorV3Reference(args);
  const json = `${JSON.stringify(receipt, null, 2)}\n`;
  if (args.outputPath) {
    const outputPath = path.resolve(args.outputPath);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, json, 'utf8');
    console.error(`[wgsl-author-reference] wrote ${args.outputPath}`);
  } else {
    process.stdout.write(json);
  }
  if (receipt.decision !== 'reference_package_mechanics_qualified') process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.stack : String(error));
    process.exitCode = 1;
  });
}
