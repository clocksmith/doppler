import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

import {
  extractExecutionContractFacts,
  renderExecutionContractLeanModule,
  sanitizeLeanModuleName,
} from './lean-execution-contract.js';

export function resolveLeanBinary() {
  const elanLean = path.join(os.homedir(), '.elan', 'bin', 'lean');
  if (fs.existsSync(elanLean)) {
    return elanLean;
  }
  const probe = spawnSync('bash', ['-lc', 'command -v lean'], { encoding: 'utf8' });
  if (probe.status === 0) {
    const resolved = probe.stdout.trim();
    if (resolved) {
      return resolved;
    }
  }
  throw new Error('lean binary not found. Install Lean with elan first.');
}

function runLeanCommand({ leanBin, toolchainRef, buildDir, rootDir, sourcePath, outputPath }) {
  const result = spawnSync(
    leanBin,
    [`+${toolchainRef}`, '-o', outputPath, sourcePath],
    {
      cwd: rootDir,
      encoding: 'utf8',
      env: {
        ...process.env,
        LEAN_PATH: `${buildDir}:${path.join(rootDir, 'lean')}`,
      },
    }
  );
  if (result.stdout) {
    process.stdout.write(result.stdout);
  }
  if (result.stderr) {
    process.stderr.write(result.stderr);
  }
  if (result.status !== 0) {
    throw new Error(`lean exited with status ${result.status}`);
  }
}

export function runLeanCheck({ sourcePath, rootDir }) {
  const toolchainVersion = process.env.DOPPLER_LEAN_VERSION ?? '4.16.0';
  const toolchainRef = toolchainVersion.startsWith('v')
    ? `leanprover/lean4:${toolchainVersion}`
    : `leanprover/lean4:v${toolchainVersion}`;
  const leanBin = resolveLeanBinary();
  const buildDir = fs.mkdtempSync(path.join(os.tmpdir(), 'doppler-lean-execution-contract-'));
  try {
    fs.mkdirSync(path.join(buildDir, 'Doppler'), { recursive: true });
    runLeanCommand({
      leanBin,
      toolchainRef,
      buildDir,
      rootDir,
      sourcePath: path.join(rootDir, 'lean', 'Doppler', 'Model.lean'),
      outputPath: path.join(buildDir, 'Doppler', 'Model.olean'),
    });
    runLeanCommand({
      leanBin,
      toolchainRef,
      buildDir,
      rootDir,
      sourcePath: path.join(rootDir, 'lean', 'Doppler', 'ExecutionContract.lean'),
      outputPath: path.join(buildDir, 'Doppler', 'ExecutionContract.olean'),
    });
    const generatedOutput = path.join(buildDir, 'GeneratedExecutionContractCheck.olean');
    const result = spawnSync(
      leanBin,
      [`+${toolchainRef}`, '-o', generatedOutput, sourcePath],
      {
        cwd: rootDir,
        encoding: 'utf8',
        env: {
          ...process.env,
          LEAN_PATH: `${buildDir}:${path.join(rootDir, 'lean')}`,
        },
      }
    );
    if (result.stdout) {
      process.stdout.write(result.stdout);
    }
    if (result.stderr) {
      process.stderr.write(result.stderr);
    }
    if (result.status !== 0) {
      throw new Error(`lean exited with status ${result.status}`);
    }
    const overallMatch = result.stdout.match(/executionContractOverall:(pass|fail)/);
    if (!overallMatch) {
      throw new Error('unable to parse executionContractOverall from Lean output.');
    }
    return { ok: overallMatch[1] === 'pass', toolchainRef };
  } finally {
    fs.rmSync(buildDir, { recursive: true, force: true });
  }
}

export function writeExecutionContractLeanModuleForManifest(manifest, options = {}) {
  const rootDir = path.resolve(String(options.rootDir ?? process.cwd()));
  const facts = extractExecutionContractFacts(manifest);
  const moduleName = sanitizeLeanModuleName(options.moduleName ?? `${facts.modelId}_ExecutionContractCheck`);
  const source = renderExecutionContractLeanModule(facts, { moduleName });
  const tempDir = options.emitPath
    ? null
    : fs.mkdtempSync(path.join(rootDir, 'lean', '.generated-'));
  const generatedPath = options.emitPath
    ? path.resolve(rootDir, String(options.emitPath))
    : path.join(tempDir, `${moduleName}.lean`);
  fs.mkdirSync(path.dirname(generatedPath), { recursive: true });
  fs.writeFileSync(generatedPath, source);
  return {
    rootDir,
    facts,
    moduleName,
    source,
    generatedPath,
    tempDir,
  };
}

export function runLeanExecutionContractForManifest(manifest, options = {}) {
  const generated = writeExecutionContractLeanModuleForManifest(manifest, options);
  try {
    if (options.check === false) {
      return {
        ok: true,
        toolchainRef: null,
        generatedPath: generated.generatedPath,
        moduleName: generated.moduleName,
        facts: generated.facts,
      };
    }
    const result = runLeanCheck({
      sourcePath: generated.generatedPath,
      rootDir: generated.rootDir,
    });
    return {
      ...result,
      generatedPath: generated.generatedPath,
      moduleName: generated.moduleName,
      facts: generated.facts,
    };
  } finally {
    if (!options.emitPath && generated.tempDir) {
      fs.rmSync(generated.tempDir, { recursive: true, force: true });
    }
  }
}
