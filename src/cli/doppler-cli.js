#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runNodeCommand } from '../tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../tooling/node-browser-command-runner.js';
import { writeProgramBundle } from '../tooling/program-bundle.js';
import {
  TOOLING_COMMANDS,
  normalizeToolingCommandRequest,
} from '../tooling/command-api.js';
import { createToolingErrorEnvelope } from '../tooling/command-envelope.js';
import {
  asStringOrNull,
  resolveBrowserModelUrl,
  resolveNodeModelUrl,
  resolveStaticRootDir,
  resolveRdrrRoot,
} from './cli-model-resolution.js';
import { isPlainObject } from '../utils/plain-object.js';
import {
  toSummary,
  formatNumber,
  formatMs,
  saveBenchResult,
  loadBaseline,
  compareBenchResults,
  printManifestSummary,
  printDeviceInfo,
  printConvertContractSummary,
  printConvertReportSummary,
  printMetricsSummary,
} from './cli-output.js';
import { formatRuntimeProfiles, listRuntimeProfiles } from './runtime-profiles.js';
import { persistBrowserRelayReport } from './browser-report-output.js';
import { isAbsoluteUrl, normalizeModelUrl, parseJsonObjectFlag, performIntake, readJsonObjectFile, readJsonObjectInput, runBoundaryCommand, runBundleCommand, runCommandOnSurface } from './commands/bundle.js';
import { DEFAULT_CLI_POLICY, createCliToolingErrorEnvelope, parseRuntimeConfigUrl, resolveConfigEnvelope, resolveSurfaceForCommand, runManifestSweep, runWithAutoSurface } from './output/manifest.js';
import {
  parseCliArguments,
  validateBoundaryFlags,
  validateBundleFlags,
  validateCommandFlags,
  validateIntakeFlags,
  validateOnboardFlags,
  validateProfilesFlags,
  validateProgramBundleFlags,
} from './argument-contract.js';
export { createCliToolingErrorEnvelope } from './output/manifest.js';
export { checkCapturePrecondition, finalizeCliCommandResponse, performIntake } from './commands/bundle.js';

export { resolveBrowserModelUrl, resolveNodeModelUrl } from './cli-model-resolution.js';

const CLI_POLICY_PATH = fileURLToPath(new URL('./config/doppler-cli-policy.json', import.meta.url));

function usage() {
  return [
    'Usage:',
    '  doppler convert --config <path|url|json> [--surface auto|node]',
    '  doppler refresh-integrity --config <path|url|json> [--surface auto|node]',
    '  doppler debug --config <path|url|json> [--runtime-profile <id>|--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler bench --config <path|url|json> [--runtime-profile <id>|--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler verify --config <path|url|json> [--runtime-profile <id>|--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler diagnose --config <path|url|json> [--runtime-profile <id>|--runtime-config <path|url|json>] [--surface auto|node]',
    '  doppler profiles [--json|--pretty]',
    '  doppler lora --config <path|url|json> [--surface auto|node]',
    '  doppler distill --config <path|url|json> [--surface auto|node]',
    '  doppler program-bundle --config <path|json>',
    '  doppler program-bundle --manifest <path> --reference-report <path> --out <path> [--conversion-config <path>]',
    '  doppler onboard inspect --source <checkpoint-dir> --out <dir> [--family-intake <source-intake.json>]',
    '  doppler boundary capture --report <diagnose-report.json> --out <runtime-boundaries.json>',
    '  doppler boundary source-pack --provider-capture <provider-capture.json> --out <source-boundaries.json>',
    '  doppler boundary token-evidence --reference-transcript <reference-transcript.json> --out <token-evidence.json>',
    '  doppler boundary compare --source-pack <source-boundaries.json> --runtime-capture <runtime-boundaries.json> --token-evidence <json> --out <receipt.json>',
    '  doppler intake --convert-config <path|json> [--manifest <path>] [--out <intake-report.json>]',
    '  doppler bundle --manifest <path> --out <dir> [--prompt <text>] [--max-tokens <n>] [--surface node|browser]',
    '  doppler bundle --convert-config <path|json> --out <dir>',
    '',
    'Flags:',
    '  --config <path|url|json>        Required command config payload (file path, URL, or JSON object string).',
    '  --runtime-config <value>        Compatibility runtime override alias (JSON object, URL, or file path).',
    '  --runtime-profile <id>          Convenience alias for request.runtimeProfile on harnessed commands.',
    '  --surface <auto|node|browser>   Optional execution surface override.',
    '  --json                          Explicitly print JSON output (default).',
    '  --pretty                        Print human-readable summary instead of JSON',
    '  --help, -h                      Show this help message',
    '',
    'Command Config Contract:',
    '  The config payload must be a JSON object and may include:',
    '    - request: tooling command request fields (workload, modelId, training fields, convertPayload, etc).',
    '      May also include `runtimeProfile`, `runtimeConfigUrl`, and `runtimeConfig`.',
    '      Unknown top-level keys are disallowed when `request` is used as the envelope key.',
    '    - run: CLI-only run controls (surface, browser options, and bench save/compare/manifest settings).',
    '    - runtimeProfile: optional runtime profile id.',
    '    - runtimeConfigUrl: optional runtime override URL or local JSON path.',
    '    - runtimeConfig: optional inline runtime override object.',
    '',
    'Example:',
    '  doppler verify --config \'{"request":{"workload":"inference","modelId":"gemma-3-270m-it-f16-af32"}}\'',
    '  doppler profiles --pretty',
    '  doppler refresh-integrity --config \'{"request":{"modelDir":"models/local/gemma-3-270m-it-q4k-ehf16-af32"}}\'',
    '  doppler verify --config \'{"request":{"workload":"inference","workloadType":"program-bundle","programBundlePath":"examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json"}}\'',
    '  doppler program-bundle --config \'{"manifestPath":"models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json","referenceReportPath":"tests/fixtures/reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-18T13-33-38.973Z.json","outputPath":"examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json"}\'',
  ].join('\n');
}

function parseUnifiedRuntimeConfig(value) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return null;
  if (normalized.startsWith('{')) {
    return {
      sourceFlag: '--runtime-config',
      runtimeProfile: null,
      runtimeConfigUrl: null,
      runtimeConfig: parseJsonObjectFlag(normalized, '--runtime-config'),
    };
  }
  if (isAbsoluteUrl(normalized)) {
    return {
      sourceFlag: '--runtime-config',
      runtimeProfile: null,
      runtimeConfigUrl: normalized,
      runtimeConfig: null,
    };
  }
  return {
    sourceFlag: '--runtime-config',
    runtimeProfile: null,
    runtimeConfigUrl: parseRuntimeConfigUrl(normalized),
    runtimeConfig: null,
  };
}

function parseRuntimeProfileFlag(value) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return null;
  if (isAbsoluteUrl(normalized)) {
    throw new Error('--runtime-profile must be a runtime profile id, not a URL. Use --runtime-config for URLs.');
  }
  const trimmed = normalized.replace(/\.json$/u, '');
  if (!trimmed) {
    throw new Error('--runtime-profile must be a non-empty runtime profile id.');
  }
  if (trimmed.startsWith('/') || trimmed.startsWith('./') || trimmed.startsWith('../')) {
    throw new Error('--runtime-profile must be a runtime profile id, not a file path. Use --runtime-config for files.');
  }
  return trimmed.includes('/') ? trimmed : `profiles/${trimmed}`;
}

function resolveRuntimeConfigFlags(parsed) {
  const runtimeProfileRaw = asStringOrNull(parsed.flags['runtime-profile']);
  const unifiedRaw = asStringOrNull(parsed.flags['runtime-config']);
  if (runtimeProfileRaw !== null && unifiedRaw !== null) {
    throw new Error('--runtime-profile cannot be combined with --runtime-config.');
  }
  if (runtimeProfileRaw !== null) {
    return {
      sourceFlag: '--runtime-profile',
      runtimeProfile: parseRuntimeProfileFlag(runtimeProfileRaw),
      runtimeConfigUrl: null,
      runtimeConfig: null,
    };
  }
  return parseUnifiedRuntimeConfig(unifiedRaw);
}

async function buildProgramBundleOptions(parsed) {
  const repoRoot = path.resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const configValue = asStringOrNull(parsed.flags.config);
  const config = configValue
    ? await readJsonObjectInput(configValue, '--config')
    : {};
  return {
    repoRoot,
    ...config,
    manifestPath: parsed.flags.manifest ?? config.manifestPath ?? null,
    modelDir: parsed.flags['model-dir'] ?? config.modelDir ?? null,
    referenceReportPath: parsed.flags['reference-report'] ?? config.referenceReportPath ?? null,
    conversionConfigPath: parsed.flags['conversion-config'] ?? config.conversionConfigPath ?? null,
    runtimeConfigPath: parsed.flags['runtime-config'] ?? config.runtimeConfigPath ?? null,
    outputPath: parsed.flags.out ?? config.outputPath ?? config.out ?? null,
    bundleId: parsed.flags['bundle-id'] ?? config.bundleId ?? null,
    createdAtUtc: parsed.flags['created-at'] ?? config.createdAtUtc ?? null,
  };
}

async function runProgramBundleCommand(parsed, jsonOutput) {
  const result = await writeProgramBundle(await buildProgramBundleOptions(parsed));
  const summary = {
    ok: true,
    outputPath: path.relative(process.cwd(), result.outputPath),
    modelId: result.bundle.modelId,
    bundleId: result.bundle.bundleId,
    executionGraphHash: result.bundle.sources.executionGraph.hash,
    artifactCount: result.bundle.artifacts.length,
    wgslModuleCount: result.bundle.wgslModules.length,
  };
  if (jsonOutput) {
    console.log(JSON.stringify(summary, null, 2));
    return;
  }
  console.log(
    `[ok] wrote ${summary.outputPath} ` +
    `(modelId=${summary.modelId}, artifacts=${summary.artifactCount}, wgsl=${summary.wgslModuleCount})`
  );
}

async function runIntakeCommand(parsed, jsonOutput) {
  const fsMod = await import('node:fs/promises');
  const { report } = await performIntake({
    convertConfigValue: asStringOrNull(parsed.flags['convert-config']),
    manifestFlag: asStringOrNull(parsed.flags.manifest),
    modelDir: asStringOrNull(parsed.flags['model-dir']),
    skipConvert: parsed.flags['skip-convert'] === true
      || String(parsed.flags['skip-convert'] ?? '').toLowerCase() === 'true',
  });

  const outputPath = asStringOrNull(parsed.flags.out);
  if (outputPath) {
    const resolved = path.resolve(outputPath);
    await fsMod.mkdir(path.dirname(resolved), { recursive: true });
    await fsMod.writeFile(resolved, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
    report.outputPath = path.relative(process.cwd(), resolved);
  }

  if (jsonOutput) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(`[ok] intake passed (${report.stages.length} stages, 0 blockers)`);
  } else {
    console.log(`[fail] intake found ${report.blockers.length} blocker(s):`);
    for (const b of report.blockers) {
      console.log(`  - [${b.stage}] ${b.code}: ${b.message}`);
    }
  }

  if (!report.ok) {
    process.exitCode = 1;
  }
}

async function runOnboardInspectCommand(parsed, jsonOutput) {
  const { inspectSourceModel } = await import('../tooling/source-intake.js');
  const policyPath = fileURLToPath(
    new URL('../config/evidence/source-intake-policy.json', import.meta.url)
  );
  const policy = await readJsonObjectFile(policyPath, 'source intake policy');
  const familyIntakePath = asStringOrNull(parsed.flags['family-intake']);
  const familyIntake = familyIntakePath
    ? await readJsonObjectFile(path.resolve(familyIntakePath), '--family-intake')
    : null;
  const result = await inspectSourceModel({
    sourceDir: path.resolve(parsed.flags.source),
    policy,
    familyIntake,
  });
  const outputDir = path.resolve(parsed.flags.out);
  await fs.mkdir(outputDir, { recursive: true });
  const outputs = {
    intake: path.join(outputDir, 'source-intake.json'),
    conversion: path.join(outputDir, 'conversion-config.skeleton.json'),
    contractTests: path.join(outputDir, 'contract-tests.plan.json'),
  };
  await Promise.all([
    fs.writeFile(outputs.intake, `${JSON.stringify(result.report, null, 2)}\n`, 'utf8'),
    fs.writeFile(outputs.conversion, `${JSON.stringify(result.artifacts.conversion, null, 2)}\n`, 'utf8'),
    fs.writeFile(outputs.contractTests, `${JSON.stringify(result.artifacts.contractTests, null, 2)}\n`, 'utf8'),
  ]);
  const summary = {
    ...result.report,
    outputPaths: Object.fromEntries(
      Object.entries(outputs).map(([key, value]) => [key, path.relative(process.cwd(), value)])
    ),
  };
  if (jsonOutput) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    const status = result.report.ok ? 'ok' : 'review';
    console.log(
      `[${status}] source intake wrote ${path.relative(process.cwd(), outputDir)} ` +
      `(${result.report.summary.accepted} accepted, ${result.report.summary.unresolved} unresolved)`
    );
  }
  if (!result.report.ok) process.exitCode = 1;
}

async function runProfilesCommand(jsonOutput) {
  const result = await listRuntimeProfiles();
  if (jsonOutput) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }
  console.log(formatRuntimeProfiles(result));
}

function resolveCommandConfigFlag(parsed) {
  const config = asStringOrNull(parsed.flags.config);
  if (!config) {
    throw new Error('command requires --config <path|url|json>.');
  }
  return config;
}

function applyRuntimeFlagOverride(requestInput, runtimeOverride) {
  if (!runtimeOverride) {
    return;
  }
  const sourceFlag = runtimeOverride.sourceFlag || '--runtime-config';
  const hasInlineRuntime = (
    requestInput.runtimeProfile != null
    || requestInput.runtimeConfigUrl != null
    || requestInput.runtimeConfig != null
  );
  if (hasInlineRuntime) {
    throw new Error(
      `${sourceFlag} cannot be combined with runtimeProfile/runtimeConfigUrl/runtimeConfig values inside --config.`
    );
  }
  requestInput.runtimeProfile = runtimeOverride.runtimeProfile;
  requestInput.runtimeConfigUrl = runtimeOverride.runtimeConfigUrl;
  requestInput.runtimeConfig = runtimeOverride.runtimeConfig;
}

function resolveBenchRunOptions(runConfig, policy = DEFAULT_CLI_POLICY) {
  const benchConfig = isPlainObject(runConfig?.bench) ? runConfig.bench : {};
  const configuredSaveDir = asStringOrNull(benchConfig.saveDir);
  const defaultSaveDir = asStringOrNull(policy?.defaults?.benchmark?.saveDir);
  return {
    shouldSave: benchConfig.save === true,
    saveDir: configuredSaveDir === null
      ? defaultSaveDir || './benchmarks/vendors/results'
      : configuredSaveDir,
    comparePath: asStringOrNull(benchConfig.compare),
    manifestPath: asStringOrNull(benchConfig.manifest),
  };
}

export async function buildRequest(parsed, policy = DEFAULT_CLI_POLICY) {
  const command = parsed.command;
  if (!command || !TOOLING_COMMANDS.includes(command)) {
    throw new Error(`Unsupported command "${command || ''}"`);
  }

  const configValue = resolveCommandConfigFlag(parsed);
  const configPayload = await readJsonObjectInput(configValue, '--config');
  const envelope = resolveConfigEnvelope(configPayload);
  const runtimeOverride = resolveRuntimeConfigFlags(parsed);

  const requestInput = { ...envelope.request };
  if (requestInput.command != null && requestInput.command !== command) {
    throw new Error(
      `--config request command mismatch: CLI command is "${command}" but config command is "${requestInput.command}".`
    );
  }
  requestInput.command = command;
  if (requestInput.modelUrl != null) {
    requestInput.modelUrl = normalizeModelUrl(requestInput.modelUrl);
  }

  applyRuntimeFlagOverride(requestInput, runtimeOverride);

  const surfaceFromCli = asStringOrNull(parsed.flags.surface) !== null;
  const surface = resolveSurfaceForCommand(command, parsed, envelope.run, policy);

  return {
    request: normalizeToolingCommandRequest(requestInput),
    runConfig: envelope.run,
    surface,
    surfaceFromCli,
    benchRunOptions: resolveBenchRunOptions(envelope.run, policy),
  };
}

export async function withJsonStdoutIsolation(enabled, callback) {
  if (!enabled) {
    return callback();
  }

  const previousConsole = {
    log: console.log,
    info: console.info,
    debug: console.debug,
    warn: console.warn,
    error: console.error,
  };
  const writeDiagnostic = (...args) => previousConsole.error(...args);

  console.log = writeDiagnostic;
  console.info = writeDiagnostic;
  console.debug = writeDiagnostic;
  console.warn = writeDiagnostic;

  try {
    return await callback();
  } finally {
    console.log = previousConsole.log;
    console.info = previousConsole.info;
    console.debug = previousConsole.debug;
    console.warn = previousConsole.warn;
  }
}

async function loadManifest(manifestPath) {
  const raw = await fs.readFile(path.resolve(manifestPath), 'utf-8');
  const manifest = JSON.parse(raw);
  if (!manifest.runs || !Array.isArray(manifest.runs) || manifest.runs.length === 0) {
    throw new Error('manifest must have a non-empty "runs" array');
  }
  return manifest;
}

async function main() {
  const argv = process.argv.slice(2);
  const jsonOutputRequested = !argv.includes('--pretty');
  const errorContext = {
    surface: null,
    request: null,
  };

  try {
    if (!argv.length || argv[0] === '--help' || argv[0] === '-h') {
      console.log(usage());
      return;
    }

    const cliPolicy = await readJsonObjectFile(CLI_POLICY_PATH, '--cli-policy');
    const parsed = parseCliArguments(argv);
    if (parsed.flags.help === true || parsed.flags.h === true) {
      console.log(usage());
      return;
    }
    if (parsed.command === 'program-bundle') {
      validateProgramBundleFlags(parsed);
      await runProgramBundleCommand(parsed, parsed.flags.pretty !== true);
      return;
    }
    if (parsed.command === 'profiles') {
      validateProfilesFlags(parsed);
      await runProfilesCommand(parsed.flags.pretty !== true);
      return;
    }
    if (parsed.command === 'onboard') {
      validateOnboardFlags(parsed);
      await runOnboardInspectCommand(parsed, parsed.flags.pretty !== true);
      return;
    }
    if (parsed.command === 'boundary') {
      validateBoundaryFlags(parsed);
      await runBoundaryCommand(parsed, parsed.flags.pretty !== true);
      return;
    }
    if (parsed.command === 'intake') {
      validateIntakeFlags(parsed);
      await runIntakeCommand(parsed, parsed.flags.pretty !== true);
      return;
    }
    if (parsed.command === 'bundle') {
      validateBundleFlags(parsed);
      await runBundleCommand(parsed, parsed.flags.pretty !== true);
      return;
    }
    validateCommandFlags(parsed);

    const jsonOutput = parsed.flags.pretty !== true;
    const commandContext = await buildRequest(parsed, cliPolicy);
    const { request, runConfig, surface, surfaceFromCli, benchRunOptions } = commandContext;
    errorContext.surface = surface === 'auto' ? null : surface;
    errorContext.request = request;
    const { saveDir, shouldSave, comparePath, manifestPath } = benchRunOptions;

    if (manifestPath) {
      const manifest = await loadManifest(String(manifestPath));
      const results = await withJsonStdoutIsolation(jsonOutput, async () => {
        const sweepResults = await runManifestSweep(
          manifest,
          {
            request,
            runConfig,
            surface,
            surfaceFromCli,
          },
          jsonOutput,
          cliPolicy
        );

        if (shouldSave) {
          for (const r of sweepResults) {
            if (r.response?.result) {
              const savedPath = await saveBenchResult(r.response.result, saveDir);
              if (!jsonOutput) console.error(`[save] ${r.label}: ${savedPath}`);
            }
          }
        }

        return sweepResults;
      });

      if (jsonOutput) {
        console.log(JSON.stringify(results.map((r) => r.response ?? r.error), null, 2));
        return;
      }

      printManifestSummary(results);
      for (const r of results) {
        if (r.response?.result) {
          console.log(`\n--- ${r.label} ---`);
          printMetricsSummary(r.response.result);
        }
      }
      return;
    }

    const response = await withJsonStdoutIsolation(jsonOutput, async () => {
      const commandResponse = surface === 'auto'
        ? await runWithAutoSurface(request, runConfig, jsonOutput, cliPolicy)
        : await runCommandOnSurface(request, surface, runConfig, jsonOutput);
      const isBench = commandResponse.result?.suite === 'bench';

      if (comparePath && isBench) {
        const baseline = await loadBaseline(String(comparePath), saveDir);
        if (baseline) {
          compareBenchResults(commandResponse.result, baseline);
        }
      }

      if (shouldSave && isBench) {
        const savedPath = await saveBenchResult(commandResponse.result, saveDir);
        if (!jsonOutput) {
          console.error(`[save] ${savedPath}`);
        }
      }

      return commandResponse;
    });

    if (jsonOutput) {
      const output = response?.result?.report !== undefined
        ? { ...response, result: { ...response.result, report: undefined } }
        : response;
      console.log(JSON.stringify(output, null, 2));
      return;
    }

    console.log(`[ok] ${toSummary(response.result)}`);
    printConvertContractSummary(response.result);
    printConvertReportSummary(response.result);
    printMetricsSummary(response.result);
  } catch (error) {
    if (jsonOutputRequested) {
      console.log(JSON.stringify(createCliToolingErrorEnvelope(error, errorContext), null, 2));
      process.exitCode = 1;
      return;
    }
    throw error;
  }
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  if (!entryPath) {
    return false;
  }
  return path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

function flushStream(stream) {
  if (!stream || stream.destroyed) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    stream.write('', () => resolve());
  });
}

async function flushCliStreams() {
  await Promise.all([
    flushStream(process.stdout),
    flushStream(process.stderr),
  ]);
}

if (isMainModule(import.meta.url)) {
  main()
    .then(async () => {
      const exitCode = process.exitCode ?? 0;
      await flushCliStreams();
      process.exit(exitCode);
    })
    .catch(async (error) => {
      console.error(`[error] ${error?.message || String(error)}`);
      await flushCliStreams();
      process.exit(1);
    });
}
