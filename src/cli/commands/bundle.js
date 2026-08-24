import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runNodeCommand } from '../../tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../../tooling/node-browser-command-runner.js';
import { writeProgramBundle } from '../../tooling/program-bundle.js';
import {
  TOOLING_COMMANDS,
  normalizeToolingCommandRequest,
} from '../../tooling/command-api.js';
import { createToolingErrorEnvelope } from '../../tooling/command-envelope.js';
import {
  asStringOrNull,
  resolveBrowserModelUrl,
  resolveNodeModelUrl,
  resolveStaticRootDir,
  resolveRdrrRoot,
} from '../cli-model-resolution.js';
import { isPlainObject } from '../../formats/plain-object.js';
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
} from '../cli-output.js';
import { formatRuntimeProfiles, listRuntimeProfiles } from '../runtime-profiles.js';
import { persistBrowserRelayReport } from '../browser-report-output.js';

export function parseJsonObjectFlag(value, label) {
  if (asStringOrNull(value) === null) return null;
  try {
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('value must be a JSON object');
    }
    return parsed;
  } catch (error) {
    throw new Error(`Invalid ${label}: ${error.message}`);
  }
}

export function isAbsoluteUrl(value) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return false;
  try {
    const parsed = new URL(normalized);
    return typeof parsed.protocol === 'string' && parsed.protocol.length > 0;
  } catch {
    return false;
  }
}

export async function readJsonObjectFile(filePath, label) {
  const resolved = path.resolve(String(filePath));
  let raw;
  try {
    raw = await fs.readFile(resolved, 'utf8');
  } catch (error) {
    throw new Error(`${label} not found or unreadable: ${resolved}`);
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} must contain valid JSON: ${error.message}`);
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return parsed;
}

export async function readJsonObjectUrl(rawUrl, label) {
  let response;
  try {
    response = await fetch(rawUrl, {
      headers: {
        Connection: 'close',
      },
      redirect: 'follow',
      signal: AbortSignal.timeout(30000),
    });
  } catch (error) {
    throw new Error(`${label} URL request failed: ${error?.message || String(error)}`);
  }
  if (!response.ok) {
    throw new Error(`${label} URL request failed: HTTP ${response.status}`);
  }
  let raw;
  try {
    raw = await response.text();
  } catch (error) {
    throw new Error(`${label} URL request failed: ${error?.message || String(error)}`);
  }
  return parseJsonObjectFlag(raw, label);
}

export async function readJsonObjectInput(inputValue, label) {
  const normalized = asStringOrNull(inputValue);
  if (normalized === null) {
    throw new Error(`${label} is required.`);
  }
  if (normalized.startsWith('{')) {
    return parseJsonObjectFlag(normalized, label);
  }
  if (isAbsoluteUrl(normalized)) {
    return readJsonObjectUrl(normalized, label);
  }
  return readJsonObjectFile(normalized, label);
}

export const INTAKE_SCHEMA_ID = 'doppler.intake-report/v1';

export async function performIntake({
  convertConfigValue = null,
  manifestFlag = null,
  modelDir = null,
  skipConvert = false,
} = {}) {
  const { extractExecutionContractFacts, validateExecutionContractFacts } = await import(
    '../../config/execution-contract-check.js'
  );
  const fsMod = await import('node:fs/promises');

  const stages = [];
  const blockers = [];
  let manifest = null;

  let manifestPath = null;
  if (manifestFlag) {
    manifestPath = path.resolve(manifestFlag);
  } else if (modelDir) {
    manifestPath = path.resolve(modelDir, 'manifest.json');
  }

  if (convertConfigValue && !skipConvert) {
    stages.push({ stage: 'convert', status: 'starting' });
    try {
      const convertConfig = await readJsonObjectInput(convertConfigValue, '--convert-config');
      const request = normalizeToolingCommandRequest({
        command: 'convert',
        convertPayload: convertConfig,
      });
      const response = await runCommandOnSurface(request, 'node', {}, true);
      stages[stages.length - 1] = {
        stage: 'convert',
        status: 'succeeded',
        convertReport: response?.result?.report ?? null,
      };
      if (!manifestPath) {
        const inferredModelBaseId =
          convertConfig?.output?.modelBaseId
          || convertConfig?.converterConfig?.output?.modelBaseId
          || null;
        const inferredBaseDir =
          convertConfig?.output?.baseDir
          || convertConfig?.converterConfig?.output?.baseDir
          || 'models/local';
        if (inferredModelBaseId) {
          manifestPath = path.resolve(inferredBaseDir, inferredModelBaseId, 'manifest.json');
        }
      }
    } catch (error) {
      stages[stages.length - 1] = {
        stage: 'convert',
        status: 'failed',
        error: error?.message || String(error),
      };
      blockers.push({
        stage: 'convert',
        code: error?.code || 'convert_failed',
        message: error?.message || String(error),
      });
    }
  } else {
    stages.push({ stage: 'convert', status: skipConvert ? 'skipped' : 'not_requested' });
  }

  if (blockers.length === 0 && manifestPath) {
    stages.push({ stage: 'manifest_load', status: 'starting' });
    try {
      const raw = await fsMod.readFile(manifestPath, 'utf8');
      manifest = JSON.parse(raw);
      stages[stages.length - 1] = {
        stage: 'manifest_load',
        status: 'succeeded',
        manifestPath,
        modelId: manifest?.modelId ?? null,
      };

      stages.push({ stage: 'execution_contract_check', status: 'starting' });
      try {
        const facts = extractExecutionContractFacts(manifest);
        const validation = validateExecutionContractFacts(facts);
        stages[stages.length - 1] = {
          stage: 'execution_contract_check',
          status: 'succeeded',
          stepCount: facts.steps.length,
          checks: Array.isArray(validation?.checks) ? validation.checks.length : null,
        };
      } catch (error) {
        stages[stages.length - 1] = {
          stage: 'execution_contract_check',
          status: 'failed',
          error: error?.message || String(error),
        };
        blockers.push({
          stage: 'execution_contract_check',
          code: 'unsupported_op_or_contract_violation',
          message: error?.message || String(error),
        });
      }
    } catch (error) {
      stages[stages.length - 1] = {
        stage: 'manifest_load',
        status: 'failed',
        error: error?.message || String(error),
      };
      blockers.push({
        stage: 'manifest_load',
        code: 'manifest_unreadable',
        message: error?.message || String(error),
      });
    }
  } else if (blockers.length === 0) {
    stages.push({ stage: 'manifest_load', status: 'skipped' });
    blockers.push({
      stage: 'manifest_load',
      code: 'no_manifest_path',
      message:
        'intake: no manifest path resolved. Provide --manifest, --model-dir, or a convert-config with output.modelBaseId.',
    });
  }

  const report = {
    schema: INTAKE_SCHEMA_ID,
    ok: blockers.length === 0,
    blockers,
    stages,
    createdAtUtc: new Date().toISOString(),
  };

  return { report, manifestPath, manifest };
}

export async function runBoundaryCommand(parsed, jsonOutput) {
  const {
    buildDeterministicTokenEvidenceFromReferenceTranscript,
    buildRuntimeBoundaryCapture,
    buildSourceBoundaryPackFromProviderCapture,
    compareBoundaryEvidence,
  } = await import('../../tooling/boundary-evidence.js');
  const policyPath = fileURLToPath(
    new URL('../../config/evidence/boundary-evidence-policy.json', import.meta.url)
  );
  const policy = await readJsonObjectFile(policyPath, 'boundary evidence policy');
  let result;
  if (parsed.action === 'capture') {
    const report = await readJsonObjectFile(
      path.resolve(parsed.flags.report),
      '--report'
    );
    result = buildRuntimeBoundaryCapture({
      report,
      policy,
      tolerancePolicyId: asStringOrNull(parsed.flags['tolerance-policy'])
        ?? 'doppler.boundary-tolerance/source-f16-v1',
    });
  } else if (parsed.action === 'source-pack') {
    const providerCapture = await readJsonObjectFile(
      path.resolve(parsed.flags['provider-capture']),
      '--provider-capture'
    );
    result = buildSourceBoundaryPackFromProviderCapture(providerCapture);
  } else if (parsed.action === 'token-evidence') {
    const transcript = await readJsonObjectFile(
      path.resolve(parsed.flags['reference-transcript']),
      '--reference-transcript'
    );
    result = buildDeterministicTokenEvidenceFromReferenceTranscript(transcript);
  } else {
    const [sourcePack, runtimeCapture, tokenEvidence] = await Promise.all([
      readJsonObjectFile(path.resolve(parsed.flags['source-pack']), '--source-pack'),
      readJsonObjectFile(path.resolve(parsed.flags['runtime-capture']), '--runtime-capture'),
      readJsonObjectFile(path.resolve(parsed.flags['token-evidence']), '--token-evidence'),
    ]);
    const sourceControlPath = asStringOrNull(parsed.flags['source-control']);
    const sourcePrecisionControlReceipt = sourceControlPath
      ? await readJsonObjectFile(path.resolve(sourceControlPath), '--source-control')
      : null;
    result = compareBoundaryEvidence({
      sourcePack,
      runtimeCapture,
      policy,
      artifactPrecision: asStringOrNull(parsed.flags['artifact-precision']) ?? 'source',
      sourcePrecisionControlReceipt,
      deterministicTokenEvidence: tokenEvidence,
    });
  }
  const outputPath = path.resolve(parsed.flags.out);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  if (jsonOutput) {
    console.log(JSON.stringify({ ...result, outputPath: path.relative(process.cwd(), outputPath) }, null, 2));
  } else if (parsed.action === 'capture') {
    console.log(`[ok] captured ${result.boundaries.length} semantic boundaries`);
  } else if (parsed.action === 'source-pack') {
    console.log(`[ok] source pack contains ${result.boundaries.length} semantic boundaries`);
  } else if (parsed.action === 'token-evidence') {
    const status = result.exact ? 'ok' : 'fail';
    console.log(`[${status}] token evidence covers ${result.tokenCount} tokens`);
  } else {
    const status = result.promotionGate.passed ? 'ok' : 'fail';
    console.log(
      `[${status}] boundary comparison ` +
      `(first divergence: ${result.firstDivergence?.boundaryId ?? 'none'})`
    );
  }
  if (
    (parsed.action === 'compare' && !result.promotionGate.passed)
    || (parsed.action === 'token-evidence' && !result.exact)
  ) {
    process.exitCode = 1;
  }
}

export const BUNDLE_SUMMARY_SCHEMA_ID = 'doppler.bundle-summary/v1';

export const DEFAULT_BUNDLE_PROMPT = 'The color of the sky is';

export const DEFAULT_BUNDLE_MAX_TOKENS = 8;

export async function checkCapturePrecondition(surface) {
  const skipCaptureHint = 'Alternatively, use --skip-capture with pre-existing --reference-report and --reference-transcript paths.';
  if (surface === 'node') {
    const { bootstrapNodeWebGPU } = await import('../../tooling/node-webgpu.js');
    const { hasNodeWebGPUSupport } = await import('../../tooling/node-command-runner.js');
    const bootstrap = await bootstrapNodeWebGPU();
    if (bootstrap.ok && hasNodeWebGPUSupport()) {
      return { ok: true, surface, provider: bootstrap.provider ?? null };
    }
    return {
      ok: false,
      surface,
      code: 'capture_precondition_node_webgpu_unavailable',
      provider: bootstrap.provider ?? null,
      detail: bootstrap.detail ?? null,
      message:
        'bundle: --surface node requires a working Node WebGPU binding. '
        + `Bootstrap attempt failed: ${bootstrap.detail ?? 'no provider produced a usable adapter'}`
        + '. Install a Node WebGPU binding (for example `npm install webgpu`) '
        + 'or set DOPPLER_NODE_WEBGPU_MODULE to a custom specifier; '
        + 'alternatively use --surface browser after `npx playwright install chromium`. '
        + skipCaptureHint,
    };
  }
  if (surface === 'browser') {
    let playwright = null;
    try {
      playwright = await import('playwright');
    } catch (error) {
      return {
        ok: false,
        surface,
        code: 'capture_precondition_playwright_missing',
        detail: error?.message ?? null,
        message:
          'bundle: --surface browser requires the `playwright` package. '
          + 'Run `npm install --save-dev playwright` then `npx playwright install chromium`. '
          + 'Alternatively use --surface node after installing a Node WebGPU binding (e.g. `npm install webgpu`). '
          + skipCaptureHint,
      };
    }
    let executablePath = null;
    try {
      executablePath = playwright.chromium?.executablePath?.() ?? null;
    } catch (error) {
      return {
        ok: false,
        surface,
        code: 'capture_precondition_playwright_chromium_missing',
        detail: error?.message ?? null,
        message:
          'bundle: Playwright is installed but Chromium binaries are missing. '
          + 'Run `npx playwright install chromium` to download them. '
          + skipCaptureHint,
      };
    }
    const fsSync = await import('node:fs');
    if (!executablePath || !fsSync.existsSync(executablePath)) {
      return {
        ok: false,
        surface,
        code: 'capture_precondition_playwright_chromium_missing',
        detail: executablePath ? `executable not found at ${executablePath}` : 'executablePath returned null',
        message:
          'bundle: Playwright Chromium binaries are not installed. '
          + 'Run `npx playwright install chromium` to download them. '
          + skipCaptureHint,
      };
    }
    return { ok: true, surface, executablePath };
  }
  return {
    ok: false,
    surface,
    code: 'capture_precondition_unsupported_surface',
    detail: null,
    message: `bundle: --surface must be 'browser' or 'node', got '${surface}'.`,
  };
}

export async function runBundleCommand(parsed, jsonOutput) {
  const {
    runReferenceCapture,
    extractReferenceReport,
    extractReferenceTranscriptSeed,
    writeReferenceReport,
    writeReferenceTranscript,
    normalizeModelUrl,
  } = await import('../../tooling/reference-verify.js');
  const { writeProgramBundle } = await import('../../tooling/program-bundle.js');
  const fsMod = await import('node:fs/promises');

  const outDir = asStringOrNull(parsed.flags.out);
  if (!outDir) {
    throw new Error('bundle: --out <dir> is required.');
  }
  const skipCapture = parsed.flags['skip-capture'] === true
    || String(parsed.flags['skip-capture'] ?? '').toLowerCase() === 'true';
  const resolvedOut = path.resolve(outDir);
  await fsMod.mkdir(resolvedOut, { recursive: true });

  const repoRoot = path.resolve(fileURLToPath(new URL('../../..', import.meta.url)));
  const stages = [];
  const blockers = [];
  const artifactPaths = {};

  stages.push({ stage: 'intake', status: 'starting' });
  const intakeResult = await performIntake({
    convertConfigValue: asStringOrNull(parsed.flags['convert-config']),
    manifestFlag: asStringOrNull(parsed.flags.manifest),
    modelDir: asStringOrNull(parsed.flags['model-dir']),
    skipConvert: parsed.flags['skip-convert'] === true
      || String(parsed.flags['skip-convert'] ?? '').toLowerCase() === 'true',
  });
  const intakeReportPath = path.join(resolvedOut, 'intake-report.json');
  await fsMod.writeFile(intakeReportPath, `${JSON.stringify(intakeResult.report, null, 2)}\n`, 'utf8');
  artifactPaths.intakeReport = path.relative(process.cwd(), intakeReportPath);
  if (!intakeResult.report.ok) {
    stages[stages.length - 1] = { stage: 'intake', status: 'failed', blockers: intakeResult.report.blockers };
    blockers.push(...intakeResult.report.blockers.map((b) => ({ ...b, stage: `intake:${b.stage}` })));
    return emitBundleSummary({ ok: false, stages, blockers, artifactPaths, jsonOutput, resolvedOut });
  }
  stages[stages.length - 1] = {
    stage: 'intake',
    status: 'succeeded',
    manifestPath: intakeResult.manifestPath,
    modelId: intakeResult.manifest?.modelId ?? null,
  };

  const manifestPath = intakeResult.manifestPath;
  const manifest = intakeResult.manifest;
  const modelDirResolved = asStringOrNull(parsed.flags['model-dir'])
    ? path.resolve(asStringOrNull(parsed.flags['model-dir']))
    : path.dirname(manifestPath);
  const modelId = manifest?.modelId ?? null;
  if (!modelId) {
    blockers.push({ stage: 'bundle', code: 'missing_model_id', message: 'manifest.modelId is required for bundle composition.' });
    return emitBundleSummary({ ok: false, stages, blockers, artifactPaths, jsonOutput, resolvedOut });
  }

  const referenceReportPath = path.join(resolvedOut, 'reference-report.json');
  const referenceTranscriptPath = path.join(resolvedOut, 'reference-transcript.json');

  if (!skipCapture) {
    const surface = asStringOrNull(parsed.flags.surface) ?? 'browser';
    stages.push({ stage: 'capture', status: 'starting', surface });
    const precondition = await checkCapturePrecondition(surface);
    if (!precondition.ok) {
      stages[stages.length - 1] = {
        stage: 'capture',
        status: 'blocked',
        surface,
        precondition: {
          code: precondition.code,
          detail: precondition.detail ?? null,
          provider: precondition.provider ?? null,
        },
      };
      blockers.push({
        stage: 'capture',
        code: precondition.code,
        message: precondition.message,
        surface,
        detail: precondition.detail ?? null,
        provider: precondition.provider ?? null,
      });
      return emitBundleSummary({ ok: false, stages, blockers, artifactPaths, jsonOutput, resolvedOut });
    }
    try {
      const modelUrl = normalizeModelUrl(asStringOrNull(parsed.flags['model-url']), modelDirResolved);
      const maxTokensRaw = asStringOrNull(parsed.flags['max-tokens']);
      const maxTokens = maxTokensRaw == null ? DEFAULT_BUNDLE_MAX_TOKENS : Number(maxTokensRaw);
      if (!Number.isFinite(maxTokens) || maxTokens <= 0) {
        throw new Error('--max-tokens must be a positive number.');
      }
      const response = await runReferenceCapture({
        manifest,
        modelId,
        modelUrl,
        surface,
        prompt: asStringOrNull(parsed.flags.prompt) ?? DEFAULT_BUNDLE_PROMPT,
        maxTokens,
        runtimeConfig: asStringOrNull(parsed.flags['runtime-config']),
        repoRoot,
      });
      const report = extractReferenceReport(response);
      await writeReferenceReport(report, referenceReportPath);
      artifactPaths.referenceReport = path.relative(process.cwd(), referenceReportPath);
      const transcript = extractReferenceTranscriptSeed(report);
      await writeReferenceTranscript(transcript, referenceTranscriptPath);
      artifactPaths.referenceTranscript = path.relative(process.cwd(), referenceTranscriptPath);
      stages[stages.length - 1] = {
        stage: 'capture',
        status: 'succeeded',
        surface,
        tokensGenerated: transcript.output?.tokensGenerated ?? null,
        stopReason: transcript.output?.stopReason ?? null,
      };
    } catch (error) {
      stages[stages.length - 1] = {
        stage: 'capture',
        status: 'failed',
        error: error?.message || String(error),
      };
      blockers.push({ stage: 'capture', code: 'capture_failed', message: error?.message || String(error) });
      return emitBundleSummary({ ok: false, stages, blockers, artifactPaths, jsonOutput, resolvedOut });
    }
  } else {
    stages.push({ stage: 'capture', status: 'skipped' });
    const preExistingReport = asStringOrNull(parsed.flags['reference-report']);
    const preExistingTranscript = asStringOrNull(parsed.flags['reference-transcript']);
    if (preExistingReport) artifactPaths.referenceReport = preExistingReport;
    if (preExistingTranscript) artifactPaths.referenceTranscript = preExistingTranscript;
    if (!preExistingReport || !preExistingTranscript) {
      blockers.push({
        stage: 'capture',
        code: 'skip_capture_requires_artifacts',
        message: 'bundle: --skip-capture requires existing --reference-report and --reference-transcript paths.',
      });
      return emitBundleSummary({ ok: false, stages, blockers, artifactPaths, jsonOutput, resolvedOut });
    }
  }

  stages.push({ stage: 'bundle', status: 'starting' });
  try {
    const bundleOutputPath = path.join(resolvedOut, 'program-bundle.json');
    const bundleResult = await writeProgramBundle({
      repoRoot,
      manifestPath,
      modelDir: modelDirResolved,
      referenceReportPath: artifactPaths.referenceReport
        ? path.resolve(artifactPaths.referenceReport)
        : referenceReportPath,
      conversionConfigPath: asStringOrNull(parsed.flags['conversion-config']),
      outputPath: bundleOutputPath,
      bundleId: asStringOrNull(parsed.flags['bundle-id']),
      createdAtUtc: asStringOrNull(parsed.flags['created-at']),
    });
    artifactPaths.programBundle = path.relative(process.cwd(), bundleResult.outputPath);
    stages[stages.length - 1] = {
      stage: 'bundle',
      status: 'succeeded',
      bundleId: bundleResult.bundle.bundleId,
      executionGraphHash: bundleResult.bundle.sources.executionGraph.hash,
      artifactCount: bundleResult.bundle.artifacts.length,
      wgslModuleCount: bundleResult.bundle.wgslModules.length,
    };
  } catch (error) {
    stages[stages.length - 1] = {
      stage: 'bundle',
      status: 'failed',
      error: error?.message || String(error),
    };
    blockers.push({ stage: 'bundle', code: 'bundle_export_failed', message: error?.message || String(error) });
    return emitBundleSummary({ ok: false, stages, blockers, artifactPaths, jsonOutput, resolvedOut });
  }

  stages.push({ stage: 'parity', status: 'starting' });
  try {
    const { checkProgramBundleParity } = await import('../../tooling/program-bundle-parity.js');
    const parityOutputPath = path.join(resolvedOut, 'program-bundle-parity.json');
    const bundlePath = artifactPaths.programBundle
      ? path.resolve(artifactPaths.programBundle)
      : path.join(resolvedOut, 'program-bundle.json');
    const parityReport = await checkProgramBundleParity({
      bundlePath,
      mode: 'contract',
      providers: ['browser-webgpu', 'node:webgpu', 'node:doe-gpu'],
      repoRoot,
    });
    await fs.writeFile(parityOutputPath, `${JSON.stringify(parityReport, null, 2)}\n`, 'utf8');
    artifactPaths.programBundleParity = path.relative(process.cwd(), parityOutputPath);
    stages[stages.length - 1] = {
      stage: 'parity',
      status: 'succeeded',
      mode: parityReport.mode,
      ok: parityReport.ok,
      parityHash: parityReport.parityHash,
      providers: parityReport.providers.map((result) => ({
        provider: result.provider,
        status: result.status,
        schemaValid: result.schemaValid,
        providerAvailable: result.providerAvailable,
        executed: result.executed,
        transcriptMatched: result.transcriptMatched,
      })),
    };
  } catch (error) {
    stages[stages.length - 1] = {
      stage: 'parity',
      status: 'failed',
      error: error?.message || String(error),
    };
    blockers.push({ stage: 'parity', code: 'parity_check_failed', message: error?.message || String(error) });
    return emitBundleSummary({ ok: false, stages, blockers, artifactPaths, jsonOutput, resolvedOut });
  }

  return emitBundleSummary({ ok: true, stages, blockers, artifactPaths, jsonOutput, resolvedOut });
}

export async function emitBundleSummary({ ok, stages, blockers, artifactPaths, jsonOutput, resolvedOut }) {
  const fsMod = await import('node:fs/promises');
  const summary = {
    schema: BUNDLE_SUMMARY_SCHEMA_ID,
    ok,
    stages,
    blockers,
    artifactPaths,
    outputDir: path.relative(process.cwd(), resolvedOut),
    createdAtUtc: new Date().toISOString(),
  };
  const summaryPath = path.join(resolvedOut, 'bundle-summary.json');
  await fsMod.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  summary.summaryPath = path.relative(process.cwd(), summaryPath);

  if (jsonOutput) {
    console.log(JSON.stringify(summary, null, 2));
  } else if (ok) {
    console.log(`[ok] bundle composed under ${summary.outputDir} (${stages.length} stages)`);
    for (const [key, value] of Object.entries(artifactPaths)) {
      console.log(`  - ${key}: ${value}`);
    }
  } else {
    console.log(`[fail] bundle composition stopped with ${blockers.length} blocker(s):`);
    for (const b of blockers) {
      console.log(`  - [${b.stage}] ${b.code}: ${b.message}`);
    }
  }
  if (!ok) {
    process.exitCode = 1;
  }
  return summary;
}

export function parseBooleanFlag(value, label) {
  const normalizedInput = asStringOrNull(value);
  if (normalizedInput === null) return null;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = normalizedInput.toLowerCase();
    if (normalized === 'true') return true;
    if (normalized === 'false') return false;
  }
  throw new Error(`${label} must be true or false`);
}

export function parseNumberFlag(value, label) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return null;
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a number`);
  }
  return parsed;
}

export function parseBrowserArgs(value) {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value.map((item) => String(item)) : [String(value)];
}

export function normalizeModelUrl(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return value;
  }
  const trimmed = value.trim();
  if (/^[a-z][a-z0-9+.-]*:\/\//u.test(trimmed)) {
    return trimmed;
  }
  if (trimmed.startsWith('/') || trimmed.startsWith('.')) {
    return pathToFileURL(path.resolve(trimmed)).href.replace(/\/$/, '');
  }
  return trimmed;
}

export function buildNodeRunOptions(jsonOutput) {
  return {
    onProgress(progress) {
      if (jsonOutput) return;
      if (!progress?.message) return;
      if (Number.isFinite(progress.current) && Number.isFinite(progress.total)) {
        console.error(`[progress] ${progress.current}/${progress.total} ${progress.message}`);
      } else {
        console.error(`[progress] ${progress.stage ?? 'run'} ${progress.message}`);
      }
    },
  };
}

export function buildBrowserRunOptions(runConfig, jsonOutput, request = {}) {
  const browser = isPlainObject(runConfig?.browser) ? runConfig.browser : {};

  const headed = parseBooleanFlag(browser.headed, 'run.browser.headed') === true;
  const explicitHeadless = parseBooleanFlag(browser.headless, 'run.browser.headless');
  if (headed && explicitHeadless !== null) {
    throw new Error('run.browser.headed is mutually exclusive with run.browser.headless.');
  }

  const options = {
    channel: asStringOrNull(browser.channel),
    executablePath: asStringOrNull(browser.executablePath),
    runnerPath: asStringOrNull(browser.runnerPath),
    staticRootDir: asStringOrNull(browser.staticRootDir),
    rdrrRoot: asStringOrNull(browser.rdrrRoot),
    baseUrl: asStringOrNull(browser.baseUrl),
    browserArgs: parseBrowserArgs(browser.browserArgs),
    headless: headed ? false : (explicitHeadless ?? true),
    reportOutputPath: asStringOrNull(browser.reportOutputPath),
  };
  const rdrrRoot = resolveRdrrRoot(options);
  options.staticMounts = [
    {
      urlPrefix: '/models/external',
      rootDir: rdrrRoot,
    },
  ];

  const port = parseNumberFlag(browser.port, 'run.browser.port');
  if (port !== null) {
    options.port = port;
  }

  const timeoutMs = parseNumberFlag(browser.timeoutMs, 'run.browser.timeoutMs');
  if (timeoutMs !== null) {
    options.timeoutMs = timeoutMs;
  }

  const opfsCache = parseBooleanFlag(browser.opfsCache, 'run.browser.opfsCache');
  if (opfsCache === false) {
    options.opfsCache = false;
  }

  const userDataDir = asStringOrNull(browser.userDataDir);
  if (userDataDir) {
    options.userDataDir = userDataDir;
  }

  if (request.cacheMode === 'cold') {
    options.wipeCacheBeforeLaunch = true;
  }

  const streamConsole = parseBooleanFlag(browser.console, 'run.browser.console');
  const shouldStreamConsole = streamConsole === true;
  if (shouldStreamConsole && !jsonOutput) {
    options.onConsole = ({ type, text }) => {
      console.error(`[browser:${type}] ${text}`);
    };
  }

  return options;
}

export function finalizeCliCommandResponse(response, request) {
  if (!isPlainObject(response)) {
    return response;
  }
  return {
    ...response,
    request,
  };
}

export async function runCommandOnSurface(request, surface, runConfig, jsonOutput) {
  if (surface === 'node') {
    const nodeRequest = await resolveNodeModelUrl(request);
    if (!jsonOutput) {
      console.error('[surface] running on: node');
      if (nodeRequest.modelUrl && nodeRequest.modelUrl !== request.modelUrl) {
        console.error(`[surface] node resolved modelUrl=${nodeRequest.modelUrl}`);
      }
    }
    const response = await runNodeCommand(nodeRequest, buildNodeRunOptions(jsonOutput));
    return finalizeCliCommandResponse(response, request);
  }

  const browserOptions = buildBrowserRunOptions(runConfig, jsonOutput, request);
  const browserRequest = await resolveBrowserModelUrl(request, browserOptions);

  if (!jsonOutput) {
    const mode = browserOptions.headless === false ? 'headed' : 'headless';
    console.error(`[surface] running on: browser (${mode})`);
    if (browserRequest.modelUrl && browserRequest.modelUrl !== request.modelUrl) {
      console.error(`[surface] browser resolved modelUrl=${browserRequest.modelUrl}`);
    }
  }

  const response = await runBrowserCommandInNode(browserRequest, browserOptions);
  return finalizeCliCommandResponse(await persistBrowserRelayReport(response, browserOptions.reportOutputPath), request);
}
