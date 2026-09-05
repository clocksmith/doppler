import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  createProgramBundleCliDefaults,
  loadProgramBundle,
  verifyClosedProgramBundle,
  writeProgramBundle,
} from './program-bundle.js';
import { runForgePipeline } from '../converter/forge-stages.js';
import { hashTargetPlan } from '../config/target-plan.js';
import { loadPackSigningKey, writePackV2 } from './pack-v2.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { buildReferenceTranscript } from './program-bundle/materialize.js';

export const FORGE_VERSION = '2.0.0';

export function usage() {
  return [
    'Doppler Forge: deterministic signed Pack v2 compiler',
    '',
    'Usage:',
    '  node tools/forge-model-pack.js --program-bundle <path> --release-manifest <path> --out <pack.json>',
    '  node tools/forge-model-pack.js --manifest <path> --reference-report <path> --release-manifest <path> --out <pack.json> [--conversion-config <path>]',
    '  node tools/forge-model-pack.js --config <path|json>',
    '',
    'Flags:',
    '  --program-bundle <path>       Existing verified Program Bundle v1',
    '  --manifest <path>             Source manifest used to build a Program Bundle',
    '  --reference-report <path>     Physical execution reference report',
    '  --qualification-report <path> Additional physical-surface qualification report (repeatable)',
    '  --conversion-config <path>    Conversion configuration',
    '  --runtime-config <path>       Runtime configuration',
    '  --model-ir-receipt <path>     Validated ModelIR receipt containing modelIR',
    '  --initial-identity <path>     Pre-dispatch observed execution identity or report',
    '  --release-manifest <path>     Required doppler.pack-release/v1 contract',
    '  --model-dir <path>            Model artifact directory',
    '  --out <path>                  Signed Pack v2 output path',
    '  --signing-private-key <path>  Ed25519 private JWK',
    '  --signing-public-key <path>   Ed25519 public JWK',
    '  --signing-authority <id>      Trusted signing authority ID',
    '  --created-at <iso8601>        Stable Program Bundle timestamp',
    '  --config <path|json>          Inline JSON or config file',
    '  --json                        Emit machine-readable JSON',
    '  --help, -h                    Show this help',
  ].join('\n');
}

export function parseArgs(argv) {
  const flags = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (token === '--json') {
      flags.json = true;
      continue;
    }
    if (!token.startsWith('--')) throw new Error(`Unsupported positional argument "${token}".`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith('--')) throw new Error(`Missing value for --${key}.`);
    if (key === 'qualification-report') {
      flags[key] = [...(Array.isArray(flags[key]) ? flags[key] : []), value];
    } else {
      flags[key] = value;
    }
    index += 1;
  }
  return flags;
}

export async function readJsonInput(value) {
  const normalized = String(value || '').trim();
  if (!normalized) throw new Error('--config must be a JSON object or path.');
  if (normalized.startsWith('{')) return JSON.parse(normalized);
  return JSON.parse(await fs.readFile(path.resolve(normalized), 'utf8'));
}

export async function buildForgeOptions(
  flags,
  metaUrl = import.meta.url
) {
  const defaults = metaUrl === import.meta.url
    ? { repoRoot: path.resolve(path.dirname(fileURLToPath(metaUrl)), '../..') }
    : createProgramBundleCliDefaults(metaUrl);
  const values = flags.config ? await readJsonInput(flags.config) : {
    programBundlePath: flags['program-bundle'] ?? null,
    manifestPath: flags.manifest ?? null,
    modelDir: flags['model-dir'] ?? null,
    referenceReportPath: flags['reference-report'] ?? null,
    qualificationReportPaths: flags['qualification-report'] ?? [],
    conversionConfigPath: flags['conversion-config'] ?? null,
    runtimeConfigPath: flags['runtime-config'] ?? null,
    modelIRReceiptPath: flags['model-ir-receipt'] ?? null,
    initialExecutionIdentityPath: flags['initial-identity'] ?? null,
    releaseManifestPath: flags['release-manifest'] ?? null,
    outputPath: flags.out ?? null,
    createdAtUtc: flags['created-at'] ?? null,
    signingPrivateKeyPath: flags['signing-private-key'] ?? null,
    signingPublicKeyPath: flags['signing-public-key'] ?? null,
    signingAuthority: flags['signing-authority'] ?? null,
  };
  if (!values || typeof values !== 'object' || Array.isArray(values)) {
    throw new Error('--config must resolve to a JSON object.');
  }
  return {
    ...defaults,
    ...values,
    outputPath: values.outputPath ?? values.out ?? null,
    allowDevelopmentSigner: false,
  };
}

async function readJsonFile(filePath, label) {
  const resolved = path.resolve(filePath);
  const raw = await fs.readFile(resolved, 'utf8');
  const json = JSON.parse(raw);
  if (!json || typeof json !== 'object' || Array.isArray(json)) throw new Error(`${label} must be a JSON object.`);
  return { path: resolved, raw, json };
}

async function hashFile(filePath) {
  const hash = createHash('sha256');
  let sizeBytes = 0;
  for await (const chunk of createReadStream(filePath)) {
    hash.update(chunk);
    sizeBytes += chunk.byteLength;
  }
  return { hash: `sha256:${hash.digest('hex')}`, sizeBytes };
}

function hashCanonical(value) {
  return `sha256:${createHash('sha256').update(JSON.stringify(stableSortObject(value))).digest('hex')}`;
}

async function loadQualificationEvidence(reportPaths, bundle) {
  if (bundle.referenceTranscript?.operation === 'rerank') {
    const evidence = [];
    for (const reportPath of reportPaths || []) {
      const result = await buildReferenceTranscript(reportPath, process.cwd(), bundle.execution.graphHash);
      const transcript = result.transcript;
      if (transcript.operation !== 'rerank' || transcript.modelId !== bundle.modelId
        || transcript.manifestHash !== bundle.referenceTranscript.manifestHash
        || transcript.referenceDigest !== bundle.referenceTranscript.referenceDigest) {
        throw new Error('Forge rerank qualification must use the same model, source reference, and request.');
      }
      const observed = await hashFile(reportPath);
      evidence.push({ surface: transcript.surface, status: 'passed', operation: 'rerank',
        rerankedDocuments: transcript.reference.input.documents.length,
        evidenceHash: observed.hash, sizeBytes: observed.sizeBytes, sourcePath: path.resolve(reportPath),
        transcriptHash: hashCanonical(transcript) });
    }
    return evidence;
  }
  if (bundle.referenceTranscript?.operation === 'encodeSequence') {
    const evidence = [];
    for (const reportPath of reportPaths || []) {
      const result = await buildReferenceTranscript(reportPath, process.cwd(), bundle.execution.graphHash);
      const reference = bundle.referenceTranscript;
      const transcript = result.transcript;
      if (transcript.operation !== 'encodeSequence' || transcript.modelId !== bundle.modelId
        || transcript.manifestHash !== reference.manifestHash
        || hashCanonical(transcript.reference) !== hashCanonical(reference.reference)
        || hashCanonical(transcript.options) !== hashCanonical(reference.options)) {
        throw new Error('Forge sequence qualification must use the same model, reference, and request.');
      }
      const observed = await hashFile(reportPath);
      evidence.push({
        surface: transcript.surface, status: 'passed', operation: 'encodeSequence', encodedSequences: 1,
        evidenceHash: observed.hash, sizeBytes: observed.sizeBytes, sourcePath: path.resolve(reportPath),
        transcriptHash: hashCanonical(transcript),
      });
    }
    return evidence;
  }
  const referenceTokens = bundle.referenceTranscript?.tokens?.ids;
  if (!Array.isArray(referenceTokens) || referenceTokens.length === 0) {
    throw new Error('Forge qualification requires Program Bundle reference token IDs.');
  }
  const evidence = [];
  for (const reportPath of reportPaths || []) {
    const reportFile = await readJsonFile(reportPath, 'qualification report');
    const report = reportFile.json;
    const transcript = report.metrics?.referenceTranscript;
    const surface = transcript?.surface;
    const tokens = transcript?.tokens?.ids;
    const generationConfig = transcript?.generationConfig;
    if (report.modelId !== bundle.modelId) {
      throw new Error(`Forge qualification report modelId "${report.modelId}" does not match "${bundle.modelId}".`);
    }
    if (!report.results?.some((result) => result?.name === 'generation' && result?.passed === true)) {
      throw new Error(`Forge qualification report "${reportPath}" has no passed generation result.`);
    }
    if (typeof surface !== 'string' || !surface.endsWith('-webgpu')) {
      throw new Error(`Forge qualification report "${reportPath}" lacks an explicit WebGPU surface.`);
    }
    if (!generationConfig || !Number.isFinite(generationConfig.temperature)) {
      throw new Error(`Forge qualification report "${reportPath}" lacks explicit generationConfig.`);
    }
    if (generationConfig.temperature > 0 && !Number.isFinite(generationConfig.seed)) {
      throw new Error(`Forge qualification report "${reportPath}" is stochastic without an explicit seed.`);
    }
    if (!Array.isArray(tokens)
      || tokens.length !== referenceTokens.length
      || tokens.some((tokenId, index) => tokenId !== referenceTokens[index])) {
      throw new Error(`Forge qualification report "${reportPath}" does not exactly match the Program Bundle transcript.`);
    }
    if (transcript.executionGraphHash !== bundle.execution.graphHash) {
      throw new Error(`Forge qualification report "${reportPath}" execution graph does not match the Program Bundle.`);
    }
    const observed = await hashFile(reportFile.path);
    evidence.push({
      surface,
      status: 'passed',
      evidenceHash: observed.hash,
      sizeBytes: observed.sizeBytes,
      sourcePath: reportFile.path,
      transcriptHash: hashCanonical(transcript),
      generatedTokens: tokens.length,
    });
  }
  return evidence;
}

function resolveBundleArtifactPath(artifact, bundlePath, repoRoot) {
  if (artifact.role === 'wgsl-source' || artifact.role === 'host-source') {
    return path.resolve(path.dirname(bundlePath), artifact.path);
  }
  return path.resolve(repoRoot, artifact.path);
}

async function verifySourceArtifactBytes(bundle, bundlePath, repoRoot) {
  const receipts = [];
  for (const artifact of bundle.artifacts) {
    const sourcePath = resolveBundleArtifactPath(artifact, bundlePath, repoRoot);
    const observed = await hashFile(sourcePath);
    if (observed.hash !== artifact.hash || observed.sizeBytes !== artifact.sizeBytes) {
      throw new Error(
        `Forge source artifact mismatch for "${artifact.path}": ` +
        `expected ${artifact.hash}/${artifact.sizeBytes}, got ${observed.hash}/${observed.sizeBytes}.`
      );
    }
    receipts.push({ path: sourcePath, ...observed });
  }
  return receipts;
}

async function materializeProgramBundle(options) {
  if (options.programBundlePath) {
    const bundlePath = path.resolve(options.programBundlePath);
    const raw = await fs.readFile(bundlePath, 'utf8');
    const bundle = await loadProgramBundle(bundlePath);
    await verifyClosedProgramBundle(bundlePath, bundle);
    return { bundle, bundlePath, raw };
  }
  if (!options.outputPath) throw new Error('Doppler Forge requires outputPath.');
  let createdAtUtc = options.createdAtUtc;
  if (!createdAtUtc && options.referenceReportPath) {
    const reference = await readJsonFile(options.referenceReportPath, 'reference report');
    createdAtUtc = reference.json.timestamp ?? reference.json.createdAtUtc ?? reference.json.generatedAt ?? null;
  }
  if (!createdAtUtc) {
    throw new Error('Forge requires a stable --created-at or a timestamp in the reference report.');
  }
  const outputPath = path.resolve(options.outputPath);
  const bundlePath = path.resolve(
    options.programBundleOutputPath
      ?? path.join(path.dirname(outputPath), 'artifacts', 'program-bundle.json')
  );
  const result = await writeProgramBundle({
    ...options,
    outputPath: bundlePath,
    createdAtUtc,
  });
  return {
    bundle: result.bundle,
    bundlePath: result.outputPath,
    raw: await fs.readFile(result.outputPath, 'utf8'),
  };
}

async function resolveSigner(options) {
  const privateKeyPath = options.signingPrivateKeyPath;
  const publicKeyPath = options.signingPublicKeyPath;
  const authority = options.signingAuthority;
  if (!privateKeyPath || !publicKeyPath || !authority) {
    throw new Error(
      'Doppler Forge requires explicit signingPrivateKeyPath, signingPublicKeyPath, and signingAuthority.'
    );
  }
  const privateKeyJwk = await loadPackSigningKey(privateKeyPath);
  const publicKeyJwk = await loadPackSigningKey(publicKeyPath);
  return { authority, privateKeyJwk, publicKeyJwk };
}

async function materializePackArtifactClosure(pack, sourceBundle, sourceBundlePath, repoRoot, outputPath, qualificationEvidence = [], modelIREvidence = null) {
  const outputRoot = path.dirname(path.resolve(outputPath));
  const remainingSources = sourceBundle.artifacts.map((artifact) => ({
    artifact,
    sourcePath: resolveBundleArtifactPath(artifact, sourceBundlePath, repoRoot),
    used: false,
  }));
  remainingSources.push(...qualificationEvidence.map((evidence) => ({
    artifact: {
      role: 'qualification-evidence',
      hash: evidence.evidenceHash,
      sizeBytes: evidence.sizeBytes,
    },
    sourcePath: evidence.sourcePath,
    used: false,
  })));
  if (modelIREvidence) {
    remainingSources.push({
      artifact: {
        role: 'source-truth-evidence',
        hash: modelIREvidence.hash,
        sizeBytes: modelIREvidence.sizeBytes,
      },
      sourcePath: modelIREvidence.sourcePath,
      used: false,
    });
  }
  for (const artifact of pack.artifacts) {
    let sourcePath;
    if (artifact.role === 'program-bundle') {
      sourcePath = sourceBundlePath;
    } else {
      const match = remainingSources.find((candidate) => (
        !candidate.used
        && candidate.artifact.role === artifact.role
        && candidate.artifact.hash === artifact.hash
        && candidate.artifact.sizeBytes === artifact.sizeBytes
      ));
      if (!match) throw new Error(`Forge cannot materialize Pack artifact "${artifact.artifactId}".`);
      match.used = true;
      sourcePath = match.sourcePath;
    }
    const destination = path.resolve(outputRoot, artifact.path);
    const relative = path.relative(outputRoot, destination);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      throw new Error(`Forge Pack artifact path escapes output root: ${artifact.path}.`);
    }
    await fs.mkdir(path.dirname(destination), { recursive: true });
    if (path.resolve(sourcePath) !== destination) await fs.copyFile(sourcePath, destination);
    const observed = await hashFile(destination);
    if (observed.hash !== artifact.hash || observed.sizeBytes !== artifact.sizeBytes) {
      throw new Error(`Forge materialized artifact verification failed for "${artifact.path}".`);
    }
  }
}

export async function forgeModelPack(options) {
  if (!options?.outputPath) throw new Error('Doppler Forge requires --out / outputPath.');
  if (!options?.releaseManifestPath) {
    throw new Error('Doppler Forge requires --release-manifest / releaseManifestPath.');
  }
  const repoRoot = path.resolve(options.repoRoot ?? process.cwd());
  const source = await materializeProgramBundle({ ...options, repoRoot });
  await verifySourceArtifactBytes(source.bundle, source.bundlePath, repoRoot);
  const manifestPath = path.resolve(
    options.manifestPath ?? path.join(repoRoot, source.bundle.sources.manifest.path)
  );
  const manifest = await readJsonFile(manifestPath, 'manifest');
  const release = (await readJsonFile(options.releaseManifestPath, 'Pack release contract')).json;
  const signer = await resolveSigner(options);
  const qualificationEvidence = await loadQualificationEvidence(
    options.qualificationReportPaths ?? [],
    source.bundle
  );
  let modelIR = null;
  let modelIREvidence = null;
  if (options.modelIRReceiptPath) {
    const modelIRReceiptFile = await readJsonFile(options.modelIRReceiptPath, 'ModelIR receipt');
    const modelIRReceipt = modelIRReceiptFile.json;
    if (!modelIRReceipt.modelIR || typeof modelIRReceipt.modelIR !== 'object'
      || Array.isArray(modelIRReceipt.modelIR)) {
      throw new Error('ModelIR receipt must contain a modelIR object.');
    }
    modelIR = modelIRReceipt.modelIR;
    modelIREvidence = {
      sourcePath: modelIRReceiptFile.path,
      hash: `sha256:${createHash('sha256').update(modelIRReceiptFile.raw).digest('hex')}`,
      sizeBytes: new TextEncoder().encode(modelIRReceiptFile.raw).byteLength,
    };
  }
  let initialExecutionIdentity = null;
  if (options.initialExecutionIdentityPath) {
    const identitySource = (await readJsonFile(
      options.initialExecutionIdentityPath,
      'initial execution identity'
    )).json;
    initialExecutionIdentity = identitySource.initialExecutionIdentity
      ?? identitySource.metrics?.initialExecutionIdentity
      ?? identitySource.runtime?.initialExecutionIdentity
      ?? identitySource;
  }
  const { pack, stages } = await runForgePipeline({
    manifest: manifest.json,
    manifestRaw: manifest.raw,
    programBundle: source.bundle,
    programBundleRaw: source.raw,
    programBundlePath: source.bundlePath,
    repoRoot,
    outputPath: path.resolve(options.outputPath),
    qualificationEvidence,
    modelIR,
    modelIREvidence,
    initialExecutionIdentity,
    release,
  }, signer);
  await materializePackArtifactClosure(
    pack,
    source.bundle,
    source.bundlePath,
    repoRoot,
    options.outputPath,
    qualificationEvidence,
    modelIREvidence
  );
  const written = await writePackV2(options.outputPath, pack);
  return {
    ok: true,
    forgeVersion: FORGE_VERSION,
    outputPath: path.relative(process.cwd(), written.outputPath),
    absoluteOutputPath: written.outputPath,
    modelId: pack.modelId,
    packId: pack.packId,
    semanticRoot: pack.semanticRoot,
    envelopeHash: written.envelopeHash,
    schema: pack.schema,
    schemaVersion: pack.schemaVersion,
    createdAtUtc: pack.createdAtUtc,
    sourceBundleId: source.bundle.bundleId,
    programBundlePath: source.bundlePath,
    executionGraphHash: pack.program.executionGraphHash,
    artifactCount: pack.artifacts.length,
    wgslModuleCount: pack.wgslModules.length,
    targetPlanDigests: pack.targetPlans.map((plan) => hashTargetPlan(plan)),
    stages,
  };
}
