import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  COLUMBO_QWEN_ADAPTER_PARITY_CHECKS,
  TINKER_PEFT_BROWSER_ADAPTER_PARITY_CHECKS,
  TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID,
  TRANSLATION_FULL_CHECKPOINT_PARITY_CHECKS,
  assertTrainerArtifactCandidateEntry,
  buildTrainerArtifactImportPlan,
  buildTrainerArtifactParityTemplate,
  verifyTrainerArtifactParityEvidence,
} from '../../src/experimental/bridge/trainer-artifact-bridge.js';
import {
  importTrainerArtifactHandoff,
  verifyTrainerArtifactHandoff,
} from '../../src/tooling/trainer-artifact-handoff.js';

const VERIFIED_AT = '2026-07-13T00:00:00.000Z';
const ARCHITECTURE = Object.freeze({
  architectures: ['UnitForCausalLM'],
  modelType: 'unit_text',
  hiddenSize: 8,
  intermediateSize: 16,
  layers: 2,
  attentionHeads: 2,
  keyValueHeads: 1,
  headDim: 4,
  vocabularySize: 32,
});

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function writeIdentity(repoRoot, rootPath, relativePath, value, id, role, repository) {
  const filePath = path.join(repoRoot, rootPath, relativePath);
  mkdirSync(path.dirname(filePath), { recursive: true });
  const bytes = Buffer.from(value);
  writeFileSync(filePath, bytes);
  return {
    id,
    role,
    repository,
    rootPath,
    path: relativePath,
    sha256: sha256(bytes),
    bytes: bytes.byteLength,
  };
}

function baseDescriptor({ artifact, tokenizer, conversion, evaluation, selection, parity }) {
  return {
    schema: TRAINER_ARTIFACT_BRIDGE_SCHEMA_ID,
    bridgeId: 'bridge.unit',
    sourceContractId: 'unit.contract',
    artifact,
    baseModel: {
      modelId: 'unit-model',
      checkpointSha256: artifact.files[0].sha256,
      tokenizer,
      architecture: ARCHITECTURE,
    },
    conversion: {
      owner: 'clocksmith/doppler',
      sourceArtifactSha256: artifact.files[0].sha256,
      config: conversion,
      runtimeArtifact: null,
    },
    evaluation: {
      populationRole: 'diagnostic_only',
      files: evaluation,
    },
    selection,
    parity,
    claimBoundary: 'Unit fixture only.',
  };
}

const root = mkdtempSync(path.join(tmpdir(), 'doppler-trainer-artifact-handoff-'));
try {
  const gammaRoot = path.join(root, 'gamma');
  const dopplerRoot = path.join(root, 'doppler');
  const columboRoot = path.join(root, 'columbo');
  const repositoryRoots = {
    'clocksmith/gamma': gammaRoot,
    'clocksmith/doppler': dopplerRoot,
    'clocksmith/columbo': columboRoot,
  };
  const checkpointRoot = 'runs/checkpoint-1';
  const weights = writeIdentity(
    gammaRoot,
    checkpointRoot,
    'model.safetensors',
    'unit-weights',
    'checkpoint.weights',
    'weights',
    'clocksmith/gamma'
  );
  const configValue = JSON.stringify({
    architectures: ARCHITECTURE.architectures,
    model_type: ARCHITECTURE.modelType,
    hidden_size: ARCHITECTURE.hiddenSize,
    intermediate_size: ARCHITECTURE.intermediateSize,
    num_hidden_layers: ARCHITECTURE.layers,
    num_attention_heads: ARCHITECTURE.attentionHeads,
    num_key_value_heads: ARCHITECTURE.keyValueHeads,
    head_dim: ARCHITECTURE.headDim,
    vocab_size: ARCHITECTURE.vocabularySize,
  });
  const config = writeIdentity(
    gammaRoot,
    checkpointRoot,
    'config.json',
    configValue,
    'checkpoint.config',
    'config',
    'clocksmith/gamma'
  );
  const generationConfig = writeIdentity(
    gammaRoot,
    checkpointRoot,
    'generation_config.json',
    '{}',
    'checkpoint.generation_config',
    'generation_config',
    'clocksmith/gamma'
  );
  const tokenizerFile = writeIdentity(
    gammaRoot,
    checkpointRoot,
    'tokenizer.json',
    '{}',
    'tokenizer.json',
    'tokenizer_asset',
    'clocksmith/gamma'
  );
  const promptContract = writeIdentity(
    gammaRoot,
    checkpointRoot,
    'chat_template.jinja',
    '{{ messages }}',
    'tokenizer.prompt',
    'prompt_contract',
    'clocksmith/gamma'
  );
  const conversionConfig = writeIdentity(
    dopplerRoot,
    '',
    'config/conversion.json',
    '{}',
    'conversion.config',
    'conversion_config',
    'clocksmith/doppler'
  );
  const evaluationFile = writeIdentity(
    gammaRoot,
    '',
    'evaluation/population.jsonl',
    '{"id":1}\n',
    'evaluation.population',
    'population',
    'clocksmith/gamma'
  );
  const fullDescriptor = baseDescriptor({
    artifact: {
      kind: 'full_checkpoint',
      role: 'diagnostic_baseline',
      format: 'huggingface_safetensors',
      repository: 'clocksmith/gamma',
      rootPath: checkpointRoot,
      files: [weights, config, generationConfig],
    },
    tokenizer: { files: [tokenizerFile], promptContract },
    conversion: conversionConfig,
    evaluation: [evaluationFile],
    selection: {
      authority: 'clocksmith/gamma',
      status: 'not_selected',
      receipt: null,
    },
    parity: {
      profile: 'translation_full_checkpoint',
      requiredChecks: [...TRANSLATION_FULL_CHECKPOINT_PARITY_CHECKS],
    },
  });

  const fullVerification = await verifyTrainerArtifactHandoff({
    contract: fullDescriptor,
    repositoryRoots,
    verifiedAt: VERIFIED_AT,
  });
  assert.equal(fullVerification.receipt.ok, true);
  assert.equal(fullVerification.receipt.admission.candidateCompetitionAllowed, false);
  assert.throws(
    () => assertTrainerArtifactCandidateEntry(fullDescriptor),
    /not a selected candidate/
  );
  const importPlan = buildTrainerArtifactImportPlan(fullDescriptor, fullVerification.receipt);
  assert.equal(importPlan.entrypoint, 'resolveNodeSourceRuntimeBundle');
  assert.equal(importPlan.admission.candidateCompetitionAllowed, false);
  assert.equal(importPlan.admission.promotionAllowed, false);

  const importedFull = await importTrainerArtifactHandoff({
    contract: fullDescriptor,
    repositoryRoots,
    verifiedAt: VERIFIED_AT,
    async runtimeResolver(options) {
      assert.equal(options.inputPath, path.join(gammaRoot, checkpointRoot));
      assert.equal(options.verifyHashes, true);
      return {
        model: { modelId: 'unit-model' },
        manifest: { modelId: 'unit-model' },
        storageContext: {},
        sourceKind: 'safetensors',
        sourceRoot: options.inputPath,
        resolvedMemoryBudgetBytes: null,
      };
    },
  });
  assert.equal(importedFull.receipt.candidateCompetitionAllowed, false);
  assert.equal(importedFull.imported.sourceKind, 'safetensors');

  const parityTemplate = buildTrainerArtifactParityTemplate(fullDescriptor, fullVerification.receipt);
  assert.equal(parityTemplate.checks[0].status, 'pass');
  assert.equal(parityTemplate.checks.at(-1).status, 'pending');
  const blockedParity = verifyTrainerArtifactParityEvidence(
    fullDescriptor,
    fullVerification.receipt,
    parityTemplate
  );
  assert.equal(blockedParity.decision, 'block');
  assert.ok(blockedParity.blockers.includes('parity_check_not_passed:first_token_logits'));
  const passingEvidence = structuredClone(parityTemplate);
  for (const check of passingEvidence.checks) {
    check.status = 'pass';
    check.upstreamDecision = 'pass';
    check.evidenceHash = fullVerification.receipt.receiptHash;
  }
  const passedParity = verifyTrainerArtifactParityEvidence(
    fullDescriptor,
    fullVerification.receipt,
    passingEvidence
  );
  assert.equal(passedParity.decision, 'pass');

  writeFileSync(path.join(gammaRoot, checkpointRoot, 'generation_config.json'), '{"changed":true}');
  const tampered = await verifyTrainerArtifactHandoff({
    contract: fullDescriptor,
    repositoryRoots,
    verifiedAt: VERIFIED_AT,
  });
  assert.equal(tampered.receipt.ok, false);
  assert.ok(
    tampered.receipt.checks.find((check) => check.id === 'source_artifact_byte_identity').errors
      .some((error) => error.includes('sha256_mismatch'))
  );

  const adapterRoot = 'generated/adapters/unit';
  const adapterWeights = writeIdentity(
    columboRoot,
    adapterRoot,
    'adapters.safetensors',
    'adapter-weights',
    'adapter.weights',
    'adapter_weights',
    'clocksmith/columbo'
  );
  const adapterConfig = writeIdentity(
    columboRoot,
    adapterRoot,
    'adapter_config.json',
    '{}',
    'adapter.config',
    'adapter_config',
    'clocksmith/columbo'
  );
  const dopplerManifest = writeIdentity(
    columboRoot,
    adapterRoot,
    'doppler-adapter-manifest.json',
    JSON.stringify({ id: 'unit-adapter', baseModel: 'unit-model' }),
    'adapter.doppler_manifest',
    'doppler_adapter_manifest',
    'clocksmith/columbo'
  );
  const runtimeManifest = writeIdentity(
    columboRoot,
    adapterRoot,
    'runtime-adapter-manifest.json',
    '{}',
    'adapter.runtime_manifest',
    'runtime_adapter_manifest',
    'clocksmith/columbo'
  );
  const trainingReport = writeIdentity(
    columboRoot,
    adapterRoot,
    'training-report.json',
    '{}',
    'adapter.training_report',
    'training_report',
    'clocksmith/columbo'
  );
  const columboTokenizer = writeIdentity(
    columboRoot,
    adapterRoot,
    'tokenizer.json',
    '{}',
    'adapter.tokenizer',
    'tokenizer_asset',
    'clocksmith/columbo'
  );
  const columboPrompt = writeIdentity(
    columboRoot,
    adapterRoot,
    'chat_template.jinja',
    '{{ messages }}',
    'adapter.prompt',
    'prompt_contract',
    'clocksmith/columbo'
  );
  const columboEvaluation = writeIdentity(
    columboRoot,
    adapterRoot,
    'selection-receipt.json',
    '{}',
    'adapter.selection_evidence',
    'selection_evidence',
    'clocksmith/columbo'
  );
  const adapterDescriptor = baseDescriptor({
    artifact: {
      kind: 'peft_adapter',
      role: 'selected_candidate',
      format: 'peft_safetensors',
      repository: 'clocksmith/columbo',
      rootPath: adapterRoot,
      files: [adapterWeights, adapterConfig, dopplerManifest, runtimeManifest, trainingReport],
    },
    tokenizer: { files: [columboTokenizer], promptContract: columboPrompt },
    conversion: null,
    evaluation: [columboEvaluation],
    selection: {
      authority: 'clocksmith/columbo',
      status: 'selected',
      receipt: 'columbo.selection-receipt/v1:unit',
    },
    parity: {
      profile: 'columbo_qwen_adapter',
      requiredChecks: [...COLUMBO_QWEN_ADAPTER_PARITY_CHECKS],
    },
  });
  adapterDescriptor.bridgeId = 'bridge.columbo.unit';
  adapterDescriptor.baseModel.checkpointSha256 = adapterWeights.sha256;
  adapterDescriptor.conversion.sourceArtifactSha256 = adapterWeights.sha256;
  assert.equal(assertTrainerArtifactCandidateEntry(adapterDescriptor).selection.status, 'selected');
  const adapterVerification = await verifyTrainerArtifactHandoff({
    contract: adapterDescriptor,
    repositoryRoots,
    verifiedAt: VERIFIED_AT,
  });
  assert.equal(adapterVerification.receipt.ok, true);
  const importedAdapter = await importTrainerArtifactHandoff({
    contract: adapterDescriptor,
    repositoryRoots,
    verifiedAt: VERIFIED_AT,
    async adapterLoader(manifest, options) {
      assert.equal(manifest.id, 'unit-adapter');
      assert.equal(typeof options.readFile, 'function');
      return { adapter: { name: manifest.id }, manifest };
    },
  });
  assert.equal(importedAdapter.imported.adapterId, 'unit-adapter');
  assert.equal(importedAdapter.plan.admission.candidateCompetitionAllowed, true);
  assert.equal(importedAdapter.plan.admission.promotionAllowed, false);

  const diagnosticAdapterDescriptor = structuredClone(adapterDescriptor);
  diagnosticAdapterDescriptor.bridgeId = 'bridge.columbo.diagnostic-unit';
  diagnosticAdapterDescriptor.artifact.role = 'diagnostic_candidate';
  diagnosticAdapterDescriptor.selection.status = 'not_selected';
  diagnosticAdapterDescriptor.selection.receipt = null;
  const diagnosticVerification = await verifyTrainerArtifactHandoff({
    contract: diagnosticAdapterDescriptor,
    repositoryRoots,
    verifiedAt: VERIFIED_AT,
  });
  assert.equal(diagnosticVerification.receipt.ok, true);
  assert.equal(diagnosticVerification.receipt.admission.candidateCompetitionAllowed, false);
  assert.throws(
    () => assertTrainerArtifactCandidateEntry(diagnosticAdapterDescriptor),
    /not a selected candidate/
  );
  const diagnosticImport = await importTrainerArtifactHandoff({
    contract: diagnosticAdapterDescriptor,
    repositoryRoots,
    verifiedAt: VERIFIED_AT,
    async adapterLoader(manifest) {
      return { adapter: { name: manifest.id }, manifest };
    },
  });
  assert.equal(diagnosticImport.imported.adapterId, 'unit-adapter');
  assert.equal(diagnosticImport.plan.admission.candidateCompetitionAllowed, false);
  assert.equal(diagnosticImport.receipt.candidateCompetitionAllowed, false);
  assert.equal(diagnosticImport.receipt.promotionAllowed, false);

  const wrongAuthority = structuredClone(adapterDescriptor);
  wrongAuthority.selection.authority = 'clocksmith/gamma';
  assert.throws(
    () => assertTrainerArtifactCandidateEntry(wrongAuthority),
    /selection authority must be "clocksmith\/columbo"/
  );

  const tinkerDescriptor = structuredClone(adapterDescriptor);
  tinkerDescriptor.bridgeId = 'bridge.tinker.unit';
  tinkerDescriptor.sourceContractId = 'gamma.same-r.tinker-browser.unit';
  tinkerDescriptor.selection.authority = 'clocksmith/gamma';
  tinkerDescriptor.selection.receipt = 'same-r.selection-receipt/v1:tinker-unit';
  tinkerDescriptor.parity = {
    profile: 'tinker_peft_browser_adapter',
    requiredChecks: [...TINKER_PEFT_BROWSER_ADAPTER_PARITY_CHECKS],
  };
  assert.equal(assertTrainerArtifactCandidateEntry(tinkerDescriptor).selection.authority, 'clocksmith/gamma');
  const tinkerVerification = await verifyTrainerArtifactHandoff({
    contract: tinkerDescriptor,
    repositoryRoots,
    verifiedAt: VERIFIED_AT,
  });
  assert.equal(tinkerVerification.receipt.ok, true);
  const tinkerTemplate = buildTrainerArtifactParityTemplate(
    tinkerDescriptor,
    tinkerVerification.receipt
  );
  assert.deepEqual(
    tinkerTemplate.checks.slice(0, 4).map((check) => check.status),
    ['pass', 'pass', 'pass', 'pass']
  );
  assert.deepEqual(
    tinkerTemplate.checks.slice(4).map((check) => check.status),
    ['pending', 'pending', 'pending', 'pending', 'pending', 'pending']
  );

  const tinkerWrongAuthority = structuredClone(tinkerDescriptor);
  tinkerWrongAuthority.selection.authority = 'clocksmith/columbo';
  assert.throws(
    () => assertTrainerArtifactCandidateEntry(tinkerWrongAuthority),
    /selection authority must be "clocksmith\/gamma"/
  );
} finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('trainer-artifact-handoff.test: ok');
