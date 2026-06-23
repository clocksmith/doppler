import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';

const matrix = JSON.parse(readFileSync('models/gemma4-targets.json', 'utf8'));
const catalog = JSON.parse(readFileSync('models/catalog.json', 'utf8'));
const quickstartRegistry = JSON.parse(readFileSync('src/client/doppler-registry.json', 'utf8'));

const REQUIRED_TARGET_IDS = [
  'gemma-4-e2b',
  'gemma-4-e4b',
  'gemma-4-12b-unified',
  'gemma-4-31b',
  'gemma-4-26b-a4b',
];

const TARGET_STATUS = new Set(['partially_verified', 'gap']);
const SURFACE_STATUS = new Set(['verified', 'unverified', 'unsupported']);
const CLAIM_STATUS = new Set(['verified', 'verified-local', 'experimental']);
const MTP_STATUS = new Set(['not_implemented']);
const SERVE_STATUS = new Set(['verified', 'unverified', 'unsupported']);
const EVIDENCE_STATUS = new Set(['pass', 'performance_evidence', 'diagnostic']);
const EVIDENCE_SURFACE = new Set(['browser', 'electron', 'node']);
const SERVE_EVIDENCE_STATUS = new Set(['pass', 'diagnostic']);
const PREFLIGHT_EVIDENCE_STATUS = new Set(['pass', 'diagnostic']);
const SOURCE_PACKAGE_STATUS = new Set(['blocked', 'unverified', 'verified']);
const BLOCKER_STATE = new Set(['missing', 'unsupported', 'unverified', 'diagnostic', 'not_implemented', 'incomplete']);
const BLOCKER_SURFACE = new Set(['browser', 'electron', 'node', 'serve', 'mtp', 'model', 'benchmark']);
const BLOCKER_CODE_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

const catalogById = new Map(catalog.models.map((model) => [model.modelId, model]));
const quickstartById = new Map(quickstartRegistry.models.map((model) => [model.modelId, model]));
const targetsById = new Map(matrix.targets.map((target) => [target.targetId, target]));

assert.equal(matrix.schemaVersion, 1);
assert.ok(matrix.sourceUrls.includes('https://ai.google.dev/gemma/docs/core'));
assert.ok(matrix.sourceUrls.includes('https://ai.google.dev/gemma/docs/releases'));
assert.ok(matrix.sourceUrls.includes('https://developers.google.com/edge/litert-lm/models/gemma-4'));
assert.deepEqual([...targetsById.keys()].sort(), [...REQUIRED_TARGET_IDS].sort());

for (const target of matrix.targets) {
  assert.ok(TARGET_STATUS.has(target.dopplerStatus), `${target.targetId}: invalid Doppler status`);
  assert.equal(target.officialMtp, true, `${target.targetId}: Gemma 4 targets must track official MTP`);
  assert.ok(MTP_STATUS.has(target.mtpStatus), `${target.targetId}: invalid MTP status`);
  assert.ok(SERVE_STATUS.has(target.serveStatus), `${target.targetId}: invalid serve status`);
  assert.ok(Array.isArray(target.currentLanes), `${target.targetId}: currentLanes must be an array`);
  assert.ok(Array.isArray(target.servedLanes), `${target.targetId}: servedLanes must be an array`);
  assert.ok(target.evidence && typeof target.evidence === 'object', `${target.targetId}: evidence is required`);
  assert.ok(Array.isArray(target.evidence.runtimeReceipts), `${target.targetId}: runtime receipts must be an array`);
  assert.ok(Array.isArray(target.evidence.benchmarkReceipts), `${target.targetId}: benchmark receipts must be an array`);
  assert.ok(Array.isArray(target.evidence.serveReceipts), `${target.targetId}: serve receipts must be an array`);
  assert.ok(Array.isArray(target.evidence.preflightReceipts), `${target.targetId}: preflight receipts must be an array`);
  assert.ok(Array.isArray(target.missing), `${target.targetId}: missing must be an array`);
  assert.ok(Array.isArray(target.blockers), `${target.targetId}: blockers must be an array`);
  assert.ok(target.missing.includes('mtp lane'), `${target.targetId}: MTP gap must stay explicit`);
  if (target.sourcePackages !== undefined) {
    assert.ok(Array.isArray(target.sourcePackages), `${target.targetId}: sourcePackages must be an array`);
    const sourcePackageIds = new Set();
    for (const sourcePackage of target.sourcePackages) {
      assert.ok(typeof sourcePackage.id === 'string' && sourcePackage.id.trim(), `${target.targetId}: source package id is required`);
      assert.ok(!sourcePackageIds.has(sourcePackage.id), `${target.targetId}: duplicate source package ${sourcePackage.id}`);
      sourcePackageIds.add(sourcePackage.id);
      assert.ok(SOURCE_PACKAGE_STATUS.has(sourcePackage.status), `${target.targetId}: invalid source package status`);
      if (sourcePackage.status === 'blocked') {
        assert.ok(BLOCKER_CODE_PATTERN.test(sourcePackage.blockerCode), `${target.targetId}: blocked source package requires blockerCode`);
      }
      assert.ok(typeof sourcePackage.reason === 'string' && sourcePackage.reason.trim(), `${target.targetId}: source package reason is required`);
    }
  }

  const blockerCodes = new Set();
  for (const blocker of target.blockers) {
    assert.ok(BLOCKER_CODE_PATTERN.test(blocker.code), `${target.targetId}: blocker code must be kebab-case`);
    assert.ok(!blockerCodes.has(blocker.code), `${target.targetId}: duplicate blocker ${blocker.code}`);
    blockerCodes.add(blocker.code);
    assert.ok(BLOCKER_SURFACE.has(blocker.surface), `${target.targetId}: invalid blocker surface`);
    assert.ok(BLOCKER_STATE.has(blocker.state), `${target.targetId}: invalid blocker state`);
    assert.ok(typeof blocker.reason === 'string' && blocker.reason.trim(), `${target.targetId}: blocker reason is required`);
  }

  if (target.serveStatus === 'verified') {
    assert.ok(target.servedLanes.length > 0, `${target.targetId}: verified serve target must list served lanes`);
    assert.ok(
      target.evidence.serveReceipts.some((receipt) => receipt.status === 'pass'),
      `${target.targetId}: verified serve target must list a serve pass receipt`
    );
  } else if (target.serveStatus === 'unverified') {
    assert.ok(target.servedLanes.length > 0, `${target.targetId}: unverified serve target must list served lanes`);
    assert.ok(
      target.missing.includes('doppler-serve runtime pass receipt'),
      `${target.targetId}: unverified serve target must keep doppler-serve runtime receipt gap explicit`
    );
    assert.ok(
      target.blockers.some((blocker) => blocker.surface === 'serve' && blocker.state === 'unverified'),
      `${target.targetId}: unverified serve status must have a blocker`
    );
  } else {
    assert.equal(target.servedLanes.length, 0, `${target.targetId}: unsupported serve target must not list lanes`);
    assert.ok(
      target.blockers.some((blocker) => blocker.surface === 'serve' && blocker.state === 'unsupported'),
      `${target.targetId}: unsupported serve status must have a blocker`
    );
    if (target.currentLanes.length > 0) {
      assert.ok(
        target.missing.includes('doppler-serve quickstart lane'),
        `${target.targetId}: unserved current lanes must keep doppler-serve gap explicit`
      );
    }
  }

  for (const surface of ['browser', 'electron', 'node']) {
    assert.ok(
      SURFACE_STATUS.has(target.surfaceStatus?.[surface]),
      `${target.targetId}: invalid ${surface} surface status`
    );
    if (target.surfaceStatus[surface] === 'unverified' || target.surfaceStatus[surface] === 'unsupported') {
      assert.ok(
        target.blockers.some((blocker) => blocker.surface === surface && blocker.state === target.surfaceStatus[surface]),
        `${target.targetId}: ${surface} ${target.surfaceStatus[surface]} status must have a blocker`
      );
    }
  }
  assert.ok(
    target.blockers.some((blocker) => blocker.surface === 'mtp' && blocker.state === target.mtpStatus),
    `${target.targetId}: MTP status must have a matching blocker`
  );

  if (target.dopplerStatus === 'gap') {
    assert.equal(target.currentLanes.length, 0, `${target.targetId}: gap targets must not list current lanes`);
    assert.equal(target.evidence.runtimeReceipts.length, 0, `${target.targetId}: gap targets must not list runtime receipts`);
    assert.equal(target.evidence.benchmarkReceipts.length, 0, `${target.targetId}: gap targets must not list benchmark receipts`);
    assert.equal(target.evidence.serveReceipts.length, 0, `${target.targetId}: gap targets must not list serve receipts`);
    assert.equal(target.evidence.preflightReceipts.length, 0, `${target.targetId}: gap targets must not list preflight receipts`);
    assert.ok(target.missing.length > 0, `${target.targetId}: gap targets must explain missing work`);
    assert.ok(target.blockers.length > 0, `${target.targetId}: gap targets must list blockers`);
    assert.ok(
      target.blockers.some((blocker) => blocker.surface === 'model'),
      `${target.targetId}: gap targets must include a model blocker`
    );
    continue;
  }

  assert.ok(target.currentLanes.length > 0, `${target.targetId}: supported targets must list at least one lane`);
  assert.ok(target.evidence.runtimeReceipts.length > 0, `${target.targetId}: supported targets must list runtime receipts`);
  const laneIds = new Set(target.currentLanes.map((lane) => lane.modelId));
  const passingEvidenceLaneIds = new Set();
  const verifiedEvidenceSurfaces = new Set();
  for (const receipt of [
    ...target.evidence.runtimeReceipts,
    ...target.evidence.benchmarkReceipts,
  ]) {
    assert.ok(laneIds.has(receipt.modelId), `${target.targetId}: receipt model ${receipt.modelId} must be a current lane`);
    assert.ok(EVIDENCE_SURFACE.has(receipt.surface), `${target.targetId}: invalid receipt surface`);
    assert.ok(EVIDENCE_STATUS.has(receipt.status), `${target.targetId}: invalid receipt status`);
    assert.ok(receipt.path.endsWith('.json'), `${target.targetId}: receipt path must be JSON`);
    assert.ok(existsSync(receipt.path), `${target.targetId}: receipt file must exist: ${receipt.path}`);
    if (receipt.status === 'pass') {
      passingEvidenceLaneIds.add(receipt.modelId);
      verifiedEvidenceSurfaces.add(receipt.surface);
    }
  }

  for (const lane of target.currentLanes) {
    assert.ok(CLAIM_STATUS.has(lane.claimStatus), `${lane.modelId}: invalid claim status`);
    const catalogModel = catalogById.get(lane.modelId);
    assert.ok(catalogModel, `${lane.modelId}: matrix lane must exist in models/catalog.json`);
    assert.equal(catalogModel.family, 'gemma4', `${lane.modelId}: matrix lane must be a Gemma 4 catalog model`);

    if (lane.claimStatus === 'verified') {
      assert.equal(
        catalogModel.lifecycle?.status?.tested,
        'verified',
        `${lane.modelId}: verified matrix lane must be catalog-verified`
      );
    }
    if (lane.claimStatus === 'verified' || lane.claimStatus === 'verified-local') {
      assert.ok(
        passingEvidenceLaneIds.has(lane.modelId),
        `${lane.modelId}: verified matrix lane must be backed by passing receipt evidence`
      );
    }
  }

  for (const servedLaneId of target.servedLanes) {
    assert.ok(
      target.currentLanes.some((lane) => lane.modelId === servedLaneId),
      `${target.targetId}: served lane ${servedLaneId} must be a current lane`
    );
    const quickstartModel = quickstartById.get(servedLaneId);
    assert.ok(quickstartModel, `${target.targetId}: served lane ${servedLaneId} must be in quickstart registry`);
    assert.ok(
      quickstartModel.modes.includes('text'),
      `${target.targetId}: served lane ${servedLaneId} must be text-generative`
    );
  }

  const servedLaneIds = new Set(target.servedLanes);
  for (const receipt of target.evidence.serveReceipts) {
    assert.ok(laneIds.has(receipt.modelId), `${target.targetId}: serve receipt model ${receipt.modelId} must be a current lane`);
    assert.ok(servedLaneIds.has(receipt.modelId), `${target.targetId}: serve receipt model ${receipt.modelId} must be a served lane`);
    assert.equal(receipt.surface, 'serve', `${target.targetId}: serve receipt surface must be serve`);
    assert.ok(SERVE_EVIDENCE_STATUS.has(receipt.status), `${target.targetId}: invalid serve receipt status`);
    assert.ok(receipt.path.endsWith('.json'), `${target.targetId}: serve receipt path must be JSON`);
    assert.ok(existsSync(receipt.path), `${target.targetId}: serve receipt file must exist: ${receipt.path}`);
  }

  for (const receipt of target.evidence.preflightReceipts) {
    assert.ok(laneIds.has(receipt.modelId), `${target.targetId}: preflight receipt model ${receipt.modelId} must be a current lane`);
    assert.ok(EVIDENCE_SURFACE.has(receipt.surface), `${target.targetId}: invalid preflight receipt surface`);
    assert.ok(PREFLIGHT_EVIDENCE_STATUS.has(receipt.status), `${target.targetId}: invalid preflight receipt status`);
    assert.ok(receipt.path.endsWith('.json'), `${target.targetId}: preflight receipt path must be JSON`);
    assert.ok(existsSync(receipt.path), `${target.targetId}: preflight receipt file must exist: ${receipt.path}`);
  }

  for (const surface of ['browser', 'node']) {
    if (target.surfaceStatus[surface] !== 'verified') {
      continue;
    }
    assert.ok(
      verifiedEvidenceSurfaces.has(surface),
      `${target.targetId}: ${surface} verified status must have same-surface runtime pass evidence`
    );
    assert.ok(
      target.currentLanes.some((lane) => {
        const catalogModel = catalogById.get(lane.modelId);
        return catalogModel?.lifecycle?.status?.tested === 'verified'
          && catalogModel?.lifecycle?.tested?.surface?.includes(surface);
      }),
      `${target.targetId}: ${surface} verified status must be backed by a verified catalog lane`
    );
  }
}

const related26B = targetsById.get('gemma-4-26b-a4b')?.relatedButNotEquivalent ?? [];
assert.ok(
  related26B.includes('diffusiongemma-26b-a4b-it-q4k-ehf16-af16'),
  'Gemma 4 26B A4B gap must explicitly distinguish DiffusionGemma from the official Gemma 4 MoE target'
);

assert.deepEqual(targetsById.get('gemma-4-e4b')?.sourcePackages, [
  {
    id: 'litert/gemma-4-e4b-it',
    status: 'blocked',
    blockerCode: 'gemma4-e4b-litert-direct-source-unverified',
    reason: 'LiteRT .task and .litertlm package identity is known, but direct-source parsing fails closed until a parity receipt and graph contraction map exist.',
  },
]);

console.log('gemma4-target-matrix-contract.test: ok');
