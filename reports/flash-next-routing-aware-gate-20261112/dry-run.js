// Dry run of doppler.routing-aware-acceptance/v1 on the retained Nov-10/Nov-12 Flash-Next
// artifacts. No GPU, no new execution. Maps what the committed artifacts contain into the
// evaluator's evidence shape, lets the evaluator fail closed on everything they lack, and
// separately applies the attribution rule to the inferred (not admissible) divergences.
//
// Usage: node reports/flash-next-routing-aware-gate-20261112/dry-run.js OUT.json
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  attributeDivergences,
  evaluateRoutingAwareAcceptance,
  hashPolicy,
  nearTieBounds,
} from '../../tools/lib/routing-aware-acceptance.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const [out] = process.argv.slice(2);
if (!out) throw new Error('Usage: dry-run.js OUT.json');
const inputs = {};
async function readJson(relative) {
  const bytes = await fs.readFile(path.join(ROOT, relative));
  inputs[relative] = crypto.createHash('sha256').update(bytes).digest('hex');
  return JSON.parse(bytes);
}
const policy = await readJson('tools/policies/routing-aware-acceptance.json');
const promptSet = await readJson(policy.taskQuality.promptSet.path);
const tolerance = await readJson(policy.numericalTolerance.policyPath);
const forced = await readJson('reports/flash-next-forced-routing-20261112/comparison.json');
const trace = await readJson('reports/flash-next-new-prompts-20261110/routing-trace.json');
const sameInput = await readJson('reports/flash-next-new-prompts-20261110/same-input-operator-controls.json');

const candidate = policy.qualificationTuples[0];
const { id: _id, ...pinned } = candidate;
// The retained receipts name the platform only as "AMD RDNA3, Node webgpu 0.4.0" and do
// not record an exact adapter string, so the tuple cannot be completed from them.
const tuple = { ...pinned, device: { ...pinned.device } };
const indicesLogits = forced.logits.find(row => row.label === 'indices');
const evidence = {
  schema: 'doppler.routing-aware-acceptance-evidence/v1',
  policySha256: null,
  promptSetSha256: null,
  runStartedAtUtc: forced.createdUtc,
  tuple,
  cases: [{
    caseId: 'inventory', numLayers: 48, numTokens: 2187, numExperts: 512, topK: 10,
    cpu: null,
    unforced: { mode: 'observe', selectionSource: 'inferred-from-captured-logits', tuple,
      prefillLogitsSha256: forced.logits.find(row => row.label === 'unforced').prefillLogitsSha256,
      uninstrumentedPrefillLogitsSha256: forced.logits.find(row => row.label === 'retained-original').prefillLogitsSha256 },
    forcedIndices: { mode: 'indices', completed: forced.routing.indices.completed, tuple,
      executedSelectionDifferingTokenLayers: forced.routing.indices.executedSelectionDifferingTokenLayers,
      logits: { policyId: forced.policy.id, count: indicesLogits.count, outsidePolicy: indicesLogits.outsidePolicy },
      layers: forced.routing.indices.layers.map(row => ({ layer: row.layer,
        routerLogitsMaxAbsDiffVsReference: row.routerLogitsMaxAbsDiffVsCpu })) },
  }],
  sameInputOperators: null,
  taskQuality: null,
};
const verdict = evaluateRoutingAwareAcceptance({ policy, promptSet,
  tolerancePolicies: tolerance.tolerancePolicies, evidence });

// Informational only: the attribution rule applied to divergences inferred from captured
// GPU router logits. Inferred rankings are not executed readbacks, so this is not evidence
// the gate can accept.
const divergences = trace.routes.flatMap(route => route.inferredSetDifferences.map(token => ({ layer: route.layer, token })))
  .sort((a, b) => a.layer - b.layer || a.token - b.token);
const { roots, cascades } = attributeDivergences(divergences);
const bounds = nearTieBounds(policy.nearTie, evidence.cases[0].forcedIndices.layers, 48);
const rootRows = roots.map(({ layer, token }) => {
  const route = trace.routes[layer];
  // The CPU rank-10/11 margin is the 60/225 flip displacement only when the retained
  // per-layer minimum sits at this token; per-token CPU logits are not committed.
  const displacement = route.sourceMinimumMarginPosition === token ? route.sourceMinimumRawLogitMargin : null;
  return { layer, token, cpuFlipDisplacement: displacement, bound: bounds[layer],
    nearTieUnderProposedBound: displacement !== null && displacement <= bounds[layer] };
});
const sameInputLayers = sameInput.layers.map(row => row.layer);
const report = {
  schema: 'doppler.routing-aware-acceptance-dry-run/v1',
  createdUtc: '2026-11-12',
  scope: 'Dry run on retained artifacts only; no GPU or CPU model execution. Not qualification.',
  policySha256: hashPolicy(policy),
  inputs,
  verdict,
  informational: {
    admissible: false,
    inferredRoutingAttribution: {
      source: 'reports/flash-next-new-prompts-20261110/routing-trace.json (rankings inferred from captured GPU router logits)',
      divergentTokenLayers: divergences.length,
      cascadeTokenLayers: cascades,
      roots: rootRows,
      executedReadbackAgrees: {
        firstExecutedDivergence: forced.routing.observe.firstExecutedDivergence,
        executedSelectionDifferingTokenLayers: forced.routing.observe.executedSelectionDifferingTokenLayers,
      },
    },
    forcedIndicesControl: {
      logitViolations: indicesLogits.outsidePolicy,
      executedSelectionDifferingTokenLayers: forced.routing.indices.executedSelectionDifferingTokenLayers,
      ownSelectionInterventions: forced.routing.indices.forcingInterventions,
      ownSelectionSetsRecorded: false,
    },
    sameInputOperatorCoverage: {
      caseId: sameInput.sequenceId,
      layers: [Math.min(...sameInputLayers), Math.max(...sameInputLayers)],
      requiredLayers: [0, 47],
      position: sameInput.targetTokenIndex,
      recordsQuantizedV1ViolationCounts: false,
    },
  },
};
await fs.writeFile(out, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
console.log(JSON.stringify({ verdict: verdict.verdict, codes: [...new Set(verdict.reasons.map(r => r.code))],
  roots: rootRows, cascades }));
