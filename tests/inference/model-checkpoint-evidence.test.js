import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MODEL_CHECKPOINT_STAGES,
  buildModelCheckpointEvidence,
  flattenModelCheckpointDigests,
} from '../../src/inference/model-checkpoint-evidence.js';

const DIGEST = `sha256:${'1'.repeat(64)}`;
const KV_DIGEST = `sha256:${'2'.repeat(64)}`;

function record(stageName, opType, layerIndex = 0) {
  return {
    opId: layerIndex === null ? stageName : `layer.${layerIndex}.${stageName}`,
    stageName,
    opType,
    layerIndex,
    dtype: 'f16',
    shapeSignature: '1x16',
    capture: {
      fullTensorDigest: DIGEST,
      sample: [0.25, -0.5],
      sampleCoordinates: [[0, 0], [0, 1]],
      stats: { maxAbs: 0.5 },
      hasNaN: false,
      hasInf: false,
    },
  };
}

function stepRecords() {
  return [
    record('embed.out', 'embedding', null),
    record('attn.normed', 'normalization'),
    record('attn.q_proj', 'projection'),
    record('attn.k_proj', 'projection'),
    record('attn.v_proj', 'projection'),
    record('attn.q_rope', 'rope'),
    record('attn.k_rope', 'rope'),
    record('attn.core_out', 'attention'),
    record('ffn.out', 'ffn'),
    record('logits.final', 'logits', null),
  ];
}

test('builds complete full-model checkpoint evidence across prefill and decode', () => {
  const timeline = [...stepRecords(), ...stepRecords(), ...stepRecords()];
  const evidence = buildModelCheckpointEvidence({
    operatorDiagnostics: { timeline },
    kvCacheByteProof: {
      digest: KV_DIGEST,
      layout: 'contiguous',
      kvDtype: 'f16',
      layers: [{
        layer: 0,
        seqLen: 5,
        keyBytes: 160,
        valueBytes: 160,
        keyDigest: DIGEST,
        valueDigest: KV_DIGEST,
      }],
    },
    expectedStepCount: 3,
    minimumDecodeSteps: 2,
  });

  assert.equal(evidence.status, 'complete');
  assert.equal(evidence.stepCount, 3);
  assert.equal(evidence.decodeStepCount, 2);
  assert.deepEqual(evidence.capturedStages, [...MODEL_CHECKPOINT_STAGES].sort());
  assert.equal(evidence.steps[0].phase, 'prefill');
  assert.equal(evidence.steps[1].phase, 'decode');
  assert.equal(evidence.steps[0].checkpoints.qkv.recordCount, 3);
  assert.equal(flattenModelCheckpointDigests(evidence).length, 22);
});

test('fails closed when a tensor class, KV proof, or decode floor is missing', () => {
  const timeline = stepRecords().filter((entry) => entry.opType !== 'rope');
  const evidence = buildModelCheckpointEvidence({
    operatorDiagnostics: { timeline },
    kvCacheByteProof: null,
    expectedStepCount: 3,
    minimumDecodeSteps: 2,
  });

  assert.equal(evidence.status, 'blocked');
  assert.ok(evidence.missingStages.includes('rope'));
  assert.ok(evidence.missingStages.includes('kv'));
  assert.ok(evidence.blockers.some((entry) => entry.includes('expected 3')));
  assert.ok(evidence.blockers.some((entry) => entry.includes('at least 2')));
});
