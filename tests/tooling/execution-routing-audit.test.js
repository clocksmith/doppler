import assert from 'node:assert/strict';
import {
  auditManifestExecutionRouting,
} from '../../src/tooling/execution-routing-audit.js';

const registry = {
  operations: {
    attention: {
      variants: {
        prefill_streaming_f16kv: {
          wgsl: 'attention_streaming_f16kv.wgsl',
          entryPoint: 'main',
          outputDtype: 'f32',
          requires: ['shader-f16'],
          variantMetadata: { tier: 'streaming' },
        },
        prefill_head256_f16kv: {
          wgsl: 'attention_head256_f16kv.wgsl',
          entryPoint: 'main',
          outputDtype: 'f32',
          requires: ['shader-f16'],
          variantMetadata: { tier: 'tiled_small', exactHeadDim: 256 },
        },
      },
    },
  },
};
const digests = {
  'attention_streaming_f16kv.wgsl#main': '1'.repeat(64),
  'attention_head256_f16kv.wgsl#main': '2'.repeat(64),
};
const audit = auditManifestExecutionRouting({
  modelId: 'test-model',
  architecture: { headDim: 256 },
  inference: {
    session: { compute: { defaults: { activationDtype: 'f32' } } },
    execution: {
      kernels: {
        attention: {
          kernel: 'attention_streaming_f16kv.wgsl',
          entry: 'main',
          digest: `sha256:${'1'.repeat(64)}`,
        },
      },
      prefill: [['attention', 'attention']],
    },
  },
}, registry, digests);

assert.equal(audit.integrity[0].status, 'verified');
assert.equal(audit.opportunities.length, 1);
assert.equal(audit.opportunities[0].reason, 'exact-head-prefill-available');
assert.equal(audit.opportunities[0].candidate.variantId, 'prefill_head256_f16kv');
assert.equal(audit.opportunities[0].disposition, 'calibration-required');
assert.equal(
  audit.opportunities[0].selectionPolicy,
  'required-after-evidence-promotion-on-compatible-hardware'
);

console.log('execution-routing-audit.test: ok');
