import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import {
  extractExecutionContractFacts,
  renderExecutionContractLeanModule,
  sanitizeLeanModuleName,
} from '../../src/tooling/lean-execution-contract.js';

{
  const manifest = {
    modelId: 'bdpa-prefill-contract-mismatch',
    architecture: {
      headDim: 256,
      maxSeqLen: 131072,
    },
    inference: {
      session: {
        kvcache: {
          layout: 'bdpa',
          tiering: {
            mode: 'off',
          },
        },
        decodeLoop: {
          batchSize: 16,
          disableCommandBatching: false,
        },
      },
      execution: {
        steps: [
          {
            id: 'prefill_attention',
            phase: 'prefill',
            op: 'attention',
          },
          {
            id: 'decode_attention',
            phase: 'decode',
            op: 'attention',
          },
        ],
      },
    },
  };

  const facts = extractExecutionContractFacts(manifest);
  assert.deepEqual(facts.session, {
    layout: 'bdpa',
    disableCommandBatching: false,
    decodeBatchSize: 16,
    headDim: 256,
    kvLen: 131072,
    coldQuantMode: 'none',
  });
  assert.deepEqual(facts.steps, [
    { id: 'prefill_attention', phase: 'prefill', opClass: 'attention' },
    { id: 'decode_attention', phase: 'decode', opClass: 'attention' },
  ]);

  const rendered = renderExecutionContractLeanModule(facts, {
    moduleName: 'BDPAPrefillContractMismatch',
  });
  assert.match(rendered, /layout := \.bdpa/);
  assert.match(rendered, /decodeBatchSize := 16/);
  assert.match(rendered, /phase := \.prefill/);
  assert.match(rendered, /executionContractOverall/);
}

{
  const manifest = {
    modelId: 'bdpa-paged-runtime-contract',
    architecture: {
      headDim: 128,
      maxSeqLen: 4096,
    },
    inference: {
      session: {
        kvcache: {
          layout: 'bdpa_paged',
          tiering: {
            mode: 'off',
          },
        },
        decodeLoop: {
          batchSize: 1,
          disableCommandBatching: true,
        },
      },
      execution: {
        steps: [
          {
            id: 'decode_attention',
            phase: 'decode',
            op: 'attention',
          },
        ],
      },
    },
  };

  const facts = extractExecutionContractFacts(manifest);
  assert.deepEqual(facts.session, {
    layout: 'bdpa_paged',
    disableCommandBatching: true,
    decodeBatchSize: 1,
    headDim: 128,
    kvLen: 4096,
    coldQuantMode: 'none',
  });

  const rendered = renderExecutionContractLeanModule(facts, {
    moduleName: 'BDPAPagedRuntimeContract',
  });
  assert.match(rendered, /layout := \.bdpa_paged/);
  assert.match(rendered, /disableCommandBatching := true/);
}

{
  const manifestPath = path.join('models/local/translategemma-4b-it-q4k-ehf16-af32', 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    console.log('lean-execution-contract.test: skipped (local model fixture missing)');
  } else {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    if (
      manifest?.inference?.session?.decodeLoop == null
      || manifest.inference.session?.kvcache?.layout == null
    ) {
      console.log('lean-execution-contract.test: skipped (local model fixture has stale session contract)');
    } else {
      const facts = extractExecutionContractFacts(manifest);
      assert.equal(facts.modelId, 'translategemma-4b-it-q4k-ehf16-af32');
      assert.equal(facts.session.layout, 'paged');
      assert.equal(facts.session.decodeBatchSize, 1);
      assert.equal(
        facts.steps.some((step) => step.phase === 'prefill' && step.opClass === 'attention'),
        true
      );

      const rendered = renderExecutionContractLeanModule(facts, {
        moduleName: sanitizeLeanModuleName(facts.modelId),
      });
      assert.match(rendered, /layout := \.paged/);
      assert.match(rendered, /translategemma-4b-it-q4k-ehf16-af32\.steps/);
    }
  }
}

console.log('lean-execution-contract.test: ok');
