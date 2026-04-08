import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import {
  extractExecutionContractFacts,
  renderExecutionContractLeanModule,
  sanitizeLeanModuleName,
} from '../../src/tooling/lean-execution-contract.js';
import { EXECUTION_V1_SCHEMA_ID } from '../../src/config/schema/index.js';
import { createExecutionContractSession } from '../helpers/execution-v1-fixtures.js';

function createExecutionV1ContractManifest(overrides = {}) {
  return {
    modelId: 'lean-execution-contract-test',
    architecture: {
      headDim: 256,
      maxSeqLen: 131072,
      ...(overrides.architecture ?? {}),
    },
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      session: createExecutionContractSession(overrides.session ?? {}),
      execution: {
        kernels: {
          attn: {
            kernel: 'attention_streaming_f16kv.wgsl',
            entry: 'main',
            digest: `sha256:${'a'.repeat(64)}`,
          },
        },
        preLayer: [],
        decode: [
          ['attention', 'attn'],
        ],
        prefill: [
          ['attention', 'attn'],
        ],
        postLayer: [],
        policies: {
          unsupportedPrecision: 'error',
          dtypeTransition: 'require_cast_step',
          unresolvedKernel: 'error',
        },
        ...(overrides.execution ?? {}),
      },
    },
  };
}

{
  const manifest = createExecutionV1ContractManifest({
    modelId: 'bdpa-prefill-contract-mismatch',
    session: {
      kvcache: {
        layout: 'bdpa',
        tiering: {
          mode: 'off',
        },
        quantization: {
          mode: 'none',
        },
      },
      decodeLoop: {
        batchSize: 16,
        disableCommandBatching: false,
      },
    },
  });

  const facts = extractExecutionContractFacts(manifest);
  assert.deepEqual(facts.session, {
    layout: 'bdpa',
    disableCommandBatching: false,
    decodeBatchSize: 16,
    headDim: 256,
    kvLen: 131072,
    coldQuantMode: 'none',
    contiguousQuantMode: 'none',
  });
  assert.deepEqual(facts.steps, [
    { id: 'layer_decode_0_attention', phase: 'decode', opClass: 'attention' },
    { id: 'layer_prefill_1_attention', phase: 'prefill', opClass: 'attention' },
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
  const manifest = createExecutionV1ContractManifest({
    modelId: 'paged-runtime-contract',
    architecture: {
      headDim: 128,
      maxSeqLen: 4096,
    },
    session: {
      kvcache: {
        layout: 'paged',
        tiering: {
          mode: 'off',
        },
        quantization: {
          mode: 'none',
        },
      },
      decodeLoop: {
        batchSize: 1,
        disableCommandBatching: true,
      },
    },
    execution: {
      prefill: [],
    },
  });

  const facts = extractExecutionContractFacts(manifest);
  assert.deepEqual(facts.session, {
    layout: 'paged',
    disableCommandBatching: true,
    decodeBatchSize: 1,
    headDim: 128,
    kvLen: 4096,
    coldQuantMode: 'none',
    contiguousQuantMode: 'none',
  });

  const rendered = renderExecutionContractLeanModule(facts, {
    moduleName: 'PagedRuntimeContract',
  });
  assert.match(rendered, /layout := \.paged/);
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
      const manifestBatchSize = manifest.inference.session.decodeLoop.batchSize;
      assert.equal(facts.modelId, 'translategemma-4b-it-q4k-ehf16-af32');
      assert.ok(['contiguous', 'paged', 'tiered', 'bdpa'].includes(facts.session.layout));
      assert.equal(facts.session.decodeBatchSize, manifestBatchSize);
      assert.equal(
        facts.steps.some((step) => step.phase === 'prefill' && step.opClass === 'attention'),
        true
      );

      const rendered = renderExecutionContractLeanModule(facts, {
        moduleName: sanitizeLeanModuleName(facts.modelId),
      });
      assert.match(rendered, new RegExp(`layout := \\.${facts.session.layout}`));
      assert.match(rendered, new RegExp(`decodeBatchSize := ${manifestBatchSize}`));
      assert.match(rendered, /translategemma-4b-it-q4k-ehf16-af32\.steps/);
    }
  }
}

console.log('lean-execution-contract.test: ok');
