import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  parseQwenPeftAdapterSafetensors,
  uploadQwenPeftAdapterToLayers,
} from '../../../src/experimental/training/qwen-peft-adapter-import.js';
import {
  exportQwenPeftAdapterFromLayers,
} from '../../../src/experimental/training/qwen-peft-adapter-export.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import {
  acquireBuffer,
  readBuffer,
  releaseBuffer,
} from '../../../src/memory/buffer-pool.js';

function bytesToHex(bytes) {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

async function sha256Hex(buffer) {
  return bytesToHex(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)));
}

function makeLayer(type) {
  return {
    type,
    inputs: {
      attention: { lora: {} },
      mlp: { lora: {} },
    },
  };
}

function projectionKey(name) {
  return name.replace(/_proj$/, '');
}

function installTarget(layers, entry, adapter, ownedTensors) {
  const layer = layers[entry.layerIndex];
  const branch = entry.branch === 'self_attn'
    ? layer.inputs.attention.lora
    : layer.inputs.mlp.lora;
  const key = projectionKey(entry.projection);
  const pair = branch[key] ?? {
    rank: adapter.rank,
    alpha: adapter.alpha,
  };
  const buffer = acquireBuffer(
    entry.data.byteLength,
    undefined,
    `qwen_peft_gpu_${entry.layerIndex}_${key}_${entry.kind}`
  );
  const tensor = createTensor(
    buffer,
    'f32',
    [...entry.shape],
    `qwen_peft_gpu_${entry.canonicalName}`
  );
  pair[entry.kind === 'a' ? 'A' : 'B'] = tensor;
  branch[key] = pair;
  ownedTensors.push(tensor);
}

function targetTensor(layers, entry) {
  const layer = layers[entry.layerIndex];
  const branch = entry.branch === 'self_attn'
    ? layer.inputs.attention.lora
    : layer.inputs.mlp.lora;
  return branch[projectionKey(entry.projection)][entry.kind === 'a' ? 'A' : 'B'];
}

function compareBits(actual, expected) {
  if (actual.byteLength !== expected.byteLength) {
    throw new Error(`Qwen PEFT GPU readback byte mismatch: ${actual.byteLength} != ${expected.byteLength}.`);
  }
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.byteLength / 4);
  const expectedBits = new Uint32Array(
    expected.buffer,
    expected.byteOffset,
    expected.byteLength / 4
  );
  let mismatchCount = 0;
  let maxAbsError = 0;
  for (let index = 0; index < expectedBits.length; index += 1) {
    if (actualBits[index] !== expectedBits[index]) mismatchCount += 1;
    maxAbsError = Math.max(maxAbsError, Math.abs(actual[index] - expected[index]));
  }
  return { elementCount: expected.length, mismatchCount, maxAbsError };
}

export async function runQwenPeftAdapterGpuUploadOracle(input) {
  if (!input?.weightsUrl || !input?.configUrl || !input?.baseModel
    || !input?.weightsSha256 || !input?.configSha256
    || !Array.isArray(input.layerTypes)) {
    throw new Error('Qwen PEFT GPU upload oracle requires weights, config, hash, and layer types.');
  }
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());

  const [weightsResponse, configResponse] = await Promise.all([
    fetch(new URL(input.weightsUrl, globalThis.location.origin)),
    fetch(new URL(input.configUrl, globalThis.location.origin)),
  ]);
  if (!weightsResponse.ok || !configResponse.ok) {
    throw new Error('Qwen PEFT GPU upload oracle could not load the staged adapter.');
  }
  const [weights, configBytes] = await Promise.all([
    weightsResponse.arrayBuffer(),
    configResponse.arrayBuffer(),
  ]);
  const [actualWeightsSha256, actualConfigSha256] = await Promise.all([
    sha256Hex(weights),
    sha256Hex(configBytes),
  ]);
  if (actualWeightsSha256 !== input.weightsSha256) {
    throw new Error(
      `Qwen PEFT GPU upload weights SHA-256 mismatch: ${actualWeightsSha256}.`
    );
  }
  if (actualConfigSha256 !== input.configSha256) {
    throw new Error(`Qwen PEFT GPU upload config SHA-256 mismatch: ${actualConfigSha256}.`);
  }
  const adapterConfig = JSON.parse(new TextDecoder().decode(configBytes));
  const adapter = parseQwenPeftAdapterSafetensors(weights, {
    r: adapterConfig.r,
    lora_alpha: adapterConfig.lora_alpha,
    target_modules: adapterConfig.target_modules,
    layerTypes: input.layerTypes,
  });

  await initDevice();
  const ownedTensors = [];
  const layers = adapter.layerTypes.map(makeLayer);
  try {
    for (const entry of adapter.tensors) installTarget(layers, entry, adapter, ownedTensors);
    const upload = uploadQwenPeftAdapterToLayers(layers, adapter);
    let mismatchCount = 0;
    let maxAbsError = 0;
    let comparedElementCount = 0;
    for (const entry of adapter.tensors) {
      const tensor = targetTensor(layers, entry);
      const actual = new Float32Array(await readBuffer(tensor.buffer, entry.data.byteLength));
      const comparison = compareBits(actual, entry.data);
      mismatchCount += comparison.mismatchCount;
      maxAbsError = Math.max(maxAbsError, comparison.maxAbsError);
      comparedElementCount += comparison.elementCount;
    }
    const exported = await exportQwenPeftAdapterFromLayers(layers, {
      rank: adapter.rank,
      alpha: adapter.alpha,
      dropout: adapterConfig.lora_dropout,
      baseModel: input.baseModel,
      targetModules: adapter.targetModules,
      layerTypes: adapter.layerTypes,
    });
    const exportedSha256 = await sha256Hex(exported.weights);
    const exportedRoundTrip = parseQwenPeftAdapterSafetensors(
      exported.weights,
      {
        r: exported.adapterConfig.r,
        lora_alpha: exported.adapterConfig.lora_alpha,
        target_modules: exported.adapterConfig.target_modules,
        layerTypes: adapter.layerTypes,
      }
    );
    let exportMismatchCount = 0;
    let exportMaxAbsError = 0;
    let exportComparedElementCount = 0;
    for (let index = 0; index < adapter.tensors.length; index += 1) {
      const expected = adapter.tensors[index];
      const actual = exportedRoundTrip.tensors[index];
      if (actual?.canonicalName !== expected.canonicalName) {
        throw new Error(`Qwen PEFT export tensor order mismatch at index ${index}.`);
      }
      const comparison = compareBits(actual.data, expected.data);
      exportMismatchCount += comparison.mismatchCount;
      exportMaxAbsError = Math.max(exportMaxAbsError, comparison.maxAbsError);
      exportComparedElementCount += comparison.elementCount;
    }
    const expectedTensorCount = 256;
    const expectedPairCount = 128;
    const expectedElementCount = 58195968;
    const passed = adapter.tensorCount === expectedTensorCount
      && adapter.pairCount === expectedPairCount
      && adapter.elementCount === expectedElementCount
      && upload.tensorCount === expectedTensorCount
      && upload.elementCount === expectedElementCount
      && comparedElementCount === expectedElementCount
      && mismatchCount === 0
      && maxAbsError === 0
      && exported.tensorCount === expectedTensorCount
      && exported.pairCount === expectedPairCount
      && exported.elementCount === expectedElementCount
      && exportedRoundTrip.elementCount === expectedElementCount
      && exportComparedElementCount === expectedElementCount
      && exportMismatchCount === 0
      && exportMaxAbsError === 0;
    const capabilities = getKernelCapabilities();
    return {
      artifactType: 'qwen35_9b_peft_adapter_gpu_upload_oracle',
      schemaVersion: 1,
      passed,
      model: {
        id: 'Qwen/Qwen3.5-9B',
        revision: 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        layerCount: adapter.layerTypes.length,
        linearAttentionLayerCount: adapter.layerTypes.filter(
          (type) => type === 'linear_attention'
        ).length,
        fullAttentionLayerCount: adapter.layerTypes.filter(
          (type) => type === 'full_attention'
        ).length,
      },
      sourceAdapter: {
        weightsSha256: actualWeightsSha256,
        weightsBytes: weights.byteLength,
        configSha256: actualConfigSha256,
        configBytes: configBytes.byteLength,
      },
      normalizedAdapter: {
        rank: adapter.rank,
        alpha: adapter.alpha,
        tensorCount: adapter.tensorCount,
        pairCount: adapter.pairCount,
        elementCount: adapter.elementCount,
        byteCountF32: adapter.elementCount * 4,
      },
      gpuUpload: {
        tensorCount: upload.tensorCount,
        elementCount: upload.elementCount,
        comparedElementCount,
        bitMismatchCount: mismatchCount,
        maxAbsError,
        exactBitMatch: mismatchCount === 0,
      },
      peftExport: {
        baseModel: exported.adapterConfig.base_model_name_or_path,
        weightsSha256: exportedSha256,
        weightsBytes: exported.weights.byteLength,
        configSha256: await sha256Hex(
          new TextEncoder().encode(exported.adapterConfigJson)
        ),
        tensorCount: exported.tensorCount,
        pairCount: exported.pairCount,
        elementCount: exported.elementCount,
        comparedElementCount: exportComparedElementCount,
        bitMismatchCount: exportMismatchCount,
        maxAbsError: exportMaxAbsError,
        exactCanonicalRoundTrip: exportMismatchCount === 0,
      },
      adapterInfo: capabilities.adapterInfo || null,
      claimBoundary: 'Exact WebGPU upload/readback and PEFT-format export round trip for all 256 production-topology rank-32 Qwen 3.5 9B LoRA tensors; an independent PEFT loader, base-model loading, forward activation, training, inference coherence, compiler capability, and semantic WGSL evidence remain absent.',
    };
  } finally {
    for (const tensor of ownedTensors) releaseBuffer(tensor.buffer);
  }
}
