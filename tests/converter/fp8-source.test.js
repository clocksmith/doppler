import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { parseSafetensorsHeader } from '../../src/formats/safetensors/types.js';
import { float8ToFloat32 } from '../../src/formats/float8.js';
import { packFP8Source, materializeFP8Source, prepareFP8SourceTensors, readFP8CompanionScales } from '../../src/converter/fp8-source.js';
import { transformTensorBytes } from '../../src/converter/tensor-transform.js';
import { buildQuantizationInfo } from '../../src/converter/quantization-info.js';
import { validateFP8Policy } from '../../src/config/schema/fp8.schema.js';
import { validateFP8ManifestContract } from '../../src/formats/rdrr/fp8-contract.js';
import { validateFP8StorageBytes } from '../../src/formats/rdrr/fp8-storage.js';
import { normalizeTensorStorageDescriptor } from '../../src/formats/rdrr/storage-descriptor.js';
import { createConverterConfig } from '../../src/config/schema/converter.schema.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';

installNodeFileFetchShim();
const { convertModel } = await import('../../src/converter/core.js');
const oracle = JSON.parse(await readFile(new URL('../fixtures/fp8-code-oracle.json', import.meta.url)));
const contract = JSON.parse(await readFile(new URL('../fixtures/fp8-operator-contract.json', import.meta.url)));
const scratch = new Float32Array(1), raw = new Uint32Array(scratch.buffer);
const fromBits = bits => { raw[0] = bits; return scratch[0]; };
for (const encoding of ['e4m3fn', 'e5m2']) {
  for (let code = 0; code < 256; code++) {
    const expected = oracle.formats[encoding][code];
    scratch[0] = float8ToFloat32(code, encoding);
    if ((expected & 0x7fffffff) > 0x7f800000) assert.ok(Number.isNaN(scratch[0]));
    else assert.equal(raw[0], expected, `${encoding} code${code}`);
  }
  const policy = { ...contract.policy, encoding };
  assert.equal(validateFP8Policy(policy), policy);
  for (const field of Object.keys(policy)) {
    const missing = { ...policy }; delete missing[field];
    assert.throws(() => validateFP8Policy(missing), /FP8/);
  }
  assert.throws(() => validateFP8Policy({ ...policy, encoding: 'e4m3fnuz' }), /unsupported/);
  const shape = [5, 131], codes = Uint8Array.from({ length: 5*131 }, (_, i) => i % 256);
  const scales = new Float32Array([0.5, 2, 4, 0.25]);
  const packed = packFP8Source(codes, shape, policy, scales);
  assert.deepEqual(normalizeTensorStorageDescriptor(packed.storage, 'fp8'), packed.storage);
  assert.deepEqual(validateFP8StorageBytes(packed.quantized, shape, packed.storage), packed.storage);
  const values = materializeFP8Source(codes, shape, policy, scales);
  for (let n = 0; n < shape[0]; n++) for (let k = 0; k < shape[1]; k++) {
    const scale = scales[Math.floor(n/3)*2 + Math.floor(k/128)];
    const expected = Math.fround(fromBits(oracle.formats[encoding][codes[n*shape[1]+k]]) * scale);
    assert.ok(Object.is(values[n*shape[1]+k], expected), `scaled FP8${n}:${k}`);
  }
  const source = { name:'model.layers.0.self_attn.q_proj.weight', dtype:encoding === 'e4m3fn' ? 'F8_E4M3' : 'F8_E5M2', shape, role:'matmul', size:codes.length };
  const companion = { name:source.name+policy.scaleSuffix, dtype:'F32', shape:[2,2], size:16 };
  const header=new TextEncoder().encode(JSON.stringify({[source.name]:{dtype:source.dtype,shape,data_offsets:[0,codes.length]},[companion.name]:{dtype:'F32',shape:[2,2],data_offsets:[codes.length,codes.length+16]}}));
  const safeFile=new Uint8Array(8+header.length+codes.length+16);
  new DataView(safeFile.buffer).setBigUint64(0,BigInt(header.length),true);safeFile.set(header,8);safeFile.set(codes,8+header.length);safeFile.set(new Uint8Array(scales.buffer),8+header.length+codes.length);
  const parsed=parseSafetensorsHeader(safeFile.buffer);
  assert.equal(parsed.tensors[0].dtype,source.dtype);assert.equal(parsed.tensors[0].elemSize,1);
  const plan = prepareFP8SourceTensors([source, companion], policy, { quantization_config:{ weight_block_size:[3,128] } });
  assert.deepEqual(plan.tensors,[source]); assert.equal(plan.companions.get(source.name),companion);
  assert.deepEqual(await readFP8CompanionScales(source, companion, policy, async () => new Uint8Array(scales.buffer)), scales);
  for (const inputs of [[source], [source,{...companion,dtype:'F16'}], [source,{...companion,shape:[1,4]}]]) {
    assert.throws(() => prepareFP8SourceTensors(inputs,policy,null), /companion/);
  }
  assert.throws(() => prepareFP8SourceTensors([source,companion],policy,{quantization_config:{weight_block_size:[128,128]}}), /weight_block_size/);
  const tag = encoding === 'e4m3fn' ? 'fp8e4' : 'fp8e5';
  const info = buildQuantizationInfo({ quantization:{weights:tag,embeddings:'f32',lmHead:'f32',computePrecision:'f32',fp8:policy} }, tag, 'f32', 'f32');
  const result = transformTensorBytes(source,codes,{targetQuant:tag,quantizationInfo:info,fp8Scales:scales});
  assert.deepEqual(result.tensorData, packed.quantized);
  assert.equal(result.sourceDtype,source.dtype);
  const location = {dtype:result.outDtype,layout:result.outLayout,shape,role:'matmul',storage:result.storage,size:result.tensorData.length};
  assert.deepEqual(validateFP8ManifestContract({quantizationInfo:info,tensors:{weight:location}}), []);
  for (const change of [t=>{t.layout='column';},t=>{t.size--;},t=>{t.storage={...t.storage,encoding:'fnuz'};},t=>{t.storage={...t.storage,blockShape:[1,128]};}]) {
    const altered=structuredClone(location);change(altered);
    assert.ok(validateFP8ManifestContract({quantizationInfo:info,tensors:{weight:altered}}).length);
  }
  for (const mode of ['length','padding','scale']) {
    const bad=packed.quantized.slice();
    if(mode==='padding')bad[bad.length-1]=1;
    if(mode==='scale')new DataView(bad.buffer).setFloat32(0,NaN,true);
    assert.throws(()=>validateFP8StorageBytes(mode==='length'?bad.slice(1):bad,shape,packed.storage), /FP8/);
  }
  assert.throws(()=>packFP8Source(codes,shape,policy,null),/scales/);
  assert.throws(()=>transformTensorBytes(source,codes,{targetQuant:tag}),/policy/);
  const identity={...policy,scaleMode:'identity',scaleSuffix:null};
  assert.throws(()=>prepareFP8SourceTensors([source],identity,{quantization_config:{}}),/cannot ignore/);
  const dense=transformTensorBytes(source,codes,{targetQuant:'f32',quantizationInfo:{...info,weights:'f32'},fp8Scales:scales});
  assert.equal(dense.outDtype,'F32');assert.deepEqual(new Float32Array(dense.tensorData.buffer),values);
  // Actual conversion writes the explicit packed artifact and consumes scale facts.
  const shards=[];let manifest;
  await convertModel({modelId:'fp8-fixture',name:'fp8-fixture',quantization:source.dtype,tensors:parsed.tensors,config:{}}, {
    readTensorData:async tensor=>safeFile.slice(tensor.offset,tensor.offset+tensor.size).buffer,
    writeShard:async(index,bytes)=>{shards[index]=bytes.slice();return 'fixture-hash';},
    writeManifest:async value=>{manifest=value;},
  }, {modelId:'fp8-fixture',modelType:'transformer',quantization:tag,quantizationInfo:info,shardSize:4096,
    architecture:{numLayers:1,hiddenSize:131,intermediateSize:131,numAttentionHeads:1,numKeyValueHeads:1,headDim:131,vocabSize:5,maxSeqLen:16,ropeTheta:10000},
    inference:{...DEFAULT_MANIFEST_INFERENCE},eosTokenId:1,converterConfig:createConverterConfig()});
  assert.ok(manifest);assert.deepEqual(Object.keys(manifest.tensors),[source.name]);
  assert.deepEqual(validateFP8ManifestContract(manifest),[]);
  const emitted=manifest.tensors[source.name];assert.equal(emitted.dtype,result.outDtype);
  assert.deepEqual(shards[emitted.shard].slice(emitted.offset,emitted.offset+emitted.size),packed.quantized);
}
console.log('FP8 source: all512 PyTorch code patterns, faithful special values, block scales/tails, explicit dense conversion, packed artifact and failure contracts passed.');
