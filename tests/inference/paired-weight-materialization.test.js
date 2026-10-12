import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/inference/pipelines/text/weight-loading-config.js', import.meta.url).href;
const hooks = registerHooks({resolve(name, context, next) {
 if(context.parentURL===target && name==='../../../gpu/device.js') return {url:'data:text/javascript,export const getKernelCapabilities=()=>({hasSubgroups:false});',shortCircuit:true};
 return next(name,context);
}});
const {resolveWeightLoadingConfig}=await import(target);
const step = (op, weight) => ({op,kernel:'matmul_paired.wgsl',entry:'main',constants:{WEIGHT_F16:weight==='f16',WEIGHT_Q4K:weight==='q4k',SCALAR_OUTPUT:op==='lm_head'}});
const path={decode:{steps:[step('linear_qkv_proj','q4k'),step('linear_a_proj','f16')]},prefill:{steps:[step('q_proj','q4k')]},postLayer:[step('lm_head','q4k')]};
const result=resolveWeightLoadingConfig({modelId:'paired-weights-fixture',quantization:'Q4_K_M',quantizationInfo:{layout:'row'}},path,'execution-v1');
assert.equal(result.useFusedQ4K,true,'Declared packed operands must remain packed.');
assert.equal(result.q4kMaterializationMode,'mixed');
assert.deepEqual(result.q4kFusedRoles,['linear_qkv_proj','lm_head','q_proj']);
assert.equal(result.keepF32Weights,false,'Sharing a shader filename does not request F32 materialization.');
hooks.deregister();
console.log('paired weight materialization: exact selected variants retain Q4 and independent F16 roles');
