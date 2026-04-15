import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { classifyProviderFailure, FAILURE_CLASSES } from '../../src/client/failure-taxonomy.js';
import { ERROR_CODES } from '../../src/errors/doppler-error.js';

describe('classifyProviderFailure — fine-grained classes', () => {
  it('classifies OOM as gpu_oom', () => {
    const result = classifyProviderFailure(new Error('WebGPU out of memory during inference'));
    assert.equal(result.failureClass, FAILURE_CLASSES.GPU_OOM);
    assert.equal(result.failureCode, ERROR_CODES.GPU_OOM);
  });

  it('classifies device lost as gpu_device_lost', () => {
    const result = classifyProviderFailure(new Error('GPU device lost'));
    assert.equal(result.failureClass, FAILURE_CLASSES.GPU_DEVICE_LOST);
    assert.equal(result.failureCode, ERROR_CODES.GPU_DEVICE_LOST);
  });

  it('classifies timeout as gpu_timeout', () => {
    const result = classifyProviderFailure(new Error('Operation timeout after 30s'));
    assert.equal(result.failureClass, FAILURE_CLASSES.GPU_TIMEOUT);
    assert.equal(result.failureCode, ERROR_CODES.GPU_TIMEOUT);
  });

  it('classifies unsupported adapter as gpu_unsupported', () => {
    const result = classifyProviderFailure(new Error('Adapter unsupported feature'));
    assert.equal(result.failureClass, FAILURE_CLASSES.GPU_UNSUPPORTED);
    assert.equal(result.failureCode, ERROR_CODES.GPU_UNSUPPORTED_ADAPTER);
  });

  it('classifies webgpu unavailable as gpu_unavailable', () => {
    const result = classifyProviderFailure(new Error('WebGPU is unavailable on this device'));
    assert.equal(result.failureClass, FAILURE_CLASSES.GPU_UNAVAILABLE);
    assert.equal(result.failureCode, ERROR_CODES.GPU_UNAVAILABLE);
  });

  it('classifies model load errors as model_load_failed', () => {
    const result = classifyProviderFailure(new Error('Failed to load manifest shard 3'));
    assert.equal(result.failureClass, FAILURE_CLASSES.MODEL_LOAD_FAILED);
  });

  it('classifies policy_denied from errors with a Doppler error code', () => {
    const err = new Error('denied');
    err.code = ERROR_CODES.PROVIDER_POLICY_DENIED;
    const result = classifyProviderFailure(err);
    assert.equal(result.failureClass, FAILURE_CLASSES.POLICY_DENIED);
  });

  it('classifies fallback network errors as fallback_failed', () => {
    const err = new Error('fetch failed');
    err.code = ERROR_CODES.PROVIDER_NETWORK_FAILED;
    const result = classifyProviderFailure(err);
    assert.equal(result.failureClass, FAILURE_CLASSES.FALLBACK_FAILED);
  });

  it('classifies unclassifiable errors as unknown', () => {
    const result = classifyProviderFailure(new Error('completely unrecognizable failure'));
    assert.equal(result.failureClass, FAILURE_CLASSES.UNKNOWN);
  });

  it('prefers errors with a Doppler error code over message regex', () => {
    const err = new Error('this message mentions out of memory but the code says otherwise');
    err.code = ERROR_CODES.GPU_DEVICE_LOST;
    const result = classifyProviderFailure(err);
    assert.equal(result.failureClass, FAILURE_CLASSES.GPU_DEVICE_LOST);
    assert.equal(result.failureCode, ERROR_CODES.GPU_DEVICE_LOST);
  });

  it('detects simulated failures via __dopplerFaultInjected', () => {
    const error = new Error('Simulated OOM');
    error.__dopplerFaultInjected = true;
    const result = classifyProviderFailure(error);
    assert.equal(result.isSimulated, true);
  });

  it('marks non-injected failures as not simulated', () => {
    const result = classifyProviderFailure(new Error('Real OOM out of memory'));
    assert.equal(result.isSimulated, false);
  });

  it('infers stage from error message', () => {
    assert.equal(classifyProviderFailure(new Error('Error during prefill phase')).stage, 'prefill');
    assert.equal(classifyProviderFailure(new Error('Decode step failed')).stage, 'decode');
    assert.equal(classifyProviderFailure(new Error('Model load error')).stage, 'load');
    assert.equal(classifyProviderFailure(new Error('Something else')).stage, 'unknown');
  });

  it('infers surface from error message', () => {
    assert.equal(classifyProviderFailure(new Error('WebGPU buffer error')).surface, 'webgpu');
    assert.equal(classifyProviderFailure(new Error('GPU memory exhausted')).surface, 'webgpu');
    assert.equal(classifyProviderFailure(new Error('OpenAI fallback failed')).surface, 'openai_compat');
    assert.equal(classifyProviderFailure(new Error('completely unrecognizable failure')).surface, 'unknown');
  });

  it('accepts context overrides for every FailureRecord field', () => {
    const result = classifyProviderFailure(new Error('out of memory'), {
      stage: 'prefill',
      surface: 'webgpu',
      device: 'apple-m1',
      modelId: 'gemma-4-e2b',
      runtimeProfile: 'approved_discrete',
      kernelPathId: 'attention.decode.v1',
    });
    assert.equal(result.stage, 'prefill');
    assert.equal(result.surface, 'webgpu');
    assert.equal(result.device, 'apple-m1');
    assert.equal(result.modelId, 'gemma-4-e2b');
    assert.equal(result.runtimeProfile, 'approved_discrete');
    assert.equal(result.kernelPathId, 'attention.decode.v1');
  });

  it('handles non-Error inputs', () => {
    const result = classifyProviderFailure('string error out of memory');
    assert.equal(result.failureClass, FAILURE_CLASSES.GPU_OOM);
    assert.equal(result.message, 'string error out of memory');
  });

  it('returns every FailureRecord field including runtimeProfile and kernelPathId', () => {
    const result = classifyProviderFailure(new Error('test'));
    assert.ok('failureClass' in result);
    assert.ok('failureCode' in result);
    assert.ok('stage' in result);
    assert.ok('surface' in result);
    assert.ok('device' in result);
    assert.ok('modelId' in result);
    assert.ok('runtimeProfile' in result);
    assert.ok('kernelPathId' in result);
    assert.ok('isSimulated' in result);
    assert.ok('message' in result);
  });

  it('defaults runtimeProfile and kernelPathId to null when not provided', () => {
    const result = classifyProviderFailure(new Error('test'));
    assert.equal(result.runtimeProfile, null);
    assert.equal(result.kernelPathId, null);
    assert.equal(result.device, null);
    assert.equal(result.modelId, null);
  });
});
