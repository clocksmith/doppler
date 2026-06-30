import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const generatorStepsSource = readFileSync(
  new URL('../../src/inference/pipelines/text/generator-steps.js', import.meta.url),
  'utf8'
);

globalThis.GPUMapMode = {
  READ: 0x0001,
};

const {
  createStopTokenLookup,
  findInvalidGeneratedToken,
  readSampledTokenFromStagingBuffer,
  readMappedBufferCopy,
  readBatchTokensFromStagingBuffers,
  resolveBatchStop,
  shouldUseFusedDecodeSampling,
} = await import('../../src/inference/pipelines/text/generator-steps.js');

function createRingTracker() {
  return {
    advanceCount: 0,
    advance() {
      this.advanceCount += 1;
    },
  };
}

function createCleanupRecorder() {
  return {
    calls: [],
    completeDeferredCleanup(options = {}) {
      this.calls.push({ discardPooled: options.discardPooled === true });
    },
  };
}

function createStagingBuffer(words, options = {}) {
  const array = words instanceof Uint32Array ? words : new Uint32Array(words);
  return {
    destroyCount: 0,
    mapCount: 0,
    unmapCount: 0,
    async mapAsync() {
      this.mapCount += 1;
      if (options.mapAsyncError) {
        throw options.mapAsyncError;
      }
    },
    getMappedRange() {
      if (options.getMappedRangeError) {
        throw options.getMappedRangeError;
      }
      return array.buffer;
    },
    unmap() {
      this.unmapCount += 1;
    },
    destroy() {
      this.destroyCount += 1;
    },
  };
}

{
  assert.ok(
    generatorStepsSource.includes("createCommandRecorder('decode', { recordLabels: opts.debug === true }, device)"),
    'single-token decode recorder must skip label collection outside debug runs'
  );
  assert.ok(
    generatorStepsSource.includes("createCommandRecorder('batch_decode', { recordLabels: opts.debug === true }, device)"),
    'batch decode recorder must skip label collection outside debug runs'
  );
  assert.equal(
    generatorStepsSource.includes("createProfilingRecorder('batch_decode', device)"),
    true,
    'batch decode profiling must keep labeled profile recorder path'
  );
  assert.ok(
    generatorStepsSource.includes('tokens.subarray(0, actualCount)'),
    'batch decode must return accepted token views without copying a second typed array'
  );
}

{
  const ring = createRingTracker();
  const cleanupRecorder = createCleanupRecorder();
  const stagingBuffer = createStagingBuffer([0], {
    mapAsyncError: new Error('mapAsync failed'),
  });

  await assert.rejects(
    () => readSampledTokenFromStagingBuffer(stagingBuffer, {
      ownsStagingBuffer: true,
      cleanupRecorder,
      ring,
    }),
    /mapAsync failed/
  );

  assert.equal(stagingBuffer.mapCount, 1);
  assert.equal(stagingBuffer.unmapCount, 0);
  assert.equal(stagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
  assert.deepEqual(cleanupRecorder.calls, [{ discardPooled: true }]);
}

{
  const ring = createRingTracker();
  const cleanupRecorder = createCleanupRecorder();
  const stagingBuffer = createStagingBuffer([42, 1, 7, 9, 0]);

  const result = await readSampledTokenFromStagingBuffer(stagingBuffer, {
    ownsStagingBuffer: false,
    hasFinitenessBuffer: true,
    cleanupRecorder,
    ring,
  });

  assert.equal(result.nextToken, 42);
  assert.equal(result.finitenessStatus.triggered, true);
  assert.equal(result.finitenessStatus.metadata, ' (layer 7, step 9)');
  assert.ok(Number.isFinite(result.timing.mapWaitMs));
  assert.ok(Number.isFinite(result.timing.cleanupMs));
  assert.ok(Number.isFinite(result.timing.copyMs));
  assert.equal(stagingBuffer.unmapCount, 1);
  assert.equal(stagingBuffer.destroyCount, 0);
  assert.equal(ring.advanceCount, 1);
  assert.deepEqual(cleanupRecorder.calls, [{ discardPooled: false }]);
}

{
  const ring = createRingTracker();
  const cleanupRecorder = createCleanupRecorder();
  const stagingBuffer = createStagingBuffer([17], {
    getMappedRangeError: new Error('mapped range failed'),
  });

  await assert.rejects(
    () => readSampledTokenFromStagingBuffer(stagingBuffer, {
      ownsStagingBuffer: true,
      cleanupRecorder,
      ring,
    }),
    /mapped range failed/
  );

  assert.equal(stagingBuffer.unmapCount, 1);
  assert.equal(stagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
  assert.deepEqual(cleanupRecorder.calls, [{ discardPooled: false }]);
}

{
  const stagingBuffer = createStagingBuffer([3, 4], {
    mapAsyncError: new Error('debug map failed'),
  });

  await assert.rejects(
    () => readMappedBufferCopy(stagingBuffer),
    /debug map failed/
  );

  assert.equal(stagingBuffer.unmapCount, 0);
  assert.equal(stagingBuffer.destroyCount, 1);
}

{
  const stagingBuffer = createStagingBuffer([11, 12], {
    getMappedRangeError: new Error('debug range failed'),
  });

  await assert.rejects(
    () => readMappedBufferCopy(stagingBuffer),
    /debug range failed/
  );

  assert.equal(stagingBuffer.unmapCount, 1);
  assert.equal(stagingBuffer.destroyCount, 1);
}

{
  const ring = createRingTracker();
  const cleanupRecorder = createCleanupRecorder();
  const tokensStagingBuffer = createStagingBuffer([5, 6, 7]);
  const stopStagingBuffer = createStagingBuffer([0, 1, 0]);
  const finitenessStagingBuffer = createStagingBuffer([1, 3, 4, 0]);

  const result = await readBatchTokensFromStagingBuffers({
    tokensStagingBuffer,
    stopStagingBuffer,
    finitenessStagingBuffer,
    tokenCount: 3,
    ownsTokensStaging: true,
    ownsStopStaging: true,
    cleanupRecorder,
    ring,
  });

  assert.deepEqual(Array.from(result.tokens), [5, 6, 7]);
  assert.deepEqual(Array.from(result.stopFlags || []), [0, 1, 0]);
  assert.equal(result.finitenessStatus.metadata, ' (layer 3, step 4)');
  assert.ok(Number.isFinite(result.timing.mapWaitMs));
  assert.ok(Number.isFinite(result.timing.cleanupMs));
  assert.ok(Number.isFinite(result.timing.copyMs));
  assert.equal(tokensStagingBuffer.unmapCount, 1);
  assert.equal(stopStagingBuffer.unmapCount, 1);
  assert.equal(tokensStagingBuffer.destroyCount, 1);
  assert.equal(stopStagingBuffer.destroyCount, 1);
  assert.equal(finitenessStagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
  assert.deepEqual(cleanupRecorder.calls, [{ discardPooled: false }]);
}

{
  const ring = createRingTracker();
  const cleanupRecorder = createCleanupRecorder();
  const tokensStagingBuffer = createStagingBuffer([1, 2], {
    mapAsyncError: new Error('batch map failed'),
  });
  const stopStagingBuffer = createStagingBuffer([0, 0]);

  await assert.rejects(
    () => readBatchTokensFromStagingBuffers({
      tokensStagingBuffer,
      stopStagingBuffer,
      tokenCount: 2,
      ownsTokensStaging: true,
      ownsStopStaging: true,
      cleanupRecorder,
      ring,
    }),
    /batch map failed/
  );

  assert.equal(tokensStagingBuffer.destroyCount, 1);
  assert.equal(stopStagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
  assert.deepEqual(cleanupRecorder.calls, [{ discardPooled: true }]);
}

{
  const ring = createRingTracker();
  const cleanupRecorder = createCleanupRecorder();
  const tokensStagingBuffer = createStagingBuffer([1, 2]);
  const stopStagingBuffer = createStagingBuffer([0, 0], {
    getMappedRangeError: new Error('batch range failed'),
  });

  await assert.rejects(
    () => readBatchTokensFromStagingBuffers({
      tokensStagingBuffer,
      stopStagingBuffer,
      tokenCount: 2,
      ownsTokensStaging: true,
      ownsStopStaging: true,
      cleanupRecorder,
      ring,
    }),
    /batch range failed/
  );

  assert.equal(tokensStagingBuffer.unmapCount, 1);
  assert.equal(stopStagingBuffer.unmapCount, 1);
  assert.equal(tokensStagingBuffer.destroyCount, 1);
  assert.equal(stopStagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
  assert.deepEqual(cleanupRecorder.calls, [{ discardPooled: false }]);
}

{
  const ring = createRingTracker();
  const tokensStagingBuffer = createStagingBuffer([5, 6, 7]);
  const finitenessStagingBuffer = createStagingBuffer([0, 0, 0, 0]);

  const result = await readBatchTokensFromStagingBuffers({
    tokensStagingBuffer,
    finitenessStagingBuffer,
    tokenCount: 3,
    ownsTokensStaging: true,
    ownsFinitenessStaging: false,
    ring,
  });

  assert.deepEqual(Array.from(result.tokens), [5, 6, 7]);
  assert.equal(finitenessStagingBuffer.destroyCount, 0);
  assert.equal(finitenessStagingBuffer.unmapCount, 1);
  assert.equal(ring.advanceCount, 1);
}

{
  const ring = createRingTracker();
  const cleanupRecorder = createCleanupRecorder();
  const tokensStagingBuffer = createStagingBuffer([5, 6, 7, 1, 3, 4, 0]);
  const finitenessStagingBuffer = createStagingBuffer([1, 9, 9, 0]);

  const result = await readBatchTokensFromStagingBuffers({
    tokensStagingBuffer,
    finitenessStagingBuffer,
    finitenessOffsetBytes: 12,
    tokenCount: 3,
    ownsTokensStaging: false,
    ownsFinitenessStaging: false,
    cleanupRecorder,
    ring,
  });

  assert.deepEqual(Array.from(result.tokens), [5, 6, 7]);
  assert.equal(result.finitenessStatus.triggered, true);
  assert.equal(result.finitenessStatus.metadata, ' (layer 3, step 4)');
  assert.equal(tokensStagingBuffer.mapCount, 1);
  assert.equal(tokensStagingBuffer.unmapCount, 1);
  assert.equal(tokensStagingBuffer.destroyCount, 0);
  assert.equal(finitenessStagingBuffer.mapCount, 0);
  assert.equal(finitenessStagingBuffer.unmapCount, 0);
  assert.equal(finitenessStagingBuffer.destroyCount, 0);
  assert.equal(ring.advanceCount, 1);
  assert.deepEqual(cleanupRecorder.calls, [{ discardPooled: false }]);
}

{
  const oneTokenLookup = createStopTokenLookup([7], null);
  assert.equal(oneTokenLookup.firstTokenId, 7);
  assert.equal(oneTokenLookup.secondTokenId, null);
  assert.equal(oneTokenLookup.tokenSet, null);
  assert.equal(
    resolveBatchStop(new Uint32Array([5, 6, 7, 8]), null, oneTokenLookup),
    3
  );

  const twoTokenLookup = createStopTokenLookup([9], 7);
  assert.equal(twoTokenLookup.firstTokenId, 9);
  assert.equal(twoTokenLookup.secondTokenId, 7);
  assert.equal(twoTokenLookup.tokenSet, null);
  assert.equal(
    resolveBatchStop(new Uint32Array([5, 7, 9]), null, twoTokenLookup),
    2
  );

  const setLookup = createStopTokenLookup([9, 7, 5], 2);
  assert.ok(setLookup.tokenSet instanceof Set);
  assert.equal(
    resolveBatchStop(new Uint32Array([1, 3, 2, 9]), null, setLookup),
    3
  );
  assert.equal(
    resolveBatchStop(
      new Uint32Array([1, 3, 2, 9]),
      new Uint32Array([0, 1, 0, 0]),
      setLookup
    ),
    2
  );

  assert.throws(
    () => createStopTokenLookup(null, 7),
    /stopTokenIds must be an array/
  );
}

{
  assert.equal(
    findInvalidGeneratedToken([11, 12, 13], 32, null),
    null
  );
  assert.deepEqual(
    findInvalidGeneratedToken([11, 0, 13], 32, null),
    { index: 1, tokenId: 0 }
  );
  assert.deepEqual(
    findInvalidGeneratedToken([11, 7, 13], 32, 7),
    { index: 1, tokenId: 7 }
  );
  assert.deepEqual(
    findInvalidGeneratedToken([11, 99], 32, null),
    { index: 1, tokenId: 99 }
  );
}

{
  assert.equal(
    shouldUseFusedDecodeSampling({
      recorderEnabled: true,
      gpuSamplingEnabled: true,
      fusedDecodeDisabled: false,
      layerTypes: ['attention', 'mlp'],
    }),
    true
  );
  assert.equal(
    shouldUseFusedDecodeSampling({
      recorderEnabled: true,
      gpuSamplingEnabled: true,
      fusedDecodeDisabled: false,
      layerTypes: ['conv', 'attention'],
    }),
    false
  );
  assert.equal(
    shouldUseFusedDecodeSampling({
      recorderEnabled: true,
      gpuSamplingEnabled: true,
      fusedDecodeDisabled: true,
      layerTypes: ['attention'],
    }),
    false
  );
}

console.log('generator-steps-cleanup.test: ok');
