import assert from 'node:assert/strict';

globalThis.GPUMapMode = {
  READ: 0x0001,
};

const {
  readSampledTokenFromStagingBuffer,
  readMappedBufferCopy,
  readBatchTokensFromStagingBuffers,
} = await import('../../src/inference/pipelines/text/generator-steps.js');

function createRingTracker() {
  return {
    advanceCount: 0,
    advance() {
      this.advanceCount += 1;
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
  const ring = createRingTracker();
  const stagingBuffer = createStagingBuffer([0], {
    mapAsyncError: new Error('mapAsync failed'),
  });

  await assert.rejects(
    () => readSampledTokenFromStagingBuffer(stagingBuffer, {
      ownsStagingBuffer: true,
      ring,
    }),
    /mapAsync failed/
  );

  assert.equal(stagingBuffer.mapCount, 1);
  assert.equal(stagingBuffer.unmapCount, 0);
  assert.equal(stagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
}

{
  const ring = createRingTracker();
  const stagingBuffer = createStagingBuffer([42, 1, 7, 9, 0]);

  const result = await readSampledTokenFromStagingBuffer(stagingBuffer, {
    ownsStagingBuffer: false,
    hasFinitenessBuffer: true,
    ring,
  });

  assert.equal(result.nextToken, 42);
  assert.equal(result.finitenessStatus.triggered, true);
  assert.equal(result.finitenessStatus.metadata, ' (layer 7, step 9)');
  assert.equal(stagingBuffer.unmapCount, 1);
  assert.equal(stagingBuffer.destroyCount, 0);
  assert.equal(ring.advanceCount, 1);
}

{
  const ring = createRingTracker();
  const stagingBuffer = createStagingBuffer([17], {
    getMappedRangeError: new Error('mapped range failed'),
  });

  await assert.rejects(
    () => readSampledTokenFromStagingBuffer(stagingBuffer, {
      ownsStagingBuffer: true,
      ring,
    }),
    /mapped range failed/
  );

  assert.equal(stagingBuffer.unmapCount, 1);
  assert.equal(stagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
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
    ring,
  });

  assert.deepEqual(result.tokens, [5, 6, 7]);
  assert.deepEqual(Array.from(result.stopFlags || []), [0, 1, 0]);
  assert.equal(result.finitenessStatus.metadata, ' (layer 3, step 4)');
  assert.equal(tokensStagingBuffer.unmapCount, 1);
  assert.equal(stopStagingBuffer.unmapCount, 1);
  assert.equal(tokensStagingBuffer.destroyCount, 1);
  assert.equal(stopStagingBuffer.destroyCount, 1);
  assert.equal(finitenessStagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
}

{
  const ring = createRingTracker();
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
      ring,
    }),
    /batch map failed/
  );

  assert.equal(tokensStagingBuffer.destroyCount, 1);
  assert.equal(stopStagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
}

{
  const ring = createRingTracker();
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
      ring,
    }),
    /batch range failed/
  );

  assert.equal(tokensStagingBuffer.unmapCount, 1);
  assert.equal(stopStagingBuffer.unmapCount, 1);
  assert.equal(tokensStagingBuffer.destroyCount, 1);
  assert.equal(stopStagingBuffer.destroyCount, 1);
  assert.equal(ring.advanceCount, 1);
}

console.log('generator-steps-cleanup.test: ok');
