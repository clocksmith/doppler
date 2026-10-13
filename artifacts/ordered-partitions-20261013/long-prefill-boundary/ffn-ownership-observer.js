// Select the installed runtime's existing operator diagnostics; never replace kernels.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
export async function observeStage({ page, runtimeBaseUrl, runtimeDirectory, evaluate, evidence, persist, inputsPath, execute }) {
  const source = await readFile(runtimeDirectory + '/src/inference/pipelines/text/partition-execution.js', 'utf8');
  const lines = source.split('\n').flatMap((line, index) => line.trim() === 'context.currentTokenIds = input.tokenIds ?? null;' ? [index] : []);
  assert.equal(lines.length, 1);
  const inputs = await readFile(inputsPath), cases = JSON.parse(inputs);
  const policyBytes = await readFile(new URL('./ffn-ownership-policy.json', import.meta.url));
  const policy = JSON.parse(policyBytes);
  evidence.diagnostic = { scope: 'Fresh resident and original long-prefix input; installed canonical operator snapshots',
    sourceSha256: hash(source), inputsSha256: hash(inputs), policySha256: hash(policyBytes),
    cases: [], snapshots: [], debuggerPauses: [] };
  const cdp = await page.context().newCDPSession(page);
  cdp.on('Debugger.paused', event => {
    evidence.diagnostic.debuggerPauses.push({ reason: event.reason, location: event.callFrames?.[0]?.location });
    void cdp.send('Debugger.resume').catch(error => { evidence.diagnostic.resumeError = error.message; });
  });
  await cdp.send('Debugger.enable');
  await page.evaluate(captureConfig => {
    globalThis.stageCaptureRecords = [];
    globalThis.stageOwnershipEvents = [];
    globalThis.stageBufferIds = new WeakMap();
    globalThis.stageBufferId = buffer => {
      if (!stageBufferIds.has(buffer)) stageBufferIds.set(buffer, stageBufferIds.sizeCounter = (stageBufferIds.sizeCounter ?? 0) + 1);
      return { id: stageBufferIds.get(buffer), label: buffer.label, size: buffer.size, usage: buffer.usage };
    };
    globalThis.stageCaptureError = null;
    globalThis.stageCaptureInputPromises = [];
    globalThis.stageCaptureDiagnostics = { enabled: true, captureConfig,
      emitter: { emitRecord(stage, record) { stageCaptureRecords.push({ stage, ...record }); } } };
  }, policy);
  let breakpoint, inputBreakpoint;
  const ownershipBreakpoints = [];
  try {
    breakpoint = await cdp.send('Debugger.setBreakpointByUrl', {
      url: new URL('inference/pipelines/text/partition-execution.js', runtimeBaseUrl).href,
      lineNumber: lines[0], condition: `(() => { try {
        state.operatorDiagnostics = globalThis.stageCaptureDiagnostics;
        context.operatorDiagnostics = globalThis.stageCaptureDiagnostics;
      } catch (error) { globalThis.stageCaptureError = error.message; } return false; })()`,
    });
    assert(breakpoint.locations.length);
    assert(breakpoint.locations.every(row => row.lineNumber === lines[0]));
    const loaded = await cdp.send('Debugger.getScriptSource', { scriptId: breakpoint.locations[0].scriptId });
    assert.equal(hash(loaded.scriptSource), hash(source));
    evidence.diagnostic.breakpoint = breakpoint;
    const inputSource = await readFile(runtimeDirectory + '/src/inference/pipelines/text/layer.js', 'utf8');
    const inputLines = inputSource.split('\n').flatMap((line, index) =>
      line.trim() === "await runProbes('layer_in', inputBuffer, {" ? [index] : []);
    assert.equal(inputLines.length, 1);
    inputBreakpoint = await cdp.send('Debugger.setBreakpointByUrl', {
      url: new URL('inference/pipelines/text/layer.js', runtimeBaseUrl).href,
      lineNumber: inputLines[0], condition: `(() => { if (layerIdx === 17) {
        try { globalThis.stageCaptureInputPromises.push(runProbes('attn_input', inputBuffer, { layerIdx, numTokens, hiddenSize,
            recorder, operatorDiagnostics: context.operatorDiagnostics, dtype: inputTensor.dtype,
            precision: inputTensor.precision, components: inputTensor.components,
            elementStrideBytes: inputTensor.elementStrideBytes }).catch(error => { globalThis.stageCaptureError = error.message; }));
        } catch (error) { globalThis.stageCaptureError = error.message; }
      } return false; })()` });
    assert(inputBreakpoint.locations.length);
    assert(inputBreakpoint.locations.every(row => row.lineNumber === inputLines[0]));
    const loadedInput = await cdp.send('Debugger.getScriptSource', { scriptId: inputBreakpoint.locations[0].scriptId });
    assert.equal(hash(loadedInput.scriptSource), hash(inputSource));
    evidence.diagnostic.inputProbe = { sourceSha256: hash(inputSource), breakpoint: inputBreakpoint,
      scope: 'Existing canonical runProbes API submits a read-only input copy before immediate layer dispatch; await capture completion at export' };
    const observationPoints = [
      { file: 'src/inference/pipelines/text/probes.js', line: "const { layerIdx, numTokens, hiddenSize, probes, recorder, dtype = 'f32' } = options;",
        condition: `(() => { if ((options.layerIdx === 16 || options.layerIdx === 17) && stage === 'layer_out') {
          globalThis.stageTraceLayer = options.layerIdx;
          stageOwnershipEvents.push({ event: 'layer-output-probe', layer: options.layerIdx, buffer: stageBufferId(buffer),
            recorder: Boolean(options.recorder), count: stageCaptureRecords.length });
        } return false; })()` },
      { file: 'src/inference/pipelines/text/ffn/standard.js', line: 'if (normedTensor !== postAttn && !usedPostAttnNormStatsPrelude) {',
        condition: `(() => { if (layerIdx === 16 || layerIdx === 17) {
          stageOwnershipEvents.push({ event: 'ffn-before-cleanup', layer: layerIdx, ffnResidualFused, residualInputOwned,
            output: stageBufferId(output.buffer), normed: stageBufferId(normedTensor.buffer),
            postAttn: stageBufferId(postAttn.buffer), ffnOutput: stageBufferId(ffnOutput.buffer),
            residualInput: stageBufferId(residualInput.buffer),
            prenorm: prenormSumBuffer ? stageBufferId(prenormSumBuffer) : null });
        } return false; })()` },
      { file: 'src/inference/pipelines/text/ffn/standard.js', line: 'return output;',
        condition: `(() => { if (layerIdx === 16 || layerIdx === 17) {
          stageOwnershipEvents.push({ event: 'ffn-after-cleanup', layer: layerIdx, output: stageBufferId(output.buffer) });
        } return false; })()` },
      { file: 'src/memory/buffer-pool.js', line: 'this.#releaseTrackedBuffer(buffer, true);',
        condition: `(() => { if (globalThis.stageTraceLayer === 16 || globalThis.stageTraceLayer === 17) {
          stageOwnershipEvents.push({ event: 'pool-release', layer: stageTraceLayer, buffer: stageBufferId(buffer) });
        } return false; })()` },
    ];
    evidence.diagnostic.ownershipBreakpoints = [];
    for (const point of observationPoints) {
      const source = await readFile(runtimeDirectory + '/' + point.file, 'utf8');
      const rows = source.split('\n').flatMap((line, index) => line.trim() === point.line ? [index] : []);
      assert.equal(rows.length, 1);
      const observed = await cdp.send('Debugger.setBreakpointByUrl', {
        url: new URL(point.file.slice(4), runtimeBaseUrl).href, lineNumber: rows[0], condition: point.condition });
      ownershipBreakpoints.push(observed);
      assert(observed.locations.length);
      assert(observed.locations.every(row => row.lineNumber === rows[0]));
      const loaded = await cdp.send('Debugger.getScriptSource', { scriptId: observed.locations[0].scriptId });
      assert.equal(hash(loaded.scriptSource), hash(source));
      evidence.diagnostic.ownershipBreakpoints.push({ file: point.file, sourceSha256: hash(source), breakpoint: observed });
    }
    for (const row of cases) {
      const result = await evaluate(page, 'canonical-observed-long-prefill', execute,
        { parameters: row.parameters, activationInput: row.activationInput, continuation: null });
      evidence.diagnostic.cases.push({ request: row.request, step: row.parameters.step, tokenId: result.tokenId,
        logits: result.logits, stopReason: result.stopReason, matchesOriginalBytes: result.logits === row.expectedLogits });
      await persist();
    }
    evidence.diagnostic.export = await evaluate(page, 'canonical-snapshot-export', async output => {
      await Promise.all(globalThis.stageCaptureInputPromises);
      if (stageCaptureError) throw Error(stageCaptureError);
      const fs = process.getBuiltinModule('fs'), zlib = process.getBuiltinModule('zlib');
      const crypto = process.getBuiltinModule('crypto');
      let bytes = 0;
      const records = stageCaptureRecords.filter(row => row.capturePolicy === 'full').map((row, ordinal) => {
        const { capture, ...meta } = row;
        if (capture.error || !Array.isArray(capture.storageComponents)) throw Error('Missing paired canonical snapshot');
        const storage = Buffer.from(Float32Array.from(capture.storageComponents).buffer);
        bytes += storage.byteLength;
        if (bytes > 33554432) throw Error('Canonical snapshot byte budget exceeded');
        const storageDigest = 'sha256:' + crypto.createHash('sha256').update(storage).digest('hex');
        if (storageDigest !== capture.storageDigest) throw Error('Snapshot digest mismatch');
        return { ...meta, ordinal, layer: row.layerIdx, shape: capture.shape, byteLength: storage.byteLength,
          data: storage.toString('base64'), storageDigest };
      });
      if (records.length === 0) throw Error('No canonical captures');
      if (stageOwnershipEvents.length > 1000) throw Error('Ownership event budget exceeded');
      const raw = Buffer.from(JSON.stringify({ records, ownershipEvents: stageOwnershipEvents }));
      const compressed = zlib.gzipSync(raw);
      fs.writeFileSync(output, compressed, { flag: 'wx' });
      return { output, bytes, compressedBytes: compressed.byteLength, count: records.length,
        sha256: crypto.createHash('sha256').update(compressed).digest('hex'),
        targets: records.map(row => ({ stage: row.stage, layer: row.layer, shape: row.shape, storageDigest: row.storageDigest })) };
    }, '/var/tmp/doppler-long-prefill-boundary-20261013/ffn-ownership-captures.json.gz', false, false);
    await persist();
    await evaluate(page, 'close-canonical-diagnostic-attempt', identity => resident.closeAttempt({ identity }),
      cases.at(-1).parameters.identity, true, false);
  } finally {
    for (const observed of ownershipBreakpoints) await cdp.send('Debugger.removeBreakpoint', { breakpointId: observed.breakpointId });
    if (inputBreakpoint) await cdp.send('Debugger.removeBreakpoint', { breakpointId: inputBreakpoint.breakpointId });
    if (breakpoint) await cdp.send('Debugger.removeBreakpoint', { breakpointId: breakpoint.breakpointId });
    await cdp.detach();
  }
}
