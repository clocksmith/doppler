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
  const policyBytes = await readFile(new URL('./capture-policy.json', import.meta.url));
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
    globalThis.stageCaptureError = null;
    globalThis.stageCaptureDiagnostics = { enabled: true, captureConfig,
      emitter: { emitRecord(stage, record) { stageCaptureRecords.push({ stage, ...record }); } } };
  }, policy);
  let breakpoint;
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
    for (const row of cases) {
      const result = await evaluate(page, 'canonical-observed-long-prefill', execute,
        { parameters: row.parameters, activationInput: row.activationInput, continuation: null });
      evidence.diagnostic.cases.push({ request: row.request, step: row.parameters.step, tokenId: result.tokenId,
        logits: result.logits, stopReason: result.stopReason, matchesOriginalBytes: result.logits === row.expectedLogits });
      await persist();
    }
    evidence.diagnostic.snapshots = await evaluate(page, 'canonical-snapshot-readback', () => {
      if (stageCaptureError) throw Error(stageCaptureError);
      let bytes = 0;
      return stageCaptureRecords.filter(row => row.capturePolicy === 'full').map(row => {
        const { capture, ...meta } = row;
        if (capture.error || !Array.isArray(capture.storageComponents)) throw Error('Missing paired canonical snapshot');
        const storage = Float32Array.from(capture.storageComponents);
        bytes += storage.byteLength;
        if (bytes > 33554432) throw Error('Canonical snapshot byte budget exceeded');
        return { ...meta, layer: row.layerIdx, shape: capture.shape, byteLength: storage.byteLength,
          data: encodeBytes(storage), storageDigest: capture.storageDigest };
      });
    }, undefined, false, false);
    assert.equal(evidence.diagnostic.snapshots.length, 9);
    await persist();
    await evaluate(page, 'close-canonical-diagnostic-attempt', identity => resident.closeAttempt({ identity }),
      cases.at(-1).parameters.identity, true, false);
  } finally {
    if (breakpoint) await cdp.send('Debugger.removeBreakpoint', { breakpointId: breakpoint.breakpointId });
    await cdp.detach();
  }
}
