// Read-only tensor copies at the unchanged installed layer-completion boundary.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

export async function observeStage({ page, runtimeBaseUrl, runtimeDirectory, evaluate, evidence, persist, inputsPath, execute }) {
  const source = await readFile(runtimeDirectory + '/src/inference/pipelines/text/partition-execution.js', 'utf8');
  const matches = source.split('\n').flatMap((line, index) => line.trim() === 'hidden = next;' ? [index] : []);
  assert.equal(matches.length, 1);
  const inputsBytes = await readFile(inputsPath), cases = JSON.parse(inputsBytes);
  evidence.diagnostic = { scope: 'Same installed output-stage inputs and request order; observation copies only',
    sourceSha256: hash(source), inputsSha256: hash(inputsBytes), cases: [], snapshots: [] };
  const cdp = await page.context().newCDPSession(page);
  evidence.diagnostic.debuggerPauses = [];
  cdp.on('Debugger.paused', event => {
    evidence.diagnostic.debuggerPauses.push({ reason: event.reason,
      location: event.callFrames?.[0]?.location, data: event.data });
    void cdp.send('Debugger.resume').catch(error => {
      evidence.diagnostic.resumeError = error.message;
    });
  });
  await cdp.send('Debugger.enable');
  await page.evaluate(() => { globalThis.stageSnapshots = []; globalThis.stageSnapshotBytes = 0; });
  const condition = `(() => {
    if (globalThis.stageDiagnosticRequest !== 2) return false;
    const size = byteLength;
    if (globalThis.stageSnapshotBytes + size > 33554432) throw Error('Diagnostic copy bound exceeded');
    const target = recorder.device.createBuffer({label:'diagnostic_layer_snapshot',size,usage:9});
    globalThis.stageSnapshots.push({layer,position:state.currentSeqLen,numTokens,byteLength:size,buffer:target});
    globalThis.stageSnapshotBytes += size;
    if (recordLayers) recorder.getEncoder().copyBufferToBuffer(next,0,target,0,size);
    else { const encoder=recorder.device.createCommandEncoder(); encoder.copyBufferToBuffer(next,0,target,0,size); recorder.device.queue.submit([encoder.finish()]); }
    return false;
  })()`;
  let breakpoint;
  try {
    breakpoint = await cdp.send('Debugger.setBreakpointByUrl', {
      url: new URL('inference/pipelines/text/partition-execution.js', runtimeBaseUrl).href,
      lineNumber: matches[0], condition,
    });
    assert(breakpoint.locations.length, 'Installed execution boundary is not loaded');
    const loaded = await cdp.send('Debugger.getScriptSource', { scriptId: breakpoint.locations[0].scriptId });
    assert.equal(hash(loaded.scriptSource), hash(source));
    let continuation = null, priorRequest = null;
    for (const row of cases) {
      if (priorRequest !== row.request) continuation = null;
      await page.evaluate(request => { globalThis.stageDiagnosticRequest = request; }, row.request);
      const result = await evaluate(page, `diagnostic-request-${row.request}-step-${row.parameters.step}`, execute,
        { parameters: row.parameters, activationInput: row.activationInput, continuation });
      continuation = result.continuation; priorRequest = row.request;
      evidence.diagnostic.cases.push({ request: row.request, step: row.parameters.step, tokenId: result.tokenId,
        stopReason: result.stopReason, logits: result.logits, matchesOriginalBytes: result.logits === row.expectedLogits });
      if (result.done || row === cases.at(-1)) await evaluate(page, 'close-diagnostic-attempt',
        identity => resident.closeAttempt({ identity }), row.parameters.identity, true);
      await persist();
    }
    evidence.diagnostic.snapshots = await evaluate(page, 'read-diagnostic-snapshots', async () => {
      const result = [];
      for (const { buffer, ...row } of stageSnapshots) {
        try {
          await buffer.mapAsync(1);
          const bytes = buffer.getMappedRange().slice(0);
          result.push({ ...row, data: encodeBytes(bytes) });
        } finally { buffer.destroy(); }
      }
      stageSnapshots.length = 0;
      return result;
    });
    assert.equal(evidence.diagnostic.snapshots.length, 8, 'One full input chunk expected for each output-stage layer');
  } finally {
    if (breakpoint) await cdp.send('Debugger.removeBreakpoint', { breakpointId: breakpoint.breakpointId });
    await cdp.detach();
    await page.evaluate(() => { for (const row of globalThis.stageSnapshots ?? []) row.buffer.destroy(); });
  }
}
