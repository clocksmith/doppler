import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { chromium } from 'playwright';

// Explicitly opt in: this uses two physical hosts serially, never a software GPU.
if (!process.env.DOPPLER_RMS_CAPTURE) {
  console.log('rmsnorm-dispatch-physical: SKIP (DOPPLER_RMS_CAPTURE not supplied)');
} else {
  const { DOPPLER_RMS_CAPTURE: capturePath, DOPPLER_INSTALLED_ROOT: installedRoot,
    DOPPLER_EXECUTOR_WS: remote, DOPPLER_RMS_OUT: output } = process.env;
  assert(installedRoot && remote && output, 'Installed package root, physical browser endpoint and output are required');
  const captureBytes = await readFile(capturePath), capture = JSON.parse(captureBytes);
  const operands = capture.inputObservations[0];
  const decode = (role, Type) => {
    const item = operands.find(item => item.role === role), bytes = Buffer.from(item.data, 'base64');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), item.sha256);
    return Array.from(new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
  };
  const packageInfo = JSON.parse(await readFile(resolve(installedRoot, 'package.json')));
  const helper = await readFile(new URL('../kernels/rmsnorm-dispatch-reuse.js', import.meta.url), 'utf8');
  const paths = ['config/kernel-path-loader.js', 'gpu/kernels/rmsnorm.js'];
  const sources = await Promise.all(paths.map(async path => ({ path,
    body: await readFile(new URL('../../src/' + path, import.meta.url), 'utf8') })));
  const graph = capture.observations[0][0].data.kernelPath;
  assert(graph && graph.activationDtype === 'f32', 'Captured resolved graph required');
  const base = process.env.DOPPLER_BROWSER_BASE_URL ?? 'http://localhost:8000';
  const moduleBase = `${base}/vendor/doppler/${packageInfo.version}/src/`;
  const runs = [];
  for (const platform of ['mac', 'linux']) for (const candidate of [false, true]) {
    const browser = platform === 'mac' ? await chromium.launch({ headless: true,
      args: ['--enable-unsafe-webgpu', '--use-angle=metal'] }) : await chromium.connect(remote);
    let context;
    try {
      context = await browser.newContext();
      await context.route('**/rmsnorm-dispatch-reuse.js', route => route.fulfill({ contentType: 'text/javascript', body: helper }));
      if (candidate) await context.route('**/vendor/doppler/**', async route => {
        const source = sources.find(source => new URL(route.request().url()).pathname.endsWith('/' + source.path));
        if (source) await route.fulfill({ contentType: 'text/javascript', body: source.body });
        else await route.continue();
      });
      const page = await context.newPage(); await page.goto(`${base}/config/chat-files.json`);
      const result = await page.evaluate(async request => {
        const { auditRMSNormReuse } = await import('/rmsnorm-dispatch-reuse.js');
        return auditRMSNormReuse(request);
      }, { moduleBase, input: decode('input-last-row', Float32Array), weights: decode('weight', Uint16Array), graph });
      runs.push({ platform, candidate, browser: browser.version(), ...result });
      await writeFile(output, JSON.stringify({ scope: 'Isolated physical wrapper creation/reuse reproduction; not model or package acceptance',
        packageVersion: packageInfo.version, captureSha256: createHash('sha256').update(captureBytes).digest('hex'),
        helperSha256: createHash('sha256').update(helper).digest('hex'),
        sourceOverrides: sources.map(s => ({ path: s.path, sha256: createHash('sha256').update(s.body).digest('hex') })), runs }, null, 2));
      assert.deepEqual(result.errors, []); assert.equal(result.calls.length, 12);
      for (const call of result.calls) {
        assert.equal(call.dispatches.length, 1);
        const dispatch = call.dispatches[0], pipeline = result.pipelines[dispatch.pipeline];
        assert.equal(pipeline.entryPoint, candidate ? 'main' : 'main_subgroup');
        assert.equal(pipeline.entryPoint, call.requested.entryPoint);
        for (const [key, value] of Object.entries(call.requested.constants)) assert.equal(pipeline.constants[key], Number(value));
        if (call.cachedPipelineId !== null) assert.equal(dispatch.pipeline, call.cachedPipelineId);
        const uniform = dispatch.group.find(b => b.binding === 0);
        assert.equal(uniform.uniformBytes.length, 32, 'Observe actual uploaded uniform bytes');
        const view = new DataView(Uint8Array.from(uniform.uniformBytes).buffer);
        assert.equal(view.getUint32(0, true), 1024); assert.equal(view.getUint32(4, true), 1);
        assert.equal(view.getFloat32(8, true), Math.fround(call.epsilon));
        assert.equal(view.getFloat32(20, true), call.outputScale);
        for (const [binding, expected, minimum] of [[1, call.input, 4096], [2, call.weight, 2048]]) {
          const actual = dispatch.group.find(b => b.binding === binding);
          assert.equal(actual.buffer.id, expected.id); assert.equal(actual.offset, 0); assert(actual.size >= minimum);
          assert(actual.offset + actual.size <= actual.buffer.bytes);
        }
        assert.deepEqual(dispatch.dimensions, [1, 1, 1]);
      }
      for (const recorded of [false, true]) {
        const rows = result.calls.filter(call => call.recorded === recorded), initial = rows[0];
        for (const index of [1, 2, 5]) assert.equal(rows[index].dispatches[0].pipeline, initial.dispatches[0].pipeline);
        for (const index of [3, 4]) assert.notEqual(rows[index].dispatches[0].pipeline, initial.dispatches[0].pipeline);
        for (const index of [1, 5]) assert.equal(rows[index].outputSha256, initial.outputSha256);
        assert.notEqual(rows[2].outputSha256, initial.outputSha256, 'Changed uniforms affect output through a reused pipeline');
      }
      for (let index = 0; index < 6; index++) assert.equal(result.calls[index].outputSha256, result.calls[index + 6].outputSha256,
        'Immediate and recorded adapters preserve output through identical creation/reuse cases');
      console.log(JSON.stringify({ platform, candidate, calls: result.calls.length, pipelines: result.pipelines.length,
        cacheHits: result.calls.filter(c => c.cachedPipelineId !== null).length }));
    } finally { await context?.close(); await browser.close(); }
  }
}
