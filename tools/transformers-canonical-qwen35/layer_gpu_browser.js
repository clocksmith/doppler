import * as ort from '/ort/ort.webgpu.min.mjs';

async function main() {
  const config = await (await fetch('/config.json')).json();
  const fixtures = await (await fetch('/model/fixtures.json')).json();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No physical WebGPU adapter');
  const info = adapter.info;
  if (!info.vendor?.toLowerCase().includes(config.requiredVendor.toLowerCase())) {
    throw new Error(`Unexpected physical adapter: ${JSON.stringify(info)}`);
  }
  const device = await adapter.requestDevice();
  const gpuErrors = [];
  device.addEventListener('uncapturederror', event => gpuErrors.push(String(event.error)));
  ort.env.webgpu.device = device;
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'verbose';
  const options = {
    executionProviders: ['webgpu'],
    extra: { session: { disable_cpu_ep_fallback: '1' } },
    preferredOutputLocation: { present_conv: 'gpu-buffer', present_recurrent: 'gpu-buffer' },
  };
  const sessions = {};
  const checks = [];
  try {
    for (const mode of ['prefill', 'decode']) {
      sessions[mode] = await ort.InferenceSession.create(`/model/${mode}.onnx`, options);
    }
    for (const fixture of fixtures.cases) {
      const payload = await (await fetch(`/model/${fixture.file}`)).arrayBuffer();
      const checksum = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', payload)),
        value => value.toString(16).padStart(2, '0')).join('');
      if (checksum !== fixture.sha256) throw new Error(`Fixture bytes differ: ${fixture.label}`);
      const feeds = {};
      let outputs;
      try {
        for (const tensor of fixture.tensors.filter(value => value.role === 'input')) {
          feeds[tensor.name] = new ort.Tensor(tensor.dtype,
            new Float32Array(payload.slice(tensor.offset, tensor.offset + tensor.bytes)), tensor.shape);
        }
        outputs = await sessions[fixture.mode].run(feeds);
        for (const tensor of fixture.tensors.filter(value => value.role === 'expected')) {
          const output = outputs[tensor.name];
          const location = output.location;
          if (tensor.name.startsWith('present_') && location !== 'gpu-buffer') {
            throw new Error(`State output not GPU resident: ${tensor.name} ${location}`);
          }
          const actual = await output.getData();
          const expected = new Float32Array(payload.slice(tensor.offset, tensor.offset + tensor.bytes));
          if (actual.length !== expected.length) throw new Error(`Shape mismatch: ${tensor.name}`);
          let maximum = 0, total = 0, failures = 0;
          for (let index = 0; index < actual.length; index++) {
            const error = Math.abs(actual[index] - expected[index]);
            if (!Number.isFinite(actual[index]) || error > fixtures.absoluteTolerance + fixtures.relativeTolerance * Math.abs(expected[index])) failures++;
            maximum = Math.max(maximum, error);
            total += error;
          }
          const check = { label: fixture.label, mode: fixture.mode, tensor: tensor.name, location,
            maxAbsoluteError: maximum, meanAbsoluteError: total / actual.length, failedElements: failures, pass: failures === 0 };
          checks.push(check);
          if (failures) throw new Error(`GPU parity failed: ${JSON.stringify(check)}`);
        }
      } finally {
        for (const tensor of Object.values(outputs ?? {})) tensor.dispose();
        for (const tensor of Object.values(feeds)) tensor.dispose();
      }
    }
    await device.queue.onSubmittedWorkDone();
    if (gpuErrors.length) throw new Error(gpuErrors.join('\n'));
    return { schema: 'doppler.qwen35-linear-gpu.v1', pass: true, options, checks, gpuErrors,
      adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
      claim: 'Untimed canonical single-layer ONNX GPU admission/parity only; no full model or speed proof' };
  } finally {
    for (const session of Object.values(sessions)) await session.release();
    device.destroy();
  }
}

try { globalThis.RESULT = await main(); }
catch (error) { globalThis.RESULT = { pass: false, error: String(error), stack: error.stack }; }
