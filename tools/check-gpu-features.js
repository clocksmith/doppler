// Quick GPU feature probe — run with: node tools/check-gpu-features.js
const { create } = await import('webgpu');
const instance = create(['enable-dawn-features=allow_unsafe_apis']);
const globals = instance.globals || {};
for (const [name, value] of Object.entries(globals)) {
  if (value != null && globalThis[name] === undefined) {
    globalThis[name] = value;
  }
}
const gpu = instance.gpu || instance;
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) { console.log('no adapter'); process.exit(); }
const feats = [...(adapter.features ?? [])];
console.log('adapter features:', feats.join(', '));
console.log('has shader-f16:', feats.includes('shader-f16'));

if (feats.includes('shader-f16')) {
  const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
  try {
    device.createShaderModule({ code: 'enable f16;\n@compute @workgroup_size(1) fn _probe() { var x: f16 = 1.0h; }' });
    console.log('f16 shader probe: OK');
  } catch(e) {
    console.log('f16 shader probe failed:', e.message);
  }
  device.destroy();
} else {
  console.log('f16 not available on adapter');
}
