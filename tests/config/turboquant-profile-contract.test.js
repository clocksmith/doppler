// TurboQuant runtime profiles must stay within what the TurboQuant decode
// kernels can execute: the KV cache dtype matches the kernels' f16 KV input,
// and maxSeqLen does not exceed the MAX_KV_LEN ceiling compiled into the
// kernel the profile's layout and quantization mode select. Values are read
// from the WGSL source, not copied.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const PROFILES_DIR = 'src/config/runtime/profiles';
const KERNELS_DIR = 'src/gpu/kernels';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function turboQuantMode(kvcache) {
  if (kvcache.layout === 'tiered') {
    const compression = kvcache.tiering?.compression;
    if (!String(compression?.mode ?? '').startsWith('turboquant')) return null;
    return compression.prodMode === true ? 'turboquant_prod' : 'turboquant';
  }
  const mode = kvcache.quantization?.mode;
  return String(mode ?? '').startsWith('turboquant') ? mode : null;
}

function decodeKernelCeiling(layout, mode) {
  const file = `attention_decode_${layout}_${mode}_f16kv.wgsl`;
  const kernelPath = path.join(KERNELS_DIR, file);
  assert.ok(fs.existsSync(kernelPath), `no TurboQuant decode kernel for layout=${layout} mode=${mode} (${file})`);
  const match = /const MAX_KV_LEN: u32 = (\d+)u;/.exec(fs.readFileSync(kernelPath, 'utf8'));
  assert.ok(match, `${file} must declare MAX_KV_LEN`);
  return { file, maxKvLen: Number(match[1]) };
}

const profiles = fs.readdirSync(PROFILES_DIR)
  .filter((name) => name.endsWith('.json'))
  .map((name) => ({ name, profile: readJson(path.join(PROFILES_DIR, name)) }))
  .filter(({ profile }) => {
    const kvcache = profile.runtime?.inference?.session?.kvcache;
    return kvcache && turboQuantMode(kvcache) !== null;
  });
assert.ok(profiles.length > 0, 'at least one TurboQuant runtime profile must exist');

for (const { name, profile } of profiles) {
  assert.equal(profile.inference, undefined, `${name}: must not use legacy top-level inference`);
  const kvcache = profile.runtime.inference.session.kvcache;
  const mirror = profile.runtime.inference.kvcache;
  if (mirror !== undefined) {
    // Whichever copy a reader resolves, it must not exceed the kernel ceiling.
    assert.equal(mirror.maxSeqLen, kvcache.maxSeqLen, `${name}: runtime.inference.kvcache.maxSeqLen disagrees with session`);
    assert.equal(mirror.kvDtype, kvcache.kvDtype, `${name}: runtime.inference.kvcache.kvDtype disagrees with session`);
  }
  const mode = turboQuantMode(kvcache);
  const { file, maxKvLen } = decodeKernelCeiling(kvcache.layout, mode);
  assert.equal(kvcache.kvDtype, 'f16', `${name}: ${file} reads an f16 KV cache`);
  assert.ok(Number.isInteger(kvcache.maxSeqLen) && kvcache.maxSeqLen > 0, `${name}: maxSeqLen must be a positive integer`);
  assert.ok(
    kvcache.maxSeqLen <= maxKvLen,
    `${name}: maxSeqLen ${kvcache.maxSeqLen} exceeds ${file} MAX_KV_LEN ${maxKvLen}`
  );
  if (kvcache.layout === 'contiguous') {
    assert.equal(kvcache.quantization.prodMode, mode === 'turboquant_prod', `${name}: prodMode must agree with quantization mode`);
  }
}

console.log(`turboquant-profile-contract.test: ok (${profiles.length} profiles)`);
