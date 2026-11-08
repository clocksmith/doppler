import assert from 'node:assert/strict';
import { resolveAudioFeatureExtractor } from '../../src/inference/pipelines/text/config/normalization.js';
import { extractLogMelSpectrogram, buildMelFilterBank } from '../../src/inference/pipelines/audio/mel.js';

// The Gemma 4 reference front end (transformers Gemma4AudioFeatureExtractor
// defaults): 128 mel bins, 16 kHz, 20 ms frames every 10 ms, no preemphasis,
// FFT 512, 0-8000 Hz, log(mel + 1e-3). The old extractor hard-coded 80 bins
// and a 25 ms window, which no manifest could change.
const gemma4 = {
  feature_size: 128,
  sampling_rate: 16000,
  frame_length_ms: 20.0,
  hop_length_ms: 10.0,
  min_frequency: 0.0,
  max_frequency: 8000.0,
  preemphasis: 0.0,
  preemphasis_htk_flavor: true,
  fft_overdrive: false,
  dither: 0.0,
  input_scale_factor: 1.0,
  mel_floor: 0.001,
  per_bin_mean: null,
  per_bin_stddev: null,
};

const resolved = resolveAudioFeatureExtractor(gemma4, 'test');
assert.equal(resolved.featureSize, 128);
assert.equal(resolved.frameLength, 320);
assert.equal(resolved.hopLength, 160);
assert.equal(resolved.fftLength, 512);
assert.equal(resolved.melFloor, 0.001);
assert.equal(resolved.preemphasis, 0);
assert.equal(resolved.perBinMean, null);
assert.equal(resolveAudioFeatureExtractor({ ...gemma4, fft_overdrive: true }, 'test').fftLength, 1024);
assert.equal(resolveAudioFeatureExtractor({ ...gemma4, frame_length_ms: 32.0 }, 'test').frameLength, 512);
assert.equal(resolveAudioFeatureExtractor(null, 'test'), null);
assert.equal(resolveAudioFeatureExtractor(undefined, 'test'), null);
assert.throws(() => resolveAudioFeatureExtractor({ ...gemma4, feature_size: undefined }, 'm'), /missing audio_config\.feature_extractor\.feature_size/);
assert.throws(() => resolveAudioFeatureExtractor({ ...gemma4, mel_floor: 0 }, 'm'), /invalid audio_config\.feature_extractor\.mel_floor/);
assert.throws(() => resolveAudioFeatureExtractor({ ...gemma4, dither: 0.1 }, 'm'), /dither=0\.1 requests random dither/);
assert.throws(() => resolveAudioFeatureExtractor({ ...gemma4, max_frequency: 9000 }, 'm'), /frequency range 0-9000 Hz is invalid/);
assert.throws(() => resolveAudioFeatureExtractor({ ...gemma4, per_bin_mean: [1, 2] }, 'm'), /expected 128 finite numbers/);
assert.throws(() => resolveAudioFeatureExtractor('128', 'm'), /expected an object/);

// Filterbank geometry: 257 bins, 128 unnormalized triangles (norm=None) from
// 0 Hz to 8000 Hz. With min_frequency 0 the reference's lowest triangle lies
// entirely below the first FFT bin (31.25 Hz) and is all zeros; that bin then
// reads log(mel_floor), exactly as the reference extractor produces it.
{
  const { filters, numBins } = buildMelFilterBank(resolved);
  assert.equal(numBins, 257);
  assert.equal(filters.length, 128 * 257);
  let withMass = 0;
  for (let m = 0; m < 128; m += 1) {
    const row = filters.subarray(m * numBins, (m + 1) * numBins);
    const peak = Math.max(...row);
    if (peak > 0) withMass += 1;
    assert.ok(peak <= 1 + 1e-6, `mel ${m} peak ${peak} is unnormalized triangle height`);
  }
  assert.ok(withMass >= 125, `${withMass} of 128 mel filters carry mass`);
  assert.equal(Math.max(...filters.subarray(0, numBins)), 0, 'the 0 Hz triangle is empty like the reference');
}

// A 1 kHz tone lands in the mel bin whose centre is nearest 1 kHz, silence
// sits exactly at log(mel_floor), and the frame count follows the reference
// framing (semicausal pad of 160, windows of 321 unfolded every 160).
{
  const sampleRate = 16000;
  const seconds = 1;
  const tone = new Float32Array(sampleRate * seconds);
  for (let i = 0; i < tone.length; i += 1) tone[i] = 0.5 * Math.sin((2 * Math.PI * 1000 * i) / sampleRate);
  const { features, numFrames, nMels } = extractLogMelSpectrogram(tone, resolved);
  assert.equal(nMels, 128);
  assert.equal(numFrames, Math.floor((160 + tone.length - 321) / 160) + 1);
  assert.equal(features.length, numFrames * 128);
  assert.ok(features.every(Number.isFinite));
  const frame = features.subarray(50 * 128, 51 * 128);
  let best = 0;
  for (let m = 1; m < 128; m += 1) if (frame[m] > frame[best]) best = m;
  const hertzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  const melToHertz = (mel) => 700 * (10 ** (mel / 2595) - 1);
  const melMax = hertzToMel(8000);
  const centre = melToHertz((melMax * (best + 1)) / 129);
  assert.ok(Math.abs(centre - 1000) < 60, `peak mel bin ${best} centred at ${centre.toFixed(1)} Hz`);

  const silence = new Float32Array(sampleRate);
  const quiet = extractLogMelSpectrogram(silence, resolved);
  assert.ok(quiet.features.every((value) => Math.abs(value - Math.log(0.001)) < 1e-6));

  // Per-bin normalization applies mean then stddev.
  const normalized = extractLogMelSpectrogram(silence, {
    ...resolved,
    perBinMean: new Float32Array(128).fill(Math.log(0.001)),
    perBinStddev: new Float32Array(128).fill(2),
  });
  assert.ok(normalized.features.every((value) => Math.abs(value) < 1e-6));
}

assert.throws(() => extractLogMelSpectrogram(new Float32Array(16000)), /requires the manifest audio_config\.feature_extractor/);
assert.throws(() => extractLogMelSpectrogram(new Float32Array(16000), { ...resolved, melFloor: undefined }), /requires feature_extractor\.melFloor/);
assert.throws(() => extractLogMelSpectrogram(new Float32Array(100), resolved), /too few for one 320-sample frame/);
assert.throws(() => extractLogMelSpectrogram(new Float32Array(0), resolved), /non-empty Float32Array/);
console.log('audio feature extractor ok');
