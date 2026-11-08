// Log-mel front end for the Gemma 4 audio encoder.
//
// Port of the reference feature extractor (transformers
// Gemma4AudioFeatureExtractor, single waveform): semicausal zero padding of
// frame_length/2 samples, frames of frame_length samples every hop_length,
// optional HTK-flavoured preemphasis, periodic Hann window, rfft zero-padded to
// fft_length, magnitude spectrum, HTK-mel triangular filterbank between
// min_frequency and max_frequency without area normalization, then
// log(mel + mel_floor) and optional per-bin mean/stddev normalization.
//
// Every parameter comes from the manifest's audio_config.feature_extractor
// (resolveAudioFeatureExtractor). There are no runtime defaults: the encoder's
// first convolution is trained for exactly one mel geometry.

const REQUIRED_FIELDS = [
  'featureSize', 'samplingRate', 'frameLength', 'hopLength', 'fftLength',
  'minFrequency', 'maxFrequency', 'preemphasis', 'preemphasisHtkFlavor',
  'melFloor', 'inputScaleFactor',
];

function requireFeatureExtractor(featureExtractor) {
  if (!featureExtractor || typeof featureExtractor !== 'object') {
    throw new Error('[Audio] extractLogMelSpectrogram requires the manifest audio_config.feature_extractor parameters.');
  }
  for (const field of REQUIRED_FIELDS) {
    if (featureExtractor[field] == null) {
      throw new Error(`[Audio] extractLogMelSpectrogram requires feature_extractor.${field}.`);
    }
  }
  const { frameLength, fftLength } = featureExtractor;
  if ((fftLength & (fftLength - 1)) !== 0 || fftLength < frameLength) {
    throw new Error(`[Audio] fftLength ${fftLength} must be a power of two no smaller than frameLength ${frameLength}.`);
  }
  return featureExtractor;
}

// Periodic Hann window (divides by N), as the reference does.
function periodicHannWindow(length) {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / length));
  }
  return window;
}

// In-place radix-2 FFT of the real frame; returns magnitudes of the first
// fftLength/2 + 1 bins.
function magnitudeSpectrum(frame, fftLength) {
  const real = new Float64Array(fftLength);
  const imag = new Float64Array(fftLength);
  real.set(frame.subarray(0, Math.min(frame.length, fftLength)));

  const bits = Math.log2(fftLength);
  for (let i = 0; i < fftLength; i++) {
    let j = 0;
    for (let b = 0; b < bits; b++) {
      j = (j << 1) | ((i >> b) & 1);
    }
    if (j > i) {
      let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
      tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
    }
  }
  for (let size = 2; size <= fftLength; size *= 2) {
    const halfSize = size / 2;
    const angle = -2 * Math.PI / size;
    for (let i = 0; i < fftLength; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const cos = Math.cos(angle * j);
        const sin = Math.sin(angle * j);
        const tpRe = real[i + j + halfSize] * cos - imag[i + j + halfSize] * sin;
        const tpIm = real[i + j + halfSize] * sin + imag[i + j + halfSize] * cos;
        real[i + j + halfSize] = real[i + j] - tpRe;
        imag[i + j + halfSize] = imag[i + j] - tpIm;
        real[i + j] += tpRe;
        imag[i + j] += tpIm;
      }
    }
  }
  const numBins = fftLength / 2 + 1;
  const magnitude = new Float64Array(numBins);
  for (let i = 0; i < numBins; i++) {
    magnitude[i] = Math.hypot(real[i], imag[i]);
  }
  return magnitude;
}

const hertzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
const melToHertz = (mel) => 700 * (10 ** (mel / 2595) - 1);

// HTK-mel triangular filters built in frequency space, norm=None; returns
// filters[mel * numBins + bin] so each mel row is contiguous.
export function buildMelFilterBank({ featureSize, fftLength, samplingRate, minFrequency, maxFrequency }) {
  const numBins = fftLength / 2 + 1;
  const fftFreqs = new Float64Array(numBins);
  for (let k = 0; k < numBins; k++) {
    fftFreqs[k] = (k * Math.floor(samplingRate / 2)) / (numBins - 1);
  }
  const melMin = hertzToMel(minFrequency);
  const melMax = hertzToMel(maxFrequency);
  const filterFreqs = new Float64Array(featureSize + 2);
  for (let i = 0; i < featureSize + 2; i++) {
    filterFreqs[i] = melToHertz(melMin + ((melMax - melMin) * i) / (featureSize + 1));
  }
  const filters = new Float32Array(featureSize * numBins);
  for (let m = 0; m < featureSize; m++) {
    const lower = filterFreqs[m];
    const center = filterFreqs[m + 1];
    const upper = filterFreqs[m + 2];
    for (let k = 0; k < numBins; k++) {
      const down = (fftFreqs[k] - lower) / (center - lower);
      const up = (upper - fftFreqs[k]) / (upper - center);
      filters[m * numBins + k] = Math.max(0, Math.min(down, up));
    }
  }
  return { filters, numBins };
}

export function extractLogMelSpectrogram(audio, featureExtractor) {
  const fe = requireFeatureExtractor(featureExtractor);
  if (!(audio instanceof Float32Array) || audio.length === 0) {
    throw new Error('[Audio] extractLogMelSpectrogram requires a non-empty Float32Array of audio samples.');
  }
  const { featureSize, frameLength, hopLength, fftLength, melFloor } = fe;
  const padLeft = Math.floor(frameLength / 2);
  const paddedLength = padLeft + audio.length;
  // The reference unfolds frames of frameLength + 1 samples and drops the extra
  // sample, so a frame exists only when its full window plus one lies inside
  // the padded signal. Frames whose window would run past the real audio are
  // masked out by the reference; they are simply not produced here.
  const numFrames = Math.floor((paddedLength - (frameLength + 1)) / hopLength) + 1;
  if (numFrames < 1) {
    throw new Error(`[Audio] ${audio.length} samples are too few for one ${frameLength}-sample frame.`);
  }
  const window = periodicHannWindow(frameLength);
  const { filters, numBins } = buildMelFilterBank(fe);
  const features = new Float32Array(numFrames * featureSize);
  const frame = new Float64Array(frameLength);
  const raw = new Float64Array(frameLength + 1);
  const sampleAt = (index) => (index < padLeft || index - padLeft >= audio.length ? 0 : audio[index - padLeft] * fe.inputScaleFactor);

  for (let t = 0; t < numFrames; t++) {
    const start = t * hopLength;
    for (let i = 0; i <= frameLength; i++) raw[i] = sampleAt(start + i);
    if (fe.preemphasis > 0) {
      if (fe.preemphasisHtkFlavor) {
        frame[0] = raw[0] * (1 - fe.preemphasis);
        for (let i = 1; i < frameLength; i++) frame[i] = raw[i] - fe.preemphasis * raw[i - 1];
      } else {
        for (let i = 0; i < frameLength; i++) frame[i] = raw[i + 1] - fe.preemphasis * raw[i];
      }
    } else {
      for (let i = 0; i < frameLength; i++) frame[i] = raw[i];
    }
    for (let i = 0; i < frameLength; i++) frame[i] *= window[i];
    const magnitude = magnitudeSpectrum(frame, fftLength);
    for (let m = 0; m < featureSize; m++) {
      let energy = 0;
      const offset = m * numBins;
      for (let k = 0; k < numBins; k++) energy += filters[offset + k] * magnitude[k];
      let value = Math.log(energy + melFloor);
      if (fe.perBinMean) value -= fe.perBinMean[m];
      if (fe.perBinStddev) value /= fe.perBinStddev[m];
      features[t * featureSize + m] = value;
    }
  }
  return { features, numFrames, nMels: featureSize };
}
