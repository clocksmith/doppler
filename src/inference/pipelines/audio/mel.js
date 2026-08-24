
const DEFAULT_SAMPLE_RATE = 16000;
const DEFAULT_N_FFT = 512;
const DEFAULT_HOP_LENGTH = 160;
const DEFAULT_N_MELS = 80;
const DEFAULT_WINDOW_LENGTH = 400;

function hannWindow(length) {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return window;
}

function powerSpectrum(frame, nFft) {
  const real = new Float32Array(nFft);
  const imag = new Float32Array(nFft);
  real.set(frame.subarray(0, Math.min(frame.length, nFft)));

  // In-place Cooley-Tukey FFT
  const bits = Math.log2(nFft);
  for (let i = 0; i < nFft; i++) {
    let j = 0;
    for (let b = 0; b < bits; b++) {
      j = (j << 1) | ((i >> b) & 1);
    }
    if (j > i) {
      let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
      tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
    }
  }

  for (let size = 2; size <= nFft; size *= 2) {
    const halfSize = size / 2;
    const angle = -2 * Math.PI / size;
    for (let i = 0; i < nFft; i += size) {
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

  const numBins = nFft / 2 + 1;
  const power = new Float32Array(numBins);
  for (let i = 0; i < numBins; i++) {
    power[i] = real[i] * real[i] + imag[i] * imag[i];
  }
  return power;
}

function melFilterBank(nMels, nFft, sampleRate) {
  const numBins = nFft / 2 + 1;
  const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  const melToHz = (mel) => 700 * (10 ** (mel / 2595) - 1);

  const melLow = hzToMel(0);
  const melHigh = hzToMel(sampleRate / 2);
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    melPoints[i] = melToHz(melLow + (melHigh - melLow) * i / (nMels + 1));
  }

  const binFreqs = new Float32Array(numBins);
  for (let i = 0; i < numBins; i++) {
    binFreqs[i] = (sampleRate / nFft) * i;
  }

  const filters = new Float32Array(nMels * numBins);
  for (let m = 0; m < nMels; m++) {
    const fLow = melPoints[m];
    const fCenter = melPoints[m + 1];
    const fHigh = melPoints[m + 2];
    for (let k = 0; k < numBins; k++) {
      const freq = binFreqs[k];
      if (freq >= fLow && freq <= fCenter && fCenter > fLow) {
        filters[m * numBins + k] = (freq - fLow) / (fCenter - fLow);
      } else if (freq > fCenter && freq <= fHigh && fHigh > fCenter) {
        filters[m * numBins + k] = (fHigh - freq) / (fHigh - fCenter);
      }
    }
  }
  return filters;
}

export function extractLogMelSpectrogram(audio, opts = {}) {
  const sampleRate = opts.sampleRate ?? DEFAULT_SAMPLE_RATE;
  const nFft = opts.nFft ?? DEFAULT_N_FFT;
  const hopLength = opts.hopLength ?? DEFAULT_HOP_LENGTH;
  const nMels = opts.nMels ?? DEFAULT_N_MELS;
  const windowLength = opts.windowLength ?? DEFAULT_WINDOW_LENGTH;

  if (!(audio instanceof Float32Array) || audio.length === 0) {
    throw new Error('[Audio] extractLogMelSpectrogram requires a non-empty Float32Array of audio samples.');
  }

  const window = hannWindow(windowLength);
  const numBins = nFft / 2 + 1;
  const filters = melFilterBank(nMels, nFft, sampleRate);

  const numFrames = Math.max(1, Math.floor((audio.length - windowLength) / hopLength) + 1);
  const features = new Float32Array(numFrames * nMels);
  const paddedFrame = new Float32Array(nFft);

  for (let t = 0; t < numFrames; t++) {
    const offset = t * hopLength;
    paddedFrame.fill(0);
    for (let i = 0; i < windowLength; i++) {
      const sampleIdx = offset + i;
      paddedFrame[i] = sampleIdx < audio.length ? audio[sampleIdx] * window[i] : 0;
    }

    const power = powerSpectrum(paddedFrame, nFft);

    for (let m = 0; m < nMels; m++) {
      let melEnergy = 0;
      const filterOffset = m * numBins;
      for (let k = 0; k < numBins; k++) {
        melEnergy += filters[filterOffset + k] * power[k];
      }
      features[t * nMels + m] = Math.log(Math.max(melEnergy, 1e-10));
    }
  }

  return { features, numFrames, nMels };
}
