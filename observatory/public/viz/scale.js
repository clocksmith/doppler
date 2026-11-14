// The encodings every view shares, defined once: what a color, a width and a
// shade mean. They return numbers and token names; the views do the drawing.

// Sign: Doppler's two ends of the spectrum. Zero has no color.
export function signToken(value) {
  if (!Number.isFinite(value)) throw new Error('A sign needs a finite number.');
  return value > 0 ? '--doppler-blue' : value < 0 ? '--doppler-red' : null;
}

// Magnitude: a share of the largest value in view, from 0 to 1.
export function magnitudeShare(value, largest) {
  if (!Number.isFinite(value) || !Number.isFinite(largest)) throw new Error('A magnitude needs finite numbers.');
  if (largest <= 0) return 0;
  return Math.min(1, Math.abs(value) / largest);
}

// Probability: bands whose widths are the probabilities, always with the
// remainder as its own band, so listed candidates never look like all there is.
export function probabilityBands(candidates) {
  let listed = 0;
  const bands = candidates.map((candidate) => {
    if (!(candidate.probability >= 0 && candidate.probability <= 1)) throw new Error('A probability must be between 0 and 1.');
    listed += candidate.probability;
    return { ...candidate, share: candidate.probability, other: false };
  });
  return [...bands, { share: Math.max(0, 1 - listed), other: true }];
}

// Relative difference: differences span many orders of magnitude, so shade on a
// log scale from 1e-7 (faint) to 1 (solid). Exactly zero has no shade.
export function differenceStrength(relative) {
  if (!Number.isFinite(relative) || relative < 0) throw new Error('A relative difference must be a finite number from 0.');
  return relative === 0 ? 0 : Math.min(1, Math.max(0.06, (Math.log10(relative) + 7) / 7));
}
