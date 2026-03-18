// =============================================================================
// Token Press — Renderer Layer
// =============================================================================
// DOM rendering for the token press visualization. Receives flush events from
// the queue layer and updates the output container.
//
// Each token is a <span> with a confidence fill and hover-accessible top-k
// alternatives. The "trail" (most recent N tokens) shows alternatives at
// fading opacity. Older tokens collapse to just the chosen text with a
// confidence underline.
//
// Rendering is purely declarative from the queue state — no internal state
// beyond the DOM itself.
//
// Usage:
//   const renderer = createTokenPressRenderer(containerEl, { trailSize: 8 });
//   // Called by queue.onFlush:
//   renderer.render({ trail, cursor, total, playing });
//   renderer.dispose();

const CONFIDENCE_THRESHOLDS = {
  high: 0.8,
  mid: 0.3,
};

function confidenceToColor(confidence) {
  // blue (1.0) → purple (0.5) → red (0.0)
  // Uses CSS custom properties defined in main.css
  if (confidence >= CONFIDENCE_THRESHOLDS.high) {
    // Interpolate blue → purple
    const t = (confidence - CONFIDENCE_THRESHOLDS.high) / (1.0 - CONFIDENCE_THRESHOLDS.high);
    return `color-mix(in oklch, var(--doppler-blue) ${(t * 100).toFixed(0)}%, var(--doppler-purple))`;
  }
  if (confidence >= CONFIDENCE_THRESHOLDS.mid) {
    // Interpolate purple → red
    const t = (confidence - CONFIDENCE_THRESHOLDS.mid) / (CONFIDENCE_THRESHOLDS.high - CONFIDENCE_THRESHOLDS.mid);
    return `color-mix(in oklch, var(--doppler-purple) ${(t * 100).toFixed(0)}%, var(--doppler-red))`;
  }
  return 'var(--doppler-red)';
}

function trailOpacity(indexFromEnd, trailSize) {
  // Newest = 1.0, oldest in trail = 0.15
  if (trailSize <= 1) return 1.0;
  return 0.15 + 0.85 * (1 - indexFromEnd / (trailSize - 1));
}

function createTokenSpan(record, isInTrail, indexFromEnd, trailSize) {
  const span = document.createElement('span');
  span.className = 'tp-token';
  span.textContent = record.text;
  span.dataset.tokenId = String(record.tokenId);
  span.dataset.confidence = record.confidence.toFixed(3);

  const color = confidenceToColor(record.confidence);
  span.style.setProperty('--conf-color', color);
  span.style.setProperty('--confidence', record.confidence.toFixed(3));

  if (isInTrail) {
    span.classList.add('tp-trail');
    span.style.setProperty('--trail-opacity', trailOpacity(indexFromEnd, trailSize).toFixed(2));
  }

  // Alternatives tooltip (accessible on hover/click)
  if (record.topK && record.topK.length > 1) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tp-alternatives';
    for (const alt of record.topK) {
      const row = document.createElement('div');
      row.className = 'tp-alt-row';

      const text = document.createElement('span');
      text.className = 'tp-alt-text';
      text.textContent = alt.text;

      const bar = document.createElement('span');
      bar.className = 'tp-alt-bar';
      bar.style.setProperty('--alt-prob', alt.prob.toFixed(3));
      bar.style.setProperty('--alt-color', confidenceToColor(alt.prob));

      const pct = document.createElement('span');
      pct.className = 'tp-alt-pct';
      pct.textContent = `${(alt.prob * 100).toFixed(1)}%`;

      row.append(text, bar, pct);
      tooltip.append(row);
    }
    span.append(tooltip);
    span.classList.add('tp-has-alts');
  }

  return span;
}

export function createTokenPressRenderer(container, options = {}) {
  const { trailSize = 8 } = options;

  // Settled tokens zone (before the trail)
  const settledZone = document.createElement('span');
  settledZone.className = 'tp-settled-zone';

  // Trail zone (most recent N tokens with alternatives visible)
  const trailZone = document.createElement('span');
  trailZone.className = 'tp-trail-zone';

  container.innerHTML = '';
  container.classList.add('tp-container');
  container.append(settledZone, trailZone);

  // Track settled token count to avoid re-rendering them
  let settledCount = 0;

  function render(state) {
    const { trail, cursor } = state;
    const settledEnd = cursor - trail.length;

    // Append new settled tokens (tokens that left the trail)
    if (settledEnd > settledCount) {
      // We need the committed array for settled tokens — but we only get
      // the trail in the flush. The settled tokens are the ones the queue
      // already committed that are no longer in the trail window. We can
      // reconstruct from previous trail renders or accept we only render
      // the trail and settled text grows from collapsed trail tokens.
      settledCount = settledEnd;
    }

    // Clear and re-render trail zone
    trailZone.innerHTML = '';
    for (let i = 0; i < trail.length; i++) {
      const indexFromEnd = trail.length - 1 - i;
      const span = createTokenSpan(trail[i], true, indexFromEnd, trailSize);
      trailZone.append(span);
    }
  }

  function appendSettledToken(record) {
    const span = document.createElement('span');
    span.className = 'tp-token tp-settled';
    span.textContent = record.text;
    span.dataset.tokenId = String(record.tokenId);
    span.dataset.confidence = record.confidence.toFixed(3);
    span.style.setProperty('--conf-color', confidenceToColor(record.confidence));
    span.style.setProperty('--confidence', record.confidence.toFixed(3));

    // Settled tokens still have hover alternatives
    if (record.topK && record.topK.length > 1) {
      const tooltip = document.createElement('div');
      tooltip.className = 'tp-alternatives';
      for (const alt of record.topK) {
        const row = document.createElement('div');
        row.className = 'tp-alt-row';
        const text = document.createElement('span');
        text.className = 'tp-alt-text';
        text.textContent = alt.text;
        const pct = document.createElement('span');
        pct.className = 'tp-alt-pct';
        pct.textContent = `${(alt.prob * 100).toFixed(1)}%`;
        row.append(text, pct);
        tooltip.append(row);
      }
      span.append(tooltip);
      span.classList.add('tp-has-alts');
    }

    settledZone.append(span);
    settledCount++;
  }

  function renderWithSettled(state, allCommitted) {
    const { trail, cursor } = state;
    const settledEnd = cursor - trail.length;

    // Append any new settled tokens
    while (settledCount < settledEnd && settledCount < allCommitted.length) {
      appendSettledToken(allCommitted[settledCount]);
    }

    // Re-render trail zone
    trailZone.innerHTML = '';
    for (let i = 0; i < trail.length; i++) {
      const indexFromEnd = trail.length - 1 - i;
      const span = createTokenSpan(trail[i], true, indexFromEnd, trailSize);
      trailZone.append(span);
    }
  }

  function clear() {
    settledZone.innerHTML = '';
    trailZone.innerHTML = '';
    settledCount = 0;
  }

  function dispose() {
    clear();
    container.classList.remove('tp-container');
  }

  return {
    render,
    renderWithSettled,
    clear,
    dispose,
  };
}
