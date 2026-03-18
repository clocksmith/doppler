// =============================================================================
// Token Press — Renderer Layer
// =============================================================================
// Renders the token stream as colored spans with a trailing alternatives
// window. Two zones:
//   - Settled zone: append-only on forward steps, only rebuilt on back steps
//   - Trail zone: rebuilt every flush (small fixed-size window)

const CONF_HIGH = 0.8;
const CONF_MID = 0.3;

function confidenceToColor(confidence) {
  if (confidence >= CONF_HIGH) {
    const t = (confidence - CONF_HIGH) / (1.0 - CONF_HIGH);
    return `color-mix(in oklch, var(--doppler-blue) ${(t * 100).toFixed(0)}%, var(--doppler-purple))`;
  }
  if (confidence >= CONF_MID) {
    const t = (confidence - CONF_MID) / (CONF_HIGH - CONF_MID);
    return `color-mix(in oklch, var(--doppler-purple) ${(t * 100).toFixed(0)}%, var(--doppler-red))`;
  }
  return 'var(--doppler-red)';
}

function createAlternativesTooltip(topK) {
  const tooltip = document.createElement('div');
  tooltip.className = 'tp-alternatives';
  for (const alt of topK) {
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
  return tooltip;
}

function createTokenSpan(record, cssClass, trailFade) {
  const span = document.createElement('span');
  span.className = `tp-token ${cssClass}`;
  span.textContent = record.text;
  span.dataset.confidence = record.confidence.toFixed(3);

  const color = confidenceToColor(record.confidence);
  span.style.setProperty('--conf-color', color);
  span.style.setProperty('--confidence', record.confidence.toFixed(3));

  if (trailFade != null) {
    span.style.setProperty('--trail-opacity', trailFade.toFixed(2));
  }

  if (record.topK && record.topK.length > 1) {
    span.append(createAlternativesTooltip(record.topK));
    span.classList.add('tp-has-alts');
  }

  return span;
}

export function createTokenPressRenderer(container, options = {}) {
  const { trailSize = 8 } = options;

  const settledZone = document.createElement('span');
  settledZone.className = 'tp-settled-zone';

  const trailZone = document.createElement('span');
  trailZone.className = 'tp-trail-zone';

  container.innerHTML = '';
  container.classList.add('tp-container');
  container.append(settledZone, trailZone);

  let settledCount = 0;
  let lastCursor = 0;

  function rebuildTrail(committed, trailStart, cursor) {
    trailZone.innerHTML = '';
    const trailLen = cursor - trailStart;
    for (let i = trailStart; i < cursor; i++) {
      const fade = trailLen <= 1 ? 1.0
        : 0.15 + 0.85 * ((i - trailStart) / (trailLen - 1));
      trailZone.append(createTokenSpan(committed[i], 'tp-trail', fade));
    }
  }

  function render(state) {
    const { committed, cursor } = state;
    const trailStart = Math.max(0, cursor - trailSize);
    const wentBack = cursor < lastCursor;

    if (wentBack) {
      // Backward step — may need to remove settled DOM nodes
      while (settledCount > trailStart) {
        settledCount--;
        const last = settledZone.lastChild;
        if (last) last.remove();
      }
    } else {
      // Forward step — append any newly settled tokens
      while (settledCount < trailStart) {
        settledZone.append(
          createTokenSpan(committed[settledCount], 'tp-settled', null)
        );
        settledCount++;
      }
    }

    rebuildTrail(committed, trailStart, cursor);
    lastCursor = cursor;
  }

  function clear() {
    settledZone.innerHTML = '';
    trailZone.innerHTML = '';
    settledCount = 0;
    lastCursor = 0;
  }

  function dispose() {
    clear();
    container.classList.remove('tp-container');
  }

  return { render, clear, dispose };
}
