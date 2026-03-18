// =============================================================================
// Token Press — Renderer Layer
// =============================================================================
// Renders the token stream as colored spans with a trailing alternatives
// window. Two zones:
//   - Settled zone: append-only on forward steps, only rebuilt on back steps
//   - Trail zone: rebuilt every flush (small fixed-size window)
//
// Color scale: perplexity (1/p) normalized to the observed min/max range.
// As new tokens stream in, all existing spans re-color via CSS transitions.

// Perplexity from probability
function probToPpl(prob) {
  return 1 / Math.max(1e-10, Math.min(1, prob));
}

// Normalize t ∈ [0,1] from log-perplexity within observed range
function normalizePpl(ppl, minPpl, maxPpl) {
  const logMin = Math.log(Math.max(1, minPpl));
  const logMax = Math.log(Math.max(1, maxPpl));
  if (logMax <= logMin) return 0;
  return Math.min(1, Math.max(0, (Math.log(ppl) - logMin) / (logMax - logMin)));
}

function tToColor(t) {
  if (t <= 0.5) {
    const mix = (t / 0.5) * 100;
    return `color-mix(in oklch, var(--doppler-purple) ${mix.toFixed(0)}%, var(--doppler-blue))`;
  }
  const mix = ((t - 0.5) / 0.5) * 100;
  return `color-mix(in oklch, var(--doppler-red) ${mix.toFixed(0)}%, var(--doppler-purple))`;
}

function confidenceToColor(prob, minPpl, maxPpl) {
  const ppl = probToPpl(prob);
  const t = normalizePpl(ppl, minPpl, maxPpl);
  return tToColor(t);
}

function createAlternativesTooltip(topK, confidence) {
  const tooltip = document.createElement('div');
  tooltip.className = 'tp-alternatives';

  // Perplexity header
  const ppl = probToPpl(confidence);
  const header = document.createElement('div');
  header.className = 'tp-alt-header';
  header.textContent = `ppl ${ppl.toFixed(1)}`;
  tooltip.append(header);

  for (const alt of topK) {
    const row = document.createElement('div');
    row.className = 'tp-alt-row';

    const text = document.createElement('span');
    text.className = 'tp-alt-text';
    text.textContent = alt.text;

    const bar = document.createElement('span');
    bar.className = 'tp-alt-bar';
    bar.style.setProperty('--alt-prob', alt.prob.toFixed(3));

    const pct = document.createElement('span');
    pct.className = 'tp-alt-pct';
    pct.textContent = `${(alt.prob * 100).toFixed(1)}%`;

    row.append(text, bar, pct);
    tooltip.append(row);
  }
  return tooltip;
}

function applyColor(span, prob, minPpl, maxPpl) {
  const color = confidenceToColor(prob, minPpl, maxPpl);
  span.style.setProperty('--conf-color', color);
}

function createTokenSpan(record, cssClass, trailFade, minPpl, maxPpl) {
  const span = document.createElement('span');
  span.className = `tp-token ${cssClass}`;
  span.textContent = record.text;
  span.dataset.confidence = record.confidence.toFixed(6);

  applyColor(span, record.confidence, minPpl, maxPpl);

  if (trailFade != null) {
    span.style.setProperty('--trail-opacity', trailFade.toFixed(2));
  }

  if (record.topK && record.topK.length > 1) {
    span.append(createAlternativesTooltip(record.topK, record.confidence));
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
  let currentMinPpl = Infinity;
  let currentMaxPpl = 0;

  // Recolor all existing spans when the range changes
  function recolorAll(committed, cursor) {
    // Settled zone
    for (let i = 0; i < settledCount; i++) {
      const span = settledZone.children[i];
      if (span) {
        const prob = parseFloat(span.dataset.confidence);
        if (Number.isFinite(prob)) applyColor(span, prob, currentMinPpl, currentMaxPpl);
      }
    }
    // Trail zone
    for (const span of trailZone.children) {
      const prob = parseFloat(span.dataset.confidence);
      if (Number.isFinite(prob)) applyColor(span, prob, currentMinPpl, currentMaxPpl);
    }
  }

  function updateRange(committed, cursor) {
    let min = currentMinPpl;
    let max = currentMaxPpl;
    for (let i = 0; i < cursor; i++) {
      const ppl = probToPpl(committed[i].confidence);
      if (ppl < min) min = ppl;
      if (ppl > max) max = ppl;
    }
    const changed = min !== currentMinPpl || max !== currentMaxPpl;
    currentMinPpl = min;
    currentMaxPpl = max;
    return changed;
  }

  function rebuildTrail(committed, trailStart, cursor) {
    const trailLen = cursor - trailStart;
    const existing = Array.from(trailZone.children);
    let reused = 0;

    for (let i = trailStart; i < cursor; i++) {
      const fade = trailLen <= 1 ? 1.0
        : 0.15 + 0.85 * ((i - trailStart) / (trailLen - 1));
      const record = committed[i];

      if (reused < existing.length) {
        const span = existing[reused];
        span.style.setProperty('--trail-opacity', fade.toFixed(2));
        if (span.dataset.tokenIndex !== String(i)) {
          span.textContent = record.text;
          span.dataset.tokenIndex = String(i);
          span.dataset.confidence = record.confidence.toFixed(6);
        }
        applyColor(span, record.confidence, currentMinPpl, currentMaxPpl);
        reused++;
      } else {
        const span = createTokenSpan(record, 'tp-trail', fade, currentMinPpl, currentMaxPpl);
        span.dataset.tokenIndex = String(i);
        trailZone.append(span);
      }
    }

    for (let j = existing.length - 1; j >= reused; j--) {
      existing[j].remove();
    }
  }

  function render(state) {
    const { committed, cursor } = state;
    const trailStart = Math.max(0, cursor - trailSize);
    const wentBack = cursor < lastCursor;
    const rangeChanged = updateRange(committed, cursor);

    if (wentBack) {
      while (settledCount > trailStart) {
        settledCount--;
        const last = settledZone.lastChild;
        if (last) last.remove();
      }
    } else {
      while (settledCount < trailStart) {
        settledZone.append(
          createTokenSpan(committed[settledCount], 'tp-settled', null, currentMinPpl, currentMaxPpl)
        );
        settledCount++;
      }
    }

    rebuildTrail(committed, trailStart, cursor);

    // Range shifted — recolor all previously rendered spans so they animate
    if (rangeChanged) {
      recolorAll(committed, cursor);
    }

    lastCursor = cursor;
  }

  function clear() {
    settledZone.innerHTML = '';
    trailZone.innerHTML = '';
    settledCount = 0;
    lastCursor = 0;
    currentMinPpl = Infinity;
    currentMaxPpl = 0;
  }

  function dispose() {
    clear();
    container.classList.remove('tp-container');
  }

  return { render, clear, dispose };
}
