// =============================================================================
// Token Press — Renderer Layer
// =============================================================================
// Renders the token stream as colored spans with a trailing alternatives
// window. Two zones:
//   - Settled zone: append-only on forward steps, only rebuilt on back steps
//   - Trail zone: rebuilt every flush (small fixed-size window)
//
// Color scale: perplexity normalized to a clipped display range so single
// outliers do not flatten the rest of the run.
// As new tokens stream in, all existing spans re-color via CSS transitions.

import {
  normalizePerplexity,
  probToPerplexity,
  summarizePerplexityRecords,
} from './metrics.js';

function tToColor(t) {
  if (t <= 0.5) {
    const mix = (t / 0.5) * 100;
    return `color-mix(in oklch, var(--doppler-purple) ${mix.toFixed(0)}%, var(--doppler-blue))`;
  }
  const mix = ((t - 0.5) / 0.5) * 100;
  return `color-mix(in oklch, var(--doppler-red) ${mix.toFixed(0)}%, var(--doppler-purple))`;
}

function confidenceToColor(prob, minPpl, maxPpl) {
  const ppl = probToPerplexity(prob);
  const t = normalizePerplexity(ppl, minPpl, maxPpl);
  return tToColor(t);
}

function createAlternativesTooltip(topK, confidence, perplexityStats) {
  const tooltip = document.createElement('div');
  tooltip.className = 'tp-alternatives';

  const perplexity = probToPerplexity(confidence);
  const header = document.createElement('div');
  header.className = 'tp-alt-header';
  header.textContent = `Perplexity ${perplexity.toFixed(1)}`;
  tooltip.append(header);

  if ((perplexityStats?.extremeLowCount ?? 0) > 0 || (perplexityStats?.extremeHighCount ?? 0) > 0) {
    const summary = document.createElement('div');
    summary.className = 'tp-alt-summary';
    summary.textContent =
      `Display range ${perplexityStats.displayMin.toFixed(1)}-${perplexityStats.displayMax.toFixed(1)}`
      + `, extremes ${perplexityStats.extremeLowCount + perplexityStats.extremeHighCount}`;
    tooltip.append(summary);
  }

  for (const alt of topK) {
    const row = document.createElement('div');
    row.className = 'tp-alt-row';

    const text = document.createElement('span');
    text.className = 'tp-alt-text';
    text.textContent = alt.text;

    const logit = document.createElement('span');
    logit.className = 'tp-alt-logit';
    logit.textContent = Number.isFinite(alt.logit) ? alt.logit.toFixed(2) : 'n/a';

    const bar = document.createElement('span');
    bar.className = 'tp-alt-bar';
    bar.style.setProperty('--alt-prob', alt.prob.toFixed(3));

    const pct = document.createElement('span');
    pct.className = 'tp-alt-pct';
    pct.textContent = `${(alt.prob * 100).toFixed(1)}%`;

    row.append(text, logit, bar, pct);
    tooltip.append(row);
  }
  return tooltip;
}

function syncAlternativesTooltip(tooltip, confidence, perplexityStats) {
  if (!tooltip) {
    return;
  }
  const perplexity = probToPerplexity(confidence);
  const header = tooltip.querySelector('.tp-alt-header');
  if (header) {
    header.textContent = `Perplexity ${perplexity.toFixed(1)}`;
  }
  let summary = tooltip.querySelector('.tp-alt-summary');
  const hasExtremes = (perplexityStats?.extremeLowCount ?? 0) > 0 || (perplexityStats?.extremeHighCount ?? 0) > 0;
  if (hasExtremes) {
    if (!summary) {
      summary = document.createElement('div');
      summary.className = 'tp-alt-summary';
      const headerNode = tooltip.querySelector('.tp-alt-header');
      if (headerNode?.nextSibling) {
        tooltip.insertBefore(summary, headerNode.nextSibling);
      } else {
        tooltip.append(summary);
      }
    }
    summary.textContent =
      `Display range ${perplexityStats.displayMin.toFixed(1)}-${perplexityStats.displayMax.toFixed(1)}`
      + `, extremes ${perplexityStats.extremeLowCount + perplexityStats.extremeHighCount}`;
  } else if (summary) {
    summary.remove();
  }
}

function applyColor(span, prob, minPpl, maxPpl) {
  const color = confidenceToColor(prob, minPpl, maxPpl);
  span.style.setProperty('--conf-color', color);
}

function createTokenSpan(record, cssClass, trailFade, minPpl, maxPpl, perplexityStats) {
  const span = document.createElement('span');
  span.className = `tp-token ${cssClass}`;
  span.textContent = record.text;
  span.dataset.confidence = record.confidence.toFixed(6);

  applyColor(span, record.confidence, minPpl, maxPpl);

  if (trailFade != null) {
    span.style.setProperty('--trail-opacity', trailFade.toFixed(2));
  }

  if (record.topK && record.topK.length > 1) {
    span.tabIndex = 0;
    span.setAttribute('role', 'button');
    span.setAttribute('aria-label', `Token details for ${record.text || 'token'}`);
    span.append(createAlternativesTooltip(record.topK, record.confidence, perplexityStats));
    span.classList.add('tp-has-alts');
    span.addEventListener('pointerenter', () => {
      span.dataset.tooltipOpen = 'true';
    });
    span.addEventListener('pointerleave', () => {
      delete span.dataset.tooltipOpen;
    });
    span.addEventListener('focus', () => {
      span.dataset.tooltipOpen = 'true';
    });
    span.addEventListener('blur', () => {
      delete span.dataset.tooltipOpen;
    });
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
  let currentMinPpl = 1;
  let currentMaxPpl = 1;
  let currentPerplexityStats = summarizePerplexityRecords([]);

  // Recolor all existing spans when the range changes
  function recolorAll(committed, cursor) {
    // Settled zone
    for (let i = 0; i < settledCount; i++) {
      const span = settledZone.children[i];
      if (span) {
        const prob = parseFloat(span.dataset.confidence);
        if (Number.isFinite(prob)) {
          applyColor(span, prob, currentMinPpl, currentMaxPpl);
          syncAlternativesTooltip(span.querySelector('.tp-alternatives'), prob, currentPerplexityStats);
        }
      }
    }
    // Trail zone
    for (const span of trailZone.children) {
      const prob = parseFloat(span.dataset.confidence);
      if (Number.isFinite(prob)) {
        applyColor(span, prob, currentMinPpl, currentMaxPpl);
        syncAlternativesTooltip(span.querySelector('.tp-alternatives'), prob, currentPerplexityStats);
      }
    }
  }

  function updateRange(committed, cursor) {
    const summary = summarizePerplexityRecords(
      committed.slice(0, cursor).map((record) => ({
        perplexity: probToPerplexity(record.confidence),
      }))
    );
    const min = summary.displayMin ?? 1;
    const max = summary.displayMax ?? min;
    const changed = min !== currentMinPpl || max !== currentMaxPpl;
    currentMinPpl = min;
    currentMaxPpl = max;
    currentPerplexityStats = summary;
    return changed;
  }

  function updateSpanContent(span, record) {
    // Update text without destroying child tooltip nodes — replace only the
    // leading text node, then rebuild the tooltip so reused spans keep hover.
    const firstChild = span.firstChild;
    if (firstChild && firstChild.nodeType === Node.TEXT_NODE) {
      firstChild.textContent = record.text;
    } else {
      span.insertBefore(document.createTextNode(record.text), span.firstChild);
    }
    // Remove stale tooltip if present
    const oldTooltip = span.querySelector('.tp-alternatives');
    if (oldTooltip) oldTooltip.remove();
    span.classList.remove('tp-has-alts');
    // Rebuild tooltip from current record
    if (record.topK && record.topK.length > 1) {
      span.append(createAlternativesTooltip(record.topK, record.confidence, currentPerplexityStats));
      span.classList.add('tp-has-alts');
    }
    span.dataset.confidence = record.confidence.toFixed(6);
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
          updateSpanContent(span, record);
          span.dataset.tokenIndex = String(i);
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

  function buildSpan(record, cssClass, trailFade = null) {
    return createTokenSpan(
      record,
      cssClass,
      trailFade,
      currentMinPpl,
      currentMaxPpl,
      currentPerplexityStats
    );
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
        settledZone.append(buildSpan(committed[settledCount], 'tp-settled'));
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

  function finalize(state) {
    const { committed, cursor } = state;
    updateRange(committed, cursor);
    // Promote all remaining trail tokens into the settled zone so they get
    // full tooltip-bearing spans and consistent settled styling.
    while (settledCount < cursor) {
      settledZone.append(buildSpan(committed[settledCount], 'tp-settled'));
      settledCount++;
    }
    trailZone.innerHTML = '';
    lastCursor = cursor;
  }

  function clear() {
    settledZone.innerHTML = '';
    trailZone.innerHTML = '';
    settledCount = 0;
    lastCursor = 0;
    currentMinPpl = 1;
    currentMaxPpl = 1;
    currentPerplexityStats = summarizePerplexityRecords([]);
  }

  function dispose() {
    clear();
    container.classList.remove('tp-container');
  }

  return { render, finalize, clear, dispose };
}
