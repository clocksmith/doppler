export const SVG_THEME = Object.freeze({
  palette: Object.freeze({
    bg: '#050607',
    panel: '#0b0d0f',
    panelAlt: '#101317',
    border: '#2a2f35',
    text: '#f2f2f0',
    muted: '#9ca3af',
    grid: '#2a2f35',
    accent: '#93c5fd',
    good: '#86efac',
    warn: '#fde68a',
    bad: '#fca5a5',
    doppler: '#93c5fd',
    transformersjs: '#fde68a',
    fail: '#2a2f35',
    failFill: '#fca5a5',
    metric: Object.freeze([
      '#93c5fd',
      '#fde68a',
      '#86efac',
      '#fca5a5',
      '#9ca3af',
    ]),
    phase: Object.freeze({
      warmLoad: '#fde68a',
      prefill: '#93c5fd',
      ttftMarker: '#f2f2f0',
      decode: '#86efac',
    }),
    architecture: Object.freeze({
      loadBorder: '#fde68a',
      inferBorder: '#93c5fd',
      edge: '#9ca3af',
      arrow: '#93c5fd',
      arrowMuted: '#9ca3af',
      nodeLoad: '#fde68a',
      nodeInfer: '#93c5fd',
    }),
  }),
  fonts: Object.freeze({
    ui: 'Inter, Segoe UI, Helvetica Neue, Arial, sans-serif',
    mono: 'SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace',
  }),
  stroke: Object.freeze({
    thin: 1.25,
    normal: 1.75,
    heavy: 2.5,
  }),
  radius: Object.freeze({
    panel: 4,
    badge: 3,
  }),
});

function quoteFamilyNames(fontStack) {
  return fontStack.map((font) => {
    const trimmed = font.trim();
    if (trimmed.startsWith('"') && trimmed.endsWith('"')) return trimmed;
    if (trimmed.startsWith("'") && trimmed.endsWith("'")) return trimmed;
    if (trimmed.includes(' ')) return `"${trimmed}"`;
    return trimmed;
  }).join(', ');
}

const FONT_STACKS = Object.freeze({
  ui: SVG_THEME.fonts.ui.split(',').map((font) => font.trim()),
  mono: SVG_THEME.fonts.mono.split(',').map((font) => font.trim()),
});

export function makeSvgTextStyle(selector = 'text') {
  const palette = SVG_THEME.palette;
  return `<defs><style>
  ${selector} { fill: ${palette.text}; font-family: ${SVG_FONTS.uiCss}; letter-spacing: 0; }
</style></defs>`;
}

export const SVG_FONTS = Object.freeze({
  ui: FONT_STACKS.ui.join(', '),
  mono: FONT_STACKS.mono.join(', '),
  uiCss: quoteFamilyNames(FONT_STACKS.ui),
  monoCss: quoteFamilyNames(FONT_STACKS.mono),
});
