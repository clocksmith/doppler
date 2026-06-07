export const SVG_THEME = Object.freeze({
  palette: Object.freeze({
    bg: '#ffffff',
    panel: '#ffffff',
    panelAlt: '#f7f7f7',
    border: '#111111',
    text: '#111111',
    muted: '#5a5a5a',
    grid: '#d8d8d8',
    accent: '#111111',
    good: '#b8a8d8',
    warn: '#9bb7d6',
    bad: '#d98b8b',
    doppler: '#111111',
    transformersjs: '#9bb7d6',
    fail: '#d8d8d8',
    failFill: '#d98b8b',
    metric: Object.freeze([
      '#111111',
      '#9bb7d6',
      '#b8a8d8',
      '#d98b8b',
      '#5a5a5a',
    ]),
    phase: Object.freeze({
      warmLoad: '#111111',
      prefill: '#9bb7d6',
      ttftMarker: '#111111',
      decode: '#b8a8d8',
    }),
    architecture: Object.freeze({
      loadBorder: '#111111',
      inferBorder: '#111111',
      edge: '#111111',
      arrow: '#111111',
      arrowMuted: '#5a5a5a',
      nodeLoad: '#111111',
      nodeInfer: '#111111',
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
