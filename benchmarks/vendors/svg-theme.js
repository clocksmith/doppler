export const SVG_THEME = Object.freeze({
  palette: Object.freeze({
    bg: '#ffffff',
    panel: '#ffffff',
    panelAlt: '#f6f7ff',
    border: '#111111',
    text: '#111111',
    muted: '#404041',
    grid: '#d8d8d8',
    accent: '#17358f',
    good: '#5c6fff',
    warn: '#6b4ee6',
    bad: '#cc3e45',
    doppler: '#17358f',
    transformersjs: '#5c6fff',
    fail: '#d7d7d7',
    failFill: '#ffeded',
    leaderDoppler: '#e8edff',
    leaderTjs: '#ffe8eb',
    metric: Object.freeze([
      '#111111',
      '#17358f',
      '#5c6fff',
      '#cc3e45',
      '#5e5e5e',
    ]),
    phase: Object.freeze({
      warmLoad: '#17358f',
      prefill: '#6b4ee6',
      ttftMarker: '#111111',
      decode: '#5c6fff',
    }),
    architecture: Object.freeze({
      loadBorder: '#111111',
      inferBorder: '#111111',
      edge: '#111111',
      arrow: '#111111',
      arrowMuted: '#404041',
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
