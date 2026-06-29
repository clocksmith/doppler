export const SVG_THEME = Object.freeze({
  palette: Object.freeze({
    bg: '#ffffff',
    panel: '#ffffff',
    panelAlt: '#fafafa',
    border: '#111111',
    text: '#111111',
    muted: '#4b5563',
    grid: '#e5e7eb',
    accent: '#8b5cf6',
    good: '#3b82f6',
    warn: '#8b5cf6',
    bad: '#f87171',
    doppler: '#3b82f6',          // Blue
    transformersjs: '#f87171',   // Light Red
    fail: '#ef4444',
    failFill: '#fee2e2',
    leaderDoppler: '#eff6ff',
    leaderTjs: '#fef2f2',
    metric: Object.freeze([
      '#3b82f6', // Blue
      '#8b5cf6', // Purple
      '#f87171', // Light Red
      '#111111', // Black
      '#4b5563', // Gray
    ]),
    phase: Object.freeze({
      warmLoad: '#3b82f6', // Blue
      prefill: '#8b5cf6',  // Purple
      ttftMarker: '#111111',
      decode: '#f87171',   // Light Red
    }),
    architecture: Object.freeze({
      loadBorder: '#111111',
      inferBorder: '#111111',
      edge: '#4b5563',
      arrow: '#3b82f6',
      arrowMuted: '#8b5cf6',
      nodeLoad: '#ffffff',
      nodeInfer: '#ffffff',
    }),
  }),
  fonts: Object.freeze({
    ui: 'Outfit, Inter, system-ui, -apple-system, sans-serif',
    mono: 'JetBrains Mono, Fira Code, SFMono-Regular, Consolas, monospace',
  }),
  stroke: Object.freeze({
    thin: 1.25,
    normal: 1.75,
    heavy: 2.5,
  }),
  radius: Object.freeze({
    panel: 6,
    badge: 4,
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
  ${selector} { fill: ${palette.text}; font-family: ${SVG_FONTS.uiCss}; letter-spacing: -0.01em; }
</style></defs>`;
}

export const SVG_FONTS = Object.freeze({
  ui: FONT_STACKS.ui.join(', '),
  mono: FONT_STACKS.mono.join(', '),
  uiCss: quoteFamilyNames(FONT_STACKS.ui),
  monoCss: quoteFamilyNames(FONT_STACKS.mono),
});
