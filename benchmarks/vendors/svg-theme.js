export const SVG_THEME = Object.freeze({
  palette: Object.freeze({
    text: '#ffffff',
    muted: '#cbd5e1',
    grid: '#1f2937',
    doppler: '#9d4edd',
    transformersjs: '#ffbd45',
    fail: '#3f3f46',
    failFill: '#7f1d1d',
    metric: Object.freeze([
      '#9d4edd',
      '#c77dff',
      '#7c3aed',
      '#ffbd45',
      '#ffd580',
      '#22d3ee',
      '#4ade80',
      '#f59e0b',
    ]),
    phase: Object.freeze({
      warmLoad: '#ef4444',
      prefill: '#fbbf24',
      ttftMarker: '#ffffff',
      decode: '#3b82f6',
    }),
    architecture: Object.freeze({
      loadBorder: '#ef4444',
      inferBorder: '#2563eb',
      edge: '#7c3aed',
      arrow: '#7c3aed',
      arrowMuted: '#2563eb',
      nodeLoad: '#ef4444',
      nodeInfer: '#2563eb',
      columnLeftStart: '#ef444412',
      columnLeftMid: '#ef444418',
      columnLeftEnd: '#7c3aed16',
      columnRightStart: '#7c3aed12',
      columnRightMid: '#7c3aed16',
      columnRightEnd: '#2563eb16',
    }),
  }),
  fonts: Object.freeze({
    ui: 'Segoe UI, Helvetica Neue, Arial, sans-serif',
    mono: 'SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace',
  }),
  textStroke: Object.freeze({
    color: '#000000',
    width: '2px',
    lineJoin: 'round',
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
  return `<defs><style>
  ${selector} { paint-order: stroke fill; stroke: ${SVG_THEME.textStroke.color}; stroke-width: ${SVG_THEME.textStroke.width}; stroke-linejoin: ${SVG_THEME.textStroke.lineJoin}; }
</style></defs>`;
}

export const SVG_FONTS = Object.freeze({
  ui: FONT_STACKS.ui.join(', '),
  mono: FONT_STACKS.mono.join(', '),
  uiCss: quoteFamilyNames(FONT_STACKS.ui),
  monoCss: quoteFamilyNames(FONT_STACKS.mono),
});
