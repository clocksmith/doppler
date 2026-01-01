/**
 * HTML Report Generation - SVG charts and HTML benchmark reports
 */

import { compareResults } from './comparison.js';

// ============================================================================
// SVG Chart Generation
// ============================================================================

export function generateSVGBarChart(
  data: Array<{ label: string; value: number; color?: string }>,
  width = 400,
  height = 200,
  title = ''
): string {
  const margin = { top: 30, right: 20, bottom: 40, left: 60 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const maxValue = Math.max(...data.map((d) => d.value)) * 1.1;
  const barWidth = chartWidth / data.length - 10;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<style>
    .chart-title { font: bold 14px sans-serif; }
    .axis-label { font: 11px sans-serif; fill: #666; }
    .bar-label { font: 10px sans-serif; fill: #333; }
    .grid-line { stroke: #e0e0e0; stroke-width: 1; }
  </style>`;

  if (title) {
    svg += `<text x="${width / 2}" y="18" text-anchor="middle" class="chart-title">${title}</text>`;
  }

  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (chartHeight * i) / 4;
    svg += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid-line"/>`;
    const val = (maxValue * (4 - i)) / 4;
    svg += `<text x="${margin.left - 5}" y="${y + 4}" text-anchor="end" class="axis-label">${val.toFixed(0)}</text>`;
  }

  data.forEach((d, i) => {
    const barHeight = (d.value / maxValue) * chartHeight;
    const x = margin.left + i * (chartWidth / data.length) + 5;
    const y = margin.top + chartHeight - barHeight;
    const color = d.color || '#4a90d9';

    svg += `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${color}" rx="2"/>`;
    svg += `<text x="${x + barWidth / 2}" y="${height - margin.bottom + 15}" text-anchor="middle" class="bar-label">${d.label}</text>`;
    svg += `<text x="${x + barWidth / 2}" y="${y - 5}" text-anchor="middle" class="bar-label">${d.value.toFixed(1)}</text>`;
  });

  svg += '</svg>';
  return svg;
}

export function generateSVGLineChart(
  data: number[],
  width = 400,
  height = 150,
  title = '',
  yLabel = ''
): string {
  const margin = { top: 30, right: 20, bottom: 30, left: 50 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const maxValue = Math.max(...data) * 1.1;
  const minValue = Math.min(...data) * 0.9;
  const range = maxValue - minValue;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<style>
    .chart-title { font: bold 12px sans-serif; }
    .axis-label { font: 10px sans-serif; fill: #666; }
    .line { fill: none; stroke: #4a90d9; stroke-width: 2; }
    .dot { fill: #4a90d9; }
    .grid-line { stroke: #e0e0e0; stroke-width: 1; }
  </style>`;

  if (title) {
    svg += `<text x="${width / 2}" y="15" text-anchor="middle" class="chart-title">${title}</text>`;
  }

  if (yLabel) {
    svg += `<text x="12" y="${height / 2}" text-anchor="middle" transform="rotate(-90, 12, ${height / 2})" class="axis-label">${yLabel}</text>`;
  }

  for (let i = 0; i <= 3; i++) {
    const y = margin.top + (chartHeight * i) / 3;
    svg += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid-line"/>`;
    const val = maxValue - (range * i) / 3;
    svg += `<text x="${margin.left - 5}" y="${y + 4}" text-anchor="end" class="axis-label">${val.toFixed(1)}</text>`;
  }

  const points = data.map((v, i) => {
    const x = margin.left + (i / (data.length - 1)) * chartWidth;
    const y = margin.top + ((maxValue - v) / range) * chartHeight;
    return `${x},${y}`;
  });
  svg += `<polyline points="${points.join(' ')}" class="line"/>`;

  data.forEach((v, i) => {
    const x = margin.left + (i / (data.length - 1)) * chartWidth;
    const y = margin.top + ((maxValue - v) / range) * chartHeight;
    svg += `<circle cx="${x}" cy="${y}" r="3" class="dot"/>`;
  });

  svg += '</svg>';
  return svg;
}

// ============================================================================
// HTML Report Generation
// ============================================================================

export function generateHTMLReport(results: any, baseline?: any): string {
  const isArray = Array.isArray(results);
  const resultList = isArray ? results : [results];
  const firstResult = resultList[0];

  const model = firstResult.model?.modelName || firstResult.model?.modelId || 'Unknown Model';
  const timestamp = new Date().toISOString();
  const env = firstResult.env || {};

  let html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DOPPLER Benchmark Report - ${model}</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; }
    .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1 { color: #333; margin-top: 0; }
    h2 { color: #555; border-bottom: 2px solid #4a90d9; padding-bottom: 8px; }
    h3 { color: #666; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
    .metric { padding: 15px; background: #f8f9fa; border-radius: 6px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #4a90d9; }
    .metric-label { font-size: 14px; color: #666; }
    .metric-unit { font-size: 14px; color: #999; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #e0e0e0; }
    th { background: #f8f9fa; font-weight: 600; }
    .better { color: #28a745; }
    .worse { color: #dc3545; }
    .chart-container { margin: 20px 0; text-align: center; }
    .env-info { font-size: 13px; color: #666; }
    .timestamp { font-size: 12px; color: #999; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>DOPPLER Benchmark Report</h1>
      <p class="timestamp">Generated: ${timestamp}</p>
      <div class="env-info">
        <strong>Model:</strong> ${model} |
        <strong>Browser:</strong> ${env.browser?.name || 'Unknown'} ${env.browser?.version || ''} |
        <strong>GPU:</strong> ${env.gpu?.description || env.gpu?.device || 'Unknown'} |
        <strong>OS:</strong> ${env.os?.name || 'Unknown'}
      </div>
    </div>
`;

  for (const result of resultList) {
    const m = result.metrics;
    const prompt = result.workload?.promptName || 'default';

    html += `
    <div class="card">
      <h2>Results: ${prompt} prompt</h2>
      <div class="grid">
        <div class="metric">
          <div class="metric-value">${m.ttft_ms}<span class="metric-unit">ms</span></div>
          <div class="metric-label">Time to First Token</div>
        </div>
        <div class="metric">
          <div class="metric-value">${m.prefill_tokens_per_sec}<span class="metric-unit">tok/s</span></div>
          <div class="metric-label">Prefill Throughput</div>
        </div>
        <div class="metric">
          <div class="metric-value">${m.decode_tokens_per_sec}<span class="metric-unit">tok/s</span></div>
          <div class="metric-label">Decode Throughput</div>
        </div>
        <div class="metric">
          <div class="metric-value">${m.gpu_submit_count_prefill + m.gpu_submit_count_decode}</div>
          <div class="metric-label">GPU Submits (${m.gpu_submit_count_prefill} prefill + ${m.gpu_submit_count_decode} decode)</div>
        </div>
      </div>
`;

    if (result.raw?.decode_latencies_ms?.length > 0) {
      const latencies = result.raw.decode_latencies_ms;
      html += `
      <div class="chart-container">
        ${generateSVGLineChart(latencies, 600, 150, 'Decode Latency per Token', 'ms')}
      </div>
`;
    }

    if (m.decode_ms_per_token_p50) {
      html += `
      <h3>Latency Percentiles</h3>
      <table>
        <tr><th>P50</th><th>P90</th><th>P99</th></tr>
        <tr>
          <td>${m.decode_ms_per_token_p50.toFixed(2)} ms</td>
          <td>${m.decode_ms_per_token_p90.toFixed(2)} ms</td>
          <td>${m.decode_ms_per_token_p99.toFixed(2)} ms</td>
        </tr>
      </table>
`;
    }

    html += `</div>`;
  }

  if (baseline) {
    const comparisons = compareResults(baseline, firstResult);
    html += `
    <div class="card">
      <h2>Comparison vs Baseline</h2>
      <table>
        <tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th><th>Status</th></tr>
`;
    for (const c of comparisons) {
      const statusClass = c.improved ? 'better' : c.delta === 0 ? '' : 'worse';
      const sign = c.delta >= 0 ? '+' : '';
      html += `
        <tr>
          <td>${c.metric}</td>
          <td>${c.baseline.toFixed(2)}</td>
          <td>${c.current.toFixed(2)}</td>
          <td class="${statusClass}">${sign}${c.deltaPercent.toFixed(1)}%</td>
          <td class="${statusClass}">${c.improved ? 'Better' : c.delta === 0 ? 'Same' : 'Worse'}</td>
        </tr>
`;
    }
    html += `</table>`;

    const chartData = comparisons.slice(0, 6).map((c) => [
      { label: `${c.metric} (base)`, value: c.baseline, color: '#94a3b8' },
      { label: `${c.metric} (curr)`, value: c.current, color: c.improved ? '#22c55e' : '#ef4444' },
    ]).flat();

    html += `
      <div class="chart-container">
        ${generateSVGBarChart(chartData.slice(0, 8), 700, 250, 'Baseline vs Current')}
      </div>
    </div>
`;
  }

  const m = firstResult.metrics;
  html += `
    <div class="card">
      <h2>All Metrics</h2>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
`;
  const allMetrics = [
    ['TTFT', `${m.ttft_ms} ms`],
    ['Prefill Time', `${m.prefill_ms} ms`],
    ['Prefill Throughput', `${m.prefill_tokens_per_sec} tok/s`],
    ['Decode Time', `${m.decode_ms_total} ms`],
    ['Decode Throughput', `${m.decode_tokens_per_sec} tok/s`],
    ['GPU Submits (Prefill)', m.gpu_submit_count_prefill],
    ['GPU Submits (Decode)', m.gpu_submit_count_decode],
    ['GPU Readback Bytes', m.gpu_readback_bytes_total ? `${(m.gpu_readback_bytes_total / 1024).toFixed(1)} KB` : 'N/A'],
    ['Peak VRAM', m.estimated_vram_bytes_peak ? `${(m.estimated_vram_bytes_peak / 1024 / 1024).toFixed(1)} MB` : 'N/A'],
    ['GPU Timestamp Available', m.gpu_timestamp_available ? 'Yes' : 'No'],
  ];

  for (const [name, value] of allMetrics) {
    html += `<tr><td>${name}</td><td>${value}</td></tr>`;
  }

  html += `
      </table>
    </div>
    <div class="card">
      <p class="timestamp">Report generated by DOPPLER CLI</p>
    </div>
  </div>
</body>
</html>`;

  return html;
}
