

import { compareResults } from './comparison.js';

// ============================================================================
// SVG Chart Generation
// ============================================================================


export function generateSVGBarChart(data, width = 400, height = 200, title = '') {
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


export function generateSVGLineChart(data, width = 400, height = 150, title = '', yLabel = '') {
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

export function generateMultiLineChart(series, width = 600, height = 200, title = '', yLabel = '') {
  if (!series || series.length === 0) return '';
  const dataLength = Math.max(...series.map(s => s.values.length));
  if (!dataLength) return '';

  const margin = { top: 30, right: 90, bottom: 30, left: 50 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const allValues = series.flatMap(s => s.values);
  const maxValue = Math.max(...allValues) * 1.1;
  const minValue = Math.min(...allValues) * 0.9;
  const range = maxValue - minValue || 1;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<style>
    .chart-title { font: bold 12px sans-serif; }
    .axis-label { font: 10px sans-serif; fill: #666; }
    .grid-line { stroke: #e0e0e0; stroke-width: 1; }
    .legend { font: 10px sans-serif; }
  </style>`;

  if (title) {
    svg += `<text x="${width / 2}" y="15" text-anchor="middle" class="chart-title">${title}</text>`;
  }
  if (yLabel) {
    svg += `<text x="12" y="${height / 2}" text-anchor="middle" transform="rotate(-90, 12, ${height / 2})" class="axis-label">${yLabel}</text>`;
  }

  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (chartHeight * i) / 4;
    svg += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid-line"/>`;
    const val = maxValue - (range * i) / 4;
    svg += `<text x="${margin.left - 5}" y="${y + 4}" text-anchor="end" class="axis-label">${val.toFixed(1)}</text>`;
  }

  series.forEach((s, idx) => {
    const points = s.values.map((v, i) => {
      const x = margin.left + (i / Math.max(dataLength - 1, 1)) * chartWidth;
      const y = margin.top + ((maxValue - v) / range) * chartHeight;
      return `${x},${y}`;
    });
    svg += `<polyline points="${points.join(' ')}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    const legendY = margin.top + idx * 14;
    svg += `<rect x="${width - margin.right + 10}" y="${legendY - 8}" width="10" height="10" fill="${s.color}"/>`;
    svg += `<text x="${width - margin.right + 25}" y="${legendY}" class="legend">${s.label}</text>`;
  });

  svg += '</svg>';
  return svg;
}

// ============================================================================
// Memory Time Series Chart
// ============================================================================

export function generateMemoryTimeSeriesChart(timeSeries, width = 700, height = 200, title = 'Memory Usage Over Time') {
  if (!timeSeries || timeSeries.length === 0) return '';

  const margin = { top: 30, right: 80, bottom: 40, left: 70 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const gpuValues = timeSeries.map(d => d.gpu || 0);
  const jsHeapValues = timeSeries.map(d => d.jsHeap || 0);
  const times = timeSeries.map(d => d.t || 0);

  const maxGpu = Math.max(...gpuValues) / (1024 * 1024 * 1024); // GB
  const maxTime = Math.max(...times);
  const maxValue = maxGpu * 1.1;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<style>
    .chart-title { font: bold 12px sans-serif; }
    .axis-label { font: 10px sans-serif; fill: #666; }
    .legend { font: 10px sans-serif; }
    .line-gpu { fill: none; stroke: #4a90d9; stroke-width: 2; }
    .line-requested { fill: none; stroke: #f59e0b; stroke-width: 2; stroke-dasharray: 4,2; }
    .grid-line { stroke: #e0e0e0; stroke-width: 1; }
    .phase-line { stroke: #ef4444; stroke-width: 1; stroke-dasharray: 2,2; }
    .phase-label { font: 9px sans-serif; fill: #ef4444; }
  </style>`;

  svg += `<text x="${width / 2}" y="15" text-anchor="middle" class="chart-title">${title}</text>`;

  // Y-axis grid and labels
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (chartHeight * i) / 4;
    svg += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid-line"/>`;
    const val = (maxValue * (4 - i)) / 4;
    svg += `<text x="${margin.left - 5}" y="${y + 4}" text-anchor="end" class="axis-label">${val.toFixed(1)} GB</text>`;
  }

  // X-axis label
  svg += `<text x="${margin.left + chartWidth / 2}" y="${height - 5}" text-anchor="middle" class="axis-label">Time (ms)</text>`;

  // GPU allocated line
  const gpuPoints = timeSeries.map((d, i) => {
    const x = margin.left + (d.t / maxTime) * chartWidth;
    const y = margin.top + chartHeight - ((d.gpu / (1024 * 1024 * 1024)) / maxValue) * chartHeight;
    return `${x},${y}`;
  });
  svg += `<polyline points="${gpuPoints.join(' ')}" class="line-gpu"/>`;

  // GPU requested line (if available)
  if (timeSeries[0]?.gpuRequested) {
    const requestedPoints = timeSeries.map((d, i) => {
      const x = margin.left + (d.t / maxTime) * chartWidth;
      const y = margin.top + chartHeight - ((d.gpuRequested / (1024 * 1024 * 1024)) / maxValue) * chartHeight;
      return `${x},${y}`;
    });
    svg += `<polyline points="${requestedPoints.join(' ')}" class="line-requested"/>`;
  }

  // Phase markers
  const phases = timeSeries.filter(d => d.phase && d.phase !== 'sample');
  phases.forEach(d => {
    const x = margin.left + (d.t / maxTime) * chartWidth;
    svg += `<line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + chartHeight}" class="phase-line"/>`;
    svg += `<text x="${x + 3}" y="${margin.top + 12}" class="phase-label">${d.phase}</text>`;
  });

  // Legend
  svg += `<rect x="${width - 75}" y="${margin.top}" width="12" height="12" fill="#4a90d9"/>`;
  svg += `<text x="${width - 60}" y="${margin.top + 10}" class="legend">Allocated</text>`;
  svg += `<rect x="${width - 75}" y="${margin.top + 18}" width="12" height="12" fill="#f59e0b"/>`;
  svg += `<text x="${width - 60}" y="${margin.top + 28}" class="legend">Requested</text>`;

  svg += '</svg>';
  return svg;
}

// ============================================================================
// Latency Histogram
// ============================================================================

export function generateLatencyHistogram(latencies, width = 400, height = 180, title = 'Latency Distribution') {
  if (!latencies || latencies.length === 0) return '';

  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Create histogram bins
  const minVal = Math.min(...latencies);
  const maxVal = Math.max(...latencies);
  const binCount = 15;
  const binWidth = (maxVal - minVal) / binCount;
  const bins = Array(binCount).fill(0);

  latencies.forEach(v => {
    const binIndex = Math.min(Math.floor((v - minVal) / binWidth), binCount - 1);
    bins[binIndex]++;
  });

  const maxBin = Math.max(...bins);
  const barW = chartWidth / binCount - 2;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<style>
    .chart-title { font: bold 12px sans-serif; }
    .axis-label { font: 10px sans-serif; fill: #666; }
    .bar { fill: #4a90d9; }
    .bar:hover { fill: #2563eb; }
    .grid-line { stroke: #e0e0e0; stroke-width: 1; }
  </style>`;

  svg += `<text x="${width / 2}" y="15" text-anchor="middle" class="chart-title">${title}</text>`;

  // Y-axis
  for (let i = 0; i <= 3; i++) {
    const y = margin.top + (chartHeight * i) / 3;
    svg += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid-line"/>`;
    const val = Math.round((maxBin * (3 - i)) / 3);
    svg += `<text x="${margin.left - 5}" y="${y + 4}" text-anchor="end" class="axis-label">${val}</text>`;
  }

  // Bars
  bins.forEach((count, i) => {
    const barHeight = (count / maxBin) * chartHeight;
    const x = margin.left + i * (chartWidth / binCount) + 1;
    const y = margin.top + chartHeight - barHeight;
    svg += `<rect x="${x}" y="${y}" width="${barW}" height="${barHeight}" class="bar"/>`;
  });

  // X-axis labels
  svg += `<text x="${margin.left}" y="${height - 5}" text-anchor="start" class="axis-label">${minVal.toFixed(0)}ms</text>`;
  svg += `<text x="${width - margin.right}" y="${height - 5}" text-anchor="end" class="axis-label">${maxVal.toFixed(0)}ms</text>`;

  svg += '</svg>';
  return svg;
}

// ============================================================================
// Buffer Pool Stats Chart
// ============================================================================

export function generateBufferPoolChart(metrics, width = 350, height = 180) {
  if (!metrics) return '';

  const hitRate = metrics.buffer_pool_hit_rate_pct || 0;
  const missRate = 100 - hitRate;

  const cx = width / 2;
  const cy = 90;
  const r = 60;

  // Calculate arc paths for donut chart
  const hitAngle = (hitRate / 100) * 2 * Math.PI - Math.PI / 2;
  const hitEndX = cx + r * Math.cos(hitAngle);
  const hitEndY = cy + r * Math.sin(hitAngle);
  const largeArc = hitRate > 50 ? 1 : 0;

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<style>
    .chart-title { font: bold 12px sans-serif; }
    .center-text { font: bold 20px sans-serif; fill: #333; }
    .center-label { font: 11px sans-serif; fill: #666; }
    .legend { font: 10px sans-serif; }
  </style>`;

  svg += `<text x="${width / 2}" y="15" text-anchor="middle" class="chart-title">Buffer Pool Efficiency</text>`;

  // Background circle (miss)
  svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#e5e7eb" stroke-width="20"/>`;

  // Hit rate arc
  if (hitRate > 0) {
    svg += `<path d="M ${cx} ${cy - r} A ${r} ${r} 0 ${largeArc} 1 ${hitEndX} ${hitEndY}" fill="none" stroke="#22c55e" stroke-width="20" stroke-linecap="round"/>`;
  }

  // Center text
  svg += `<text x="${cx}" y="${cy + 5}" text-anchor="middle" class="center-text">${hitRate.toFixed(1)}%</text>`;
  svg += `<text x="${cx}" y="${cy + 20}" text-anchor="middle" class="center-label">hit rate</text>`;

  // Legend and stats
  const stats = [
    { label: 'Allocations', value: metrics.buffer_pool_allocations_total || 0, color: '#94a3b8' },
    { label: 'Reuses', value: metrics.buffer_pool_reuses_total || 0, color: '#22c55e' },
  ];

  stats.forEach((s, i) => {
    const y = height - 30 + i * 14;
    svg += `<rect x="20" y="${y - 8}" width="10" height="10" fill="${s.color}"/>`;
    svg += `<text x="35" y="${y}" class="legend">${s.label}: ${s.value.toLocaleString()}</text>`;
  });

  svg += '</svg>';
  return svg;
}

// ============================================================================
// HTML Report Generation
// ============================================================================


export function generateHTMLReport(results, baseline) {
  const isArray = Array.isArray(results);
  const resultList = isArray ? results : [results];
  const firstResult = resultList[0];

  const model = firstResult.model?.modelName || firstResult.model?.modelId || 'Unknown Model';
  const timestamp = new Date().toISOString();
  const env = firstResult.env || {};

  const modelInfo = firstResult.model || {};
  const modelSizeMB = modelInfo.totalSizeBytes ? (modelInfo.totalSizeBytes / 1024 / 1024).toFixed(1) : null;

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
    .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
    .metric { padding: 15px; background: #f8f9fa; border-radius: 6px; }
    .metric-sm { padding: 12px; background: #f8f9fa; border-radius: 6px; text-align: center; }
    .metric-value { font-size: 28px; font-weight: bold; color: #4a90d9; }
    .metric-value-sm { font-size: 20px; font-weight: bold; color: #4a90d9; }
    .metric-label { font-size: 14px; color: #666; }
    .metric-label-sm { font-size: 12px; color: #666; }
    .metric-unit { font-size: 14px; color: #999; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #e0e0e0; }
    th { background: #f8f9fa; font-weight: 600; }
    .better { color: #28a745; }
    .worse { color: #dc3545; }
    .chart-container { margin: 20px 0; text-align: center; }
    .env-info { font-size: 13px; color: #666; }
    .timestamp { font-size: 12px; color: #999; }
    .model-badge { display: inline-block; background: #4a90d9; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px; margin-right: 8px; }
    .quant-badge { background: #f59e0b; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>DOPPLER Benchmark Report</h1>
      <p class="timestamp">Generated: ${timestamp}</p>
      <div class="env-info">
        <strong>Browser:</strong> ${env.browser?.name || 'Unknown'} ${env.browser?.version || ''} |
        <strong>GPU:</strong> ${env.gpu?.description || env.gpu?.device || 'Unknown'} |
        <strong>OS:</strong> ${env.os?.name || 'Unknown'}
        ${env.webgpu?.hasF16 ? ' | <strong>F16:</strong> Yes' : ''}
        ${env.webgpu?.hasSubgroups ? ' | <strong>Subgroups:</strong> Yes' : ''}
      </div>
    </div>

    <div class="card">
      <h2>Model: ${model}</h2>
      <div style="margin-bottom: 15px;">
        ${modelInfo.quantization ? `<span class="model-badge quant-badge">${modelInfo.quantization}</span>` : ''}
        ${modelInfo.numLayers ? `<span class="model-badge">${modelInfo.numLayers} layers</span>` : ''}
        ${modelInfo.hiddenSize ? `<span class="model-badge">${modelInfo.hiddenSize} hidden</span>` : ''}
      </div>
      <div class="grid-4">
        ${modelSizeMB ? `
        <div class="metric-sm">
          <div class="metric-value-sm">${modelSizeMB}<span class="metric-unit">MB</span></div>
          <div class="metric-label-sm">Model Size</div>
        </div>` : ''}
        ${modelInfo.tensorCount ? `
        <div class="metric-sm">
          <div class="metric-value-sm">${modelInfo.tensorCount}</div>
          <div class="metric-label-sm">Tensors</div>
        </div>` : ''}
        ${modelInfo.numLayers ? `
        <div class="metric-sm">
          <div class="metric-value-sm">${modelInfo.numLayers}</div>
          <div class="metric-label-sm">Layers</div>
        </div>` : ''}
        ${modelInfo.hiddenSize ? `
        <div class="metric-sm">
          <div class="metric-value-sm">${modelInfo.hiddenSize}</div>
          <div class="metric-label-sm">Hidden Size</div>
        </div>` : ''}
      </div>
    </div>
`;

  for (const result of resultList) {
    const m = result.metrics;
    const prompt = result.workload?.promptName || 'default';
    const quality = result.quality;
    const peakVramMB = m.estimated_vram_bytes_peak
      ? (m.estimated_vram_bytes_peak / 1024 / 1024).toFixed(1)
      : null;
    const peakVramRequestedMB = m.estimated_vram_bytes_peak_requested
      ? (m.estimated_vram_bytes_peak_requested / 1024 / 1024).toFixed(1)
      : null;
    const decodeProfileSteps = (result.raw?.decode_step_profile_ms || [])
      .filter((step) => step && !step.batch && Number.isFinite(step.step))
      .sort((a, b) => a.step - b.step);

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
        ${peakVramMB ? `
        <div class="metric">
          <div class="metric-value">${peakVramMB}<span class="metric-unit">MB</span></div>
          <div class="metric-label">Peak VRAM (allocated)</div>
        </div>
        ` : ''}
        ${peakVramRequestedMB ? `
        <div class="metric">
          <div class="metric-value">${peakVramRequestedMB}<span class="metric-unit">MB</span></div>
          <div class="metric-label">Peak VRAM (requested)</div>
        </div>
        ` : ''}
      </div>
`;

    if (result.raw?.decode_latencies_ms?.length > 0) {
      const latencies = result.raw.decode_latencies_ms;
      const tokPerSec = latencies.map((ms) => (ms > 0 ? 1000 / ms : 0));
      html += `
      <h3>Decode Performance</h3>
      <div class="chart-container" style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
        ${generateSVGLineChart(latencies, 550, 150, 'Decode Latency per Token', 'ms')}
        ${generateLatencyHistogram(latencies, 350, 150, 'Latency Distribution')}
      </div>
      <div class="chart-container">
        ${generateSVGLineChart(tokPerSec, 800, 150, 'Decode Throughput per Token', 'tok/s')}
      </div>
`;
    }

    if (decodeProfileSteps.length > 1) {
      const kernelTotals = new Map();
      for (const step of decodeProfileSteps) {
        for (const [label, time] of Object.entries(step.timings || {})) {
          const prev = kernelTotals.get(label) || { total: 0, count: 0 };
          kernelTotals.set(label, { total: prev.total + time, count: prev.count + 1 });
        }
      }
      const kernelsByAvg = [...kernelTotals.entries()]
        .map(([label, stats]) => ({ label, avg: stats.total / Math.max(stats.count, 1) }))
        .sort((a, b) => b.avg - a.avg)
        .slice(0, 3);
      const colors = ['#2563eb', '#10b981', '#f59e0b'];
      const series = kernelsByAvg.map((kernel, idx) => ({
        label: kernel.label,
        color: colors[idx % colors.length],
        values: decodeProfileSteps.map((step) => step.timings?.[kernel.label] ?? 0),
      }));
      const attentionEntry = kernelsByAvg.find((k) => k.label === 'attention');
      let crossover = null;
      if (attentionEntry) {
        for (const step of decodeProfileSteps) {
          const timings = step.timings || {};
          const attention = timings.attention ?? 0;
          const matmul = timings.matmul ?? 0;
          const fused = timings.matmul_rmsnorm_fused ?? 0;
          if (attention >= Math.max(matmul, fused)) {
            crossover = step.step ?? null;
            break;
          }
        }
      }

      if (series.length > 0) {
        html += `
      <h3>Decode Kernel Timing</h3>
      <div class="chart-container">
        ${generateMultiLineChart(series, 800, 220, 'Top Decode Kernels per Step', 'ms')}
      </div>
      <table>
        <tr><th>Top Kernels (avg ms)</th><th>Attention Crossover Step</th></tr>
        <tr>
          <td>${kernelsByAvg.map((k) => `${k.label}: ${k.avg.toFixed(2)}ms`).join(', ')}</td>
          <td>${crossover ?? 'n/a'}</td>
        </tr>
      </table>
`;
      }
    }

    // Memory time series chart
    if (result.raw?.memory_time_series?.length > 0) {
      html += `
      <h3>Memory Usage</h3>
      <div class="chart-container">
        ${generateMemoryTimeSeriesChart(result.raw.memory_time_series, 800, 200, 'GPU Memory Over Time')}
      </div>
`;
    }

    // Buffer pool efficiency
    if (m.buffer_pool_hit_rate_pct !== undefined) {
      html += `
      <h3>Resource Efficiency</h3>
      <div class="chart-container" style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
        ${generateBufferPoolChart(m, 300, 180)}
      </div>
      <table>
        <tr><th>Buffer Pool Hit Rate</th><th>Decode Ring Reuse</th><th>Effective Reuse</th><th>FFN Fused Down+Norm</th></tr>
        <tr>
          <td>${m.buffer_pool_hit_rate_pct}%</td>
          <td>${m.decode_ring_reuse_rate_pct ?? 'n/a'}%</td>
          <td>${m.buffer_reuse_effective_pct ?? 'n/a'}%</td>
          <td>${m.decode_fused_down_norm_used === undefined ? 'n/a' : (m.decode_fused_down_norm_used ? 'yes' : 'no')}</td>
        </tr>
      </table>
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

    if (quality) {
      const status = quality.ok ? 'ok' : 'fail';
      const reasons = quality.reasons?.length ? quality.reasons.join(', ') : 'none';
      const warnings = quality.warnings?.length ? quality.warnings.join(', ') : 'none';
      html += `
      <h3>Output Quality</h3>
      <table>
        <tr><th>Status</th><th>Reasons</th><th>Warnings</th></tr>
        <tr>
          <td>${status}</td>
          <td>${reasons}</td>
          <td>${warnings}</td>
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
