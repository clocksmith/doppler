// Derive means from retained request samples; never select the fastest sample.
import fs from 'node:fs/promises';
const directory = new URL('./benchmarks/', import.meta.url);
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const timing = samples => ({
  samples: samples.length,
  completeRequestMs: mean(samples.map(sample => sample.completeRequestMs)),
  firstTokenMs: mean(samples.map(sample => sample.firstTokenMs)),
  decodeTokensPerSecond: mean(samples.map(sample => sample.decodeTokensPerSecond)),
  minCompleteMs: Math.min(...samples.map(sample => sample.completeRequestMs)),
  maxCompleteMs: Math.max(...samples.map(sample => sample.completeRequestMs)),
});
const rows = [];
for (const file of (await fs.readdir(directory)).sort()) {
  if (!/^compare_\d+T\d+\.json$/.test(file)) continue;
  const report = JSON.parse(await fs.readFile(new URL(file, directory)));
  const sections = report.sections.compute ?? { warm: report.sections.warm };
  for (const [mode, section] of Object.entries(sections)) {
    if (!section?.doppler?.result) continue;
    const samples = section.doppler.result.metrics.requestTiming.samples;
    rows.push({ file, model: report.dopplerModelId, mode,
      prompt: report.prompt, promptTokens: section.doppler.result.comparisonFingerprint.identity.promptTokenIds.length,
      outputTokens: report.maxTokens,
      pairedComparable: section.pairedComparable,
      invalidReason: section.invalidReason,
      parity: section.outputParity,
      releaseClaimable: report.fairness?.releaseClaimable ?? false,
      doppler: timing(samples.map(sample => ({ completeRequestMs: sample.totalRunMs,
        firstTokenMs: sample.firstTokenMs, decodeTokensPerSecond: sample.decodeTokensPerSec }))),
      transformersjs: timing(section.transformersjs.runs.map(sample => ({
        completeRequestMs: sample.totalMs, firstTokenMs: sample.ttftMs,
        decodeTokensPerSecond: sample.decodeTokensPerSec,
      }))),
    });
  }
}
const native = [];
for (const file of (await fs.readdir(directory)).sort()) {
  if (!file.includes('llamacpp') || !file.endsWith('.json')) continue;
  const report = JSON.parse(await fs.readFile(new URL(file, directory)));
  const samples = report.samples.filter(sample => !sample.warmup);
  native.push({ file, error: report.error ?? null, serverVersion: report.serverVersion,
    exactTokenMatch: samples.every(sample => sample.exactTokenMatch === true),
    ...timing(samples.map(sample => ({ completeRequestMs: sample.completeRequestMs,
      firstTokenMs: sample.firstTokenMs, decodeTokensPerSecond: sample.timings.predicted_per_second }))),
  });
}
const doppler27 = JSON.parse(await fs.readFile(new URL('qwen27-doppler-node.json', directory)));
const onnx27 = JSON.parse(await fs.readFile(new URL('qwen27-onnx-webgpu.json', directory)));
const native27 = JSON.parse(await fs.readFile(new URL('qwen27-llamacpp-vulkan.json', directory)));
const prefix = (a, b) => {
  const position = a.findIndex((token, i) => token !== b[i]);
  return position < 0 ? Math.min(a.length, b.length) : position;
};
const dTokens = doppler27.samples[1].evidence.tokenIds;
const oTokens = onnx27.samples[1].generatedIds;
const nTokens = native27.samples[1].tokens;
const largeModel = {
  inputTokensMatch: JSON.stringify(doppler27.contract.inputTokenIds) === JSON.stringify(onnx27.samples[1].inputIds)
    && JSON.stringify(doppler27.contract.inputTokenIds) === JSON.stringify(native27.inputTokenIds),
  dopplerNativeMatchingPrefix: prefix(dTokens, nTokens),
  dopplerOnnxMatchingPrefix: prefix(dTokens, oTokens),
  doppler: timing(doppler27.samples.filter(sample => !sample.warmup).map(sample => ({
    completeRequestMs: sample.completeRequestMs, firstTokenMs: sample.evidence.stats.ttftMs,
    decodeTokensPerSecond: (sample.evidence.tokenIds.length - 1) * 1000 / sample.evidence.stats.decodeTimeMs,
  }))),
  onnx: timing(onnx27.samples.filter(sample => !sample.warmup).map(sample => ({
    completeRequestMs: sample.elapsedMs, firstTokenMs: sample.firstTokenMs,
    decodeTokensPerSecond: sample.tokensPerSecondAfterFirst,
  }))),
  claimBoundary: 'Doppler Node/WebGPU, ONNX browser/WebGPU, llama.cpp Vulkan; different quantized artifacts. Short-prompt evidence only; no Flash-Next qualification.',
};
await fs.writeFile(new URL('./summary.json', import.meta.url), `${JSON.stringify({
  schema: 'doppler.release-comparison-summary/v1', statistic: 'arithmetic mean of all timed samples; no outlier removal',
  rows, native, largeModel,
}, null, 2)}\n`);
console.log(JSON.stringify({ browserPairs: rows.length, nativeRuns: native.length, largeModel }));
