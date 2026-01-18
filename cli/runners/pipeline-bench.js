

/**
 * Simple pipeline benchmark (for bench:all).
 */

import { appendRuntimeConfigParams } from '../args/index.js';

/**
 * Run pipeline benchmark.
 * @param {import('playwright').Page} page
 * @param {import('../args/index.js').CLIOptions} opts
 * @returns {Promise<import('../output.js').SuiteResult>}
 */
export async function runPipelineBenchmark(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('PIPELINE BENCHMARK');
  console.log('='.repeat(60));
  console.log(`  Model: ${opts.model}`);

  const runtimeConfig = opts.runtimeConfig;
  if (!runtimeConfig) {
    throw new Error('Runtime config is required for benchmarks.');
  }
  const benchmarkRun = runtimeConfig.shared.benchmark.run;

  const demoParams = new URLSearchParams();
  appendRuntimeConfigParams(demoParams, opts);
  await page.goto(`${opts.baseUrl}/d${demoParams.toString() ? `?${demoParams.toString()}` : ''}`, { timeout: opts.timeout });
  await page.waitForTimeout(1000);

  // Build script as string to avoid TypeScript module resolution issues
  // The import runs in browser context, not Node.js
  const config = JSON.stringify({
    promptName: benchmarkRun.promptName,
    customPrompt: benchmarkRun.customPrompt ?? undefined,
    maxNewTokens: benchmarkRun.maxNewTokens,
    warmupRuns: benchmarkRun.warmupRuns,
    timedRuns: benchmarkRun.timedRuns,
    sampling: benchmarkRun.sampling,
    debug: benchmarkRun.debug,
    profile: benchmarkRun.profile,
    useChatTemplate: benchmarkRun.useChatTemplate,
    captureMemoryTimeSeries: benchmarkRun.captureMemoryTimeSeries,
    memoryTimeSeriesIntervalMs: benchmarkRun.memoryTimeSeriesIntervalMs,
    runtimeConfig,
  });

  const script = `
    (async () => {
      const { PipelineBenchmark } = await import('./tests/benchmark/index.js');
      const config = ${config};
      const harness = new PipelineBenchmark(config);
      return await harness.run();
    })()
  `;

  try {
    const result = /** @type {any} */ (await page.evaluate(script));

    console.log(`\n  TTFT: ${result.metrics?.ttft_ms || 'N/A'}ms`);
    console.log(`  Prefill: ${result.metrics?.prefill_tokens_per_sec || 'N/A'} tok/s`);
    console.log(`  Decode: ${result.metrics?.decode_tokens_per_sec || 'N/A'} tok/s`);

    return {
      suite: 'bench:pipeline',
      passed: 1,
      failed: 0,
      skipped: 0,
      duration: result.metrics?.decode_ms_total || 0,
      results: [
        {
          name: 'pipeline',
          passed: true,
          duration: result.metrics?.decode_ms_total || 0,
        },
      ],
    };
  } catch (err) {
    console.log(`  \x1b[31mFAIL\x1b[0m: ${/** @type {Error} */ (err).message}`);
    return {
      suite: 'bench:pipeline',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration: 0,
      results: [
        {
          name: 'pipeline',
          passed: false,
          duration: 0,
          error: /** @type {Error} */ (err).message,
        },
      ],
    };
  }
}
