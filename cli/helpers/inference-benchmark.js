/**
 * Inference benchmark runner (Playwright + browser harness).
 *
 * Uses a persistent Playwright profile to keep OPFS state between runs.
 */

import { runBenchmarkBuild, ensureServerRunning, createBrowserContext, installLocalDopplerRoutes } from './utils.js';

/**
 * @param {import('./types.js').CLIOptions} opts
 * @param {string} modelPath
 * @returns {string}
 */
function buildBenchmarkScript(opts, modelPath) {
  const runtimeConfig = opts.runtimeConfig;
  if (!runtimeConfig) {
    throw new Error('Runtime config is required for benchmarks.');
  }
  const benchmarkRun = runtimeConfig.shared.benchmark.run;
  const benchmarkSampling = benchmarkRun.sampling;
  const customPrompt = benchmarkRun.customPrompt;
  const promptName = customPrompt ? 'custom' : benchmarkRun.promptName;

  /** @type {Record<string, unknown>} */
  const configObj = {
    modelPath,
    promptName,
    maxNewTokens: benchmarkRun.maxNewTokens,
    warmupRuns: benchmarkRun.warmupRuns,
    timedRuns: benchmarkRun.timedRuns,
    sampling: {
      temperature: benchmarkSampling.temperature,
      topK: benchmarkSampling.topK,
      topP: benchmarkSampling.topP,
    },
    debug: benchmarkRun.debug,
    profile: benchmarkRun.profile,
    runtimeConfig,
  };
  if (customPrompt) {
    configObj.customPrompt = customPrompt;
  }
  if (benchmarkRun.useChatTemplate !== undefined) {
    configObj.useChatTemplate = benchmarkRun.useChatTemplate;
  }

  const config = JSON.stringify(configObj);

  return `
    (async () => {
      const bench = await import('/doppler/tests/benchmark/index.js');

      // Enable benchmark mode to silence console.log during timing (unless debug mode)
      const debugMode = ${benchmarkRun.debug};
      if (!debugMode) {
        bench.setBenchmarkMode(true);
      }

      const progress = (phase, current, total) => {
        // Use console.warn since benchmark mode only silences log/debug/info
        console.warn('[Benchmark] ' + phase + ': ' + current + '/' + total);
      };

      try {
        progress('Loading model', 1, 1);
        const config = ${config};
        const harness = new bench.PipelineBenchmark(config);
        progress('Running benchmark', 1, 1);
        const result = await harness.run();
        progress('Complete', 1, 1);

        // Restore logging (if it was disabled)
        if (!debugMode) {
          bench.setBenchmarkMode(false);
        }

        // Emit standardized completion signals for CLI/automation detection
        console.log('[DOPPLER:RESULT] ' + JSON.stringify(result));
        console.log('[DOPPLER:DONE] ' + JSON.stringify({ status: 'success', elapsed: result.metrics?.total_ms || 0 }));

        return result;
      } catch (err) {
        // Restore logging before error signal
        if (!debugMode) {
          bench.setBenchmarkMode(false);
        }
        // Emit standardized error signals for CLI/automation detection
        console.log('[DOPPLER:ERROR] ' + JSON.stringify({ error: err.message }));
        console.log('[DOPPLER:DONE] ' + JSON.stringify({ status: 'error', elapsed: 0, error: err.message }));
        throw err;
      }
    })()
  `;
}

/**
 * @param {import('./types.js').CLIOptions} opts
 * @returns {Promise<any>}
 */
export async function runFullInferenceBenchmark(opts) {
  await runBenchmarkBuild(opts.verbose);
  if (!opts.noServer) {
    await ensureServerRunning(opts.baseUrl, opts.verbose);
  } else {
    console.log('No-server mode enabled (serving assets from disk)...');
  }

  const runtimeConfig = opts.runtimeConfig;
  if (!runtimeConfig) {
    throw new Error('Runtime config is required for benchmarks.');
  }
  const benchmarkRun = runtimeConfig.shared.benchmark.run;
  const customPromptText = benchmarkRun.customPrompt;
  const promptDisplay = customPromptText
    ? `custom: "${customPromptText.slice(0, 50)}${customPromptText.length > 50 ? '...' : ''}"`
    : benchmarkRun.promptName;

  console.log(`\n${'─'.repeat(60)}`);
  console.log('DOPPLER Inference Benchmark');
  console.log(`${'─'.repeat(60)}`);
  console.log(`Model:      ${opts.model}`);
  console.log(`Prompt:     ${promptDisplay}`);
  console.log(`Max tokens: ${benchmarkRun.maxNewTokens}`);
  console.log(`Warmup:     ${benchmarkRun.warmupRuns}`);
  console.log(`Runs:       ${benchmarkRun.timedRuns}`);
  console.log(`Retries:    ${opts.retries}`);
  console.log(`${'─'.repeat(60)}\n`);

  /** @type {Error | null} */
  let lastError = null;

  for (let attempt = 0; attempt <= opts.retries; attempt++) {
    if (attempt > 0) {
      const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
      console.log(`\nRetrying in ${delay}ms... (attempt ${attempt + 1}/${opts.retries + 1})`);
      await new Promise((r) => setTimeout(r, delay));
    }

    const context = await createBrowserContext(opts, { scope: 'bench', devtools: true });
    const page = context.pages()[0] || await context.newPage();
    if (opts.noServer) {
      await installLocalDopplerRoutes(page, opts);
    }

    const relevantTags = ['[Benchmark]', '[Pipeline]', '[Loader]', '[DopplerLoader]', '[GPU]', '[Kernel]', '[Layer', '[LAYER', '[KERNEL]', '[KV]', '[ATTN]', '[FFN]', '[Embed]', '[Gather]', '[FUSED]', '[Profile]', 'ERROR', 'WARN', 'error', 'Error'];
    page.on('console', (msg) => {
      const text = msg.text();
      const isRelevant = relevantTags.some((tag) => text.includes(tag));
      if (opts.verbose || isRelevant) {
        console.log(`[browser] ${text}`);
      }
    });

    page.on('pageerror', (err) => {
      console.error(`[browser error] ${err.message}`);
    });

    try {
      console.log('Opening browser...');
      const benchUrl = `${opts.baseUrl}/doppler/tests/harness.html?mode=bench`;
      await page.goto(benchUrl, { timeout: 30000 });

      console.log('Waiting for WebGPU...');
      await page.waitForFunction(
        () => typeof navigator !== 'undefined' && 'gpu' in navigator,
        { timeout: 10000 }
      );

      await page.waitForFunction(
        () => /** @type {any} */ (window).dopplerReady === true,
        { timeout: 5000 }
      ).catch(() => {});

      await page.waitForTimeout(500);

      const modelPath = `${opts.baseUrl}/models/${opts.model}`;
      const script = buildBenchmarkScript(opts, modelPath);

      console.log('Running benchmark...');
      const startTime = Date.now();

      const result = await Promise.race([
        page.evaluate(script),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Benchmark timeout')), opts.timeout)
        ),
      ]);

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`\nBenchmark complete! (${elapsed}s)`);

      if (!opts.headless) {
        console.log('Keeping browser open for 10s (headed mode)...');
        await page.waitForTimeout(10000);
      }

      await context.close();
      return result;
    } catch (err) {
      lastError = /** @type {Error} */ (err);
      console.error(`\nAttempt ${attempt + 1} failed:`, lastError.message);
      await context.close();

      if (lastError.message.includes('timeout') && attempt === opts.retries) {
        break;
      }
    }
  }

  throw lastError || new Error('Benchmark failed after all retries');
}

/**
 * @param {any} result
 * @returns {void}
 */
export function formatBenchmarkResult(result) {
  const m = result.metrics;
  const model = result.model?.modelName ?? result.model?.modelId ?? 'unknown';
  const prompt = result.workload?.promptName ?? 'unknown';

  console.log(`\n--- ${model} (${prompt}) ---`);
  console.log(`TTFT:           ${m.ttft_ms} ms`);
  console.log(`Prefill:        ${m.prefill_ms} ms (${m.prefill_tokens_per_sec} tok/s)`);
  console.log(`Decode:         ${m.decode_ms_total} ms (${m.decode_tokens_per_sec} tok/s)`);
  console.log(`GPU Submits:    ${m.gpu_submit_count_prefill} prefill, ${m.gpu_submit_count_decode} decode`);

  if (m.decode_ms_per_token_p50) {
    console.log(`Latency P50/90/99: ${m.decode_ms_per_token_p50}/${m.decode_ms_per_token_p90}/${m.decode_ms_per_token_p99} ms`);
  }

  if (m.estimated_vram_bytes_peak) {
    const vramMB = (m.estimated_vram_bytes_peak / 1024 / 1024).toFixed(1);
    console.log(`Peak VRAM:      ${vramMB} MB`);
  }
}
