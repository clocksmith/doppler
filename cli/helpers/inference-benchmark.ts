/**
 * Inference benchmark runner (Playwright + browser harness).
 *
 * Uses a persistent Playwright profile to keep OPFS state between runs.
 */

import { resolve } from 'path';
import { readFile } from 'fs/promises';

import type { CLIOptions } from './types.js';

import { runBenchmarkBuild, ensureServerRunning, createBrowserContext, installLocalDopplerRoutes } from './utils.js';

function buildBenchmarkScript(opts: CLIOptions, modelPath: string, customPromptText: string | null): string {
  const kernelHints: Record<string, unknown> = {
    ...(opts.kernelHints ?? {}),
  };
  if (opts.computePrecision) kernelHints.computePrecision = opts.computePrecision;
  if (opts.q4kMatmul) kernelHints.q4kMatmul = opts.q4kMatmul;
  if (opts.f16Matmul) kernelHints.f16Matmul = opts.f16Matmul;
  if (opts.attentionPrefill) kernelHints.attentionPrefill = opts.attentionPrefill;
  if (opts.attentionDecode) kernelHints.attentionDecode = opts.attentionDecode;

  const runtime: Record<string, unknown> = {};
  if (opts.attentionKernel) runtime.attentionKernel = opts.attentionKernel;
  if (Object.keys(kernelHints).length > 0) runtime.kernelHints = kernelHints;

  const configObj: Record<string, unknown> = {
    modelPath,
    maxNewTokens: opts.maxTokens,
    warmupRuns: opts.warmup,
    timedRuns: opts.runs,
    sampling: {
      temperature: opts.temperature,
      topK: 1,
      topP: 1,
    },
    debug: opts.debug ?? false,  // When true, disables batching for layer-by-layer debugging
    debugLayers: opts.debugLayers,
    profile: opts.gpuProfile,  // GPU timestamp profiling for per-kernel timing
  };
  if (opts.runtimeConfig) {
    configObj.runtimeConfig = opts.runtimeConfig;
  }
  if (Object.keys(runtime).length > 0) {
    configObj.runtime = runtime;
  }

  if (customPromptText) {
    configObj.promptName = 'custom';
    configObj.customPrompt = customPromptText;
  } else {
    configObj.promptName = opts.prompt;
  }

  const config = JSON.stringify(configObj);

  let traceInit = '';
  if (opts.trace) {
    // Map old presets to new trace categories
    const traceCategories = opts.trace === 'full' ? 'all' : opts.trace === 'quick' ? 'logits,sample' : opts.trace;
    const traceLayers = opts.traceLayers?.length ? opts.traceLayers : [];
    traceInit = `
      // Set trace categories (unified API)
      const traceCategories = '${traceCategories}';
      const traceLayers = ${JSON.stringify(traceLayers)};
      const traceOptions = traceLayers.length ? { layers: traceLayers } : {};
      bench.setTrace(traceCategories, traceOptions);
      bench.setLogLevel('verbose');
      console.log('[Trace] Categories enabled: ' + traceCategories);
    `;
  }

  return `
    (async () => {
      const bench = await import('./tests/benchmark/index.js');

      // Enable benchmark mode to silence console.log during timing (unless debug mode)
      const debugMode = ${opts.debug ?? false};
      if (!debugMode) {
        bench.setBenchmarkMode(true);
      }

      ${traceInit}
      const progress = (phase, current, total) => {
        // Use console.warn since benchmark mode only silences log/debug/info
        console.warn('[Benchmark] ' + phase + ': ' + current + '/' + total);
      };
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

      return result;
    })()
  `;
}

export async function runFullInferenceBenchmark(opts: CLIOptions): Promise<any> {
  await runBenchmarkBuild(opts.verbose);
  if (!opts.noServer) {
    await ensureServerRunning(opts.baseUrl, opts.verbose);
  } else {
    console.log('No-server mode enabled (serving assets from disk)...');
  }

  let customPromptText: string | null = null;
  if (opts.text) {
    customPromptText = opts.text;
  } else if (opts.file) {
    try {
      customPromptText = await readFile(resolve(opts.file), 'utf-8');
      customPromptText = customPromptText.trim();
    } catch (err) {
      throw new Error(`Failed to read prompt file: ${opts.file} - ${(err as Error).message}`);
    }
  }

  const promptDisplay = customPromptText
    ? `custom: "${customPromptText.slice(0, 50)}${customPromptText.length > 50 ? '...' : ''}"`
    : opts.prompt;

  console.log(`\n${'─'.repeat(60)}`);
  console.log('DOPPLER Inference Benchmark');
  console.log(`${'─'.repeat(60)}`);
  console.log(`Model:      ${opts.model}`);
  console.log(`Prompt:     ${promptDisplay}`);
  console.log(`Max tokens: ${opts.maxTokens}`);
  console.log(`Warmup:     ${opts.warmup}`);
  console.log(`Runs:       ${opts.runs}`);
  console.log(`Retries:    ${opts.retries}`);
  console.log(`${'─'.repeat(60)}\n`);

  let lastError: Error | null = null;

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

    const relevantTags = ['[Benchmark]', '[Pipeline]', '[Loader]', '[DopplerLoader]', '[GPU]', '[Kernel]', '[Layer', '[LAYER', '[KERNEL]', '[KV]', '[ATTN]', '[FFN]', '[Embed]', '[Gather]', '[FUSED]', 'ERROR', 'WARN', 'error', 'Error'];
    page.on('console', (msg) => {
      const text = msg.text();
      const isRelevant = relevantTags.some((tag) => text.includes(tag));
      // If debugLayers are specified, suppress per-layer logs for non-checkpoint layers.
      // This keeps output focused during inference debugging while still allowing --verbose.
      if (opts.debugLayers?.length) {
        const m1 = text.match(/^\[Layer(\d+)\]/);
        const m2 = text.match(/^\[LAYER\]\[L(\d+)\]/);
        const layerStr = m1?.[1] ?? m2?.[1] ?? null;
        if (layerStr) {
          const layerIdx = Number(layerStr);
          if (!opts.debugLayers.includes(layerIdx)) {
            return;
          }
        }
      }
      if (opts.verbose || isRelevant) {
        console.log(`[browser] ${text}`);
      }
    });

    page.on('pageerror', (err) => {
      console.error(`[browser error] ${err.message}`);
    });

    try {
      console.log('Opening browser...');
      const benchUrl = `${opts.baseUrl}/d`;
      await page.goto(benchUrl, { timeout: 30000 });

      console.log('Waiting for WebGPU...');
      await page.waitForFunction(
        () => typeof navigator !== 'undefined' && 'gpu' in navigator,
        { timeout: 10000 }
      );

      await page.waitForFunction(
        () => typeof (window as any).dopplerReady === 'undefined' || (window as any).dopplerReady === true,
        { timeout: 5000 }
      ).catch(() => {});

      await page.waitForTimeout(500);

      const modelPath = `${opts.baseUrl}/models/${opts.model}`;
      const script = buildBenchmarkScript(opts, modelPath, customPromptText);

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
      lastError = err as Error;
      console.error(`\nAttempt ${attempt + 1} failed:`, lastError.message);
      await context.close();

      if (lastError.message.includes('timeout') && attempt === opts.retries) {
        break;
      }
    }
  }

  throw lastError || new Error('Benchmark failed after all retries');
}

export function formatBenchmarkResult(result: any): void {
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
