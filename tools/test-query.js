#!/usr/bin/env node


import { chromium } from 'playwright';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import * as readline from 'readline';
import { loadConfig } from '../cli/config/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);


function parseArgs(argv) {
  const opts = { config: null, help: false };
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      i++;
      continue;
    }
    if (arg === '--config' || arg === '-c') {
      opts.config = argv[i + 1] || null;
      i += 2;
      continue;
    }
    if (!arg.startsWith('-') && !opts.config) {
      opts.config = arg;
      i++;
      continue;
    }
    console.error(`Unknown argument: ${arg}`);
    opts.help = true;
    break;
  }
  return opts;
}

function printHelp() {
  console.log(`
Quick test query for DOPPLER

Usage:
  doppler --config <ref>

Config requirements:
  model (string, required)
  tools.testQuery.baseUrl (string, required)
  tools.testQuery.repl (boolean, required)
  tools.testQuery.prompt (string|null, required; null uses runtime.inference.prompt)

REPL Commands:
  <text>               Run inference with prompt
  /clear               Clear KV cache (new conversation)
  /reload              Reload model from scratch
  /quit                Exit
`);
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be a boolean`);
  }
}

function assertStringOrNull(value, label) {
  if (value === null) return;
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string or null`);
  }
}


async function launchBrowser() {
  const userDataDir = resolve(__dirname, '../.benchmark-cache');
  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: false,
    devtools: true,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--auto-open-devtools-for-tabs'],
  });

  const page = context.pages()[0] || await context.newPage();

  // Forward ALL console logs
  page.on('console', (msg) => {
    const type = msg.type();
    const prefix = type === 'error' ? '\x1b[31m[err]\x1b[0m' :
                   type === 'warning' ? '\x1b[33m[warn]\x1b[0m' :
                   '\x1b[90m[log]\x1b[0m';
    console.log(`${prefix} ${msg.text()}`);
  });

  page.on('pageerror', (err) => {
    console.error(`\x1b[31m[page error]\x1b[0m ${err.message}`);
  });

  return { context, page };
}


async function loadModel(page, baseUrl, model) {
  // Use the DopplerDemo app that's already loaded on the page
  // This avoids module loading issues and reuses the existing infrastructure

  await page.evaluate(
    async ({ model }) => {
      
      const w =  (window);

      // Wait for the app to be ready
      let retries = 0;
      while (!w.dopplerDemo && retries < 50) {
        await new Promise(r => setTimeout(r, 100));
        retries++;
      }

      if (!w.dopplerDemo) {
        throw new Error('DopplerDemo app not found on page');
      }

      console.log(`[Test] Looking for model matching: ${model}`);

      // Find the first available model that matches the search term
      // The model selector stores the registry, but it's private
      // Let's just click the first model button in the UI instead
      const modelBtns = document.querySelectorAll('[data-model-key]');
      
      let targetBtn = null;

      for (const btn of modelBtns) {
        const key =  (btn).dataset.modelKey || '';
        const text = btn.textContent?.toLowerCase() || '';
        if (key.includes(model.toLowerCase()) || text.includes(model.toLowerCase())) {
          targetBtn =  (btn);
          break;
        }
      }

      // If no match, just use the first available model
      if (!targetBtn && modelBtns.length > 0) {
        targetBtn =  (modelBtns[0]);
        console.log(`[Test] No exact match, using first model`);
      }

      if (!targetBtn) {
        throw new Error('No models available. Is the server running?');
      }

      const modelKey = targetBtn.dataset.modelKey;
      console.log(`[Test] Selecting model: ${modelKey}`);

      // Click the Run button for this model
      const runBtn =  (targetBtn.querySelector('.model-run-btn'));
      if (runBtn) {
        runBtn.click();
      } else {
        // Fallback: click the model item itself
        targetBtn.click();
      }

      // Wait for model to load
      let loadRetries = 0;
      while (loadRetries < 300) { // 30 seconds max
        await new Promise(r => setTimeout(r, 100));
        const status = w.dopplerDemo.getStatus();
        if (status.model) {
          console.log(`[Test] Model loaded: ${status.model}`);
          return;
        }
        loadRetries++;
      }

      throw new Error('Model load timed out');
    },
    { model }
  );
}


async function runQuery(page, prompt) {
  return await page.evaluate(
    async ({ prompt }) => {
      
      const w =  (window);
      const { getRuntimeConfig } = await import('/doppler/src/config/index.js');

      if (!w.dopplerDemo?.pipeline) {
        throw new Error('Model not loaded. Load a model first.');
      }

      const runtimeConfig = getRuntimeConfig();
      const maxTokens = runtimeConfig.inference.batching.maxTokens;
      const sampling = runtimeConfig.inference.sampling;

      const pipeline = w.dopplerDemo.pipeline;
      
      const tokens = [];
      const start = performance.now();

      console.log(`[Test] Generating: "${prompt.slice(0, 50)}${prompt.length > 50 ? '...' : ''}"`);

      for await (const token of pipeline.generate(prompt, {
        maxTokens,
        temperature: sampling.temperature,
        topK: sampling.topK,
        topP: sampling.topP,
      })) {
        tokens.push(token);
        // Stream to console
        console.log(`[tok] ${JSON.stringify(token)}`);
      }

      const elapsed = performance.now() - start;
      return {
        output: tokens.join(''),
        tokenCount: tokens.length,
        elapsedMs: elapsed,
        tokensPerSec: (tokens.length / elapsed) * 1000,
      };
    },
    { prompt }
  );
}


async function clearKVCache(page) {
  await page.evaluate(() => {
    
    const w =  (window);
    if (w.dopplerDemo?.clearConversation) {
      w.dopplerDemo.clearConversation();
      console.log('[Test] Conversation cleared');
    } else if (w.dopplerDemo?.pipeline?.clearKVCache) {
      w.dopplerDemo.pipeline.clearKVCache();
      console.log('[Test] KV cache cleared');
    } else {
      console.log('[Test] No KV cache to clear');
    }
  });
}


async function runRepl(page, opts) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log('\n\x1b[36mDOPPLER REPL\x1b[0m - Type prompts or commands');
  console.log('Commands: /clear /reload /quit\n');

  
  const prompt = () => {
    rl.question('\x1b[32m>\x1b[0m ', async (input) => {
      const line = input.trim();

      if (!line) {
        prompt();
        return;
      }

      try {
        if (line === '/quit' || line === '/exit' || line === '/q') {
          console.log('Bye!');
          rl.close();
          process.exit(0);
        } else if (line === '/clear' || line === '/c') {
          await clearKVCache(page);
          console.log('KV cache cleared.\n');
        } else if (line === '/reload' || line === '/r') {
          console.log('Reloading model...');
          await page.reload();
          await page.waitForFunction(() => 'gpu' in navigator, { timeout: 10000 });
          await loadModel(page, opts.baseUrl, opts.model);
          console.log('Model reloaded.\n');
        } else if (line.startsWith('/')) {
          console.log('Unknown command. Try: /clear /reload /quit\n');
        } else {
          // Run inference
          const result = await runQuery(page, line);
          console.log(`\n\x1b[33m${result.output}\x1b[0m`);
          console.log(`\x1b[90m[${result.tokenCount} tokens, ${result.elapsedMs.toFixed(0)}ms, ${result.tokensPerSec.toFixed(1)} tok/s]\x1b[0m\n`);
        }
      } catch (err) {
        console.error(`\x1b[31mError: ${ (err).message}\x1b[0m\n`);
      }

      prompt();
    });
  };

  prompt();
}


async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.help) {
    printHelp();
    process.exit(0);
  }
  if (!parsed.config) {
    console.error('Error: --config is required');
    printHelp();
    process.exit(1);
  }

  const loadedConfig = await loadConfig(parsed.config);
  const raw = loadedConfig.raw ?? {};
  const toolConfig = raw.tools?.testQuery;

  assertString(raw.model, 'model');
  if (!toolConfig || typeof toolConfig !== 'object') {
    throw new Error('tools.testQuery is required in config');
  }
  assertString(toolConfig.baseUrl, 'tools.testQuery.baseUrl');
  assertBoolean(toolConfig.repl, 'tools.testQuery.repl');
  if (!('prompt' in toolConfig)) {
    throw new Error('tools.testQuery.prompt is required in config');
  }
  assertStringOrNull(toolConfig.prompt, 'tools.testQuery.prompt');

  const opts = {
    prompt: toolConfig.prompt,
    model: raw.model,
    baseUrl: toolConfig.baseUrl,
    repl: toolConfig.repl,
  };

  console.log(`\n\x1b[36mDOPPLER Test Query\x1b[0m`);
  console.log(`${'─'.repeat(50)}`);
  console.log(`Model:      ${opts.model}`);
  console.log(`Config:     ${(loadedConfig.chain ?? []).join(' -> ')}`);
  console.log(`Mode:       ${opts.repl ? 'REPL (interactive)' : 'single query'}`);
  console.log(`${'─'.repeat(50)}\n`);

  console.log('Launching browser...');
  const { context, page } = await launchBrowser();

  try {
    await page.goto(opts.baseUrl, { timeout: 30000 });
    console.log('Waiting for WebGPU...');
    await page.waitForFunction(() => 'gpu' in navigator, { timeout: 10000 });

    await page.evaluate(async (runtimeConfig) => {
      const { setRuntimeConfig } = await import('/doppler/src/config/index.js');
      setRuntimeConfig(runtimeConfig);
    }, loadedConfig.runtime);

    console.log('Loading model...');
    await loadModel(page, opts.baseUrl, opts.model);

    if (opts.repl) {
      // Interactive REPL mode
      await runRepl(page, opts);
    } else {
      // Single query mode
      console.log('\nRunning inference...');
      const prompt = opts.prompt ?? loadedConfig.runtime?.inference?.prompt;
      if (!prompt) {
        throw new Error('runtime.inference.prompt must be set for single-query mode.');
      }
      const result = await runQuery(page, prompt);

      console.log(`\n${'─'.repeat(50)}`);
      console.log(`\x1b[33mOutput:\x1b[0m ${result.output}`);
      console.log(`Tokens: ${result.tokenCount} in ${result.elapsedMs.toFixed(0)}ms`);
      console.log(`Speed:  ${result.tokensPerSec.toFixed(1)} tok/s`);
      console.log(`${'─'.repeat(50)}`);

      console.log('\nKeeping browser open. Press Ctrl+C to exit.');
      await new Promise(() => {});
    }
  } catch (err) {
    console.error('\nTest failed:',  (err).message);
    await context.close();
    process.exit(1);
  }
}

main();
