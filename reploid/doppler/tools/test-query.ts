#!/usr/bin/env node
/**
 * Quick test query runner - minimal Playwright script for ad-hoc inference testing
 *
 * Usage:
 *   npx tsx tools/test-query.ts "the color of the sky is "
 *   npx tsx tools/test-query.ts --model gemma-3-1b-it-q4 "hello world"
 *   npx tsx tools/test-query.ts --repl  # Interactive mode with cached model
 */

import { chromium, type BrowserContext, type Page } from 'playwright';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import * as readline from 'readline';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface Options {
  prompt: string;
  model: string;
  baseUrl: string;
  maxTokens: number;
  timeout: number;
  repl: boolean;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    prompt: 'the color of the sky is ',
    model: 'gemma-3-1b-it-q4',
    baseUrl: 'http://localhost:8080/d',
    maxTokens: 32,
    timeout: 120000,
    repl: false,
  };

  const tokens = [...argv];
  while (tokens.length) {
    const arg = tokens.shift()!;
    switch (arg) {
      case '--model':
      case '-m':
        opts.model = tokens.shift() || opts.model;
        break;
      case '--base-url':
      case '-u':
        opts.baseUrl = tokens.shift() || opts.baseUrl;
        break;
      case '--max-tokens':
      case '-t':
        opts.maxTokens = parseInt(tokens.shift() || '32', 10);
        break;
      case '--timeout':
        opts.timeout = parseInt(tokens.shift() || '120000', 10);
        break;
      case '--repl':
      case '-i':
        opts.repl = true;
        break;
      case '--help':
      case '-h':
        console.log(`
Quick test query for DOPPLER

Usage:
  npx tsx tools/test-query.ts [options] "prompt text"
  npx tsx tools/test-query.ts --repl   # Interactive REPL mode

Options:
  --model, -m <name>   Model name (default: gemma-3-1b-it-q4)
  --base-url, -u <url> Server URL (default: http://localhost:8080/d)
  --max-tokens, -t <n> Max tokens (default: 32)
  --timeout <ms>       Timeout (default: 120000)
  --repl, -i           Interactive REPL mode (model cached in memory)
  --help, -h           Show this help

REPL Commands:
  <text>               Run inference with prompt
  /clear               Clear KV cache (new conversation)
  /reload              Reload model from scratch
  /tokens <n>          Set max tokens
  /quit                Exit

Examples:
  npx tsx tools/test-query.ts "the color of the sky is "
  npx tsx tools/test-query.ts --repl
  npx tsx tools/test-query.ts -m gemma-3-1b-it-q4 -t 64 "once upon a time"
`);
        process.exit(0);
      default:
        if (!arg.startsWith('-')) {
          opts.prompt = arg;
        }
    }
  }

  return opts;
}

async function launchBrowser(): Promise<{ context: BrowserContext; page: Page }> {
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

async function loadModel(page: Page, baseUrl: string, model: string): Promise<void> {
  // Use the DOPPLERDemo app that's already loaded on the page
  // This avoids module loading issues and reuses the existing infrastructure

  await page.evaluate(
    async ({ model }) => {
      const w = window as Window & {
        dopplerDemo?: {
          selectModel: (modelOrKey: unknown) => Promise<void>;
          getStatus: () => { model: string | null };
          modelSelector?: {
            getModels?: () => Array<{ key: string; name: string; sources?: { server?: { id: string } } }>;
          };
        };
      };

      // Wait for the app to be ready
      let retries = 0;
      while (!w.dopplerDemo && retries < 50) {
        await new Promise(r => setTimeout(r, 100));
        retries++;
      }

      if (!w.dopplerDemo) {
        throw new Error('DOPPLERDemo app not found on page');
      }

      console.log(`[Test] Looking for model matching: ${model}`);

      // Find the first available model that matches the search term
      // The model selector stores the registry, but it's private
      // Let's just click the first model button in the UI instead
      const modelBtns = document.querySelectorAll('[data-model-key]');
      let targetBtn: HTMLElement | null = null;

      for (const btn of modelBtns) {
        const key = (btn as HTMLElement).dataset.modelKey || '';
        const text = btn.textContent?.toLowerCase() || '';
        if (key.includes(model.toLowerCase()) || text.includes(model.toLowerCase())) {
          targetBtn = btn as HTMLElement;
          break;
        }
      }

      // If no match, just use the first available model
      if (!targetBtn && modelBtns.length > 0) {
        targetBtn = modelBtns[0] as HTMLElement;
        console.log(`[Test] No exact match, using first model`);
      }

      if (!targetBtn) {
        throw new Error('No models available. Is the server running?');
      }

      const modelKey = targetBtn.dataset.modelKey;
      console.log(`[Test] Selecting model: ${modelKey}`);

      // Click the Run button for this model
      const runBtn = targetBtn.querySelector('.model-run-btn') as HTMLButtonElement | null;
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
        const status = w.dopplerDemo!.getStatus();
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

async function runQuery(page: Page, prompt: string, maxTokens: number): Promise<{ output: string; tokenCount: number; elapsedMs: number; tokensPerSec: number }> {
  return await page.evaluate(
    async ({ prompt, maxTokens }) => {
      const w = window as Window & {
        dopplerDemo?: {
          pipeline?: {
            generate: (p: string, o: unknown) => AsyncGenerator<string>;
          };
        };
      };

      if (!w.dopplerDemo?.pipeline) {
        throw new Error('Model not loaded. Load a model first.');
      }

      const pipeline = w.dopplerDemo.pipeline;
      const tokens: string[] = [];
      const start = performance.now();

      console.log(`[Test] Generating: "${prompt.slice(0, 50)}${prompt.length > 50 ? '...' : ''}"`);

      for await (const token of pipeline.generate(prompt, {
        maxTokens,
        temperature: 0,
        topK: 1,
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
    { prompt, maxTokens }
  );
}

async function clearKVCache(page: Page): Promise<void> {
  await page.evaluate(() => {
    const w = window as Window & {
      dopplerDemo?: {
        pipeline?: { clearKVCache?: () => void };
        clearConversation?: () => void;
      };
    };
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

async function runRepl(page: Page, opts: Options): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  let maxTokens = opts.maxTokens;

  console.log('\n\x1b[36mDOPPLER REPL\x1b[0m - Type prompts or commands');
  console.log('Commands: /clear /reload /tokens <n> /quit\n');

  const prompt = (): void => {
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
        } else if (line.startsWith('/tokens ') || line.startsWith('/t ')) {
          const n = parseInt(line.split(' ')[1], 10);
          if (n > 0) {
            maxTokens = n;
            console.log(`Max tokens set to ${maxTokens}\n`);
          }
        } else if (line.startsWith('/')) {
          console.log('Unknown command. Try: /clear /reload /tokens <n> /quit\n');
        } else {
          // Run inference
          const result = await runQuery(page, line, maxTokens);
          console.log(`\n\x1b[33m${result.output}\x1b[0m`);
          console.log(`\x1b[90m[${result.tokenCount} tokens, ${result.elapsedMs.toFixed(0)}ms, ${result.tokensPerSec.toFixed(1)} tok/s]\x1b[0m\n`);
        }
      } catch (err) {
        console.error(`\x1b[31mError: ${(err as Error).message}\x1b[0m\n`);
      }

      prompt();
    });
  };

  prompt();
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));

  console.log(`\n\x1b[36mDOPPLER Test Query\x1b[0m`);
  console.log(`${'─'.repeat(50)}`);
  console.log(`Model:      ${opts.model}`);
  console.log(`Max tokens: ${opts.maxTokens}`);
  console.log(`Mode:       ${opts.repl ? 'REPL (interactive)' : 'single query'}`);
  console.log(`${'─'.repeat(50)}\n`);

  console.log('Launching browser...');
  const { context, page } = await launchBrowser();

  try {
    await page.goto(opts.baseUrl, { timeout: 30000 });
    console.log('Waiting for WebGPU...');
    await page.waitForFunction(() => 'gpu' in navigator, { timeout: 10000 });

    console.log('Loading model...');
    await loadModel(page, opts.baseUrl, opts.model);

    if (opts.repl) {
      // Interactive REPL mode
      await runRepl(page, opts);
    } else {
      // Single query mode
      console.log('\nRunning inference...');
      const result = await runQuery(page, opts.prompt, opts.maxTokens);

      console.log(`\n${'─'.repeat(50)}`);
      console.log(`\x1b[33mOutput:\x1b[0m ${result.output}`);
      console.log(`Tokens: ${result.tokenCount} in ${result.elapsedMs.toFixed(0)}ms`);
      console.log(`Speed:  ${result.tokensPerSec.toFixed(1)} tok/s`);
      console.log(`${'─'.repeat(50)}`);

      console.log('\nKeeping browser open. Press Ctrl+C to exit.');
      await new Promise(() => {});
    }
  } catch (err) {
    console.error('\nTest failed:', (err as Error).message);
    await context.close();
    process.exit(1);
  }
}

main();
