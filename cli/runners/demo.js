

/**
 * Demo UI test.
 */

import { appendRuntimeConfigParams } from '../args/index.js';

/** Good token patterns for quality analysis */
const GOOD_TOKENS = ['blue', 'sky', 'the', 'is', 'clear', 'clouds', 'sun', 'day', 'night', 'color'];

/** Bad token patterns (garbage output) */
const BAD_TOKENS = ['<unk>', '####', '\u0000', '\uFFFD'];

/**
 * Run demo UI test.
 * @param {import('playwright').Page} page
 * @param {import('../args/index.js').CLIOptions} opts
 * @returns {Promise<import('../output.js').SuiteResult>}
 */
export async function runDemoTest(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('DEMO UI TEST');
  console.log('='.repeat(60));
  console.log(`  Model: ${opts.model}`);

  const prompt = opts.runtimeConfig?.inference?.prompt;
  if (!prompt) {
    throw new Error('runtime.inference.prompt must be set for demo tests.');
  }
  console.log(`  Prompt: "${prompt}"`);
  const followupPrompt = 'and at night?';

  const startTime = Date.now();
  /** @type {string[]} */
  const errors = [];
  /** @type {string[]} */
  const logs = [];

  // Setup console capture
  page.on('console', (msg) => {
    const text = msg.text();
    logs.push(text);
    console.log(`  [browser] ${text}`);
  });

  page.on('pageerror', (err) => {
    errors.push(err.message);
    console.error(`  [browser error] ${err.message}`);
  });

  try {
    // Navigate to demo
    console.log('\n  Step 1: Opening demo page...');
    const demoParams = new URLSearchParams();
    appendRuntimeConfigParams(demoParams, opts);
    const demoUrl = `${opts.baseUrl}/d${demoParams.toString() ? `?${demoParams.toString()}` : ''}`;
    await page.goto(demoUrl, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Wait for model list
    console.log('  Step 2: Waiting for model list...');
    await page.waitForSelector('#model-list', { timeout: 10000 }).catch(() => {
      console.log('    (model-list selector not found, trying alternative)');
    });

    // Select model matching pattern
    console.log(`  Step 3: Selecting model matching "${opts.model}"...`);
    const modelSelected = await page.evaluate(async (modelPattern) => {
      const modelList = document.querySelector('#model-list');
      if (!modelList) return false;

      const buttons = modelList.querySelectorAll('button, a, div');
      for (const btn of buttons) {
        const text = btn.textContent?.toLowerCase() || '';
        if (text.includes(modelPattern.toLowerCase())) {
          /** @type {HTMLElement} */ (btn).click();
          return true;
        }
      }
      return false;
    }, opts.model);

    if (!modelSelected) {
      throw new Error(`Model "${opts.model}" not found in model list`);
    }

    // Wait for model to load
    console.log('  Step 4: Waiting for model to load...');
    await page.waitForFunction(
      () => {
        const textarea = /** @type {HTMLTextAreaElement} */ (document.querySelector('#chat-input'));
        return textarea && !textarea.disabled;
      },
      { timeout: 90000 }
    );

    console.log('  Model loaded successfully!');

    // Send prompt
    console.log(`  Step 5: Sending prompt: "${prompt}"...`);
    await page.fill('#chat-input', prompt);
    await page.click('#send-btn').catch(async () => {
      await page.press('#chat-input', 'Enter');
    });

    // Wait for generation (check logs for output)
    console.log('  Step 6: Waiting for generation...');
    const generationTimeout = 30000;
    const genStartTime = Date.now();

    while (Date.now() - genStartTime < generationTimeout) {
      await page.waitForTimeout(1000);

      const hasOutput = logs.some(l =>
        l.includes('OUTPUT') ||
        l.includes('Generated') ||
        l.includes('generation complete') ||
        l.includes('[Pipeline] Decode complete')
      );

      if (hasOutput) {
        console.log('  Generation complete!');
        await page.waitForTimeout(1000);
        break;
      }
    }

    // Capture message counts after first response
    const firstTurnCounts = await page.evaluate(() => ({
      users: document.querySelectorAll('.message.user').length,
      assistants: document.querySelectorAll('.message.assistant').length,
    }));

    // Send follow-up prompt to validate multi-turn chat flow
    console.log(`  Step 7: Sending follow-up prompt: "${followupPrompt}"...`);
    await page.fill('#chat-input', followupPrompt);
    await page.click('#send-btn').catch(async () => {
      await page.press('#chat-input', 'Enter');
    });

    console.log('  Step 8: Waiting for follow-up generation...');
    await page.waitForFunction(
      (counts) => {
        const users = document.querySelectorAll('.message.user').length;
        const assistants = document.querySelectorAll('.message.assistant').length;
        const cursor = document.querySelector('.message.assistant .cursor');
        return users >= counts.users + 1 && assistants >= counts.assistants + 1 && !cursor;
      },
      { timeout: generationTimeout },
      firstTurnCounts
    );

    const roleSequence = await page.evaluate(() =>
      Array.from(document.querySelectorAll('.message-role'))
        .map((el) => el.textContent?.trim().toLowerCase())
        .filter(Boolean)
    );
    const rolesOk = roleSequence.length >= 4
      && roleSequence[0] === 'user'
      && roleSequence[1] === 'assistant'
      && roleSequence[2] === 'user'
      && roleSequence[3] === 'assistant';

    const statsOk = await page.evaluate(() => {
      const stats = Array.from(document.querySelectorAll('.message.assistant .message-stats'));
      return stats.some((el) => /\d+ tokens/.test(el.textContent || ''));
    });

    // Analyze token quality
    const allText = logs.join(' ');
    const goodFound = GOOD_TOKENS.filter(t => allText.toLowerCase().includes(t.toLowerCase()));
    const badFound = BAD_TOKENS.filter(t => allText.includes(t));

    const hasGood = goodFound.length > 0;
    const hasBad = badFound.length > 0;
    const passed = hasGood && !hasBad && errors.length === 0 && rolesOk && statsOk;

    // Print summary
    console.log('\n  ' + '-'.repeat(50));
    console.log('  Token Quality Analysis:');
    console.log(`    Good tokens found: ${goodFound.join(', ') || 'none'}`);
    console.log(`    Bad tokens found: ${badFound.join(', ') || 'none'}`);
    console.log(`    Errors: ${errors.length}`);
    console.log(`    Multi-turn roles: ${rolesOk ? 'ok' : 'invalid'}`);
    console.log(`    Message stats: ${statsOk ? 'present' : 'missing'}`);

    const duration = Date.now() - startTime;

    if (passed) {
      console.log(`\n  \x1b[32mPASS\x1b[0m Demo test completed successfully (${(duration / 1000).toFixed(1)}s)`);
    } else {
      console.log(`\n  \x1b[31mFAIL\x1b[0m Demo test failed`);
      if (!hasGood) console.log('    - No coherent tokens detected');
      if (hasBad) console.log(`    - Garbage tokens found: ${badFound.join(', ')}`);
      if (errors.length > 0) console.log(`    - Page errors: ${errors.join(', ')}`);
      if (!rolesOk) console.log('    - Message roles did not alternate user/assistant');
      if (!statsOk) console.log('    - Assistant token stats missing');
    }

    return {
      suite: 'demo',
      passed: passed ? 1 : 0,
      failed: passed ? 0 : 1,
      skipped: 0,
      duration,
      results: [
        {
          name: `demo:${opts.model}`,
          passed,
          duration,
          error: passed ? undefined : (errors[0] || (hasBad ? `Bad tokens: ${badFound.join(', ')}` : 'No coherent output')),
        },
      ],
    };
  } catch (err) {
    const duration = Date.now() - startTime;
    console.log(`\n  \x1b[31mFAIL\x1b[0m ${/** @type {Error} */ (err).message}`);

    return {
      suite: 'demo',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration,
      results: [
        {
          name: `demo:${opts.model}`,
          passed: false,
          duration,
          error: /** @type {Error} */ (err).message,
        },
      ],
    };
  }
}
