

/**
 * Inference smoke test.
 */

import { setHarnessConfig, appendRuntimeConfigParams } from '../args/index.js';

/**
 * Run inference test (model load + generate).
 * @param {import('playwright').Page} page
 * @param {import('../args/index.js').CLIOptions} opts
 * @returns {Promise<import('../output.js').SuiteResult>}
 */
export async function runInferenceTest(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('INFERENCE TEST');
  console.log('='.repeat(60));
  console.log(`  Model: ${opts.model}`);

  setHarnessConfig(opts, {
    mode: 'inference',
    autorun: true,
    skipLoad: false,
    modelId: opts.model,
  });
  const testParams = new URLSearchParams();
  appendRuntimeConfigParams(testParams, opts);

  const testUrl = `${opts.baseUrl}/doppler/tests/harness.html?${testParams.toString()}`;
  console.log(`  URL: ${testUrl}`);

  await page.goto(testUrl, { timeout: opts.timeout });

  const startTime = Date.now();

  try {
    await page.waitForFunction(
      () => {
        const state = /** @type {any} */ (window).testState;
        return state && state.done === true;
      },
      { timeout: opts.timeout }
    );

    const testState = await page.evaluate(() => /** @type {any} */ (window).testState);
    const duration = Date.now() - startTime;

    const passed = testState.loaded && testState.tokens?.length > 0 && testState.errors?.length === 0;

    if (passed) {
      const output = /** @type {string} */ (testState.output || '');
      const outputPreview = output.slice(0, 100);
      const outputLabel = outputPreview.trim().length === 0
        ? '<empty>'
        : `${outputPreview}${output.length > 100 ? '...' : ''}`;
      console.log(`\n  \x1b[32mPASS\x1b[0m Model loaded and generated ${testState.tokens?.length || 0} tokens`);
      console.log(`  Output: ${outputLabel}`);
    } else {
      console.log(`\n  \x1b[31mFAIL\x1b[0m`);
      if (!testState.loaded) console.log('    Model failed to load');
      if (testState.errors?.length > 0) {
        for (const err of testState.errors) {
          console.log(`    Error: ${err}`);
        }
      }
    }

    return {
      suite: 'inference',
      passed: passed ? 1 : 0,
      failed: passed ? 0 : 1,
      skipped: 0,
      duration,
      results: [
        {
          name: `inference:${opts.model}`,
          passed,
          duration,
          error: passed ? undefined : (testState.errors?.[0] || 'Unknown error'),
        },
      ],
    };
  } catch (err) {
    const duration = Date.now() - startTime;
    console.log(`\n  \x1b[31mFAIL\x1b[0m ${/** @type {Error} */ (err).message}`);

    return {
      suite: 'inference',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration,
      results: [
        {
          name: `inference:${opts.model}`,
          passed: false,
          duration,
          error: /** @type {Error} */ (err).message,
        },
      ],
    };
  }
}
