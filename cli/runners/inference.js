


import { setHarnessConfig, appendRuntimeConfigParams } from '../args/index.js';

export async function runInferenceTest(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('INFERENCE TEST');
  console.log('='.repeat(60));
  console.log(`  Model: ${opts.model}`);

  // Strip 'models/' prefix if present - harness BASE_URL already includes /models
  const modelId = opts.model.replace(/^models\//, '');
  setHarnessConfig(opts, {
    mode: 'inference',
    autorun: true,
    skipLoad: false,
    modelId,
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
        const state =  (window).testState;
        return state && state.done === true;
      },
      { timeout: opts.timeout }
    );

    const testState = await page.evaluate(() =>  (window).testState);
    const duration = Date.now() - startTime;

    const passed = testState.loaded && testState.tokens?.length > 0 && testState.errors?.length === 0;

    if (passed) {
      const output =  (testState.output || '');
      const outputPreview = output.slice(0, 100);
      const outputEscaped = JSON.stringify(outputPreview);
      const outputLabel = output.length === 0
        ? '<empty>'
        : `${outputEscaped}${output.length > 100 ? '...' : ''}`;
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
    console.log(`\n  \x1b[31mFAIL\x1b[0m ${ (err).message}`);

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
          error:  (err).message,
        },
      ],
    };
  }
}
