


import { resolve } from 'path';
import { writeFile } from 'fs/promises';

import { setHarnessConfig, appendRuntimeConfigParams } from '../args/index.js';

export async function runTrainingTests(page, opts, tests) {
  console.log('\n' + '='.repeat(60));
  console.log('TRAINING CORRECTNESS TESTS');
  console.log('='.repeat(60));

  setHarnessConfig(opts, {
    mode: 'training',
    autorun: false,
    skipLoad: false,
    modelId: null,
  });
  const harnessParams = new URLSearchParams();
  appendRuntimeConfigParams(harnessParams, opts);
  await page.goto(`${opts.baseUrl}/doppler/tests/harness.html?${harnessParams.toString()}`, {
    timeout: opts.timeout,
  });

  await page.waitForTimeout(500);

  await page.waitForFunction(
    () => {
      const w =  (window);
      return Boolean(w.trainingHarness);
    },
    { timeout: 30000 }
  );

    const results = [];
  const startTime = Date.now();

  const testsToRun = opts.filter
    ? tests.filter((t) => t.includes(opts.filter))
    : tests;

  for (const testName of testsToRun) {
    console.log(`\n  Running: ${testName}...`);
    const testStart = Date.now();

    try {
      const result = await page.evaluate(
        async (name) => {
          const harness =  (window).trainingHarness;
          if (harness?.getGPU) {
            await harness.getGPU();
          }
          const testResult = await harness.runTest(name);
          return testResult;
        },
        testName
      );

      const duration = Date.now() - testStart;
      results.push({
        name: testName,
        passed: result.passed,
        duration,
        error: result.error,
      });

      const status = result.passed ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
      console.log(`  ${status} ${testName} (${duration}ms)`);
      if (result.error) {
        console.log(`    Error: ${result.error}`);
      }
    } catch (err) {
      const duration = Date.now() - testStart;
      results.push({
        name: testName,
        passed: false,
        duration,
        error: err?.message || String(err),
      });
      console.log(`  \x1b[31mFAIL\x1b[0m ${testName} (${duration}ms)`);
    }
  }

  const duration = Date.now() - startTime;
  const passed = results.filter(r => r.passed).length;
  const failed = results.length - passed;

  if (opts.output && results.length > 0) {
    const outputPath = resolve(opts.output);
    await writeFile(outputPath, JSON.stringify({ suite: 'training', results }, null, 2));
    console.log(`\nResults written to ${outputPath}`);
  }

  if (await page.evaluate(() =>  (window).renderResults)) {
    await page.evaluate((res) =>  (window).renderResults(res), results);
  }

  return {
    suite: 'training',
    passed,
    failed,
    skipped: tests.length - testsToRun.length,
    duration,
    results,
  };
}
