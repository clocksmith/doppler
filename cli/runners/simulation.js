
import { appendRuntimeConfigParams } from '../args/index.js';

export async function runSimulationTest(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('SIMULATION TEST');
  console.log('='.repeat(60));

  const harness = opts.runtimeConfig?.shared?.harness;
  if (!harness) {
    throw new Error('runtime.shared.harness is required for simulation.');
  }
  if (harness.mode !== 'simulation') {
    throw new Error('Simulation requires runtime.shared.harness.mode="simulation".');
  }
  if (!opts.runtimeConfig?.emulation?.enabled) {
    throw new Error('Simulation requires runtime.emulation.enabled=true.');
  }

  const params = new URLSearchParams();
  appendRuntimeConfigParams(params, opts);
  const testUrl = `${opts.baseUrl}/doppler/tests/harness.html?${params.toString()}`;
  console.log(`  URL: ${testUrl}`);

  await page.goto(testUrl, { timeout: opts.timeout });

  const startTime = Date.now();

  try {
    await page.waitForFunction(
      () => {
        const state = window.testState;
        return state && state.done === true;
      },
      { timeout: opts.timeout }
    );

    const testState = await page.evaluate(() => window.testState);
    const duration = Date.now() - startTime;

    const passed = Boolean(testState) && (testState.errors?.length || 0) === 0;

    if (passed) {
      console.log('\n  \x1b[32mPASS\x1b[0m Simulation context initialized');
    } else {
      console.log('\n  \x1b[31mFAIL\x1b[0m');
      if (testState?.errors?.length > 0) {
        for (const err of testState.errors) {
          console.log(`    Error: ${err}`);
        }
      }
    }

    return {
      suite: 'simulation',
      passed: passed ? 1 : 0,
      failed: passed ? 0 : 1,
      skipped: 0,
      duration,
      results: [
        {
          name: 'simulation',
          passed,
          duration,
          error: passed ? undefined : (testState?.errors?.[0] || 'Unknown error'),
        },
      ],
    };
  } catch (err) {
    const duration = Date.now() - startTime;
    const errMessage = err && err.message ? err.message : String(err);
    console.log(`\n  \x1b[31mFAIL\x1b[0m ${errMessage}`);

    return {
      suite: 'simulation',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration,
      results: [
        {
          name: 'simulation',
          passed: false,
          duration,
          error: errMessage,
        },
      ],
    };
  }
}
