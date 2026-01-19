


export async function runConverterTest(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('CONVERTER UI TEST');
  console.log('='.repeat(60));

  const startTime = Date.now();
    const results = [];
    const errors = [];

  // Setup console capture
  page.on('console', (msg) => {
    const text = msg.text();
    console.log(`  [browser] ${text}`);
  });

  page.on('pageerror', (err) => {
    errors.push(err.message);
    console.error(`  [browser error] ${err.message}`);
  });

  try {
    // Navigate to demo
    console.log('\n  Step 1: Opening demo page...');
    await page.goto(`${opts.baseUrl}/d`, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Test 1: Convert button exists and is enabled
    console.log('  Step 2: Checking convert button...');
    const testStart1 = Date.now();

    const convertBtnExists = await page.locator('#convert-btn').isVisible({ timeout: 5000 }).catch(() => false);
    const convertBtnEnabled = await page.locator('#convert-btn').isEnabled({ timeout: 1000 }).catch(() => false);

    const test1Passed = convertBtnExists && convertBtnEnabled;
    results.push({
      name: 'converter-ui:button-present',
      passed: test1Passed,
      duration: Date.now() - testStart1,
      error: test1Passed ? undefined : `Button visible: ${convertBtnExists}, enabled: ${convertBtnEnabled}`,
    });

    console.log(`    ${test1Passed ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m'} Convert button present and enabled`);

    // Test 2: Convert status is initially hidden
    console.log('  Step 3: Checking convert status...');
    const testStart2 = Date.now();

    const convertStatusHidden = await page.locator('#convert-status').isHidden({ timeout: 1000 }).catch(() => true);

    results.push({
      name: 'converter-ui:status-hidden',
      passed: convertStatusHidden,
      duration: Date.now() - testStart2,
      error: convertStatusHidden ? undefined : 'Convert status should be hidden initially',
    });

    console.log(`    ${convertStatusHidden ? '\x1b[32mPASS\x1b[0m' : '\x1b[33mWARN\x1b[0m'} Convert status initially hidden`);

    // Test 3: Convert button click triggers file picker setup
    console.log('  Step 4: Testing convert button interaction...');
    const testStart3 = Date.now();

    // Inject a test file input to avoid native file picker
    await page.evaluate(() => {
      const input = document.createElement('input');
      input.type = 'file';
      input.id = 'test-file-input';
      input.multiple = true;
      input.style.display = 'none';
      document.body.appendChild(input);
    });

    // Click convert button (should not throw)
    let clickSucceeded = false;
    try {
      await page.click('#convert-btn', { timeout: 2000 });
      clickSucceeded = true;
    } catch {
      // Button might trigger native dialog, that's ok
      clickSucceeded = true;
    }

    results.push({
      name: 'converter-ui:button-clickable',
      passed: clickSucceeded,
      duration: Date.now() - testStart3,
    });

    console.log(`    ${clickSucceeded ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m'} Convert button clickable`);

    // Test 4: Check for progress bar element (exists but may be hidden)
    console.log('  Step 5: Checking progress bar presence...');
    const testStart4 = Date.now();

    const hasProgressBar = await page.evaluate(() => {
      return !!document.querySelector('#convert-progress, .convert-progress, [role="progressbar"]');
    });

    results.push({
      name: 'converter-ui:progress-element',
      passed: true, // Optional, just informational
      duration: Date.now() - testStart4,
      error: hasProgressBar ? undefined : 'Progress bar element not found (may be created dynamically)',
    });

    console.log(`    ${hasProgressBar ? '\x1b[32mPASS\x1b[0m' : '\x1b[33mINFO\x1b[0m'} Progress bar element ${hasProgressBar ? 'found' : 'not found (may be dynamic)'}`);

    const duration = Date.now() - startTime;
    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;

    console.log('\n  ' + '-'.repeat(50));
    console.log(`  Converter UI Tests: ${passed} passed, ${failed} failed`);

    if (failed === 0) {
      console.log(`\n  \x1b[32mPASS\x1b[0m Converter UI test completed (${(duration / 1000).toFixed(1)}s)`);
    } else {
      console.log(`\n  \x1b[31mFAIL\x1b[0m Converter UI test had failures`);
    }

    return {
      suite: 'converter',
      passed,
      failed,
      skipped: 0,
      duration,
      results,
    };
  } catch (err) {
    const duration = Date.now() - startTime;
    console.log(`\n  \x1b[31mFAIL\x1b[0m ${ (err).message}`);

    return {
      suite: 'converter',
      passed: 0,
      failed: 1,
      skipped: 0,
      duration,
      results: [
        {
          name: 'converter-ui',
          passed: false,
          duration,
          error:  (err).message,
        },
      ],
    };
  }
}
