

/**
 * CLI output formatting.
 */

/**
 * @typedef {Object} SuiteResult
 * @property {string} suite
 * @property {number} passed
 * @property {number} failed
 * @property {number} skipped
 * @property {number} duration
 * @property {Array<{name: string, passed: boolean, duration: number, error?: string}>} results
 */

/**
 * Print summary of all test suites.
 * @param {SuiteResult[]} suites
 */
export function printSummary(suites) {
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));

  let totalPassed = 0;
  let totalFailed = 0;
  let totalSkipped = 0;

  for (const suite of suites) {
    totalPassed += suite.passed;
    totalFailed += suite.failed;
    totalSkipped += suite.skipped;

    const status = suite.failed === 0 ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
    console.log(
      `  ${status} ${suite.suite}: ${suite.passed} passed, ${suite.failed} failed` +
      (suite.skipped > 0 ? `, ${suite.skipped} skipped` : '') +
      ` (${(suite.duration / 1000).toFixed(1)}s)`
    );
  }

  console.log('');
  console.log(`Total: ${totalPassed} passed, ${totalFailed} failed, ${totalSkipped} skipped`);

  if (totalFailed > 0) {
    console.log('\n\x1b[31mTests failed!\x1b[0m');
  } else {
    console.log('\n\x1b[32mAll tests passed!\x1b[0m');
  }
}
