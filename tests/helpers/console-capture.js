/**
 * console-capture.js - Browser Console Capture Utility
 *
 * Captures and analyzes browser console output during e2e tests.
 *
 * @module tests/helpers/console-capture
 */

import {
  BASE_LOG_PATTERNS,
  GOOD_TOKENS,
  BAD_TOKENS,
  analyzeTokens,
} from './test-config.js';

/**
 * Console capture helper
 */
export class ConsoleCapture {
  logs = [];
  errors = [];
  importantPatterns;

  constructor() {
    this.importantPatterns = new Set(BASE_LOG_PATTERNS);
  }

  /**
   * Attach to a Playwright page
   * @param page - Playwright page instance
   * @param options - Attach options
   */
  attach(page, options = {}) {
    const { printImportant = true, printAll = false } = options;

    page.on('console', msg => {
      const entry = {
        type: msg.type(),
        text: msg.text(),
        timestamp: Date.now(),
      };

      this.logs.push(entry);

      // Check if important
      const isImportant = this._isImportant(entry.text);

      if (printAll || (printImportant && isImportant)) {
        console.log(`[${entry.type}] ${entry.text}`);
      }
    });

    page.on('pageerror', err => {
      this.errors.push(err.message);
      console.error('PAGE ERROR:', err.message);
    });
  }

  /**
   * Check if a log message matches important patterns
   * @private
   */
  _isImportant(text) {
    for (const pattern of this.importantPatterns) {
      if (text.includes(pattern)) return true;
    }
    return false;
  }

  /**
   * Add custom important patterns
   * @param patterns - Array of pattern strings
   */
  addImportantPatterns(patterns) {
    patterns.forEach(p => this.importantPatterns.add(p));
  }

  /**
   * Get all log texts
   * @returns Array of log text strings
   */
  getLogTexts() {
    return this.logs.map(l => l.text);
  }

  /**
   * Get logs matching a pattern
   * @param pattern - String or regex pattern to match
   * @returns Array of matching log entries
   */
  filter(pattern) {
    return this.logs.filter(l => {
      if (typeof pattern === 'string') {
        return l.text.includes(pattern);
      }
      return pattern.test(l.text);
    });
  }

  /**
   * Check if any log contains the pattern
   * @param pattern - String or regex pattern to check
   * @returns Whether any log matches
   */
  contains(pattern) {
    return this.filter(pattern).length > 0;
  }

  /**
   * Get logs by type
   * @param type - Log type ('log', 'warn', 'error', 'info', 'debug')
   * @returns Array of log entries of specified type
   */
  byType(type) {
    return this.logs.filter(l => l.type === type);
  }

  /**
   * Get error logs
   * @returns Array of error log entries
   */
  getErrors() {
    return this.logs.filter(l =>
      l.type === 'error' ||
      l.text.toLowerCase().includes('error')
    );
  }

  /**
   * Clear captured logs
   */
  clear() {
    this.logs = [];
    this.errors = [];
  }

  /**
   * Get the last N logs
   * @param n - Number of logs to retrieve
   * @returns Array of the last n log entries
   */
  last(n) {
    return this.logs.slice(-n);
  }

  // ============================================
  // DOPPLER-specific analysis
  // ============================================

  /**
   * Extract logits information from logs
   * @returns Array of logits info objects
   */
  getLogitsInfo() {
    const logitsLogs = this.filter(/logits:|top-5:/);
    return logitsLogs.map(l => {
      const match = l.text.match(/top-5: (.+)$/);
      return {
        raw: l.text,
        tokens: match ? match[1] : null,
      };
    });
  }

  /**
   * Check for generation output
   * @returns Generation output info
   */
  getGenerationOutput() {
    const outputLog = this.logs.find(l =>
      l.text.includes('OUTPUT') ||
      l.text.includes('Output text:') ||
      l.text.includes('Generated:')
    );

    return {
      found: !!outputLog,
      text: outputLog?.text || null,
    };
  }

  /**
   * Analyze token quality for expected vs unexpected output
   * @param options - Token quality options
   * @returns Token quality analysis result
   */
  analyzeTokenQuality(options = {}) {
    const allText = this.logs.map(l => l.text).join(' ');

    // Use centralized token lists from test-config.js
    const result = analyzeTokens(allText, {
      goodTokens: options.goodTokens || GOOD_TOKENS,
      badTokens: options.badTokens || BAD_TOKENS,
    });

    return {
      hasGood: result.hasGood,
      hasBad: result.hasBad,
      details: {
        goodTokensFound: result.goodFound,
        badTokensFound: result.badFound,
      },
    };
  }

  /**
   * Print a summary of captured logs
   */
  printSummary() {
    console.log('\n' + '='.repeat(60));
    console.log('CONSOLE CAPTURE SUMMARY');
    console.log('='.repeat(60));

    console.log(`Total logs: ${this.logs.length}`);
    console.log(`Errors: ${this.errors.length}`);

    // Logits info
    const logitsInfo = this.getLogitsInfo();
    if (logitsInfo.length > 0) {
      console.log('\nLogits/Top-5 samples:');
      logitsInfo.slice(0, 5).forEach(l => console.log('  ', l.tokens || l.raw));
    }

    // Generation output
    const output = this.getGenerationOutput();
    if (output.found) {
      console.log('\nGeneration output:', output.text);
    }

    // Token quality
    const quality = this.analyzeTokenQuality();
    console.log('\nToken quality:');
    console.log('  Good tokens found:', quality.details.goodTokensFound.join(', ') || 'none');
    console.log('  Bad tokens found:', quality.details.badTokensFound.join(', ') || 'none');

    // Errors
    if (this.errors.length > 0) {
      console.log('\nPage errors:');
      this.errors.forEach(e => console.log('  ', e));
    }

    console.log('='.repeat(60) + '\n');
  }
}

export default ConsoleCapture;
