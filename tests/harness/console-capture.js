

import {
  BASE_LOG_PATTERNS,
  GOOD_TOKENS,
  BAD_TOKENS,
  analyzeTokens,
} from './test-config.js';


export class ConsoleCapture {
  logs = [];
  errors = [];
  importantPatterns;

  constructor() {
    this.importantPatterns = new Set(BASE_LOG_PATTERNS);
  }

  
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

  
  _isImportant(text) {
    for (const pattern of this.importantPatterns) {
      if (text.includes(pattern)) return true;
    }
    return false;
  }

  
  addImportantPatterns(patterns) {
    patterns.forEach(p => this.importantPatterns.add(p));
  }

  
  getLogTexts() {
    return this.logs.map(l => l.text);
  }

  
  filter(pattern) {
    return this.logs.filter(l => {
      if (typeof pattern === 'string') {
        return l.text.includes(pattern);
      }
      return pattern.test(l.text);
    });
  }

  
  contains(pattern) {
    return this.filter(pattern).length > 0;
  }

  
  byType(type) {
    return this.logs.filter(l => l.type === type);
  }

  
  getErrors() {
    return this.logs.filter(l =>
      l.type === 'error' ||
      l.text.toLowerCase().includes('error')
    );
  }

  
  clear() {
    this.logs = [];
    this.errors = [];
  }

  
  last(n) {
    return this.logs.slice(-n);
  }

  // ============================================
  // DOPPLER-specific analysis
  // ============================================

  
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
