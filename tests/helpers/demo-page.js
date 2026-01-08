/**
 * demo-page.js - Page Object Model for DOPPLER Demo
 *
 * Provides reusable methods for interacting with the demo UI in e2e tests.
 *
 * @module tests/helpers/demo-page
 */

import { URLS } from './test-config.js';

/**
 * Page Object Model for DOPPLER demo
 */
export class DemoPage {
  page;
  baseUrl;

  /**
   * @param page - Playwright page
   * @param options - Demo page options
   */
  constructor(page, options = {}) {
    this.page = page;
    this.baseUrl = options.baseUrl || URLS.demo;
  }

  // ============================================
  // Navigation
  // ============================================

  /**
   * Navigate to demo page
   * @param options - Navigation options
   */
  async goto(options = {}) {
    const timeout = options.timeout || 30000;
    await this.page.goto(this.baseUrl, {
      waitUntil: 'networkidle',
      timeout,
    });
    await this.page.waitForSelector('#model-list', { timeout: 10000 });
  }

  // ============================================
  // Model Selection & Loading
  // ============================================

  /**
   * Select a model by name pattern
   * @param namePattern - Model name to match (e.g., 'gemma 1b')
   * @returns Whether model was found and clicked
   */
  async selectModel(namePattern) {
    // Wait for model list to populate
    await this.page.waitForTimeout(2000);

    const pattern = typeof namePattern === 'string'
      ? namePattern.toLowerCase()
      : namePattern;

    // Look for model items
    const allElements = await this.page.locator('#model-list *').all();

    for (const elem of allElements) {
      const text = await elem.textContent().catch(() => '');
      const tagName = await elem.evaluate(el => el.tagName).catch(() => '');

      const matches = typeof pattern === 'string'
        ? text.toLowerCase().includes(pattern)
        : pattern.test(text);

      if (matches && ['BUTTON', 'A', 'DIV'].includes(tagName)) {
        try {
          await elem.click({ timeout: 2000 });
          return true;
        } catch {
          // Element not clickable, continue
        }
      }
    }

    // Fallback: try text pattern directly
    const textLocator = this.page.locator(`text=/${namePattern}/i`).first();
    if (await textLocator.isVisible({ timeout: 2000 }).catch(() => false)) {
      await textLocator.click();
      return true;
    }

    return false;
  }

  /**
   * Wait for model to finish loading
   * @param options - Load options
   */
  async waitForModelLoad(options = {}) {
    const timeout = options.timeout || 90000;

    await this.page.waitForFunction(() => {
      const textarea = document.querySelector('#chat-input');
      return textarea && !textarea.disabled;
    }, { timeout });
  }

  /**
   * Get list of available model names
   * @returns Array of model names
   */
  async getAvailableModels() {
    const content = await this.page.locator('#model-list').textContent();
    return content.split('\n').filter(s => s.trim());
  }

  // ============================================
  // Chat / Generation
  // ============================================

  /**
   * Send a prompt and wait for generation to start
   * @param prompt - Text prompt
   */
  async sendPrompt(prompt) {
    const textarea = this.page.locator('#chat-input');
    await textarea.fill(prompt);

    const sendBtn = this.page.locator('#send-btn');
    if (await sendBtn.isEnabled({ timeout: 1000 }).catch(() => false)) {
      await sendBtn.click();
    } else {
      await textarea.press('Enter');
    }
  }

  /**
   * Wait for generation to complete
   * @param options - Generation options
   */
  async waitForGeneration(options = {}) {
    const timeout = options.timeout || 30000;
    const logs = options.logs || [];

    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      await this.page.waitForTimeout(1000);

      // Check for generation completion in logs
      const hasOutput = logs.some(l =>
        l.includes('OUTPUT') ||
        l.includes('Generated') ||
        l.includes('generation complete')
      );

      if (hasOutput) {
        // Give it a moment to finish
        await this.page.waitForTimeout(2000);
        return;
      }
    }
  }

  /**
   * Get the last assistant response from the chat
   * @returns Response text
   */
  async getLastResponse() {
    const responseElement = this.page.locator(
      '.assistant-message, .response, .output, .message-content'
    ).last();

    if (await responseElement.isVisible({ timeout: 5000 }).catch(() => false)) {
      return await responseElement.textContent() || '';
    }

    return '';
  }

  /**
   * Clear the conversation
   */
  async clearConversation() {
    const clearBtn = this.page.locator('#clear-btn');
    if (await clearBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      await clearBtn.click();
    }
  }

  // ============================================
  // Model Conversion
  // ============================================

  /**
   * Click the convert button to open file picker
   */
  async clickConvert() {
    const convertBtn = this.page.locator('#convert-btn');
    await convertBtn.click();
  }

  /**
   * Wait for conversion to complete
   * @param options - Conversion options
   */
  async waitForConversion(options = {}) {
    const timeout = options.timeout || 300000;

    // Wait for progress bar to show completion or status to show "Done"
    await this.page.waitForFunction(() => {
      const progress = document.querySelector('#convert-progress');
      const message = document.querySelector('#convert-message');

      if (message?.textContent?.toLowerCase().includes('done')) return true;
      if (message?.textContent?.toLowerCase().includes('error')) return 'error';
      if (progress?.style.width === '100%') return true;

      return false;
    }, { timeout });

    // Check if it was an error
    const message = await this.page.locator('#convert-message').textContent() || '';
    if (message.toLowerCase().includes('error')) {
      throw new Error(`Conversion failed: ${message}`);
    }
  }

  /**
   * Get conversion progress message
   * @returns Conversion status text
   */
  async getConversionStatus() {
    return await this.page.locator('#convert-message').textContent() || '';
  }

  // ============================================
  // Status & Info
  // ============================================

  /**
   * Get current status text
   * @returns Status text
   */
  async getStatus() {
    return await this.page.locator('.status-text').textContent() || '';
  }

  /**
   * Check if chat input is enabled (model loaded)
   * @returns Whether chat is enabled
   */
  async isChatEnabled() {
    const textarea = this.page.locator('#chat-input');
    return !(await textarea.isDisabled());
  }

  /**
   * Get performance stats from the UI
   * @returns Performance statistics
   */
  async getStats() {
    return {
      tps: await this.page.locator('#stat-tps').textContent().catch(() => '--') || '--',
      memory: await this.page.locator('#stat-memory').textContent().catch(() => '--') || '--',
      gpu: await this.page.locator('#stat-gpu').textContent().catch(() => '--') || '--',
      kv: await this.page.locator('#stat-kv').textContent().catch(() => '--') || '--',
    };
  }
}

export default DemoPage;
