/**
 * Playwright config for pipeline-level correctness tests
 *
 * Run with: npx playwright test -c tests/correctness/playwright.config.ts
 */

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [
    ['html', { outputFolder: '../../results/correctness-html' }],
    ['json', { outputFile: '../../results/correctness-report.json' }],
    ['list'],
  ],

  timeout: 180000, // 3 minutes for model loading + generation
  expect: {
    timeout: 30000,
  },

  use: {
    baseURL: 'http://localhost:8080',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        headless: false,
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-angle=vulkan',
            '--no-activate',
            '--silent-launch',
            '--no-first-run',
            '--no-default-browser-check',
            '--window-position=-9999,-9999',
            '--window-size=50,50',
          ],
        },
      },
    },
  ],

  webServer: {
    command: 'python3 -m http.server 8080 --directory ../..',
    url: 'http://localhost:8080',
    reuseExistingServer: !process.env.CI,
    timeout: 10000,
  },
});
