/**
 * Playwright config for kernel correctness tests
 *
 * Run with: npx playwright test -c tests/kernels/correctness/playwright.config.js
 */

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [
    ['list'],
  ],

  timeout: 60000, // 1 minute per test
  expect: {
    timeout: 10000,
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
        headless: true,
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-angle=metal',
            '--headless=new',
            '--no-first-run',
            '--no-default-browser-check',
          ],
        },
      },
    },
  ],

  webServer: {
    command: 'python3 -m http.server 8080 --directory ../../..',
    url: 'http://localhost:8080',
    reuseExistingServer: !process.env.CI,
    timeout: 10000,
  },
});
