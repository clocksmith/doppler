

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [
    ['html', { outputFolder: '../../test-results/correctness-html' }],
    ['json', { outputFile: '../../test-results/correctness-report.json' }],
    ['list'],
  ],
  outputDir: '../../test-results/correctness-output',

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
        headless: false,  // Handle via --headless=new in args (supports real GPU)
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--headless=new',
            // Platform-specific GPU backend
            ...(process.platform === 'darwin'
              ? ['--use-angle=metal']
              : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-vulkan-surface']),
            '--no-first-run',
            '--no-default-browser-check',
          ],
        },
      },
    },
  ],

  webServer: {
    command: 'node ../../serve.js --port 8080',
    url: 'http://localhost:8080/doppler/tests/harness.html',
    reuseExistingServer: !process.env.CI,
    timeout: 10000,
  },
});
