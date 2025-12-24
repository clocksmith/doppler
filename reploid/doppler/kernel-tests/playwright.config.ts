import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [
    ['html', { outputFolder: 'results/html' }],
    ['json', { outputFile: 'results/report.json' }],
    ['list'],
  ],

  timeout: 120000,
  expect: {
    timeout: 30000,
  },

  use: {
    baseURL: 'http://localhost:8080/doppler/kernel-tests/browser',
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
            // Prevent focus stealing
            '--no-activate',
            '--silent-launch',
            '--no-first-run',
            '--no-default-browser-check',
            // Tiny window off-screen
            '--window-position=-9999,-9999',
            '--window-size=50,50',
          ],
        },
      },
    },
  ],

  webServer: {
    command: 'python3 -m http.server 8080 --directory ..',
    url: 'http://localhost:8080',
    reuseExistingServer: !process.env.CI,
    timeout: 10000,
  },
});
