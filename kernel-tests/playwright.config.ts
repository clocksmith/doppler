import { defineConfig, devices } from '@playwright/test';

const isCI = !!process.env.CI;
const parseBool = (value: string | undefined, fallback: boolean) => {
  if (value === undefined) return fallback;
  return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
};
const channelEnv = process.env.PLAYWRIGHT_CHANNEL;
const resolvedChannel = channelEnv === 'none'
  ? undefined
  : channelEnv || (isCI ? undefined : 'chrome');
const headless = parseBool(process.env.PLAYWRIGHT_HEADLESS, isCI);
const angleBackend = (process.env.PLAYWRIGHT_ANGLE || (isCI ? 'swiftshader' : 'vulkan')).toLowerCase();
const angleArgs = angleBackend === 'vulkan'
  ? ['--enable-features=Vulkan', '--use-angle=vulkan']
  : ['--use-angle=swiftshader', '--use-gl=swiftshader'];

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
        ...(resolvedChannel ? { channel: resolvedChannel } : {}),
        headless,
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            ...angleArgs,
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
