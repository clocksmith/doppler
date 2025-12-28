import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Only include unit tests (*.test.ts), not playwright tests (*.spec.ts)
    include: ['**/*.test.ts'],
    // Explicitly exclude playwright test directories
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      'kernel-tests/tests/correctness/**',
      'kernel-tests/tests/benchmarks/**',
      'tests/correctness/**',
      'tests/benchmark/**',
    ],
    environment: 'node',
    globals: true,
    // Don't fail when no tests found (all tests are currently playwright)
    passWithNoTests: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
});
