import type { Page } from 'playwright';
import type { CLIOptions, SuiteResult } from '../helpers/types.js';

export function runTrainingTests(
  page: Page,
  opts: CLIOptions,
  tests: string[]
): Promise<SuiteResult>;
