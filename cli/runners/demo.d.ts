import type { Page } from 'playwright';
import type { CLIOptions, SuiteResult } from '../helpers/types.js';

export function runDemoTest(page: Page, opts: CLIOptions): Promise<SuiteResult>;
