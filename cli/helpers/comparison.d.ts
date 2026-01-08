/**
 * Comparison Utilities - Baseline comparison and statistical analysis
 */

import type { ComparisonResult, TTestResult } from './types.js';

export function compareResults(baseline: any, current: any): ComparisonResult[];

export function formatComparison(comparisons: ComparisonResult[]): string;

export function welchTTest(a: number[], b: number[]): TTestResult;

export function formatTTestResult(metric: string, result: TTestResult): string;
