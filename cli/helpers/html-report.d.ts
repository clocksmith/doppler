/**
 * HTML Report Generation - SVG charts and HTML benchmark reports
 */

export function generateSVGBarChart(
  data: Array<{ label: string; value: number; color?: string }>,
  width?: number,
  height?: number,
  title?: string
): string;

export function generateSVGLineChart(
  data: number[],
  width?: number,
  height?: number,
  title?: string,
  yLabel?: string
): string;

export function generateHTMLReport(results: any, baseline?: any): string;
