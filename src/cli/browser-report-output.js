import fs from 'node:fs/promises';
import path from 'node:path';
import { isPlainObject } from '../formats/plain-object.js';

export async function persistBrowserRelayReport(response, outputPath) {
  if (!outputPath) return response;
  if (!isPlainObject(response?.result?.report)) {
    throw new Error('run.browser.reportOutputPath requires the browser command to return a report object.');
  }
  const resolved = path.resolve(outputPath);
  await fs.mkdir(path.dirname(resolved), { recursive: true });
  await fs.writeFile(resolved, `${JSON.stringify(response.result.report, null, 2)}\n`, 'utf8');
  return {
    ...response,
    result: {
      ...response.result,
      reportInfo: {
        ...(isPlainObject(response.result.reportInfo) ? response.result.reportInfo : {}),
        path: path.relative(process.cwd(), resolved),
        persistedBy: 'node-browser-command-relay',
      },
    },
  };
}
