import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  loadChartMetricContract,
  resolveSection,
} from '../../benchmarks/vendors/compare-chart.js';

{
  const contract = loadChartMetricContract();
  assert.ok(Array.isArray(contract.metrics));
  assert.ok(contract.metrics.length > 0);
  assert.ok(contract.metricPathHints.decodeTokensPerSec.length > 0);
}

{
  const report = {
    sections: {
      warm: {
        doppler: { result: { timing: { decodeTokensPerSec: 1 } } },
        transformersjs: { result: { timing: { decodeTokensPerSec: 1 } } },
      },
    },
  };

  assert.equal(resolveSection(report, 'compute/parity'), null);
  assert.deepEqual(resolveSection(report, 'warm'), {
    section: 'warm',
    payload: report.sections.warm,
  });
}

{
  const source = readFileSync(new URL('../../benchmarks/vendors/compare-chart.js', import.meta.url), 'utf8');
  assert.match(source, /Prompt → First Token \${resolved\.ttft\.toFixed\(1\)} ms/);
  assert.match(source, /label: 'Prompt → First Token'/);
  assert.match(source, /font-weight="bold">TTFT</);
}

for (const fixtureName of [
  'compare_20260303T175640.json',
  'compare_20260303T210150.json',
]) {
  const fixture = JSON.parse(readFileSync(new URL(`../../benchmarks/vendors/results/${fixtureName}`, import.meta.url), 'utf8'));
  assert.equal(fixture.metricContract.metrics[1].id, 'promptTokensPerSecToFirstToken');
  assert.equal(
    fixture.methodology.promptTokensPerSecToFirstToken,
    'prompt_tokens / first_token_ms',
  );
  assert.ok(
    fixture.harnesses.doppler.requiredMetrics.includes('promptTokensPerSecToFirstToken'),
  );
  assert.ok(
    fixture.harnesses.transformersjs.requiredMetrics.includes('promptTokensPerSecToFirstToken'),
  );
  assert.ok(
    !fixture.harnesses.doppler.requiredMetrics.includes('prefillTokensPerSec'),
  );
  assert.ok(
    !fixture.harnesses.transformersjs.requiredMetrics.includes('prefillTokensPerSec'),
  );
}

console.log('compare-chart-contract.test: ok');
