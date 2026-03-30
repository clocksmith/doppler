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

  assert.deepEqual(resolveSection(report, 'compute/parity'), {
    section: 'warm',
    payload: report.sections.warm,
  });
  assert.deepEqual(resolveSection(report, 'compute/throughput'), {
    section: 'warm',
    payload: report.sections.warm,
  });
  assert.deepEqual(resolveSection(report, 'warm'), {
    section: 'warm',
    payload: report.sections.warm,
  });
}

{
  const report = {
    sections: {
      compute: {
        parity: {
          doppler: { result: { timing: { decodeTokensPerSec: 1 } } },
          transformersjs: { result: { timing: { decodeTokensPerSec: 1 } } },
        },
      },
    },
  };

  assert.deepEqual(resolveSection(report, 'warm'), {
    section: 'compute/parity',
    payload: report.sections.compute.parity,
  });
}

{
  const source = readFileSync(new URL('../../benchmarks/vendors/compare-chart.js', import.meta.url), 'utf8');
  assert.match(source, /title: 'First token'/);
  assert.match(source, /SHORTER BAR = FASTER/);
  assert.match(source, /normalized === 'warm' \|\| normalized === 'compute\/parity'/);
  assert.match(source, /font-weight="bold" stroke="none">TOTAL<\/text>/);
}

for (const fixtureName of [
  'g3-1b-p064-d064-t0-k1.compare.json',
  'lfm2-5-1-2b-p064-d064-t0-k1.compare.json',
]) {
  const fixture = JSON.parse(readFileSync(new URL(`../../benchmarks/vendors/fixtures/${fixtureName}`, import.meta.url), 'utf8'));
  assert.equal(fixture.metricContract.metrics[1].id, 'promptTokensPerSecToFirstToken');
  assert.equal(
    fixture.methodology.promptTokensPerSecToFirstToken,
    'prompt_tokens / firstTokenMs',
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
