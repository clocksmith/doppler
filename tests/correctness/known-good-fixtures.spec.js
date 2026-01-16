

import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { createHash } from 'crypto';

const FIXTURES_PATH = join(process.cwd(), 'tests', 'fixtures', 'known-good-outputs.json');
const EXPECTED = JSON.parse(readFileSync(FIXTURES_PATH, 'utf-8'));
const SHOULD_UPDATE = process.env.DOPPLER_UPDATE_KNOWN_GOOD === '1';

function hashOutput(text) {
  return createHash('sha256').update(text).digest('hex');
}

test.describe('known-good fixtures', () => {
  for (const [fixtureId, entry] of Object.entries(EXPECTED)) {
    test(`${fixtureId} output checksum`, async ({ page, baseURL }) => {
      const origin = baseURL || 'http://localhost:8080';
      await page.goto(`${origin}/doppler/tests/harness.html`);
      await page.waitForLoadState('domcontentloaded');

      const output = await page.evaluate(async ({ fixtureId, entry }) => {
        const { initDevice, getDevice } = await import('/doppler/dist/gpu/device.js');
        await initDevice();
        const device = getDevice();
        const origin = window.location.origin;
        const modelUrl = `${origin}/doppler/tests/fixtures/${fixtureId}`;
        const manifestResp = await fetch(`${modelUrl}/manifest.json`);
        const manifest = await manifestResp.json();

        const { createPipeline } = await import('/doppler/dist/inference/pipeline.js');
        const loadShard = async (idx) => {
          const shard = manifest.shards[idx];
          const resp = await fetch(`${modelUrl}/${shard.fileName}`);
          return new Uint8Array(await resp.arrayBuffer());
        };

        const pipeline = await createPipeline(manifest, {
          storage: { loadShard },
          gpu: { device },
          baseUrl: modelUrl,
        });

        const chunks = [];
        for await (const text of pipeline.generate(entry.prompt, {
          maxTokens: entry.maxTokens,
          temperature: 0,
        })) {
          chunks.push(text);
        }
        return chunks.join('');
      }, { fixtureId, entry });

      const checksum = hashOutput(output);
      if (SHOULD_UPDATE) {
        EXPECTED[fixtureId].checksum = checksum;
        writeFileSync(FIXTURES_PATH, JSON.stringify(EXPECTED, null, 2) + '\n');
        expect(true).toBe(true);
        return;
      }

      if (!entry.checksum) {
        throw new Error(`Missing known-good checksum for ${fixtureId}. Set DOPPLER_UPDATE_KNOWN_GOOD=1 to update.`);
      }

      expect(checksum).toBe(entry.checksum);
    });
  }
});
