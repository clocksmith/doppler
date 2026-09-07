import assert from 'node:assert/strict';

async function rejected(page, operation) {
  const result = await page.evaluate(async operation => {
    try { await globalThis.documentSearch[operation](); return { rejected: false }; }
    catch (error) { return { rejected: true, name: error.name, message: error.message }; }
  }, operation);
  assert(result.rejected, `${operation} must reject the injected failure`);
  return result;
}

export async function checkInterruptedInstallation(page, context, origin) {
  const results = {};
  let interrupted = false;
  const pattern = '**/capsules/embedding/artifacts/model/shard_00000.bin';
  await context.route(pattern, async route => {
    interrupted = true;
    await page.click('#cancel');
    try { await route.abort('aborted'); }
    catch (error) { if (!/Route is already handled/.test(error.message)) throw error; }
  });
  try { results.cancellation = await rejected(page, 'install'); }
  finally { await context.unroute(pattern); }
  assert(interrupted, 'cancellation must reach an actual model-file request');
  results.interrupted = interrupted;
  results.incomplete = await rejected(page, 'open');
  assert.match(results.incomplete.message, /No completed model installation/);
  const cdp = await context.newCDPSession(page);
  const before = await cdp.send('Storage.getUsageAndQuota', { origin });
  results.quota = { mechanism: 'Chromium origin storage quota override', before };
  await cdp.send('Storage.overrideQuotaForOrigin', { origin, quotaSize: before.usage + 1024 });
  try {
    results.quota.failure = await rejected(page, 'install');
    assert.match(results.quota.failure.name + ' ' + results.quota.failure.message, /quota/i);
    results.quota.incomplete = await rejected(page, 'open');
    assert.match(results.quota.incomplete.message, /No completed model installation/);
  } finally {
    await cdp.send('Storage.overrideQuotaForOrigin', { origin });
    await cdp.detach();
  }
  return results;
}

export async function checkDamagedInstallation(page, fixture) {
  await page.evaluate(() => globalThis.documentSearch.close());
  const damage = await page.evaluate(async () => {
    const config = await (await fetch('./models.json')).json();
    const model = config.models.find(model => model.role === 'embedding');
    const root = await navigator.storage.getDirectory();
    const storage = await root.getDirectoryHandle(config.storage.opfsRootDir);
    const directory = await storage.getDirectoryHandle(model.storageId);
    const installation = JSON.parse(await (await (await directory.getFileHandle('installation.json')).getFile()).text());
    const artifact = installation.capsule.artifacts.find(artifact => artifact.path.endsWith('shard_00000.bin'));
    const checkpoint = await (await (await directory.getFileHandle('release-checkpoint.json')).getFile()).text();
    const artifacts = await directory.getDirectoryHandle('artifacts');
    const file = await artifacts.getFileHandle(artifact.hash.slice(7));
    const first = new Uint8Array(await (await file.getFile()).slice(0, 1).arrayBuffer());
    first[0] ^= 1;
    const writer = await file.createWritable({ keepExistingData: true });
    await writer.write({ type: 'write', position: 0, data: first }); await writer.close();
    return { artifact, checkpoint, storageId: model.storageId, opfsRootDir: config.storage.opfsRootDir };
  });
  const result = { damage, rejected: await rejected(page, 'open') };
  assert.match(result.rejected.message, /integrity failed/);
  result.repaired = await page.evaluate(() => globalThis.documentSearch.repair());
  result.checkpointPreserved = await page.evaluate(async damage => {
    const root = await navigator.storage.getDirectory();
    const storage = await root.getDirectoryHandle(damage.opfsRootDir);
    const directory = await storage.getDirectoryHandle(damage.storageId);
    return await (await (await directory.getFileHandle('release-checkpoint.json')).getFile()).text() === damage.checkpoint;
  }, damage);
  assert(result.checkpointPreserved);
  result.search = await page.evaluate(query => globalThis.documentSearch.search(query), fixture.queries[0].text);
  assert.equal(result.search.results[0].document.id, fixture.queries[0].expectedTopId);
  return result;
}

export async function checkIncompatibleIndex(page, fixture) {
  await page.evaluate(() => globalThis.documentSearch.close());
  const result = { mechanism: 'Retained index binding changed to simulate an incompatible prior embedding release.' };
  result.priorIdentity = await page.evaluate(async () => {
    const config = await (await fetch('./models.json')).json();
    const root = await navigator.storage.getDirectory();
    const storage = await root.getDirectoryHandle(config.storage.opfsRootDir);
    const directory = await storage.getDirectoryHandle('documents');
    const file = await directory.getFileHandle('index.json');
    const record = JSON.parse(await (await file.getFile()).text());
    const prior = record.index.embeddingIdentity;
    record.index.embeddingIdentity = JSON.stringify({ semanticRoot: 'injected-incompatible-prior-release' });
    const bytes = new TextEncoder().encode(JSON.stringify(record.index));
    record.digest = 'sha256:' + Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte => byte.toString(16).padStart(2, '0')).join('');
    const writer = await file.createWritable(); await writer.write(JSON.stringify(record)); await writer.close();
    return prior;
  });
  result.opened = await page.evaluate(() => globalThis.documentSearch.open());
  assert.equal(result.opened.indexInvalidated, true);
  result.rebuilt = await page.evaluate(() => globalThis.documentSearch.rebuildIndex());
  assert.equal(result.rebuilt.documents, fixture.documents.length);
  result.search = await page.evaluate(query => globalThis.documentSearch.search(query), fixture.queries[0].text);
  assert.equal(result.search.results[0].document.id, fixture.queries[0].expectedTopId);
  return result;
}

export async function checkDeviceLoss(page, fixture) {
  const result = await page.evaluate(async query => {
    const { getDevice } = await import('./runtime/src/gpu/device.js');
    const device = getDevice();
    if (!device) throw new Error('The physical device must exist before the loss check.');
    device.destroy();
    const lost = await device.lost;
    try { await globalThis.documentSearch.search(query); return { rejected: false }; }
    catch (error) { return { rejected: true, reason: lost.reason, message: error.message }; }
  }, fixture.queries[0].text);
  assert(result.rejected);
  assert.match(result.message, /lost|closed|destroyed/i);
  await page.evaluate(() => globalThis.documentSearch.close());
  result.reopened = await page.evaluate(() => globalThis.documentSearch.open());
  result.search = await page.evaluate(query => globalThis.documentSearch.search(query), fixture.queries[0].text);
  assert.equal(result.search.results[0].document.id, fixture.queries[0].expectedTopId);
  return result;
}
